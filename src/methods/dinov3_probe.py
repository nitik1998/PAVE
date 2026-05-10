"""Frozen DINOv3 (or DINOv2 fallback) + sklearn linear probe.

Two-step lifecycle:
  1. fit(): extract patch features over the train split, fit a multinomial
     logistic regression in feature space (per-patch labels are pooled from
     ground truth via mode-of-pixels under each patch).
  2. predict_map(): extract features for one image, apply the fitted probe,
     bilinear upsample logits to image resolution, softmax → (C, H, W).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.methods.base import AffordancePredictor

log = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    hf_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    fallback_hf_id: str | None = "facebook/dinov2-base"
    image_size: int = 448
    patch_size: int = 16
    feature_dim: int = 768
    device: str = "cpu"
    dtype: str = "float32"
    pool_per_patch_label: str = "mode"     # mode | majority


class DinoLinearProbe(AffordancePredictor):
    name = "dinov3"

    def __init__(self, cfg: ProbeConfig, num_classes: int, foreground_names: list[str]):
        self.cfg = cfg
        self.num_classes = num_classes
        self.foreground_class_names = [n for n in foreground_names if n != "background"]
        self._backbone = None
        self._processor = None
        self._clf = None

    # --- backbone loading ---
    def warmup(self) -> None:
        if self._backbone is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        model_id = self.cfg.hf_id
        try:
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._backbone = AutoModel.from_pretrained(model_id)
        except Exception as e:
            if self.cfg.fallback_hf_id is None:
                raise
            log.warning("Falling back to %s (%s)", self.cfg.fallback_hf_id, e)
            model_id = self.cfg.fallback_hf_id
            self.name = "dinov2"
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._backbone = AutoModel.from_pretrained(model_id)
        self._backbone.eval().to(self.cfg.device)
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        cfg_obj = getattr(self._backbone, "config", None)
        actual_patch = getattr(cfg_obj, "patch_size", None)
        actual_image = getattr(cfg_obj, "image_size", None)
        if actual_patch is not None and actual_patch != self.cfg.patch_size:
            log.warning("Adjusting patch_size %d -> %d to match loaded backbone", self.cfg.patch_size, actual_patch)
            self.cfg.patch_size = int(actual_patch)
        # SigLIP and similar use fixed position embeddings, so we must use
        # exactly the image size the model was trained at. DINOv2/3 support
        # arbitrary sizes via position-embedding interpolation, but we still
        # prefer divisible-by-patch sizes.
        family = type(self._backbone).__name__.lower()
        force_train_size = "siglip" in family or "clip" in family
        if force_train_size and actual_image is not None:
            if self.cfg.image_size != int(actual_image):
                log.warning("Forcing image_size %d -> %d to match %s position embeddings",
                            self.cfg.image_size, int(actual_image), family)
                self.cfg.image_size = int(actual_image)
        if self.cfg.image_size % self.cfg.patch_size != 0:
            new_sz = (self.cfg.image_size // self.cfg.patch_size) * self.cfg.patch_size
            log.warning("Adjusting image_size %d -> %d to be divisible by patch_size %d", self.cfg.image_size, new_sz, self.cfg.patch_size)
            self.cfg.image_size = new_sz
        log.info("Loaded %s on %s (patch=%d, image=%d)", model_id, self.cfg.device, self.cfg.patch_size, self.cfg.image_size)

    # --- feature extraction ---
    def _extract_patch_features(self, rgb: np.ndarray) -> np.ndarray:
        """rgb (H, W, 3) → (N_patches, D) numpy.

        We resize+normalize manually so the patch grid has exactly
        ``(image_size/patch_size)^2`` tokens, regardless of the HF image
        processor's default crop_size.
        """
        import torch
        from PIL import Image

        self.warmup()
        S = int(self.cfg.image_size)
        pil = Image.fromarray(rgb).resize((S, S), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        # Pull mean/std from the processor if available; else ImageNet defaults.
        mean = getattr(self._processor, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(self._processor, "image_std", [0.229, 0.224, 0.225])
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        arr = (arr - mean) / std
        pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.cfg.device).float()
        import inspect

        with torch.no_grad():
            sig = inspect.signature(self._backbone.forward)
            if "interpolate_pos_encoding" in sig.parameters:
                out = self._backbone(pix, interpolate_pos_encoding=True)
            else:
                out = self._backbone(pix)
        feats = out.last_hidden_state[0]
        gh = S // self.cfg.patch_size
        n_patches = gh * gh
        if feats.shape[0] > n_patches:
            feats = feats[-n_patches:]
        elif feats.shape[0] != n_patches:
            raise RuntimeError(
                f"backbone returned {feats.shape[0]} tokens, expected {n_patches} "
                f"for image={S}, patch={self.cfg.patch_size}"
            )
        return feats.cpu().numpy().astype(np.float32)

    # --- patch-level GT pooling ---
    def _pool_label_to_patches(self, label: np.ndarray) -> np.ndarray:
        gh = self.cfg.image_size // self.cfg.patch_size
        ps = self.cfg.patch_size
        # label has been resized to image_size already.
        out = np.zeros(gh * gh, dtype=np.int64)
        for i in range(gh):
            for j in range(gh):
                tile = label[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps].ravel()
                vals, counts = np.unique(tile, return_counts=True)
                out[i * gh + j] = int(vals[counts.argmax()])
        return out

    # --- fit on a list of (rgb, label) ---
    def fit(self, samples: list[tuple[np.ndarray, np.ndarray]], C: float = 1.0):
        from sklearn.linear_model import LogisticRegression

        Xs, ys = [], []
        for rgb, label in samples:
            feats = self._extract_patch_features(rgb)
            ys_patch = self._pool_label_to_patches(label)
            Xs.append(feats)
            ys.append(ys_patch)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        log.info("Fitting linear probe on X=%s, y=%s, classes=%s", X.shape, y.shape, np.unique(y))
        self._clf = LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=1000,
            n_jobs=-1,
        )
        self._clf.fit(X, y)
        return self

    # --- inference ---
    def predict_map(self, rgb: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Probe is not fitted. Call .fit(samples) first.")
        feats = self._extract_patch_features(rgb)
        gh = self.cfg.image_size // self.cfg.patch_size
        proba = self._clf.predict_proba(feats)              # (N, K)
        K = proba.shape[1]
        grid = proba.reshape(gh, gh, K).transpose(2, 0, 1)  # (K, gh, gh)
        # Upsample to image_size.
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(grid).unsqueeze(0).float()
        t = F.interpolate(t, size=self.cfg.image_size, mode="bilinear", align_corners=False)
        full = t.squeeze(0).cpu().numpy()                   # (K, H, W)
        # Drop background channel for the (C, H, W) foreground convention.
        if 0 in self._clf.classes_:
            bg_idx = int(np.where(self._clf.classes_ == 0)[0][0])
            mask = np.ones(full.shape[0], dtype=bool)
            mask[bg_idx] = False
            full = full[mask]
        return full.astype(np.float32)

    # --- persistence ---
    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "cfg": self.cfg, "name": self.name}, path)

    def load(self, path: str | Path) -> "DinoLinearProbe":
        import joblib

        blob = joblib.load(path)
        self._clf = blob["clf"]
        self.cfg = blob["cfg"]
        self.name = blob["name"]
        return self
