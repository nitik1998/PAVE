"""Probe the SigLIP-So400m vision tower extracted from OpenVLA-7b.

OpenVLA (Kim et al., CoRL 2024) is a different VLA family from π0/π0.5:
  - Backbone family: Prismatic VLM (Karamcheti et al.)
  - Vision: fused DINOv2-large + SigLIP-So400m (timm; both at 224×224)
  - LM: Llama-2-7b
  - Fine-tuning corpus: 970k OXE trajectories (different from π0's data mix).

We extract just the SigLIP vision tower (`vision_backbone.fused_featurizer.*`)
and load it into a fresh `timm.create_model('vit_so400m_patch14_siglip_224')`.
This makes OpenVLA's affordance probe directly comparable to:
  - standalone SigLIP-So400m@224 (HF version)
  - π0 SigLIP-So400m@224
  - π0.5 SigLIP-So400m@224

Output protocol matches `Pi0SigLIPProbe`: per-patch features → linear probe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.methods.dinov3_probe import DinoLinearProbe, ProbeConfig

log = logging.getLogger(__name__)


@dataclass
class OpenVLASigLIPConfig(ProbeConfig):
    openvla_repo: str = "openvla/openvla-7b"
    timm_arch: str = "vit_so400m_patch14_siglip_224"
    image_size: int = 224
    patch_size: int = 14
    feature_dim: int = 1152


class OpenVLASigLIPProbe(DinoLinearProbe):
    name = "openvla_siglip"

    def warmup(self) -> None:
        if self._backbone is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import timm

        log.info("Building timm skeleton %s ...", self.cfg.timm_arch)
        model = timm.create_model(self.cfg.timm_arch, pretrained=False, num_classes=0)
        model = model.eval().to(self.cfg.device)

        log.info("Locating OpenVLA weight shards ...")
        snap = snapshot_download(repo_id=self.cfg.openvla_repo,
                                 allow_patterns=["*.safetensors", "*.json"])
        snap = Path(snap)
        # Find shards.
        import json as _json

        idx_path = snap / "model.safetensors.index.json"
        with open(idx_path) as f:
            wmap = _json.load(f)["weight_map"]
        shards: dict[str, list[str]] = {}
        for k, fname in wmap.items():
            if k.startswith("vision_backbone.fused_featurizer."):
                shards.setdefault(fname, []).append(k)
        log.info("vision_backbone.fused_featurizer keys spread across %d shard(s)", len(shards))

        timm_state = model.state_dict()
        loaded = {}
        timm_keys_set = set(timm_state.keys())
        prefix = "vision_backbone.fused_featurizer."
        for fname, keys in shards.items():
            with safe_open(snap / fname, framework="pt") as f:
                for k in keys:
                    rest = k[len(prefix):]
                    if rest in timm_keys_set:
                        t = f.get_tensor(k)
                        if t.shape == timm_state[rest].shape:
                            loaded[rest] = t
                        else:
                            log.warning("shape mismatch %s: %s vs %s",
                                        rest, t.shape, timm_state[rest].shape)
        log.info("Loaded %d / %d timm tensors from OpenVLA", len(loaded), len(timm_state))

        missing, unexpected = model.load_state_dict(loaded, strict=False)
        if missing:
            log.warning("Missing keys after OpenVLA load: %d (e.g. %s)",
                        len(missing), missing[:5])
        if unexpected:
            log.warning("Unexpected keys: %s", unexpected[:5])
        for p in model.parameters():
            p.requires_grad_(False)
        self._backbone = model

        # Build a HF-style processor for the same SigLIP-So400m@224. This gives
        # us the right mean/std stats for normalization.
        from transformers import AutoImageProcessor

        self._processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-224")
        log.info("OpenVLA SigLIP loaded; patch=%d image=%d feature_dim=%d",
                 self.cfg.patch_size, self.cfg.image_size, self.cfg.feature_dim)

    def _extract_patch_features(self, rgb: np.ndarray) -> np.ndarray:
        """timm models don't take HF kwargs; we run them as plain forward and
        intercept the last attention-pool output via a forward hook.
        For ViT, ``forward_features`` returns a (B, N, D) tensor with patch
        tokens. timm's SigLIP has no CLS by default (it uses GAP/attn pool).
        """
        import torch
        from PIL import Image

        self.warmup()
        S = int(self.cfg.image_size)
        pil = Image.fromarray(rgb).resize((S, S), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        mean = np.asarray(getattr(self._processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        std = np.asarray(getattr(self._processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        arr = (arr - mean) / std
        pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.cfg.device).float()
        with torch.no_grad():
            feats = self._backbone.forward_features(pix)
        # timm SigLIP: (B, N_patches, D). N_patches = (224/14)^2 = 256.
        feats = feats[0]
        gh = S // self.cfg.patch_size
        n_patches = gh * gh
        if feats.shape[0] > n_patches:
            feats = feats[-n_patches:]
        elif feats.shape[0] != n_patches:
            raise RuntimeError(
                f"backbone returned {feats.shape[0]} tokens, expected {n_patches}"
            )
        return feats.cpu().numpy().astype(np.float32)


def build(num_classes: int, foreground_names: list[str], device: str = "cpu") -> OpenVLASigLIPProbe:
    cfg = OpenVLASigLIPConfig(device=device)
    return OpenVLASigLIPProbe(cfg=cfg, num_classes=num_classes, foreground_names=foreground_names)
