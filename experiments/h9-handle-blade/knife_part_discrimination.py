"""H9 — Knife handle-vs-blade patch-level part discrimination.

The sharpest local test of "does VLA cut-class loss matter for the part-
discrimination task that the affordance-manipulation literature relies on?"

Setup:
  - UMD images restricted to {knife, shears, scissors, saw} categories.
  - Pool patches whose GROUND-TRUTH label is grasp (handle) or cut (blade).
    Discard background, scoop, contain, support, and other class-foreign patches.
  - Train per encoder: binary logistic regression  features → {0=handle, 1=blade}.
  - Evaluate on held-out test images. Metrics: balanced accuracy, AUC, F1.

Encoders compared:
  DINOv2-base/large, standalone SigLIP-So400m, π0 SigLIP, π0.5 SigLIP,
  OpenVLA SigLIP, π0+adapter (256-d hidden), π0.5+adapter,
  random-projection of standalone SigLIP to 256-d (dim-reduction control).

Predictions (under the "VLA loses cut → manipulation suffers" story):
  - Standalone SigLIP, DINOv2-base/large >> π0 SigLIP, OpenVLA on this metric.
  - π0+adapter recovers most of the gap.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.dataset_umd import UMDSubset

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "h8_predict_actions",
    str(ROOT / "experiments" / "h8-action-proxy" / "predict_actions.py"),
)
_pa = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pa)

featurize_dinov2 = _pa.featurize_dinov2
featurize_hf_siglip = _pa.featurize_hf_siglip
featurize_timm_siglip = _pa.featurize_timm_siglip
build_pi0_full = _pa.build_pi0_full
build_openvla_full = _pa.build_openvla_full
build_standalone = _pa.build_standalone

# Class IDs in our 5+bg taxonomy:
BG, GRASP, CUT, SCOOP, CONTAIN, SUPPORT = 0, 1, 2, 3, 4, 5
CUT_CATS = {"knife", "shears", "scissors", "saw"}


def select_cut_objects(subset: UMDSubset) -> list:
    return [s for s in subset.samples if s.category in CUT_CATS]


def pool_label_to_patches(label: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    """Returns a (gh*gh,) array of majority-class labels per patch."""
    gh = image_size // patch_size
    out = np.zeros(gh * gh, dtype=np.int64)
    ps = patch_size
    for i in range(gh):
        for j in range(gh):
            tile = label[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps].ravel()
            vals, counts = np.unique(tile, return_counts=True)
            out[i * gh + j] = int(vals[counts.argmax()])
    return out


def get_patch_features_full_image(rgbs, model_kind, processor, model, image_size, device):
    """Returns (N_images, n_patches, D) full per-patch features."""
    from PIL import Image

    feats = []
    mean = np.asarray(getattr(processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(getattr(processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    n_patches_target = (image_size // 14) ** 2

    for rgb in rgbs:
        pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
        arr = (np.asarray(pil, dtype=np.float32) / 255.0 - mean) / std
        pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            if model_kind == "hf_siglip":
                out = model(pix)
                f = out.last_hidden_state[0]
            elif model_kind == "timm":
                f = model.forward_features(pix)[0]
            elif model_kind == "dinov2":
                import inspect

                sig = inspect.signature(model.forward)
                if "interpolate_pos_encoding" in sig.parameters:
                    out = model(pix, interpolate_pos_encoding=True)
                else:
                    out = model(pix)
                f = out.last_hidden_state[0]
            else:
                raise ValueError(model_kind)
        if f.shape[0] > n_patches_target:
            f = f[-n_patches_target:]
        feats.append(f.cpu().numpy())
    return np.stack(feats)


def get_patch_features_pi(rgbs, probe, image_size, device, adapter=None):
    """π0/π0.5 patch features. If adapter is provided, returns its 256-d hidden."""
    from PIL import Image

    proc = probe._processor
    mean = np.asarray(getattr(proc, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(getattr(proc, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    n_patches_target = (image_size // 14) ** 2
    feats = []
    for rgb in rgbs:
        pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
        arr = (np.asarray(pil, dtype=np.float32) / 255.0 - mean) / std
        pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            out = probe._backbone(pix)
            f = out.last_hidden_state[0]
            if f.shape[0] > n_patches_target:
                f = f[-n_patches_target:]
            if adapter is not None:
                _, h = adapter(f)
                f = h
        feats.append(f.cpu().numpy())
    return np.stack(feats)


class MLPAdapter(nn.Module):
    def __init__(self, in_dim=1152, num_classes=6, hidden=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.act(self.proj(x))
        return self.cls(self.drop(h)), h

    @classmethod
    def from_state_dict(cls, sd, in_dim=1152, num_classes=6, hidden=256):
        m = cls(in_dim=in_dim, num_classes=num_classes, hidden=hidden)
        m.load_state_dict({
            "proj.weight": sd["net.0.weight"], "proj.bias": sd["net.0.bias"],
            "cls.weight": sd["net.3.weight"], "cls.bias": sd["net.3.bias"],
        })
        return m


def featurize_for_encoder(name, rgbs, image_size, device):
    if name in ("dinov2_base", "dinov2_large"):
        from transformers import AutoImageProcessor, AutoModel

        hf = "facebook/dinov2-base" if name == "dinov2_base" else "facebook/dinov2-large"
        proc = AutoImageProcessor.from_pretrained(hf)
        model = AutoModel.from_pretrained(hf).eval().to(device)
        return get_patch_features_full_image(rgbs, "dinov2", proc, model, image_size, device)

    if name == "standalone_siglip":
        model, proc = build_standalone(device)
        return get_patch_features_full_image(rgbs, "hf_siglip", proc, model, image_size, device)

    if name in ("pi0_siglip", "pi05_siglip"):
        p = build_pi0_full(use_pi05=(name == "pi05_siglip"), device=device)
        return get_patch_features_pi(rgbs, p, image_size, device, adapter=None)

    if name == "openvla_siglip":
        p = build_openvla_full(device)
        return get_patch_features_full_image(rgbs, "timm", p._processor, p._backbone,
                                             image_size, device)

    if name in ("pi0_adapter", "pi05_adapter"):
        which = "pi0" if name == "pi0_adapter" else "pi05"
        p = build_pi0_full(use_pi05=(which == "pi05"), device=device)
        sd_path = ROOT / "outputs/intervention" / f"adapter_{which}_siglip_h256.pt"
        sd = torch.load(sd_path, map_location=device)
        adapter = MLPAdapter.from_state_dict(sd, in_dim=1152, hidden=256).to(device).eval()
        return get_patch_features_pi(rgbs, p, image_size, device, adapter=adapter)

    raise ValueError(name)


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("h9")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 224
    patch_size = 14
    gh = image_size // patch_size

    taxonomy = ROOT / "configs/affordance_taxonomy.yaml"
    train_split = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/train.json", taxonomy, image_size=image_size)
    val_split = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/val.json", taxonomy, image_size=image_size)
    test_split = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/test.json", taxonomy, image_size=image_size)

    train_cuts = select_cut_objects(train_split)
    val_cuts = select_cut_objects(val_split)
    test_cuts = select_cut_objects(test_split)
    log.info("knife/shears/scissors/saw images: train=%d val=%d test=%d",
             len(train_cuts), len(val_cuts), len(test_cuts))

    mapping = train_split.mapping

    def gather_rgbs_labels(samples):
        rgbs, labels = [], []
        for s in samples:
            rgb = s.load_rgb(size=image_size)
            lbl = s.load_label(mapping, size=image_size)
            rgbs.append(rgb)
            labels.append(lbl)
        return np.stack(rgbs).astype(np.uint8), np.stack(labels).astype(np.uint8)

    rgb_tr, lbl_tr = gather_rgbs_labels(train_cuts)
    rgb_va, lbl_va = gather_rgbs_labels(val_cuts)
    rgb_te, lbl_te = gather_rgbs_labels(test_cuts)
    log.info("loaded RGBs: train=%s val=%s test=%s", rgb_tr.shape, rgb_va.shape, rgb_te.shape)

    # Patch-level labels.
    def labels_to_patch_labels(labels):
        """Returns (N, n_patches) int array."""
        out = np.zeros((labels.shape[0], gh * gh), dtype=np.int64)
        for i in range(labels.shape[0]):
            out[i] = pool_label_to_patches(labels[i], image_size, patch_size)
        return out

    plbl_tr = labels_to_patch_labels(lbl_tr)
    plbl_va = labels_to_patch_labels(lbl_va)
    plbl_te = labels_to_patch_labels(lbl_te)

    def keep_handle_blade_mask(plbls):
        """Boolean mask for patches whose label is GRASP (1) or CUT (2)."""
        return (plbls == GRASP) | (plbls == CUT)

    mask_tr = keep_handle_blade_mask(plbl_tr)
    mask_va = keep_handle_blade_mask(plbl_va)
    mask_te = keep_handle_blade_mask(plbl_te)
    log.info("usable patches (handle+blade only): train=%d val=%d test=%d",
             mask_tr.sum(), mask_va.sum(), mask_te.sum())
    log.info("class balance train: handle=%d blade=%d",
             (plbl_tr[mask_tr] == GRASP).sum(), (plbl_tr[mask_tr] == CUT).sum())

    encoders = ["dinov2_base", "dinov2_large", "standalone_siglip",
                "pi0_siglip", "pi05_siglip", "openvla_siglip",
                "pi0_adapter", "pi05_adapter"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    pred_grids_per_enc = {}  # for qualitative figure later

    for name in encoders:
        log.info(">>> %s ...", name)
        t0 = time.time()
        try:
            feats_tr = featurize_for_encoder(name, rgb_tr, image_size, device)
            feats_va = featurize_for_encoder(name, rgb_va, image_size, device)
            feats_te = featurize_for_encoder(name, rgb_te, image_size, device)
        except Exception as e:
            log.exception("FAILED %s: %s", name, e)
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        D = feats_tr.shape[-1]
        log.info("    feats train=%s val=%s test=%s (D=%d) in %.1fs",
                 feats_tr.shape, feats_va.shape, feats_te.shape, D, time.time() - t0)

        X_tr = feats_tr.reshape(-1, D)[mask_tr.reshape(-1)]
        y_tr = (plbl_tr.reshape(-1)[mask_tr.reshape(-1)] == CUT).astype(np.int64)
        X_va = feats_va.reshape(-1, D)[mask_va.reshape(-1)]
        y_va = (plbl_va.reshape(-1)[mask_va.reshape(-1)] == CUT).astype(np.int64)
        X_te = feats_te.reshape(-1, D)[mask_te.reshape(-1)]
        y_te = (plbl_te.reshape(-1)[mask_te.reshape(-1)] == CUT).astype(np.int64)

        # Standardize on train.
        mu, sigma = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-6
        X_tr = (X_tr - mu) / sigma
        X_va = (X_va - mu) / sigma
        X_te = (X_te - mu) / sigma

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score

        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, class_weight="balanced")
        clf.fit(X_tr, y_tr)

        pred_va = clf.predict(X_va)
        prob_va = clf.predict_proba(X_va)[:, 1]
        pred_te = clf.predict(X_te)
        prob_te = clf.predict_proba(X_te)[:, 1]

        record = dict(
            feat_dim=int(D),
            n_train=int(len(y_tr)),
            n_val=int(len(y_va)),
            n_test=int(len(y_te)),
            val=dict(
                accuracy=float(accuracy_score(y_va, pred_va)),
                balanced_accuracy=float(balanced_accuracy_score(y_va, pred_va)),
                f1_blade=float(f1_score(y_va, pred_va, pos_label=1)),
                auc=float(roc_auc_score(y_va, prob_va)),
            ),
            test=dict(
                accuracy=float(accuracy_score(y_te, pred_te)),
                balanced_accuracy=float(balanced_accuracy_score(y_te, pred_te)),
                f1_blade=float(f1_score(y_te, pred_te, pos_label=1)),
                auc=float(roc_auc_score(y_te, prob_te)),
            ),
        )
        results[name] = record
        log.info("    %s test bal_acc=%.3f f1_blade=%.3f auc=%.3f",
                 name, record["test"]["balanced_accuracy"],
                 record["test"]["f1_blade"], record["test"]["auc"])

        # Save full-image prob grids for qualitative panel.
        # Re-run on test images: every patch, full prob (not just handle+blade).
        full_test_feats = feats_te  # (N, n_patches, D)
        Nt, Np, _ = full_test_feats.shape
        flat = full_test_feats.reshape(-1, D)
        flat_z = (flat - mu) / sigma
        prob_full = clf.predict_proba(flat_z)[:, 1].reshape(Nt, gh, gh)
        pred_grids_per_enc[name] = prob_full.astype(np.float32)

    # Save raw results JSON.
    with open(out_dir / "h9_handle_blade_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out_dir / "h9_handle_blade_results.json")

    # Save prob grids for qualitative panel.
    np.savez(out_dir / "h9_test_pred_grids.npz",
             rgbs=rgb_te,
             labels=lbl_te,
             **pred_grids_per_enc)
    log.info("wrote %s", out_dir / "h9_test_pred_grids.npz")

    # ===== Plot summary bar chart =====
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["dinov2_base", "dinov2_large", "standalone_siglip",
             "pi0_siglip", "pi0_adapter", "pi05_siglip", "pi05_adapter",
             "openvla_siglip"]
    names = [n for n in order if n in results]
    test_bal = [results[n]["test"]["balanced_accuracy"] for n in names]
    test_auc = [results[n]["test"]["auc"] for n in names]

    color_map = {
        "dinov2_base": "#1e5b34", "dinov2_large": "#3a8a4f",
        "standalone_siglip": "#4a72c9",
        "pi0_siglip": "#c94a72", "pi0_adapter": "#d3a45f",
        "pi05_siglip": "#aa6c39", "pi05_adapter": "#7a4d27",
        "openvla_siglip": "#5fa05f",
    }
    colors = [color_map[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    for ax, vals, title, ylim in [
        (ax1, test_bal, "Balanced accuracy (handle vs blade, test)", (0.5, 1.0)),
        (ax2, test_auc, "ROC-AUC (handle vs blade, test)", (0.5, 1.0)),
    ]:
        bars = ax.bar(np.arange(len(names)), vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    fig.suptitle("H9 — Knife handle-vs-blade discrimination on UMD\n"
                 "(per-patch binary classification on knife/shears/scissors/saw)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "h9_handle_blade.png")
    log.info("wrote %s", out_dir / "h9_handle_blade.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/h9-handle-blade/results")
    args = ap.parse_args()
    main(args)
