"""H10a — Cut-vs-rest-foreground binary classification across the WHOLE UMD dataset.

Tests where on the "complexity spectrum" the H2 cut-class loss actually
manifests behaviourally.

  H2: 5-class IoU across all UMD objects.            π0 cut IoU 0.181, std 0.455
  H9: binary handle/blade on single-object knives.   π0 0.985, std 1.000  → no gap
  H10a: binary CUT vs OTHER FOREGROUND on all UMD.   ?

Setup:
  - Use ALL UMD train (n=345), val (n=73), test (n=75) — every category.
  - Pool patch labels (224 image, 14 patch, 16x16=256 patches per image).
  - Keep every patch whose label is in {grasp, cut, scoop, contain, support}
    (drop background only). Binary target: 1 if cut, 0 otherwise.
  - Train balanced LogisticRegression per encoder; evaluate val and test.

If π0 / OpenVLA show a gap here but not in H9, the H2 multi-class loss
arises specifically from class-confusion when CUT must be discriminated
from OTHER foreground affordances — not from a perception deficit.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
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

_spec = _ilu.spec_from_file_location(
    "h9_kpd",
    str(ROOT / "experiments" / "h9-handle-blade" / "knife_part_discrimination.py"),
)
_h9 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_h9)
featurize_for_encoder = _h9.featurize_for_encoder
pool_label_to_patches = _h9.pool_label_to_patches

# Class IDs: 0=bg, 1=grasp, 2=cut, 3=scoop, 4=contain, 5=support
BG, GRASP, CUT, SCOOP, CONTAIN, SUPPORT = 0, 1, 2, 3, 4, 5


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("h10a")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 224
    patch_size = 14
    gh = image_size // patch_size

    taxonomy = ROOT / "configs/affordance_taxonomy.yaml"
    train = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/train.json", taxonomy, image_size=image_size)
    val = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/val.json", taxonomy, image_size=image_size)
    test = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/test.json", taxonomy, image_size=image_size)
    log.info("UMD splits: train=%d val=%d test=%d", len(train), len(val), len(test))

    mapping = train.mapping

    def gather(samples):
        rgbs, labels = [], []
        for s in samples:
            rgbs.append(s.load_rgb(size=image_size))
            labels.append(s.load_label(mapping, size=image_size))
        return np.stack(rgbs).astype(np.uint8), np.stack(labels).astype(np.uint8)

    rgb_tr, lbl_tr = gather(train.samples)
    rgb_va, lbl_va = gather(val.samples)
    rgb_te, lbl_te = gather(test.samples)
    log.info("loaded RGBs: train=%s val=%s test=%s", rgb_tr.shape, rgb_va.shape, rgb_te.shape)

    def patch_labels(labels):
        out = np.zeros((labels.shape[0], gh * gh), dtype=np.int64)
        for i in range(labels.shape[0]):
            out[i] = pool_label_to_patches(labels[i], image_size, patch_size)
        return out

    plbl_tr = patch_labels(lbl_tr)
    plbl_va = patch_labels(lbl_va)
    plbl_te = patch_labels(lbl_te)

    # Mask: keep any foreground patch (1..5).
    def fg_mask(plbls):
        return plbls > 0

    mask_tr = fg_mask(plbl_tr)
    mask_va = fg_mask(plbl_va)
    mask_te = fg_mask(plbl_te)
    log.info("foreground patches: train=%d val=%d test=%d (cut-fraction train=%.3f)",
             mask_tr.sum(), mask_va.sum(), mask_te.sum(),
             (plbl_tr[mask_tr] == CUT).mean())

    encoders = ["dinov2_base", "dinov2_large", "standalone_siglip",
                "pi0_siglip", "pi05_siglip", "openvla_siglip",
                "pi0_adapter", "pi05_adapter"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    pred_grids_te = {}

    for name in encoders:
        log.info(">>> %s ...", name)
        t0 = time.time()
        try:
            f_tr = featurize_for_encoder(name, rgb_tr, image_size, device)
            f_va = featurize_for_encoder(name, rgb_va, image_size, device)
            f_te = featurize_for_encoder(name, rgb_te, image_size, device)
        except Exception as e:
            log.exception("FAILED %s: %s", name, e)
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        D = f_tr.shape[-1]
        log.info("    feats train=%s val=%s test=%s in %.1fs",
                 f_tr.shape, f_va.shape, f_te.shape, time.time() - t0)

        X_tr = f_tr.reshape(-1, D)[mask_tr.reshape(-1)]
        y_tr = (plbl_tr.reshape(-1)[mask_tr.reshape(-1)] == CUT).astype(np.int64)
        X_va = f_va.reshape(-1, D)[mask_va.reshape(-1)]
        y_va = (plbl_va.reshape(-1)[mask_va.reshape(-1)] == CUT).astype(np.int64)
        X_te = f_te.reshape(-1, D)[mask_te.reshape(-1)]
        y_te = (plbl_te.reshape(-1)[mask_te.reshape(-1)] == CUT).astype(np.int64)

        mu, sigma = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-6
        X_tr = (X_tr - mu) / sigma
        X_va = (X_va - mu) / sigma
        X_te = (X_te - mu) / sigma

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score

        clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=-1, class_weight="balanced")
        clf.fit(X_tr, y_tr)

        def metrics(X, y):
            pred = clf.predict(X)
            prob = clf.predict_proba(X)[:, 1]
            return dict(
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                f1_cut=float(f1_score(y, pred, pos_label=1)),
                auc=float(roc_auc_score(y, prob)),
            )

        record = dict(
            feat_dim=int(D),
            n_train=int(len(y_tr)),
            n_val=int(len(y_va)),
            n_test=int(len(y_te)),
            cut_fraction_train=float(y_tr.mean()),
            val=metrics(X_va, y_va),
            test=metrics(X_te, y_te),
        )
        results[name] = record
        log.info("    %s test bal_acc=%.3f f1_cut=%.3f auc=%.3f",
                 name, record["test"]["balanced_accuracy"],
                 record["test"]["f1_cut"], record["test"]["auc"])

        # Save full prob grids for test (for the qualitative panel later).
        flat = f_te.reshape(-1, D)
        flat_z = (flat - mu) / sigma
        prob_full = clf.predict_proba(flat_z)[:, 1].reshape(f_te.shape[0], gh, gh)
        pred_grids_te[name] = prob_full.astype(np.float32)

    with open(out_dir / "h10a_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out_dir / "h10a_results.json")
    np.savez(out_dir / "h10a_test_pred_grids.npz",
             rgbs=rgb_te, labels=lbl_te, **pred_grids_te)
    log.info("wrote %s", out_dir / "h10a_test_pred_grids.npz")

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["dinov2_base", "dinov2_large", "standalone_siglip",
             "pi0_siglip", "pi0_adapter", "pi05_siglip", "pi05_adapter",
             "openvla_siglip"]
    names = [n for n in order if n in results]
    test_bal = [results[n]["test"]["balanced_accuracy"] for n in names]
    test_auc = [results[n]["test"]["auc"] for n in names]
    test_f1 = [results[n]["test"]["f1_cut"] for n in names]

    color_map = {
        "dinov2_base": "#1e5b34", "dinov2_large": "#3a8a4f",
        "standalone_siglip": "#4a72c9",
        "pi0_siglip": "#c94a72", "pi0_adapter": "#d3a45f",
        "pi05_siglip": "#aa6c39", "pi05_adapter": "#7a4d27",
        "openvla_siglip": "#5fa05f",
    }
    colors = [color_map[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=140)
    for ax, vals, ylab, title in [
        (axes[0], test_bal, "balanced accuracy",
         "H10a: cut vs rest-foreground (test bal-acc)"),
        (axes[1], test_auc, "AUC",
         "H10a: cut vs rest-foreground (test AUC)"),
        (axes[2], test_f1, "F1 cut",
         "H10a: cut vs rest-foreground (test F1 cut)"),
    ]:
        bars = ax.bar(np.arange(len(names)), vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    fig.suptitle("H10a — cut-class detection across the full UMD test set\n"
                 "(binary cut vs other foreground; >32K patches across all categories)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "h10a_cut_vs_rest.png")
    log.info("wrote %s", out_dir / "h10a_cut_vs_rest.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/h10-multitool/results")
    args = ap.parse_args()
    main(args)
