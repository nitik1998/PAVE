"""Linear-probe regularization ablation: vary C in LogisticRegression.

Re-uses pre-extracted DINOv2 features so each LR fit takes ~30 s.
Output: outputs/tables/c_ablation.csv + outputs/figures/c_ablation.png.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from src.eval.dataset_umd import UMDSubset
from src.eval.metrics import compute_metrics
from src.methods.dinov2_probe import build as build_dinov2
from src.utils.seed import set_seed


def _predict_to_label(pred_chw: np.ndarray) -> np.ndarray:
    bg = np.clip(1.0 - pred_chw.max(axis=0, keepdims=True), 0, 1)
    full = np.concatenate([bg, pred_chw], axis=0)
    return full.argmax(axis=0).astype(np.uint8)


def main(out_csv: str, out_fig: str, c_values: list[float], image_size: int):
    log = logging.getLogger("c_ablation")
    set_seed(0)
    train = UMDSubset.from_split_file("data/umd/splits/train.json",
                                      "configs/affordance_taxonomy.yaml",
                                      image_size=image_size)
    val = UMDSubset.from_split_file("data/umd/splits/val.json",
                                    "configs/affordance_taxonomy.yaml",
                                    image_size=image_size)

    foreground = [n for n in train.class_names if n != "background"]
    probe = build_dinov2(num_classes=len(train.class_names),
                         foreground_names=foreground, device="cpu")
    probe.cfg.image_size = image_size
    probe.cfg.patch_size = 14
    probe.warmup()

    log.info("Extracting features ...")
    Xs, ys = [], []
    for s, rgb, lbl in train:
        Xs.append(probe._extract_patch_features(rgb))
        ys.append(probe._pool_label_to_patches(lbl))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    val_pairs = [(rgb, lbl) for s, rgb, lbl in val]

    rows = []
    from sklearn.linear_model import LogisticRegression

    for c in c_values:
        clf = LogisticRegression(solver="lbfgs", C=c, max_iter=1000, n_jobs=-1)
        clf.fit(X, y)
        probe._clf = clf
        cm_total = np.zeros((len(train.class_names), len(train.class_names)), dtype=np.int64)
        for rgb, lbl in val_pairs:
            pred = probe.predict_map(rgb)
            cm_total += compute_metrics(_predict_to_label(pred), lbl,
                                        num_classes=len(train.class_names)).confusion
        tp = np.diag(cm_total)
        denom = tp + cm_total.sum(0) - tp + cm_total.sum(1) - tp
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        rows.append({"C": c, "mIoU": float(np.nanmean(iou))})
        log.info("C=%.4g  mIoU=%.4f", c, rows[-1]["mIoU"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=140)
    ax.semilogx([r["C"] for r in rows], [r["mIoU"] for r in rows], marker="o", linewidth=2, color="#c94a72")
    for r in rows:
        ax.text(r["C"], r["mIoU"] + 0.005, f"{r['mIoU']:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("LogisticRegression C (inverse regularization)")
    ax.set_ylabel("mIoU on UMD val (n=28)")
    ax.set_title(f"DINOv2-base @ {image_size}² — regularization sweep")
    ax.set_ylim(0, max(0.7, max(r["mIoU"] for r in rows) * 1.1))
    fig.tight_layout()
    fig.savefig(out_fig)
    log.info("wrote %s and %s", out_csv, out_fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default="outputs/tables/c_ablation.csv")
    ap.add_argument("--out-fig", default="outputs/figures/c_ablation.png")
    ap.add_argument("--C", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--image-size", type=int, default=448)
    args = ap.parse_args()
    main(args.out_csv, args.out_fig, args.C, args.image_size)
