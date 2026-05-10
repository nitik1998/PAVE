"""Probe data-scaling curve: mIoU vs n_train for DINOv2 @ 448.

Runs the linear probe at multiple train-set sizes, evaluates on val,
and writes a CSV + line plot.
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


def main(out_csv: str, out_fig: str, sizes: list[int], image_size: int):
    log = logging.getLogger("scaling_curve")
    set_seed(0)
    train = UMDSubset.from_split_file("data/umd/splits/train.json",
                                      "configs/affordance_taxonomy.yaml",
                                      image_size=image_size)
    val = UMDSubset.from_split_file("data/umd/splits/val.json",
                                    "configs/affordance_taxonomy.yaml",
                                    image_size=image_size)

    # Pre-extract all train and val features ONCE.
    foreground = [n for n in train.class_names if n != "background"]
    probe = build_dinov2(num_classes=len(train.class_names),
                         foreground_names=foreground, device="cpu")
    probe.cfg.image_size = image_size
    probe.cfg.patch_size = 14
    probe.warmup()

    log.info("Extracting train features (%d images) ...", len(train))
    train_feats: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    for s, rgb, lbl in train:
        f = probe._extract_patch_features(rgb)
        y = probe._pool_label_to_patches(lbl)
        train_feats.append(f)
        train_labels.append(y)

    log.info("Extracting val features (%d images) ...", len(val))
    val_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for s, rgb, lbl in val:
        val_pairs.append((rgb, lbl))

    rows = []
    for n in sizes:
        n = min(n, len(train_feats))
        X = np.concatenate(train_feats[:n], axis=0)
        y = np.concatenate(train_labels[:n], axis=0)
        log.info("Fitting LR on %d images (X=%s, y=%s)", n, X.shape, y.shape)
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, n_jobs=-1)
        clf.fit(X, y)
        probe._clf = clf

        # Eval on val.
        cm_total = np.zeros((len(train.class_names), len(train.class_names)), dtype=np.int64)
        for rgb, lbl in val_pairs:
            pred = probe.predict_map(rgb)
            pred_label = _predict_to_label(pred)
            m = compute_metrics(pred_label, lbl, num_classes=len(train.class_names))
            cm_total += m.confusion
        tp = np.diag(cm_total)
        denom = tp + cm_total.sum(0) - tp + cm_total.sum(1) - tp
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        rows.append({"n_train": n, "mIoU": float(np.nanmean(iou))})
        log.info("n_train=%d  mIoU=%.4f", n, rows[-1]["mIoU"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=140)
    xs = [r["n_train"] for r in rows]
    ys = [r["mIoU"] for r in rows]
    ax.plot(xs, ys, marker="o", linewidth=2, color="#4a72c9")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("Number of UMD train images used to fit the linear probe")
    ax.set_ylabel("mIoU on UMD val (n=28)")
    ax.set_title(f"DINOv2-base @ {image_size}² — data scaling")
    ax.set_ylim(0, max(0.7, max(ys) * 1.1))
    fig.tight_layout()
    fig.savefig(out_fig)
    log.info("wrote %s and %s", out_csv, out_fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default="outputs/tables/scaling_curve.csv")
    ap.add_argument("--out-fig", default="outputs/figures/scaling_curve.png")
    ap.add_argument("--sizes", nargs="+", type=int, default=[10, 30, 60, 100, 130])
    ap.add_argument("--image-size", type=int, default=448)
    args = ap.parse_args()
    main(args.out_csv, args.out_fig, args.sizes, args.image_size)
