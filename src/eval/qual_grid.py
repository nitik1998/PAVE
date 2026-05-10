"""Cross-method qualitative figure builder.

Reads a small number of UMD samples and stacks predictions from each method
under `outputs/predictions/<method>/<sample_id>.npy`. Each .npy is a
(C, H, W) float32 array in [0, 1].
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.eval.dataset_umd import UMDSubset
from src.utils.viz import colorize_label_map, grid_figure, overlay_multi_heatmap


def _argmax_to_label(probs: np.ndarray, num_classes: int) -> np.ndarray:
    """probs: (C, H, W) for foreground classes; prepend a background slot."""
    c, h, w = probs.shape
    if c == num_classes:
        full = probs
    else:
        # Insert background as 1 - max(foreground).
        bg = np.clip(1.0 - probs.max(axis=0, keepdims=True), 0, 1)
        full = np.concatenate([bg, probs], axis=0)
    return full.argmax(axis=0).astype(np.uint8)


def main(
    methods: list[str],
    n: int,
    split_file: str,
    taxonomy: str,
    pred_root: str,
    out_path: str,
):
    subset = UMDSubset.from_split_file(split_file, taxonomy, image_size=448)
    samples = subset.samples[:n]
    rows: list[list[np.ndarray]] = []
    row_labels: list[str] = []
    col_labels = ["RGB", "GT"] + methods
    for s in samples:
        rgb = s.load_rgb(size=448)
        gt = s.load_label(subset.mapping, size=448)
        cells = [rgb, colorize_label_map(gt)]
        for m in methods:
            f = Path(pred_root) / m / f"{s.object_id}.npy"
            if f.exists():
                arr = np.load(f).astype(np.float32)
                if arr.shape[1:] != rgb.shape[:2]:
                    import cv2

                    resized = np.zeros((arr.shape[0], rgb.shape[0], rgb.shape[1]), dtype=np.float32)
                    for ci in range(arr.shape[0]):
                        resized[ci] = cv2.resize(arr[ci], (rgb.shape[1], rgb.shape[0]),
                                                 interpolation=cv2.INTER_LINEAR)
                    arr = resized
                cells.append(overlay_multi_heatmap(rgb, arr))
            else:
                placeholder = np.zeros_like(rgb)
                cells.append(placeholder)
        rows.append(cells)
        row_labels.append(s.object_id)
    grid_figure(rows, row_labels=row_labels, col_labels=col_labels, out_path=out_path)
    logging.info("wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["dinov3", "siglip2", "qwen25vl", "molmoe"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--split-file", default="data/umd/splits/test.json")
    ap.add_argument("--taxonomy", default="configs/affordance_taxonomy.yaml")
    ap.add_argument("--pred-root", default="outputs/predictions")
    ap.add_argument("--out", default="outputs/figures/qual_grid.png")
    args = ap.parse_args()
    main(args.methods, args.n, args.split_file, args.taxonomy, args.pred_root, args.out)
