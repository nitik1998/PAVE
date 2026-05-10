"""Qualitative panel: predicted cut probability map on multi-tool composites,
per encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def overlay_blade(ax, rgb, prob, title=None, alpha=0.55):
    H, W = rgb.shape[:2]
    from PIL import Image

    pmap = Image.fromarray((prob * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    pmap = np.asarray(pmap, dtype=np.float32) / 255.0
    ax.imshow(rgb)
    ov = np.zeros((H, W, 4), dtype=np.float32)
    ov[..., 0] = (1.0 - pmap)
    ov[..., 2] = pmap
    ov[..., 3] = np.maximum(1.0 - pmap, pmap) * alpha
    ax.imshow(ov)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def gt_overlay(ax, rgb, lbl, title=None, alpha=0.55):
    H, W = rgb.shape[:2]
    ax.imshow(rgb)
    ov = np.zeros((H, W, 4), dtype=np.float32)
    fg = lbl > 0
    cut = (lbl == 2)
    ov[..., 2] = cut.astype(np.float32)
    ov[..., 0] = (fg & ~cut).astype(np.float32)
    ov[..., 3] = fg.astype(np.float32) * alpha
    ax.imshow(ov)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def main(args):
    data = np.load(ROOT / args.data)
    rgbs = data["rgbs"]
    labels = data["labels"]
    enc_names = ["dinov2_base", "dinov2_large", "standalone_siglip",
                 "pi0_siglip", "pi05_siglip",
                 "pi0_adapter", "pi05_adapter"]
    enc_names = [n for n in enc_names if n in data.files]
    pred_grids = {n: data[n] for n in enc_names}

    pretty = {
        "dinov2_base": "DINOv2-base",
        "dinov2_large": "DINOv2-large",
        "standalone_siglip": "SigLIP-So400m",
        "pi0_siglip": "π0 SigLIP",
        "pi05_siglip": "π0.5 SigLIP",
        "pi0_adapter": "π0+adapter",
        "pi05_adapter": "π0.5+adapter",
    }

    # Pick composites with both cut and non-cut foreground present.
    n = rgbs.shape[0]
    keep = []
    for i in range(n):
        n_cut = (labels[i] == 2).sum()
        n_other_fg = ((labels[i] > 0) & (labels[i] != 2)).sum()
        if n_cut > 200 and n_other_fg > 200:
            keep.append((i, n_cut + n_other_fg))
    keep.sort(key=lambda x: -x[1])
    selected = [k[0] for k in keep[:args.n_rows]]
    print(f"selected: {selected}")

    n_rows = len(selected)
    n_cols = 2 + len(enc_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows), dpi=120)
    if n_rows == 1:
        axes = axes[None, :]

    for r, idx in enumerate(selected):
        rgb = rgbs[idx]
        lbl = labels[idx]
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("composite RGB", fontsize=9)
        gt_overlay(axes[r, 1], rgb, lbl,
                   title="GT (red=other, blue=cut)" if r == 0 else None)
        for c, name in enumerate(enc_names):
            overlay_blade(axes[r, 2 + c], rgb, pred_grids[name][idx],
                          title=pretty[name] if r == 0 else None)

    fig.suptitle("H10b — multi-tool composite cut detection (red=predicted not-cut, blue=predicted cut)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="experiments/h10-multitool/results/h10b_cmp_pred_grids.npz")
    ap.add_argument("--n-rows", type=int, default=4)
    ap.add_argument("--out", default="outputs/figures/h10b_qualitative.png")
    args = ap.parse_args()
    main(args)
