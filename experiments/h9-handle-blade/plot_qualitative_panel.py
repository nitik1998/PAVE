"""Qualitative side-by-side panel: each encoder's handle/blade prediction
overlaid on UMD knife test images. The slide-ready figure for H9.

Layout: rows = test images (5 selected); columns = (RGB, GT, then each encoder).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def overlay(ax, rgb, prob_map, title=None, alpha=0.55):
    """Overlay a blade-probability heatmap (red=handle, blue=blade) on RGB."""
    H, W = rgb.shape[:2]
    # Upsample prob_map to image size.
    from PIL import Image

    pmap = Image.fromarray((prob_map * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    pmap = np.asarray(pmap, dtype=np.float32) / 255.0  # in [0, 1]

    ax.imshow(rgb)
    # Colorize: red where prob<0.5 (handle), blue where prob>0.5 (blade).
    overlay_img = np.zeros((H, W, 4), dtype=np.float32)
    handle_strength = (1.0 - pmap)
    blade_strength = pmap
    overlay_img[..., 0] = handle_strength       # red channel = handle
    overlay_img[..., 2] = blade_strength        # blue channel = blade
    overlay_img[..., 3] = np.maximum(handle_strength, blade_strength) * alpha
    ax.imshow(overlay_img)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def gt_overlay(ax, rgb, label, title=None, alpha=0.55):
    """Ground-truth overlay: handle (label==1) in red, blade (label==2) in blue."""
    H, W = rgb.shape[:2]
    ax.imshow(rgb)
    overlay_img = np.zeros((H, W, 4), dtype=np.float32)
    overlay_img[..., 0] = (label == 1).astype(np.float32)
    overlay_img[..., 2] = (label == 2).astype(np.float32)
    overlay_img[..., 3] = ((label == 1) | (label == 2)).astype(np.float32) * alpha
    ax.imshow(overlay_img)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def main(args):
    data = np.load(ROOT / args.data)
    rgbs = data["rgbs"]                    # (N, 224, 224, 3) uint8
    labels = data["labels"]                # (N, 224, 224)
    enc_names = ["dinov2_base", "dinov2_large", "standalone_siglip",
                 "pi0_siglip", "pi0_adapter", "pi05_siglip", "pi05_adapter",
                 "openvla_siglip"]
    enc_names = [n for n in enc_names if n in data.files]
    pred_grids = {n: data[n] for n in enc_names}  # each (N, gh, gh)

    # Pick a small number of indicative images: one shears, two knives, one scissors, one saw.
    N = rgbs.shape[0]
    # Heuristic: pick images with both handle (>50 px) and blade (>50 px).
    keep = []
    for i in range(N):
        n_h = (labels[i] == 1).sum()
        n_b = (labels[i] == 2).sum()
        if n_h > 200 and n_b > 200:
            keep.append((i, n_h + n_b))
    keep.sort(key=lambda x: -x[1])
    selected = [k[0] for k in keep[:args.n_rows]]
    print(f"selected indices: {selected}")

    n_rows = len(selected)
    n_cols = 2 + len(enc_names)  # RGB + GT + encoders
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows), dpi=120)
    if n_rows == 1:
        axes = axes[None, :]

    pretty_names = {
        "dinov2_base": "DINOv2-base",
        "dinov2_large": "DINOv2-large",
        "standalone_siglip": "SigLIP-So400m\n(standalone)",
        "pi0_siglip": "π0 SigLIP",
        "pi0_adapter": "π0 + adapter\n(ours)",
        "pi05_siglip": "π0.5 SigLIP",
        "pi05_adapter": "π0.5 + adapter\n(ours)",
        "openvla_siglip": "OpenVLA SigLIP",
    }

    for r, idx in enumerate(selected):
        rgb = rgbs[idx]
        lbl = labels[idx]
        # Col 0: RGB.
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("RGB", fontsize=9)
        # Col 1: GT.
        gt_overlay(axes[r, 1], rgb, lbl,
                   title="GT (red=handle, blue=blade)" if r == 0 else None)
        # Cols 2..: encoder predictions.
        for c, name in enumerate(enc_names):
            ax = axes[r, 2 + c]
            overlay(ax, rgb, pred_grids[name][idx],
                    title=pretty_names[name] if r == 0 else None)

    fig.suptitle("H9 — Per-patch handle/blade prediction across encoders\n"
                 "(red = predicted handle, blue = predicted blade)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/h9-handle-blade/results/h9_test_pred_grids.npz")
    ap.add_argument("--n-rows", type=int, default=5)
    ap.add_argument("--out", default="outputs/figures/h9_qualitative_panel.png")
    args = ap.parse_args()
    main(args)
