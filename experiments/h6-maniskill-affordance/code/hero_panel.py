"""Combine H6 figures into a single 16:9 hero panel for the talk."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main(args):
    figs = Path("outputs/figures")
    rob = np.asarray(Image.open(figs / "h6_robustness_pickcube.png").convert("RGB"))
    pred = np.asarray(Image.open(figs / "h6_predictor_quality.png").convert("RGB"))
    rec = np.asarray(Image.open(figs / "h6_recovery_pickcube.png").convert("RGB"))

    fig = plt.figure(figsize=(16, 9), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.05, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    for ax, img in [(ax1, rob), (ax2, pred), (ax3, rec)]:
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "H6 — Pretrained policy is fragile to perception noise; "
        "π0's vision tower predicts cube_pos better than DINOv2 (H2's preserved-contain prediction).",
        fontsize=13,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    logging.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/h6_hero_panel.png")
    args = ap.parse_args()
    main(args)
