"""Single 16:9 PNG that combines the headline figure assets for one slide:

  ┌──────────────────────────────┬──────────────────────────────┐
  │   probe_miou bar chart        │   per-class IoU bar chart   │
  ├──────────────────────────────┴──────────────────────────────┤
  │            qual_grid (5 rows × N methods, scaled)            │
  └──────────────────────────────────────────────────────────────┘

All inputs are existing PNGs under outputs/figures/. We just read and stack.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main(out_path: str):
    figs = Path("outputs/figures")
    miou = np.asarray(Image.open(figs / "probe_miou.png").convert("RGB"))
    perclass = np.asarray(Image.open(figs / "probe_miou_perclass.png").convert("RGB"))
    qual = np.asarray(Image.open(figs / "qual_grid.png").convert("RGB"))

    fig = plt.figure(figsize=(16, 9), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.2], hspace=0.05, wspace=0.05)
    ax_top_l = fig.add_subplot(gs[0, 0])
    ax_top_r = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, :])
    for ax, img, title in [
        (ax_top_l, miou, ""),
        (ax_top_r, perclass, ""),
        (ax_bot, qual, ""),
    ]:
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=11)
    fig.suptitle(
        "Affordance probing on UMD: linear probes on frozen vision foundation models",
        fontsize=14,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    logging.info("wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/hero_panel.png")
    args = ap.parse_args()
    main(args.out)
