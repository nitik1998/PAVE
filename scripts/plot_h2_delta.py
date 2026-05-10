"""H2 figure: per-class IoU comparison between standalone SigLIP-So400m and
π0-extracted SigLIP. The delta = "affordance representation drift caused by
VLA fine-tuning."
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict:
    with open(path) as f:
        return next(csv.DictReader(f))


def main(out: str, val_root: str, test_root: str):
    log = logging.getLogger("h2_delta")
    classes = ["grasp", "cut", "scoop", "contain", "support"]

    rows = {
        ("val", "standalone"): _read(Path(val_root) / "openpi_siglip_overall.csv"),
        ("val", "pi0"): _read(Path(val_root) / "pi0_siglip_overall.csv"),
        ("test", "standalone"): _read(Path(test_root) / "openpi_siglip_overall.csv"),
        ("test", "pi0"): _read(Path(test_root) / "pi0_siglip_overall.csv"),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=140, sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        std = rows[(split, "standalone")]
        pi = rows[(split, "pi0")]
        std_vals = [float(std[f"IoU_{c}"]) for c in classes]
        pi_vals = [float(pi[f"IoU_{c}"]) for c in classes]
        deltas = [p - s for p, s in zip(pi_vals, std_vals)]
        xs = np.arange(len(classes))
        w = 0.35
        ax.bar(xs - w / 2, std_vals, w, label="standalone SigLIP-So400m", color="#4a72c9")
        ax.bar(xs + w / 2, pi_vals, w, label=r"$\pi_0$ SigLIP (post-VLA)", color="#c94a72")
        for x, d in zip(xs, deltas):
            ax.text(x, max(std_vals + pi_vals) * 0.05 + 0.85,
                    f"{d:+.2f}", ha="center", fontsize=9,
                    color="#c94a72" if d < 0 else "#2a8a2a")
        std_miou = float(std["mIoU"])
        pi_miou = float(pi["mIoU"])
        ax.set_xticks(xs)
        ax.set_xticklabels(classes)
        ax.set_ylabel("IoU")
        ax.set_title(
            f"{split} (n={int(float(std['n']))}) — Δ mIoU = {pi_miou - std_miou:+.3f}"
        )
        ax.set_ylim(0, 0.95)
        ax.grid(True, axis="y", alpha=0.3)
        if split == "val":
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(r"H2: VLA fine-tuning shifts the affordance representation. "
                 r"Per-class IoU($\pi_0$) − IoU(standalone) above each bar group.",
                 fontsize=11)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    log.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default="outputs/tables_500")
    ap.add_argument("--test-root", default="outputs/tables_500_test")
    ap.add_argument("--out", default="outputs/figures/h2_delta.png")
    args = ap.parse_args()
    main(args.out, args.val_root, args.test_root)
