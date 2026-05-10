"""Per-class IoU comparison across standalone SigLIP-So400m, π0, and π0.5.

Shows the asymmetric degradation (H2) plus partial recovery in π0.5 (H5).
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


def main(out: str, val_root: str):
    classes = ["grasp", "cut", "scoop", "contain", "support"]

    standalone = _read(Path(val_root) / "openpi_siglip_overall.csv")
    pi0 = _read(Path(val_root) / "pi0_siglip_overall.csv")
    pi05 = _read(Path(val_root) / "pi05_siglip_overall.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    xs = np.arange(len(classes))
    w = 0.27

    std_vals = [float(standalone[f"IoU_{c}"]) for c in classes]
    pi0_vals = [float(pi0[f"IoU_{c}"]) for c in classes]
    pi05_vals = [float(pi05[f"IoU_{c}"]) for c in classes]

    ax.bar(xs - w, std_vals, w, label=f"standalone SigLIP-So400m  (mIoU {float(standalone['mIoU']):.3f})", color="#4a72c9")
    ax.bar(xs, pi0_vals, w, label=f"π0 SigLIP (post-VLA)  (mIoU {float(pi0['mIoU']):.3f})", color="#c94a72")
    ax.bar(xs + w, pi05_vals, w, label=f"π0.5 SigLIP (improved recipe)  (mIoU {float(pi05['mIoU']):.3f})", color="#aa6c39")

    for i, c in enumerate(classes):
        d_pi0 = pi0_vals[i] - std_vals[i]
        d_pi05 = pi05_vals[i] - std_vals[i]
        ax.text(xs[i] - w, std_vals[i] + 0.01, f"{std_vals[i]:.2f}", ha="center", fontsize=8)
        ax.text(xs[i], pi0_vals[i] + 0.01, f"{pi0_vals[i]:.2f}\n({d_pi0:+.2f})", ha="center", fontsize=7, color="#aa274a")
        ax.text(xs[i] + w, pi05_vals[i] + 0.01, f"{pi05_vals[i]:.2f}\n({d_pi05:+.2f})", ha="center", fontsize=7, color="#7a4d27")

    ax.set_xticks(xs)
    ax.set_xticklabels(classes)
    ax.set_ylabel("IoU")
    ax.set_title(f"H2 + H5 — VLA fine-tuning degrades affordance class-asymmetrically; π0.5's recipe partially recovers cut/support")
    ax.set_ylim(0, max(std_vals + pi0_vals + pi05_vals) * 1.2)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default="outputs/tables_500")
    ap.add_argument("--out", default="outputs/figures/h2_h5_delta.png")
    args = ap.parse_args()
    main(args.out, args.val_root)
