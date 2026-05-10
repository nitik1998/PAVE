"""Per-class IoU comparison across standalone SigLIP-So400m, π0, π0.5,
and OpenVLA. The OpenVLA bar makes the asymmetric-degradation finding
generalize across VLA families (Prismatic-Llama vs PaliGemma).
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


def main(out: str, val_root: str, test_root: str | None):
    classes = ["grasp", "cut", "scoop", "contain", "support"]

    standalone = _read(Path(val_root) / "openpi_siglip_overall.csv")
    pi0 = _read(Path(val_root) / "pi0_siglip_overall.csv")
    pi05 = _read(Path(val_root) / "pi05_siglip_overall.csv")
    openvla = _read(Path(val_root) / "openvla_siglip_overall.csv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    xs = np.arange(len(classes))
    w = 0.21

    std_vals = [float(standalone[f"IoU_{c}"]) for c in classes]
    pi0_vals = [float(pi0[f"IoU_{c}"]) for c in classes]
    pi05_vals = [float(pi05[f"IoU_{c}"]) for c in classes]
    openvla_vals = [float(openvla[f"IoU_{c}"]) for c in classes]

    ax.bar(xs - 1.5 * w, std_vals, w,
           label=f"standalone SigLIP-So400m  (mIoU {float(standalone['mIoU']):.3f})", color="#4a72c9")
    ax.bar(xs - 0.5 * w, pi0_vals, w,
           label=f"π0 SigLIP (PaliGemma)  (mIoU {float(pi0['mIoU']):.3f})", color="#c94a72")
    ax.bar(xs + 0.5 * w, pi05_vals, w,
           label=f"π0.5 SigLIP (improved recipe)  (mIoU {float(pi05['mIoU']):.3f})", color="#aa6c39")
    ax.bar(xs + 1.5 * w, openvla_vals, w,
           label=f"OpenVLA SigLIP (Prismatic-Llama)  (mIoU {float(openvla['mIoU']):.3f})", color="#3a8a4f")

    for i in range(len(classes)):
        for x_off, vals, color in [
            (-1.5 * w, std_vals, "black"),
            (-0.5 * w, pi0_vals, "#aa274a"),
            (+0.5 * w, pi05_vals, "#7a4d27"),
            (+1.5 * w, openvla_vals, "#1e5b34"),
        ]:
            ax.text(xs[i] + x_off, vals[i] + 0.005, f"{vals[i]:.2f}",
                    ha="center", fontsize=7.2, color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels(classes)
    ax.set_ylabel("IoU on UMD val")
    ax.set_title("Class-asymmetric VLA degradation generalizes across families:\n"
                 "contain preserved, cut/support most degraded — for all three VLAs")
    ax.set_ylim(0, max(std_vals + pi0_vals + pi05_vals + openvla_vals) * 1.18)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default="outputs/tables_500")
    ap.add_argument("--test-root", default="outputs/tables_500_test")
    ap.add_argument("--out", default="outputs/figures/h2_h5_h7_delta.png")
    args = ap.parse_args()
    main(args.out, args.val_root, args.test_root)
