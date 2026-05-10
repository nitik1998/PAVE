"""Drift→IoU correlation across all 3 VLAs (π0, π0.5, OpenVLA)."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np


def load_iou_per_class(p):
    with open(p) as f:
        row = next(csv.DictReader(f))
    return {k.replace("IoU_", ""): float(v) for k, v in row.items() if k.startswith("IoU_")}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("drift_iou_all")
    root = Path(".")
    with open(root / "outputs/mechanism/per_class_drift_all.json") as f:
        drift = json.load(f)

    iou_std = load_iou_per_class(root / "outputs/tables_500/openpi_siglip_overall.csv")
    iou_pi0 = load_iou_per_class(root / "outputs/tables_500/pi0_siglip_overall.csv")
    iou_pi05 = load_iou_per_class(root / "outputs/tables_500/pi05_siglip_overall.csv")
    iou_openvla = load_iou_per_class(root / "outputs/tables_500/openvla_siglip_overall.csv")

    classes = ["grasp", "cut", "scoop", "contain", "support"]

    rows = []
    for c in classes:
        d = drift[c]
        rows.append(dict(
            cls=c,
            drift_pi0=1 - d["cos_standalone_pi0"],
            drift_pi05=1 - d["cos_standalone_pi05"],
            drift_openvla=1 - d["cos_standalone_openvla"],
            drop_pi0=iou_std[c] - iou_pi0[c],
            drop_pi05=iou_std[c] - iou_pi05[c],
            drop_openvla=iou_std[c] - iou_openvla[c],
        ))

    from scipy.stats import pearsonr, spearmanr

    summary = {}
    for tag in ("pi0", "pi05", "openvla"):
        x = np.array([r[f"drift_{tag}"] for r in rows])
        y = np.array([r[f"drop_{tag}"] for r in rows])
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        summary[tag] = dict(pearson=float(pr), pearson_p=float(pp),
                            spearman=float(sr), spearman_p=float(sp))
        log.info("%s — Pearson r=%.3f (p=%.3f), Spearman ρ=%.3f (p=%.3f)",
                 tag, pr, pp, sr, sp)

    out_dir = root / "outputs/mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "correlation_summary_all.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=140)
    palette = {"pi0": "#c94a72", "pi05": "#aa6c39", "openvla": "#3a8a4f"}
    markers = {"pi0": "o", "pi05": "s", "openvla": "^"}
    label_map = {"pi0": "π0", "pi05": "π0.5", "openvla": "OpenVLA"}
    for tag in ("pi0", "pi05", "openvla"):
        x = np.array([r[f"drift_{tag}"] for r in rows])
        y = np.array([r[f"drop_{tag}"] for r in rows])
        ax.scatter(x, y, color=palette[tag], marker=markers[tag], s=110,
                   edgecolor="black",
                   label=f"{label_map[tag]}  (ρ={summary[tag]['spearman']:.2f}, p={summary[tag]['spearman_p']:.3f})")
        for r in rows:
            ax.annotate(r["cls"], (r[f"drift_{tag}"] + 0.005, r[f"drop_{tag}"]),
                        fontsize=8, color=palette[tag])
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, m * xx + b, "--", color=palette[tag], alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("per-class final-layer drift (1 − cosine similarity)")
    ax.set_ylabel("per-class IoU drop vs standalone SigLIP-So400m")
    ax.set_title("Mechanism → behavior across 3 VLA families\n"
                 "Per-class final-layer drift predicts per-class IoU degradation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "drift_vs_iou_all.png")
    log.info("wrote %s", out_dir / "drift_vs_iou_all.png")


if __name__ == "__main__":
    main()
