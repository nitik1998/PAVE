"""Test whether per-class final-layer feature drift predicts per-class IoU drop.

The mechanism→behavior link: if VLA fine-tuning destroys class-c feature
geometry, the linear-probe IoU on class c should drop. We compute Pearson
and Spearman correlations between:

  x = 1 - cos_similarity(standalone_class_mean, π0_class_mean)   (drift)
  y = IoU(standalone, class) - IoU(π0, class)                    (degradation)

across the 5 foreground classes, for π0 and π0.5 separately. A strong
positive correlation closes the loop "mechanism predicts measured behavior."
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np


def load_iou_per_class(csv_path: Path) -> dict[str, float]:
    with open(csv_path) as f:
        row = next(csv.DictReader(f))
    return {k.replace("IoU_", ""): float(v) for k, v in row.items() if k.startswith("IoU_")}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("drift_iou")

    root = Path(".")
    drift_path = root / "outputs/mechanism/per_class_drift.json"
    with open(drift_path) as f:
        drift = json.load(f)

    iou_standalone = load_iou_per_class(root / "outputs/tables_500/openpi_siglip_overall.csv")
    iou_pi0 = load_iou_per_class(root / "outputs/tables_500/pi0_siglip_overall.csv")
    iou_pi05 = load_iou_per_class(root / "outputs/tables_500/pi05_siglip_overall.csv")

    classes = ["grasp", "cut", "scoop", "contain", "support"]

    rows = []
    for c in classes:
        d = drift[c]
        rows.append(dict(
            cls=c,
            drift_pi0=1.0 - d["cos_standalone_pi0"],
            drift_pi05=1.0 - d["cos_standalone_pi05"],
            iou_standalone=iou_standalone[c],
            iou_pi0=iou_pi0[c],
            iou_pi05=iou_pi05[c],
            iou_drop_pi0=iou_standalone[c] - iou_pi0[c],
            iou_drop_pi05=iou_standalone[c] - iou_pi05[c],
        ))
    out_csv = root / "outputs/mechanism/drift_vs_iou.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("wrote %s", out_csv)
    for r in rows:
        log.info("%s: drift_pi0=%.3f drop_pi0=%.3f drift_pi05=%.3f drop_pi05=%.3f",
                 r["cls"], r["drift_pi0"], r["iou_drop_pi0"],
                 r["drift_pi05"], r["iou_drop_pi05"])

    # Correlations.
    from scipy.stats import pearsonr, spearmanr

    x_pi0 = np.array([r["drift_pi0"] for r in rows])
    y_pi0 = np.array([r["iou_drop_pi0"] for r in rows])
    x_pi05 = np.array([r["drift_pi05"] for r in rows])
    y_pi05 = np.array([r["iou_drop_pi05"] for r in rows])

    pearsP, pearsP_p = pearsonr(x_pi0, y_pi0)
    spearP, spearP_p = spearmanr(x_pi0, y_pi0)
    pearsQ, pearsQ_p = pearsonr(x_pi05, y_pi05)
    spearQ, spearQ_p = spearmanr(x_pi05, y_pi05)

    summary = dict(
        pi0=dict(pearson=float(pearsP), pearson_p=float(pearsP_p),
                 spearman=float(spearP), spearman_p=float(spearP_p)),
        pi05=dict(pearson=float(pearsQ), pearson_p=float(pearsQ_p),
                  spearman=float(spearQ), spearman_p=float(spearQ_p)),
        n_classes=int(len(classes)),
    )
    with open(root / "outputs/mechanism/correlation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("π0 — Pearson r=%.3f (p=%.3f), Spearman ρ=%.3f (p=%.3f)",
             pearsP, pearsP_p, spearP, spearP_p)
    log.info("π0.5 — Pearson r=%.3f (p=%.3f), Spearman ρ=%.3f (p=%.3f)",
             pearsQ, pearsQ_p, spearQ, spearQ_p)

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=140)
    for r in rows:
        ax.scatter(r["drift_pi0"], r["iou_drop_pi0"], color="#c94a72", s=110, edgecolor="black",
                   label="π0" if r["cls"] == "grasp" else None)
        ax.annotate(r["cls"], (r["drift_pi0"] + 0.005, r["iou_drop_pi0"]), fontsize=9, color="#c94a72")
        ax.scatter(r["drift_pi05"], r["iou_drop_pi05"], color="#aa6c39", s=110, edgecolor="black", marker="s",
                   label="π0.5" if r["cls"] == "grasp" else None)
        ax.annotate(r["cls"], (r["drift_pi05"] + 0.005, r["iou_drop_pi05"]), fontsize=9, color="#aa6c39")
    # Trend lines.
    for x, y, color in [(x_pi0, y_pi0, "#c94a72"), (x_pi05, y_pi05, "#aa6c39")]:
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, m * xx + b, "--", color=color, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("per-class final-layer feature drift (1 − cosine similarity)")
    ax.set_ylabel("per-class IoU drop vs standalone SigLIP-So400m")
    ax.set_title(
        f"Mechanism → behavior:\n"
        f"π0 Pearson r={pearsP:.3f} (p={pearsP_p:.3f}),  "
        f"π0.5 Pearson r={pearsQ:.3f} (p={pearsQ_p:.3f})"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(root / "outputs/mechanism/drift_vs_iou.png")
    log.info("wrote outputs/mechanism/drift_vs_iou.png")


if __name__ == "__main__":
    main()
