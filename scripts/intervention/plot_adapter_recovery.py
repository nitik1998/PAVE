"""Hero figure: linear-probe baselines vs MLP adapter on each backbone.

Shows that a 297k-parameter adapter on top of frozen π0 SigLIP recovers most
of the per-class affordance signal that a linear probe could not extract.

Order of bars per class:
    DINOv2-large (linear)   — purely visual ceiling, no VLA influence
    standalone SigLIP-So400m (linear)
    π0 SigLIP (linear)         ← H2 result
    π0.5 SigLIP (linear)       ← H5 result
    π0 SigLIP (adapter)        ← intervention
    π0.5 SigLIP (adapter)      ← intervention
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["grasp", "cut", "scoop", "contain", "support"]


def read_overall(path: Path) -> dict:
    with open(path) as f:
        return next(csv.DictReader(f))


def read_adapter(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    standalone = read_overall(ROOT / "outputs/tables_500/openpi_siglip_overall.csv")
    pi0_lin = read_overall(ROOT / "outputs/tables_500/pi0_siglip_overall.csv")
    pi05_lin = read_overall(ROOT / "outputs/tables_500/pi05_siglip_overall.csv")
    dinov2l = read_overall(ROOT / "outputs/tables_500/dinov2_large_overall.csv")

    pi0_ad = read_adapter(ROOT / "outputs/intervention/adapter_pi0_siglip_h256.json")
    pi05_ad = read_adapter(ROOT / "outputs/intervention/adapter_pi05_siglip_h256.json")
    openvla_lin = read_overall(ROOT / "outputs/tables_500/openvla_siglip_overall.csv")
    openvla_ad = read_adapter(ROOT / "outputs/intervention/adapter_openvla_siglip_h256.json")

    methods = [
        ("standalone SigLIP-So400m (linear)", standalone, "linear", "#4a72c9"),
        ("π0 SigLIP (linear)", pi0_lin, "linear", "#c94a72"),
        ("π0.5 SigLIP (linear)", pi05_lin, "linear", "#aa6c39"),
        ("OpenVLA SigLIP (linear)", openvla_lin, "linear", "#3a8a4f"),
        ("π0 SigLIP + MLP adapter (ours)", pi0_ad, "adapter", "#d3a45f"),
        ("π0.5 SigLIP + MLP adapter (ours)", pi05_ad, "adapter", "#7a4d27"),
        ("OpenVLA SigLIP + MLP adapter (ours)", openvla_ad, "adapter", "#6fbf80"),
    ]
    n_methods = len(methods)

    def per_class_value(item, kind):
        if kind == "linear":
            return [float(item[f"IoU_{c}"]) for c in CLASSES]
        else:
            return [item["val"]["per_class"][c] for c in CLASSES]

    def miou_value(item, kind):
        if kind == "linear":
            return float(item["mIoU"])
        else:
            # adapter dict's "miou" includes bg, recompute foreground-only
            return float(np.mean([item["val"]["per_class"][c] for c in CLASSES]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140,
                                   gridspec_kw=dict(width_ratios=[1, 2.4]))

    # --- panel 1: overall foreground-mean IoU ---
    xs1 = np.arange(n_methods)
    vals1 = [miou_value(m[1], m[2]) for m in methods]
    bars = ax1.bar(xs1, vals1, color=[m[3] for m in methods])
    ax1.set_xticks(xs1)
    ax1.set_xticklabels([m[0].replace(" + MLP adapter (ours)", "\n+adapter (ours)").replace(" (linear)", "")
                         for m in methods], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("foreground-mean IoU on UMD val")
    ax1.set_title("Overall affordance score")
    ax1.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, vals1):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", fontsize=8)
    ax1.set_ylim(0, max(vals1) * 1.18)

    # --- panel 2: per-class IoU bars ---
    xs2 = np.arange(len(CLASSES))
    w = 0.13
    for k, (lbl, item, kind, c) in enumerate(methods):
        vals = per_class_value(item, kind)
        offs = (k - (n_methods - 1) / 2) * w
        bars = ax2.bar(xs2 + offs, vals, w, color=c, edgecolor="black", linewidth=0.3,
                       label=lbl)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(CLASSES)
    ax2.set_ylabel("per-class IoU on UMD val")
    ax2.set_title("Per-class IoU — adapter recovers cut/support lost in π0/π0.5")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax2.set_ylim(0, 1.0)

    fig.suptitle("Frozen-backbone adapter recovery: VLA-degraded affordance is recoverable, not deleted",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / "outputs/figures/adapter_recovery.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
