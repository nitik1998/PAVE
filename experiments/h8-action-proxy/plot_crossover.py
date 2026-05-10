"""H8 crossover figure: combine PickCube (contain task) + UMD-cut (cut task) results.

Tests the central question: do encoders that preserve cut affordance predict
cut-task quantities better, while encoders that preserve contain affordance
predict contain-task quantities better?

If yes: a single encoder choice trades off across affordance classes — i.e.,
affordance perception is causally relevant for downstream prediction tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    res_dir = ROOT / "experiments/h8-action-proxy/results"
    with open(res_dir / "pickcube_action_results.json") as f:
        pc = json.load(f)
    with open(res_dir / "umd_cut_action_results.json") as f:
        cut = json.load(f)

    encoders = ["dinov2_base", "dinov2_large", "standalone_siglip",
                "pi0_siglip", "pi0_adapter", "pi05_siglip", "pi05_adapter",
                "openvla_siglip"]
    encoders = [e for e in encoders if e in pc and e in cut]

    pc_l2 = [pc[e]["mean_l2"] for e in encoders]
    cut_angle = [cut[e]["orient_test_angle_deg_mean"] for e in encoders]
    cut_handle = [cut[e]["handle"]["test"]["mean_l2"] for e in encoders]

    color_map = {
        "dinov2_base": "#1e5b34",
        "dinov2_large": "#3a8a4f",
        "standalone_siglip": "#4a72c9",
        "pi0_siglip": "#c94a72",
        "pi0_adapter": "#d3a45f",
        "pi05_siglip": "#aa6c39",
        "pi05_adapter": "#7a4d27",
        "openvla_siglip": "#3a8a4f",
    }
    colors = [color_map[e] for e in encoders]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=140)

    ax = axes[0]
    bars = ax.bar(np.arange(len(encoders)), pc_l2, color=colors)
    for b, v in zip(bars, pc_l2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(encoders)))
    ax.set_xticklabels([e.replace("_", "\n") for e in encoders], fontsize=8)
    ax.set_ylabel("Mean L2 action error")
    ax.set_title("contain-task: PickCube action prediction\n(lower = better)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(np.arange(len(encoders)), cut_angle, color=colors)
    for b, v in zip(bars, cut_angle):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}°",
                ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(encoders)))
    ax.set_xticklabels([e.replace("_", "\n") for e in encoders], fontsize=8)
    ax.set_ylabel("Mean angle error (deg)")
    ax.set_title("cut-task: knife orientation prediction\n(blade→handle direction, lower = better)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    bars = ax.bar(np.arange(len(encoders)), cut_handle, color=colors)
    for b, v in zip(bars, cut_handle):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(encoders)))
    ax.set_xticklabels([e.replace("_", "\n") for e in encoders], fontsize=8)
    ax.set_ylabel("Handle-centroid L2 (image plane)")
    ax.set_title("cut-task: handle position prediction\n(lower = better)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("H8 — Action-prediction crossover: do affordance-preserving encoders predict task-relevant quantities better?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / "outputs/figures/h8_crossover.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)

    # Also plot a 2D scatter: x = contain-task error, y = cut-task error.
    # Encoders that win on contain (low x) but lose on cut (high y) live in
    # the bottom-right; the opposite trade-off is top-left. A "Pareto frontier"
    # would show whether any encoder dominates on both axes.
    fig2, ax2 = plt.subplots(figsize=(7.5, 6), dpi=140)
    for e, x, y in zip(encoders, pc_l2, cut_angle):
        ax2.scatter(x, y, s=140, color=color_map[e], edgecolor="black", zorder=3)
        ax2.annotate(e.replace("_siglip", "").replace("_adapter", "+adp"),
                     (x + 0.005, y + 0.3), fontsize=8)
    ax2.set_xlabel("contain-task error: PickCube action L2 (lower = better)")
    ax2.set_ylabel("cut-task error: knife-orient angle deg (lower = better)")
    ax2.set_title("Encoder trade-off: contain vs cut prediction")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = ROOT / "outputs/figures/h8_pareto.png"
    fig2.savefig(out2)
    print("wrote", out2)


if __name__ == "__main__":
    main()
