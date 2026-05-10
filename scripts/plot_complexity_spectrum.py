"""Compose the complexity-spectrum figure linking H2 (5-class), H10a (binary
on whole UMD), H9 (binary on knives only), and H10b (binary on multi-tool
composites). Shows where on the task-difficulty spectrum the VLA cut-class
loss actually appears.

Y-axis: per-encoder cut-class detection score (mIoU for H2, balanced-acc
for H9/H10a/H10b).
X-axis: ordered task complexity (left = harder, right = easier).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_iou(path):
    with open(path) as f:
        row = next(csv.DictReader(f))
    return float(row["IoU_cut"])


def main():
    # H2 — 5-class IoU (cut row).
    h2 = {
        "dinov2_base": read_iou(ROOT / "outputs/tables_500/dinov2_overall.csv"),
        "dinov2_large": read_iou(ROOT / "outputs/tables_500/dinov2_large_overall.csv"),
        "standalone_siglip": read_iou(ROOT / "outputs/tables_500/openpi_siglip_overall.csv"),
        "pi0_siglip": read_iou(ROOT / "outputs/tables_500/pi0_siglip_overall.csv"),
        "pi05_siglip": read_iou(ROOT / "outputs/tables_500/pi05_siglip_overall.csv"),
        "openvla_siglip": read_iou(ROOT / "outputs/tables_500/openvla_siglip_overall.csv"),
    }

    with open(ROOT / "experiments/h10-multitool/results/h10a_results.json") as f:
        h10a = json.load(f)
    h10a_bal = {k: v["test"]["balanced_accuracy"] for k, v in h10a.items()}

    with open(ROOT / "experiments/h9-handle-blade/results/h9_handle_blade_results.json") as f:
        h9 = json.load(f)
    h9_bal = {k: v["test"]["balanced_accuracy"] for k, v in h9.items()}

    h10b_path = ROOT / "experiments/h10-multitool/results/h10b_results.json"
    has_h10b = h10b_path.exists()
    if has_h10b:
        with open(h10b_path) as f:
            h10b = json.load(f)
        h10b_bal = {k: v["composite_test"]["balanced_accuracy"] for k, v in h10b.items()}
    else:
        h10b_bal = {}

    # Encoders we report.
    enc_show = ["dinov2_base", "dinov2_large", "standalone_siglip",
                "pi0_siglip", "pi05_siglip", "openvla_siglip"]

    # Tasks ordered from MOST CHALLENGING to LEAST.
    tasks = [
        ("H2 5-class IoU\n(cut among grasp/scoop/contain/support, all UMD)", h2),
        ("H10a binary cut-vs-rest\n(all UMD, multi-class confusion)", h10a_bal),
    ]
    if has_h10b:
        tasks.append(
            ("H10b multi-tool composite\n(cut on 2-tool scenes, train on single-tool)",
             h10b_bal))
    tasks.append(
        ("H9 binary handle-vs-blade\n(knives only, single-object)", h9_bal),
    )

    color_map = {
        "dinov2_base": "#1e5b34", "dinov2_large": "#3a8a4f",
        "standalone_siglip": "#4a72c9",
        "pi0_siglip": "#c94a72",
        "pi05_siglip": "#aa6c39",
        "openvla_siglip": "#5fa05f",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    xs = np.arange(len(tasks))
    for enc in enc_show:
        ys = []
        for tname, tdict in tasks:
            v = tdict.get(enc, np.nan)
            ys.append(v)
        ax.plot(xs, ys, marker="o", linewidth=2, color=color_map[enc], label=enc.replace("_", " "))
        for x, y in zip(xs, ys):
            if not np.isnan(y):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5),
                            fontsize=7, color=color_map[enc])
    ax.set_xticks(xs)
    ax.set_xticklabels([t[0] for t in tasks], fontsize=8)
    ax.set_ylabel("cut detection metric (IoU or balanced acc)")
    ax.set_title("Complexity spectrum: when does the VLA cut-class loss actually appear?\n"
                 "(left = harder multi-class confusion, right = easier binary-on-single-object)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = ROOT / "outputs/figures/complexity_spectrum.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
