"""Plot SAC+HER training curves across the 4 H3 arms with seed shading.

Reads outputs/h3/{A,B,C,D}/seed*/train_log.csv (success_rate_window col)
and outputs/h3/{arm}/seed*/eval.json. Produces:
  - outputs/figures/policy_curves.png : 4 lines, mean ± std across seeds
  - outputs/figures/policy_final_bar.png : final success rate bar chart
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np


def _load_arm(root: Path, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Returns (steps, mean_curve, std_curve, final_success_per_seed)."""
    arm_dir = root / arm
    seeds = sorted(arm_dir.glob("seed*"))
    curves = []
    finals = []
    for sd in seeds:
        log_csv = sd / "train_log.csv"
        eval_json = sd / "eval.json"
        if log_csv.exists():
            with open(log_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                steps = np.array([int(r["step"]) for r in rows])
                rate = np.array([float(r["success_rate_window"]) for r in rows])
                curves.append((steps, rate))
        if eval_json.exists():
            d = json.load(open(eval_json))
            finals.append(float(d.get("success_rate", 0.0)))
    if not curves:
        return np.array([]), np.array([]), np.array([]), finals

    def _smooth(y: np.ndarray, w: int = 25) -> np.ndarray:
        if len(y) < 3:
            return y
        # cumulative-mean style windowed moving average.
        w = min(w, len(y))
        kernel = np.ones(w) / w
        # 'same' pads with reflection-equivalent at the edges via linear ramp.
        padded = np.concatenate([np.full(w // 2, y[0]), y, np.full(w - 1 - w // 2, y[-1])])
        return np.convolve(padded, kernel, mode="valid")

    grid = curves[0][0]
    interp = np.stack([_smooth(np.interp(grid, c[0], c[1])) for c in curves])
    return grid, interp.mean(0), interp.std(0), finals


def main(root: str, out_curves: str, out_bar: str):
    log = logging.getLogger("plot_h3")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_arms = ["A", "B", "C", "D"]
    arm_labels = {
        "A": "A: Full state (baseline)",
        "B": "B: Degraded state (object_pos zeroed)",
        "C": "C: B + ORACLE affordance centroid",
        "D": "D: B + PREDICTED affordance centroid",
    }
    palette = {"A": "#4a72c9", "B": "#888888", "C": "#c94a72", "D": "#4ac972"}
    # Drop arms with no completed seeds.
    arms = []
    for a in all_arms:
        seeds_with_data = list((Path(root) / a).glob("seed*/eval.json"))
        if seeds_with_data:
            arms.append(a)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    finals_by_arm: dict[str, list[float]] = {}
    for arm in arms:
        steps, mean, std, finals = _load_arm(Path(root), arm)
        finals_by_arm[arm] = finals
        if len(steps) == 0:
            log.warning("No data for arm %s", arm)
            continue
        ax.plot(steps, mean, color=palette[arm], linewidth=2, label=arm_labels[arm])
        ax.fill_between(steps, mean - std, mean + std, color=palette[arm], alpha=0.18)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Rolling success rate (1k-step window)")
    ax.set_title("H3 — SAC+HER on PandaPush-v3 across 4 observation arms (3 seeds)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_curves).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_curves)
    log.info("wrote %s", out_curves)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5.5, 3.2), dpi=140)
    xs = np.arange(len(arms))
    means = [float(np.mean(finals_by_arm[a])) if finals_by_arm[a] else 0.0 for a in arms]
    stds = [float(np.std(finals_by_arm[a])) if finals_by_arm[a] else 0.0 for a in arms]
    bars = ax2.bar(xs, means, yerr=stds, capsize=4,
                   color=[palette[a] for a in arms])
    ax2.set_xticks(xs)
    ax2.set_xticklabels(arms)
    ax2.set_ylabel("Eval success rate (30 episodes)")
    ax2.set_ylim(0, 1.05)
    for b, m in zip(bars, means):
        ax2.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    ax2.set_title("H3 — final eval success per arm (mean ± std across seeds)")
    fig2.tight_layout()
    fig2.savefig(out_bar)
    log.info("wrote %s", out_bar)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/h3")
    ap.add_argument("--out-curves", default="outputs/figures/policy_curves.png")
    ap.add_argument("--out-bar", default="outputs/figures/policy_final_bar.png")
    args = ap.parse_args()
    main(args.root, args.out_curves, args.out_bar)
