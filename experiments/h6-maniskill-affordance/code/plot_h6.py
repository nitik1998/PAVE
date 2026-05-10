"""Generate H6 figures from CSV results.

Inputs:
  - pickcube_robustness.csv  (sigma, success)
  - recovery_pickcube_v2.csv (variant, sigma, success)
  - cubepos_*_v2.joblib      (predictor val L2 errors)

Outputs:
  outputs/figures/h6_robustness_pickcube.png   — single-curve baseline collapse
  outputs/figures/h6_recovery_pickcube.png     — 4 curves (baseline/oracle/dinov2/pi0)
  outputs/figures/h6_predictor_quality.png     — L2 error bar chart per backbone
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_robustness(rob_csv: str, out_path: str):
    rows = load_csv(rob_csv)
    rows = sorted(rows, key=lambda r: float(r["sigma"]))
    xs = [float(r["sigma"]) for r in rows]
    ys = [float(r["mean_success"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    ax.plot(xs, ys, marker="o", linewidth=2, color="#4a72c9", label="Pretrained PPO (PickCube-v1)")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=8)
    ax.set_xlabel("Gaussian noise σ added to cube_pos in observation (m)")
    ax.set_ylabel("Success rate")
    ax.set_title("ManiSkill3 PickCube: pretrained policy collapses under perception noise")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    logging.info("wrote %s", out_path)


def plot_recovery(rec_csv: str, out_path: str):
    rows = load_csv(rec_csv)
    by_variant: dict[str, list] = {}
    for r in rows:
        v = r["variant"]
        by_variant.setdefault(v, []).append((float(r["sigma"]), float(r["mean_success"]), int(r["n_episodes"])))
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    colors = {"baseline": "#4a72c9", "oracle": "#2a8a2a", "dinov2": "#c94a72", "pi0": "#aa6c39"}
    labels = {
        "baseline": "Baseline (noisy obs, no override)",
        "oracle": "Oracle (clean cube_pos override)",
        "dinov2": "DINOv2-predicted cube_pos",
        "pi0": "π0 SigLIP-predicted cube_pos",
    }
    for v, pts in by_variant.items():
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, color=colors.get(v, "#888"),
                label=labels.get(v, v))
    ax.set_xlabel("Gaussian noise σ added to cube_pos in observation (m)")
    ax.set_ylabel("Success rate")
    ax.set_title("H6 — vision-predicted affordance recovers from perception noise")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    logging.info("wrote %s", out_path)


def plot_predictor_quality(blob_paths: list[str], out_path: str):
    import joblib

    names = []
    errs_mean = []
    errs_med = []
    for p in blob_paths:
        d = joblib.load(p)
        names.append(d.get("backbone", Path(p).stem))
        errs_mean.append(d["val_l2_mean"] * 100)  # m → cm
        errs_med.append(d["val_l2_median"] * 100)
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=140)
    xs = np.arange(len(names))
    w = 0.35
    ax.bar(xs - w / 2, errs_mean, w, label="mean", color="#4a72c9")
    ax.bar(xs + w / 2, errs_med, w, label="median", color="#aa6c39")
    for x, m, md in zip(xs, errs_mean, errs_med):
        ax.text(x - w / 2, m + 0.1, f"{m:.2f}", ha="center", fontsize=8)
        ax.text(x + w / 2, md + 0.1, f"{md:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("Cube xyz prediction error (cm)")
    ax.set_title("H6 vision predictor accuracy per backbone (val 100 frames)")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    logging.info("wrote %s", out_path)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.robustness).exists():
        plot_robustness(args.robustness, out_dir / "h6_robustness_pickcube.png")
    else:
        logging.warning("missing %s", args.robustness)
    if Path(args.recovery).exists():
        plot_recovery(args.recovery, out_dir / "h6_recovery_pickcube.png")
    else:
        logging.warning("missing %s", args.recovery)
    if args.predictors:
        plot_predictor_quality(args.predictors, out_dir / "h6_predictor_quality.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--robustness", default="experiments/h6-maniskill-affordance/results/pickcube_robustness.csv")
    ap.add_argument("--recovery", default="experiments/h6-maniskill-affordance/results/recovery_pickcube_v3.csv")
    ap.add_argument("--predictors", nargs="+",
                    default=["experiments/h6-maniskill-affordance/results/cubepos_dinov2_v2.joblib",
                             "experiments/h6-maniskill-affordance/results/cubepos_pi0_v2.joblib"])
    ap.add_argument("--out-dir", default="outputs/figures")
    args = ap.parse_args()
    main(args)
