"""Cross-arm robustness evaluation: take trained policies and stress-test
them under perturbations to ``achieved_goal`` at *test time* only.

The story: arm A (vanilla state) and arm C (vanilla state + oracle
affordance) reach similar success rates in-distribution, but C should be
more robust when achieved_goal is perturbed because the affordance channel
provides a redundant view of the object position.

Output:
  outputs/h3/robustness.csv
  outputs/figures/h3_robustness.png

Note: at this layer we only perturb the OBSERVED achieved_goal and the
OBSERVATION's object_pos slice; the env's compute_reward (HER's signal at
training time) is unaffected because we don't re-train. We just deterministically
roll out the trained models with perturbed inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np


def _perturb_obs(obs: dict, noise: float, rng: np.random.Generator) -> dict:
    obs = dict(obs)
    if "achieved_goal" in obs and noise > 0:
        ag = np.asarray(obs["achieved_goal"]).astype(np.float32, copy=True)
        ag = ag + rng.normal(0, noise, size=ag.shape).astype(np.float32)
        obs["achieved_goal"] = ag
    if "observation" in obs and noise > 0:
        o = np.asarray(obs["observation"]).astype(np.float32, copy=True)
        if o.shape[-1] >= 9:
            o[..., 6:9] = o[..., 6:9] + rng.normal(0, noise, size=3).astype(np.float32)
        obs["observation"] = o
    return obs


def _eval_policy_under_noise(model, env, noise: float, episodes: int) -> dict:
    rng = np.random.default_rng(42)
    successes = 0
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=20_000 + ep)
        ret = 0.0
        succ = False
        for _ in range(80):
            obs_perturbed = _perturb_obs(obs, noise, rng)
            a, _ = model.predict(obs_perturbed, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ret += float(r)
            if info.get("is_success"):
                succ = True
            if term or trunc:
                break
        successes += int(succ)
        returns.append(ret)
    return {"success_rate": successes / episodes, "mean_return": float(np.mean(returns))}


def main(out_csv: str, out_fig: str, noise_levels: list[float], episodes: int, root: str):
    log = logging.getLogger("h3_robust")
    import panda_gym  # noqa: F401
    from sb3_contrib import TQC
    from src.inject.camera import CameraParams
    from src.inject.degraded_obs import make_panda_env_for_arm
    from scripts.train_h3 import _load_predictor

    cam = CameraParams(width=84, height=84)
    predictor = _load_predictor("outputs/checkpoints/panda_heatmap_head.joblib")

    rows = []
    for arm in ["A", "B", "C", "D"]:
        for seed_dir in sorted(Path(root, arm).glob("seed*")):
            seed = int(seed_dir.name.replace("seed", ""))
            ckpt = seed_dir / "model.zip"
            if not ckpt.exists():
                log.warning("no model at %s; skipping", ckpt)
                continue
            env = make_panda_env_for_arm(arm, "PandaPush-v3",
                                         predictor=predictor if arm == "D" else None,
                                         camera=cam)
            model = TQC.load(ckpt, env=env, device="cuda")
            for noise in noise_levels:
                m = _eval_policy_under_noise(model, env, noise, episodes)
                rows.append({"arm": arm, "seed": seed, "noise": noise, **m})
                log.info("arm=%s seed=%d noise=%.3f success=%.3f",
                         arm, seed, noise, m["success_rate"])
            env.close()

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", out_csv)

    # Figure: success vs noise per arm (mean across seeds).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=140)
    arms = ["A", "B", "C", "D"]
    palette = {"A": "#4a72c9", "B": "#888888", "C": "#c94a72", "D": "#4ac972"}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        by_noise: dict[float, list[float]] = {}
        for r in arm_rows:
            by_noise.setdefault(r["noise"], []).append(r["success_rate"])
        if not by_noise:
            continue
        xs = sorted(by_noise.keys())
        means = [float(np.mean(by_noise[x])) for x in xs]
        stds = [float(np.std(by_noise[x])) for x in xs]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2,
                    color=palette[arm], label=arm)
    ax.set_xlabel("Noise σ added to achieved_goal at test time (m)")
    ax.set_ylabel("Eval success rate")
    ax.set_title("H3 — robustness to perception noise across arms")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    log.info("wrote %s", out_fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default="outputs/h3_saved/robustness.csv")
    ap.add_argument("--out-fig", default="outputs/figures/h3_robustness.png")
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--root", default="outputs/h3_saved")
    args = ap.parse_args()
    main(args.out_csv, args.out_fig, args.noise, args.episodes, args.root)
