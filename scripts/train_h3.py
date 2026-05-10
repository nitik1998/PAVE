"""H3 experiment: train SAC+HER on PandaPush-v3 across 4 observation arms.

  A: full state (baseline)
  B: degraded state (object xyz zeroed)
  C: degraded + ORACLE affordance centroid (4 dims appended: u_obj, v_obj, u_goal, v_goal)
     — peak intensity also added if --with-peak.
  D: degraded + PREDICTED affordance centroid (vision predictor)

Usage:
    python scripts/train_h3.py --arm A --seed 0 --steps 200000

Outputs (per run):
    outputs/h3/<arm>/seed<seed>/{model.zip, vec_normalize.pkl, train_log.csv,
                                  eval.json, learning_curve.png}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np


@dataclass
class TrainCfg:
    arm: str = "A"
    seed: int = 0
    steps: int = 200_000
    eval_episodes: int = 30
    learning_starts: int = 5_000
    buffer_size: int = 200_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    tau: float = 0.05
    gamma: float = 0.95
    n_envs: int = 1
    her_n_sampled_goal: int = 4
    out_dir: str = "outputs/h3"
    eval_every: int = 25_000
    log_every: int = 1_000
    predictor_ckpt: str | None = None  # arm D probe head joblib
    image_size: int = 448
    render_size: int = 240
    no_save_model: bool = False


def _make_env(arm: str, seed: int, predictor=None, image_size: int = 448, render_size: int = 240):
    import panda_gym  # noqa: F401

    from src.inject.camera import CameraParams
    from src.inject.degraded_obs import make_panda_env_for_arm

    cam = CameraParams(width=render_size, height=render_size)
    env = make_panda_env_for_arm(arm, "PandaPush-v3", predictor=predictor, camera=cam)
    env.reset(seed=seed)
    return env


def _load_predictor(ckpt: str | None):
    """For arm D: a frozen DINOv2 backbone + a SAVED LinearProbe head fitted on
    Panda renders → oracle heatmap (regression). The predict_map signature must
    return (C, H, W) in [0, 1]."""
    if ckpt is None:
        return None
    import joblib

    from src.methods.dinov2_probe import build as build_dinov2

    blob = joblib.load(ckpt)
    if blob.get("kind") != "panda_heatmap_head":
        raise ValueError(f"Unknown ckpt kind: {blob.get('kind')!r}")

    backbone = build_dinov2(num_classes=3, foreground_names=["object", "goal"], device="cuda")
    backbone.cfg.image_size = blob["image_size"]
    backbone.cfg.patch_size = 14
    backbone.warmup()

    head = blob["head"]                # (n_patches * C_out,) sklearn regressor
    n_channels = blob["n_channels"]
    image_size = blob["image_size"]
    patch = blob["patch_size"]
    grid = image_size // patch

    class _PandaHeatmapPredictor:
        foreground_class_names = ["object", "goal"]

        def predict_map(self, rgb: np.ndarray) -> np.ndarray:
            feats = backbone._extract_patch_features(rgb)        # (N_patches, D)
            preds = head.predict(feats)                           # (N_patches, C_out)
            preds = preds.reshape(grid, grid, n_channels).transpose(2, 0, 1)
            preds = np.clip(preds, 0, 1).astype(np.float32)
            # Bilinear up to image_size for shape match.
            import torch
            import torch.nn.functional as F

            t = torch.from_numpy(preds).unsqueeze(0).float()
            t = F.interpolate(t, size=image_size, mode="bilinear", align_corners=False)
            return t.squeeze(0).cpu().numpy()

    return _PandaHeatmapPredictor()


def _eval(model, env_factory, episodes: int) -> dict:
    successes = 0
    returns = []
    eval_env = env_factory()
    for ep in range(episodes):
        obs, _ = eval_env.reset(seed=10_000 + ep)
        ret = 0.0
        succ = False
        for _ in range(80):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(a)
            ret += float(r)
            if info.get("is_success"):
                succ = True
            if term or trunc:
                break
        returns.append(ret)
        successes += int(succ)
    eval_env.close()
    return {
        "success_rate": successes / max(1, episodes),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "episodes": episodes,
    }


def main(cfg: TrainCfg):
    log = logging.getLogger("train_h3")
    out_dir = Path(cfg.out_dir) / cfg.arm / f"seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = _load_predictor(cfg.predictor_ckpt)
    env_factory = lambda: _make_env(cfg.arm, cfg.seed, predictor=predictor,
                                    image_size=cfg.image_size, render_size=cfg.render_size)

    train_env = env_factory()
    log.info("arm=%s obs_space=%s action_space=%s",
             cfg.arm, train_env.observation_space, train_env.action_space)

    from sb3_contrib import TQC
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback

    class _Logger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows: list[dict] = []
            self.t0 = time.time()

        def _on_step(self) -> bool:
            if self.num_timesteps % cfg.log_every == 0 and self.num_timesteps > 0:
                infos = self.locals.get("infos", [])
                successes = [int(i.get("is_success", 0)) for i in infos]
                self.rows.append({
                    "step": self.num_timesteps,
                    "success_rate_window": float(np.mean(successes)) if successes else 0.0,
                    "wallclock_s": time.time() - self.t0,
                })
            return True

    model = TQC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_selection_strategy": "future",
        },
        verbose=0,
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        learning_starts=cfg.learning_starts,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        device="cuda",
    )

    cb = _Logger()
    log.info("Training %d steps ...", cfg.steps)
    model.learn(total_timesteps=cfg.steps, callback=cb, log_interval=50)

    if not cfg.no_save_model:
        model.save(out_dir / "model.zip")

    log_csv = out_dir / "train_log.csv"
    if cb.rows:
        with open(log_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cb.rows[0].keys()))
            w.writeheader()
            w.writerows(cb.rows)

    log.info("Evaluating ...")
    metrics = _eval(model, env_factory, cfg.eval_episodes)
    metrics["arm"] = cfg.arm
    metrics["seed"] = cfg.seed
    metrics["steps"] = cfg.steps
    with open(out_dir / "eval.json", "w") as f:
        json.dump({**asdict(cfg), **metrics}, f, indent=2)
    log.info("DONE arm=%s seed=%s success=%.3f return=%.2f",
             cfg.arm, cfg.seed, metrics["success_rate"], metrics["mean_return"])
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--out-dir", default="outputs/h3")
    ap.add_argument("--predictor-ckpt", default=None)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--render-size", type=int, default=240)
    ap.add_argument("--no-save-model", action="store_true")
    args = ap.parse_args()
    cfg = TrainCfg(
        arm=args.arm,
        seed=args.seed,
        steps=args.steps,
        eval_episodes=args.eval_episodes,
        out_dir=args.out_dir,
        predictor_ckpt=args.predictor_ckpt,
        image_size=args.image_size,
        render_size=args.render_size,
        no_save_model=args.no_save_model,
    )
    main(cfg)
