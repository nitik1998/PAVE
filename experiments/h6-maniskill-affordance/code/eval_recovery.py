"""Robustness recovery eval: noisy state + vision-predicted cube_pos override.

For each noise σ on the cube_pos slice, compare three policies:
  - baseline:      noisy obs, no override.
  - oracle:        noisy obs but cube_pos slice replaced with sim ground-truth.
  - predicted:     noisy obs but cube_pos slice replaced with predictor(rgb).

The "predicted" curve sandwiched between baseline (lowest at high σ) and
oracle (full recovery) quantifies how much vision-based affordance recovers
from perception noise. Predictor backbones: DINOv2 vs π0 SigLIP.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class OfficialPPOAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))


CUBE_POS_SLICE = (29, 32)  # absolute cube xyz
CUBE_REL_SLICE = (36, 39)  # cube xyz - tcp xyz (relative)
TCP_SLICE = (19, 22)       # tcp xyz


def load_predictor(path: str):
    import joblib
    blob = joblib.load(path)
    return blob


def get_features(rgb_batch: torch.Tensor, backbone_obj) -> np.ndarray:
    """Extract DINOv2/SigLIP mean-pooled features from a batch of RGBs.

    rgb_batch: (B, H, W, 3) uint8 numpy or torch. Returns (B, D) numpy.
    """
    feats = []
    for i in range(rgb_batch.shape[0]):
        rgb = rgb_batch[i]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        f = backbone_obj._extract_patch_features(rgb)
        feats.append(f.mean(axis=0))
    return np.stack(feats)


def main(args):
    log = logging.getLogger("recovery")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    device = torch.device("cuda")

    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim = sd["actor_mean.0.weight"].shape[1]
    action_dim = sd["actor_mean.6.weight"].shape[0]

    # Two envs kept in sync via set_state: state env (for observations to policy) + rgb env (for vision predictor).
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state",
                   control_mode=args.control_mode, sim_backend="gpu",
                   reward_mode="dense")
    rgb_env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb",
                       control_mode=args.control_mode, sim_backend="gpu",
                       reward_mode="dense", sensor_configs=dict(width=args.image_size, height=args.image_size))

    agent = OfficialPPOAgent(obs_dim, action_dim).to(device)
    agent.load_state_dict(sd)
    agent.eval()

    # Load predictors.
    predictors = {}
    backbones = {}
    if args.dinov2_pred:
        predictors["dinov2"] = load_predictor(args.dinov2_pred)
        from src.methods.dinov2_probe import build as build_probe
        backbones["dinov2"] = build_probe(num_classes=3, foreground_names=["a","b","c"], device="cuda")
        backbones["dinov2"].cfg.image_size = predictors["dinov2"]["image_size"]
        backbones["dinov2"].cfg.patch_size = 14
        backbones["dinov2"].warmup()
    if args.pi0_pred:
        predictors["pi0"] = load_predictor(args.pi0_pred)
        from src.methods.pi0_siglip_probe import build as build_probe
        backbones["pi0"] = build_probe(num_classes=3, foreground_names=["a","b","c"], device="cuda")
        backbones["pi0"].cfg.image_size = predictors["pi0"]["image_size"]
        backbones["pi0"].warmup()
    log.info("loaded %d predictors: %s", len(predictors), list(predictors.keys()))

    rows = []
    for sigma in args.noise:
        for variant in ["baseline", "oracle"] + list(predictors.keys()):
            torch.manual_seed(args.seed)
            obs_state, _ = env.reset(seed=args.seed)
            obs_rgb, _ = rgb_env.reset(seed=args.seed)
            # Sync rgb_env to state_env's scene state so renders match exactly.
            sync_state = env.unwrapped.get_state()
            rgb_env.unwrapped.set_state(sync_state)
            running_max_succ = torch.zeros(args.num_envs, device=device)
            eps_succs = []

            for step in range(args.steps):
                noisy_obs = obs_state.clone()
                a, b = CUBE_POS_SLICE
                c, d = CUBE_REL_SLICE
                tx, ty = TCP_SLICE
                if sigma > 0:
                    # Apply consistent noise to both absolute and relative cube slices.
                    noise = sigma * torch.randn_like(noisy_obs[..., a:b])
                    noisy_obs[..., a:b] = noisy_obs[..., a:b] + noise
                    noisy_obs[..., c:d] = noisy_obs[..., c:d] + noise

                if variant == "baseline":
                    eff_obs = noisy_obs
                elif variant == "oracle":
                    eff_obs = noisy_obs.clone()
                    eff_obs[..., a:b] = obs_state[..., a:b]
                    eff_obs[..., c:d] = obs_state[..., c:d]
                else:
                    rgb_env.unwrapped.set_state(env.unwrapped.get_state())
                    obs_rgb_now = rgb_env.unwrapped.get_obs()
                    cam = list(obs_rgb_now["sensor_data"].values())[0]
                    rgb = cam["rgb"]
                    feats = get_features(rgb, backbones[variant])
                    pred = predictors[variant]["head"].predict(feats)
                    pred_t = torch.from_numpy(pred).to(device).float()
                    eff_obs = noisy_obs.clone()
                    eff_obs[..., a:b] = pred_t
                    # Recompute relative slice from predicted cube_pos and clean tcp.
                    tcp = obs_state[..., tx:ty]
                    eff_obs[..., c:d] = pred_t - tcp

                with torch.no_grad():
                    mean = agent.actor_mean(eff_obs)
                action = mean.clamp(-1.0, 1.0)
                obs_state, _, term, trunc, info = env.step(action)
                if "success" in info:
                    running_max_succ = torch.maximum(running_max_succ, info["success"].float().flatten())
                done = (term | trunc).flatten().bool()
                if done.any():
                    for i in done.nonzero(as_tuple=False).flatten().tolist():
                        eps_succs.append(float(running_max_succ[i].item()))
                    running_max_succ[done] = 0.0
            mean_succ = sum(eps_succs)/max(len(eps_succs),1)
            log.info("variant=%s sigma=%.3f n_eps=%d mean_succ=%.3f", variant, sigma, len(eps_succs), mean_succ)
            rows.append({"variant": variant, "sigma": sigma,
                         "n_episodes": len(eps_succs), "mean_success": mean_succ})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--ckpt", default="/home/njalgo/.maniskill/demos/PickCube-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt")
    ap.add_argument("--dinov2-pred", default="experiments/h6-maniskill-affordance/results/cubepos_dinov2.joblib")
    ap.add_argument("--pi0-pred", default="experiments/h6-maniskill-affordance/results/cubepos_pi0.joblib")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--out", default="experiments/h6-maniskill-affordance/results/recovery_pickcube.csv")
    args = ap.parse_args()
    main(args)
