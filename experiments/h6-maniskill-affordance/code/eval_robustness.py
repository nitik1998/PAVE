"""Evaluate pretrained ManiSkill3 PPO checkpoints under test-time observation
perturbation. Sweeps Gaussian noise σ on the OBJECT POSE slice of the state
observation across multiple noise levels and records success rate.

The intent is to find the regime where state-conditioned perception fails —
that's the regime where affordance injection should be tested as a remedy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

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


# ManiSkill3 PickCube state layout (42 dims), per the task source:
#   [0:7]   robot qpos (7)
#   [7:14]  robot qvel (7)
#   [14:17] tcp pos
#   [17:20] tcp velocity
#   [20:23] cube pos        ← perturb this
#   [23:27] cube quaternion
#   [27:30] tcp-to-cube vector
#   [30:33] target/goal pos
#   [33:42] additional task-specific state
# We pertub the slice [20:23] for cube_pos noise.
DEFAULT_PERTURB_SLICES = {
    "PickCube-v1": (29, 32),     # verified: state[29:32] == cube actor pose.p
    "PushCube-v1": (29, 32),
    "StackCube-v1": (29, 32),
}

# All slices that carry cube-position information (full perturbation needs all).
ALL_CUBE_SLICES = {
    "PickCube-v1": [(29, 32), (36, 39)],   # absolute cube xyz, cube-tcp relative
}


def perturb_obs_all_cube(obs: torch.Tensor, slices, sigma: float, tcp_slice=(19, 22)) -> torch.Tensor:
    if sigma <= 0:
        return obs
    out = obs.clone()
    # Apply consistent noise to absolute cube_pos, then RECOMPUTE the relative slice.
    a, b = slices[0]
    noise = sigma * torch.randn_like(out[..., a:b])
    out[..., a:b] = out[..., a:b] + noise
    # If a relative slice exists, perturb it consistently with the absolute noise.
    if len(slices) > 1:
        c, d = slices[1]
        # Relative is cube - tcp, so noise on cube propagates to relative.
        out[..., c:d] = out[..., c:d] + noise
    return out


def perturb_obs(obs: torch.Tensor, slc: tuple[int, int], sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return obs
    out = obs.clone()
    a, b = slc
    out[..., a:b] = out[..., a:b] + sigma * torch.randn_like(out[..., a:b])
    return out


def main(args):
    log = logging.getLogger("robust_eval")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    device = torch.device("cuda")
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim = sd["actor_mean.0.weight"].shape[1]
    action_dim = sd["actor_mean.6.weight"].shape[0]

    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state",
                   control_mode=args.control_mode, sim_backend="gpu",
                   reward_mode="dense")

    agent = OfficialPPOAgent(obs_dim, action_dim).to(device)
    agent.load_state_dict(sd)
    agent.eval()

    if args.full_cube:
        slices = ALL_CUBE_SLICES.get(args.env_id, [(29, 32)])
        log.info("env=%s ALL cube slices=%s (full perturbation)", args.env_id, slices)
    else:
        slc = DEFAULT_PERTURB_SLICES.get(args.env_id, (20, 23))
        slices = None
        log.info("env=%s perturb_slice=%s n_envs=%d steps=%d", args.env_id, slc, args.num_envs, args.steps)

    rows = []
    for sigma in args.noise:
        torch.manual_seed(args.seed)
        obs, _ = env.reset(seed=args.seed)
        if args.full_cube:
            obs = perturb_obs_all_cube(obs, slices, sigma)
        else:
            obs = perturb_obs(obs, slc, sigma)
        running_ret = torch.zeros(args.num_envs, device=device)
        running_max_succ = torch.zeros(args.num_envs, device=device)
        eps_returns, eps_succs = [], []
        for _ in range(args.steps):
            with torch.no_grad():
                mean = agent.actor_mean(obs)
            action = mean.clamp(-1.0, 1.0)
            obs, rew, term, trunc, info = env.step(action)
            if args.full_cube:
                obs = perturb_obs_all_cube(obs, slices, sigma)
            else:
                obs = perturb_obs(obs, slc, sigma)
            running_ret += rew.flatten()
            if "success" in info:
                running_max_succ = torch.maximum(running_max_succ, info["success"].float().flatten())
            done = (term | trunc).flatten().bool()
            if done.any():
                for i in done.nonzero(as_tuple=False).flatten().tolist():
                    eps_returns.append(float(running_ret[i].item()))
                    eps_succs.append(float(running_max_succ[i].item()))
                running_ret[done] = 0.0
                running_max_succ[done] = 0.0
        mean_succ = sum(eps_succs)/max(len(eps_succs),1)
        mean_ret = sum(eps_returns)/max(len(eps_returns),1)
        log.info("sigma=%.3f n_eps=%d mean_succ=%.3f mean_ret=%.2f", sigma, len(eps_succs), mean_succ, mean_ret)
        rows.append({"env_id": args.env_id, "sigma": sigma,
                     "n_episodes": len(eps_succs),
                     "mean_success": mean_succ, "mean_return": mean_ret})

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
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--full-cube", action="store_true", help="Perturb both [29:32] and [36:39]")
    ap.add_argument("--out", default="experiments/h6-maniskill-affordance/results/pickcube_robustness.csv")
    args = ap.parse_args()
    main(args)
