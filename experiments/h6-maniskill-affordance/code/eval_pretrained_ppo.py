"""Evaluate the ManiSkill3 official pretrained PPO checkpoint.

Provides our arm A baseline without needing to train PPO from scratch — the
checkpoint at ~/.maniskill/demos/<env>/rl/ppo_*_ckpt.pt was trained on the
official ManiSkill3 PPO pipeline and is known to converge.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn


class OfficialPPOAgent(nn.Module):
    """Mirrors the architecture of ManiSkill3's official PPO baseline.

    Layout deduced from `~/.maniskill/demos/PickCube-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt`:
      actor_mean: 4-layer MLP (256 hidden, indices 0/2/4/6, Tanh activations)
      critic: same shape, scalar output
      actor_logstd: per-action-dim scalar param
    """
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


def main(args):
    log = logging.getLogger("eval_ppo")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    device = torch.device("cuda")

    ckpt_path = args.ckpt
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    log.info("Loaded ckpt %s", ckpt_path)

    obs_dim = sd["actor_mean.0.weight"].shape[1]
    action_dim = sd["actor_mean.6.weight"].shape[0]

    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state",
                   control_mode=args.control_mode, sim_backend="gpu",
                   reward_mode="dense")
    log.info("env=%s control=%s obs_dim=%d action_dim=%d", args.env_id, args.control_mode, obs_dim, action_dim)

    agent = OfficialPPOAgent(obs_dim, action_dim).to(device)
    agent.load_state_dict(sd)
    agent.eval()

    obs, _ = env.reset(seed=args.seed)
    n = args.num_envs
    sucs = torch.zeros(n, device=device)
    rets = torch.zeros(n, device=device)
    eval_completed_returns = []
    eval_completed_successes = []
    running_ret = torch.zeros(n, device=device)
    running_max_succ = torch.zeros(n, device=device)
    for _ in range(args.steps):
        with torch.no_grad():
            mean = agent.actor_mean(obs)
        action = mean if args.deterministic else mean + torch.exp(agent.actor_logstd) * torch.randn_like(mean)
        action = action.clamp(-1.0, 1.0)
        obs, rew, term, trunc, info = env.step(action)
        running_ret += rew.flatten()
        if "success" in info:
            running_max_succ = torch.maximum(running_max_succ, info["success"].float().flatten())
        done = (term | trunc).flatten().bool()
        if done.any():
            for i in done.nonzero(as_tuple=False).flatten().tolist():
                eval_completed_returns.append(float(running_ret[i].item()))
                eval_completed_successes.append(float(running_max_succ[i].item()))
            running_ret[done] = 0.0
            running_max_succ[done] = 0.0
    mean_ret = sum(eval_completed_returns)/max(len(eval_completed_returns),1)
    mean_succ = sum(eval_completed_successes)/max(len(eval_completed_successes),1)
    log.info("RESULT env=%s det=%s n_episodes=%d mean_return=%.2f mean_success=%.3f",
             args.env_id, args.deterministic, len(eval_completed_returns), mean_ret, mean_succ)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "env_id": args.env_id, "control_mode": args.control_mode,
            "ckpt": str(ckpt_path),
            "n_eval_envs": n, "n_eval_steps": args.steps,
            "n_episodes": len(eval_completed_returns),
            "mean_return": mean_ret, "mean_success": mean_succ,
            "deterministic": args.deterministic,
        }, f, indent=2)
    log.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--ckpt", default="/home/njalgo/.maniskill/demos/PickCube-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--out", default="experiments/h6-maniskill-affordance/results/pretrained_ppo_eval.json")
    args = ap.parse_args()
    main(args)
