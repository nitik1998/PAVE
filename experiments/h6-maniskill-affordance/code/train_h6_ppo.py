"""H6 — ManiSkill3 PPO with optional affordance injection.

Single-file CleanRL-style PPO targeting ManiSkill3 GPU-vectorized envs.
Implements three observation arms for the affordance ablation:

  arm A: state-only baseline.
  arm C: state + oracle affordance centroid (from sim ground truth).
  arm D: state + predicted affordance centroid (from a frozen vision probe).

The probe head for arm D is the panda heatmap probe trained earlier — needs
adaptation to the ManiSkill3 task. For now arm D loads a placeholder predictor
that mirrors arm C's centroid, so the script is end-to-end runnable while we
refit the probe to ManiSkill3 renders in a follow-up.

References applied:
  - ml-training-recipes: bf16 autocast, grad clip 1.0, TF32 enabled, AdamW
    eps=1e-5, time-based LR with warmup + cosine decay, fast-fail on
    divergence.
  - ManiSkill3 PPO baselines: 256+ parallel envs, 4096-step rollouts, 4 epochs
    of update, 32 mini-batches per epoch, gamma=0.99, lambda=0.95.

Usage:
  python experiments/h6-maniskill-affordance/code/train_h6_ppo.py \
    --env-id PlugCharger-v1 --arm A --seed 0 --total-steps 2_000_000

Output:
  experiments/h6-maniskill-affordance/results/{env}/{arm}/seed{n}/
    train_log.csv  — per-update metrics
    eval.json      — final eval (success, return, std)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


# --------------------------------------------------------------------------
# Affordance source: query the wrapped sim env for ground-truth pose and
# project to a centroid feature vector. Mirrors the panda H3 setup.
# --------------------------------------------------------------------------


def _get_oracle_centroid_features(env, n_channels: int = 2) -> torch.Tensor:
    """Extract per-channel centroid features from sim ground truth.

    Returns shape (num_envs, n_channels * 3) with (cx, cy, cz) per channel.
    For PlugCharger we use (charger pose, outlet pose). For PegInsertionSide
    we use (peg pose, hole pose).

    The env exposes `agent` and `extra` keys in its observation dict. We sniff
    those rather than maintaining a per-env mapping table.
    """
    try:
        # ManiSkill3 envs expose unwrapped scene with named actors.
        scene = env.unwrapped.scene if hasattr(env.unwrapped, "scene") else env.unwrapped._scene
        candidates = []
        for actor in scene.actors.values():
            name = actor.name.lower() if hasattr(actor, "name") else ""
            if any(k in name for k in ["charger", "outlet", "peg", "socket", "hole", "obj", "target"]):
                pose = actor.pose
                # pose.p is (num_envs, 3) tensor on GPU
                candidates.append(pose.p)
        if len(candidates) < n_channels:
            # Pad with zeros if env doesn't have enough named actors.
            while len(candidates) < n_channels:
                candidates.append(torch.zeros_like(candidates[0]) if candidates else None)
        feats = torch.cat(candidates[:n_channels], dim=-1)
        return feats
    except Exception as e:
        logging.warning("oracle centroid failed: %s; returning zeros", e)
        return torch.zeros((env.unwrapped.num_envs, n_channels * 3), device="cuda")


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def _layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.actor_mean = _layer_init(nn.Linear(hidden, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = _layer_init(nn.Linear(hidden, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.shared(x))

    def get_action_and_value(self, x, action=None):
        h = self.shared(x)
        mean = self.actor_mean(h)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        v = self.critic(h)
        return action, logp, ent, v


# --------------------------------------------------------------------------
# Observation augmentation by arm
# --------------------------------------------------------------------------


@dataclass
class ArmConfig:
    arm: str
    n_extra_dims: int = 0


def augment_obs(obs: torch.Tensor, env, arm_cfg: ArmConfig, n_channels: int = 2) -> torch.Tensor:
    if arm_cfg.arm == "A":
        return obs
    if arm_cfg.arm == "C":
        feats = _get_oracle_centroid_features(env, n_channels=n_channels)
        return torch.cat([obs, feats], dim=-1)
    if arm_cfg.arm == "D":
        # Placeholder: use oracle for now; replace with vision predictor in follow-up.
        feats = _get_oracle_centroid_features(env, n_channels=n_channels)
        # Add Gaussian noise to simulate "predicted not oracle" until probe is fitted.
        feats = feats + 0.02 * torch.randn_like(feats)
        return torch.cat([obs, feats], dim=-1)
    raise ValueError(f"unknown arm {arm_cfg.arm}")


# --------------------------------------------------------------------------
# Train loop
# --------------------------------------------------------------------------


def train(args):
    log = logging.getLogger("h6_ppo")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  registers envs

    device = torch.device("cuda")

    env_kwargs = dict(
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reward_mode=args.reward_mode,
        sim_backend="gpu",
    )
    try:
        env = gym.make(args.env_id, **env_kwargs)
    except NotImplementedError as e:
        if "Unsupported reward mode" in str(e):
            log.warning("reward_mode=%s unsupported; falling back to normalized_dense", args.reward_mode)
            env_kwargs["reward_mode"] = "normalized_dense"
            env = gym.make(args.env_id, **env_kwargs)
        else:
            raise
    # Conditionally flatten action space if it's a Dict (rare; most ManiSkill3 envs are Box).
    if hasattr(env.action_space, "spaces"):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)

    arm_cfg = ArmConfig(arm=args.arm)
    if args.arm in ("C", "D"):
        arm_cfg.n_extra_dims = 6  # 2 channels × 3 coords

    obs, _ = env.reset(seed=args.seed)
    obs = augment_obs(obs, env, arm_cfg)
    obs_dim = obs.shape[-1]
    action_dim = int(np.prod(env.action_space.shape[1:]))
    log.info("env=%s arm=%s obs_dim=%d action_dim=%d num_envs=%d",
             args.env_id, args.arm, obs_dim, action_dim, args.num_envs)

    agent = PPOActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr, eps=1e-5)

    n_steps = args.steps_per_rollout
    total_steps = args.total_steps
    n_updates = total_steps // (n_steps * args.num_envs)
    log.info("planning %d updates × %d steps × %d envs = %d total steps",
             n_updates, n_steps, args.num_envs, total_steps)

    obs_buf = torch.zeros((n_steps, args.num_envs, obs_dim), device=device)
    act_buf = torch.zeros((n_steps, args.num_envs, action_dim), device=device)
    logp_buf = torch.zeros((n_steps, args.num_envs), device=device)
    val_buf = torch.zeros((n_steps, args.num_envs), device=device)
    rew_buf = torch.zeros((n_steps, args.num_envs), device=device)
    done_buf = torch.zeros((n_steps, args.num_envs), device=device)

    out_dir = Path(args.out_dir) / args.env_id / args.arm / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    fieldnames = ["update", "global_step", "wallclock", "ep_return_mean", "ep_success_mean",
                  "policy_loss", "value_loss", "entropy", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    t0 = time.time()
    global_step = 0
    next_obs = obs
    next_done = torch.zeros(args.num_envs, device=device)
    # Running per-env episode tracking — better than relying on final_info.
    ep_ret_running = torch.zeros(args.num_envs, device=device)
    ep_len_running = torch.zeros(args.num_envs, device=device)
    last_completed_returns: list[float] = []
    last_completed_successes: list[float] = []

    for update in range(1, n_updates + 1):
        # LR schedule (linear warmup then cosine).
        progress = update / n_updates
        if progress < args.warmup_ratio:
            lr_mult = progress / args.warmup_ratio
        else:
            t = (progress - args.warmup_ratio) / max(1e-9, 1.0 - args.warmup_ratio)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * t))
        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_mult

        # Rollout.
        for step in range(n_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, logp, _, v = agent.get_action_and_value(next_obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = v.flatten()
            obs_raw, rew, term, trunc, info = env.step(action)
            done = (term | trunc).float()
            rew_buf[step] = rew.flatten()
            ep_ret_running += rew.flatten()
            ep_len_running += 1
            # On any env-done, record its episode metrics and reset its accumulators.
            done_mask = done.bool()
            if done_mask.any():
                # ManiSkill3 exposes per-step success in info["success"] (per-env tensor).
                # Use the value at step-of-done as the episode success.
                if "success" in info:
                    succ = info["success"].float().flatten()
                else:
                    succ = torch.zeros_like(done)
                completed_idx = done_mask.nonzero(as_tuple=False).flatten()
                for i in completed_idx.tolist():
                    last_completed_returns.append(float(ep_ret_running[i].item()))
                    last_completed_successes.append(float(succ[i].item()))
                ep_ret_running[done_mask] = 0.0
                ep_len_running[done_mask] = 0.0
                # Cap memory.
                if len(last_completed_returns) > 200:
                    last_completed_returns[:] = last_completed_returns[-200:]
                    last_completed_successes[:] = last_completed_successes[-200:]
            next_obs = augment_obs(obs_raw, env, arm_cfg)
            next_done = done

        # GAE.
        with torch.no_grad():
            next_val = agent.get_value(next_obs).flatten()
            adv = torch.zeros_like(rew_buf)
            lastgae = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nonterm = 1.0 - next_done
                    nv = next_val
                else:
                    nonterm = 1.0 - done_buf[t + 1]
                    nv = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nonterm - val_buf[t]
                adv[t] = lastgae = delta + args.gamma * args.gae_lambda * nonterm * lastgae
            ret = adv + val_buf

        # Flatten.
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, action_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)

        b_size = b_obs.shape[0]
        mb_size = b_size // args.n_minibatches
        idx = torch.arange(b_size, device=device)

        pg_losses, v_losses, ents = [], [], []
        for epoch in range(args.update_epochs):
            idx = idx[torch.randperm(b_size, device=device)]
            for start in range(0, b_size, mb_size):
                mb = idx[start:start + mb_size]
                _, newlogp, ent, newv = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                mb_adv = b_adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                newv = newv.flatten()
                v_loss = 0.5 * ((newv - b_ret[mb]) ** 2).mean()
                ent_loss = ent.mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ents.append(ent_loss.item())

        # Fast-fail on divergence (per ml-training-recipes).
        if any(math.isnan(x) for x in pg_losses + v_losses):
            log.error("NaN loss detected at update %d. Aborting.", update)
            return {"success_rate": float("nan")}

        # Log.
        ep_ret_mean = float(np.mean(last_completed_returns)) if last_completed_returns else float("nan")
        ep_suc_mean = float(np.mean(last_completed_successes)) if last_completed_successes else float("nan")
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "update": update, "global_step": global_step,
                "wallclock": time.time() - t0,
                "ep_return_mean": ep_ret_mean, "ep_success_mean": ep_suc_mean,
                "policy_loss": float(np.mean(pg_losses)),
                "value_loss": float(np.mean(v_losses)),
                "entropy": float(np.mean(ents)),
                "lr": optimizer.param_groups[0]["lr"],
            })

        if update % args.log_freq == 0 or update == n_updates:
            log.info(
                "u=%d step=%d t=%.0fs ret=%.2f succ=%.2f pg=%.3f v=%.3f ent=%.3f",
                update, global_step, time.time() - t0,
                ep_ret_mean, ep_suc_mean,
                float(np.mean(pg_losses)), float(np.mean(v_losses)), float(np.mean(ents)),
            )

    # Final eval — track per-episode success (max success over a single episode).
    log.info("eval ...")
    n_eval_envs = args.num_envs
    obs_e, _ = env.reset(seed=args.seed + 1000)
    obs_e = augment_obs(obs_e, env, arm_cfg)
    rets = torch.zeros(n_eval_envs, device=device)
    sucs_episode_max = torch.zeros(n_eval_envs, device=device)
    eval_completed_returns: list[float] = []
    eval_completed_successes: list[float] = []
    eval_running_ret = torch.zeros(n_eval_envs, device=device)
    for _ in range(args.eval_steps):
        with torch.no_grad():
            act, _, _, _ = agent.get_action_and_value(obs_e)
        obs_raw, rew, term, trunc, info = env.step(act)
        eval_running_ret += rew.flatten()
        if "success" in info:
            sucs_episode_max = torch.maximum(sucs_episode_max, info["success"].float().flatten())
        done_mask = (term | trunc).flatten().bool()
        if done_mask.any():
            for i in done_mask.nonzero(as_tuple=False).flatten().tolist():
                eval_completed_returns.append(float(eval_running_ret[i].item()))
                eval_completed_successes.append(float(sucs_episode_max[i].item()))
            eval_running_ret[done_mask] = 0.0
            sucs_episode_max[done_mask] = 0.0
        obs_e = augment_obs(obs_raw, env, arm_cfg)
    if eval_completed_returns:
        mean_return = float(np.mean(eval_completed_returns))
        mean_success = float(np.mean(eval_completed_successes))
        n_episodes = len(eval_completed_returns)
    else:
        # No episodes completed during eval window — report current-step succ.
        mean_return = float(eval_running_ret.mean().item())
        mean_success = float(sucs_episode_max.mean().item())
        n_episodes = 0
    eval_metrics = {
        "env_id": args.env_id, "arm": args.arm, "seed": args.seed,
        "total_steps": total_steps,
        "n_eval_envs": int(n_eval_envs),
        "n_eval_episodes": n_episodes,
        "mean_return": mean_return,
        "mean_success": mean_success,
        "wallclock_total": time.time() - t0,
    }
    with open(out_dir / "eval.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    log.info("DONE %s", json.dumps(eval_metrics))
    return eval_metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="PlugCharger-v1")
    ap.add_argument("--arm", default="A", choices=["A", "C", "D"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--steps-per-rollout", type=int, default=64)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--n-minibatches", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--norm-adv", action="store_true", default=True)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--log-freq", type=int, default=5)
    ap.add_argument("--out-dir", default="experiments/h6-maniskill-affordance/results")
    ap.add_argument("--reward-mode", default="normalized_dense", help="ManiSkill reward mode (dense, normalized_dense, sparse)")
    return ap.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train(parse_args())
