"""Roll out the pretrained ManiSkill3 PickCube PPO and record (rgb, state, action) tuples.

Used by H8 (action-prediction proxy): each encoder will be tested on how well its
features predict the policy's chosen action, given only RGB.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class OfficialPPOAgent(nn.Module):
    """Mirrors ManiSkill3 PPO baseline architecture."""
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("collect")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    device = torch.device("cuda")
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim = sd["actor_mean.0.weight"].shape[1]
    action_dim = sd["actor_mean.6.weight"].shape[0]
    log.info("ckpt obs_dim=%d action_dim=%d", obs_dim, action_dim)

    # We need BOTH the state (to feed the policy) AND the rgb (for our predictor).
    # `obs_mode='rgbd'` returns a dict with sensor_data + agent state slices, but
    # it doesn't give the same state vector the policy was trained on. The
    # cleanest hack: roll out two parallel envs, one in state mode (drives the
    # policy), one in rgb mode (renders the same scene from the same seed).
    state_env = gym.make(args.env_id, num_envs=1, obs_mode="state",
                         control_mode=args.control_mode, sim_backend="gpu",
                         reward_mode="dense")
    rgb_env = gym.make(args.env_id, num_envs=1, obs_mode="rgb",
                       control_mode=args.control_mode, sim_backend="gpu",
                       reward_mode="dense",
                       sensor_configs=dict(width=args.image_size, height=args.image_size))

    agent = OfficialPPOAgent(obs_dim, action_dim).to(device)
    agent.load_state_dict(sd)
    agent.eval()

    rgbs, actions, cube_poses = [], [], []
    successes = 0
    for ep in range(args.n_episodes):
        seed = ep + args.seed_base
        s_obs, _ = state_env.reset(seed=seed)
        r_obs, _ = rgb_env.reset(seed=seed)
        # Sync rgb_env to state_env (paranoia — same seed gives same scene if
        # both envs share the underlying sim semantics, but force-sync anyway).
        try:
            rgb_env.unwrapped.set_state(state_env.unwrapped.get_state())
        except Exception as e:
            log.warning("state sync failed (%s) — falling back to seed-only", e)

        for step in range(args.steps_per_episode):
            with torch.no_grad():
                action = agent.actor_mean(s_obs)
            action = action.clamp(-1.0, 1.0)
            # Save rgb + cube_pos at the current state.
            sd_dict = r_obs.get("sensor_data") if isinstance(r_obs, dict) else None
            if sd_dict is not None:
                cam = list(sd_dict.values())[0]
                rgb = cam["rgb"][0].cpu().numpy() if hasattr(cam["rgb"], "cpu") else np.array(cam["rgb"][0])
                rgbs.append(rgb)
                actions.append(action[0].cpu().numpy())
                try:
                    cube_actor = (state_env.unwrapped.scene.actors.get("cube")
                                  or state_env.unwrapped.scene.actors.get("object")
                                  or list(state_env.unwrapped.scene.actors.values())[0])
                    cube_pos = cube_actor.pose.p[0].cpu().numpy()
                except Exception:
                    cube_pos = np.zeros(3, dtype=np.float32)
                cube_poses.append(cube_pos)

            # Step both envs with the same action.
            s_obs, _, term, trunc, info = state_env.step(action)
            r_obs, _, _, _, _ = rgb_env.step(action)
            try:
                rgb_env.unwrapped.set_state(state_env.unwrapped.get_state())
            except Exception:
                pass
            if "success" in info:
                successes += int(info["success"][0].item() if hasattr(info["success"], "item") else bool(info["success"]))
            if term.any() or trunc.any():
                break
        if (ep + 1) % 10 == 0:
            log.info("[%d/%d] collected %d frames, %d successes",
                     ep + 1, args.n_episodes, len(rgbs), successes)

    state_env.close()
    rgb_env.close()

    rgbs = np.stack(rgbs)             # (N, H, W, 3) uint8
    actions = np.stack(actions)       # (N, action_dim)
    cube_poses = np.stack(cube_poses) # (N, 3)
    log.info("final: rgbs=%s actions=%s cubes=%s",
             rgbs.shape, actions.shape, cube_poses.shape)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "pickcube_action_data.npz",
             rgbs=rgbs.astype(np.uint8),
             actions=actions.astype(np.float32),
             cube_poses=cube_poses.astype(np.float32))
    log.info("wrote %s", out / "pickcube_action_data.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / ".maniskill/demos/PickCube-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt"))
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--steps-per-episode", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out", default="experiments/h8-action-proxy/data")
    args = ap.parse_args()
    main(args)
