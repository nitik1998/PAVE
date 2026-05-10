"""Record a PickCube demo MP4 with the vision predictor overlaid.

Per frame:
  - render RGB at high resolution
  - run DINOv2 (or π0 SigLIP) cube_pos predictor
  - draw a marker at the predicted cube_pos (projected to pixels)
  - draw a marker at the true cube_pos
  - composite frames into MP4

This is the talk's headline visual: pretrained PPO solves PickCube while a
frozen vision predictor tracks the cube within ~1.5 cm in real time.
"""

from __future__ import annotations

import argparse
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


def project_world_to_pixel(world_xyz, cam_extrinsic, cam_intrinsic, image_size):
    """Project world coords (N, 3) to pixel (u, v). cam_extrinsic = world→camera 4x4."""
    pts = np.concatenate([world_xyz, np.ones((world_xyz.shape[0], 1))], axis=1)  # (N, 4)
    cam = pts @ cam_extrinsic.T  # (N, 4)
    cam = cam[:, :3] / np.maximum(np.abs(cam[:, 3:4]), 1e-9)  # but cam_extrinsic is 4x4 so [:, 3] is 1
    proj = cam_intrinsic @ cam.T  # (3, N)
    uv = proj[:2] / np.maximum(proj[2:3], 1e-9)
    return uv.T  # (N, 2)


def main(args):
    log = logging.getLogger("record_demo")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import joblib
    import cv2

    device = torch.device("cuda")
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim = sd["actor_mean.0.weight"].shape[1]
    action_dim = sd["actor_mean.6.weight"].shape[0]

    state_env = gym.make(args.env_id, num_envs=1, obs_mode="state",
                         control_mode=args.control_mode, sim_backend="gpu",
                         reward_mode="dense")
    rgb_env = gym.make(args.env_id, num_envs=1, obs_mode="rgb",
                       control_mode=args.control_mode, sim_backend="gpu",
                       reward_mode="dense", sensor_configs=dict(width=args.image_size, height=args.image_size))

    agent = OfficialPPOAgent(obs_dim, action_dim).to(device)
    agent.load_state_dict(sd)
    agent.eval()

    pred_blob = joblib.load(args.predictor)
    head = pred_blob["head"]
    backbone_name = pred_blob["backbone"]
    if backbone_name == "dinov2":
        from src.methods.dinov2_probe import build as build_probe
        backbone = build_probe(num_classes=3, foreground_names=["a","b","c"], device="cuda")
        backbone.cfg.image_size = pred_blob["image_size"]; backbone.cfg.patch_size = 14
    elif backbone_name == "pi0":
        from src.methods.pi0_siglip_probe import build as build_probe
        backbone = build_probe(num_classes=3, foreground_names=["a","b","c"], device="cuda")
        backbone.cfg.image_size = pred_blob["image_size"]
    backbone.warmup()

    obs_state, _ = state_env.reset(seed=args.seed)
    obs_rgb, _ = rgb_env.reset(seed=args.seed)
    rgb_env.unwrapped.set_state(state_env.unwrapped.get_state())

    frames = []
    for step in range(args.steps):
        # Re-sync rgb env state.
        rgb_env.unwrapped.set_state(state_env.unwrapped.get_state())
        rgb_obs_now = rgb_env.unwrapped.get_obs()
        cam_data = list(rgb_obs_now["sensor_data"].values())[0]
        rgb = cam_data["rgb"][0].cpu().numpy().astype(np.uint8)

        # Predict cube_pos.
        feats = backbone._extract_patch_features(rgb)
        feat = feats.mean(axis=0, keepdims=True)
        pred_xyz = head.predict(feat)[0]
        true_xyz = state_env.unwrapped.scene.actors["cube"].pose.p[0].cpu().numpy()

        # Get camera matrices to project xyz→uv.
        cam_param = list(rgb_env.unwrapped._sensors.values())[0]
        try:
            extr = cam_param.get_extrinsic_matrix()[0].cpu().numpy()
            intr = cam_param.get_intrinsic_matrix()[0].cpu().numpy()
        except Exception:
            # Fallback: just label without overlay.
            frames.append(rgb)
            with torch.no_grad():
                a = agent.actor_mean(obs_state).clamp(-1, 1)
            obs_state, _, _, _, _ = state_env.step(a)
            continue

        pts_world = np.stack([true_xyz, pred_xyz])
        uv = project_world_to_pixel(pts_world, extr, intr, args.image_size)
        uv = uv.astype(int)

        canvas = rgb.copy()
        # True cube: green circle
        cv2.circle(canvas, tuple(uv[0]), 6, (0, 255, 0), 2)
        cv2.putText(canvas, "true", (uv[0][0] + 8, uv[0][1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        # Predicted cube: red circle
        cv2.circle(canvas, tuple(uv[1]), 6, (255, 50, 50), 2)
        cv2.putText(canvas, f"pred ({backbone_name})", (uv[1][0] + 8, uv[1][1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 50, 50), 1, cv2.LINE_AA)
        # L2 error label.
        l2 = np.linalg.norm(true_xyz - pred_xyz) * 100  # m → cm
        cv2.putText(canvas, f"step {step}  L2 {l2:.1f}cm", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(canvas)

        with torch.no_grad():
            a = agent.actor_mean(obs_state).clamp(-1, 1)
        obs_state, _, term, trunc, info = state_env.step(a)
        if "success" in info and info["success"][0].item():
            cv2.putText(canvas, "SUCCESS", (args.image_size // 2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            frames.append(canvas)
            break

    state_env.close(); rgb_env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    with imageio.get_writer(str(out), fps=15, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
    log.info("wrote %s (%d frames)", out, len(frames))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--ckpt", default="/home/njalgo/.maniskill/demos/PickCube-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt")
    ap.add_argument("--predictor", default="experiments/h6-maniskill-affordance/results/cubepos_pi0_v2.joblib")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/figures/h6_pickcube_demo.mp4")
    args = ap.parse_args()
    main(args)
