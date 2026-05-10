"""Train a vision-based cube_pos predictor for ManiSkill3 PickCube.

Pipeline: render RGB → DINOv2 (or π0 SigLIP) patch features → Ridge regression
→ 3-dim cube xyz prediction.

This is the keystone for the H6 affordance-injection experiment:

  Vanilla pretrained PPO    +  noisy cube_pos in obs   →  policy fails (per
  experiments/h6-maniskill-affordance/results/pickcube_robustness.csv).

  Same policy + REPLACE noisy slice with VISION-PREDICTED cube_pos →
  policy should recover. The recovery margin is the H6 evidence.

  Substituting π0 SigLIP for DINOv2 in the predictor → recovery should
  degrade (because π0's vision tower lost cut/support perception). That's
  the H2-predicts-H3 link.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch


def collect_state_image_pairs(env_id: str, n: int, image_size: int, control_mode: str):
    """Collect (rgb, cube_pos) pairs from DIVERSE timesteps, not just resets.

    For each of n//5 episodes, take 5 random-action steps and record a sample
    at each. This gives the predictor a wide distribution of cube positions
    (during interaction, not just initial drops).
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    log = logging.getLogger("collect")
    env = gym.make(env_id, num_envs=1, obs_mode="rgb",
                   control_mode=control_mode, sim_backend="gpu",
                   reward_mode="dense", sensor_configs=dict(width=image_size, height=image_size))
    rgbs = []
    cube_poses = []
    n_episodes = max(1, n // 5)
    samples_per_ep = max(1, n // n_episodes)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for s in range(samples_per_ep):
            # Skip first sample at random to bias toward mid-interaction.
            if s > 0:
                action = env.action_space.sample()
                obs, _, term, trunc, _ = env.step(action)
                if term.any() or trunc.any():
                    break
            sd = obs.get("sensor_data") if isinstance(obs, dict) else None
            if sd is None:
                continue
            cam = list(sd.values())[0]
            rgb = cam["rgb"][0].cpu().numpy() if hasattr(cam["rgb"], "cpu") else np.array(cam["rgb"][0])
            try:
                cube_actor = env.unwrapped.scene.actors.get("cube") or env.unwrapped.scene.actors.get("object") or list(env.unwrapped.scene.actors.values())[0]
                cube_pos = cube_actor.pose.p[0].cpu().numpy()
            except Exception as e:
                log.warning("could not extract cube pose: %s", e)
                continue
            rgbs.append(rgb)
            cube_poses.append(cube_pos)
        if (ep + 1) % 20 == 0:
            log.info("collected %d/%d episodes (%d samples)", ep + 1, n_episodes, len(rgbs))
    env.close()
    return np.stack(rgbs), np.stack(cube_poses)


def extract_features(rgbs: np.ndarray, backbone: str, image_size: int):
    log = logging.getLogger("features")
    import torch
    from PIL import Image

    device = torch.device("cuda")

    if backbone == "dinov2":
        from src.methods.dinov2_probe import build as build_probe
        backbone_obj = build_probe(num_classes=3, foreground_names=["object","goal","gripper"], device="cuda")
        backbone_obj.cfg.image_size = image_size
        backbone_obj.cfg.patch_size = 14
    elif backbone == "pi0":
        from src.methods.pi0_siglip_probe import build as build_probe
        backbone_obj = build_probe(num_classes=3, foreground_names=["object","goal","gripper"], device="cuda")
        backbone_obj.cfg.image_size = 224
    else:
        raise ValueError(f"unknown backbone {backbone}")
    backbone_obj.warmup()

    feats = []
    for i, rgb in enumerate(rgbs):
        # Resize for the backbone.
        target_size = backbone_obj.cfg.image_size
        if rgb.shape[0] != target_size:
            pil = Image.fromarray(rgb).resize((target_size, target_size), Image.BILINEAR)
            rgb = np.asarray(pil)
        f = backbone_obj._extract_patch_features(rgb)
        # Mean-pool patch features → single vector per image.
        feats.append(f.mean(axis=0))
        if (i + 1) % 20 == 0:
            log.info("features %d/%d", i + 1, len(rgbs))
    return np.stack(feats)


def main(args):
    log = logging.getLogger("cubepos")
    rgbs, cubes = collect_state_image_pairs(args.env_id, args.n_samples, args.image_size, args.control_mode)
    log.info("collected RGB %s, cube_pos %s", rgbs.shape, cubes.shape)

    feats = extract_features(rgbs, args.backbone, args.image_size)
    log.info("features %s", feats.shape)

    # Train Ridge regression: features → cube_pos
    from sklearn.linear_model import Ridge

    n_train = int(0.8 * len(feats))
    X_train, X_val = feats[:n_train], feats[n_train:]
    y_train, y_val = cubes[:n_train], cubes[n_train:]
    head = Ridge(alpha=1.0)
    head.fit(X_train, y_train)
    pred_val = head.predict(X_val)
    err = np.linalg.norm(pred_val - y_val, axis=1)
    log.info("val mean L2 error: %.4f m, median: %.4f m, std: %.4f", err.mean(), np.median(err), err.std())

    import joblib
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "kind": "cubepos_predictor",
        "head": head,
        "backbone": args.backbone,
        "env_id": args.env_id,
        "image_size": args.image_size,
        "n_train": n_train,
        "n_val": len(feats) - n_train,
        "val_l2_mean": float(err.mean()),
        "val_l2_median": float(np.median(err)),
        "val_l2_std": float(err.std()),
    }, out)
    log.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "pi0"])
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default="experiments/h6-maniskill-affordance/results/cubepos_dinov2.joblib")
    args = ap.parse_args()
    main(args)
