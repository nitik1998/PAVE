"""Render N frames of PandaPush-v3 with the oracle affordance overlay.

Writes outputs/figures/oracle_overlay_{i}.png and a single oracle_overlay.mp4.
GPU not used.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.inject.camera import CameraParams
from src.inject.wrapper import AffordanceWrapper
from src.utils.viz import overlay_multi_heatmap, save_image, save_video


def main(env_id: str, frames: int, out_dir: str, render_size: int):
    import panda_gym  # noqa: F401  registers envs

    log = logging.getLogger("oracle_demo")
    env = gym.make(env_id, render_mode="rgb_array")
    cam = CameraParams(width=render_size, height=render_size)
    wrapped = AffordanceWrapper(env, source="oracle", camera=cam, downsample=render_size, include_rgb=True)

    obs, _ = wrapped.reset(seed=0)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    composites: list[np.ndarray] = []
    for i in range(frames):
        rgb = obs["rgb"]
        heat = obs["affordance"]
        comp = overlay_multi_heatmap(rgb, heat, alpha=0.55)
        composites.append(comp)
        save_image(comp, out_dir_p / f"oracle_overlay_{i:02d}.png")
        a = wrapped.action_space.sample()
        obs, _, term, trunc, _ = wrapped.step(a)
        if term or trunc:
            obs, _ = wrapped.reset()
    wrapped.close()

    save_video(composites, out_dir_p / "oracle_overlay.mp4", fps=4)
    log.info("wrote %d overlay frames + mp4 to %s", frames, out_dir_p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="PandaPush-v3")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--render-size", type=int, default=480)
    args = ap.parse_args()
    main(args.env_id, args.frames, args.out_dir, args.render_size)
