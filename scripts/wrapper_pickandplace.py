"""Sanity check that AffordanceWrapper + oracle heatmap also works on
`PandaPickAndPlace-v3` (secondary task from the proposal).

We do NOT have a pretrained policy for this task in the same form; this is
purely a wrapper sanity check + visual proof that the oracle generalizes
to a different task with different region semantics.
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


def main(out_dir: str, frames: int, render_size: int):
    import panda_gym  # noqa: F401

    log = logging.getLogger("pickandplace_wrapper")
    env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
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
        save_image(comp, out_dir_p / f"pickandplace_overlay_{i:02d}.png")
        a = wrapped.action_space.sample()
        obs, _, term, trunc, _ = wrapped.step(a)
        if term or trunc:
            obs, _ = wrapped.reset()
    wrapped.close()
    save_video(composites, out_dir_p / "pickandplace_overlay.mp4", fps=4)
    log.info("wrote %d frames + mp4 (PandaPickAndPlace-v3)", frames)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--render-size", type=int, default=480)
    args = ap.parse_args()
    main(args.out_dir, args.frames, args.render_size)
