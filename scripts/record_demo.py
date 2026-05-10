"""Headline demo: pretrained TQC plays PandaPush-v3, side panel shows the
oracle affordance heatmap. Runs episodes until a successful one is captured
(up to --max-episodes), then writes that one to MP4.

We use VecNormalize for the policy obs (the way the checkpoint was trained)
and a parallel single env for rendering + oracle heatmap extraction. Actions
are computed against the normalized vec env and applied identically to the
render env so the two trajectories stay in lockstep.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.inject.camera import CameraParams
from src.inject.oracle_panda import render_oracle
from src.utils.viz import overlay_multi_heatmap, save_video


HF_REPO = "enaitzb/TQC-PandaPush-v3"
HF_FILE = "TQC_no_load_env-PandaPush-v3.zip"
VN_FILE = "vec_normalize.pkl"


def _composite(rgb: np.ndarray, heat: np.ndarray, success: bool, step: int) -> np.ndarray:
    import cv2

    overlay = overlay_multi_heatmap(rgb, heat, alpha=0.55)
    h, w = rgb.shape[:2]
    pad = np.zeros((h, 16, 3), dtype=np.uint8)
    panel = np.concatenate([rgb, pad, overlay], axis=1)
    label = f"step {step}" + ("  SUCCESS" if success else "")
    cv2.putText(
        panel,
        label,
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _seed_render_env_to_vec(render_env, vec_env):
    """Best-effort match the render env's state to the inner vec env state."""
    inner = vec_env.envs[0].unwrapped
    rsim = render_env.unwrapped.sim
    # Copy object and goal poses from vec inner env to render env.
    for body in ("object", "target"):
        try:
            pos = inner.sim.get_base_position(body)
            rot = inner.sim.get_base_rotation(body)
            rsim.set_base_pose(body, pos, rot)
        except Exception:
            pass


def main(out: str, max_episodes: int, max_steps: int, render_size: int):
    import panda_gym  # noqa: F401
    from huggingface_hub import hf_hub_download
    from huggingface_sb3 import load_from_hub
    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    log = logging.getLogger("record_demo")
    log.info("Downloading %s ...", HF_REPO)
    ckpt = load_from_hub(HF_REPO, HF_FILE)
    vn_path = hf_hub_download(HF_REPO, VN_FILE)

    cam = CameraParams(width=render_size, height=render_size)
    render_env = gym.make("PandaPush-v3", render_mode="rgb_array")
    vec = DummyVecEnv([lambda: gym.make("PandaPush-v3")])
    vec = VecNormalize.load(vn_path, vec)
    vec.training = False
    vec.norm_reward = False

    model = TQC.load(
        ckpt,
        env=vec,
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
    )

    success_frames: list[np.ndarray] | None = None
    for ep in range(max_episodes):
        obs = vec.reset()
        render_env.reset(seed=ep)
        _seed_render_env_to_vec(render_env, vec)
        frames: list[np.ndarray] = []
        succ = False
        for t in range(max_steps):
            rgb = render_env.unwrapped.render()
            if rgb.shape[0] != cam.height or rgb.shape[1] != cam.width:
                import cv2

                rgb = cv2.resize(rgb, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
            heat = render_oracle(render_env, cam)
            frames.append(_composite(rgb, heat, succ, t))
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = vec.step(a)
            try:
                render_env.step(a[0])
            except Exception:
                pass
            if info[0].get("is_success"):
                succ = True
            if done[0] or succ:
                # Capture one extra frame at the success state.
                if succ:
                    rgb_end = render_env.unwrapped.render()
                    if rgb_end.shape[0] != cam.height or rgb_end.shape[1] != cam.width:
                        import cv2
                        rgb_end = cv2.resize(rgb_end, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
                    heat_end = render_oracle(render_env, cam)
                    frames.append(_composite(rgb_end, heat_end, True, t + 1))
                break
        log.info("episode=%d frames=%d success=%s", ep, len(frames), succ)
        if succ and success_frames is None:
            success_frames = frames
            break
    render_env.close()
    vec.close()

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_video(success_frames or frames, out_p, fps=15)
    log.info("wrote %s", out_p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/push_demo.mp4")
    ap.add_argument("--max-episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--render-size", type=int, default=320)
    args = ap.parse_args()
    main(args.out, args.max_episodes, args.max_steps, args.render_size)
