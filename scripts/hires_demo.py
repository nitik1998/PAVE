"""V1 hi-res multi-panel demo MP4.

Records 3 successful PandaPush episodes from a (pretrained) policy and
composites:

  ┌──────────────┬──────────────┬──────────────┐
  │   RGB front  │   RGB top-45  │  Affordance  │
  │   (720×720)  │   (720×720)   │   overlay    │
  └──────────────┴──────────────┴──────────────┘
  step counter / success bar at bottom

The two RGB panels orbit slightly per frame for parallax. The affordance
overlay is the oracle 2-channel (object red, goal yellow).

Output: outputs/figures/hero_demo_4k.mp4 (~60 sec, 3 episodes).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np


HF_REPO = "enaitzb/TQC-PandaPush-v3"
HF_FILE = "TQC_no_load_env-PandaPush-v3.zip"
VN_FILE = "vec_normalize.pkl"


def _composite_3panel(rgb_front: np.ndarray, rgb_top: np.ndarray,
                      heat: np.ndarray, episode: int, step: int, success: bool) -> np.ndarray:
    import cv2

    from src.utils.viz import overlay_multi_heatmap

    overlay = overlay_multi_heatmap(rgb_front, heat, alpha=0.55)
    h, w = rgb_front.shape[:2]
    pad = np.full((h, 16, 3), 16, dtype=np.uint8)
    panel = np.concatenate([rgb_front, pad, rgb_top, pad, overlay], axis=1)
    H, W = panel.shape[:2]
    bar = np.full((48, W, 3), 16, dtype=np.uint8)
    label = f"Episode {episode}  Step {step:02d}" + ("  SUCCESS" if success else "")
    cv2.putText(bar, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(bar, "RGB front  |  RGB top-45  |  Oracle affordance", (W - 760, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    return np.concatenate([panel, bar], axis=0)


def main(out: str, episodes: int, max_steps: int, render_size: int):
    import panda_gym  # noqa: F401
    from huggingface_hub import hf_hub_download
    from huggingface_sb3 import load_from_hub
    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from src.inject.camera import CameraParams, render_camera
    from src.inject.oracle_panda import render_oracle
    from src.utils.viz import save_video

    log = logging.getLogger("hires_demo")
    log.info("Downloading pretrained policy %s", HF_REPO)
    ckpt = load_from_hub(HF_REPO, HF_FILE)
    vn_path = hf_hub_download(HF_REPO, VN_FILE)

    cam_front = CameraParams(width=render_size, height=render_size, yaw=45, pitch=-30, distance=1.4)
    cam_top = CameraParams(width=render_size, height=render_size, yaw=45, pitch=-55, distance=1.6)

    render_env = gym.make("PandaPush-v3", render_mode="rgb_array")
    vec = DummyVecEnv([lambda: gym.make("PandaPush-v3")])
    vec = VecNormalize.load(vn_path, vec)
    vec.training = False
    vec.norm_reward = False

    model = TQC.load(
        ckpt, env=vec,
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
    )

    all_frames: list[np.ndarray] = []
    successes = 0
    seed = 0
    while successes < episodes and seed < episodes * 4:
        obs = vec.reset()
        render_env.reset(seed=seed)
        try:
            inner = vec.envs[0].unwrapped
            for body in ("object", "target"):
                pos = inner.sim.get_base_position(body)
                rot = inner.sim.get_base_rotation(body)
                render_env.unwrapped.sim.set_base_pose(body, pos, rot)
        except Exception:
            pass

        ep_frames: list[np.ndarray] = []
        succ = False
        for t in range(max_steps):
            cam_front.yaw = 45 + 5 * np.sin(t * 0.15)
            cam_top.yaw = 45 - 5 * np.sin(t * 0.12)
            rgb_front = render_camera(None, cam_front)
            rgb_top = render_camera(None, cam_top)
            heat = render_oracle(render_env, cam_front)
            ep_frames.append(_composite_3panel(rgb_front, rgb_top, heat, successes + 1, t, succ))
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = vec.step(a)
            try:
                render_env.step(a[0])
            except Exception:
                pass
            if info[0].get("is_success"):
                succ = True
            if done[0] or succ:
                if succ:
                    rgb_f2 = render_camera(None, cam_front)
                    rgb_t2 = render_camera(None, cam_top)
                    heat2 = render_oracle(render_env, cam_front)
                    for k in range(15):
                        ep_frames.append(_composite_3panel(rgb_f2, rgb_t2, heat2, successes + 1, t + 1, True))
                break
        seed += 1
        if succ:
            all_frames += ep_frames
            successes += 1
            log.info("captured ep %d (seed=%d, %d frames)", successes, seed - 1, len(ep_frames))
    render_env.close()
    vec.close()

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_video(all_frames, out_p, fps=20)
    log.info("wrote %s (%d frames, %d episodes)", out_p, len(all_frames), successes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/hero_demo_4k.mp4")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--render-size", type=int, default=720)
    args = ap.parse_args()
    main(args.out, args.episodes, args.max_steps, args.render_size)
