"""Record N successful PandaPush episodes back-to-back into a single MP4.

Each episode is separated by a 5-frame fade-in title card showing the
episode index. Heatmap side panel is the oracle 2-channel.

Useful for the talk's headline video — shorter than the full sweep but more
visually compelling than a single 9-frame win.
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


def _composite(rgb: np.ndarray, heat: np.ndarray, episode: int, step: int, success: bool) -> np.ndarray:
    import cv2

    overlay = overlay_multi_heatmap(rgb, heat, alpha=0.55)
    h, w = rgb.shape[:2]
    pad = np.zeros((h, 16, 3), dtype=np.uint8)
    panel = np.concatenate([rgb, pad, overlay], axis=1)
    label = f"ep {episode}  step {step}" + ("  SUCCESS" if success else "")
    cv2.putText(panel, label, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _title_card(width: int, height: int, episode: int, hold: int = 5) -> list[np.ndarray]:
    import cv2

    frame = np.full((height, 2 * width + 16, 3), 16, dtype=np.uint8)
    text = f"Episode {episode}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.putText(frame, text, ((frame.shape[1] - tw) // 2, (height + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2, cv2.LINE_AA)
    return [frame] * hold


def main(out: str, episodes: int, max_steps: int, render_size: int):
    import panda_gym  # noqa: F401
    from huggingface_hub import hf_hub_download
    from huggingface_sb3 import load_from_hub
    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    log = logging.getLogger("multi_demo")
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
            rgb = render_env.unwrapped.render()
            if rgb.shape[0] != render_size or rgb.shape[1] != render_size:
                import cv2
                rgb = cv2.resize(rgb, (render_size, render_size), interpolation=cv2.INTER_AREA)
            heat = render_oracle(render_env, cam)
            ep_frames.append(_composite(rgb, heat, successes + 1, t, succ))
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
                    rgb_end = render_env.unwrapped.render()
                    if rgb_end.shape[0] != render_size:
                        import cv2
                        rgb_end = cv2.resize(rgb_end, (render_size, render_size))
                    heat_end = render_oracle(render_env, cam)
                    for _ in range(8):
                        ep_frames.append(_composite(rgb_end, heat_end, successes + 1, t + 1, True))
                break
        seed += 1
        if succ:
            all_frames += _title_card(render_size, render_size, successes + 1, hold=10)
            all_frames += ep_frames
            successes += 1
            log.info("captured episode %d (seed=%d, steps=%d)", successes, seed - 1, len(ep_frames))
        else:
            log.info("seed=%d failed; skipping", seed - 1)

    render_env.close()
    vec.close()

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_video(all_frames, out_p, fps=15)
    log.info("wrote %s (%d frames, %d successful episodes)", out_p, len(all_frames), successes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/push_demo_multi.mp4")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--render-size", type=int, default=320)
    args = ap.parse_args()
    main(args.out, args.episodes, args.max_steps, args.render_size)
