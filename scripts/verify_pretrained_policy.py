"""Smoke test: download `enaitzb/TQC-PandaPush-v3` and run N episodes.

Loads the matching `vec_normalize.pkl` so the policy sees the obs distribution
it was trained on. Writes outputs/figures/push_pretrained.mp4 of the first
successful episode (or of episode 0 if none succeed).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.utils.viz import save_video


HF_REPO = "enaitzb/TQC-PandaPush-v3"
HF_FILE = "TQC_no_load_env-PandaPush-v3.zip"
VN_FILE = "vec_normalize.pkl"


def _make_envs():
    import panda_gym  # noqa: F401
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from huggingface_hub import hf_hub_download

    base = gym.make("PandaPush-v3", render_mode="rgb_array")
    vec = DummyVecEnv([lambda: gym.make("PandaPush-v3")])
    vn_path = hf_hub_download(HF_REPO, VN_FILE)
    vec = VecNormalize.load(vn_path, vec)
    vec.training = False
    vec.norm_reward = False
    return base, vec


def main(out_dir: str, episodes: int, max_steps: int):
    from huggingface_sb3 import load_from_hub
    from sb3_contrib import TQC

    log = logging.getLogger("verify_policy")
    log.info("Downloading %s ...", HF_REPO)
    ckpt = load_from_hub(HF_REPO, HF_FILE)

    render_env, vec = _make_envs()
    model = TQC.load(
        ckpt,
        env=vec,
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
    )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    saved_video = False
    successes = 0
    for ep in range(episodes):
        obs = vec.reset()
        render_env.reset(seed=ep)
        # Sync the render env to the same starting state by re-seeding both.
        # NOTE: reset(seed=ep) on render_env is not bit-identical to vec.reset();
        # the rendered video is for visual confirmation only.
        frames: list[np.ndarray] = []
        ret = 0.0
        succ = False
        for t in range(max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = vec.step(a)
            ret += float(r[0])
            if info[0].get("is_success"):
                succ = True
            try:
                a_render, _ = model.predict(obs, deterministic=True)
                _ = render_env.step(a_render)
            except Exception:
                pass
            frames.append(render_env.render())
            if done[0]:
                break
        successes += int(succ)
        log.info("episode=%d return=%.2f success=%s", ep, ret, succ)
        if not saved_video:
            save_video(frames, out_dir_p / "push_pretrained.mp4", fps=20)
            saved_video = True
    log.info("Total: %d/%d successes", successes, episodes)
    render_env.close()
    vec.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=50)
    args = ap.parse_args()
    main(args.out_dir, args.episodes, args.max_steps)
