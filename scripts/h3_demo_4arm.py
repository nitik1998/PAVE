"""4-arm side-by-side rollout video for the H3 ablation.

Loads trained models from outputs/h3/{A,B,C,D}/seed0/model.zip and rolls
each one out for one episode (deterministic). Composites all 4 into a 2×2
grid with arm labels and success indicators. Intended for the talk hero.

Output: outputs/figures/h3_arms_4panel.mp4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np


def _label_panel(rgb: np.ndarray, arm: str, label: str, step: int, success: bool) -> np.ndarray:
    import cv2

    img = rgb.copy()
    h, w = img.shape[:2]
    # bottom title bar
    bar = np.full((48, w, 3), 16, dtype=np.uint8)
    text = f"{arm}: {label}"
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    img = np.concatenate([img, bar], axis=0)
    # success indicator
    if success:
        cv2.rectangle(img, (8, 8), (28, 28), (40, 200, 40), -1)
        cv2.putText(img, "OK", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main(out: str, max_steps: int, render_size: int):
    import panda_gym  # noqa: F401
    from sb3_contrib import TQC
    from src.inject.camera import CameraParams, render_camera
    from src.inject.degraded_obs import make_panda_env_for_arm
    from src.utils.viz import save_video
    from scripts.train_h3 import _load_predictor

    log = logging.getLogger("h3_demo")
    cam = CameraParams(width=render_size, height=render_size)
    arm_labels = {
        "A": "Full state",
        "B": "Degraded state",
        "C": "B + ORACLE affordance",
        "D": "B + PREDICTED affordance",
    }
    predictor = _load_predictor("outputs/checkpoints/panda_heatmap_head.joblib")

    envs = {}
    models = {}
    for arm in ["A", "B", "C", "D"]:
        env = make_panda_env_for_arm(arm, "PandaPush-v3",
                                     predictor=predictor if arm == "D" else None,
                                     camera=cam)
        envs[arm] = env
        ckpt = Path(f"outputs/h3/{arm}/seed0/model.zip")
        if not ckpt.exists():
            log.warning("missing %s; using untrained model", ckpt)
            models[arm] = TQC("MultiInputPolicy", env, verbose=0, device="cuda")
        else:
            models[arm] = TQC.load(ckpt, env=env, device="cuda")

    seed = 0
    frames: list[np.ndarray] = []
    for arm in ["A", "B", "C", "D"]:
        envs[arm].reset(seed=seed)

    succ = {a: False for a in envs}
    obs = {a: envs[a].reset(seed=seed)[0] for a in envs}
    for t in range(max_steps):
        rgb = render_camera(None, cam)
        panels = []
        for arm in ["A", "B", "C", "D"]:
            panels.append(_label_panel(rgb, arm, arm_labels[arm], t, succ[arm]))
        # 2×2 grid
        h, w = panels[0].shape[:2]
        pad_h = np.full((4, w, 3), 32, dtype=np.uint8)
        pad_v = np.full((h * 2 + 4, 4, 3), 32, dtype=np.uint8)
        top = np.concatenate([panels[0], panels[1]], axis=1)
        bot = np.concatenate([panels[2], panels[3]], axis=1)
        grid = np.concatenate([top, pad_h.repeat(2, axis=0)[:4], bot], axis=0)
        frames.append(grid)
        # Step each env
        for arm in ["A", "B", "C", "D"]:
            if succ[arm]:
                continue
            a, _ = models[arm].predict(obs[arm], deterministic=True)
            obs[arm], r, term, trunc, info = envs[arm].step(a)
            if info.get("is_success"):
                succ[arm] = True
        if all(succ.values()):
            break
    for env in envs.values():
        env.close()

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, out_p, fps=15)
    log.info("wrote %s with %d frames; final success=%s", out_p, len(frames), succ)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/h3_arms_4panel.mp4")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--render-size", type=int, default=320)
    args = ap.parse_args()
    main(args.out, args.max_steps, args.render_size)
