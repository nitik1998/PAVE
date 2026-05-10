"""Cross-domain qualitative test: fit DINOv2 probe on UMD, apply to Panda-Gym renders.

This is the experiment that motivates Stage 3 (predicted-instead-of-oracle
heatmaps in sim). UMD is photo-realistic 3D objects on a tan tabletop;
Panda-Gym renders are flat-shaded synthetic cubes on a navy table. We do NOT
expect transfer — but visualizing the failure mode is the point.

Outputs:
  outputs/figures/cross_domain_grid.png     5 frames x {RGB, dinov2 probe overlay}
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.eval.dataset_umd import UMDSubset
from src.inject.camera import CameraParams
from src.methods.dinov2_probe import build as build_dinov2
from src.utils.viz import grid_figure, overlay_multi_heatmap


def main(out_path: str, n_frames: int, image_size: int, n_train: int):
    log = logging.getLogger("cross_domain")
    import gymnasium as gym
    import panda_gym  # noqa: F401

    # --- 1. Fit the probe on UMD ---
    log.info("Loading UMD train split ...")
    train = UMDSubset.from_split_file("data/umd/splits/train.json",
                                      "configs/affordance_taxonomy.yaml",
                                      image_size=image_size)
    foreground = [n for n in train.class_names if n != "background"]
    probe = build_dinov2(num_classes=len(train.class_names),
                         foreground_names=foreground, device="cpu")
    probe.cfg.image_size = image_size
    probe.cfg.patch_size = 14
    probe.warmup()

    pairs = []
    for s, rgb, lbl in train:
        pairs.append((rgb, lbl))
        if len(pairs) >= n_train:
            break
    log.info("Fitting linear probe on %d UMD images ...", len(pairs))
    probe.fit(pairs)

    # --- 2. Roll out random actions in PandaPush, predict on each frame ---
    log.info("Generating Panda-Gym frames ...")
    env = gym.make("PandaPush-v3", render_mode="rgb_array")
    cam = CameraParams(width=480, height=480)
    env.reset(seed=0)
    rgbs: list[np.ndarray] = []
    for t in range(n_frames):
        from src.inject.camera import render_camera

        rgb = render_camera(None, cam)
        rgbs.append(rgb)
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
    env.close()

    # --- 3. Apply probe to each rendered frame ---
    log.info("Applying probe to %d Panda frames ...", len(rgbs))
    rows: list[list[np.ndarray]] = []
    for rgb in rgbs:
        # Resize to probe's image_size so feature grid lines up.
        from PIL import Image
        rgb_for_probe = np.asarray(Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR))
        pred = probe.predict_map(rgb_for_probe)
        # Resize prediction back to original RGB shape.
        import cv2
        h, w = rgb.shape[:2]
        resized = np.zeros((pred.shape[0], h, w), dtype=np.float32)
        for ci in range(pred.shape[0]):
            resized[ci] = cv2.resize(pred[ci], (w, h), interpolation=cv2.INTER_LINEAR)
        rows.append([rgb, overlay_multi_heatmap(rgb, resized, alpha=0.55)])

    grid_figure(
        rows,
        col_labels=["Panda RGB", "UMD-trained DINOv2 probe overlay"],
        row_labels=[f"frame {i}" for i in range(len(rows))],
        out_path=out_path,
    )
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/cross_domain_grid.png")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--n-train", type=int, default=130)
    args = ap.parse_args()
    main(args.out, args.frames, args.image_size, args.n_train)
