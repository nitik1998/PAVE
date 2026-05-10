"""Train a small per-pixel regressor that predicts the 2-channel oracle
heatmap (object, goal) for ``PandaPush-v3`` renders, given DINOv2 patch
features. Used by H3 arm D — *predicted* affordance from vision.

Procedure:
  * Sample N random (object_pos, goal_pos) configurations.
  * Render each, generate the oracle heatmap.
  * Pool oracle heatmap to per-patch 2-vec (mean intensity inside each 14×14 patch).
  * Fit `sklearn.MultiOutputRegressor(Ridge)` on (DINOv2 patch features → 2-vec).
  * Save head as joblib that ``train_h3.py`` can load.

Output: outputs/checkpoints/panda_heatmap_head.joblib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np


def main(out_path: str, n_samples: int, image_size: int, render_size: int):
    log = logging.getLogger("train_panda_head")
    import gymnasium as gym
    import panda_gym  # noqa: F401

    from src.inject.camera import CameraParams, render_camera
    from src.inject.oracle_panda import render_oracle
    from src.methods.dinov2_probe import build as build_dinov2

    cam = CameraParams(width=render_size, height=render_size)
    env = gym.make("PandaPush-v3", render_mode="rgb_array")

    backbone = build_dinov2(num_classes=3, foreground_names=["object", "goal"], device="cuda")
    backbone.cfg.image_size = image_size
    backbone.cfg.patch_size = 14
    backbone.warmup()
    grid = image_size // 14

    Xs, Ys = [], []
    log.info("Generating %d Panda render samples ...", n_samples)
    for i in range(n_samples):
        env.reset(seed=i)
        rgb = render_camera(None, cam)
        # Resize rgb to image_size for DINOv2.
        from PIL import Image

        rgb_for_probe = np.asarray(Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR))
        feats = backbone._extract_patch_features(rgb_for_probe)         # (N_patches, D)
        oracle = render_oracle(env, cam)                                # (2, render_size, render_size)
        # Resize oracle to image_size.
        oracle_resized = np.zeros((oracle.shape[0], image_size, image_size), dtype=np.float32)
        for c in range(oracle.shape[0]):
            oracle_resized[c] = np.asarray(
                Image.fromarray((oracle[c] * 255).astype(np.uint8)).resize((image_size, image_size), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        # Patch-pool: mean per 14×14 tile, per channel.
        ph = oracle_resized.reshape(2, grid, 14, grid, 14).mean(axis=(2, 4))  # (2, grid, grid)
        Y = ph.transpose(1, 2, 0).reshape(-1, 2)                              # (N_patches, 2)
        Xs.append(feats)
        Ys.append(Y)
        if (i + 1) % 20 == 0:
            log.info("  sampled %d/%d", i + 1, n_samples)
    env.close()

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    log.info("Fitting Ridge regressor on X=%s Y=%s", X.shape, Y.shape)
    from sklearn.linear_model import Ridge

    head = Ridge(alpha=1.0)
    head.fit(X, Y)

    # Quick sanity: predict on training data and report R².
    pred = head.predict(X)
    ss_res = ((pred - Y) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-9)
    log.info("Training R² per channel: %s", r2.tolist())

    import joblib

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "kind": "panda_heatmap_head",
        "head": head,
        "image_size": image_size,
        "patch_size": 14,
        "n_channels": 2,
        "channel_names": ["object", "goal"],
        "n_samples": n_samples,
        "render_size": render_size,
        "r2": r2.tolist(),
    }, out_path)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/checkpoints/panda_heatmap_head.joblib")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--render-size", type=int, default=240)
    args = ap.parse_args()
    main(args.out, args.n_samples, args.image_size, args.render_size)
