"""Wrappers that selectively erase parts of the PandaPush-v3 observation.

The H3 experiment uses these to construct controlled "perception is noisy"
scenarios. Specifically:

  * ``DegradedStateWrapper`` zeros out the object xyz from the obs, so the
    policy can no longer read the cube position from state.
  * ``AffordanceCentroidWrapper`` adds 4 extra dims at the end of the
    observation: (u, v) projected pixel of the object, (u, v) of the goal,
    each normalised to [0, 1]. The pixels can come from oracle sim state
    or from a vision predictor.

These compose: typically you wrap with Degraded first, then add
AffordanceCentroid on top.
"""

from __future__ import annotations

import logging
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.inject.camera import CameraParams, render_camera, view_proj_matrices, world_to_pixel

log = logging.getLogger(__name__)


# Index of the object xyz inside the PandaPush observation vector. From
# panda-gym source: the obs is [ee_pos(3), ee_vel(3), object_pos(3),
# object_rot(3), object_velp(3), object_velr(3)] = 18 dims (or 25 dims
# with extras). The object_pos is at indices 6:9.
_OBJECT_POS_OBS_SLICE = slice(6, 9)


class DegradedStateWrapper(gym.ObservationWrapper):
    """Zero out the cube xyz from achieved_goal and from observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Observation space stays the same shape; values just get zeroed.
        self.observation_space = env.observation_space

    def observation(self, obs):
        # NOTE: we deliberately leave ``achieved_goal`` untouched so that
        # HER's relabeled-reward computation still uses the ground-truth
        # object xyz. We only erase the redundant copy that lives inside
        # ``observation`` (slice 6:9 in panda_gym v3.0.7). The policy
        # cannot read the object_pos slice but achieved_goal still
        # provides its goal-conditioned signal.
        obs = dict(obs) if isinstance(obs, dict) else {"observation": obs}
        if "observation" in obs:
            o = np.asarray(obs["observation"]).astype(np.float32, copy=True)
            if o.shape[-1] >= 9:
                o[..., _OBJECT_POS_OBS_SLICE] = 0.0
            obs["observation"] = o
        return obs


HeatmapSource = Callable[[gym.Env, np.ndarray, CameraParams], np.ndarray]
"""Signature: (env, rgb_image_optional, camera) -> (C, H, W) heatmap in [0, 1].
RGB may be None for sources that ignore pixels (oracle)."""


class AffordanceCentroidWrapper(gym.ObservationWrapper):
    """Append (u, v) centroid + peak intensity for each affordance channel."""

    def __init__(
        self,
        env: gym.Env,
        heatmap_source: HeatmapSource,
        camera: CameraParams | None = None,
        n_channels: int = 2,
        include_peak: bool = True,
        skip_rgb: bool = True,
    ):
        super().__init__(env)
        self.heatmap_source = heatmap_source
        self.camera = camera or CameraParams(width=84, height=84)
        self.n_channels = n_channels
        self.include_peak = include_peak
        # ``skip_rgb`` lets oracle source bypass the expensive PyBullet RGB
        # render — oracle works from sim state, not pixels. Saves ~30 ms/step.
        self.skip_rgb = skip_rgb
        extra_dims = n_channels * (3 if include_peak else 2)

        if isinstance(env.observation_space, spaces.Dict):
            inner = env.observation_space["observation"]
            new_inner = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(inner.shape[-1] + extra_dims,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Dict({
                **env.observation_space.spaces,
                "observation": new_inner,
            })
        else:
            inner = env.observation_space
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(inner.shape[-1] + extra_dims,),
                dtype=np.float32,
            )

    def _extract_centroid(self, heat: np.ndarray) -> np.ndarray:
        """heat: (C, H, W) → (C * features,) where features = (u, v) or (u, v, peak)."""
        c, h, w = heat.shape
        out: list[float] = []
        for i in range(c):
            m = heat[i]
            total = float(m.sum())
            if total <= 1e-6:
                u, v, peak = 0.5, 0.5, 0.0
            else:
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                u = float((xx * m).sum() / total) / max(1, w - 1)
                v = float((yy * m).sum() / total) / max(1, h - 1)
                peak = float(m.max())
            out.extend([u, v])
            if self.include_peak:
                out.append(peak)
        return np.asarray(out, dtype=np.float32)

    def observation(self, obs):
        rgb = None if self.skip_rgb else render_camera(None, self.camera)
        heat = self.heatmap_source(self.env, rgb, self.camera)
        # Pad/truncate to expected n_channels.
        if heat.shape[0] != self.n_channels:
            if heat.shape[0] > self.n_channels:
                heat = heat[:self.n_channels]
            else:
                pad = np.zeros((self.n_channels - heat.shape[0], *heat.shape[1:]), dtype=heat.dtype)
                heat = np.concatenate([heat, pad], axis=0)
        feats = self._extract_centroid(heat)
        if isinstance(obs, dict):
            obs = dict(obs)
            o = np.asarray(obs["observation"]).astype(np.float32)
            obs["observation"] = np.concatenate([o, feats]).astype(np.float32)
            return obs
        return np.concatenate([np.asarray(obs).astype(np.float32), feats]).astype(np.float32)


# --- heatmap sources ---


def oracle_source(env: gym.Env, rgb: np.ndarray, camera: CameraParams) -> np.ndarray:
    from src.inject.oracle_panda import render_oracle

    return render_oracle(env, camera)


def make_predicted_source(predictor):
    """Predicted source: feed rgb to a frozen predictor (DINOv2 + linear head)."""
    def _src(env: gym.Env, rgb: np.ndarray, camera: CameraParams) -> np.ndarray:
        if rgb is None:
            from src.inject.camera import render_camera as _r

            rgb = _r(None, camera)
        return predictor.predict_map(rgb)
    return _src


def make_panda_env_for_arm(
    arm: str,
    env_id: str = "PandaPush-v3",
    predictor=None,
    camera: CameraParams | None = None,
) -> gym.Env:
    """Build the env for an H3 ablation arm.

    Arms:
      A: full state — no degradation, no affordance
      B: degraded state — zero object xyz
      C: degraded + oracle affordance centroid
      D: degraded + predicted affordance centroid (requires predictor)
    """
    import panda_gym  # noqa: F401

    env = gym.make(env_id)
    if arm == "A":
        return env
    if arm == "B":
        return DegradedStateWrapper(env)
    if arm == "C":
        env = DegradedStateWrapper(env)
        # Oracle source doesn't need rgb → skip the expensive PyBullet render.
        return AffordanceCentroidWrapper(env, oracle_source, camera=camera, skip_rgb=True)
    if arm == "D":
        if predictor is None:
            raise ValueError("arm D requires a predictor")
        env = DegradedStateWrapper(env)
        # Predicted source DOES need rgb (DINOv2 input). Cannot skip.
        return AffordanceCentroidWrapper(env, make_predicted_source(predictor), camera=camera, skip_rgb=False)
    raise ValueError(f"unknown arm {arm!r}")
