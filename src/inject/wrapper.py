"""AffordanceWrapper — adds an `affordance` key to every observation.

Compatible with goal-conditioned panda-gym envs (Dict obs space) and with
plain Box-obs envs. Source of the heatmap is configurable:

  source = "oracle"     →  reads sim ground-truth via src.inject.oracle_panda
  source = "predictor"  →  applies a `AffordancePredictor` to the rendered RGB
"""

from __future__ import annotations

import logging
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.inject.camera import CameraParams, render_camera
from src.inject.oracle_panda import render_oracle, task_channel_layout
from src.methods.base import AffordancePredictor

log = logging.getLogger(__name__)


class AffordanceWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        source: str = "oracle",
        predictor: AffordancePredictor | None = None,
        camera: CameraParams | None = None,
        downsample: int = 84,
        include_rgb: bool = False,
    ):
        super().__init__(env)
        self.source = source
        self.predictor = predictor
        self.camera = camera or CameraParams(width=480, height=480)
        self.downsample = downsample
        self.include_rgb = include_rgb
        if source == "predictor" and predictor is None:
            raise ValueError("source='predictor' requires a predictor instance")
        if source not in ("oracle", "predictor"):
            raise ValueError(f"unknown source {source!r}")

        env_id = env.unwrapped.spec.id if env.unwrapped.spec is not None else ""
        if source == "oracle":
            self._n_channels = len(task_channel_layout(env_id))
        else:
            self._n_channels = len(predictor.foreground_class_names)

        new_space = spaces.Dict(
            {
                **(env.observation_space.spaces if isinstance(env.observation_space, spaces.Dict)
                   else {"obs": env.observation_space}),
                "affordance": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._n_channels, downsample, downsample),
                    dtype=np.float32,
                ),
                **(
                    {"rgb": spaces.Box(low=0, high=255, shape=(downsample, downsample, 3), dtype=np.uint8)}
                    if include_rgb else {}
                ),
            }
        )
        self.observation_space = new_space

    # --- core hook ---
    def observation(self, obs):
        if not isinstance(obs, dict):
            obs = {"obs": obs}
        rgb_full = render_camera(None, self.camera)
        if self.source == "oracle":
            heat_full = render_oracle(self.env, self.camera)
        else:
            heat_full = self.predictor.predict_map(rgb_full)
        heat = _resize_chw(heat_full, self.downsample)
        obs = dict(obs)
        obs["affordance"] = heat.astype(np.float32)
        if self.include_rgb:
            obs["rgb"] = _resize_hwc(rgb_full, self.downsample)
        return obs

    def render_full_rgb(self) -> np.ndarray:
        return render_camera(None, self.camera)


def _resize_chw(x: np.ndarray, size: int) -> np.ndarray:
    import cv2

    out = np.zeros((x.shape[0], size, size), dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = cv2.resize(x[i], (size, size), interpolation=cv2.INTER_AREA)
    return out


def _resize_hwc(x: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)


def make_oracle_wrapped(env_id: str = "PandaPush-v3", render_size: int = 480) -> AffordanceWrapper:
    """Convenience: returns an AffordanceWrapper with oracle source."""
    import gymnasium as gym
    import panda_gym  # noqa: F401  (registers envs)

    env = gym.make(env_id, render_mode="rgb_array")
    cam = CameraParams(width=render_size, height=render_size)
    return AffordanceWrapper(env, source="oracle", camera=cam, include_rgb=True)


PredictorFactory = Callable[[], AffordancePredictor]


def make_predictor_wrapped(
    env_id: str, factory: PredictorFactory, render_size: int = 480
) -> AffordanceWrapper:
    import gymnasium as gym
    import panda_gym  # noqa: F401

    env = gym.make(env_id, render_mode="rgb_array")
    cam = CameraParams(width=render_size, height=render_size)
    return AffordanceWrapper(env, source="predictor", predictor=factory(), camera=cam, include_rgb=True)
