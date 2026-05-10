"""Oracle affordance heatmaps for Panda-Gym tasks.

Each task exposes a small set of named regions (cube surface, target zone,
gripper, etc.). For each region we project its world-frame center to a pixel
and render an isotropic Gaussian. The result is a (C, H, W) float array in
[0, 1] suitable for `AffordanceWrapper`.

Channel order is taxonomy-aligned where possible, otherwise we use a per-task
mapping documented in `task_channel_layout()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.inject.camera import CameraParams, view_proj_matrices, world_to_pixel

log = logging.getLogger(__name__)


@dataclass
class HeatmapSpec:
    name: str          # human-readable label, e.g. "push-target"
    sigma_world: float  # in metres, will be projected to pixels via depth


def task_channel_layout(env_id: str) -> list[HeatmapSpec]:
    if env_id.startswith("PandaPush"):
        return [
            HeatmapSpec("object", 0.03),
            HeatmapSpec("goal", 0.04),
        ]
    if env_id.startswith("PandaPickAndPlace") or env_id.startswith("PandaSlide"):
        return [
            HeatmapSpec("object", 0.03),
            HeatmapSpec("goal", 0.04),
        ]
    if env_id.startswith("PandaReach"):
        return [HeatmapSpec("goal", 0.04)]
    return [HeatmapSpec("object", 0.03), HeatmapSpec("goal", 0.04)]


def _query_region_centers(env, env_id: str) -> dict[str, np.ndarray]:
    """Pull world-frame XYZ centers for each region we care about.

    panda-gym exposes the underlying PyBullet sim at `env.unwrapped.sim` and
    the goal at `env.unwrapped.task.goal` (numpy array) or via
    `env.unwrapped.task.get_goal()`.
    """
    out: dict[str, np.ndarray] = {}
    task = env.unwrapped.task
    sim = env.unwrapped.sim
    body_names = []
    for attr in ("object", "_object", "main_object"):
        if hasattr(task, attr):
            body_names.append(getattr(task, attr))
    try:
        out["object"] = np.asarray(sim.get_base_position("object"))
    except Exception:
        try:
            out["object"] = np.asarray(task.get_obs()["achieved_goal"])
        except Exception as e:
            log.warning("Could not query object pose for %s: %s", env_id, e)
    try:
        out["goal"] = np.asarray(task.get_goal())
    except Exception:
        try:
            out["goal"] = np.asarray(task.goal)
        except Exception as e:
            log.warning("Could not query goal pose for %s: %s", env_id, e)
    return out


def _gaussian_at(width: int, height: int, u: float, v: float, sigma_px: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    sigma_px = float(max(sigma_px, 1.0))
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    return np.exp(-d2 / (2.0 * sigma_px ** 2)).astype(np.float32)


def render_oracle(env, cam: CameraParams) -> np.ndarray:
    """Returns (C, H, W) float32. C = len(task_channel_layout(env_id))."""
    env_id = env.unwrapped.spec.id if env.unwrapped.spec is not None else "Panda-v3"
    layout = task_channel_layout(env_id)
    centers = _query_region_centers(env, env_id)
    view, proj = view_proj_matrices(cam)
    chans = np.zeros((len(layout), cam.height, cam.width), dtype=np.float32)
    for i, spec in enumerate(layout):
        if spec.name not in centers:
            continue
        xyz = centers[spec.name]
        proj_pt = world_to_pixel(xyz, view, proj, cam)
        if proj_pt is None:
            continue
        u, v, depth = proj_pt
        # Convert sigma from metres to pixels using the image-plane focal length.
        # f_pix = (height/2) / tan(fov/2)
        f_pix = (cam.height * 0.5) / np.tan(np.deg2rad(cam.fov) * 0.5)
        sigma_px = max(2.0, f_pix * spec.sigma_world / max(depth, 0.05))
        chans[i] = _gaussian_at(cam.width, cam.height, u, v, sigma_px)
    return chans
