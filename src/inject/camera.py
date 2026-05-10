"""Camera math for projecting Panda-Gym world coordinates onto rendered pixels.

Panda-Gym defaults at v3.0.7 use::

    target_position=(0, 0, 0), distance=1.4, yaw=45, pitch=-30, roll=0,
    fov=45, near=0.1, far=100.

We replicate that below so the projection is consistent with whatever the
env returned in `render(mode="rgb_array")`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraParams:
    width: int = 480
    height: int = 480
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance: float = 1.4
    yaw: float = 45.0
    pitch: float = -30.0
    roll: float = 0.0
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0


def view_proj_matrices(cam: CameraParams) -> tuple[np.ndarray, np.ndarray]:
    import pybullet as p

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam.target,
        distance=cam.distance,
        yaw=cam.yaw,
        pitch=cam.pitch,
        roll=cam.roll,
        upAxisIndex=2,
    )
    aspect = cam.width / cam.height
    proj = p.computeProjectionMatrixFOV(cam.fov, aspect, cam.near, cam.far)
    return np.array(view, dtype=np.float64), np.array(proj, dtype=np.float64)


def world_to_pixel(xyz, view: np.ndarray, proj: np.ndarray, cam: CameraParams) -> tuple[float, float, float] | None:
    """Returns (u, v, depth_clip). Returns None if the point is behind the camera."""
    V = view.reshape(4, 4, order="F")
    P = proj.reshape(4, 4, order="F")
    p_h = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    cam_h = V @ p_h
    if cam_h[2] >= 0:           # behind camera in OpenGL convention
        return None
    clip = P @ cam_h
    if clip[3] == 0:
        return None
    ndc = clip[:3] / clip[3]
    u = (ndc[0] * 0.5 + 0.5) * cam.width
    v = (1.0 - (ndc[1] * 0.5 + 0.5)) * cam.height
    return float(u), float(v), float(-cam_h[2])


def render_camera(physics_client_id: int | None, cam: CameraParams) -> np.ndarray:
    """Render an RGB image with PyBullet's CPU renderer (RGBA → RGB)."""
    import pybullet as p

    view, proj = view_proj_matrices(cam)
    kwargs = dict(
        width=cam.width,
        height=cam.height,
        viewMatrix=view.tolist(),
        projectionMatrix=proj.tolist(),
        renderer=p.ER_TINY_RENDERER,
    )
    if physics_client_id is not None:
        kwargs["physicsClientId"] = physics_client_id
    _, _, rgba, _, _ = p.getCameraImage(**kwargs)
    arr = np.array(rgba, dtype=np.uint8).reshape(cam.height, cam.width, 4)
    return arr[..., :3]
