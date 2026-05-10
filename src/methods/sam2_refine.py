"""SAM 2.1 wrapper for converting points or boxes into per-pixel masks.

Lazy-loaded. Falls back to a Gaussian splat around each point if SAM 2 weights
are missing — the eval pipeline keeps running, with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SAM2Config:
    weights_path: str = "outputs/checkpoints/sam2/sam2.1_hiera_base_plus.pt"
    config_name: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    device: str = "cpu"


class SAM2Refiner:
    def __init__(self, cfg: SAM2Config | None = None):
        self.cfg = cfg or SAM2Config()
        self._predictor = None
        self._available = None

    def _try_load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            log.warning("SAM2 package not installed. Falling back to Gaussian splat.")
            self._available = False
            return False
        if not Path(self.cfg.weights_path).exists():
            log.warning("SAM2 weights not at %s. Falling back to Gaussian splat.", self.cfg.weights_path)
            self._available = False
            return False
        sam2 = build_sam2(self.cfg.config_name, self.cfg.weights_path, device=self.cfg.device)
        self._predictor = SAM2ImagePredictor(sam2)
        self._available = True
        return True

    def mask_from_points(self, rgb: np.ndarray, points_xy: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        """rgb: (H, W, 3); points_xy: (N, 2) in image coords; labels: (N,) {1=fg, 0=bg}."""
        if labels is None:
            labels = np.ones(len(points_xy), dtype=np.int32)
        if not self._try_load():
            return _gaussian_splat(rgb.shape[:2], points_xy, sigma=18)
        self._predictor.set_image(rgb)
        masks, scores, _ = self._predictor.predict(
            point_coords=points_xy.astype(np.float32),
            point_labels=labels.astype(np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(np.float32)

    def mask_from_box(self, rgb: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        if not self._try_load():
            h, w = rgb.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            m = np.zeros((h, w), dtype=np.float32)
            m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1.0
            return m
        self._predictor.set_image(rgb)
        masks, scores, _ = self._predictor.predict(
            box=xyxy.astype(np.float32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(np.float32)


def _gaussian_splat(hw: tuple[int, int], points_xy: np.ndarray, sigma: float) -> np.ndarray:
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    for x, y in points_xy:
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        out = np.maximum(out, np.exp(-d2 / (2 * sigma ** 2)))
    return out
