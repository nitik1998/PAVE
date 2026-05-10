"""Random-projection probe — sanity baseline.

Replaces the DINOv2 patch features with a fixed random projection of raw
RGB pixels. Same linear-probe head (sklearn LogisticRegression) sits on top.
If a learned backbone produces affordance signal, this baseline should be
catastrophically worse — a clean control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.methods.dinov3_probe import DinoLinearProbe, ProbeConfig

log = logging.getLogger(__name__)


@dataclass
class RandomProbeConfig(ProbeConfig):
    hf_id: str = "random-baseline"
    fallback_hf_id: str | None = None
    image_size: int = 448
    patch_size: int = 14
    feature_dim: int = 768
    seed: int = 0


class RandomFeatureProbe(DinoLinearProbe):
    name = "random_features"

    def __init__(self, cfg: RandomProbeConfig, num_classes: int, foreground_names: list[str]):
        super().__init__(cfg=cfg, num_classes=num_classes, foreground_names=foreground_names)
        self._proj = None  # set in warmup

    def warmup(self) -> None:
        if self._proj is not None:
            return
        rng = np.random.default_rng(self.cfg.seed)
        d_in = 3 * self.cfg.patch_size * self.cfg.patch_size
        self._proj = rng.standard_normal((d_in, self.cfg.feature_dim)).astype(np.float32)
        self._proj /= np.sqrt(d_in)
        log.info("Random-feature probe initialized: in=%d → out=%d", d_in, self.cfg.feature_dim)

    def _extract_patch_features(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        self.warmup()
        S = int(self.cfg.image_size)
        ps = int(self.cfg.patch_size)
        pil = Image.fromarray(rgb).resize((S, S), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        gh = S // ps
        # Tile into patches and project.
        patches = arr.reshape(gh, ps, gh, ps, 3).transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(gh * gh, ps * ps * 3)
        return (patches @ self._proj).astype(np.float32)


def build(num_classes: int, foreground_names: list[str], device: str = "cpu") -> RandomFeatureProbe:
    cfg = RandomProbeConfig(device=device)
    return RandomFeatureProbe(cfg=cfg, num_classes=num_classes, foreground_names=foreground_names)
