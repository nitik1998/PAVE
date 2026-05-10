"""Common interface for every affordance-extraction method.

A predictor turns a single RGB image (HxWx3 uint8) into a per-pixel,
per-class affordance map of shape (C, H, W) with values in [0, 1].
C = number of foreground classes (no background). Argmax over a synthetic
background channel is done downstream.

The number-of-classes is dictated by the locked taxonomy
(configs/affordance_taxonomy.yaml). Predictors that natively output coarser
categories (e.g. a single "grasp" point) must broadcast zeros for the other
channels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionMeta:
    method: str
    image_id: str
    extra: dict | None = None


class AffordancePredictor(ABC):
    name: str = "abstract"
    foreground_class_names: list[str] = []

    @abstractmethod
    def predict_map(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (H, W, 3) uint8 → (C, H, W) float32 in [0, 1]."""

    def predict_batch(self, rgbs: list[np.ndarray]) -> list[np.ndarray]:
        return [self.predict_map(x) for x in rgbs]

    def warmup(self) -> None:
        """Optional override: load weights, compile, etc. Called once before a sweep."""
        return None
