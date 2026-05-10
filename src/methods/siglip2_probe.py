"""SigLIP 2 vision tower + linear probe.

The vision encoder is `SiglipVisionModel` inside transformers. Patch tokens
live in `last_hidden_state` after dropping the optional pooled token. For
naflex models we still feed a fixed square image to keep the patch grid
deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.methods.dinov3_probe import DinoLinearProbe, ProbeConfig

log = logging.getLogger(__name__)


@dataclass
class SigLIP2ProbeConfig(ProbeConfig):
    hf_id: str = "google/siglip2-base-patch16-naflex"
    fallback_hf_id: str | None = "google/siglip-base-patch16-256"
    image_size: int = 384
    patch_size: int = 16
    feature_dim: int = 768


class SigLIP2LinearProbe(DinoLinearProbe):
    name = "siglip2"

    def warmup(self) -> None:
        if self._backbone is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        try:
            self._processor = AutoImageProcessor.from_pretrained(self.cfg.hf_id)
            full = AutoModel.from_pretrained(self.cfg.hf_id)
        except Exception as e:
            if self.cfg.fallback_hf_id is None:
                raise
            log.warning("Falling back to %s (%s)", self.cfg.fallback_hf_id, e)
            self._processor = AutoImageProcessor.from_pretrained(self.cfg.fallback_hf_id)
            full = AutoModel.from_pretrained(self.cfg.fallback_hf_id)
            self.name = "siglip"
        # Use the vision tower only.
        self._backbone = full.vision_model.eval().to(self.cfg.device)
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        # Force image_size to match the position-embedding grid the model
        # was trained at. SigLIP variants do NOT support arbitrary input
        # sizes, so we cannot resize freely.
        vc = getattr(self._backbone, "config", None)
        if vc is not None:
            actual_patch = getattr(vc, "patch_size", None)
            actual_image = getattr(vc, "image_size", None)
            if actual_patch is not None and actual_patch != self.cfg.patch_size:
                log.warning("SigLIP patch_size %d -> %d", self.cfg.patch_size, actual_patch)
                self.cfg.patch_size = int(actual_patch)
            if actual_image is not None and actual_image != self.cfg.image_size:
                log.warning("SigLIP image_size %d -> %d (fixed posemb)", self.cfg.image_size, actual_image)
                self.cfg.image_size = int(actual_image)
        log.info("Loaded SigLIP vision tower on %s (patch=%d image=%d)",
                 self.cfg.device, self.cfg.patch_size, self.cfg.image_size)


def build(num_classes: int, foreground_names: list[str], device: str = "cpu") -> SigLIP2LinearProbe:
    cfg = SigLIP2ProbeConfig(device=device)
    return SigLIP2LinearProbe(cfg=cfg, num_classes=num_classes, foreground_names=foreground_names)
