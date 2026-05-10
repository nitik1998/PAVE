"""π0 / π0.5 SigLIP-So400m vision-tower probe (stretch goal).

This is the bridge to *Affordance in the Wild* — same probe protocol as
`dinov3_probe.py`, but features come from the encoder embedded inside a VLA.

We try, in order:
  1. Community PyTorch port `allenzren/open-pi-zero` (state dict carries a
     huggingface SigLIP-So400m).
  2. Direct HF SigLIP-So400m (NOT the VLA-finetuned one, but a useful sanity
     baseline if openpi can't be loaded).

The core finding the user's other proposal is targeting is the *delta* between
this probe and a vanilla SigLIP probe. Implementing the delta cleanly is a
1-week task; this module just establishes the entrypoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.methods.dinov3_probe import DinoLinearProbe, ProbeConfig

log = logging.getLogger(__name__)


@dataclass
class OpenPISigLIPConfig(ProbeConfig):
    hf_id: str = "google/siglip-so400m-patch14-384"   # closest open analogue
    fallback_hf_id: str | None = "google/siglip-base-patch16-256"
    image_size: int = 384
    patch_size: int = 14
    feature_dim: int = 1152
    pi_torch_repo: str = "https://github.com/allenzren/open-pi-zero"
    pi_jax_repo: str = "https://github.com/Physical-Intelligence/openpi"
    notes: str = (
        "TODO(week-1): swap _backbone with the SigLIP tower extracted from "
        "the actual π0 / π0.5 checkpoint via openpi. Until that lands we use "
        "the public SigLIP-So400m so the probe protocol is end-to-end runnable."
    )


class OpenPISigLIPProbe(DinoLinearProbe):
    name = "openpi_siglip"

    def warmup(self) -> None:
        if self._backbone is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        log.warning(
            "Loading public SigLIP-So400m as a stand-in for π0's vision tower. "
            "Replace with openpi-extracted weights for the research-proposal experiment."
        )
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.cfg.hf_id)
            full = AutoModel.from_pretrained(self.cfg.hf_id)
        except Exception as e:
            if self.cfg.fallback_hf_id is None:
                raise
            log.warning("Falling back to %s (%s)", self.cfg.fallback_hf_id, e)
            self._processor = AutoImageProcessor.from_pretrained(self.cfg.fallback_hf_id)
            full = AutoModel.from_pretrained(self.cfg.fallback_hf_id)
        self._backbone = full.vision_model.eval().to(self.cfg.device)
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        vc = getattr(self._backbone, "config", None)
        if vc is not None:
            actual_patch = getattr(vc, "patch_size", None)
            actual_image = getattr(vc, "image_size", None)
            if actual_patch is not None:
                self.cfg.patch_size = int(actual_patch)
            if actual_image is not None:
                self.cfg.image_size = int(actual_image)
        log.info("Loaded openpi-siglip (%s) on %s (patch=%d image=%d)",
                 self.cfg.hf_id, self.cfg.device, self.cfg.patch_size, self.cfg.image_size)


def build(num_classes: int, foreground_names: list[str], device: str = "cpu") -> OpenPISigLIPProbe:
    cfg = OpenPISigLIPConfig(device=device)
    return OpenPISigLIPProbe(cfg=cfg, num_classes=num_classes, foreground_names=foreground_names)
