"""Probe π0's vision tower (PaliGemma's SigLIP-So400m) for affordance.

H2 direct test: ``Δ = mIoU(π0-SigLIP probe) - mIoU(standalone SigLIP-So400m probe)``
quantifies how VLA fine-tuning shifts the affordance signal.

We extract just the vision-tower keys from ``lerobot/pi0_base/model.safetensors``,
load them into a fresh ``transformers.SiglipVisionModel``, and run the same
linear-probe protocol as ``DinoLinearProbe``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.methods.dinov3_probe import DinoLinearProbe, ProbeConfig

log = logging.getLogger(__name__)


@dataclass
class Pi0SigLIPConfig(ProbeConfig):
    pi0_repo: str = "lerobot/pi0_base"
    pi0_safetensors: str = "model.safetensors"
    siglip_arch: str = "google/siglip-so400m-patch14-224"
    fallback_arch: str | None = "google/siglip-so400m-patch14-384"
    image_size: int = 224                     # π0 uses 224×224 inputs
    patch_size: int = 14
    feature_dim: int = 1152
    use_pi05: bool = False


class Pi0SigLIPProbe(DinoLinearProbe):
    name = "pi0_siglip"

    def warmup(self) -> None:
        if self._backbone is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import AutoImageProcessor, AutoModel

        repo = "lerobot/pi05_base" if self.cfg.use_pi05 else self.cfg.pi0_repo
        log.info("Loading π0 vision tower from %s ...", repo)
        ckpt_path = hf_hub_download(repo, self.cfg.pi0_safetensors)

        # Architecture skeleton: SigLIP vision tower at the matching hf id.
        try:
            log.info("Building skeleton from %s", self.cfg.siglip_arch)
            self._processor = AutoImageProcessor.from_pretrained(self.cfg.siglip_arch)
            full = AutoModel.from_pretrained(self.cfg.siglip_arch)
        except Exception as e:
            if self.cfg.fallback_arch is None:
                raise
            log.warning("Falling back to %s (%s)", self.cfg.fallback_arch, type(e).__name__)
            self._processor = AutoImageProcessor.from_pretrained(self.cfg.fallback_arch)
            full = AutoModel.from_pretrained(self.cfg.fallback_arch)
        skeleton = full.vision_model

        skeleton_keys = set(skeleton.state_dict().keys())

        # Find the vision-tower keys inside the π0 checkpoint.
        log.info("Scanning π0 tensor keys for SigLIP/PaliGemma vision tower ...")
        pi0_to_skeleton: dict[str, str] = {}
        with safe_open(ckpt_path, framework="pt") as f:
            all_keys = list(f.keys())

        # Heuristic: PaliGemma stores vision tower under
        # 'paligemma_with_expert.paligemma.vision_tower.vision_model.<...>'
        # or 'paligemma.vision_tower.vision_model.<...>'.
        prefixes_to_try = (
            "paligemma_with_expert.paligemma.model.vision_tower.vision_model.",
            "paligemma_with_expert.paligemma.vision_tower.vision_model.",
            "paligemma.model.vision_tower.vision_model.",
            "paligemma.vision_tower.vision_model.",
            "model.vision_tower.vision_model.",
            "vision_tower.vision_model.",
            "vision_model.",
        )
        prefix = None
        for p in prefixes_to_try:
            sample = next((k for k in all_keys if k.startswith(p)), None)
            if sample is not None:
                prefix = p
                break
        if prefix is None:
            log.warning(
                "Could not find vision-tower prefix in π0 keys. "
                "First 10 keys: %s", all_keys[:10]
            )
            raise RuntimeError("π0 SigLIP weights not found in expected prefix")
        log.info("Using prefix=%r", prefix)

        loaded = {}
        with safe_open(ckpt_path, framework="pt") as f:
            for k in all_keys:
                if not k.startswith(prefix):
                    continue
                stem = k[len(prefix):]
                if stem in skeleton_keys:
                    loaded[stem] = f.get_tensor(k)
        log.info("Loaded %d / %d vision-tower tensors from π0", len(loaded), len(skeleton_keys))
        if not loaded:
            raise RuntimeError("No matching SigLIP keys found in π0 checkpoint")

        missing, unexpected = skeleton.load_state_dict(loaded, strict=False)
        if missing:
            log.warning("Missing keys after π0 load (using skeleton init): %d (e.g. %s)",
                        len(missing), missing[:3])
        self._backbone = skeleton.eval().to(self.cfg.device)
        for p in self._backbone.parameters():
            p.requires_grad_(False)

        # Force the image_size to match the π0 training config (224×224).
        vc = getattr(self._backbone, "config", None)
        if vc is not None:
            actual_image = getattr(vc, "image_size", None)
            actual_patch = getattr(vc, "patch_size", None)
            # Note: even though SigLIP-So400m's *own* config says 384, π0 fine-tunes at 224.
            # We trust π0's config here.
            if actual_patch is not None:
                self.cfg.patch_size = int(actual_patch)
        log.info("π0 SigLIP loaded; patch=%d image=%d feature_dim=%d",
                 self.cfg.patch_size, self.cfg.image_size, self.cfg.feature_dim)


def build(num_classes: int, foreground_names: list[str], device: str = "cpu",
          use_pi05: bool = False) -> Pi0SigLIPProbe:
    cfg = Pi0SigLIPConfig(device=device, use_pi05=use_pi05)
    p = Pi0SigLIPProbe(cfg=cfg, num_classes=num_classes, foreground_names=foreground_names)
    if use_pi05:
        p.name = "pi05_siglip"
    return p
