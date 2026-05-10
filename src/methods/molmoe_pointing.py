"""MolmoE-1B-0924 → `<point x="..." y="...">` tokens → SAM 2.1 masks.

Molmo's outputs use absolute coordinates in [0, 100] (percent of image size).
We parse all points per class, scale to image pixels, and refine via SAM 2.1.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import yaml

from src.methods.base import AffordancePredictor
from src.methods.sam2_refine import SAM2Config, SAM2Refiner

log = logging.getLogger(__name__)


_POINT_RE = re.compile(r'<point[^>]*x\s*=\s*"([\d.]+)"[^>]*y\s*=\s*"([\d.]+)"', re.IGNORECASE)
_POINTS_RE = re.compile(r'<points[^>]*>(.*?)</points>', re.IGNORECASE | re.DOTALL)
_INNER_PT_RE = re.compile(r'x\d*\s*=\s*"([\d.]+)"\s+y\d*\s*=\s*"([\d.]+)"', re.IGNORECASE)


@dataclass
class MolmoEConfig:
    hf_id: str = "allenai/MolmoE-1B-0924"
    device: str = "cpu"
    max_new_tokens: int = 200
    refine_with_sam2: bool = True


class MolmoEPointer(AffordancePredictor):
    name = "molmoe"

    def __init__(self, cfg: MolmoEConfig, taxonomy_path: str):
        self.cfg = cfg
        with open(taxonomy_path) as f:
            tax = yaml.safe_load(f)
        self.classes = [c for c in tax["classes"] if c["id"] != 0]
        self.foreground_class_names = [c["name"] for c in self.classes]
        self._model = None
        self._processor = None
        self._sam2 = SAM2Refiner(SAM2Config(device=cfg.device)) if cfg.refine_with_sam2 else None

    def warmup(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        log.info("Loading %s on %s", self.cfg.hf_id, self.cfg.device)
        self._processor = AutoProcessor.from_pretrained(self.cfg.hf_id, trust_remote_code=True)
        torch_dtype = torch.float16 if self.cfg.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.hf_id, trust_remote_code=True, torch_dtype=torch_dtype
        ).eval().to(self.cfg.device)

    def _generate(self, rgb: np.ndarray, phrase: str) -> str:
        import torch

        self.warmup()
        prompt = f"Point to the {phrase} in this image."
        inputs = self._processor.process(images=[rgb], text=prompt)
        inputs = {k: v.to(self.cfg.device).unsqueeze(0) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate_from_batch(
                inputs,
                generation_config={
                    "max_new_tokens": self.cfg.max_new_tokens,
                    "stop_strings": ["<|endoftext|>"],
                    "do_sample": False,
                },
                tokenizer=self._processor.tokenizer,
            )
        gen_tokens = out[0, inputs["input_ids"].size(1):]
        return self._processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    @staticmethod
    def _parse_points(text: str) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = [
            (float(x), float(y)) for x, y in _POINT_RE.findall(text)
        ]
        for inner in _POINTS_RE.findall(text):
            pts.extend((float(x), float(y)) for x, y in _INNER_PT_RE.findall(inner))
        return pts

    def predict_map(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        c = len(self.classes)
        out = np.zeros((c, h, w), dtype=np.float32)
        for i, cls in enumerate(self.classes):
            phrase = cls["prompts"][0]
            text = self._generate(rgb, phrase)
            pts = self._parse_points(text)
            if not pts:
                continue
            xy = np.array([[x / 100.0 * w, y / 100.0 * h] for x, y in pts], dtype=np.float32)
            if self._sam2 is not None:
                mask = self._sam2.mask_from_points(rgb, xy)
            else:
                from src.methods.sam2_refine import _gaussian_splat

                mask = _gaussian_splat((h, w), xy, sigma=h * 0.04)
            out[i] = np.maximum(out[i], mask)
        return out


def build(taxonomy_path: str, device: str = "cpu") -> MolmoEPointer:
    return MolmoEPointer(MolmoEConfig(device=device), taxonomy_path)
