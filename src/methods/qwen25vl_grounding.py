"""Qwen2.5-VL-3B-Instruct → bounding boxes per affordance class → SAM 2.1 masks.

Per-class procedure:
    1. Build a prompt asking for `[{"bbox_2d": [x1,y1,x2,y2], "label": "..."}]`.
    2. Parse JSON from the model output (best-effort regex if JSON parse fails).
    3. Convert each box to a SAM 2.1 mask. Combine into a single channel by
       per-pixel max.

The result is (C, H, W) where C = number of foreground classes in the
taxonomy, ordered by class id 1..C.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import numpy as np
import yaml

from src.methods.base import AffordancePredictor
from src.methods.sam2_refine import SAM2Config, SAM2Refiner

log = logging.getLogger(__name__)


_BBOX_RE = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")


@dataclass
class QwenGroundingConfig:
    hf_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    fallback_hf_id: str | None = None
    device: str = "cpu"
    image_size: int = 448
    max_new_tokens: int = 256
    refine_with_sam2: bool = True


class Qwen25VLGrounder(AffordancePredictor):
    name = "qwen25vl"

    def __init__(self, cfg: QwenGroundingConfig, taxonomy_path: str):
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
        from transformers import AutoProcessor

        torch_dtype = torch.bfloat16 if self.cfg.device == "cuda" else torch.float32
        log.info("Loading %s on %s (dtype=%s)", self.cfg.hf_id, self.cfg.device, torch_dtype)
        from transformers import Qwen2VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.cfg.hf_id)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.hf_id, torch_dtype=torch_dtype, device_map=self.cfg.device,
            low_cpu_mem_usage=True,
        ).eval()
        if self.cfg.hf_id.startswith("Qwen/Qwen2-VL"):
            self.name = "qwen2vl_2b"

    def _build_messages(self, rgb: np.ndarray, affordance_phrase: str, class_name: str):
        from PIL import Image

        pil = Image.fromarray(rgb)
        prompt = (
            f"Detect every region of the image that is the {affordance_phrase}. "
            f"Return ONLY a JSON list of bounding boxes in absolute pixel "
            f"coordinates of the form "
            f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{class_name}"}}]. '
            f"If there is no such region, return []."
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _parse_boxes(self, text: str) -> list[tuple[int, int, int, int]]:
        try:
            j = json.loads(text)
            return [tuple(int(v) for v in obj["bbox_2d"]) for obj in j if "bbox_2d" in obj]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return [
            tuple(int(v) for v in re.findall(r"-?\d+", m)) for m in _BBOX_RE.findall(text)
        ]

    def _generate(self, rgb: np.ndarray, phrase: str, class_name: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        self.warmup()
        messages = self._build_messages(rgb, phrase, class_name)
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.cfg.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def predict_map(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        c = len(self.classes)
        out = np.zeros((c, h, w), dtype=np.float32)
        for i, cls in enumerate(self.classes):
            phrase = cls["prompts"][0]
            text = self._generate(rgb, phrase, cls["name"])
            boxes = self._parse_boxes(text)
            for box in boxes:
                x1, y1, x2, y2 = (
                    max(0, min(w - 1, box[0])),
                    max(0, min(h - 1, box[1])),
                    max(0, min(w, box[2])),
                    max(0, min(h, box[3])),
                )
                if x2 <= x1 or y2 <= y1:
                    continue
                if self._sam2 is not None:
                    mask = self._sam2.mask_from_box(rgb, np.array([x1, y1, x2, y2]))
                else:
                    mask = np.zeros((h, w), dtype=np.float32)
                    mask[y1:y2, x1:x2] = 1.0
                out[i] = np.maximum(out[i], mask)
        return out


def build(taxonomy_path: str, device: str = "cpu") -> Qwen25VLGrounder:
    return Qwen25VLGrounder(QwenGroundingConfig(device=device), taxonomy_path)
