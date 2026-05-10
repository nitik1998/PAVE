"""Microsoft Florence-2 (770M, open-weight) → phrase-grounded boxes → SAM 2.1 masks.

Per affordance class we issue a `<CAPTION_TO_PHRASE_GROUNDING>` task with the
class's natural-language prompt; Florence-2 returns bounding boxes in absolute
pixel coords. Boxes are then refined by SAM 2.1 (or fall back to filled
rectangles when SAM 2 weights are missing).

Open-weight, no HF auth required. Fits in <2 GB CPU memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import yaml

from src.methods.base import AffordancePredictor
from src.methods.sam2_refine import SAM2Config, SAM2Refiner

log = logging.getLogger(__name__)


@dataclass
class Florence2Config:
    hf_id: str = "microsoft/Florence-2-base"
    device: str = "cpu"
    refine_with_sam2: bool = True
    max_new_tokens: int = 256


class Florence2Grounder(AffordancePredictor):
    name = "florence2"

    def __init__(self, cfg: Florence2Config, taxonomy_path: str):
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

    def _ground(self, rgb: np.ndarray, phrase: str) -> list[tuple[int, int, int, int]]:
        import torch
        from PIL import Image

        self.warmup()
        pil = Image.fromarray(rgb)
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = task_prompt + " " + phrase
        inputs = self._processor(text=prompt, images=pil, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            text, task=task_prompt, image_size=(pil.width, pil.height)
        )
        out = parsed.get(task_prompt, {})
        boxes = out.get("bboxes", [])
        return [tuple(int(round(v)) for v in b) for b in boxes]

    def predict_map(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        c = len(self.classes)
        out = np.zeros((c, h, w), dtype=np.float32)
        for i, cls in enumerate(self.classes):
            phrase = cls["prompts"][0]
            try:
                boxes = self._ground(rgb, phrase)
            except Exception as e:
                log.warning("Florence2 grounding failed for %r: %s", phrase, e)
                continue
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if self._sam2 is not None:
                    mask = self._sam2.mask_from_box(rgb, np.array([x1, y1, x2, y2]))
                else:
                    mask = np.zeros((h, w), dtype=np.float32)
                    mask[y1:y2, x1:x2] = 1.0
                out[i] = np.maximum(out[i], mask)
        return out


def build(taxonomy_path: str, device: str = "cpu") -> Florence2Grounder:
    return Florence2Grounder(Florence2Config(device=device), taxonomy_path)
