"""UMD Part Affordance Dataset loader.

The published dataset layout is::

    part-affordance-dataset/tools/<category>/<id>/<id>_rgb.jpg
                                              /<id>_label.mat   # {'gt': HxW uint8}
                                              /<id>_depth.png   # 16-bit depth (optional)

Native UMD class ids (1..7): grasp, cut, scoop, contain, pound, support, w-grasp.
Background = 0.

We remap to the 5-class taxonomy in configs/affordance_taxonomy.yaml:
    {0: bg, 1: grasp(+w-grasp), 2: cut, 3: scoop, 4: contain, 5: support}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image


def load_taxonomy(path: str | Path) -> tuple[dict[int, int], list[str]]:
    """Returns (umd_id → our_id) mapping and ordered class-name list."""
    with open(path) as f:
        tax = yaml.safe_load(f)
    mapping = {}
    names = []
    for cls in tax["classes"]:
        names.append(cls["name"])
        for u in cls["umd_ids"]:
            mapping[int(u)] = int(cls["id"])
    return mapping, names


_LABEL_KEYS = ("gt_label", "gt", "label", "labels")


def _read_label_mat(path: str | Path) -> np.ndarray:
    """Read the per-pixel label field from a UMD .mat label file.

    UMD's `_label.mat` files store the per-pixel affordance map under the key
    ``gt_label`` (older releases used ``gt``). We try scipy first (v5/v6 .mat),
    fall through to h5py for v7.3 .mat, and try several known key names.
    """
    p = str(path)
    last_err: Exception | None = None
    try:
        from scipy.io import loadmat

        data = loadmat(p)
        for key in _LABEL_KEYS:
            if key in data and data[key] is not None:
                return np.asarray(data[key], dtype=np.uint8)
        last_err = KeyError(f"no known label key in {p}; got {list(data.keys())}")
    except (NotImplementedError, ValueError) as e:
        last_err = e
    try:
        import h5py

        with h5py.File(p, "r") as f:
            for key in _LABEL_KEYS:
                if key in f:
                    return np.array(f[key]).T.astype(np.uint8)
            last_err = KeyError(f"no known label key in {p}; got {list(f.keys())}")
    except Exception as e:
        last_err = e
    raise RuntimeError(f"could not read label from {p}: {last_err}")


def remap_label(label: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(label, dtype=np.uint8)
    for src, dst in mapping.items():
        out[label == src] = dst
    return out


@dataclass
class UMDSample:
    rgb_path: Path
    label_path: Path
    category: str
    object_id: str

    def load_rgb(self, size: int | None = None) -> np.ndarray:
        img = Image.open(self.rgb_path).convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.BILINEAR)
        return np.asarray(img)

    def load_label(self, mapping: dict[int, int], size: int | None = None) -> np.ndarray:
        lbl = _read_label_mat(self.label_path)
        lbl = remap_label(lbl, mapping)
        if size is not None:
            img = Image.fromarray(lbl, mode="L").resize((size, size), Image.NEAREST)
            lbl = np.asarray(img)
        return lbl


def discover_samples(root: str | Path) -> list[UMDSample]:
    """Walk root looking for <root>/**/*_rgb.jpg with a sibling *_label.mat.

    UMD tools layout: <root>/<category>_<instance>/<full_id>_rgb.jpg.
    Category is the prefix of the parent directory name before the first
    underscore (e.g. ``mug_19`` → ``mug``).
    """
    root = Path(root)
    out: list[UMDSample] = []
    for rgb in root.rglob("*_rgb.jpg"):
        stem = rgb.name[: -len("_rgb.jpg")]
        label = rgb.with_name(f"{stem}_label.mat")
        if not label.exists():
            continue
        parent_name = rgb.parent.name
        category = parent_name.split("_", 1)[0] if "_" in parent_name else parent_name
        out.append(UMDSample(rgb_path=rgb, label_path=label, category=category, object_id=stem))
    out.sort(key=lambda s: s.rgb_path.as_posix())
    return out


@dataclass
class UMDSubset:
    samples: list[UMDSample]
    mapping: dict[int, int]
    class_names: list[str]
    image_size: int = 448

    @classmethod
    def from_split_file(
        cls,
        split_file: str | Path,
        taxonomy_path: str | Path,
        image_size: int = 448,
    ) -> "UMDSubset":
        mapping, names = load_taxonomy(taxonomy_path)
        with open(split_file) as f:
            entries = json.load(f)
        samples = [
            UMDSample(
                rgb_path=Path(e["rgb"]),
                label_path=Path(e["label"]),
                category=e.get("category", "_"),
                object_id=e["id"],
            )
            for e in entries
        ]
        return cls(samples=samples, mapping=mapping, class_names=names, image_size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[tuple[UMDSample, np.ndarray, np.ndarray]]:
        for s in self.samples:
            rgb = s.load_rgb(size=self.image_size)
            lbl = s.load_label(self.mapping, size=self.image_size)
            yield s, rgb, lbl
