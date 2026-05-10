"""Build a stratified 200-image train/val/test split over UMD tools.

Stratification is per-category (mug, knife, hammer, ...) so each split has
coverage of every object class. Outputs JSON files at
data/umd/splits/{train,val,test}.json with shape:

    [{"id": "..._00012", "rgb": "...", "label": "...", "category": "mug"}, ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from src.eval.dataset_umd import discover_samples


def main(root: str, out_dir: str, n: int, seed: int, ratios: tuple[float, float, float]):
    random.seed(seed)
    samples = discover_samples(root)
    if not samples:
        raise SystemExit(f"No samples found under {root}. Did you run scripts/download_umd.sh?")
    by_cat: dict[str, list] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)

    # Take n stratified by category.
    n_per = max(1, n // max(1, len(by_cat)))
    picked = []
    for cat, lst in by_cat.items():
        random.shuffle(lst)
        picked.extend(lst[:n_per])
    random.shuffle(picked)
    picked = picked[:n]

    n_train = int(len(picked) * ratios[0])
    n_val = int(len(picked) * ratios[1])
    train, val, test = picked[:n_train], picked[n_train:n_train + n_val], picked[n_train + n_val:]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(out / f"{name}.json", "w") as f:
            json.dump(
                [
                    {
                        "id": s.object_id,
                        "rgb": str(s.rgb_path),
                        "label": str(s.label_path),
                        "category": s.category,
                    }
                    for s in split
                ],
                f,
                indent=2,
            )
        logging.info("[%s] %d samples → %s", name, len(split), out / f"{name}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/umd/tools")
    ap.add_argument("--out", default="data/umd/splits")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.root, args.out, args.n, args.seed, (0.7, 0.15, 0.15))
