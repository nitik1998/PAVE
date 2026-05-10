"""Dump every overall CSV under outputs/tables/ and outputs/tables_test/ as a
single results.json for the report or downstream tooling.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = dict(rows[0])
    for k, v in list(out.items()):
        if k in {"method", "actual_backbone"}:
            continue
        try:
            out[k] = float(v)
        except (ValueError, TypeError):
            pass
    return out


import math


def _scrub_nans(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _scrub_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_nans(v) for v in obj]
    return obj


def main(out_path: str):
    res = {"val": {}, "test": {}}
    for split, root in [("val", Path("outputs/tables")), ("test", Path("outputs/tables_test"))]:
        for f in sorted(root.glob("*_overall.csv")):
            d = _load(f)
            if d is None:
                continue
            res[split][f.stem.replace("_overall", "")] = d
    res = _scrub_nans(res)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    logging.info("wrote %s with %d val and %d test entries",
                 out_path, len(res["val"]), len(res["test"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main("outputs/results.json")
