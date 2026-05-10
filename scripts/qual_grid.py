"""Thin wrapper around src.eval.qual_grid.main."""

from __future__ import annotations

import argparse
import logging

from src.eval.qual_grid import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["dinov3", "siglip2", "qwen25vl", "molmoe"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--split-file", default="data/umd/splits/test.json")
    ap.add_argument("--taxonomy", default="configs/affordance_taxonomy.yaml")
    ap.add_argument("--pred-root", default="outputs/predictions")
    ap.add_argument("--out", default="outputs/figures/qual_grid.png")
    args = ap.parse_args()
    main(args.methods, args.n, args.split_file, args.taxonomy, args.pred_root, args.out)
