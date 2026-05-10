"""Outer driver: run train_h3.py across all 4 arms × N seeds, sequentially.

Output:
  outputs/h3/{A,B,C,D}/seed*/{model.zip, eval.json, train_log.csv}
  outputs/h3/sweep_results.csv
  outputs/figures/policy_curves.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np


def main(arms: list[str], seeds: list[int], steps: int, predictor_ckpt: str | None,
         out_dir: str, eval_episodes: int):
    log = logging.getLogger("h3_sweep")
    from scripts.train_h3 import TrainCfg, main as train_one

    rows = []
    for arm in arms:
        ckpt = predictor_ckpt if arm == "D" else None
        for seed in seeds:
            cfg = TrainCfg(
                arm=arm, seed=seed, steps=steps,
                eval_episodes=eval_episodes,
                out_dir=out_dir, predictor_ckpt=ckpt,
                no_save_model=False,
            )
            log.info(">>> arm=%s seed=%d steps=%d", arm, seed, steps)
            metrics = train_one(cfg)
            rows.append({
                "arm": arm,
                "seed": seed,
                "steps": steps,
                "success_rate": metrics["success_rate"],
                "mean_return": metrics["mean_return"],
                "std_return": metrics["std_return"],
            })

    out_csv = Path(out_dir) / "sweep_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", out_csv)

    by_arm: dict[str, list[float]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r["success_rate"])
    summary = {a: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
               for a, v in by_arm.items()}
    with open(Path(out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--predictor-ckpt", default="outputs/checkpoints/panda_heatmap_head.joblib")
    ap.add_argument("--out-dir", default="outputs/h3")
    args = ap.parse_args()
    main(args.arms, args.seeds, args.steps, args.predictor_ckpt, args.out_dir, args.eval_episodes)
