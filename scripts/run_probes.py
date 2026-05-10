"""Single entrypoint to run any method on the UMD subset and write metrics.

Usage:
    python scripts/run_probes.py --method dinov3 --device cuda
    python scripts/run_probes.py --method qwen25vl --device cpu --n 50

For probe-style methods (dinov3, dinov2, siglip2, openpi_siglip) we fit on the
train split, evaluate on val, and dump per-pixel predictions for the test
split into outputs/predictions/<method>/<id>.npy.

For zero-shot methods (qwen25vl, molmoe) we skip the fit step and run on
val ∪ test directly.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from src.eval.dataset_umd import UMDSubset
from src.eval.metrics import compute_metrics, metrics_to_row
from src.utils.seed import set_seed


def _predict_to_label(pred_chw: np.ndarray) -> np.ndarray:
    bg = np.clip(1.0 - pred_chw.max(axis=0, keepdims=True), 0, 1)
    full = np.concatenate([bg, pred_chw], axis=0)
    return full.argmax(axis=0).astype(np.uint8)


def _build_method(name: str, device: str, taxonomy: str, num_classes: int, foreground_names: list[str], image_size: int):
    if name == "dinov3":
        from src.methods import dinov3_probe

        return dinov3_probe.DinoLinearProbe(
            cfg=dinov3_probe.ProbeConfig(device=device, image_size=image_size, patch_size=16),
            num_classes=num_classes,
            foreground_names=foreground_names,
        )
    if name == "dinov2":
        from src.methods import dinov2_probe

        m = dinov2_probe.build(num_classes=num_classes, foreground_names=foreground_names, device=device)
        m.cfg.image_size = image_size
        m.cfg.patch_size = 14
        return m
    if name == "siglip2":
        from src.methods import siglip2_probe

        m = siglip2_probe.build(num_classes=num_classes, foreground_names=foreground_names, device=device)
        m.cfg.image_size = image_size
        m.cfg.patch_size = 16
        return m
    if name == "openpi_siglip":
        from src.methods import openpi_siglip_probe

        m = openpi_siglip_probe.build(num_classes=num_classes, foreground_names=foreground_names, device=device)
        m.cfg.image_size = image_size
        return m
    if name == "pi0_siglip":
        from src.methods import pi0_siglip_probe

        m = pi0_siglip_probe.build(num_classes=num_classes, foreground_names=foreground_names, device=device)
        m.cfg.image_size = image_size
        return m
    if name == "pi05_siglip":
        from src.methods import pi0_siglip_probe

        m = pi0_siglip_probe.build(num_classes=num_classes, foreground_names=foreground_names, device=device, use_pi05=True)
        m.cfg.image_size = image_size
        return m
    if name == "openvla_siglip":
        from src.methods import openvla_siglip_probe

        m = openvla_siglip_probe.build(num_classes=num_classes, foreground_names=foreground_names, device=device)
        m.cfg.image_size = image_size
        return m
    if name == "qwen25vl":
        from src.methods import qwen25vl_grounding

        return qwen25vl_grounding.build(taxonomy_path=taxonomy, device=device)
    if name == "molmoe":
        from src.methods import molmoe_pointing

        return molmoe_pointing.build(taxonomy_path=taxonomy, device=device)
    if name == "florence2":
        from src.methods import florence2_grounding

        return florence2_grounding.build(taxonomy_path=taxonomy, device=device)
    if name == "dinov2_large":
        from src.methods import dinov2_large_probe

        m = dinov2_large_probe.build(num_classes=num_classes, foreground_names=foreground_names, device=device)
        m.cfg.image_size = image_size
        m.cfg.patch_size = 14
        return m
    if name == "random_features":
        from src.methods import random_baseline

        m = random_baseline.build(num_classes=num_classes, foreground_names=foreground_names, device=device)
        m.cfg.image_size = image_size
        m.cfg.patch_size = 14
        return m
    raise ValueError(f"unknown method {name!r}")


def main(args):
    set_seed(args.seed)
    log = logging.getLogger("run_probes")
    splits = Path(args.splits)
    train_subset = UMDSubset.from_split_file(splits / "train.json", args.taxonomy, image_size=args.image_size)
    val_subset = UMDSubset.from_split_file(splits / "val.json", args.taxonomy, image_size=args.image_size)
    test_subset = UMDSubset.from_split_file(splits / "test.json", args.taxonomy, image_size=args.image_size)
    num_classes = len(train_subset.class_names)
    foreground = [n for n in train_subset.class_names if n != "background"]

    if args.method == "dinov2" and args.image_size % 14 != 0:
        args.image_size = (args.image_size // 14) * 14
        log.info("Adjusted image_size to %d for DINOv2 patch grid", args.image_size)
    if args.method in ("dinov3", "siglip2", "openpi_siglip") and args.image_size % 16 != 0:
        args.image_size = (args.image_size // 16) * 16
        log.info("Adjusted image_size to %d for patch16 grid", args.image_size)
    train_subset.image_size = args.image_size
    val_subset.image_size = args.image_size
    test_subset.image_size = args.image_size
    method = _build_method(args.method, args.device, args.taxonomy, num_classes, foreground, args.image_size)
    method.warmup()

    fit_methods = {"dinov3", "dinov2", "dinov2_large", "siglip2", "openpi_siglip", "pi0_siglip", "pi05_siglip", "openvla_siglip", "random_features"}
    if args.method in fit_methods:
        log.info("Fitting linear probe on %d train samples ...", len(train_subset))
        train_pairs = []
        for _, rgb, lbl in train_subset:
            train_pairs.append((rgb, lbl))
            if args.n is not None and len(train_pairs) >= args.n:
                break
        method.fit(train_pairs, C=args.C)

    pred_dir = Path(args.pred_root) / args.method
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_split == "val":
        eval_subset = val_subset
    elif args.eval_split == "test":
        eval_subset = test_subset
    elif args.eval_split == "auto":
        eval_subset = val_subset if args.method in fit_methods else test_subset
    else:
        raise ValueError(f"unknown --eval-split {args.eval_split!r}")
    rows = []
    confusion_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    eval_cap = args.eval_n if args.eval_n is not None else len(eval_subset)
    log.info("Evaluating on %d images (cap=%d)...", len(eval_subset), eval_cap)
    eval_count = 0
    for s, rgb, lbl in eval_subset:
        if eval_count >= eval_cap:
            break
        eval_count += 1
        pred = method.predict_map(rgb)
        np.save(pred_dir / f"{s.object_id}.npy", pred.astype(np.float32))
        pred_label = _predict_to_label(pred)
        m = compute_metrics(pred_label, lbl, num_classes=num_classes)
        confusion_total += m.confusion
        rows.append({"id": s.object_id, **metrics_to_row(m, train_subset.class_names)})

    # Aggregate metrics from the totalized confusion matrix.
    tp = np.diag(confusion_total)
    fp = confusion_total.sum(0) - tp
    fn = confusion_total.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    actual_name = getattr(method, "name", args.method)
    if actual_name != args.method:
        log.warning("Method requested=%r but actually loaded=%r (fallback)", args.method, actual_name)
    overall = {
        "method": args.method,
        "actual_backbone": actual_name,
        "n": len(rows),
        "mIoU": float(np.nanmean(iou)),
        "pixel_acc": float(tp.sum() / max(confusion_total.sum(), 1)),
        **{f"IoU_{n}": float(iou[i]) for i, n in enumerate(train_subset.class_names)},
    }
    log.info("OVERALL %s", overall)

    out_csv = Path(args.tables_root) / f"{args.method}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    out_overall = Path(args.tables_root) / f"{args.method}_overall.csv"
    with open(out_overall, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overall.keys()))
        w.writeheader()
        w.writerow(overall)
    log.info("wrote %s and %s", out_csv, out_overall)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=[
        "dinov3", "dinov2", "dinov2_large", "siglip2", "openpi_siglip",
        "pi0_siglip", "pi05_siglip", "openvla_siglip",
        "qwen25vl", "molmoe", "florence2", "random_features"
    ])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=None, help="cap train samples used for the linear probe fit")
    ap.add_argument("--eval-n", type=int, default=None, help="cap eval (val/test) samples")
    ap.add_argument("--eval-split", default="auto", choices=["auto", "val", "test"])
    ap.add_argument("--C", type=float, default=1.0, help="logistic regression regularization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--splits", default="data/umd/splits")
    ap.add_argument("--taxonomy", default="configs/affordance_taxonomy.yaml")
    ap.add_argument("--pred-root", default="outputs/predictions")
    ap.add_argument("--tables-root", default="outputs/tables")
    args = ap.parse_args()
    main(args)
