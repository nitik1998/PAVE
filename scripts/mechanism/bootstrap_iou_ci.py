"""Bootstrap confidence intervals on per-class IoU for each backbone.

Image-level bootstrap: resample images with replacement (1000 times),
sum per-image confusion matrices in each resample, recompute per-class
IoU. Reports 95% CI lower/upper for each (method, class) cell.

This makes the H2 "cut −0.27" claim defensible: the CI for π0's cut IoU
should sit clearly below the CI for standalone's cut IoU.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.dataset_umd import UMDSubset
from src.eval.metrics import confusion_matrix as confmat

CLASSES = ["bg", "grasp", "cut", "scoop", "contain", "support"]
FOREGROUND = CLASSES[1:]


METHOD_IMAGE_SIZE = {
    "dinov2": 448,
    "dinov2_large": 448,
    "siglip2": 256,
    "openpi_siglip": 384,
    "pi0_siglip": 224,
    "pi05_siglip": 224,
    "florence2": 256,
    "random_features": 224,
}


def predict_to_label(pred_chw: np.ndarray) -> np.ndarray:
    """5-channel softmax (no bg) -> hard 6-class label.

    Background is reconstructed as 1 - max(foreground_prob).
    """
    bg = np.clip(1.0 - pred_chw.max(axis=0, keepdims=True), 0, 1)
    full = np.concatenate([bg, pred_chw], axis=0)
    return full.argmax(axis=0).astype(np.uint8)


def load_subset_for_method(method: str, split_file: Path, taxonomy: Path) -> UMDSubset:
    sz = METHOD_IMAGE_SIZE[method]
    return UMDSubset.from_split_file(split_file, taxonomy, image_size=sz)


def per_image_confmats(method: str, split_name: str, predictions_root: Path,
                       split_file: Path, taxonomy: Path, num_classes: int = 6
                       ) -> list[np.ndarray]:
    sub = load_subset_for_method(method, split_file, taxonomy)
    pred_dir = predictions_root / method
    if split_name == "test":
        pred_dir = predictions_root / "test" / method
    cms = []
    for s, rgb, lbl in sub:
        pred_path = pred_dir / f"{s.object_id}.npy"
        if not pred_path.exists():
            return None
        pred = np.load(pred_path)
        pred_label = predict_to_label(pred)
        if pred_label.shape != lbl.shape:
            # resize prediction to match label using nearest
            from PIL import Image
            pred_img = Image.fromarray(pred_label, mode="L").resize(lbl.shape[::-1], Image.NEAREST)
            pred_label = np.asarray(pred_img, dtype=np.uint8)
        cm = confmat(pred_label, lbl, num_classes=num_classes)
        cms.append(cm)
    return cms


def iou_from_confmat(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    return iou


def bootstrap(cms: list[np.ndarray], n_boot: int, num_classes: int = 6, rng=None) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(cms)
    cms_arr = np.stack(cms, axis=0)  # (n, K, K)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        agg = cms_arr[idx].sum(axis=0)
        iou = iou_from_confmat(agg)
        samples.append(iou)
    samples = np.stack(samples, axis=0)  # (n_boot, K)
    mean = np.nanmean(samples, axis=0)
    lo = np.nanpercentile(samples, 2.5, axis=0)
    hi = np.nanpercentile(samples, 97.5, axis=0)
    full_cm = cms_arr.sum(axis=0)
    iou_full = iou_from_confmat(full_cm)
    return dict(
        iou_point=iou_full,
        iou_mean=mean,
        iou_ci_lo=lo,
        iou_ci_hi=hi,
        n_images=n,
        n_boot=n_boot,
    )


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("bootstrap")

    methods = args.methods.split(",")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    table = []
    for split_name, split_file in [
        ("val", ROOT / "data/umd/splits_500/val.json"),
        ("test", ROOT / "data/umd/splits_500/test.json"),
    ]:
        for m in methods:
            cms = per_image_confmats(
                method=m,
                split_name=split_name,
                predictions_root=Path(args.predictions),
                split_file=split_file,
                taxonomy=ROOT / "configs/affordance_taxonomy.yaml",
            )
            if cms is None:
                log.warning("skip %s on %s (no predictions)", m, split_name)
                continue
            res = bootstrap(cms, n_boot=args.n_boot, rng=rng)
            log.info("[%s/%s] n=%d miou=%.3f (CI [%.3f, %.3f])",
                     m, split_name, res["n_images"],
                     float(np.nanmean(res["iou_point"][1:])),
                     float(np.nanmean(res["iou_ci_lo"][1:])),
                     float(np.nanmean(res["iou_ci_hi"][1:])))
            for ci, name in enumerate(CLASSES):
                table.append(dict(
                    method=m, split=split_name, cls=name,
                    iou=float(res["iou_point"][ci]) if ci < len(res["iou_point"]) else float("nan"),
                    ci_lo=float(res["iou_ci_lo"][ci]) if ci < len(res["iou_ci_lo"]) else float("nan"),
                    ci_hi=float(res["iou_ci_hi"][ci]) if ci < len(res["iou_ci_hi"]) else float("nan"),
                    n_images=int(res["n_images"]),
                ))
            with open(out_dir / f"{m}_{split_name}_bootstrap.json", "w") as f:
                json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in res.items()}, f, indent=2)
    out_csv = out_dir / "bootstrap_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "split", "cls", "iou", "ci_lo", "ci_hi", "n_images"])
        w.writeheader()
        for r in table:
            w.writerow(r)
    log.info("wrote %s", out_csv)

    # === Plot: per-class IoU with CIs for SigLIP family on val ===
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    siglip_methods = ["openpi_siglip", "pi0_siglip", "pi05_siglip"]
    cls_show = FOREGROUND
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    w = 0.27
    xs = np.arange(len(cls_show))
    colors = ["#4a72c9", "#c94a72", "#aa6c39"]
    labels = ["standalone SigLIP-So400m", "π0 SigLIP", "π0.5 SigLIP"]
    for k, m in enumerate(siglip_methods):
        rows = [r for r in table if r["method"] == m and r["split"] == "val"]
        rows = {r["cls"]: r for r in rows}
        vals = [rows[c]["iou"] for c in cls_show]
        lo = [rows[c]["iou"] - rows[c]["ci_lo"] for c in cls_show]
        hi = [rows[c]["ci_hi"] - rows[c]["iou"] for c in cls_show]
        ax.bar(xs + (k - 1) * w, vals, w, yerr=[lo, hi], capsize=3,
               color=colors[k], label=labels[k])
    ax.set_xticks(xs)
    ax.set_xticklabels(cls_show)
    ax.set_ylabel(f"IoU on UMD val (95% CI, n_boot={args.n_boot})")
    ax.set_title("H2: VLA fine-tuning degrades affordance class-asymmetrically — "
                 "CIs separate the loss")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "h2_per_class_ci.png")
    log.info("wrote %s", out_dir / "h2_per_class_ci.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=
        "dinov2,dinov2_large,siglip2,openpi_siglip,pi0_siglip,pi05_siglip,florence2,random_features")
    ap.add_argument("--predictions", default="outputs/predictions_500")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/bootstrap")
    args = ap.parse_args()
    main(args)
