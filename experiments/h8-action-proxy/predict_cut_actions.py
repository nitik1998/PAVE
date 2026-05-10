"""H8 Part 2: cut-task action proxy on UMD knife / shears / scissors images.

Defines a per-image "optimal action" purely from cut-affordance ground truth:
  - HANDLE_xy  = centroid of grasp pixels (where to grab)
  - BLADE_xy   = centroid of cut pixels (where the cutting edge is)
  - ORIENTATION = unit vector from blade to handle (which way is the knife pointing)

A sane manipulation policy that grasps the knife correctly must encode this
information. We test whether each encoder's frozen features can predict
these targets with a Ridge regression.

Hypothesis: encoders that lost cut-class affordance (π0, OpenVLA, π0.5)
should predict orientation worse than encoders that preserved it
(DINOv2-large, standalone SigLIP). The adapter on top of π0 should recover
most of the gap.

This is the cut-class half of the H8 crossover.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.dataset_umd import UMDSubset
import inspect

# Reuse featurizers from Part 1 for consistency. Directory has hyphens so we
# load the file by absolute path.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "h8_predict_actions",
    str(ROOT / "experiments" / "h8-action-proxy" / "predict_actions.py"),
)
_pa = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pa)
featurize_dinov2 = _pa.featurize_dinov2
featurize_hf_siglip = _pa.featurize_hf_siglip
featurize_timm_siglip = _pa.featurize_timm_siglip
build_pi0_full = _pa.build_pi0_full
build_openvla_full = _pa.build_openvla_full
build_standalone = _pa.build_standalone
featurize_adapter_pi0 = _pa.featurize_adapter_pi0
ridge_eval = _pa.ridge_eval

CLASSES = ["bg", "grasp", "cut", "scoop", "contain", "support"]


def select_cut_objects(subset: UMDSubset) -> list:
    """Pick categories whose objects predominantly carry both grasp+cut: knife, shears, scissors, saw."""
    cut_cats = {"knife", "shears", "scissors", "saw"}
    return [s for s in subset.samples if s.category in cut_cats]


def compute_targets(samples, mapping, image_size=224) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rgbs, handle_xy, blade_xy, orient_xy).
    Coordinates normalized to [-1, 1] image plane.
    Drops images where either handle or blade pixels are absent.
    """
    rgbs, handles, blades, orients = [], [], [], []
    for s in samples:
        rgb = s.load_rgb(size=image_size)
        lbl = s.load_label(mapping, size=image_size)
        H, W = lbl.shape
        ys, xs = np.indices((H, W))
        # In the 5-class taxonomy, grasp=1, cut=2.
        grasp_mask = lbl == 1
        cut_mask = lbl == 2
        if grasp_mask.sum() < 20 or cut_mask.sum() < 20:
            continue
        # Normalize to [-1,1].
        hx = 2 * (xs[grasp_mask].mean() / W) - 1
        hy = 2 * (ys[grasp_mask].mean() / H) - 1
        bx = 2 * (xs[cut_mask].mean() / W) - 1
        by = 2 * (ys[cut_mask].mean() / H) - 1
        ox = bx - hx
        oy = by - hy
        norm = np.sqrt(ox * ox + oy * oy) + 1e-9
        ox /= norm
        oy /= norm
        rgbs.append(rgb)
        handles.append([hx, hy])
        blades.append([bx, by])
        orients.append([ox, oy])
    rgbs = np.stack(rgbs).astype(np.uint8)
    handles = np.array(handles, dtype=np.float32)
    blades = np.array(blades, dtype=np.float32)
    orients = np.array(orients, dtype=np.float32)
    return rgbs, handles, blades, orients


def angle_error_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred_n = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-9)
    true_n = true / (np.linalg.norm(true, axis=-1, keepdims=True) + 1e-9)
    cos = np.clip((pred_n * true_n).sum(axis=-1), -1, 1)
    return np.degrees(np.arccos(cos))


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("h8_cut")

    taxonomy = ROOT / "configs/affordance_taxonomy.yaml"
    train_subset = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/train.json", taxonomy, image_size=224)
    val_subset = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/val.json", taxonomy, image_size=224)
    test_subset = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/test.json", taxonomy, image_size=224)

    train_cut = select_cut_objects(train_subset)
    val_cut = select_cut_objects(val_subset)
    test_cut = select_cut_objects(test_subset)
    log.info("cut-affordance objects: train=%d val=%d test=%d", len(train_cut), len(val_cut), len(test_cut))

    mapping = train_subset.mapping
    train_rgbs, train_h, train_b, train_o = compute_targets(train_cut, mapping)
    val_rgbs, val_h, val_b, val_o = compute_targets(val_cut, mapping)
    test_rgbs, test_h, test_b, test_o = compute_targets(test_cut, mapping)
    log.info("after target filter: train=%d val=%d test=%d",
             len(train_rgbs), len(val_rgbs), len(test_rgbs))

    rgbs_all = np.concatenate([train_rgbs, val_rgbs, test_rgbs])
    h_all = np.concatenate([train_h, val_h, test_h])
    b_all = np.concatenate([train_b, val_b, test_b])
    o_all = np.concatenate([train_o, val_o, test_o])

    n_train = len(train_rgbs)
    n_val = len(val_rgbs)
    n_test = len(test_rgbs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoders = ["dinov2_base", "dinov2_large", "standalone_siglip",
                "pi0_siglip", "pi05_siglip", "openvla_siglip",
                "pi0_adapter", "pi05_adapter"]

    def featurize(name, rgbs):
        if name == "dinov2_base":
            return featurize_dinov2(rgbs, 224, base=True, device=device)
        if name == "dinov2_large":
            return featurize_dinov2(rgbs, 224, base=False, device=device)
        if name == "standalone_siglip":
            model, proc = build_standalone(device)
            f = featurize_hf_siglip(rgbs, model, proc, 224, device)
            del model
            torch.cuda.empty_cache()
            return f
        if name == "pi0_siglip":
            p = build_pi0_full(use_pi05=False, device=device)
            f = featurize_hf_siglip(rgbs, p._backbone, p._processor, 224, device)
            del p
            torch.cuda.empty_cache()
            return f
        if name == "pi05_siglip":
            p = build_pi0_full(use_pi05=True, device=device)
            f = featurize_hf_siglip(rgbs, p._backbone, p._processor, 224, device)
            del p
            torch.cuda.empty_cache()
            return f
        if name == "openvla_siglip":
            p = build_openvla_full(device)
            f = featurize_timm_siglip(rgbs, p._backbone, p._processor, 224, device)
            del p
            torch.cuda.empty_cache()
            return f
        if name == "pi0_adapter":
            return featurize_adapter_pi0(rgbs, 224, device, "pi0")
        if name == "pi05_adapter":
            return featurize_adapter_pi0(rgbs, 224, device, "pi05")
        raise ValueError(name)

    results = {}
    for name in encoders:
        t0 = time.time()
        log.info(">>> %s featurize ...", name)
        try:
            X = featurize(name, rgbs_all)
        except Exception as e:
            log.exception("FAILED %s: %s", name, e)
            continue
        log.info("    feats=%s in %.1fs", X.shape, time.time() - t0)

        # Standardize.
        Xtr = X[:n_train]
        mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        Xz = (X - mu) / sigma
        Xtrz, Xvz, Xtez = Xz[:n_train], Xz[n_train:n_train + n_val], Xz[n_train + n_val:]

        record = {}
        # Per-target Ridge eval.
        for target_name, ytr, yv, yt in [
            ("handle", train_h, val_h, test_h),
            ("blade", train_b, val_b, test_b),
            ("orient", train_o, val_o, test_o),
        ]:
            r_val = ridge_eval(Xtrz, ytr, Xvz, yv, alpha=args.alpha)
            r_test = ridge_eval(Xtrz, ytr, Xtez, yt, alpha=args.alpha)
            record[target_name] = dict(val=r_val, test=r_test)

        # Orientation: also angle error in degrees.
        from sklearn.linear_model import Ridge

        clf_o = Ridge(alpha=args.alpha)
        clf_o.fit(Xtrz, train_o)
        for split_name, X_, y_ in [("val", Xvz, val_o), ("test", Xtez, test_o)]:
            yp = clf_o.predict(X_)
            ang = angle_error_deg(yp, y_)
            record[f"orient_{split_name}_angle_deg_mean"] = float(ang.mean())
            record[f"orient_{split_name}_angle_deg_median"] = float(np.median(ang))

        record["feat_dim"] = int(X.shape[1])
        record["n_train"] = int(n_train)
        record["n_val"] = int(n_val)
        record["n_test"] = int(n_test)
        results[name] = record
        log.info("    %s orient_angle val=%.1f° test=%.1f°  handle_test_L2=%.3f",
                 name, record["orient_val_angle_deg_mean"],
                 record["orient_test_angle_deg_mean"],
                 record["handle"]["test"]["mean_l2"])

    with open(out_dir / "umd_cut_action_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out_dir / "umd_cut_action_results.json")

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["dinov2_base", "dinov2_large", "standalone_siglip",
             "pi0_siglip", "pi0_adapter", "pi05_siglip", "pi05_adapter",
             "openvla_siglip"]
    names = [n for n in order if n in results]
    angle_test = [results[n]["orient_test_angle_deg_mean"] for n in names]
    handle_test = [results[n]["handle"]["test"]["mean_l2"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    colors = ["#3a8a4f" if "dinov2" in n else
              "#4a72c9" if "standalone" in n else
              "#c94a72" if n == "pi0_siglip" else
              "#d3a45f" if n == "pi0_adapter" else
              "#aa6c39" if n == "pi05_siglip" else
              "#7a4d27" if n == "pi05_adapter" else
              "#3a8a4f" for n in names]
    for ax, vals, ylab, title in [
        (ax1, angle_test, "blade→handle angle error (degrees)",
         "Cut-task: knife orientation prediction (test)"),
        (ax2, handle_test, "handle-centroid L2 error",
         "Cut-task: handle position prediction (test)"),
    ]:
        b = ax.bar(np.arange(len(names)), vals, color=colors)
        for bar_, v in zip(b, vals):
            ax.text(bar_.get_x() + bar_.get_width() / 2, v + 0.005 * max(vals), f"{v:.2f}",
                    ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "umd_cut_action_l2.png")
    log.info("wrote %s", out_dir / "umd_cut_action_l2.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/h8-action-proxy/results")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    main(args)
