"""H10b — Multi-tool composite scenes.

Generates 2-tool composite test images by spatially tiling pairs of UMD
test objects: a cut-affordance object (knife/shears/scissors/saw) on the
LEFT and a contain-affordance object (mug/bowl/cup) on the RIGHT, each
resized to 112×112 and assembled into a 224×224 canvas. Labels are tiled
identically.

Probes are trained ONLY on the original single-tool UMD train set and
then evaluated on (a) the original single-tool UMD test set, and
(b) the composite multi-tool test set. The gap between (a) and (b)
quantifies how much the cut signal degrades when an alternative-class
distractor is present in the same scene.

Why this matters: the H2 metric is multi-class IoU. H9 showed all
encoders are perfect at single-object binary discrimination. H10a tests
whether the multi-class confusion alone produces a gap (still on
single-object scenes). H10b tests whether *visual co-presence* of
distractor classes produces a gap.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.dataset_umd import UMDSubset

_spec = _ilu.spec_from_file_location(
    "h9_kpd",
    str(ROOT / "experiments" / "h9-handle-blade" / "knife_part_discrimination.py"),
)
_h9 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_h9)
featurize_for_encoder = _h9.featurize_for_encoder
pool_label_to_patches = _h9.pool_label_to_patches

CUT_CATS = {"knife", "shears", "scissors", "saw"}
CONTAIN_CATS = {"mug", "cup", "bowl", "pot"}
GRASP_CATS = {"hammer", "ladle"}

BG, GRASP, CUT, SCOOP, CONTAIN, SUPPORT = 0, 1, 2, 3, 4, 5


def crop_to_foreground(rgb, lbl, pad=10):
    """Tighten an RGB+label pair to the bounding box of foreground pixels."""
    fg = lbl > 0
    if fg.sum() == 0:
        return rgb, lbl
    ys, xs = np.where(fg)
    y0, y1 = max(0, ys.min() - pad), min(lbl.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(lbl.shape[1], xs.max() + pad + 1)
    return rgb[y0:y1, x0:x1], lbl[y0:y1, x0:x1]


def make_composite(rgb_a, lbl_a, rgb_b, lbl_b, layout="horizontal"):
    """Crop each tool tightly, scale into a 112-pixel half-canvas, place
    side-by-side. The crop step ensures each tool has substantial foreground
    coverage in its half (otherwise UMD's small foreground regions get
    quartered when naively resizing the full image)."""
    from PIL import Image

    rgb_a_c, lbl_a_c = crop_to_foreground(rgb_a, lbl_a, pad=8)
    rgb_b_c, lbl_b_c = crop_to_foreground(rgb_b, lbl_b, pad=8)

    def fit_to_square(rgb, lbl, target=112):
        H, W = lbl.shape
        s = max(H, W)
        # Pad to square with background = 0.
        pad_y = s - H
        pad_x = s - W
        rgb_p = np.zeros((s, s, 3), dtype=np.uint8)
        lbl_p = np.zeros((s, s), dtype=np.uint8)
        oy, ox = pad_y // 2, pad_x // 2
        rgb_p[oy:oy + H, ox:ox + W] = rgb
        lbl_p[oy:oy + H, ox:ox + W] = lbl
        rgb_q = np.asarray(Image.fromarray(rgb_p).resize((target, target), Image.BILINEAR))
        lbl_q = np.asarray(Image.fromarray(lbl_p, mode="L").resize((target, target), Image.NEAREST))
        return rgb_q, lbl_q

    rgb_a_s, lbl_a_s = fit_to_square(rgb_a_c, lbl_a_c, 112)
    rgb_b_s, lbl_b_s = fit_to_square(rgb_b_c, lbl_b_c, 112)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    lbl = np.zeros((224, 224), dtype=np.uint8)
    if layout == "horizontal":
        rgb[56:56 + 112, 0:112] = rgb_a_s
        rgb[56:56 + 112, 112:224] = rgb_b_s
        lbl[56:56 + 112, 0:112] = lbl_a_s
        lbl[56:56 + 112, 112:224] = lbl_b_s
    else:
        rgb[0:112, 56:56 + 112] = rgb_a_s
        rgb[112:224, 56:56 + 112] = rgb_b_s
        lbl[0:112, 56:56 + 112] = lbl_a_s
        lbl[112:224, 56:56 + 112] = lbl_b_s
    return rgb, lbl


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("h10b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 224
    patch_size = 14
    gh = image_size // patch_size

    taxonomy = ROOT / "configs/affordance_taxonomy.yaml"
    train = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/train.json", taxonomy, image_size=image_size)
    test = UMDSubset.from_split_file(ROOT / "data/umd/splits_500/test.json", taxonomy, image_size=image_size)
    mapping = train.mapping

    def gather(samples):
        rgbs, labels = [], []
        for s in samples:
            rgbs.append(s.load_rgb(size=image_size))
            labels.append(s.load_label(mapping, size=image_size))
        return np.stack(rgbs).astype(np.uint8), np.stack(labels).astype(np.uint8)

    rgb_tr, lbl_tr = gather(train.samples)
    rgb_te, lbl_te = gather(test.samples)
    log.info("loaded UMD train=%d test=%d", len(rgb_tr), len(rgb_te))

    # Build composites: for each cut-test object, pair with a random non-cut test object.
    rng = random.Random(0)
    cut_test_idx = [i for i, s in enumerate(test.samples) if s.category in CUT_CATS]
    distractor_test_idx = [i for i, s in enumerate(test.samples)
                           if s.category in (CONTAIN_CATS | GRASP_CATS | {"spoon", "shovel"})]
    rng.shuffle(distractor_test_idx)
    log.info("cut test objects: %d, distractor test objects: %d", len(cut_test_idx), len(distractor_test_idx))
    if len(distractor_test_idx) == 0:
        log.error("no distractor objects found! falling back to other cut objects")
        distractor_test_idx = cut_test_idx[::-1]

    composites_rgb, composites_lbl = [], []
    for i, ci in enumerate(cut_test_idx):
        di = distractor_test_idx[i % len(distractor_test_idx)]
        layout = "horizontal" if i % 2 == 0 else "vertical"
        cmp_rgb, cmp_lbl = make_composite(rgb_te[ci], lbl_te[ci],
                                           rgb_te[di], lbl_te[di], layout)
        composites_rgb.append(cmp_rgb)
        composites_lbl.append(cmp_lbl)
    composites_rgb = np.stack(composites_rgb)
    composites_lbl = np.stack(composites_lbl)
    log.info("built %d composites", composites_rgb.shape[0])

    # Patch labels.
    def patch_labels(labels):
        out = np.zeros((labels.shape[0], gh * gh), dtype=np.int64)
        for i in range(labels.shape[0]):
            out[i] = pool_label_to_patches(labels[i], image_size, patch_size)
        return out

    plbl_tr = patch_labels(lbl_tr)
    plbl_te = patch_labels(lbl_te)
    plbl_cmp = patch_labels(composites_lbl)

    def fg_mask(plbls):
        return plbls > 0

    mask_tr = fg_mask(plbl_tr)
    mask_te = fg_mask(plbl_te)
    mask_cmp = fg_mask(plbl_cmp)
    log.info("foreground patches: train=%d single_test=%d composite=%d (cut frac cmp=%.3f)",
             mask_tr.sum(), mask_te.sum(), mask_cmp.sum(),
             (plbl_cmp[mask_cmp] == CUT).mean())

    encoders = ["dinov2_base", "dinov2_large", "standalone_siglip",
                "pi0_siglip", "pi05_siglip", "openvla_siglip",
                "pi0_adapter", "pi05_adapter"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    pred_grids_cmp = {}

    # Save composite RGBs/labels for the qualitative panel.
    np.savez(out_dir / "h10b_composites.npz",
             rgbs=composites_rgb, labels=composites_lbl)

    for name in encoders:
        log.info(">>> %s ...", name)
        t0 = time.time()
        try:
            f_tr = featurize_for_encoder(name, rgb_tr, image_size, device)
            f_te = featurize_for_encoder(name, rgb_te, image_size, device)
            f_cmp = featurize_for_encoder(name, composites_rgb, image_size, device)
        except Exception as e:
            log.exception("FAILED %s: %s", name, e)
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        D = f_tr.shape[-1]
        log.info("    feats train=%s test=%s cmp=%s (D=%d) in %.1fs",
                 f_tr.shape, f_te.shape, f_cmp.shape, D, time.time() - t0)

        X_tr = f_tr.reshape(-1, D)[mask_tr.reshape(-1)]
        y_tr = (plbl_tr.reshape(-1)[mask_tr.reshape(-1)] == CUT).astype(np.int64)
        X_te = f_te.reshape(-1, D)[mask_te.reshape(-1)]
        y_te = (plbl_te.reshape(-1)[mask_te.reshape(-1)] == CUT).astype(np.int64)
        X_cmp = f_cmp.reshape(-1, D)[mask_cmp.reshape(-1)]
        y_cmp = (plbl_cmp.reshape(-1)[mask_cmp.reshape(-1)] == CUT).astype(np.int64)

        mu, sigma = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-6
        X_tr = (X_tr - mu) / sigma
        X_te = (X_te - mu) / sigma
        X_cmp = (X_cmp - mu) / sigma

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

        clf = LogisticRegression(C=1.0, max_iter=3000, n_jobs=-1, class_weight="balanced")
        clf.fit(X_tr, y_tr)

        def metrics(X, y):
            pred = clf.predict(X)
            prob = clf.predict_proba(X)[:, 1]
            return dict(
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                f1_cut=float(f1_score(y, pred, pos_label=1)),
                auc=float(roc_auc_score(y, prob)),
            )

        record = dict(
            feat_dim=int(D),
            single_test=metrics(X_te, y_te),
            composite_test=metrics(X_cmp, y_cmp),
        )
        results[name] = record
        log.info("    %s single_bal=%.3f cmp_bal=%.3f  Δ=%+.3f  single_auc=%.3f cmp_auc=%.3f",
                 name,
                 record["single_test"]["balanced_accuracy"],
                 record["composite_test"]["balanced_accuracy"],
                 record["composite_test"]["balanced_accuracy"] -
                 record["single_test"]["balanced_accuracy"],
                 record["single_test"]["auc"],
                 record["composite_test"]["auc"])

        # Save composite prob grids.
        flat = f_cmp.reshape(-1, D)
        flat_z = (flat - mu) / sigma
        prob_full = clf.predict_proba(flat_z)[:, 1].reshape(f_cmp.shape[0], gh, gh)
        pred_grids_cmp[name] = prob_full.astype(np.float32)

    with open(out_dir / "h10b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out_dir / "h10b_results.json")
    np.savez(out_dir / "h10b_cmp_pred_grids.npz",
             rgbs=composites_rgb, labels=composites_lbl, **pred_grids_cmp)
    log.info("wrote %s", out_dir / "h10b_cmp_pred_grids.npz")

    # === Plot single vs composite gap ===
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["dinov2_base", "dinov2_large", "standalone_siglip",
             "pi0_siglip", "pi0_adapter", "pi05_siglip", "pi05_adapter",
             "openvla_siglip"]
    names = [n for n in order if n in results]
    single_bal = [results[n]["single_test"]["balanced_accuracy"] for n in names]
    cmp_bal = [results[n]["composite_test"]["balanced_accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    xs = np.arange(len(names))
    w = 0.4
    ax.bar(xs - w / 2, single_bal, w, label="single-tool test", color="#4a72c9")
    ax.bar(xs + w / 2, cmp_bal, w, label="multi-tool composite test", color="#c94a72")
    for x, sv, cv in zip(xs, single_bal, cmp_bal):
        ax.text(x - w / 2, sv + 0.005, f"{sv:.3f}", ha="center", fontsize=7)
        ax.text(x + w / 2, cv + 0.005, f"{cv:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("balanced accuracy (cut vs other foreground)")
    ax.set_title("H10b — does multi-tool clutter expose VLA encoder degradation?\n"
                 "single-tool test → multi-tool composite test, all encoders trained on single-tool UMD")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "h10b_single_vs_composite.png")
    log.info("wrote %s", out_dir / "h10b_single_vs_composite.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/h10-multitool/results")
    args = ap.parse_args()
    main(args)
