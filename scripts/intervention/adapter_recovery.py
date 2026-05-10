"""Tiny adapter on top of frozen π0 SigLIP recovers lost cut-class affordance.

Question: H2 showed π0 SigLIP loses 27 pp IoU on cut compared to standalone
SigLIP-So400m. Is this loss recoverable from the residual signal (i.e., the
information is still there but mis-rotated), or is it actually deleted by the
fine-tuning? A cheap test: freeze π0 SigLIP and train a 2-layer MLP adapter
(≈ 100k params) over its patch features against UMD's per-pixel labels.

If the adapter recovers most of the cut-class IoU back to standalone level,
that says the loss is a *rotation*, not a *deletion* — a much more useful
finding for practitioners (you can recover affordance from a fine-tuned VLA
without retraining the encoder; you just need a small adapter).

Design:
  - Backbone: frozen π0 SigLIP (1152-d patch features at 16x16 grid for 224 input).
  - Head: 2-layer MLP (1152 → 256 → 6). Trained with cross-entropy.
  - Compared against:
      * Linear probe (matching the H2 "0.519" result baseline).
      * Linear probe on standalone SigLIP-So400m (the "ceiling" 0.610).
      * Linear probe on DINOv2-large @ 224 (an alternative architecture-of-the-same-budget
        ceiling).
  - Train on UMD train split (n=345); eval on val + test.
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
from src.eval.metrics import compute_metrics
from src.methods.pi0_siglip_probe import Pi0SigLIPConfig, Pi0SigLIPProbe

CLASSES = ["bg", "grasp", "cut", "scoop", "contain", "support"]
FOREGROUND = CLASSES[1:]


def build_pi0_probe(use_pi05: bool, device: str) -> Pi0SigLIPProbe:
    cfg = Pi0SigLIPConfig(device=device, use_pi05=use_pi05)
    p = Pi0SigLIPProbe(cfg=cfg, num_classes=len(CLASSES), foreground_names=FOREGROUND)
    p.warmup()
    if use_pi05:
        p.name = "pi05_siglip"
    return p


@torch.no_grad()
def extract_patch_features(probe: Pi0SigLIPProbe, rgb: np.ndarray) -> torch.Tensor:
    """Returns (gh*gh, D) torch tensor on the same device as the backbone."""
    from PIL import Image

    S = int(probe.cfg.image_size)
    pil = Image.fromarray(rgb).resize((S, S), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.asarray(getattr(probe._processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(getattr(probe._processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    arr = (arr - mean) / std
    pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(probe.cfg.device).float()
    out = probe._backbone(pix)
    feats = out.last_hidden_state[0]
    gh = S // probe.cfg.patch_size
    if feats.shape[0] > gh * gh:
        feats = feats[-gh * gh:]
    return feats


def pool_label_to_patches(label: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    gh = image_size // patch_size
    out = np.zeros(gh * gh, dtype=np.int64)
    ps = patch_size
    for i in range(gh):
        for j in range(gh):
            tile = label[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps].ravel()
            vals, counts = np.unique(tile, return_counts=True)
            out[i * gh + j] = int(vals[counts.argmax()])
    return out


class MLPAdapter(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_adapter(adapter, probe, subset: UMDSubset, num_classes: int) -> dict:
    """Compute per-class IoU using the trained adapter."""
    adapter.eval()
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    for s, rgb, lbl in subset:
        feats = extract_patch_features(probe, rgb)
        with torch.no_grad():
            logits = adapter(feats)
        gh = probe.cfg.image_size // probe.cfg.patch_size
        K = logits.shape[-1]
        grid = logits.reshape(gh, gh, K).permute(2, 0, 1).unsqueeze(0)
        full = F.interpolate(grid, size=probe.cfg.image_size, mode="bilinear", align_corners=False)
        pred_label = full.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        m = compute_metrics(pred_label, lbl, num_classes=num_classes)
        cm_total += m.confusion
    tp = np.diag(cm_total)
    fp = cm_total.sum(axis=0) - tp
    fn = cm_total.sum(axis=1) - tp
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    return dict(
        miou=float(np.nanmean(iou)),
        per_class={CLASSES[i]: float(iou[i]) for i in range(num_classes)},
    )


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("adapter")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)

    train_split = ROOT / "data/umd/splits_500/train.json"
    val_split = ROOT / "data/umd/splits_500/val.json"
    test_split = ROOT / "data/umd/splits_500/test.json"
    taxonomy = ROOT / "configs/affordance_taxonomy.yaml"

    log.info("loading π0 SigLIP backbone (use_pi05=%s) ...", args.use_pi05)
    probe = build_pi0_probe(use_pi05=args.use_pi05, device=device)
    in_dim = 1152
    num_classes = len(CLASSES)

    # === Pre-extract train features ===
    log.info("pre-extracting train features ...")
    sub_train = UMDSubset.from_split_file(train_split, taxonomy, image_size=probe.cfg.image_size)
    Xs, ys = [], []
    t0 = time.time()
    for i, (s, rgb, lbl) in enumerate(sub_train):
        feats = extract_patch_features(probe, rgb).cpu().numpy()
        labs = pool_label_to_patches(lbl, image_size=probe.cfg.image_size,
                                     patch_size=probe.cfg.patch_size)
        Xs.append(feats)
        ys.append(labs)
        if i % 50 == 0:
            log.info("[train] %d/%d (%.1fs)", i, len(sub_train), time.time() - t0)
    X_train = np.concatenate(Xs, axis=0).astype(np.float32)
    y_train = np.concatenate(ys, axis=0).astype(np.int64)
    log.info("X_train=%s y_train=%s class hist=%s", X_train.shape, y_train.shape,
             np.bincount(y_train, minlength=num_classes))

    # === Train MLP adapter ===
    adapter = MLPAdapter(in_dim, num_classes, hidden=args.hidden, dropout=0.1).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in adapter.parameters())
    log.info("adapter has %d parameters", n_params)

    # Class-balanced weights so cut/support don't get drowned by bg.
    counts = np.bincount(y_train, minlength=num_classes)
    cw = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        cw[c] = (1.0 / max(counts[c], 1)) * counts.sum() / num_classes
    log.info("class weights: %s", dict(zip(CLASSES, cw.round(3))))
    cw_t = torch.from_numpy(cw).to(device)

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    bs = 8192
    n = Xt.shape[0]

    for epoch in range(args.epochs):
        adapter.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            xb, yb = Xt[idx], yt[idx]
            logits = adapter(xb)
            loss = F.cross_entropy(logits, yb, weight=cw_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * xb.shape[0]
        log.info("epoch %d/%d loss=%.4f", epoch + 1, args.epochs, epoch_loss / n)

    # === Eval ===
    sub_val = UMDSubset.from_split_file(val_split, taxonomy, image_size=probe.cfg.image_size)
    sub_test = UMDSubset.from_split_file(test_split, taxonomy, image_size=probe.cfg.image_size)
    log.info("evaluating on val ...")
    val_res = evaluate_adapter(adapter, probe, sub_val, num_classes)
    log.info("val mIoU=%.3f per_class=%s", val_res["miou"], val_res["per_class"])
    log.info("evaluating on test ...")
    test_res = evaluate_adapter(adapter, probe, sub_test, num_classes)
    log.info("test mIoU=%.3f per_class=%s", test_res["miou"], test_res["per_class"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        backbone="pi05_siglip" if args.use_pi05 else "pi0_siglip",
        adapter_hidden=args.hidden,
        adapter_params=int(n_params),
        epochs=args.epochs,
        lr=args.lr,
        val=val_res,
        test=test_res,
    )
    out_json = out_dir / f"adapter_{summary['backbone']}_h{args.hidden}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote %s", out_json)
    torch.save(adapter.state_dict(), out_dir / f"adapter_{summary['backbone']}_h{args.hidden}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-pi05", action="store_true")
    ap.add_argument("--out", default="outputs/intervention")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    main(args)
