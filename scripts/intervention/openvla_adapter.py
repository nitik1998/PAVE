"""Run the same MLP adapter recovery on OpenVLA's SigLIP tower.

If the asymmetric loss is a rotation (not a deletion) for OpenVLA too, the
same 297K-parameter adapter should recover most of the lost cut signal.
This generalizes the recovery claim across VLA families.
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
from src.methods.openvla_siglip_probe import OpenVLASigLIPConfig, OpenVLASigLIPProbe

CLASSES = ["bg", "grasp", "cut", "scoop", "contain", "support"]


@torch.no_grad()
def extract_patch_features(probe: OpenVLASigLIPProbe, rgb: np.ndarray) -> torch.Tensor:
    from PIL import Image

    S = int(probe.cfg.image_size)
    pil = Image.fromarray(rgb).resize((S, S), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.asarray(getattr(probe._processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(getattr(probe._processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    arr = (arr - mean) / std
    pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(probe.cfg.device).float()
    feats = probe._backbone.forward_features(pix)[0]
    gh = S // probe.cfg.patch_size
    if feats.shape[0] > gh * gh:
        feats = feats[-gh * gh:]
    return feats


def pool_label(label, image_size, patch_size):
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
    def __init__(self, in_dim, num_classes, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_adapter(adapter, probe, subset, num_classes):
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
    return dict(miou=float(np.nanmean(iou)),
                per_class={CLASSES[i]: float(iou[i]) for i in range(num_classes)})


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("openvla_adapter")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OpenVLASigLIPConfig(device=device)
    probe = OpenVLASigLIPProbe(cfg=cfg, num_classes=len(CLASSES), foreground_names=CLASSES[1:])
    probe.warmup()

    train_split = ROOT / "data/umd/splits_500/train.json"
    val_split = ROOT / "data/umd/splits_500/val.json"
    test_split = ROOT / "data/umd/splits_500/test.json"
    taxonomy = ROOT / "configs/affordance_taxonomy.yaml"

    sub_train = UMDSubset.from_split_file(train_split, taxonomy, image_size=probe.cfg.image_size)

    log.info("pre-extracting train features ...")
    Xs, ys = [], []
    t0 = time.time()
    for i, (s, rgb, lbl) in enumerate(sub_train):
        feats = extract_patch_features(probe, rgb).cpu().numpy()
        labs = pool_label(lbl, probe.cfg.image_size, probe.cfg.patch_size)
        Xs.append(feats)
        ys.append(labs)
        if i % 50 == 0:
            log.info("[train] %d/%d (%.1fs)", i, len(sub_train), time.time() - t0)
    X_train = np.concatenate(Xs, 0).astype(np.float32)
    y_train = np.concatenate(ys, 0).astype(np.int64)

    num_classes = len(CLASSES)
    in_dim = X_train.shape[1]
    log.info("X_train=%s y_train=%s", X_train.shape, y_train.shape)

    adapter = MLPAdapter(in_dim, num_classes, hidden=args.hidden, dropout=0.1).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    counts = np.bincount(y_train, minlength=num_classes)
    cw = (counts.sum() / num_classes) / np.maximum(counts, 1)
    cw_t = torch.from_numpy(cw.astype(np.float32)).to(device)

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
            epoch_loss += float(loss.detach()) * xb.shape[0]
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info("epoch %d/%d loss=%.4f", epoch + 1, args.epochs, epoch_loss / n)

    sub_val = UMDSubset.from_split_file(val_split, taxonomy, image_size=probe.cfg.image_size)
    sub_test = UMDSubset.from_split_file(test_split, taxonomy, image_size=probe.cfg.image_size)
    val_res = evaluate_adapter(adapter, probe, sub_val, num_classes)
    log.info("val mIoU=%.3f per_class=%s", val_res["miou"], val_res["per_class"])
    test_res = evaluate_adapter(adapter, probe, sub_test, num_classes)
    log.info("test mIoU=%.3f per_class=%s", test_res["miou"], test_res["per_class"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        backbone="openvla_siglip", adapter_hidden=args.hidden, epochs=args.epochs,
        adapter_params=int(sum(p.numel() for p in adapter.parameters())),
        val=val_res, test=test_res,
    )
    out_json = out_dir / f"adapter_openvla_siglip_h{args.hidden}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/intervention")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    main(args)
