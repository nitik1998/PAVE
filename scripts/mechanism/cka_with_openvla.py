"""Layer-wise CKA + per-class drift, extended to OpenVLA.

Adds OpenVLA's timm SigLIP-So400m to the existing analysis. Uses timm's
``forward_intermediates`` to get per-block hidden states.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.dataset_umd import UMDSubset
from src.methods.openvla_siglip_probe import OpenVLASigLIPConfig, OpenVLASigLIPProbe
from src.methods.pi0_siglip_probe import Pi0SigLIPConfig, Pi0SigLIPProbe

CLASSES = ["bg", "grasp", "cut", "scoop", "contain", "support"]
FOREGROUND = CLASSES[1:]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xtx = X.T @ X
    yty = Y.T @ Y
    xty = X.T @ Y
    num = float((xty * xty).sum())
    den = float(np.sqrt((xtx * xtx).sum()) * np.sqrt((yty * yty).sum()) + 1e-12)
    return num / den


@torch.no_grad()
def hf_hidden_states(model, pix: torch.Tensor) -> list[np.ndarray]:
    out = model(pix, output_hidden_states=True)
    hidden = out.hidden_states
    arrs = []
    n_patches = hidden[-1].shape[1]
    for h in hidden:
        if h.shape[1] != n_patches:
            h = h[:, -n_patches:, :]
        arrs.append(h[0].float().cpu().numpy())
    return arrs


@torch.no_grad()
def timm_hidden_states(model, pix: torch.Tensor) -> list[np.ndarray]:
    """Returns list of (n_patches, d) numpy arrays per block, plus the input
    patch embedding. We use forward_intermediates which returns
    (intermediates_list, final).

    For ViT-SigLIP timm models, forward_intermediates(x, indices=range(L)) gives
    one intermediate per block. We also compute the patch embedding as layer 0.
    """
    L = len(model.blocks)
    inters = model.forward_intermediates(pix, indices=list(range(L)),
                                         output_fmt="NLC", intermediates_only=True)
    # `inters` is a list length L of (B, N, D) tensors.
    # Compose layer 0 = post-patch-embedding pre-blocks
    pe = model.patch_embed(pix)
    if hasattr(model, "_pos_embed"):
        pe = model._pos_embed(pe)
    arrs = [pe[0].float().cpu().numpy()]
    for x in inters:
        arrs.append(x[0].float().cpu().numpy())
    return arrs


def pool_label_to_patches(label, image_size, patch_size):
    gh = image_size // patch_size
    out = np.zeros(gh * gh, dtype=np.int64)
    ps = patch_size
    for i in range(gh):
        for j in range(gh):
            tile = label[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps].ravel()
            vals, counts = np.unique(tile, return_counts=True)
            out[i * gh + j] = int(vals[counts.argmax()])
    return out


def featurize_hf(probe: Pi0SigLIPProbe, rgb: np.ndarray) -> list[np.ndarray]:
    from PIL import Image

    S = int(probe.cfg.image_size)
    pil = Image.fromarray(rgb).resize((S, S), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.asarray(getattr(probe._processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(getattr(probe._processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    arr = (arr - mean) / std
    pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(probe.cfg.device).float()
    return hf_hidden_states(probe._backbone, pix)


def featurize_timm(probe: OpenVLASigLIPProbe, rgb: np.ndarray) -> list[np.ndarray]:
    from PIL import Image

    S = int(probe.cfg.image_size)
    pil = Image.fromarray(rgb).resize((S, S), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.asarray(getattr(probe._processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(getattr(probe._processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    arr = (arr - mean) / std
    pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(probe.cfg.device).float()
    return timm_hidden_states(probe._backbone, pix)


def build_standalone_hf(device):
    from transformers import AutoImageProcessor, AutoModel

    full = AutoModel.from_pretrained("google/siglip-so400m-patch14-224")
    cfg = Pi0SigLIPConfig(device=device)
    p = Pi0SigLIPProbe(cfg=cfg, num_classes=len(CLASSES), foreground_names=FOREGROUND)
    p._processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-224")
    p._backbone = full.vision_model.eval().to(device)
    for x in p._backbone.parameters():
        x.requires_grad_(False)
    p.cfg.patch_size = 14
    p.cfg.image_size = 224
    p.name = "standalone"
    return p


def build_pi0_hf(device, use_pi05=False):
    cfg = Pi0SigLIPConfig(device=device, use_pi05=use_pi05)
    p = Pi0SigLIPProbe(cfg=cfg, num_classes=len(CLASSES), foreground_names=FOREGROUND)
    p.warmup()
    return p


def build_openvla_timm(device):
    cfg = OpenVLASigLIPConfig(device=device)
    p = OpenVLASigLIPProbe(cfg=cfg, num_classes=len(CLASSES), foreground_names=FOREGROUND)
    p.warmup()
    return p


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("cka_v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_split = ROOT / "data/umd/splits_500/val.json"
    taxonomy = ROOT / "configs/affordance_taxonomy.yaml"
    sub = UMDSubset.from_split_file(val_split, taxonomy, image_size=224)
    log.info("val n=%d", len(sub))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    for tag, build, featurize in [
        ("standalone", build_standalone_hf, featurize_hf),
        ("pi0", lambda d: build_pi0_hf(d), featurize_hf),
        ("pi05", lambda d: build_pi0_hf(d, use_pi05=True), featurize_hf),
        ("openvla", build_openvla_timm, featurize_timm),
    ]:
        log.info("=== %s ===", tag)
        probe = build(device)
        all_layers = None
        all_labels = []
        for idx, (s, rgb, lbl) in enumerate(sub):
            layers = featurize(probe, rgb)
            ys = pool_label_to_patches(lbl, 224, 14)
            if all_layers is None:
                all_layers = [[] for _ in layers]
            for li, l in enumerate(layers):
                all_layers[li].append(l)
            all_labels.append(ys)
            if idx % 20 == 0:
                log.info("[%s] %d/%d", tag, idx, len(sub))
        cache[tag] = dict(
            layers=[np.concatenate(L, axis=0) for L in all_layers],
            labels=np.concatenate(all_labels),
        )
        log.info("[%s] cached %d layers, %d patches",
                 tag, len(cache[tag]["layers"]), cache[tag]["labels"].shape[0])
        del probe
        torch.cuda.empty_cache()

    n_layers_min = min(len(cache[t]["layers"]) for t in cache)
    log.info("computing CKA over min layers=%d ...", n_layers_min)
    cka = {t: [] for t in ("pi0", "pi05", "openvla")}
    for li in range(n_layers_min):
        Xs = cache["standalone"]["layers"][li]
        for t in ("pi0", "pi05", "openvla"):
            cka[t].append(linear_cka(Xs, cache[t]["layers"][li]))
        log.info("layer %2d  pi0=%.3f  pi05=%.3f  openvla=%.3f",
                 li, cka["pi0"][-1], cka["pi05"][-1], cka["openvla"][-1])

    np.savez(out_dir / "cka_layers_all.npz",
             pi0=np.array(cka["pi0"]),
             pi05=np.array(cka["pi05"]),
             openvla=np.array(cka["openvla"]))

    # Per-class drift, final layer.
    final = n_layers_min - 1
    Xs = cache["standalone"]["layers"][final]
    labels = cache["standalone"]["labels"]
    per_class = {}
    for cls_id in range(len(CLASSES)):
        mask = labels == cls_id
        if mask.sum() < 5:
            continue
        ms = Xs[mask].mean(axis=0)
        out_row = dict(n_patches=int(mask.sum()))
        for t in ("pi0", "pi05", "openvla"):
            mp = cache[t]["layers"][final][mask].mean(axis=0)
            cs = float(np.dot(ms, mp) / (np.linalg.norm(ms) * np.linalg.norm(mp) + 1e-12))
            out_row[f"cos_standalone_{t}"] = cs
        per_class[CLASSES[cls_id]] = out_row
        log.info("class=%s n=%d cos: pi0=%.3f pi05=%.3f openvla=%.3f",
                 CLASSES[cls_id], int(mask.sum()),
                 out_row["cos_standalone_pi0"],
                 out_row["cos_standalone_pi05"],
                 out_row["cos_standalone_openvla"])
    with open(out_dir / "per_class_drift_all.json", "w") as f:
        json.dump(per_class, f, indent=2)

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=140)
    xs = np.arange(n_layers_min)
    ax1.plot(xs, cka["pi0"], marker="o", label="standalone vs π0", color="#c94a72")
    ax1.plot(xs, cka["pi05"], marker="s", label="standalone vs π0.5", color="#aa6c39")
    ax1.plot(xs, cka["openvla"], marker="^", label="standalone vs OpenVLA", color="#3a8a4f")
    ax1.set_xlabel("transformer layer")
    ax1.set_ylabel("linear CKA")
    ax1.set_title("Layer-wise divergence from standalone SigLIP-So400m\n(across three VLA families)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend()

    cls_names_ord = [c for c in FOREGROUND if c in per_class]
    cos_pi0 = [per_class[c]["cos_standalone_pi0"] for c in cls_names_ord]
    cos_pi05 = [per_class[c]["cos_standalone_pi05"] for c in cls_names_ord]
    cos_openvla = [per_class[c]["cos_standalone_openvla"] for c in cls_names_ord]
    w = 0.27
    xc = np.arange(len(cls_names_ord))
    ax2.bar(xc - w, cos_pi0, w, label="π0", color="#c94a72")
    ax2.bar(xc, cos_pi05, w, label="π0.5", color="#aa6c39")
    ax2.bar(xc + w, cos_openvla, w, label="OpenVLA", color="#3a8a4f")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(cls_names_ord)
    ax2.set_ylabel("cosine similarity to standalone class-mean (final layer)")
    ax2.set_title("Per-class final-layer drift across 3 VLA families")
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "cka_and_drift_all.png")
    log.info("wrote %s", out_dir / "cka_and_drift_all.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/mechanism")
    args = ap.parse_args()
    main(args)
