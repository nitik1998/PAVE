"""Layer-wise CKA + per-class feature drift between standalone SigLIP-So400m,
π0 SigLIP, and π0.5 SigLIP.

Why this experiment matters
---------------------------
H2 showed *that* π0's SigLIP loses 9 pp affordance mIoU vs a frozen
standalone SigLIP-So400m, with class-asymmetric pattern (cut −0.27,
contain −0.01). H5 showed π0.5 partially recovers. This script asks
the **mechanism** question:

  - At which transformer block does the divergence appear?
  - Is the divergence concentrated in patches whose ground-truth label
    is `cut`, vs spread uniformly across classes?

We measure:
  1. **Layer-wise linear CKA** between the layer-l hidden states of
     standalone SigLIP-So400m and π0-SigLIP, across all UMD val patches.
     Plotted as a CKA-vs-layer curve. Lower CKA = greater divergence.
  2. **Per-class cosine similarity** of class-mean patch features in the
     final layer. Per-class similarity should track per-class IoU drop.
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
from src.methods.pi0_siglip_probe import Pi0SigLIPConfig, Pi0SigLIPProbe

log = logging.getLogger("cka")
CLASSES = ["bg", "grasp", "cut", "scoop", "contain", "support"]
FOREGROUND = CLASSES[1:]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n, d) feature matrices.

    Uses the closed-form expression for centered features. This is
    Kornblith et al. 2019, Eq. (5).
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    xtx = X.T @ X
    yty = Y.T @ Y
    xty = X.T @ Y
    num = float((xty * xty).sum())
    den = float(np.sqrt((xtx * xtx).sum()) * np.sqrt((yty * yty).sum()) + 1e-12)
    return num / den


@torch.no_grad()
def forward_all_layers(model, pix: torch.Tensor) -> list[np.ndarray]:
    """Returns a list of (n_patches, d) numpy arrays, one per layer.

    SigLIP exposes intermediate states via ``output_hidden_states=True``.
    Layer 0 is the patch embedding pre-transformer; layers 1..L are post
    each transformer block. We strip the optional CLS token if present.
    """
    out = model(pix, output_hidden_states=True)
    hidden = out.hidden_states  # tuple length L+1
    arrs = []
    n_patches = hidden[-1].shape[1]
    for h in hidden:
        # SigLIP-So400m has no CLS — n_tokens = n_patches.
        if h.shape[1] != n_patches:
            h = h[:, -n_patches:, :]
        arrs.append(h[0].float().cpu().numpy())
    return arrs


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


def build_probe(use_pi05: bool, use_standalone: bool, device: str) -> Pi0SigLIPProbe:
    cfg = Pi0SigLIPConfig(device=device, use_pi05=use_pi05)
    probe = Pi0SigLIPProbe(cfg=cfg, num_classes=len(CLASSES), foreground_names=FOREGROUND)
    if use_standalone:
        # Override warmup to load standalone SigLIP-So400m at 224×224
        # (same skeleton, no checkpoint surgery).
        from transformers import AutoImageProcessor, AutoModel

        full = AutoModel.from_pretrained("google/siglip-so400m-patch14-224")
        probe._processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-224")
        probe._backbone = full.vision_model.eval().to(device)
        for p in probe._backbone.parameters():
            p.requires_grad_(False)
        probe.cfg.patch_size = 14
        probe.cfg.image_size = 224
        probe.name = "standalone_siglip"
    else:
        probe.warmup()
        if use_pi05:
            probe.name = "pi05_siglip"
    return probe


def featurize_image(probe: Pi0SigLIPProbe, rgb: np.ndarray) -> list[np.ndarray]:
    import torch
    from PIL import Image

    S = int(probe.cfg.image_size)
    pil = Image.fromarray(rgb).resize((S, S), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.asarray(getattr(probe._processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(getattr(probe._processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    arr = (arr - mean) / std
    pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(probe.cfg.device).float()
    return forward_all_layers(probe._backbone, pix)


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)

    val_split = ROOT / "data/umd/splits_500/val.json"
    taxonomy = ROOT / "configs/affordance_taxonomy.yaml"
    sub = UMDSubset.from_split_file(val_split, taxonomy, image_size=224)
    log.info("UMD val: %d samples", len(sub))

    # Build the three models. Loading them sequentially keeps VRAM usage
    # small — each finishes before the next starts.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache features per model to avoid re-loading.
    cache: dict[str, dict] = {}
    for tag, kwargs in [
        ("standalone", dict(use_pi05=False, use_standalone=True)),
        ("pi0", dict(use_pi05=False, use_standalone=False)),
        ("pi05", dict(use_pi05=True, use_standalone=False)),
    ]:
        log.info("=== %s ===", tag)
        probe = build_probe(device=device, **kwargs)
        all_layer_feats: list[list[np.ndarray]] = None
        all_labels = []
        for idx, (s, rgb, lbl) in enumerate(sub):
            layers = featurize_image(probe, rgb)
            ys = pool_label_to_patches(lbl, image_size=probe.cfg.image_size,
                                       patch_size=probe.cfg.patch_size)
            if all_layer_feats is None:
                all_layer_feats = [[] for _ in layers]
            for li, l in enumerate(layers):
                all_layer_feats[li].append(l)
            all_labels.append(ys)
            if idx % 20 == 0:
                log.info("[%s] %d/%d", tag, idx, len(sub))
        layer_arrs = [np.concatenate(L, axis=0) for L in all_layer_feats]
        labels = np.concatenate(all_labels, axis=0)
        cache[tag] = dict(layers=layer_arrs, labels=labels)
        # Drop the model from VRAM before next.
        del probe
        torch.cuda.empty_cache()
        log.info("[%s] cached %d layers, total patches=%d, label classes=%s",
                 tag, len(layer_arrs), labels.shape[0], np.unique(labels))

    # === Layer-wise CKA: standalone vs π0, standalone vs π0.5 ===
    n_layers = len(cache["standalone"]["layers"])
    log.info("computing layer-wise CKA over %d layers ...", n_layers)
    cka_pi0 = []
    cka_pi05 = []
    for li in range(n_layers):
        Xs = cache["standalone"]["layers"][li]
        Xp = cache["pi0"]["layers"][li]
        Xq = cache["pi05"]["layers"][li]
        cka_pi0.append(linear_cka(Xs, Xp))
        cka_pi05.append(linear_cka(Xs, Xq))
        log.info("layer %2d: CKA(stand,π0)=%.4f  CKA(stand,π0.5)=%.4f",
                 li, cka_pi0[-1], cka_pi05[-1])

    np.save(out_dir / "cka_layers_pi0.npy", np.array(cka_pi0))
    np.save(out_dir / "cka_layers_pi05.npy", np.array(cka_pi05))

    # === Per-class cosine similarity at the final layer ===
    final_idx = n_layers - 1
    Xs = cache["standalone"]["layers"][final_idx]
    Xp = cache["pi0"]["layers"][final_idx]
    Xq = cache["pi05"]["layers"][final_idx]
    labels = cache["standalone"]["labels"]
    classes_present = sorted(set(int(c) for c in np.unique(labels)))
    log.info("final-layer classes present: %s", classes_present)

    per_class = {}
    for cls_id in classes_present:
        mask = labels == cls_id
        n = int(mask.sum())
        if n < 5:
            continue
        ms = Xs[mask].mean(axis=0)
        mp = Xp[mask].mean(axis=0)
        mq = Xq[mask].mean(axis=0)

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        per_class[CLASSES[cls_id]] = dict(
            n_patches=n,
            cos_standalone_pi0=cos(ms, mp),
            cos_standalone_pi05=cos(ms, mq),
            l2_standalone_pi0=float(np.linalg.norm(ms - mp) / (np.linalg.norm(ms) + 1e-12)),
            l2_standalone_pi05=float(np.linalg.norm(ms - mq) / (np.linalg.norm(ms) + 1e-12)),
        )
        log.info("class=%s n=%d cos(s,π0)=%.4f cos(s,π0.5)=%.4f",
                 CLASSES[cls_id], n,
                 per_class[CLASSES[cls_id]]["cos_standalone_pi0"],
                 per_class[CLASSES[cls_id]]["cos_standalone_pi05"])

    with open(out_dir / "per_class_drift.json", "w") as f:
        json.dump(per_class, f, indent=2)

    # === Plot ===
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)

    xs = np.arange(n_layers)
    ax1.plot(xs, cka_pi0, marker="o", label="standalone vs π0", color="#c94a72")
    ax1.plot(xs, cka_pi05, marker="s", label="standalone vs π0.5", color="#aa6c39")
    ax1.set_xlabel("transformer layer (0 = patch embedding)")
    ax1.set_ylabel("linear CKA")
    ax1.set_title("Where does VLA fine-tuning diverge from the standalone encoder?")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend()

    cls_names_ord = [c for c in FOREGROUND if c in per_class]
    cos_pi0 = [per_class[c]["cos_standalone_pi0"] for c in cls_names_ord]
    cos_pi05 = [per_class[c]["cos_standalone_pi05"] for c in cls_names_ord]
    w = 0.4
    xc = np.arange(len(cls_names_ord))
    ax2.bar(xc - w / 2, cos_pi0, w, label="π0", color="#c94a72")
    ax2.bar(xc + w / 2, cos_pi05, w, label="π0.5", color="#aa6c39")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(cls_names_ord)
    ax2.set_ylabel("cosine similarity to standalone-SigLIP class-mean")
    ax2.set_title("Per-class drift in final layer (lower = more degraded)")
    ax2.set_ylim(0.5, 1.0)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend()

    fig.tight_layout()
    out_path = out_dir / "cka_and_drift.png"
    fig.savefig(out_path)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/mechanism")
    args = ap.parse_args()
    main(args)
