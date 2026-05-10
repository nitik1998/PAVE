"""H8 Part 1: action-prediction from each encoder's RGB features (PickCube).

Pipeline per encoder:
  1. Mean-pool patch features → (N, D) feature matrix.
  2. Ridge regression features → 8-dim action.
  3. Report mean L2 action-prediction error on held-out frames.

We also evaluate the "adapter-on-π0" transform: using the 256-d hidden
representation of the trained UMD adapter as the feature for the Ridge head.
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


def featurize_dinov2(rgbs, image_size, base=True, device="cuda"):
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    name = "facebook/dinov2-base" if base else "facebook/dinov2-large"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to(device)
    feats = []
    mean = np.asarray(getattr(proc, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(getattr(proc, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    for i, rgb in enumerate(rgbs):
        pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
        arr = (np.asarray(pil, dtype=np.float32) / 255.0 - mean) / std
        pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
        import inspect

        with torch.no_grad():
            sig = inspect.signature(model.forward)
            if "interpolate_pos_encoding" in sig.parameters:
                out = model(pix, interpolate_pos_encoding=True)
            else:
                out = model(pix)
        f = out.last_hidden_state[0]
        # Strip CLS
        if f.shape[0] > (image_size // 14) ** 2:
            f = f[-(image_size // 14) ** 2:]
        feats.append(f.mean(0).cpu().numpy())
    return np.stack(feats)


def featurize_hf_siglip(rgbs, model, processor, image_size, device="cuda"):
    from PIL import Image

    feats = []
    mean = np.asarray(getattr(processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(getattr(processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    n_patches_target = (image_size // 14) ** 2
    for rgb in rgbs:
        pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
        arr = (np.asarray(pil, dtype=np.float32) / 255.0 - mean) / std
        pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            out = model(pix)
        f = out.last_hidden_state[0]
        if f.shape[0] > n_patches_target:
            f = f[-n_patches_target:]
        feats.append(f.mean(0).cpu().numpy())
    return np.stack(feats)


def featurize_timm_siglip(rgbs, model, processor, image_size, device="cuda"):
    from PIL import Image

    feats = []
    mean = np.asarray(getattr(processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(getattr(processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    for rgb in rgbs:
        pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
        arr = (np.asarray(pil, dtype=np.float32) / 255.0 - mean) / std
        pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            f = model.forward_features(pix)[0]
        feats.append(f.mean(0).cpu().numpy())
    return np.stack(feats)


def build_pi0_full(use_pi05=False, device="cuda"):
    from src.methods.pi0_siglip_probe import Pi0SigLIPConfig, Pi0SigLIPProbe

    cfg = Pi0SigLIPConfig(device=device, use_pi05=use_pi05)
    p = Pi0SigLIPProbe(cfg=cfg, num_classes=6, foreground_names=["grasp", "cut", "scoop", "contain", "support"])
    p.warmup()
    return p


def build_openvla_full(device="cuda"):
    from src.methods.openvla_siglip_probe import OpenVLASigLIPConfig, OpenVLASigLIPProbe

    cfg = OpenVLASigLIPConfig(device=device)
    p = OpenVLASigLIPProbe(cfg=cfg, num_classes=6, foreground_names=["grasp", "cut", "scoop", "contain", "support"])
    p.warmup()
    return p


def build_standalone(device="cuda"):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-224")
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-224").vision_model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, proc


class MLPAdapter(nn.Module):
    """Same architecture as scripts/intervention/adapter_recovery.py.
    We use the *post-GELU hidden* representation as the recovered feature."""
    def __init__(self, in_dim=1152, num_classes=6, hidden=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.act(self.proj(x))
        return self.cls(self.drop(h)), h

    @classmethod
    def from_state_dict(cls, sd, in_dim=1152, num_classes=6, hidden=256):
        m = cls(in_dim=in_dim, num_classes=num_classes, hidden=hidden)
        # Source state dict (`adapter_*_h256.pt`) used `nn.Sequential` whose
        # keys are `net.0.*` and `net.3.*`. Map them.
        new_sd = {
            "proj.weight": sd["net.0.weight"],
            "proj.bias": sd["net.0.bias"],
            "cls.weight": sd["net.3.weight"],
            "cls.bias": sd["net.3.bias"],
        }
        m.load_state_dict(new_sd)
        return m


def featurize_adapter_pi0(rgbs, image_size=224, device="cuda", which_pi="pi0"):
    """Extract π0 features → push through adapter → return 256-d hidden.
    `which_pi` ∈ {pi0, pi05} — selects which π model AND which adapter file."""
    pi_full = build_pi0_full(use_pi05=(which_pi == "pi05"), device=device)
    sd_path = Path("outputs/intervention") / f"adapter_{which_pi}_siglip_h256.pt"
    sd = torch.load(sd_path, map_location=device)
    adapter = MLPAdapter.from_state_dict(sd, in_dim=1152, hidden=256).to(device).eval()

    from PIL import Image

    proc = pi_full._processor
    mean = np.asarray(getattr(proc, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(getattr(proc, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    n_patches = (image_size // 14) ** 2
    feats = []
    for rgb in rgbs:
        pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
        arr = (np.asarray(pil, dtype=np.float32) / 255.0 - mean) / std
        pix = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            out = pi_full._backbone(pix)
            f = out.last_hidden_state[0]
            if f.shape[0] > n_patches:
                f = f[-n_patches:]
            _, h = adapter(f)
        feats.append(h.mean(0).cpu().numpy())
    return np.stack(feats)


def ridge_eval(X_train, y_train, X_test, y_test, alpha=1.0):
    from sklearn.linear_model import Ridge

    clf = Ridge(alpha=alpha)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    err = np.linalg.norm(y_pred - y_test, axis=-1)
    return dict(
        mean_l2=float(err.mean()),
        median_l2=float(np.median(err)),
        std_l2=float(err.std()),
        per_dim_mae=np.abs(y_pred - y_test).mean(axis=0).tolist(),
    )


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("h8")

    data = np.load(args.data)
    rgbs, actions = data["rgbs"], data["actions"]
    log.info("loaded %d frames, action_dim=%d", rgbs.shape[0], actions.shape[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train/test split (deterministic).
    rng = np.random.default_rng(0)
    n = rgbs.shape[0]
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr_idx, te_idx = perm[:cut], perm[cut:]
    log.info("train n=%d test n=%d", len(tr_idx), len(te_idx))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Encoders to evaluate.
    cfgs = [
        ("dinov2_base",      lambda: featurize_dinov2(rgbs, 224, base=True, device=device)),
        ("dinov2_large",     lambda: featurize_dinov2(rgbs, 224, base=False, device=device)),
        ("standalone_siglip", lambda: _featurize_via("standalone")),
        ("pi0_siglip",       lambda: _featurize_via("pi0")),
        ("pi05_siglip",      lambda: _featurize_via("pi05")),
        ("openvla_siglip",   lambda: _featurize_via("openvla")),
        ("pi0_adapter",      lambda: featurize_adapter_pi0(rgbs, 224, device, "pi0")),
        ("pi05_adapter",     lambda: featurize_adapter_pi0(rgbs, 224, device, "pi05")),
    ]

    def _featurize_via(tag):
        if tag == "standalone":
            model, proc = build_standalone(device)
            feats = featurize_hf_siglip(rgbs, model, proc, 224, device)
        elif tag == "pi0":
            p = build_pi0_full(use_pi05=False, device=device)
            feats = featurize_hf_siglip(rgbs, p._backbone, p._processor, 224, device)
        elif tag == "pi05":
            p = build_pi0_full(use_pi05=True, device=device)
            feats = featurize_hf_siglip(rgbs, p._backbone, p._processor, 224, device)
        elif tag == "openvla":
            p = build_openvla_full(device)
            feats = featurize_timm_siglip(rgbs, p._backbone, p._processor, 224, device)
        else:
            raise ValueError(tag)
        torch.cuda.empty_cache()
        return feats

    for name, fn in cfgs:
        t0 = time.time()
        log.info(">>> %s ...", name)
        try:
            X = fn()
        except Exception as e:
            log.exception("FAILED %s: %s", name, e)
            continue
        log.info("    feats=%s extracted in %.1fs", X.shape, time.time() - t0)
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = actions[tr_idx], actions[te_idx]
        # Standardize features (Ridge handles this well, but center anyway).
        mu, sigma = X_train.mean(0, keepdims=True), X_train.std(0, keepdims=True) + 1e-6
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        res = ridge_eval(X_train, y_train, X_test, y_test, alpha=args.alpha)
        results[name] = dict(res, n_train=int(len(tr_idx)), n_test=int(len(te_idx)),
                             feat_dim=int(X.shape[1]))
        log.info("    %s mean_L2=%.4f median=%.4f", name, res["mean_l2"], res["median_l2"])
        torch.cuda.empty_cache()

    with open(out_dir / "pickcube_action_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out_dir / "pickcube_action_results.json")

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["dinov2_base", "dinov2_large", "standalone_siglip",
             "pi0_siglip", "pi0_adapter", "pi05_siglip", "pi05_adapter",
             "openvla_siglip"]
    names = [n for n in order if n in results]
    vals = [results[n]["mean_l2"] for n in names]
    colors = ["#3a8a4f" if "dinov2" in n else
              "#4a72c9" if "standalone" in n else
              "#c94a72" if n == "pi0_siglip" else
              "#d3a45f" if n == "pi0_adapter" else
              "#aa6c39" if n == "pi05_siglip" else
              "#7a4d27" if n == "pi05_adapter" else
              "#3a8a4f" for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
    bars = ax.bar(np.arange(len(names)), vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("mean L2 action-prediction error (held-out frames)")
    ax.set_title("PickCube (contain-class task) — action prediction from RGB features\n"
                 "lower = features better predict the pretrained-PPO action")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "pickcube_action_l2.png")
    log.info("wrote %s", out_dir / "pickcube_action_l2.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/h8-action-proxy/data/pickcube_action_data.npz")
    ap.add_argument("--out", default="experiments/h8-action-proxy/results")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    main(args)
