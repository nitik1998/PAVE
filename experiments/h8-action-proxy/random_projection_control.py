"""Disambiguating the adapter's win: is it dim-reduction or affordance-specific?

The pi0_adapter (256-d output) beat raw pi0_siglip (1152-d output) on PickCube
action prediction. Two possible explanations:
  (a) Dim reduction — Ridge with 256-d features generalizes better than 1152-d.
  (b) Affordance-specific compression — adapter actively concentrates
      action-relevant information.

We test (a) with a random projection control: project pi0_siglip features
to 256-d via a fixed random Gaussian, then run the same Ridge.
If random_projection ≈ adapter, the win is dim-reduction-only.
If adapter > random_projection, the affordance-specific compression matters.

This script also adds the standalone-SigLIP analogue (project standalone to 256-d).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "h8_predict_actions",
    str(ROOT / "experiments" / "h8-action-proxy" / "predict_actions.py"),
)
_pa = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pa)
ridge_eval = _pa.ridge_eval
build_pi0_full = _pa.build_pi0_full
build_standalone = _pa.build_standalone
featurize_hf_siglip = _pa.featurize_hf_siglip


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("h8_control")

    data = np.load(ROOT / "experiments/h8-action-proxy/data/pickcube_action_data.npz")
    rgbs, actions = data["rgbs"], data["actions"]
    log.info("loaded %d frames", rgbs.shape[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(0)
    n = rgbs.shape[0]
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr_idx, te_idx = perm[:cut], perm[cut:]

    out = {}
    for tag, builder in [
        ("pi0_random_proj_256", "pi0"),
        ("standalone_random_proj_256", "standalone"),
    ]:
        log.info(">>> %s ...", tag)
        if builder == "pi0":
            p = build_pi0_full(use_pi05=False, device=device)
            X = featurize_hf_siglip(rgbs, p._backbone, p._processor, 224, device)
            del p
            torch.cuda.empty_cache()
        else:
            model, proc = build_standalone(device)
            X = featurize_hf_siglip(rgbs, model, proc, 224, device)
            del model
            torch.cuda.empty_cache()
        # Random projection.
        rng2 = np.random.default_rng(42)
        W = rng2.standard_normal((X.shape[1], 256)).astype(np.float32)
        W /= np.sqrt(X.shape[1])
        Xr = X @ W
        # Standardize.
        mu, sigma = Xr[tr_idx].mean(0, keepdims=True), Xr[tr_idx].std(0, keepdims=True) + 1e-6
        Xr = (Xr - mu) / sigma
        res = ridge_eval(Xr[tr_idx], actions[tr_idx], Xr[te_idx], actions[te_idx], alpha=1.0)
        out[tag] = res
        log.info("    %s mean_L2=%.4f median=%.4f", tag, res["mean_l2"], res["median_l2"])

    out_path = ROOT / "experiments/h8-action-proxy/results/random_projection_control.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
