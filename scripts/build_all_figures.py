"""One-shot rebuild of every paper figure from current CSV/JSON state.

After all probes + H3 sweep are done, run this to regenerate:
  - probe_miou.png, probe_miou_perclass.png (n=200 talk versions)
  - probe_miou_n500.png, probe_perclass_n500.png (paper versions)
  - probe_miou_test.png, probe_miou_test_perclass.png
  - scaling_curve.png
  - cross_domain_grid.png (if cross_domain_demo run already)
  - hero_panel.png
  - policy_curves.png, policy_final_bar.png (if H3 finished)
  - h3_robustness.png (if eval_h3_robustness run)
  - results.json
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


PY = sys.executable


def _run(cmd: list[str]) -> bool:
    log = logging.getLogger("rebuild")
    log.info("RUN %s", " ".join(cmd))
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("FAIL [%d]: %s", r.returncode, r.stderr.splitlines()[-3:] if r.stderr else "")
            return False
        return True
    except Exception as e:
        log.warning("EXC %s", e)
        return False


def main():
    log = logging.getLogger("rebuild")
    cmds = [
        # n=200 (legacy talk versions)
        [PY, "scripts/summarize_probes.py"],
        [PY, "scripts/summarize_test.py"],
        [PY, "scripts/qual_grid.py", "--methods", "dinov2", "dinov2_448_full",
         "dinov2_large", "siglip2", "openpi_siglip", "florence2", "qwen25vl",
         "--n", "5", "--split-file", "data/umd/splits/val.json",
         "--out", "outputs/figures/qual_grid.png"],
        [PY, "scripts/hero_panel.py"],
        # n=500
        [PY, "scripts/summarize_n500.py"],
        # JSON dump
        [PY, "scripts/dump_results_json.py"],
        # H3 if available
        [PY, "scripts/plot_h3_curves.py"],
    ]
    for cmd in cmds:
        _run(cmd)
    log.info("DONE rebuilding figures")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.parse_args()
    main()
