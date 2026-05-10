"""Aggregate outputs/tables_500/*_overall.csv into figures + markdown for the
n=500 evaluation. Mirrors summarize_probes.py / summarize_test.py but for
the bigger split.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def _load_all(root: Path, methods: list[str]) -> list[dict]:
    rows = []
    for m in methods:
        d = _read(root / f"{m}_overall.csv")
        if d is not None:
            rows.append(d)
    return rows


def main(out_fig: str, out_perclass: str, out_md: str, val_root: str, test_root: str):
    log = logging.getLogger("summarize_n500")
    methods = [
        "random_features", "florence2", "siglip2", "openpi_siglip",
        "pi0_siglip", "pi05_siglip",
        "dinov2", "dinov2_large",
    ]
    val_rows = _load_all(Path(val_root), methods)
    test_rows = _load_all(Path(test_root), methods)
    by_method_val = {r["method"]: r for r in val_rows}
    by_method_test = {r["method"]: r for r in test_rows}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: val + test side-by-side bars per method.
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=140)
    xs = np.arange(len(methods))
    w = 0.35
    val_vals = [float(by_method_val.get(m, {"mIoU": "nan"}).get("mIoU", "nan")) for m in methods]
    test_vals = [float(by_method_test.get(m, {"mIoU": "nan"}).get("mIoU", "nan")) for m in methods]
    bars1 = ax.bar(xs - w / 2, val_vals, w, label="val (n=73)", color="#4a72c9")
    bars2 = ax.bar(xs + w / 2, test_vals, w, label="test (n=75)", color="#c94a72")
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("mIoU")
    ax.set_title("Probe mIoU on UMD @ n=500 (train=345)")
    ax.set_ylim(0, max([v for v in val_vals + test_vals if not np.isnan(v)], default=0.8) * 1.15)
    ax.legend(fontsize=9)
    for b, v in zip(bars1, val_vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for b, v in zip(bars2, test_vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    log.info("wrote %s", out_fig)
    plt.close(fig)

    # Per-class on test
    classes = ["grasp", "cut", "scoop", "contain", "support"]
    fig2, ax2 = plt.subplots(figsize=(9.5, 4), dpi=140)
    palette = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    width = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        d = by_method_test.get(m)
        if d is None:
            continue
        vals = [float(d.get(f"IoU_{c}", "nan")) for c in classes]
        ax2.bar(np.arange(len(classes)) + i * width - 0.4 + width / 2, vals, width,
                label=m, color=palette[i])
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes)
    ax2.set_ylabel("IoU")
    ax2.set_title("Per-class IoU on UMD test (n=75)")
    ax2.legend(fontsize=8, ncol=3)
    fig2.tight_layout()
    fig2.savefig(out_perclass)
    log.info("wrote %s", out_perclass)
    plt.close(fig2)

    # Markdown table
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("| method | val mIoU | test mIoU | val pixel-acc | test pixel-acc |\n")
        f.write("|---|---|---|---|---|\n")
        for m in methods:
            v = by_method_val.get(m)
            t = by_method_test.get(m)
            f.write(f"| {m} "
                    f"| {float(v['mIoU']):.3f} " if v else f"| {m} | — "
                    f"| {float(t['mIoU']):.3f} " if t else "| — "
                    f"| {float(v['pixel_acc']):.3f} " if v else "| — "
                    f"| {float(t['pixel_acc']):.3f} |\n" if t else "| — |\n")
    log.info("wrote %s", out_md)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default="outputs/tables_500")
    ap.add_argument("--test-root", default="outputs/tables_500_test")
    ap.add_argument("--out-fig", default="outputs/figures/probe_miou_n500.png")
    ap.add_argument("--out-perclass", default="outputs/figures/probe_perclass_n500.png")
    ap.add_argument("--out-md", default="outputs/tables_500/probe_summary_n500.md")
    args = ap.parse_args()
    main(args.out_fig, args.out_perclass, args.out_md, args.val_root, args.test_root)
