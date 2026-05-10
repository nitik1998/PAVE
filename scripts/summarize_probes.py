"""Aggregate outputs/tables/<method>_overall.csv into a single figure + LaTeX-ready table.

Usage:
    python scripts/summarize_probes.py
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np


def _read_overall(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def main(tables_root: str, out_fig: str, out_md: str):
    log = logging.getLogger("summarize")
    methods = ["dinov3", "dinov2", "dinov2_448", "dinov2_448_full", "dinov2_large", "siglip2", "openpi_siglip", "qwen25vl", "molmoe", "florence2", "random_features"]
    rows = []
    for m in methods:
        d = _read_overall(Path(tables_root) / f"{m}_overall.csv")
        if d is not None:
            rows.append(d)
    if not rows:
        raise SystemExit("No overall CSVs found under " + tables_root)

    class_names = [k.replace("IoU_", "") for k in rows[0].keys() if k.startswith("IoU_")]

    # Figure 1: mIoU bar chart.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=140)
    xs = np.arange(len(rows))
    miou = [float(r["mIoU"]) for r in rows]
    bars = ax.bar(xs, miou, color="#4a72c9")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["method"] for r in rows], rotation=15, ha="right")
    ax.set_ylabel("mIoU (UMD val/test subset)")
    ax.set_ylim(0, max(0.5, max(miou) * 1.2))
    for b, v in zip(bars, miou):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title("Cross-method affordance mIoU on UMD")
    fig.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    log.info("wrote %s", out_fig)
    plt.close(fig)

    # Figure 2: per-class IoU bar chart (skip background).
    fg = [c for c in class_names if c != "background"]
    fig2, ax2 = plt.subplots(figsize=(7.5, 3.6), dpi=140)
    width = 0.8 / max(1, len(rows))
    palette = plt.cm.tab10(np.linspace(0, 1, len(rows)))
    for i, r in enumerate(rows):
        vals = [float(r.get(f"IoU_{c}", "nan")) for c in fg]
        ax2.bar(np.arange(len(fg)) + i * width - 0.4 + width / 2, vals, width, label=r["method"], color=palette[i])
    ax2.set_xticks(np.arange(len(fg)))
    ax2.set_xticklabels(fg)
    ax2.set_ylabel("IoU")
    ax2.set_title("Per-class IoU (foreground)")
    ax2.legend(fontsize=8, ncol=min(3, len(rows)))
    fig2.tight_layout()
    out_fig2 = str(Path(out_fig).with_name(Path(out_fig).stem + "_perclass.png"))
    fig2.savefig(out_fig2)
    log.info("wrote %s", out_fig2)
    plt.close(fig2)

    # Markdown table.
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        head = ["method", "n", "mIoU", "pixel_acc"] + [f"IoU_{c}" for c in fg]
        f.write("| " + " | ".join(head) + " |\n")
        f.write("|" + "|".join(["---"] * len(head)) + "|\n")
        for r in rows:
            cells = [
                r.get("method", "?"),
                r.get("n", "?"),
                f"{float(r.get('mIoU', 'nan')):.3f}",
                f"{float(r.get('pixel_acc', 'nan')):.3f}",
            ] + [f"{float(r.get(f'IoU_{c}', 'nan')):.3f}" for c in fg]
            f.write("| " + " | ".join(str(c) for c in cells) + " |\n")
    log.info("wrote %s", out_md)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-root", default="outputs/tables")
    ap.add_argument("--out-fig", default="outputs/figures/probe_miou.png")
    ap.add_argument("--out-md", default="outputs/tables/probe_summary.md")
    args = ap.parse_args()
    main(args.tables_root, args.out_fig, args.out_md)
