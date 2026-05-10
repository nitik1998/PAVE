from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Stable per-class colors for figures and overlays.
CLASS_COLORS = {
    0: (0, 0, 0),          # background
    1: (255, 64, 64),      # grasp
    2: (255, 196, 0),      # cut
    3: (0, 196, 255),      # scoop
    4: (128, 64, 255),     # contain
    5: (64, 255, 128),     # support
}


def colorize_label_map(label: np.ndarray) -> np.ndarray:
    """label: (H, W) int → (H, W, 3) uint8."""
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        out[label == cid] = color
    return out


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """rgb: (H, W, 3) uint8; heatmap: (H, W) float in [0,1] → uint8 overlay."""
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    h = np.clip(heatmap, 0, 1)
    cmap = plt.get_cmap("inferno")
    rgba = (cmap(h) * 255).astype(np.uint8)[..., :3]
    out = (rgb.astype(np.float32) * (1 - alpha) + rgba.astype(np.float32) * alpha).astype(np.uint8)
    return out


def overlay_multi_heatmap(rgb: np.ndarray, channels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """channels: (C, H, W) float in [0,1]. Each channel rendered in its CLASS_COLORS color."""
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    out = rgb.astype(np.float32)
    c = channels.shape[0]
    for i in range(c):
        color = np.array(CLASS_COLORS.get(i + 1, (255, 255, 255)), dtype=np.float32)
        m = np.clip(channels[i], 0, 1)[..., None]
        out = out * (1 - alpha * m) + color * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def grid_figure(
    images: Sequence[Sequence[np.ndarray]],
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    dpi: int = 120,
):
    rows = len(images)
    cols = len(images[0])
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows), squeeze=False)
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.imshow(images[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and col_labels is not None:
                ax.set_title(col_labels[c], fontsize=9)
            if c == 0 and row_labels is not None:
                ax.set_ylabel(row_labels[r], fontsize=9)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig


def save_image(arr: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_video(frames: Iterable[np.ndarray], path: str | Path, fps: int = 20) -> None:
    import imageio.v2 as imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
