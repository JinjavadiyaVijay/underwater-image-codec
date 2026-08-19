"""Visualization tools: side-by-side comparisons, rate curves, failure galleries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_comparison_grid(
    originals: list[np.ndarray],
    reconstructions: dict[str, list[np.ndarray]],
    labels: list[str] | None = None,
    title: str = "",
    max_samples: int = 8,
    cell_size: int = 128,
    padding: int = 4,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Create a side-by-side comparison grid.

    Layout: each row is one sample.
    Columns: Original | Recon@64B | Recon@96B | Recon@124B (or custom columns).

    Args:
        originals: List of original images (H, W, 3).
        reconstructions: Dict mapping column name → list of reconstructed images.
        labels: Optional per-sample labels (e.g., species names).
        title: Grid title.
        max_samples: Maximum number of rows.
        cell_size: Size of each cell in pixels.
        padding: Padding between cells.
        save_path: If provided, save the grid image.

    Returns:
        Grid image as uint8 numpy array.
    """
    n_samples = min(len(originals), max_samples)
    col_names = ["Original"] + list(reconstructions.keys())
    n_cols = len(col_names)

    header_height = 30
    label_width = 120 if labels else 0

    grid_w = label_width + n_cols * (cell_size + padding) + padding
    grid_h = header_height + n_samples * (cell_size + padding) + padding

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # light gray bg

    # Create PIL image for text rendering
    grid_pil = Image.fromarray(grid)
    draw = ImageDraw.Draw(grid_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # Draw column headers
    for col_idx, col_name in enumerate(col_names):
        x = label_width + col_idx * (cell_size + padding) + padding + cell_size // 4
        draw.text((x, 5), col_name, fill=(0, 0, 0), font=font)

    # Draw title
    if title:
        draw.text((grid_w // 4, 2), title, fill=(0, 0, 128), font=font)

    grid = np.array(grid_pil)

    # Fill cells
    for row in range(n_samples):
        y_offset = header_height + row * (cell_size + padding) + padding

        # Label
        if labels and row < len(labels):
            grid_pil = Image.fromarray(grid)
            draw = ImageDraw.Draw(grid_pil)
            draw.text(
                (4, y_offset + cell_size // 2 - 6),
                labels[row][:15],
                fill=(0, 0, 0),
                font=font_small,
            )
            grid = np.array(grid_pil)

        # Original
        x_offset = label_width + padding
        orig_resized = _resize_cell(originals[row], cell_size)
        _paste_cell(grid, orig_resized, x_offset, y_offset)

        # Reconstructions
        for col_idx, (col_name, recon_list) in enumerate(reconstructions.items(), 1):
            x_offset = label_width + col_idx * (cell_size + padding) + padding
            if row < len(recon_list):
                recon_resized = _resize_cell(recon_list[row], cell_size)
                _paste_cell(grid, recon_resized, x_offset, y_offset)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grid).save(str(save_path))

    return grid


def create_rate_quality_plot(
    byte_budgets: list[int],
    metrics_per_budget: dict[str, list[float]],
    metric_names: list[str] | None = None,
    title: str = "Rate-Quality Curves",
    save_path: str | Path | None = None,
) -> None:
    """Create rate-quality curve plots.

    Args:
        byte_budgets: List of byte budgets tested.
        metrics_per_budget: Dict mapping metric name → list of values (one per budget).
        metric_names: Which metrics to plot. If None, plot all.
        title: Plot title.
        save_path: If provided, save the plot.
    """
    import matplotlib.pyplot as plt

    if metric_names is None:
        metric_names = list(metrics_per_budget.keys())

    fig, axes = plt.subplots(
        1, len(metric_names), figsize=(5 * len(metric_names), 4), squeeze=False
    )

    for idx, metric in enumerate(metric_names):
        ax = axes[0, idx]
        values = metrics_per_budget.get(metric, [])
        if values:
            ax.plot(byte_budgets[: len(values)], values, "o-", linewidth=2, markersize=8)
            ax.set_xlabel("Payload Bytes")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(byte_budgets)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    plt.close()


def create_failure_gallery(
    failures: list[dict[str, Any]],
    save_path: str | Path | None = None,
    max_samples: int = 16,
    cell_size: int = 128,
) -> np.ndarray:
    """Create a gallery of failure cases for inspection.

    Args:
        failures: List of dicts with 'original', 'reconstruction', 'species', 'failure_mode'.
        save_path: If provided, save the gallery.

    Returns:
        Gallery image as numpy array.
    """
    n = min(len(failures), max_samples)
    if n == 0:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    padding = 4
    header = 20
    grid_w = cols * 2 * (cell_size + padding) + padding  # 2 cols per sample (orig + recon)
    grid_h = rows * (cell_size + header + padding) + padding

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    for idx in range(n):
        f = failures[idx]
        row = idx // cols
        col = idx % cols

        y = row * (cell_size + header + padding) + padding
        x_orig = col * 2 * (cell_size + padding) + padding
        x_recon = x_orig + cell_size + padding

        # Paste original and reconstruction
        orig = _resize_cell(f["original"], cell_size)
        recon = _resize_cell(f["reconstruction"], cell_size)
        _paste_cell(grid, orig, x_orig, y + header)
        _paste_cell(grid, recon, x_recon, y + header)

        # Red border on reconstruction (failure indicator)
        _draw_border(grid, x_recon, y + header, cell_size, cell_size, color=(255, 0, 0))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grid).save(str(save_path))

    return grid


def _resize_cell(image: np.ndarray, size: int) -> np.ndarray:
    """Resize image to cell size."""
    img = Image.fromarray(image)
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def _paste_cell(grid: np.ndarray, cell: np.ndarray, x: int, y: int) -> None:
    """Paste a cell into the grid."""
    h, w = cell.shape[:2]
    gh, gw = grid.shape[:2]
    h = min(h, gh - y)
    w = min(w, gw - x)
    if h > 0 and w > 0:
        grid[y : y + h, x : x + w] = cell[:h, :w]


def _draw_border(
    grid: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple = (255, 0, 0),
    thickness: int = 2,
) -> None:
    """Draw a colored border on the grid."""
    for t in range(thickness):
        # Top
        if y + t < grid.shape[0]:
            grid[y + t, x : x + w] = color
        # Bottom
        if y + h - 1 - t < grid.shape[0] and y + h - 1 - t >= 0:
            grid[y + h - 1 - t, x : x + w] = color
        # Left
        if x + t < grid.shape[1]:
            grid[y : y + h, x + t] = color
        # Right
        if x + w - 1 - t < grid.shape[1] and x + w - 1 - t >= 0:
            grid[y : y + h, x + w - 1 - t] = color
