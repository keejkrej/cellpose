"""Show image with segmentation colored by morphology thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from cellpose.app_core import read_session
from cellpose.io import imread_2D
from cellpose.plot import image_to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from morphology_common import (  # noqa: E402
    flat_mask_overlay,
    mask_plane,
    per_cell_pixel_cv,
    per_cell_size_aspect,
    resolve_image_path,
)

BINARY_COLORS = np.array(
    [
        [68, 114, 196],
        [112, 173, 71],
    ],
    dtype=np.uint8,
)
QUAD_COLORS = np.array(
    [
        [68, 114, 196],
        [237, 125, 49],
        [165, 165, 165],
        [112, 173, 71],
    ],
    dtype=np.uint8,
)
METRIC_NAMES = ("size", "aspect", "cv")
METRIC_LABELS = {
    "size": ("small", "large"),
    "aspect": ("round", "elongated"),
    "cv": ("low CV", "high CV"),
}
METRIC_FORMAT = {
    "size": "{:.1f} px",
    "aspect": "{:.2f}",
    "cv": "{:.3f}",
}


def classify_cells(
    metrics: dict[str, np.ndarray],
    thresholds: dict[str, float],
    *,
    valid: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, str]:
    active = [(name, thresholds[name]) for name in METRIC_NAMES if name in thresholds]
    if not active:
        raise ValueError("At least one threshold must be set")

    ncells = next(iter(metrics.values())).shape[0]
    colors = np.zeros((ncells, 3), dtype=np.uint8)

    if len(active) == 1:
        name, threshold = active[0]
        values = metrics[name]
        low_label, high_label = METRIC_LABELS[name]
        fmt = METRIC_FORMAT[name]
        counts = np.zeros(2, dtype=int)
        for ic in range(ncells):
            if not valid[ic] or not np.isfinite(values[ic]):
                continue
            bucket = 0 if values[ic] <= threshold else 1
            colors[ic] = BINARY_COLORS[bucket]
            counts[bucket] += 1
        legend = [
            f"{low_label} (<= {fmt.format(threshold)}) ({counts[0]})",
            f"{high_label} (> {fmt.format(threshold)}) ({counts[1]})",
        ]
        text = f"{name} threshold = {fmt.format(threshold)}"
        return colors, legend, BINARY_COLORS, text

    if len(active) == 2:
        (name_a, threshold_a), (name_b, threshold_b) = active
        values_a = metrics[name_a]
        values_b = metrics[name_b]
        labels_a = METRIC_LABELS[name_a]
        labels_b = METRIC_LABELS[name_b]
        counts = np.zeros(4, dtype=int)
        quad_labels = [
            f"{labels_a[0]}, {labels_b[0]}",
            f"{labels_a[0]}, {labels_b[1]}",
            f"{labels_a[1]}, {labels_b[1]}",
            f"{labels_a[1]}, {labels_b[0]}",
        ]
        for ic in range(ncells):
            if not valid[ic]:
                continue
            if not np.isfinite(values_a[ic]) or not np.isfinite(values_b[ic]):
                continue
            low_a = values_a[ic] <= threshold_a
            low_b = values_b[ic] <= threshold_b
            if low_a and low_b:
                bucket = 0
            elif low_a and not low_b:
                bucket = 1
            elif not low_a and not low_b:
                bucket = 2
            else:
                bucket = 3
            colors[ic] = QUAD_COLORS[bucket]
            counts[bucket] += 1
        legend = [f"{quad_labels[i]} ({counts[i]})" for i in range(4)]
        text = (
            f"{name_a} threshold = {METRIC_FORMAT[name_a].format(threshold_a)}   "
            f"{name_b} threshold = {METRIC_FORMAT[name_b].format(threshold_b)}"
        )
        return colors, legend, QUAD_COLORS, text

    names = ", ".join(name for name, _ in active)
    raise ValueError(f"At most two thresholds allowed, got {len(active)}: {names}")


def show_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    colors: np.ndarray,
    *,
    title: str,
    legend_labels: list[str],
    legend_colors: np.ndarray,
    threshold_text: str,
    alpha: float,
    output: Path | None,
    show: bool,
) -> None:
    rgb = image_to_rgb(image)
    overlay = flat_mask_overlay(image, masks, colors, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)

    axes[0].imshow(rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Morphology overlay")
    axes[1].axis("off")

    legend = [
        mpatches.Patch(color=legend_colors[i] / 255.0, label=legend_labels[i])
        for i in range(len(legend_labels))
    ]
    axes[1].legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(legend_labels), 2),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.02,
        f"{threshold_text}   overlay opacity = {alpha:.0%}",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Wrote {output}")
    if show or output is None:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  two colors by size:  --size-threshold 30\n"
            "  two colors by CV:    --cv-threshold 0.15\n"
            "  four colors:         --size-threshold 30 --aspect-threshold 1.2\n"
            "  four colors (defaults): omit all thresholds (size + aspect medians)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "seg_path",
        nargs="?",
        default="/Users/jack/data/rcc_jb4/Pos27/img_000000000_Durchlicht_000_seg.npy",
        help="Path to a Cellpose _seg.npy file",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Source image path (default: resolve from seg filename or session metadata)",
    )
    parser.add_argument(
        "-z",
        type=int,
        default=None,
        help="Z plane index for 3D stacks (default: 0)",
    )
    parser.add_argument(
        "--size-threshold",
        type=float,
        default=None,
        help="Size threshold in px",
    )
    parser.add_argument(
        "--aspect-threshold",
        type=float,
        default=None,
        help="Aspect-ratio threshold",
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=None,
        help="Coefficient-of-variation threshold (std/mean inside mask)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay opacity for flat compositing (default: 0.5, matching GUI)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively even when --output is set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seg_path = Path(args.seg_path)
    session = read_session(seg_path)
    image_path = resolve_image_path(seg_path, session.source_image, args.image)
    image = imread_2D(str(image_path))
    masks = mask_plane(session.masks, args.z)

    if image.shape[:2] != masks.shape[:2]:
        raise SystemExit(
            f"Image shape {image.shape[:2]} does not match mask shape {masks.shape[:2]}"
        )

    size, aspect = per_cell_size_aspect(masks)
    cv = per_cell_pixel_cv(masks, image)
    metrics = {
        "size": size,
        "aspect": aspect,
        "cv": cv,
    }

    valid = np.isfinite(size) & np.isfinite(aspect) & (aspect >= 1.0)
    if not np.any(valid):
        raise SystemExit("No valid cells found.")

    explicit = {
        name: value
        for name, value in {
            "size": args.size_threshold,
            "aspect": args.aspect_threshold,
            "cv": args.cv_threshold,
        }.items()
        if value is not None
    }
    if len(explicit) > 2:
        raise SystemExit("Pass at most two thresholds.")

    thresholds = dict(explicit)
    if not thresholds:
        thresholds = {
            "size": float(np.median(size[valid])),
            "aspect": float(np.median(aspect[valid])),
        }

    colors, legend_labels, legend_colors, threshold_text = classify_cells(
        metrics,
        thresholds,
        valid=valid,
    )

    z_label = f", z={args.z if args.z is not None else 0}" if np.asarray(session.masks).squeeze().ndim == 3 else ""
    title = f"{seg_path.name}{z_label}"
    print(
        f"image={image_path.name}  cells={int(valid.sum())}  "
        f"{threshold_text}  legend={legend_labels}"
    )
    show_overlay(
        image,
        masks,
        colors,
        title=title,
        legend_labels=legend_labels,
        legend_colors=legend_colors,
        threshold_text=threshold_text,
        alpha=args.alpha,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
