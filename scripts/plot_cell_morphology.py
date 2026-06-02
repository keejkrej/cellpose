"""Plot cell morphology distributions from a Cellpose _seg.npy file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cellpose.gui.core.session import read_session
from cellpose.io import imread_2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from morphology_common import (  # noqa: E402
    mask_plane,
    per_cell_pixel_cv,
    per_cell_size_aspect,
    resolve_image_path,
)


def plot_morphology(
    size: np.ndarray,
    aspect: np.ndarray,
    cv: np.ndarray,
    *,
    title: str,
    output: Path | None,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title)

    axes[0].hist(size, bins=40, color="#4c72b0", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Cell size (px)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Mean diameter\n(major + minor) / 2")
    axes[0].axvline(size.mean(), color="#c44e52", linestyle="--", linewidth=1, label=f"mean={size.mean():.1f}")
    axes[0].legend()

    axes[1].hist(aspect, bins=40, color="#55a868", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Aspect ratio")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Major / minor")
    axes[1].axvline(aspect.mean(), color="#c44e52", linestyle="--", linewidth=1, label=f"mean={aspect.mean():.2f}")
    axes[1].legend()

    axes[2].hist(cv, bins=40, color="#dd8452", edgecolor="white", linewidth=0.5)
    axes[2].set_xlabel("Coefficient of variation")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Intensity CV\n(std / mean inside mask)")
    axes[2].axvline(cv.mean(), color="#c44e52", linestyle="--", linewidth=1, label=f"mean={cv.mean():.3f}")
    axes[2].legend()

    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Wrote {output}")
    if show or output is None:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Z plane index for 3D mask stacks (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path instead of only showing it",
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
    plane = mask_plane(session.masks, args.z)

    if image.shape[:2] != plane.shape[:2]:
        raise SystemExit(
            f"Image shape {image.shape[:2]} does not match mask shape {plane.shape[:2]}"
        )

    size_all, aspect_all = per_cell_size_aspect(plane)
    cv_all = per_cell_pixel_cv(plane, image)
    valid = (
        np.isfinite(size_all)
        & np.isfinite(aspect_all)
        & (aspect_all >= 1.0)
        & np.isfinite(cv_all)
    )
    size = size_all[valid]
    aspect = aspect_all[valid]
    cv = cv_all[valid]

    if size.size == 0:
        raise SystemExit("No valid cells found (need masks with >= 5 pixels per cell).")

    z_label = f", z={args.z if args.z is not None else 0}" if np.asarray(session.masks).squeeze().ndim == 3 else ""
    title = f"{seg_path.name}: {size.size} cells{z_label}"
    print(
        f"cells={size.size}  "
        f"size px: mean={size.mean():.1f} median={np.median(size):.1f}  "
        f"aspect: mean={aspect.mean():.2f} median={np.median(aspect):.2f}  "
        f"CV: mean={cv.mean():.3f} median={np.median(cv):.3f}"
    )
    plot_morphology(size, aspect, cv, title=title, output=args.output, show=args.show)


if __name__ == "__main__":
    main()
