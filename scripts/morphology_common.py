"""Shared helpers for morphology plotting scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cellpose.plot import image_to_rgb
from cellpose.utils import get_mask_ellipse_diameters, get_mask_pixel_cv


def mask_plane(masks: np.ndarray, z: int | None) -> np.ndarray:
    masks = np.asarray(masks).squeeze()
    if masks.ndim == 2:
        return masks
    if masks.ndim != 3:
        raise ValueError(f"Expected 2D or 3D masks, got shape {masks.shape}")
    index = 0 if z is None else z
    if index < 0 or index >= masks.shape[0]:
        raise ValueError(f"z={index} out of range for stack with {masks.shape[0]} planes")
    return masks[index]


def resolve_image_path(seg_path: Path, session_source: str, image_arg: str | None) -> Path:
    if image_arg:
        path = Path(image_arg)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    candidates = []
    if session_source:
        candidates.append(Path(session_source))
    stem = seg_path.name.removesuffix("_seg.npy")
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        candidates.append(seg_path.with_name(stem + ext))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
        alt = Path(str(candidate).replace("/home/jack/", "/Users/jack/"))
        if alt.is_file():
            return alt

    raise FileNotFoundError(
        "Could not find source image. Pass --image explicitly."
    )


def per_cell_size_aspect(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    major, minor = get_mask_ellipse_diameters(masks)
    size = (major + minor) / 2.0
    aspect = np.divide(
        major,
        minor,
        out=np.full_like(major, np.nan),
        where=np.isfinite(minor) & (minor > 0),
    )
    return size, aspect


def per_cell_pixel_cv(masks: np.ndarray, image: np.ndarray) -> np.ndarray:
    return get_mask_pixel_cv(masks, image)


def flat_mask_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    colors: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """Normal blend at fixed opacity, matching the GUI mask compositing."""
    rgb = image_to_rgb(image)
    base = rgb.astype(np.float32)
    if base.max() <= 1.0:
        base *= 255.0

    out = base.copy()
    blend = float(np.clip(alpha, 0.0, 1.0))
    inv = 1.0 - blend
    ncells = int(masks.max())
    for ic in range(ncells):
        pixels = masks == (ic + 1)
        if not np.any(pixels):
            continue
        color = colors[ic].astype(np.float32)
        out[pixels] = out[pixels] * inv + color * blend

    return np.clip(out, 0, 255).astype(np.uint8)
