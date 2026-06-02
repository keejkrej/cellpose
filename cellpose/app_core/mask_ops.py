"""Shared mask pixel operations for desktop GUI and sidecar."""

from __future__ import annotations

import cv2
import fastremap
import numpy as np

from cellpose.utils import masks_to_outlines


def ensure_3d_masks(masks: np.ndarray) -> np.ndarray:
    if masks.ndim == 2:
        return masks[np.newaxis, ...]
    return masks


def compute_outlines(cellpix: np.ndarray) -> np.ndarray:
    cellpix = ensure_3d_masks(cellpix)
    outpix = np.zeros_like(cellpix)
    for z in range(cellpix.shape[0]):
        outlines = masks_to_outlines(cellpix[z])
        outpix[z] = outlines * cellpix[z]
    return outpix


def renumber_masks(masks: np.ndarray) -> np.ndarray:
    shape = masks.shape
    if len(fastremap.unique(masks)) != masks.max() + 1:
        fastremap.renumber(masks, in_place=True)
        masks = masks.reshape(shape)
    return masks


def normalize_mask_dtype(masks: np.ndarray) -> np.ndarray:
    masks = ensure_3d_masks(masks)
    if masks.max() < 2**16 - 1:
        return masks.astype(np.uint16)
    return masks.astype(np.uint32)


def remove_cells_from_arrays(
    cellpix: np.ndarray,
    outpix: np.ndarray | None,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray | None]:
    masks = cellpix.copy()
    outlines = outpix.copy() if outpix is not None else None
    for idx in sorted(indices, reverse=True):
        if idx <= 0 or idx > masks.max():
            continue
        for z in range(masks.shape[0]):
            masks[z, masks[z] == idx] = 0
            if outlines is not None:
                outlines[z, outlines[z] == idx] = 0
        masks[masks > idx] -= 1
        if outlines is not None:
            outlines[outlines > idx] -= 1
    return masks, outlines


def paint_mask(
    cellpix: np.ndarray,
    outpix: np.ndarray,
    z: int,
    ar: np.ndarray,
    ac: np.ndarray,
    vr: np.ndarray,
    vc: np.ndarray,
    idx: int,
) -> None:
    cellpix[z, vr, vc] = idx
    cellpix[z, ar, ac] = idx
    outpix[z, vr, vc] = idx


def cell_bounds(
    cellpix: np.ndarray, idx: int, ly: int, lx: int, margin: int = 2
) -> tuple[int, int, int, int] | None:
    mask = cellpix == idx
    if not np.any(mask):
        return None
    ar, ac = np.nonzero(mask)
    y0 = max(0, int(ar.min()) - margin)
    y1 = min(ly - 1, int(ar.max()) + margin)
    x0 = max(0, int(ac.min()) - margin)
    x1 = min(lx - 1, int(ac.max()) + margin)
    return x0, y0, x1, y1


def normalize_rect(
    x0: int, y0: int, x1: int, y1: int, ly: int, lx: int
) -> tuple[int, int, int, int] | None:
    x0, x1 = sorted([int(x0), int(x1)])
    y0, y1 = sorted([int(y0), int(y1)])
    x0 = max(0, min(lx - 1, x0))
    x1 = max(0, min(lx, x1))
    y0 = max(0, min(ly - 1, y0))
    y1 = max(0, min(ly, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def cells_fully_in_rect(
    cellpix: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    ly: int,
    lx: int,
    filter_class_id: int | None = None,
    instance_classes: np.ndarray | None = None,
) -> list[int]:
    bounds = normalize_rect(x0, y0, x1, y1, ly, lx)
    if bounds is None:
        return []
    x0, y0, x1, y1 = bounds
    candidates = np.unique(cellpix[y0:y1, x0:x1])
    candidates = np.trim_zeros(candidates)
    fully_covered = []
    for idx in candidates:
        ar, ac = np.nonzero(cellpix == idx)
        if ar.min() < y0 or ar.max() >= y1 or ac.min() < x0 or ac.max() >= x1:
            continue
        row = int(idx) - 1
        if filter_class_id is not None and instance_classes is not None:
            if row >= len(instance_classes) or int(instance_classes[row]) != filter_class_id:
                continue
        fully_covered.append(int(idx))
    return sorted(fully_covered)
