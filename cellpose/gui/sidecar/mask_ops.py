"""Mask editing operations for sidecar sessions."""

from __future__ import annotations

import cv2
import numpy as np

from cellpose.gui.mask_ops import (
    compute_outlines,
    ensure_3d_masks,
    normalize_mask_dtype,
    remove_cells_from_arrays,
    renumber_masks,
)

from .session import SidecarSession

__all__ = [
    "apply_masks",
    "remove_cells",
    "merge_cells",
    "add_mask_from_strokes",
    "compute_outlines",
    "ensure_3d_masks",
    "normalize_mask_dtype",
    "remove_cells_from_arrays",
    "renumber_masks",
]


def _default_colors(ncells: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    colors = rng.integers(50, 255, size=(max(ncells, 1), 3), dtype=np.uint8)
    return colors


def apply_masks(session: SidecarSession, masks: np.ndarray) -> None:
    masks = renumber_masks(masks)
    masks = normalize_mask_dtype(masks)
    session.masks = masks
    session.outlines = compute_outlines(masks)
    ncells = int(masks.max())
    if session.colors is None or len(session.colors) < ncells:
        session.colors = _default_colors(ncells)
    if session.instance_classes is None or len(session.instance_classes) < ncells:
        current = session.instance_classes
        session.instance_classes = np.zeros(ncells, dtype=np.int32)
        if current is not None and len(current) > 0:
            n = min(ncells, len(current))
            session.instance_classes[:n] = current[:n]
    if session.ismanual is None or len(session.ismanual) < ncells:
        current = session.ismanual
        session.ismanual = np.zeros(ncells, dtype=bool)
        if current is not None and len(current) > 0:
            n = min(ncells, len(current))
            session.ismanual[:n] = current[:n]


def remove_cells(session: SidecarSession, indices: list[int]) -> None:
    if session.masks is None:
        return
    masks, outlines = remove_cells_from_arrays(
        session.masks, session.outlines, indices
    )
    session.masks = masks
    session.outlines = outlines
    for idx in sorted(indices, reverse=True):
        if session.instance_classes is not None and idx - 1 < len(session.instance_classes):
            session.instance_classes = np.delete(session.instance_classes, idx - 1)
        if session.ismanual is not None and idx - 1 < len(session.ismanual):
            session.ismanual = np.delete(session.ismanual, idx - 1)
        if session.colors is not None and idx < len(session.colors) + 1:
            session.colors = np.delete(session.colors, idx - 1, axis=0)


def merge_cells(session: SidecarSession, source_index: int, target_index: int) -> None:
    if session.masks is None or source_index == target_index:
        return
    masks = session.masks.copy()
    masks[masks == source_index] = target_index
    apply_masks(session, masks)
    remove_cells(session, [source_index] if source_index > target_index else [source_index])


def add_mask_from_strokes(
    session: SidecarSession,
    strokes: list[list[list[float]]],
    color: tuple[int, int, int] | None = None,
    class_id: int = 0,
) -> int | None:
    if session.masks is None:
        ly, lx = session.image.shape[-3], session.image.shape[-2]
        session.masks = np.zeros((1, ly, lx), dtype=np.uint16)
        session.outlines = np.zeros_like(session.masks)
        session.colors = np.zeros((0, 3), dtype=np.uint8)

    points_all = np.concatenate(
        [np.asarray(stroke, dtype=np.float32) for stroke in strokes], axis=0
    )
    z = 0
    ars, acs, vrs, vcs = (
        np.zeros(0, int),
        np.zeros(0, int),
        np.zeros(0, int),
        np.zeros(0, int),
    )
    for stroke in strokes:
        stroke_arr = np.asarray(stroke, dtype=np.float32).reshape(-1, 4)
        vr = stroke_arr[:, 1].astype(int)
        vc = stroke_arr[:, 2].astype(int)
        mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
        pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2), axis=-1)[:, np.newaxis, :]
        mask = cv2.fillPoly(mask, [pts.astype(np.int32)], (255, 0, 0))
        ar, ac = np.nonzero(mask)
        ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pvc, pvr = contours[-2][0][:, 0].T
        vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
        ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
        ioverlap = session.masks[z, ar, ac] > 0
        if (~ioverlap).sum() < 10:
            return None
        if ioverlap.sum() > 0:
            ar, ac = ar[~ioverlap], ac[~ioverlap]
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            mask[ar - vr.min() + 2, ac - vc.min() + 2] = 1
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0][:, 0].T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
        ars = np.concatenate((ars, ar), axis=0)
        acs = np.concatenate((acs, ac), axis=0)
        vrs = np.concatenate((vrs, vr), axis=0)
        vcs = np.concatenate((vcs, vc), axis=0)

    idx = int(session.masks.max()) + 1
    session.masks[z, vrs, vcs] = idx
    session.masks[z, ars, acs] = idx
    if session.outlines is not None:
        session.outlines[z, vrs, vcs] = idx
    if color is None:
        color = (100, 200, 50)
    session.colors = np.vstack([session.colors, np.array(color, dtype=np.uint8)])
    session.instance_classes = np.append(
        session.instance_classes if session.instance_classes is not None else np.zeros(0, np.int32),
        class_id,
    )
    session.ismanual = np.append(
        session.ismanual if session.ismanual is not None else np.zeros(0, bool),
        True,
    )
    return idx
