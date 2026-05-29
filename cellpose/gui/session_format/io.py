"""Read and write pickled `_seg.npy` session files (original Cellpose format)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from cellpose.utils import masks_to_outlines

from .models import SessionData, SegmentationMetadata


def default_session_path(image_path: str) -> str:
    base = os.path.splitext(image_path)[0]
    return base + "_seg.npy"


def _default_colors(ncells: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(50, 255, size=(max(ncells, 1), 3), dtype=np.uint8)


def _compute_outlines(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks).squeeze()
    return masks * masks_to_outlines(masks)


def _resolve_source_image(filename: str, session_path: Path) -> str:
    if os.path.isabs(filename):
        return filename
    return str((session_path.parent / filename).resolve())


def session_to_pickle_dict(session: SessionData) -> dict[str, Any]:
    """Build a legacy `_seg.npy` dict payload from ``SessionData``."""
    masks = np.ascontiguousarray(np.asarray(session.masks).squeeze())
    max_value = int(masks.max()) if masks.size else 0
    dtype = np.uint16 if max_value < 2**16 - 1 else np.uint32
    masks = masks.astype(dtype, copy=False)
    outlines = _compute_outlines(masks).astype(dtype, copy=False)

    colors = session.colors
    if colors is None and session.ncells > 0:
        colors = _default_colors(session.ncells)
    if colors is not None:
        colors = np.ascontiguousarray(colors, dtype=np.uint8)

    dat: dict[str, Any] = {
        "outlines": outlines,
        "masks": masks,
        "filename": session.source_image,
        "flows": session.flows or [],
        "flow_threshold": session.segmentation.flow_threshold,
        "cellprob_threshold": session.segmentation.cellprob_threshold,
        "diameter": session.segmentation.diameter,
        "model_path": session.model if session.model not in ("cpsam", "0") else 0,
        "restore": None,
        "ratio": 1.0,
    }
    if colors is not None:
        dat["colors"] = colors
    if session.ismanual is not None and len(session.ismanual):
        dat["ismanual"] = np.asarray(session.ismanual, dtype=bool)
    if session.instance_classes is not None and len(session.instance_classes):
        dat["instance_classes"] = np.asarray(session.instance_classes, dtype=np.int32)
    return dat


def session_from_pickle_dict(dat: dict[str, Any], session_path: Path) -> SessionData:
    """Convert a legacy `_seg.npy` dict payload to ``SessionData``."""
    if "outlines" not in dat:
        raise ValueError("Invalid _seg.npy file: missing outlines")

    masks = np.asarray(dat["masks"]).squeeze()
    filename = str(dat.get("filename", ""))
    if filename:
        source_image = _resolve_source_image(filename, session_path)
    else:
        source_image = str(session_path.with_name(session_path.stem.replace("_seg", "")))

    flows = None
    if "flows" in dat and dat["flows"] is not None:
        try:
            flows = [np.asarray(flow) for flow in dat["flows"]]
        except TypeError:
            flows = None

    colors = np.asarray(dat["colors"], dtype=np.uint8) if "colors" in dat else None
    instance_classes = (
        np.asarray(dat["instance_classes"], dtype=np.int32)
        if "instance_classes" in dat
        else None
    )
    ismanual = np.asarray(dat["ismanual"], dtype=bool) if "ismanual" in dat else None

    model = dat.get("model_path", dat.get("model", "cpsam"))
    if model in (0, "0", None):
        model = "cpsam"
    else:
        model = str(model)

    recompute_masks = bool(dat.get("recompute_masks", False))
    if not recompute_masks and flows:
        try:
            recompute_masks = flows[0].shape[-3] == masks.shape[-2]
        except Exception:
            recompute_masks = False

    return SessionData(
        source_image=source_image,
        masks=masks,
        flows=flows,
        colors=colors,
        instance_classes=instance_classes,
        ismanual=ismanual,
        model=model,
        recompute_masks=recompute_masks,
        segmentation=SegmentationMetadata(
            flow_threshold=float(dat.get("flow_threshold", 0.4)),
            cellprob_threshold=float(dat.get("cellprob_threshold", 0.0)),
            diameter=dat.get("diameter"),
            niter=int(dat.get("niter", 200)),
            min_size=int(dat.get("min_size", 15)),
        ),
    )


def write_session(path: str | os.PathLike[str], session: SessionData | dict[str, Any]) -> str:
    """Write session data to a pickled `_seg.npy` file."""
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")

    payload = session if isinstance(session, dict) else session_to_pickle_dict(session)
    np.save(path, payload)
    return str(path)


def read_session(path: str | os.PathLike[str]) -> SessionData:
    """Read session data from a pickled `_seg.npy` file."""
    path = Path(path)
    loaded = np.load(path, allow_pickle=True)
    if getattr(loaded, "ndim", None) != 0:
        raise ValueError("Invalid _seg.npy file: expected pickled dict payload")
    dat = loaded.item()
    if not isinstance(dat, dict):
        raise ValueError("Invalid _seg.npy file: expected pickled dict payload")
    return session_from_pickle_dict(dat, path)
