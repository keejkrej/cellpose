"""I/O routes for images and _seg.npy files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fastapi import APIRouter, HTTPException

from cellpose.io import imread, imread_2D, imread_3D

from ..arrays import decode_array, encode_array
from ..mask_ops import apply_masks
from ..routes.segment import _session_response
from ..schemas import LoadImageRequest, LoadSegRequest, SaveSegRequest, SessionResponse
from ..segmentation import display_image_from_stack
from ..session import SESSIONS, SidecarSession

router = APIRouter(tags=["io"])


def _load_seg_data(path: str, load_3D: bool) -> SidecarSession:
    dat = np.load(path, allow_pickle=True).item()
    filename = dat.get("filename")
    image = None
    if filename and os.path.isfile(filename):
        image = imread_2D(filename) if not load_3D else imread_3D(filename)
    elif "img" in dat:
        image = dat["img"]
    else:
        raise ValueError("No image found in seg file or at filename path")

    session = SESSIONS.create(image=np.asarray(image), filename=filename or path)
    masks = dat.get("masks")
    if masks is not None:
        apply_masks(session, np.asarray(masks))
        session.outlines = dat.get("outlines")
        session.colors = dat.get("colors")
        session.instance_classes = dat.get("instance_classes")
        session.ismanual = dat.get("ismanual")
        session.flows = dat.get("flows")
        session.normalize_params = dat.get("normalize_params", {})
        session.restore = dat.get("restore")
        session.ratio = float(dat.get("ratio", 1.0))
        session.manual_changes = dat.get("manual_changes", [])
        session.zdraw = dat.get("zdraw", [])
        session.model_path = dat.get("model_path", 0)
        session.segmentation_params = {
            "flow_threshold": dat.get("flow_threshold", 0.4),
            "cellprob_threshold": dat.get("cellprob_threshold", 0.0),
            "diameter": dat.get("diameter"),
        }
        if session.flows and session.masks is not None and session.masks.ndim >= 2:
            session.recompute_masks = session.masks.shape[0] == 1
        if dat.get("img_restore") is not None:
            session.stack_filtered = dat["img_restore"]
    return session


@router.post("/io/load-image", response_model=SessionResponse)
def load_image(request: LoadImageRequest) -> SessionResponse:
    path = os.path.expanduser(request.path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        image = imread_2D(path) if not request.load_3D else imread_3D(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session = SESSIONS.create(image=np.asarray(image), filename=path)
    return _session_response(session)


@router.post("/io/load-seg", response_model=SessionResponse)
def load_seg(request: LoadSegRequest) -> SessionResponse:
    path = os.path.expanduser(request.path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        session = _load_seg_data(path, request.load_3D)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _session_response(session)


@router.post("/io/save-seg")
def save_seg(request: SaveSegRequest) -> dict:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if session.masks is None:
        raise HTTPException(status_code=400, detail="No masks to save")

    filename = request.path
    if filename is None:
        if not session.filename:
            raise HTTPException(status_code=400, detail="path required")
        filename = os.path.splitext(session.filename)[0] + "_seg.npy"
    else:
        filename = os.path.expanduser(filename)
        if not filename.endswith("_seg.npy"):
            filename = os.path.splitext(filename)[0] + "_seg.npy"

    params = session.segmentation_params
    dat = {
        "outlines": session.outlines.squeeze() if session.outlines is not None else None,
        "colors": session.colors,
        "masks": session.masks.squeeze(),
        "filename": session.filename,
        "flows": session.flows,
        "ismanual": session.ismanual,
        "manual_changes": session.manual_changes,
        "model_path": session.model_path,
        "flow_threshold": params.get("flow_threshold", 0.4),
        "cellprob_threshold": params.get("cellprob_threshold", 0.0),
        "normalize_params": session.normalize_params,
        "restore": session.restore,
        "ratio": session.ratio,
        "diameter": params.get("diameter"),
        "instance_classes": session.instance_classes,
    }
    if session.stack_filtered is not None:
        dat["img_restore"] = session.stack_filtered
    if session.series_metadata is not None:
        dat["image_series"] = session.series_metadata

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.save(filename, dat)
    return {"path": filename, "ncells": session.ncells}
