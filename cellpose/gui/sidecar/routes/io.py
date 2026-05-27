"""I/O routes for images and `.cellpose` session files."""

from __future__ import annotations

import os

import numpy as np

from fastapi import APIRouter, HTTPException

from cellpose.gui.session_format import default_session_path, read_session, write_session
from cellpose.gui.session_format.models import SegmentationMetadata, SessionData
from cellpose.io import imread_2D, imread_3D

from ..arrays import decode_array, encode_array
from ..mask_ops import apply_masks
from ..routes.segment import _session_response
from ..schemas import LoadImageRequest, LoadSegRequest, SaveSegRequest, SessionResponse
from ..segmentation import display_image_from_stack
from ..session import SESSIONS, SidecarSession

router = APIRouter(tags=["io"])


def _segmentation_metadata(params: dict) -> SegmentationMetadata:
    return SegmentationMetadata(
        flow_threshold=float(params.get("flow_threshold", 0.4)),
        cellprob_threshold=float(params.get("cellprob_threshold", 0.0)),
        diameter=params.get("diameter"),
        niter=int(params.get("niter", 200)),
        min_size=int(params.get("min_size", 15)),
    )


def _sidecar_to_session_data(session: SidecarSession) -> SessionData:
    if session.masks is None or not session.filename:
        raise ValueError("Nothing to save")

    model = str(session.model_path) if session.model_path else "cpsam"
    if model == "0":
        model = "cpsam"

    return SessionData(
        source_image=session.filename,
        masks=np.asarray(session.masks).squeeze(),
        flows=session.flows,
        colors=session.colors,
        instance_classes=session.instance_classes,
        ismanual=session.ismanual,
        model=model,
        recompute_masks=session.recompute_masks,
        segmentation=_segmentation_metadata(session.segmentation_params),
    )


def _load_cellpose_session(path: str, load_3D: bool) -> SidecarSession:
    session_data = read_session(path)
    image_path = session_data.source_image
    if not os.path.isfile(image_path):
        raise ValueError(f"Source image not found: {image_path}")

    image = imread_2D(image_path) if not load_3D else imread_3D(image_path)
    session = SESSIONS.create(image=np.asarray(image), filename=image_path)
    masks = np.asarray(session_data.masks)
    apply_masks(session, masks)
    session.colors = session_data.colors
    session.instance_classes = session_data.instance_classes
    session.ismanual = session_data.ismanual
    session.flows = session_data.flows
    session.recompute_masks = session_data.recompute_masks
    session.model_path = session_data.model
    session.segmentation_params = {
        "flow_threshold": session_data.segmentation.flow_threshold,
        "cellprob_threshold": session_data.segmentation.cellprob_threshold,
        "diameter": session_data.segmentation.diameter,
        "niter": session_data.segmentation.niter,
        "min_size": session_data.segmentation.min_size,
    }
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
        session = _load_cellpose_session(path, request.load_3D)
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
        filename = default_session_path(session.filename)
    else:
        filename = os.path.expanduser(filename)
        if not filename.endswith(".cellpose"):
            filename = default_session_path(filename)

    try:
        session_data = _sidecar_to_session_data(session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    written = write_session(filename, session_data)
    return {"path": written, "ncells": session.ncells}
