"""Additional export routes."""

from __future__ import annotations

import os

import numpy as np
from fastapi import APIRouter, HTTPException

from cellpose.io import imsave, outlines_to_text, save_rois
from cellpose.utils import outlines_list

from ..schemas import ExportMasksRequest
from ..session import SESSIONS

router = APIRouter(tags=["export"])


class ExportPathRequest(ExportMasksRequest):
    pass


@router.post("/export/outlines")
def export_outlines(request: ExportPathRequest) -> dict:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if session.masks is None:
        raise HTTPException(status_code=400, detail="No masks to export")

    path = os.path.expanduser(request.path)
    if not path.endswith(".txt"):
        path = os.path.splitext(path)[0] + ".txt"
    outlines = outlines_list(session.masks.squeeze())
    outlines_to_text(os.path.splitext(path)[0], outlines)
    return {"path": path}


@router.post("/export/flows")
def export_flows(request: ExportPathRequest) -> dict:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not session.flows:
        raise HTTPException(status_code=400, detail="No flows to export")

    base = os.path.splitext(os.path.expanduser(request.path))[0]
    imsave(base + "_cp_cellprob.tif", session.flows[1])
    imsave(base + "_cp_flows.tif", session.flows[0])
    return {"path": base + "_cp_flows.tif"}


@router.post("/export/rois")
def export_rois(request: ExportPathRequest) -> dict:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if session.masks is None:
        raise HTTPException(status_code=400, detail="No masks to export")
    if session.masks.squeeze().ndim != 2:
        raise HTTPException(status_code=400, detail="ROI export supports 2D only")

    path = os.path.expanduser(request.path)
    if not path.endswith(".zip"):
        path = os.path.splitext(path)[0] + "_rois.zip"
    save_rois(session.masks.squeeze(), os.path.splitext(path)[0])
    return {"path": path}
