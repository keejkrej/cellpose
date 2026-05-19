"""Export routes."""

from __future__ import annotations

import os

import numpy as np
from fastapi import APIRouter, HTTPException

from cellpose.io import imsave

from ..schemas import ExportMasksRequest
from ..session import SESSIONS

router = APIRouter(tags=["export"])


@router.post("/export/masks")
def export_masks(request: ExportMasksRequest) -> dict:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if session.masks is None:
        raise HTTPException(status_code=400, detail="No masks to export")

    path = os.path.expanduser(request.path)
    masks = session.masks.squeeze()
    if request.format == "tif":
        if not path.endswith((".tif", ".tiff")):
            path = os.path.splitext(path)[0] + "_cp_masks.tif"
        imsave(path, masks.astype(np.uint16))
    else:
        if not path.endswith(".png"):
            path = os.path.splitext(path)[0] + "_cp_masks.png"
        imsave(path, masks.astype(np.uint16))
    return {"path": path, "ncells": session.ncells}
