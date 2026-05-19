"""Mask editing routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..mask_ops import add_mask_from_strokes, merge_cells, remove_cells
from ..routes.segment import _session_response
from ..schemas import AddMaskRequest, MergeCellsRequest, RemoveCellsRequest, SessionResponse
from ..session import SESSIONS

router = APIRouter(tags=["masks"])


@router.post("/masks/remove", response_model=SessionResponse)
def remove(request: RemoveCellsRequest) -> SessionResponse:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    remove_cells(session, request.cell_indices)
    return _session_response(session)


@router.post("/masks/add", response_model=SessionResponse)
def add(request: AddMaskRequest) -> SessionResponse:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    color = tuple(int(v) for v in request.color) if request.color else None
    idx = add_mask_from_strokes(session, request.strokes, color=color, class_id=request.class_id)
    if idx is None:
        raise HTTPException(status_code=400, detail="Cell too small to draw")
    return _session_response(session)


@router.post("/masks/merge", response_model=SessionResponse)
def merge(request: MergeCellsRequest) -> SessionResponse:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    merge_cells(session, request.source_index, request.target_index)
    return _session_response(session)
