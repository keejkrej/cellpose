"""Segmentation routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..arrays import decode_array, encode_array, encode_optional
from ..schemas import (
    ArrayPayload,
    RecomputeRequest,
    SegmentRequest,
    SessionResponse,
)
from ..segmentation import display_image_from_stack, recompute_masks, run_segmentation
from ..session import SESSIONS

router = APIRouter(tags=["segment"])


def _session_response(session) -> SessionResponse:
    image = session.stack_filtered if session.stack_filtered is not None else session.image
    return SessionResponse(
        session_id=session.session_id,
        filename=session.filename,
        shape=list(session.image.shape),
        ncells=session.ncells,
        masks=encode_optional(session.masks.squeeze() if session.masks is not None else None),
        outlines=encode_optional(
            session.outlines.squeeze() if session.outlines is not None else None
        ),
        display_image=encode_array(display_image_from_stack(image)),
        colors=encode_optional(session.colors),
        instance_classes=encode_optional(session.instance_classes),
        flows=[encode_array(flow) for flow in session.flows] if session.flows else None,
        recompute_masks=session.recompute_masks,
        metadata={
            "normalize_params": session.normalize_params,
            "segmentation_params": session.segmentation_params,
            "restore": session.restore,
            "ratio": session.ratio,
        },
    )


@router.post("/segment", response_model=SessionResponse)
def segment(request: SegmentRequest) -> SessionResponse:
    if request.session_id:
        try:
            session = SESSIONS.get(request.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    elif request.image is not None:
        image = decode_array(request.image.model_dump())
        session = SESSIONS.create(image=image, filename=request.filename)
    else:
        raise HTTPException(status_code=400, detail="session_id or image required")

    try:
        run_segmentation(
            session,
            params=request.params,
            preprocess=request.preprocess,
            model_name=request.model_name,
            custom_model=request.custom_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _session_response(session)


@router.post("/recompute-masks", response_model=SessionResponse)
def recompute(request: RecomputeRequest) -> SessionResponse:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        recompute_masks(session, request.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _session_response(session)


@router.get("/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    try:
        session = SESSIONS.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _session_response(session)
