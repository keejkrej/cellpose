"""Preprocessing routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..session_response import session_response
from ..schemas import PreprocessParams, SessionResponse
from ..segmentation import apply_preprocessing
from ..session import SESSIONS

router = APIRouter(tags=["preprocess"])


@router.post("/preprocess/{session_id}", response_model=SessionResponse)
def preprocess(session_id: str, params: PreprocessParams) -> SessionResponse:
    try:
        session = SESSIONS.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        apply_preprocessing(session, params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return session_response(session)
