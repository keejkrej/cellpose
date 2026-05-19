"""Health and model listing routes."""

from __future__ import annotations

from fastapi import APIRouter

from cellpose.version import version_str
from cellpose.models import MODEL_NAMES, get_user_models

from ..schemas import HealthResponse, ModelsResponse
from ..segmentation import _MODEL, model_device

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=version_str,
        model_loaded=_MODEL is not None,
        device=model_device() if _MODEL is not None else None,
    )


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(builtin=list(MODEL_NAMES), custom=get_user_models())
