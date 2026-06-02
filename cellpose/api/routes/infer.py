"""ML inference routes for the stateless sidecar."""

from __future__ import annotations

import os

import numpy as np
from fastapi import APIRouter, HTTPException

from cellpose.io import imread_2D

from ..arrays import decode_array, encode_array, encode_optional
from ..schemas import InferRequest, InferResponse, RecomputeFlowsRequest, RecomputeResponse
from ..segmentation import recompute_from_flows, run_inference
from cellpose.gui.core.mask_ops import normalize_mask_dtype, renumber_masks

router = APIRouter(tags=["infer"])


def _encode_flows(flows: list[np.ndarray]) -> list[dict]:
    return [encode_array(flow) for flow in flows]


@router.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    if request.path:
        path = os.path.expanduser(request.path)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        try:
            image = imread_2D(path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif request.image is not None:
        image = decode_array(request.image.model_dump())
    else:
        raise HTTPException(status_code=400, detail="path or image required")

    try:
        result = run_inference(
            np.asarray(image),
            params=request.params,
            model_name=request.model_name,
            custom_model=request.custom_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return InferResponse(
        masks=encode_optional(result.masks.squeeze()),
        flows=_encode_flows(result.flows),
        ncells=result.ncells,
        recompute_masks=result.recompute_masks,
    )


@router.post("/recompute", response_model=RecomputeResponse)
def recompute(request: RecomputeFlowsRequest) -> RecomputeResponse:
    if not request.flows:
        raise HTTPException(status_code=400, detail="flows required")
    try:
        flows = [decode_array(flow.model_dump()) for flow in request.flows]
        masks = recompute_from_flows(flows, request.params)
        masks = renumber_masks(masks)
        masks = normalize_mask_dtype(masks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RecomputeResponse(
        masks=encode_optional(masks.squeeze()),
        ncells=int(masks.max()),
    )
