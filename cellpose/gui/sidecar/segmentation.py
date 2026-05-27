"""Stateless segmentation and recompute helpers for the ML sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cellpose import dynamics
from cellpose.models import CellposeModel, normalize_default
from cellpose.gui.mask_ops import normalize_mask_dtype, renumber_masks
from cellpose.transforms import normalize99, resize_image

from .schemas import SegmentationParams

_MODEL: CellposeModel | None = None
_MODEL_NAME: str | None = None


class ProgressTracker:
    def __init__(self):
        self.value = 0

    def setValue(self, value: int) -> None:
        self.value = int(value)


@dataclass
class InferResult:
    masks: np.ndarray
    flows: list[np.ndarray]
    ncells: int
    recompute_masks: bool


def get_model(model_name: str | None = None, custom: bool = False) -> CellposeModel:
    global _MODEL, _MODEL_NAME
    resolved = model_name or "cpsam"
    if custom and model_name:
        resolved = model_name
    if _MODEL is not None and _MODEL_NAME == resolved:
        return _MODEL
    if custom and model_name:
        from pathlib import Path

        from cellpose.models import MODEL_DIR

        model_path = Path(MODEL_DIR) / "custom" / model_name
        _MODEL = CellposeModel(pretrained_model=str(model_path))
    else:
        _MODEL = CellposeModel(model_type="cpsam")
    _MODEL_NAME = resolved
    return _MODEL


def model_device() -> str:
    model = get_model()
    return str(model.device)


def build_normalize_params(
    params: SegmentationParams,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    normalize_params = dict(normalize_default)
    normalize_params.update(
        {
            "percentile": [params.percentile_low, params.percentile_high],
        }
    )
    if image_shape is not None:
        ly, lx = image_shape
        if normalize_params["tile_norm_blocksize"] > ly and normalize_params["tile_norm_blocksize"] > lx:
            normalize_params["tile_norm_blocksize"] = 0
    return normalize_params


def run_inference(
    image: np.ndarray,
    params: SegmentationParams,
    model_name: str | None = None,
    custom_model: bool = False,
) -> InferResult:
    model = get_model(model_name=model_name, custom=custom_model)
    progress = ProgressTracker()
    image = np.asarray(image)
    normalize_params = build_normalize_params(
        params, image_shape=image.shape[-2:]
    )

    data = np.squeeze(image.copy())
    do_3D = params.do_3D and params.stitch_threshold <= 0.0

    masks, flows = model.eval(
        data,
        diameter=params.diameter if params.diameter and params.diameter > 0 else None,
        cellprob_threshold=params.cellprob_threshold,
        flow_threshold=params.flow_threshold,
        do_3D=do_3D,
        niter=params.niter,
        normalize=normalize_params,
        stitch_threshold=params.stitch_threshold,
        anisotropy=params.anisotropy,
        flow3D_smooth=params.flow3D_smooth,
        min_size=params.min_size,
        channel_axis=-1,
        progress=progress,
        z_axis=0 if data.ndim > 3 else None,
    )[:2]

    flows_new: list[np.ndarray] = []
    flows_new.append(flows[0].copy())
    flows_new.append(
        (np.clip(normalize99(flows[2].copy()), 0, 1) * 255).astype(np.uint8)
    )
    flows_new.append(flows[1].copy())
    flows_new.append(flows[2].copy())

    ly, lx = image.shape[-2], image.shape[-1]
    if flows_new[0].shape[-3:-1] != (ly, lx):
        resized = []
        for flow in flows_new:
            resized.append(
                resize_image(flow, Ly=ly, Lx=lx, interpolation=cv2.INTER_NEAREST)
            )
        flows_new = resized

    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
        flows_new = [flow[np.newaxis, ...] for flow in flows_new]

    masks = renumber_masks(masks)
    masks = normalize_mask_dtype(masks)
    recompute_masks = not do_3D and params.stitch_threshold <= 0.0
    return InferResult(
        masks=masks,
        flows=flows_new,
        ncells=int(masks.max()),
        recompute_masks=recompute_masks,
    )


def recompute_from_flows(
    flows: list[np.ndarray],
    params: SegmentationParams,
) -> np.ndarray:
    if len(flows) < 4:
        raise ValueError("Expected at least 4 flow arrays")
    dP = flows[2].squeeze()
    cellprob = flows[3].squeeze()
    maski = dynamics.resize_and_compute_masks(
        dP=dP,
        cellprob=cellprob,
        niter=params.niter,
        do_3D=params.do_3D,
        min_size=params.min_size,
        cellprob_threshold=params.cellprob_threshold,
        flow_threshold=params.flow_threshold,
    )
    if maski.ndim < 3:
        maski = maski[np.newaxis, ...]
    return maski
