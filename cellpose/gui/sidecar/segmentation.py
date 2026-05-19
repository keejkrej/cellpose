"""Segmentation and preprocessing helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cellpose import dynamics, transforms
from cellpose.models import CellposeModel, MODEL_NAMES, normalize_default
from cellpose.transforms import normalize99, resize_image

from .mask_ops import apply_masks
from .schemas import PreprocessParams, SegmentationParams
from .session import SidecarSession


_MODEL: CellposeModel | None = None
_MODEL_NAME: str | None = None


class ProgressTracker:
    def __init__(self):
        self.value = 0

    def setValue(self, value: int) -> None:
        self.value = int(value)


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
    preprocess: PreprocessParams,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    params = dict(normalize_default)
    params.update(
        {
            "sharpen_radius": preprocess.sharpen_radius,
            "smooth_radius": preprocess.smooth_radius,
            "tile_norm_blocksize": preprocess.tile_norm_blocksize,
            "tile_norm_smooth3D": preprocess.tile_norm_smooth3D,
            "norm3D": preprocess.norm3D,
            "invert": preprocess.invert,
            "percentile": [preprocess.percentile_low, preprocess.percentile_high],
        }
    )
    if image_shape is not None:
        ly, lx = image_shape
        if params["tile_norm_blocksize"] > ly and params["tile_norm_blocksize"] > lx:
            params["tile_norm_blocksize"] = 0
    return params


def display_image_from_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack)
    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 2, 3, 4):
            img = arr[..., :3]
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
        else:
            img = arr[0]
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
    else:
        img = arr[0]
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
    img = img.astype(np.float32)
    if img.ndim == 2:
        lo, hi = np.percentile(img, (1, 99))
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0, 1)
        img = np.stack([img, img, img], axis=-1)
    else:
        for c in range(min(3, img.shape[-1])):
            channel = img[..., c]
            lo, hi = np.percentile(channel, (1, 99))
            if hi > lo:
                img[..., c] = np.clip((channel - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8)


def run_segmentation(
    session: SidecarSession,
    params: SegmentationParams,
    preprocess: PreprocessParams,
    model_name: str | None = None,
    custom_model: bool = False,
) -> None:
    model = get_model(model_name=model_name, custom=custom_model)
    progress = ProgressTracker()
    normalize_params = build_normalize_params(
        preprocess, image_shape=session.image.shape[-2:]
    )
    session.normalize_params = normalize_params
    session.segmentation_params = params.model_dump()

    data = session.stack_filtered.copy() if session.stack_filtered is not None else session.image.copy()
    data = np.squeeze(data)
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

    ly, lx = session.image.shape[-2], session.image.shape[-1]
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

    session.flows = flows_new
    session.recompute_masks = not do_3D and params.stitch_threshold <= 0.0
    apply_masks(session, masks)


def recompute_masks(session: SidecarSession, params: SegmentationParams) -> None:
    if not session.recompute_masks or session.flows is None:
        raise ValueError("Flows not available for recompute")
    dP = session.flows[2].squeeze()
    cellprob = session.flows[3].squeeze()
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
    apply_masks(session, maski)


def apply_preprocessing(session: SidecarSession, preprocess: PreprocessParams) -> np.ndarray:
    normalize_params = build_normalize_params(
        preprocess, image_shape=session.image.shape[-2:]
    )
    data = transforms.convert_image(session.image, channel_axis=-1, do_3D=False)
    percentile = normalize_params.get("percentile") or [
        preprocess.percentile_low,
        preprocess.percentile_high,
    ]
    filtered = transforms.normalize_img(
        data,
        normalize=normalize_params.get("normalize", True),
        norm3D=normalize_params.get("norm3D", True),
        invert=normalize_params.get("invert", False),
        lowhigh=normalize_params.get("lowhigh"),
        percentile=tuple(percentile),
        sharpen_radius=normalize_params.get("sharpen_radius", 0),
        smooth_radius=normalize_params.get("smooth_radius", 0),
        tile_norm_blocksize=normalize_params.get("tile_norm_blocksize", 0),
        tile_norm_smooth3D=normalize_params.get("tile_norm_smooth3D", 1),
    )
    session.stack_filtered = filtered
    session.normalize_params = normalize_params
    session.restore = "filter"
    return filtered
