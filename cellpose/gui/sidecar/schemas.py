"""Pydantic schemas for sidecar API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArrayPayload(BaseModel):
    dtype: str
    shape: list[int]
    data_b64: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    device: str | None = None


class ModelsResponse(BaseModel):
    builtin: list[str]
    custom: list[str]


class SegmentationParams(BaseModel):
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    niter: int = 200
    min_size: int = 15
    stitch_threshold: float = 0.0
    anisotropy: float = 1.0
    flow3D_smooth: float = 0.0
    do_3D: bool = False


class InferRequest(BaseModel):
    path: str | None = None
    image: ArrayPayload | None = None
    load_3D: bool = False
    model_name: str | None = None
    custom_model: bool = False
    params: SegmentationParams = Field(default_factory=SegmentationParams)


class InferResponse(BaseModel):
    masks: ArrayPayload | None = None
    flows: list[ArrayPayload] = Field(default_factory=list)
    ncells: int = 0
    recompute_masks: bool = False


class RecomputeFlowsRequest(BaseModel):
    flows: list[ArrayPayload]
    params: SegmentationParams = Field(default_factory=SegmentationParams)


class RecomputeResponse(BaseModel):
    masks: ArrayPayload | None = None
    ncells: int = 0


class SessionResponse(BaseModel):
    session_id: str
    filename: str | None = None
    shape: list[int]
    ncells: int
    masks: ArrayPayload | None = None
    outlines: ArrayPayload | None = None
    display_image: ArrayPayload | None = None
    colors: ArrayPayload | None = None
    instance_classes: ArrayPayload | None = None
    flows: list[ArrayPayload] | None = None
    recompute_masks: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RemoveCellsRequest(BaseModel):
    session_id: str
    cell_indices: list[int]


class AddMaskRequest(BaseModel):
    session_id: str
    strokes: list[list[list[float]]]
    color: list[float] | None = None
    class_id: int = 0


class MergeCellsRequest(BaseModel):
    session_id: str
    source_index: int
    target_index: int


class LoadImageRequest(BaseModel):
    path: str
    load_3D: bool = False


class LoadSegRequest(BaseModel):
    path: str
    load_3D: bool = False


class SaveSegRequest(BaseModel):
    session_id: str
    path: str | None = None


class ExportMasksRequest(BaseModel):
    session_id: str
    path: str
    format: str = "png"


class SeriesDiscoverRequest(BaseModel):
    folder: str
    subfolder_template: str = ""
    filename_template: str = "img_{t}_{c}_{z}.jpg"


class SeriesNavigateRequest(BaseModel):
    session_id: str
    record_index: int


class TrainRequest(BaseModel):
    train_data_folder: str
    model_name: str
    model_index: int = 0
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    n_epochs: int = 100
    model_save_folder: str | None = None


class AddModelRequest(BaseModel):
    path: str


class RemoveModelRequest(BaseModel):
    model_name: str
