"""Series discovery and navigation routes."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from cellpose.gui.core import series

from ..session_response import session_response
from ..schemas import SeriesDiscoverRequest, SeriesNavigateRequest, SessionResponse
from ..session import SESSIONS

router = APIRouter(tags=["series"])


@router.post("/series/discover")
def discover(request: SeriesDiscoverRequest) -> dict:
    folder = os.path.expanduser(request.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")
    try:
        dataset = series.build_series_dataset(
            folder,
            subfolder_template=request.subfolder_template,
            filename_template=request.filename_template,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "folder": folder,
        "record_count": len(dataset["records"]),
        "axes": {axis: values for axis, values in dataset.get("axes", {}).items()},
        "records": [
            {"index": i, "label": record["label"], "path": record["path"]}
            for i, record in enumerate(dataset["records"])
        ],
        "dataset": {
            "folder": dataset["folder"],
            "subfolder_template": dataset["subfolder_template"],
            "filename_template": dataset["filename_template"],
            "axes": dataset.get("axes", {}),
            "axis_index": dataset.get("axis_index", {}),
            "lookup": dataset.get("lookup", {}),
            "records": dataset["records"],
        },
    }


@router.post("/series/navigate", response_model=SessionResponse)
def navigate(request: SeriesNavigateRequest) -> SessionResponse:
    try:
        session = SESSIONS.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if session.series_metadata is None:
        raise HTTPException(status_code=400, detail="No series loaded on session")

    dataset = session.series_metadata.get("dataset")
    if dataset is None:
        raise HTTPException(status_code=400, detail="Series dataset missing")

    record = dataset["records"][request.record_index]
    from cellpose.io import imread_2D

    image = imread_2D(record["path"])
    new_session = SESSIONS.create(image=image, filename=record["path"])
    new_session.series_metadata = {
        "dataset": dataset,
        "record_index": request.record_index,
        "metadata": series.build_series_metadata(dataset, request.record_index),
    }
    return session_response(new_session)
