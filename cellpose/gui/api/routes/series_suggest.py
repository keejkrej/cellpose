"""Series template suggestion route."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cellpose.gui.core import series

router = APIRouter(tags=["series"])


class SeriesSuggestRequest(BaseModel):
    folder: str


@router.post("/series/suggest")
def suggest(request: SeriesSuggestRequest) -> dict:
    folder = os.path.expanduser(request.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")
    suggestion = series.suggest_series_templates(folder)
    return {
        "subfolder_template": suggestion["subfolder_template"],
        "filename_template": suggestion["filename_template"],
    }
