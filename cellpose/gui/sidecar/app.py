"""Build FastAPI application for sidecar routes."""

from __future__ import annotations

from fastapi import FastAPI

from .routes import export, export_extra, health, io, masks, preprocess, segment, series, series_suggest, train


def create_app() -> FastAPI:
    app = FastAPI(title="Cellpose Sidecar", version="1.0.0")
    app.include_router(health.router)
    app.include_router(segment.router)
    app.include_router(masks.router)
    app.include_router(preprocess.router)
    app.include_router(io.router)
    app.include_router(series.router)
    app.include_router(series_suggest.router)
    app.include_router(export.router)
    app.include_router(export_extra.router)
    app.include_router(train.router)
    return app
