"""Build FastAPI application for sidecar routes."""

from __future__ import annotations

from fastapi import FastAPI

from .routes import health, infer, train


def create_app() -> FastAPI:
    app = FastAPI(title="Cellpose Sidecar", version="2.0.0")
    app.include_router(health.router)
    app.include_router(infer.router)
    app.include_router(train.router)
    return app
