"""In-memory session store for sidecar segmentation state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SidecarSession:
    session_id: str
    image: np.ndarray
    filename: str | None = None
    masks: np.ndarray | None = None
    outlines: np.ndarray | None = None
    flows: list[np.ndarray] | None = None
    colors: np.ndarray | None = None
    instance_classes: np.ndarray | None = None
    ismanual: np.ndarray | None = None
    normalize_params: dict[str, Any] = field(default_factory=dict)
    segmentation_params: dict[str, Any] = field(default_factory=dict)
    restore: str | None = None
    ratio: float = 1.0
    model_path: str | int = 0
    manual_changes: list[Any] = field(default_factory=list)
    zdraw: list[Any] = field(default_factory=list)
    stack_filtered: np.ndarray | None = None
    series_metadata: dict[str, Any] | None = None
    recompute_masks: bool = False

    @property
    def ncells(self) -> int:
        if self.masks is None:
            return 0
        return int(self.masks.max())

    @property
    def nz(self) -> int:
        if self.masks is not None:
            return int(self.masks.shape[0]) if self.masks.ndim == 3 else 1
        if self.image.ndim == 2:
            return 1
        if self.image.ndim == 3 and self.image.shape[-1] in (1, 2, 3, 4):
            return 1
        return int(self.image.shape[0])


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, SidecarSession] = {}

    def create(
        self,
        image: np.ndarray,
        filename: str | None = None,
    ) -> SidecarSession:
        session_id = str(uuid.uuid4())
        session = SidecarSession(
            session_id=session_id,
            image=image,
            filename=filename,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SidecarSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session: {session_id}")
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


SESSIONS = SessionStore()
