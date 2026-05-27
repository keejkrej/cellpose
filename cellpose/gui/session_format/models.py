"""Data models for `.cellpose` session archives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SESSION_FORMAT_VERSION = 1
SESSION_EXTENSION = ".cellpose"
SESSION_SUFFIX = "_seg.cellpose"


@dataclass
class SegmentationMetadata:
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    diameter: float | None = None
    niter: int = 200
    min_size: int = 15

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "diameter": self.diameter,
            "niter": self.niter,
            "min_size": self.min_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationMetadata:
        return cls(
            flow_threshold=float(data.get("flow_threshold", 0.4)),
            cellprob_threshold=float(data.get("cellprob_threshold", 0.0)),
            diameter=data.get("diameter"),
            niter=int(data.get("niter", 200)),
            min_size=int(data.get("min_size", 15)),
        )


@dataclass
class SessionData:
    source_image: str
    masks: np.ndarray
    flows: list[np.ndarray] | None = None
    colors: np.ndarray | None = None
    instance_classes: np.ndarray | None = None
    ismanual: np.ndarray | None = None
    model: str = "cpsam"
    recompute_masks: bool = False
    segmentation: SegmentationMetadata = field(default_factory=SegmentationMetadata)

    @property
    def ncells(self) -> int:
        return int(self.masks.max()) if self.masks.size else 0
