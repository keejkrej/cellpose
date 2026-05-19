"""
Model objects for the Cellpose GUI.

These classes intentionally avoid Qt widgets. They hold GUI state and perform
the small validation/coercion steps that are independent of the concrete view.
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import series


@dataclass
class SeriesState:
    dataset: dict[str, Any] | None = None
    record_index: int | None = None
    output_filename: str | None = None
    display_filename: str | None = None
    filename: str | None = None

    @classmethod
    def empty(cls) -> "SeriesState":
        return cls()

    @classmethod
    def from_record(cls, dataset: dict[str, Any], record_index: int) -> "SeriesState":
        record = dataset["records"][record_index]
        return cls(
            dataset=dataset,
            record_index=record_index,
            output_filename=series.get_output_filename(dataset, record_index),
            display_filename=record["label"],
            filename=record["path"],
        )


@dataclass
class TrainingParameters:
    model_index: int = 0
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    n_epochs: int = 100
    model_name: str = ""
    train_data_folder: str = ""
    model_save_folder: str = ""

    @classmethod
    def create_default(
        cls, model_save_folder: str, now: _datetime.datetime | None = None
    ) -> "TrainingParameters":
        now = now or _datetime.datetime.now()
        return cls(
            model_name="cpsam" + now.strftime("_%Y%m%d_%H%M%S"),
            model_save_folder=model_save_folder,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_index": int(self.model_index),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "n_epochs": int(self.n_epochs),
            "model_name": str(self.model_name),
            "train_data_folder": str(self.train_data_folder),
            "model_save_folder": str(self.model_save_folder),
        }


@dataclass(frozen=True)
class SegmentationParameters:
    diameter: float | None
    flow_threshold: float
    cellprob_threshold: float
    percentile: tuple[float, float]
    niter: int

    @classmethod
    def from_values(
        cls,
        diameter: float,
        flow_threshold: float,
        cellprob_threshold: float,
        percentile_low: float,
        percentile_high: float,
        niter: int,
    ) -> "SegmentationParameters":
        diameter_value = float(diameter)
        low = float(percentile_low)
        high = float(percentile_high)
        if not 0 <= low <= 100 or not 0 <= high <= 100 or low >= high:
            raise ValueError(
                "normalization percentile range must be 0 <= low < high <= 100"
            )
        niter_value = int(niter)
        if niter_value < 1:
            niter_value = 200
        return cls(
            diameter=diameter_value if diameter_value > 0 else None,
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
            percentile=(low, high),
            niter=niter_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "percentile": self.percentile,
            "niter": self.niter,
        }


@dataclass(frozen=True)
class PreprocessingParameters:
    sharpen_radius: float
    smooth_radius: float
    tile_norm_blocksize: float
    tile_norm_smooth3D: float
    norm3D: bool
    invert: bool = False

    @classmethod
    def from_values(
        cls,
        sharpen_radius: float,
        smooth_radius: float,
        tile_norm_blocksize: float,
        tile_norm_smooth3D: float,
        norm3D: bool,
        image_shape: tuple[int, int] | None = None,
        invert: bool = False,
    ) -> "PreprocessingParameters":
        blocksize = max(0, float(tile_norm_blocksize))
        if image_shape is not None:
            ly, lx = image_shape
            if blocksize > ly and blocksize > lx:
                print(
                    "GUI_ERROR: tile size (tile_norm) bigger than both image "
                    "dimensions, disabling"
                )
                blocksize = 0
        return cls(
            sharpen_radius=max(0, float(sharpen_radius)),
            smooth_radius=max(0, float(smooth_radius)),
            tile_norm_blocksize=blocksize,
            tile_norm_smooth3D=max(0, float(tile_norm_smooth3D)),
            norm3D=bool(norm3D),
            invert=bool(invert),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sharpen_radius": self.sharpen_radius,
            "smooth_radius": self.smooth_radius,
            "tile_norm_blocksize": self.tile_norm_blocksize,
            "tile_norm_smooth3D": self.tile_norm_smooth3D,
            "norm3D": self.norm3D,
            "invert": self.invert,
        }


@dataclass
class InstanceClasses:
    values: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )

    def ensure_size(
        self, ncells: int, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        values = self.values if current_values is None else current_values
        values = np.asarray(values, dtype=np.int32).ravel()
        values = np.maximum(values, 0)
        if len(values) < ncells:
            pad = np.zeros(ncells - len(values), dtype=np.int32)
            values = np.concatenate((values, pad))
        elif len(values) > ncells:
            values = values[:ncells]
        self.values = values
        return self.values

    def replace(
        self, ncells: int, values: np.ndarray | list[int] | None = None
    ) -> np.ndarray:
        result = np.zeros(ncells, dtype=np.int32)
        if values is not None:
            loaded = np.asarray(values, dtype=np.int32).ravel()
            loaded = np.maximum(loaded, 0)
            n = min(ncells, len(loaded))
            result[:n] = loaded[:n]
        self.values = result
        return self.values

    def set_class(self, row: int, class_id: int) -> np.ndarray:
        if class_id < 0:
            raise ValueError("class_id must be non-negative")
        if row >= len(self.values):
            return self.values
        self.values[row] = int(class_id)
        return self.values

    @staticmethod
    def parse_filter(text: str) -> int | None:
        text = text.strip()
        if text == "":
            return None
        try:
            class_id = int(text)
        except ValueError:
            return None
        return class_id if class_id >= 0 else None

    def visible_cell_pixels(
        self, cellpix: np.ndarray, filter_class_id: int | None
    ) -> np.ndarray:
        if filter_class_id is None:
            return cellpix > 0

        max_label = int(cellpix.max())
        visible_labels = np.zeros(max_label + 1, dtype=bool)
        nlabels = min(max_label, len(self.values))
        if nlabels > 0:
            visible_labels[1 : nlabels + 1] = (
                self.values[:nlabels] == filter_class_id
            )
        return visible_labels[cellpix]
