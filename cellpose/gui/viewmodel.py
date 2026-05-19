"""
View-model layer for the Cellpose GUI.

The Qt window stays responsible for rendering widgets and graphics items. This
object owns the mutable GUI model state and exposes view-friendly operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from PySide6 import QtCore

from .model import (
    InstanceClasses,
    PreprocessingParameters,
    SegmentationParameters,
    SeriesState,
    TrainingParameters,
)


class MainViewModel(QtCore.QObject):
    seriesChanged = QtCore.Signal(object)
    instanceClassesChanged = QtCore.Signal(object)
    trainingParametersChanged = QtCore.Signal(object)

    def __init__(self, model_save_folder: str, parent=None):
        super().__init__(parent)
        self._model_save_folder = model_save_folder
        self.series_state = SeriesState.empty()
        self.instances = InstanceClasses()
        self.training_params: dict[str, Any] = {}
        self.reset_training_parameters()

    def reset_training_parameters(self):
        defaults = TrainingParameters.create_default(self._model_save_folder)
        self.training_params.clear()
        self.training_params.update(defaults.to_dict())
        self.trainingParametersChanged.emit(self.training_params)
        return self.training_params

    def set_training_parameters(self, values: dict[str, Any]):
        values = dict(values)
        self.training_params.clear()
        self.training_params.update(values)
        self.trainingParametersChanged.emit(self.training_params)

    def reset_series(self) -> SeriesState:
        self.series_state = SeriesState.empty()
        self.seriesChanged.emit(self.series_state)
        return self.series_state

    def set_series(
        self, dataset: dict[str, Any] | None = None, record_index: int | None = None
    ) -> SeriesState:
        if dataset is None or record_index is None:
            return self.reset_series()
        self.series_state = SeriesState.from_record(dataset, record_index)
        self.seriesChanged.emit(self.series_state)
        return self.series_state

    def output_filename(self, fallback_filename):
        return self.series_state.output_filename or fallback_filename

    def get_segmentation_parameters(
        self,
        diameter: float,
        flow_threshold: float,
        cellprob_threshold: float,
        percentile_low: float,
        percentile_high: float,
        niter: int,
    ) -> SegmentationParameters:
        return SegmentationParameters.from_values(
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            niter=niter,
        )

    def get_preprocessing_parameters(
        self,
        sharpen_radius: float,
        smooth_radius: float,
        tile_norm_blocksize: float,
        tile_norm_smooth3D: float,
        norm3D: bool,
        image_shape: tuple[int, int] | None = None,
        invert: bool = False,
    ) -> PreprocessingParameters:
        return PreprocessingParameters.from_values(
            sharpen_radius=sharpen_radius,
            smooth_radius=smooth_radius,
            tile_norm_blocksize=tile_norm_blocksize,
            tile_norm_smooth3D=tile_norm_smooth3D,
            norm3D=norm3D,
            image_shape=image_shape,
            invert=invert,
        )

    def ensure_instance_classes(
        self, ncells: int, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        values = self.instances.ensure_size(ncells, current_values=current_values)
        self.instanceClassesChanged.emit(values)
        return values

    def set_instance_classes(
        self, ncells: int, values: np.ndarray | list[int] | None = None
    ) -> np.ndarray:
        result = self.instances.replace(ncells, values)
        self.instanceClassesChanged.emit(result)
        return result

    def set_instance_class(self, row: int, class_id: int) -> np.ndarray:
        result = self.instances.set_class(row, class_id)
        self.instanceClassesChanged.emit(result)
        return result

    def instance_class_filter(self, text: str) -> int | None:
        return InstanceClasses.parse_filter(text)

    def visible_cell_pixels(
        self, cellpix: np.ndarray, filter_class_id: int | None
    ) -> np.ndarray:
        return self.instances.visible_cell_pixels(cellpix, filter_class_id)


class ObservableVariable(QtCore.QObject):
    valueChanged = QtCore.Signal(object)

    def __init__(self, initial=None):
        super().__init__()
        self._value = initial

    def set(self, new_value):
        if new_value != self._value:
            self._value = new_value
            self.valueChanged.emit(new_value)

    def get(self):
        return self._value

    def __call__(self):
        return self._value

    def reset(self):
        self.set(0)

    def __iadd__(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Value must be numeric.")
        self.set(self._value + amount)
        return self

    def __radd__(self, other):
        return other + self._value

    def __add__(self, other):
        return other + self._value

    def __isub__(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Value must be numeric.")
        self.set(self._value - amount)
        return self

    def __str__(self):
        return str(self._value)

    def __lt__(self, x):
        return self._value < x

    def __gt__(self, x):
        return self._value > x

    def __eq__(self, x):
        return self._value == x
