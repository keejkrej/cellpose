"""
Presenter layer for the Cellpose GUI (MVP).

The presenter coordinates user actions between the passive Qt view and the
non-Qt model objects. It owns mutable GUI state and tells the view when to
refresh; the view forwards input events here instead of mutating model state
directly.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .model import (
    InstanceClasses,
    InstanceVisibility,
    PreprocessingParameters,
    SegmentationParameters,
    SeriesState,
    TrainingParameters,
)


class MainView(Protocol):
    """Surface the presenter needs from the main window."""

    filename: Any
    Ly: int
    Lx: int

    def sync_series_state(self, state: SeriesState) -> None: ...
    def set_series_navigation_state(
        self, dataset: dict[str, Any] | None = None, record_index: int | None = None
    ) -> None: ...
    def refresh_instance_table(self) -> None: ...
    def draw_layer(self) -> None: ...
    def update_layer(self) -> None: ...
    def read_segmentation_widgets(self) -> dict[str, Any]: ...
    def apply_segmentation_widgets(self, params: SegmentationParameters) -> None: ...
    def read_preprocessing_widgets(self) -> dict[str, Any]: ...
    def apply_preprocessing_widgets(self, params: PreprocessingParameters) -> None: ...
    def instance_class_filter_text(self) -> str: ...
    def ncells(self) -> int: ...


class MainPresenter:
    def __init__(self, view: MainView, model_save_folder: str):
        self._view = view
        self._model_save_folder = model_save_folder
        self.series_state = SeriesState.empty()
        self.instances = InstanceClasses()
        self.instance_visibility = InstanceVisibility()
        self.training_params: dict[str, Any] = {}
        self.reset_training_parameters()

    @property
    def instance_classes(self) -> np.ndarray:
        return self.instances.values

    @property
    def instance_visible(self) -> np.ndarray:
        return self.instance_visibility.values

    # ---- training ----

    def reset_training_parameters(self) -> dict[str, Any]:
        defaults = TrainingParameters.create_default(self._model_save_folder)
        self.training_params.clear()
        self.training_params.update(defaults.to_dict())
        return self.training_params

    def set_training_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        self.training_params.clear()
        self.training_params.update(dict(values))
        return self.training_params

    # ---- series ----

    def reset_series(self) -> SeriesState:
        self.series_state = SeriesState.empty()
        self._view.sync_series_state(self.series_state)
        self._view.set_series_navigation_state()
        return self.series_state

    def set_series(
        self, dataset: dict[str, Any] | None = None, record_index: int | None = None
    ) -> SeriesState:
        if dataset is None or record_index is None:
            return self.reset_series()
        self.series_state = SeriesState.from_record(dataset, record_index)
        self._view.sync_series_state(self.series_state)
        self._view.set_series_navigation_state(dataset, record_index)
        return self.series_state

    def output_filename(self, fallback_filename: str) -> str:
        return self.series_state.output_filename or fallback_filename

    # ---- parameters ----

    def segmentation_parameters(self) -> SegmentationParameters:
        widgets = self._view.read_segmentation_widgets()
        params = SegmentationParameters.from_values(
            diameter=float(widgets["diameter"]),
            flow_threshold=float(widgets["flow_threshold"]),
            cellprob_threshold=float(widgets["cellprob_threshold"]),
            percentile_low=float(widgets["percentile_low"]),
            percentile_high=float(widgets["percentile_high"]),
            niter=int(widgets["niter"]),
        )
        self._view.apply_segmentation_widgets(params)
        return params

    def segmentation_parameters_dict(self) -> dict[str, Any]:
        return self.segmentation_parameters().to_dict()

    def preprocessing_parameters(self) -> PreprocessingParameters:
        widgets = self._view.read_preprocessing_widgets()
        params = PreprocessingParameters.from_values(
            sharpen_radius=float(widgets["sharpen_radius"]),
            smooth_radius=float(widgets["smooth_radius"]),
            tile_norm_blocksize=float(widgets["tile_norm_blocksize"]),
            tile_norm_smooth3D=float(widgets["tile_norm_smooth3D"]),
            norm3D=bool(widgets["norm3D"]),
            image_shape=(self._view.Ly, self._view.Lx),
            invert=bool(widgets.get("invert", False)),
        )
        return params

    def preprocessing_parameters_dict(self) -> dict[str, Any]:
        return self.preprocessing_parameters().to_dict()

    def set_preprocessing_parameters(self, params: dict[str, Any]) -> None:
        model_params = PreprocessingParameters.from_values(
            sharpen_radius=float(params["sharpen_radius"]),
            smooth_radius=float(params["smooth_radius"]),
            tile_norm_blocksize=float(params["tile_norm_blocksize"]),
            tile_norm_smooth3D=float(params["tile_norm_smooth3D"]),
            norm3D=bool(params["norm3D"]),
            invert=bool(params.get("invert", False)),
        )
        self._view.apply_preprocessing_widgets(model_params)

    # ---- instances ----

    def ensure_instance_classes(
        self, ncells: int | None = None, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self._view.ncells()
        return self.instances.ensure_size(ncells, current_values=current_values)

    def set_instance_classes(
        self, values: np.ndarray | list[int] | None = None, ncells: int | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self._view.ncells()
        result = self.instances.replace(ncells, values)
        self._view.refresh_instance_table()
        return result

    def set_instance_class(self, row: int, class_id: int) -> np.ndarray:
        result = self.instances.set_class(row, class_id)
        self._view.refresh_instance_table()
        self._view.draw_layer()
        self._view.update_layer()
        return result

    def ensure_instance_visible(
        self, ncells: int | None = None, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self._view.ncells()
        return self.instance_visibility.ensure_size(
            ncells, current_values=current_values
        )

    def set_instance_visible(
        self, values: np.ndarray | list[bool] | None = None, ncells: int | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self._view.ncells()
        result = self.instance_visibility.replace(ncells, values)
        self._view.refresh_instance_table()
        return result

    def set_instance_visible_row(self, row: int, visible: bool) -> np.ndarray:
        result = self.instance_visibility.set_visible(row, visible)
        self._view.draw_layer()
        self._view.update_layer()
        return result

    def set_all_instance_visible(self, visible: bool, ncells: int | None = None) -> None:
        if ncells is None:
            ncells = self._view.ncells()
        if ncells == 0:
            return
        self.ensure_instance_visible(ncells)
        self.instance_visibility.values[:ncells] = visible
        self._view.draw_layer()
        self._view.update_layer()

    def remove_instance_metadata(self, row: int) -> tuple[int, bool]:
        """Remove class/visibility entries for a deleted cell index (0-based row)."""
        removed_class_id = 0
        removed_visible = True
        if row < len(self.instances.values):
            removed_class_id = int(self.instances.values[row])
            self.instances.values = np.delete(self.instances.values, row)
        if row < len(self.instance_visibility.values):
            removed_visible = bool(self.instance_visibility.values[row])
            self.instance_visibility.values = np.delete(
                self.instance_visibility.values, row
            )
        return removed_class_id, removed_visible

    def append_instance_metadata(
        self, class_id: int = 0, visible: bool = True
    ) -> None:
        self.instances.values = np.append(
            self.instances.values, np.int32(class_id)
        )
        self.instance_visibility.values = np.append(
            self.instance_visibility.values, visible
        )

    def reset_instance_metadata(self) -> None:
        self.instances.values = np.zeros(0, dtype=np.int32)
        self.instance_visibility.values = np.zeros(0, dtype=bool)

    def instance_class_filter(self) -> int | None:
        return InstanceClasses.parse_filter(self._view.instance_class_filter_text())

    def visible_cell_pixels(self, cellpix: np.ndarray) -> np.ndarray:
        self.ensure_instance_classes()
        self.ensure_instance_visible()
        return self.instances.visible_cell_pixels(
            cellpix, self.instance_class_filter(), self.instance_visibility.values
        )

    def on_instance_filter_changed(self) -> None:
        self._view.refresh_instance_table()
        self._view.draw_layer()
        self._view.update_layer()

    def on_instance_visibility_changed(self) -> None:
        self._view.draw_layer()
        self._view.update_layer()
