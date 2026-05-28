"""
Presenter-facing view contract for the Cellpose GUI MVP stack.

MainView implements this protocol: widget I/O, canvas rendering, and chrome only.
Session state lives on MainModel; MainPresenter reads/writes the model directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .model import SegmentationParameters, SeriesState


@dataclass
class LabelRow:
    """Display row for the labels table."""

    roi: int
    class_id: int
    visible: bool
    hidden_by_filter: bool
    selected: bool


@dataclass
class SeriesNavViewState:
    """Slider positions for series navigation widgets."""

    enabled: bool
    axis_ranges: dict[str, tuple[int, int]]
    axis_values: dict[str, int]


class MainViewProtocol(Protocol):
    """Bare-minimum surface the presenter may call on MainView."""

    colormap: np.ndarray
    logger: Any

    # ---- widget input (read) ----

    def read_segmentation_widgets(self) -> dict[str, Any]: ...

    def read_labels_class_filter(self) -> str: ...

    def read_default_class_id(self) -> int: ...

    def read_selected_model(self) -> tuple[str, bool]: ...

    def read_inference_options(self) -> dict[str, Any]: ...

    def read_series_slider_axes(self) -> dict[str, int]: ...

    def read_saturation_range(self) -> tuple[float, float]: ...

    def read_view_mode_index(self) -> int: ...

    # ---- widget output (write) ----

    def apply_segmentation_widgets(self, params: SegmentationParameters) -> None: ...

    def set_progress(self, value: int) -> None: ...

    def show_message(self, title: str, text: str) -> None: ...

    def set_window_title(self, title: str) -> None: ...

    def set_ncells_count(self, n: int) -> None: ...

    def apply_series_labels(self, state: SeriesState) -> None: ...

    def set_series_navigation(self, nav: SeriesNavViewState) -> None: ...

    def set_view_mode(self, index: int, restored_enabled: bool) -> None: ...

    def set_mask_action_enabled(self, enabled: bool) -> None: ...

    def set_run_enabled(self, enabled: bool) -> None: ...

    def set_loaded_chrome(self, enabled: bool) -> None: ...

    def refresh_labels_table(
        self,
        rows: list[LabelRow] | None = None,
        header_state: str | None = None,
    ) -> None: ...

    def set_series_slider_value(self, axis_name: str, value: int) -> None: ...

    def is_updating_series_navigation(self) -> bool: ...

    def set_updating_series_navigation(self, updating: bool) -> None: ...

    def progress_widget(self) -> Any: ...

    # ---- canvas rendering ----

    def render_image_plane(
        self,
        image: np.ndarray,
        levels: list[float] | tuple[float, float],
        lut: np.ndarray | None = None,
    ) -> None: ...

    def render_mask_overlay(self, layerz: np.ndarray) -> None: ...

    def render_diameter_scale(self, radii: np.ndarray) -> None: ...

    def render_selection_boxes(
        self, bounds_list: list[tuple[int, int, int, int]]
    ) -> None: ...

    def render_rect_select_preview(
        self, bounds: tuple[int, int, int, int] | None
    ) -> None: ...

    def sync_saturation_slider(self, low: float, high: float) -> None: ...

    def refresh_plot_from_model(self) -> None: ...

    def refresh_scale_from_model(self) -> None: ...

    def show_window(self) -> None: ...

    def set_redo_enabled(self, enabled: bool) -> None: ...

    def set_undo_enabled(self, enabled: bool) -> None: ...

    def apply_segmentation_metadata_widgets(self, segmentation: Any) -> None: ...

    def set_model_list(self, models: list[str], current: str | None = None) -> None: ...
