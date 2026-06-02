"""
Qt view layer for the Cellpose GUI (MVP).

MainView owns widgets and rendering only; the presenter wires signals in start().

Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer,
Michael Rariden and Marius Pachitariu.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..utils.qt import QtCore, QtGui
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree

from ... import version
from ..utils import series
from .model import SegmentationParameters, SeriesState
from .widgets import (
    CheckBoxHeader,
    ImageDraw,
    ObservableVariable,
    SeriesAxisSlider,
    Slider,
    ViewBoxNoRightDrag,
    as_gray_image,
    brush_cursor,
    make_bwr,
    select_cursor,
)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False



@dataclass
class LabelRow:
    """Display row for the labels table."""

    roi: int
    class_id: int
    major_diameter: float | None
    minor_diameter: float | None
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
class MainView(QMainWindow):
    """Passive Qt view for the Cellpose GUI MVP stack."""

    files_dropped = QtCore.Signal(list)
    rect_select_press = QtCore.Signal(object, object)
    rect_select_move = QtCore.Signal(object)
    rect_select_release = QtCore.Signal(object, object)
    plot_double_clicked = QtCore.Signal()
    saturation_changed = QtCore.Signal(str)

    def __init__(self, logger=None):
        super().__init__()

        self.logger = logger
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(100, 100, 1280, 720)
        self.setWindowTitle(f"cellpose v{version}")

        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
        icon_path = str(icon_path.resolve())
        for size in (16, 24, 32, 48, 64, 256):
            app_icon.addFile(icon_path, QtCore.QSize(size, size))
        self.setWindowIcon(app_icon)

        self.imask = 0
        self.rect_select_mode = False
        self._rect_select_start = None
        self._rect_select_dragging = False
        self._rect_select_drag_threshold = 4
        self._canvas_loaded = False
        self.scale_on = False
        self.view = 0
        self.ly = 512
        self.lx = 512
        self.min_size = 15
        self.model_strings = []
        self.last_series_subfolder_template = ""
        self.last_series_filename_template = ""
        self._updating_series_navigation = False
        self._refreshing_labels_table = False
        self._syncing_labels_table_selection = False

        self.cwidget = QWidget(self)
        self.lmain = QHBoxLayout()
        self.cwidget.setLayout(self.lmain)
        self.setCentralWidget(self.cwidget)
        self.lmain.setContentsMargins(0, 0, 0, 10)

        self.left_sidebar = QGridLayout()
        self.left_sidebar_widget = QWidget(self)
        self.left_sidebar_widget.setLayout(self.left_sidebar)
        self.right_sidebar = QGridLayout()
        self.right_sidebar_widget = QWidget(self)
        self.right_sidebar_widget.setLayout(self.right_sidebar)
        b = self.make_buttons()

        self.win = pg.GraphicsLayoutWidget()
        self.lmain.addWidget(self.left_sidebar_widget, 1)
        self.lmain.addWidget(self.win, 2)
        self.lmain.addWidget(self.right_sidebar_widget, 1)

        self.make_viewbox()
        viewport = self._canvas_viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        self.apply_theme()

        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        if MATPLOTLIB:
            self.colormap = (
                plt.get_cmap("gist_ncar")(np.linspace(0.0, 0.9, 1000000)) * 255
            ).astype(np.uint8)
            np.random.seed(42)
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42)
            self.colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(
                np.uint8
            )

    def set_canvas_controller(self, controller) -> None:
        if hasattr(self.layer, "set_controller"):
            self.layer.set_controller(controller)
        else:
            self.layer.controller = controller

    def apply_theme(self, dark: bool | None = None) -> None:
        if dark is None:
            from .theme import is_dark_mode
            from qtpy.QtWidgets import QApplication

            app = QApplication.instance()
            dark = is_dark_mode(app) if app is not None else False
        self.win.setBackground("default")

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.Type.ThemeChange:
            from .theme import apply_app_theme, apply_pyqtgraph_theme, is_dark_mode
            from qtpy.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                apply_app_theme(app)
                dark = is_dark_mode(app)
                apply_pyqtgraph_theme(dark)
                self.apply_theme(dark)
        super().changeEvent(event)

    def level_change(self, name):
        self.saturation_changed.emit(name)

    # ---- MainViewProtocol: widget input ----

    def read_segmentation_widgets(self):
        return {
            "diameter": self.seg_param_root.param("diameter").value(),
            "flow_threshold": self.seg_param_root.param("flow_threshold").value(),
            "cellprob_threshold": self.seg_param_root.param("cellprob_threshold").value(),
            "percentile_low": self.seg_param_root.param("norm_percentile_low").value(),
            "percentile_high": self.seg_param_root.param("norm_percentile_high").value(),
            "niter": self.seg_param_root.param("niter").value(),
        }

    def read_labels_class_filter(self) -> str:
        if not hasattr(self, "LabelsClassFilter"):
            return ""
        return self.LabelsClassFilter.text()

    def read_default_class_id(self) -> int:
        text = self.DefaultClassEdit.text().strip()
        return int(text) if text else 0

    def read_selected_model(self) -> tuple[str, bool]:
        model_name = self.ModelChooseC.currentText().strip()
        is_cpsam = model_name.lower() == "cpsam"
        return ("cpsam" if is_cpsam else model_name, not is_cpsam)

    def read_inference_options(self) -> dict:
        def _read_value(attr, cast):
            value = getattr(self, attr)
            if isinstance(value, (int, float)):
                return cast(value)
            return cast(value.text())

        return {
            "min_size": _read_value("min_size", int),
        }

    def read_series_slider_axes(self) -> dict[str, int]:
        return {
            axis_name: control["slider"].value()
            for axis_name, control in self.series_nav_controls.items()
        }

    def read_saturation_range(self) -> tuple[float, float]:
        low, high = self.sliders[0].value()
        return float(low), float(high)

    def read_view_mode_index(self) -> int:
        return int(self.ViewDropDown.currentIndex())

    # ---- MainViewProtocol: widget output ----

    def apply_segmentation_widgets(self, params: SegmentationParameters) -> None:
        low, high = params.percentile
        if self.seg_param_root.param("norm_percentile_low").value() != low:
            self.seg_param_root.param("norm_percentile_low").setValue(low)
        if self.seg_param_root.param("norm_percentile_high").value() != high:
            self.seg_param_root.param("norm_percentile_high").setValue(high)
        if self.seg_param_root.param("niter").value() != params.niter:
            self.seg_param_root.param("niter").setValue(params.niter)

    def set_progress(self, value: int) -> None:
        self.progress.setValue(value)

    def show_message(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def set_window_title(self, title: str) -> None:
        self.setWindowTitle(title)

    def set_ncells_count(self, n: int) -> None:
        if hasattr(self, "ncells_counter"):
            self.ncells_counter.set(n)

    def apply_series_labels(self, state: SeriesState) -> None:
        if state.display_filename:
            self.set_window_title(str(state.display_filename))

    def set_series_navigation(self, nav: SeriesNavViewState) -> None:
        self._updating_series_navigation = True
        try:
            self.navBox.setEnabled(nav.enabled)
            for axis_name, control in self.series_nav_controls.items():
                slider = control["slider"]
                prev_btn = control["prev_btn"]
                next_btn = control["next_btn"]
                if not nav.enabled:
                    slider.setRange(0, 0)
                    slider.setValue(0)
                    slider.setEnabled(False)
                    prev_btn.setEnabled(False)
                    next_btn.setEnabled(False)
                    continue
                low, high = nav.axis_ranges.get(axis_name, (0, 0))
                slider.setEnabled(True)
                slider.setRange(low, high)
                slider.setValue(nav.axis_values.get(axis_name, 0))
                prev_btn.setEnabled(high > 0)
                next_btn.setEnabled(high > 0)
        finally:
            self._updating_series_navigation = False

    def set_view_mode(self, index: int, restored_enabled: bool) -> None:
        last = self.ViewDropDown.count() - 1
        self.ViewDropDown.model().item(last).setEnabled(restored_enabled)
        self.ViewDropDown.setCurrentIndex(index)
        self.view = index

    def set_mask_action_enabled(self, enabled: bool) -> None:
        self.saveResults.setEnabled(enabled)
        self.ClearButton.setEnabled(enabled)
        self.remcell.setEnabled(enabled)
        self.undo.setEnabled(enabled)
        if hasattr(self, "RectSelectButton"):
            self.RectSelectButton.setEnabled(enabled)
            if not enabled and self.RectSelectButton.isChecked():
                self.RectSelectButton.setChecked(False)
        if hasattr(self, "DeleteSelectedButton"):
            self.DeleteSelectedButton.setEnabled(enabled)
        if hasattr(self, "EditSelectedButton"):
            self.EditSelectedButton.setEnabled(enabled)

    def set_run_enabled(self, enabled: bool) -> None:
        self.ModelButtonC.setEnabled(enabled)

    def set_loaded_chrome(self, enabled: bool, ncells: int = 0) -> None:
        self._canvas_loaded = enabled
        if enabled:
            self.set_run_enabled(True)
            for i in range(len(self.StyleButtons)):
                self.StyleButtons[i].setEnabled(True)
            self.autoSaturationButton.setEnabled(True)
            self.newmodel.setEnabled(True)
            self.sliders[0].setEnabled(True)
            self.set_mask_action_enabled(ncells > 0)
            self._update_canvas_cursor()
        else:
            self.disable_buttons_removeROIs()

    def set_series_slider_value(self, axis_name: str, value: int) -> None:
        control = self.series_nav_controls.get(axis_name)
        if control is None:
            return
        control["slider"].setValue(value)

    def is_updating_series_navigation(self) -> bool:
        return bool(getattr(self, "_updating_series_navigation", False))

    def set_updating_series_navigation(self, updating: bool) -> None:
        self._updating_series_navigation = updating

    def progress_widget(self):
        return self.progress

    def set_redo_enabled(self, enabled: bool) -> None:
        self.redo.setEnabled(enabled)

    def set_undo_enabled(self, enabled: bool) -> None:
        self.undo.setEnabled(enabled)

    def apply_segmentation_metadata_widgets(self, segmentation) -> None:
        if not hasattr(self, "seg_param_root"):
            return
        self.seg_param_root.param("flow_threshold").setValue(
            float(segmentation.flow_threshold)
        )
        self.seg_param_root.param("cellprob_threshold").setValue(
            float(segmentation.cellprob_threshold)
        )
        self.seg_param_root.param("niter").setValue(int(segmentation.niter))
        if segmentation.diameter is not None:
            self.seg_param_root.param("diameter").setValue(float(segmentation.diameter))
        if hasattr(segmentation, "min_size"):
            self.min_size = int(segmentation.min_size)

    def set_model_list(self, models: list[str], current: str | None = None) -> None:
        self.model_strings = list(models)
        self.ModelChooseC.clear()
        self.ModelChooseC.addItems(["CPSAM"])
        if models:
            self.ModelChooseC.addItems(models)
        if current:
            self.ModelChooseC.setCurrentText(current)
        elif models:
            self.ModelChooseC.setCurrentIndex(len(models))

    def show_window(self) -> None:
        self.win.show()
        self.show()

    # ---- MainViewProtocol: canvas rendering ----

    def render_image_plane(
        self,
        image: np.ndarray,
        levels: list[float] | tuple[float, float],
        lut: np.ndarray | None = None,
    ) -> None:
        self.img.setImage(as_gray_image(image), autoLevels=False, lut=lut)
        self.img.setLevels(list(levels))

    def render_mask_overlay(self, layerz: np.ndarray) -> None:
        self.layer.setImage(layerz, autoLevels=False)

    def render_diameter_scale(self, radii: np.ndarray) -> None:
        self.scale.setImage(radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])

    def render_selection_boxes(
        self, bounds_list: list[tuple[int, int, int, int]]
    ) -> None:
        if not bounds_list:
            self._clear_selection_boxes()
            return
        xs = []
        ys = []
        for x0, y0, x1, y1 in bounds_list:
            xs.extend([x0, x1, x1, x0, x0, np.nan])
            ys.extend([y0, y0, y1, y1, y0, np.nan])
        self._ensure_selection_box()
        self.selection_box.setData(
            np.array(xs[:-1], dtype=float),
            np.array(ys[:-1], dtype=float),
            connect="finite",
        )

    def render_rect_select_preview(
        self, bounds: tuple[int, int, int, int] | None
    ) -> None:
        if bounds is None:
            self._clear_rect_select_preview()
            return
        x0, y0, x1, y1 = bounds
        self._ensure_rect_select_preview()
        self.rect_select_preview.setData(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]
        )

    def sync_saturation_slider(self, low: float, high: float) -> None:
        self.sliders[0].setValue([low, high])

    def refresh_plot_from_model(self) -> None:
        pass

    def refresh_scale_from_model(self) -> None:
        pass

    def refresh_labels_table(
        self,
        rows: list[LabelRow] | None = None,
        header_state: str | None = None,
    ):
        if rows is None or not hasattr(self, "LabelsTable"):
            return
        self._refreshing_labels_table = True
        self.LabelsTable.blockSignals(True)
        self.LabelsTable.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            visible_item = QTableWidgetItem()
            visible_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            visible_item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if row.visible
                else QtCore.Qt.CheckState.Unchecked
            )
            roi_item = QTableWidgetItem(str(row.roi))
            roi_item.setFlags(roi_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            class_item = QTableWidgetItem(str(row.class_id))
            major_item = QTableWidgetItem(
                "" if row.major_diameter is None else f"{row.major_diameter:.1f}"
            )
            major_item.setFlags(major_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            minor_item = QTableWidgetItem(
                "" if row.minor_diameter is None else f"{row.minor_diameter:.1f}"
            )
            minor_item.setFlags(minor_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.LabelsTable.setItem(row_index, 0, visible_item)
            self.LabelsTable.setItem(row_index, 1, roi_item)
            self.LabelsTable.setItem(row_index, 2, class_item)
            self.LabelsTable.setItem(row_index, 3, major_item)
            self.LabelsTable.setItem(row_index, 4, minor_item)
            self.LabelsTable.setRowHidden(row_index, row.hidden_by_filter)
        self.LabelsTable.blockSignals(False)
        selected_cells = [row.roi for row in rows if row.selected]
        if selected_cells:
            self._sync_labels_table_selection_multi(selected_cells)
        elif not self._syncing_labels_table_selection:
            self._sync_labels_table_selection_multi([])
        self._refreshing_labels_table = False
        if header_state is not None and hasattr(self, "_visibility_header"):
            state_map = {
                "unchecked": QtCore.Qt.CheckState.Unchecked,
                "checked": QtCore.Qt.CheckState.Checked,
                "partial": QtCore.Qt.CheckState.PartiallyChecked,
            }
            self._visibility_header.set_check_state(
                state_map.get(header_state, QtCore.Qt.CheckState.Unchecked)
            )

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def disable_buttons_removeROIs(self):
        self.ModelButtonC.setEnabled(False)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(False)
        self.newmodel.setEnabled(False)
        self.saveResults.setEnabled(False)
        if hasattr(self, "RectSelectButton"):
            self.RectSelectButton.setEnabled(False)
            if self.RectSelectButton.isChecked():
                self.RectSelectButton.setChecked(False)
        if hasattr(self, "DeleteSelectedButton"):
            self.DeleteSelectedButton.setEnabled(False)
        if hasattr(self, "EditSelectedButton"):
            self.EditSelectedButton.setEnabled(False)

    def make_buttons(self):
        b = 0
        self.navBox = QGroupBox("Navigation")
        self.navBoxG = QGridLayout()
        self.navBox.setLayout(self.navBoxG)
        self.left_sidebar.addWidget(self.navBox, b, 0, 1, 9)
        self.series_nav_controls = {}
        axis_labels = {
            "position": "P",
            "time": "T",
            "channel": "C",
            "z": "Z",
        }
        for column, axis_name in enumerate(series.SERIES_AXES):
            row = column
            label = QLabel(axis_labels[axis_name])
            self.navBoxG.addWidget(label, row, 0, 1, 1)

            prev_btn = QPushButton("<")
            prev_btn.setEnabled(False)
            self.navBoxG.addWidget(prev_btn, row, 1, 1, 1)

            slider = SeriesAxisSlider(QtCore.Qt.Orientation.Horizontal, self)
            slider.setRange(0, 0)
            slider.setEnabled(False)
            slider.setTracking(True)
            self.navBoxG.addWidget(slider, row, 2, 1, 1)

            next_btn = QPushButton(">")
            next_btn.setEnabled(False)
            self.navBoxG.addWidget(next_btn, row, 3, 1, 1)

            self.series_nav_controls[axis_name] = {
                "slider": slider,
                "prev_btn": prev_btn,
                "next_btn": next_btn,
            }
        self.navBox.setEnabled(False)

        b += 1
        self.satBox = QGroupBox("Views")
        self.satBoxV = QVBoxLayout()
        self.satBox.setLayout(self.satBoxV)
        self.left_sidebar.addWidget(self.satBox, b, 0, 1, 9)

        self.ViewDropDown = QComboBox()
        self.ViewDropDown.addItems(["image", "gradXY", "cellprob", "restored"])
        self.ViewDropDown.model().item(3).setEnabled(False)
        self.satBoxV.addWidget(self.ViewDropDown)

        self.autoSaturationButton = QPushButton("auto saturation")
        self.autoSaturationButton.setEnabled(False)
        self.satBoxV.addWidget(self.autoSaturationButton)

        self.sliders = []
        self.sliders.append(Slider(self, "gray", [100, 100, 100]))
        self.sliders[-1].setMinimum(-0.1)
        self.sliders[-1].setMaximum(255.1)
        self.sliders[-1].setValue([0, 255])
        self.satBoxV.addWidget(self.sliders[-1])

        b += 1
        self.drawBox = QGroupBox("Drawing")
        self.drawBoxV = QVBoxLayout()
        self.drawBox.setLayout(self.drawBoxV)
        self.left_sidebar.addWidget(self.drawBox, b, 0, 1, 9)

        self.brush_size = 1

        default_class_layout = QHBoxLayout()
        default_class_label = QLabel("default class")
        self.DefaultClassEdit = QLineEdit("0")
        self.DefaultClassEdit.setValidator(
            QtGui.QIntValidator(0, np.iinfo(np.int32).max, self)
        )
        self.DefaultClassEdit.setMaximumWidth(80)
        default_class_layout.addWidget(default_class_label)
        default_class_layout.addWidget(self.DefaultClassEdit)
        self.drawBoxV.addLayout(default_class_layout)

        self.BrushButton = QPushButton("brush")
        self.BrushButton.setCheckable(True)
        self.BrushButton.setToolTip(
            "Toggle brush mode: click to start drawing, move to draw, then click again to finish"
        )
        self.drawBoxV.addWidget(self.BrushButton)

        select_layout = QVBoxLayout()
        self.RectSelectButton = QPushButton("select")
        self.RectSelectButton.setCheckable(True)
        self.RectSelectButton.setEnabled(False)
        self.RectSelectButton.setToolTip(
            "Click a cell to select it, Shift+click to add to selection, "
            "or drag a rectangle for multiple selection"
        )
        select_layout.addWidget(self.RectSelectButton)
        self.DeleteSelectedButton = QPushButton("delete")
        self.DeleteSelectedButton.setEnabled(False)
        self.DeleteSelectedButton.setToolTip(
            "Delete the currently selected cell(s)"
        )
        select_layout.addWidget(self.DeleteSelectedButton)
        self.EditSelectedButton = QPushButton("edit")
        self.EditSelectedButton.setEnabled(False)
        self.EditSelectedButton.setToolTip(
            "Change the class ID for all selected cell(s)"
        )
        select_layout.addWidget(self.EditSelectedButton)
        self.drawBoxV.addLayout(select_layout)

        b += 1
        self.segBox = QGroupBox("Segmentation")
        self.segBoxV = QVBoxLayout()
        self.segBox.setLayout(self.segBoxV)
        self.right_sidebar.addWidget(self.segBox, 0, 0, 1, 1)

        seg_controls = QHBoxLayout()
        self.segBoxV.addLayout(seg_controls)

        self.StyleButtons = []
        model_label = QLabel("model:")
        seg_controls.addWidget(model_label)

        self.ModelChooseC = QComboBox()
        self.ModelChooseC.addItems(["CPSAM"])
        if len(self.model_strings) > 0:
            self.ModelChooseC.addItems(self.model_strings)
        self.ModelChooseC.setCurrentIndex(0)
        self.ModelChooseC.setEditable(True)
        self.ModelChooseC.setInsertPolicy(QComboBox.NoInsert)
        self.ModelChooseC.setCurrentText("CPSAM")
        self.ModelChooseC.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.ModelChooseC.completer().setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        seg_controls.addWidget(self.ModelChooseC)

        self.ModelButtonC = QPushButton("run")
        self.ModelButtonC.setEnabled(False)

        self.ncells_counter = ObservableVariable(0)

        self.labels_box = QGroupBox("Labels table")
        self.labels_box_v = QVBoxLayout()
        self.labels_box.setLayout(self.labels_box_v)
        self.right_sidebar.addWidget(self.labels_box, 2, 0, 1, 1)

        labels_filter_layout = QHBoxLayout()
        labels_filter_layout.addWidget(QLabel("class filter:"))
        self.LabelsClassFilter = QLineEdit()
        self.LabelsClassFilter.setPlaceholderText("all")
        labels_filter_layout.addWidget(self.LabelsClassFilter)
        self.labels_box_v.addLayout(labels_filter_layout)

        self.LabelsTable = QTableWidget(0, 5)
        visibility_header = CheckBoxHeader(QtCore.Qt.Orientation.Horizontal, self.LabelsTable)
        self.LabelsTable.setHorizontalHeader(visibility_header)
        self.LabelsTable.setHorizontalHeaderLabels(
            ["", "ROI", "Class ID", "Major diam.", "Minor diam."]
        )
        self._visibility_header = visibility_header
        self.LabelsTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.LabelsTable.setColumnWidth(0, 32)
        self.LabelsTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.LabelsTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.LabelsTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.LabelsTable.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.LabelsTable.verticalHeader().setVisible(False)
        self.LabelsTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.LabelsTable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.labels_box_v.addWidget(self.LabelsTable)

        self.progress = QProgressBar(self)

        self.seg_param_root = Parameter.create(
            name="segmentation",
            type="group",
            children=[
                {"name": "diameter", "type": "float", "value": 0.0, "step": 1.0},
                {
                    "name": "flow_threshold",
                    "title": "flow threshold",
                    "type": "float",
                    "value": 0.4,
                    "step": 0.1,
                },
                {
                    "name": "cellprob_threshold",
                    "title": "cellprob threshold",
                    "type": "float",
                    "value": 0.0,
                    "step": 0.1,
                },
                {
                    "name": "norm_percentile_low",
                    "title": "norm percentile lower",
                    "type": "float",
                    "value": 1.0,
                    "limits": (0.0, 100.0),
                    "step": 1.0,
                },
                {
                    "name": "norm_percentile_high",
                    "title": "norm percentile upper",
                    "type": "float",
                    "value": 99.0,
                    "limits": (0.0, 100.0),
                    "step": 1.0,
                },
                {
                    "name": "niter",
                    "title": "niter dynamics",
                    "type": "int",
                    "value": 0,
                    "step": 1,
                },
            ],
        )
        self.seg_params_tree = ParameterTree(showHeader=False)
        self.seg_params_tree.setParameters(self.seg_param_root, showTop=False)
        self.segBoxV.addWidget(self.seg_params_tree)

        seg_progress = QHBoxLayout()
        self.segBoxV.addLayout(seg_progress)
        seg_progress.addWidget(self.ModelButtonC)
        seg_progress.addWidget(self.progress, 1)

        return b

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.files_dropped.emit(files)
        event.accept()

    def make_viewbox(self):
        self.p0 = ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True,
        )
        self.p0.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.brush_size = 1
        self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0, 255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0, 255])
        self.p0.scene().contextMenuItem = self.p0
        self.ly, self.lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)
        self.selection_box = None
        self.rect_select_preview = None

    def plot_clicked(self, event):
        if event.double():
            self.plot_double_clicked.emit()

    def _ensure_selection_box(self):
        if self.selection_box is None:
            pen = pg.mkPen(
                color=(255, 255, 0),
                width=2,
                style=QtCore.Qt.PenStyle.DashLine,
            )
            self.selection_box = pg.PlotCurveItem(pen=pen)
            self.selection_box.setZValue(10)
            self.p0.addItem(self.selection_box)

    def _clear_selection_boxes(self):
        if self.selection_box is not None:
            self.selection_box.setData([], [])

    def _ensure_rect_select_preview(self):
        if self.rect_select_preview is None:
            pen = pg.mkPen(
                color=(0, 255, 255),
                width=2,
                style=QtCore.Qt.PenStyle.DashLine,
            )
            self.rect_select_preview = pg.PlotCurveItem(pen=pen)
            self.rect_select_preview.setZValue(11)
            self.p0.addItem(self.rect_select_preview)

    def _clear_rect_select_preview(self):
        if self.rect_select_preview is not None:
            self.rect_select_preview.setData([], [])

    def _normalize_rect(self, x0, y0, x1, y1):
        x0, x1 = sorted([int(x0), int(x1)])
        y0, y1 = sorted([int(y0), int(y1)])
        x0 = max(0, min(self.lx - 1, x0))
        x1 = max(0, min(self.lx, x1))
        y0 = max(0, min(self.ly - 1, y0))
        y1 = max(0, min(self.ly, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _set_rect_preview(self, x0, y0, x1, y1):
        bounds = self._normalize_rect(x0, y0, x1, y1)
        if bounds is None:
            self._clear_rect_select_preview()
            return
        x0, y0, x1, y1 = bounds
        self._ensure_rect_select_preview()
        self.rect_select_preview.setData(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]
        )

    def _sync_labels_table_selection_multi(self, indices):
        if not hasattr(self, "LabelsTable"):
            return
        selection_model = self.LabelsTable.selectionModel()
        if selection_model is None:
            return

        self._syncing_labels_table_selection = True
        selection_model.clearSelection()
        first_index = None
        for idx in indices:
            row = int(idx) - 1
            if 0 <= row < self.LabelsTable.rowCount():
                index = self.LabelsTable.model().index(row, 0)
                if first_index is None:
                    first_index = index
                selection_model.select(
                    index,
                    QtCore.QItemSelectionModel.SelectionFlag.Select
                    | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                )
        if first_index is not None:
            selection_model.setCurrentIndex(
                first_index,
                QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        self._syncing_labels_table_selection = False

        viewport = self.LabelsTable.viewport()
        if viewport is not None:
            viewport.update()

        if indices:
            first_row = int(indices[0]) - 1
            if 0 <= first_row < self.LabelsTable.rowCount():
                item = self.LabelsTable.item(first_row, 0)
                if item is not None:
                    self.LabelsTable.scrollToItem(
                        item,
                        QAbstractItemView.ScrollHint.EnsureVisible,
                    )

    def _update_canvas_cursor(self):
        if not hasattr(self, "p0"):
            return
        if not self._canvas_loaded:
            cursor = QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor)
        elif hasattr(self, "BrushButton") and self.BrushButton.isChecked():
            cursor = brush_cursor()
        elif self.rect_select_mode:
            cursor = select_cursor()
        else:
            cursor = QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.p0.setCursor(cursor)
        self.win.setCursor(cursor)

    def _canvas_view(self):
        views = self.win.scene().views()
        return views[0] if views else None

    def _canvas_viewport(self):
        view = self._canvas_view()
        return view.viewport() if view is not None else None

    def _view_pos_from_viewport_event(self, event):
        view = self._canvas_view()
        if view is None:
            return None
        scene_pos = view.mapToScene(event.position().toPoint())
        return self.p0.mapSceneToView(scene_pos)

    def eventFilter(self, obj, event):
        if obj is self._canvas_viewport() and self.rect_select_mode:
            event_type = event.type()
            if (
                event_type == QtCore.QEvent.Type.MouseButtonPress
                and event.button() == QtCore.Qt.MouseButton.LeftButton
            ):
                pos = self._view_pos_from_viewport_event(event)
                if pos is not None:
                    self._rect_select_start = (int(pos.y()), int(pos.x()))
                    self._rect_select_dragging = False
                    self.rect_select_press.emit(pos, event.modifiers())
                return True
            if (
                event_type == QtCore.QEvent.Type.MouseMove
                and event.buttons() & QtCore.Qt.MouseButton.LeftButton
                and self._rect_select_start is not None
            ):
                pos = self._view_pos_from_viewport_event(event)
                if pos is not None:
                    y0, x0 = self._rect_select_start
                    y1, x1 = int(pos.y()), int(pos.x())
                    if not self._rect_select_dragging:
                        if (
                            max(abs(y1 - y0), abs(x1 - x0))
                            < self._rect_select_drag_threshold
                        ):
                            return True
                        self._rect_select_dragging = True
                    self._set_rect_preview(x0, y0, x1, y1)
                    self.rect_select_move.emit(pos)
                return True
            if (
                event_type == QtCore.QEvent.Type.MouseButtonRelease
                and event.button() == QtCore.Qt.MouseButton.LeftButton
                and self._rect_select_start is not None
            ):
                pos = self._view_pos_from_viewport_event(event)
                if pos is not None:
                    self.rect_select_release.emit(pos, event.modifiers())
                return True
        return super().eventFilter(obj, event)


MainW = MainView