"""
Qt view layer for the Cellpose GUI (MVP).

MainView owns widgets and rendering only; user actions delegate to MainPresenter.

Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer,
Michael Rariden and Marius Pachitariu.
"""

import copy
import datetime
import os
import pathlib
import sys
import time
import warnings

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
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

from .. import models, version
from ..io import get_image_files
from ..models import normalize_default
from ..utils import download_url_to_file
from . import io, menus, series
from .model import MainModel, SegmentationParameters, SeriesState
from .presenter import MainPresenter
from .view_protocol import LabelRow, SeriesNavViewState
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
except:
    MATPLOTLIB = False

def run(image=None):
    from ..io import logger_setup

    logger, log_file = logger_setup()
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath(".cellpose")
        cp_dir.mkdir(exist_ok=True)
        print("downloading logo")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_transparent.png",
            icon_path,
            progress=True,
        )
    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    MainView(image=image, logger=logger)
    ret = app.exec()
    sys.exit(ret)


class MainView(QMainWindow):
    """Passive Qt view for the Cellpose GUI MVP stack."""

    def __init__(self, image=None, logger=None):
        super(MainView, self).__init__()

        self.logger = logger
        self.model = MainModel(
            model_save_folder=os.fspath(models.MODEL_DIR.joinpath("custom")),
        )
        self.presenter = MainPresenter(self, self.model)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(100, 100, 1280, 720)
        self.setWindowTitle(f"cellpose v{version}")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
        icon_path = str(icon_path.resolve())
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)

        self.model.session.loaded = False
        self.model.session.recompute_masks = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.lmain = QHBoxLayout()
        self.cwidget.setLayout(self.lmain)
        self.setCentralWidget(self.cwidget)
        self.lmain.setContentsMargins(0, 0, 0, 10)

        self.imask = 0
        self.rect_select_mode = False
        self._rect_select_start = None
        self._rect_select_dragging = False
        self._rect_select_drag_threshold = 4
        self.left_sidebar = QGridLayout()
        self.left_sidebar_widget = QWidget(self)
        self.left_sidebar_widget.setLayout(self.left_sidebar)
        self.right_sidebar = QGridLayout()
        self.right_sidebar_widget = QWidget(self)
        self.right_sidebar_widget.setLayout(self.right_sidebar)
        b = self.make_buttons()

        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()
        self.lmain.addWidget(self.left_sidebar_widget, 1)
        self.lmain.addWidget(self.win, 2)
        self.lmain.addWidget(self.right_sidebar_widget, 1)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        viewport = self._canvas_viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        if MATPLOTLIB:
            self.colormap = (
                plt.get_cmap("gist_ncar")(np.linspace(0.0, 0.9, 1000000)) * 255
            ).astype(np.uint8)
            np.random.seed(42)  # make colors stable
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42)  # make colors stable
            self.colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(
                np.uint8
            )
        self.model.session.nz = 1
        self.model.session.restore = None
        self.model.session.ratio = 1.0
        self.last_series_subfolder_template = ""
        self.last_series_filename_template = ""
        self.reset()


        # if called with image, load it
        if image is not None:
            self.model.filename = image
            io._load_image(self, self.model.filename)

        self.min_size = 15

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    @property
    def training_params(self):
        return self.presenter.training_params

    @training_params.setter
    def training_params(self, params):
        self.presenter.set_training_parameters(params)

    def ncells(self) -> int:
        return self.model.ncells

    def set_progress(self, value: int) -> None:
        self.progress.setValue(value)

    def show_message(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def set_window_title(self, title: str) -> None:
        self.setWindowTitle(title)

    def sync_series_state(self, state):
        self.model.series_state.dataset = state.dataset
        self.model.series_state.record_index = state.record_index
        self.model.series_state.output_filename = state.output_filename
        self.model.series_state.display_filename = state.display_filename
        if state.filename is not None:
            self.model.filename = state.filename

    def set_series_state(self, dataset=None, record_index=None):
        self.presenter.set_series(dataset=dataset, record_index=record_index)

    def read_segmentation_widgets(self):
        return {
            "diameter": self.seg_param_root.param("diameter").value(),
            "flow_threshold": self.seg_param_root.param("flow_threshold").value(),
            "cellprob_threshold": self.seg_param_root.param("cellprob_threshold").value(),
            "percentile_low": self.seg_param_root.param("norm_percentile_low").value(),
            "percentile_high": self.seg_param_root.param("norm_percentile_high").value(),
            "niter": self.seg_param_root.param("niter").value(),
        }

    def apply_segmentation_widgets(self, params):
        low, high = params.percentile
        if self.seg_param_root.param("norm_percentile_low").value() != low:
            self.seg_param_root.param("norm_percentile_low").setValue(low)
        if self.seg_param_root.param("norm_percentile_high").value() != high:
            self.seg_param_root.param("norm_percentile_high").setValue(high)
        if self.seg_param_root.param("niter").value() != params.niter:
            self.seg_param_root.param("niter").setValue(params.niter)

    def labels_class_filter_text(self):
        if not hasattr(self, "LabelsClassFilter"):
            return ""
        return self.LabelsClassFilter.text()

    # ---- MainViewProtocol: widget input ----

    def read_labels_class_filter(self) -> str:
        return self.labels_class_filter_text()

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
        z = self.model.session.current_z
        low, high = self.model.session.saturation[0][z]
        return float(low), float(high)

    def read_view_mode_index(self) -> int:
        return int(self.ViewDropDown.currentIndex())

    # ---- MainViewProtocol: widget output ----

    def set_ncells_count(self, n: int) -> None:
        if hasattr(self, "ncells_counter"):
            self.ncells_counter.set(n)

    def _sync_ncells_counter(self) -> None:
        self.set_ncells_count(self.model.ncells)

    def apply_series_labels(self, state: SeriesState) -> None:
        self.sync_series_state(state)

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

    def set_loaded_chrome(self, enabled: bool) -> None:
        if enabled:
            self.set_run_enabled(True)
            for i in range(len(self.StyleButtons)):
                self.StyleButtons[i].setEnabled(True)
            self.autoSaturationButton.setEnabled(True)
            self.newmodel.setEnabled(True)
            self.sliders[0].setEnabled(True)
            self.set_mask_action_enabled(self.ncells() > 0)
            self.refresh_plot_from_model()
            self._update_canvas_cursor()
            title = self.model.series_state.display_filename or self.model.filename
            if title:
                self.set_window_title(str(title))
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
        io._apply_cellpose_segmentation_widgets(self, segmentation)

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
        self._update_selection_boxes()

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
        self.update_plot()

    def refresh_scale_from_model(self) -> None:
        self.update_scale()

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
            prev_btn.clicked.connect(
                lambda _checked=False,
                axis_name=axis_name: self.navigate_series_from_sliders(axis_name, -1)
            )
            self.navBoxG.addWidget(prev_btn, row, 1, 1, 1)

            slider = SeriesAxisSlider(QtCore.Qt.Orientation.Horizontal, self)
            slider.setRange(0, 0)
            slider.setEnabled(False)
            slider.setTracking(True)
            slider.sliderReleased.connect(
                lambda axis_name=axis_name: self._commit_series_slider(axis_name)
            )
            self.navBoxG.addWidget(slider, row, 2, 1, 1)

            next_btn = QPushButton(">")
            next_btn.setEnabled(False)
            next_btn.clicked.connect(
                lambda _checked=False,
                axis_name=axis_name: self.navigate_series_from_sliders(axis_name, +1)
            )
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

        self.view = 0  # 0=image, 1=gradXY, 2=cellprob, 3=restored
        self.ViewDropDown = QComboBox()
        self.ViewDropDown.addItems(["image", "gradXY", "cellprob", "restored"])
        self.ViewDropDown.model().item(3).setEnabled(False)
        self.ViewDropDown.currentIndexChanged.connect(self.update_plot)
        self.satBoxV.addWidget(self.ViewDropDown)

        self.autoSaturationButton = QPushButton("auto saturation")
        self.autoSaturationButton.setEnabled(False)
        self.autoSaturationButton.clicked.connect(self.compute_saturation)
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
        self.BrushButton.toggled.connect(self.toggle_brush_mode)
        self.drawBoxV.addWidget(self.BrushButton)

        select_layout = QVBoxLayout()
        self.RectSelectButton = QPushButton("select")
        self.RectSelectButton.setCheckable(True)
        self.RectSelectButton.setEnabled(False)
        self.RectSelectButton.setToolTip(
            "Click a cell to select it, Shift+click to add to selection, "
            "or drag a rectangle for multiple selection"
        )
        self.RectSelectButton.toggled.connect(self.toggle_rect_select_mode)
        select_layout.addWidget(self.RectSelectButton)
        self.DeleteSelectedButton = QPushButton("delete")
        self.DeleteSelectedButton.setEnabled(False)
        self.DeleteSelectedButton.setToolTip(
            "Delete the currently selected cell(s)"
        )
        self.DeleteSelectedButton.clicked.connect(self.delete_selected_cells)
        select_layout.addWidget(self.DeleteSelectedButton)
        self.EditSelectedButton = QPushButton("edit")
        self.EditSelectedButton.setEnabled(False)
        self.EditSelectedButton.setToolTip(
            "Change the class ID for all selected cell(s)"
        )
        self.EditSelectedButton.clicked.connect(self.edit_selected_cells)
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
        self.ModelChooseC.activated.connect(lambda: self.model_choose(custom=True))
        seg_controls.addWidget(self.ModelChooseC)

        self.ModelButtonC = QPushButton("run")
        self.ModelButtonC.clicked.connect(self.run_selected_model)
        self.ModelButtonC.setEnabled(False)

        self.ncells_counter = ObservableVariable(0)
        self.ncells_counter.valueChanged.connect(lambda *_: self.refresh_labels_table())

        self.labels_box = QGroupBox("Labels table")
        self.labels_box_v = QVBoxLayout()
        self.labels_box.setLayout(self.labels_box_v)
        self.right_sidebar.addWidget(self.labels_box, 2, 0, 1, 1)

        labels_filter_layout = QHBoxLayout()
        labels_filter_layout.addWidget(QLabel("class filter:"))
        self.LabelsClassFilter = QLineEdit()
        self.LabelsClassFilter.setPlaceholderText("all")
        self.LabelsClassFilter.returnPressed.connect(self.labels_filter_changed)
        labels_filter_layout.addWidget(self.LabelsClassFilter)
        self.labels_box_v.addLayout(labels_filter_layout)

        self._refreshing_labels_table = False
        self._syncing_labels_table_selection = False
        self.LabelsTable = QTableWidget(0, 3)
        visibility_header = CheckBoxHeader(QtCore.Qt.Orientation.Horizontal, self.LabelsTable)
        visibility_header.checkboxClicked.connect(self._toggle_all_labels_visibility)
        self.LabelsTable.setHorizontalHeader(visibility_header)
        self.LabelsTable.setHorizontalHeaderLabels(["", "ROI", "Class ID"])
        self._visibility_header = visibility_header
        self.LabelsTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.LabelsTable.setColumnWidth(0, 32)
        self.LabelsTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.LabelsTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.LabelsTable.verticalHeader().setVisible(False)
        self.LabelsTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.LabelsTable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.LabelsTable.itemChanged.connect(self.on_labels_table_item_changed)
        self.LabelsTable.itemSelectionChanged.connect(self.select_cell_from_labels_table)
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

        self.seg_param_root.param("diameter").sigValueChanged.connect(
            lambda *_: self.update_scale()
        )
        self.seg_param_root.param("flow_threshold").sigValueChanged.connect(
            lambda *_: self.compute_cprob()
        )
        self.seg_param_root.param("cellprob_threshold").sigValueChanged.connect(
            lambda *_: self.compute_cprob()
        )
        self.seg_param_root.param("niter").sigValueChanged.connect(
            lambda *_: self.compute_cprob()
        )
        self.seg_param_root.param("norm_percentile_low").sigValueChanged.connect(
            lambda *_: self.validate_normalization_range()
        )
        self.seg_param_root.param("norm_percentile_high").sigValueChanged.connect(
            lambda *_: self.validate_normalization_range()
        )

        self.model.session.restore = None
        self.model.session.ratio = 1.0

        return b

    def validate_normalization_range(self):
        try:
            self.get_segmentation_parameters()
        except ValueError:
            print("GUI_ERROR: normalization percentile lower must be less than upper")

    def get_segmentation_parameters(self):
        return self.presenter.segmentation_parameters_dict()

    def level_change(self, r):
        if self.model.session.loaded:
            sval = self.sliders[0].value()
            self.model.session.saturation[0][self.model.session.current_z] = sval
            self.update_plot()

    def default_class_id(self):
        text = self.DefaultClassEdit.text().strip()
        return int(text) if text else 0

    def _selected_segmentation_model(self):
        model_name = self.ModelChooseC.currentText().strip()
        is_cpsam = model_name.lower() == "cpsam"
        return "cpsam" if is_cpsam else model_name, not is_cpsam

    def run_selected_model(self):
        self.presenter.run_selected_model()

    def model_choose(self, custom=False):
        if not custom:
            return
        model_name, is_custom = self._selected_segmentation_model()
        if model_name:
            print(f"GUI_INFO: selected model {model_name}, loading now")
            self.initialize_model(model_name=model_name, custom=is_custom)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def enable_buttons(self):
        self.set_loaded_chrome(True)

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

    def toggle_mask_ops(self):
        self.update_layer()
        self.toggle_saving()
        self.toggle_removals()
        self.refresh_labels_table()

    def _ensure_instance_classes(self):
        self.presenter.ensure_instance_classes()

    @property
    def instance_classes(self):
        return self.presenter.instance_classes

    def set_instance_classes(self, instance_classes=None):
        self.presenter.set_instance_classes(instance_classes)

    def labels_class_filter(self):
        return self.presenter.labels_class_filter()

    def labels_filter_changed(self):
        self.presenter.on_labels_filter_changed()

    def _ensure_instance_visible(self):
        self.presenter.ensure_instance_visible()

    @property
    def instance_visible(self):
        return self.presenter.instance_visible

    def set_instance_visible(self, instance_visible=None):
        self.presenter.set_instance_visible(instance_visible)

    def visible_cell_pixels(self, cellpix):
        return self.presenter.visible_cell_pixels(cellpix)

    def refresh_labels_table(
        self,
        rows: list[LabelRow] | None = None,
        header_state: str | None = None,
    ):
        if not hasattr(self, "LabelsTable"):
            return
        if rows is None:
            self.presenter.refresh_labels_table()
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
            self.LabelsTable.setItem(row_index, 0, visible_item)
            self.LabelsTable.setItem(row_index, 1, roi_item)
            self.LabelsTable.setItem(row_index, 2, class_item)
            self.LabelsTable.setRowHidden(row_index, row.hidden_by_filter)
        self.LabelsTable.blockSignals(False)
        selected_cells = [row.roi for row in rows if row.selected]
        if selected_cells:
            self._sync_labels_table_selection_multi(selected_cells)
        elif not self._syncing_labels_table_selection:
            selection_model = self.LabelsTable.selectionModel()
            if selection_model is not None:
                self._syncing_labels_table_selection = True
                selection_model.blockSignals(True)
                selection_model.clearSelection()
                selection_model.blockSignals(False)
                self._syncing_labels_table_selection = False
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

    def _sync_visibility_header_checkbox(self):
        if not hasattr(self, "_visibility_header"):
            return
        ncells = self.ncells()
        if ncells == 0:
            self._visibility_header.set_check_state(QtCore.Qt.CheckState.Unchecked)
            return
        self._ensure_instance_visible()
        visible_count = int(self.instance_visible[:ncells].sum())
        if visible_count == 0:
            state = QtCore.Qt.CheckState.Unchecked
        elif visible_count == ncells:
            state = QtCore.Qt.CheckState.Checked
        else:
            state = QtCore.Qt.CheckState.PartiallyChecked
        self._visibility_header.set_check_state(state)

    def _toggle_all_labels_visibility(self, state):
        if self._refreshing_labels_table:
            return
        ncells = self.ncells()
        if ncells == 0:
            return
        # Match WinUI: only explicit unchecked hides all; partial/checked show all.
        visible = state != QtCore.Qt.CheckState.Unchecked
        self._refreshing_labels_table = True
        self.LabelsTable.blockSignals(True)
        try:
            self.presenter.set_all_instance_visible(visible, ncells)
            for row in range(ncells):
                item = self.LabelsTable.item(row, 0)
                if item is not None:
                    item.setCheckState(
                        QtCore.Qt.CheckState.Checked
                        if visible
                        else QtCore.Qt.CheckState.Unchecked
                    )
        finally:
            self.LabelsTable.blockSignals(False)
            self._refreshing_labels_table = False
        self._sync_visibility_header_checkbox()

    def on_labels_table_item_changed(self, item):
        column = item.column()
        if column == 0:
            self.set_instance_visible_from_table(item)
        elif column == 2:
            self.set_instance_class_from_labels_table(item)

    def set_instance_visible_from_table(self, item):
        if self._refreshing_labels_table:
            return
        row = item.row()
        self._ensure_instance_visible()
        if row >= len(self.instance_visible):
            return
        visible = item.checkState() == QtCore.Qt.CheckState.Checked
        self.presenter.set_instance_visible_row(row, visible)
        self._sync_visibility_header_checkbox()

    def set_instance_class_from_labels_table(self, item):
        if self._refreshing_labels_table or item.column() != 2:
            return
        row = item.row()
        self._ensure_instance_classes()
        if row >= len(self.instance_classes):
            return
        old_class_id = int(self.instance_classes[row])
        try:
            class_id = int(item.text())
            if class_id < 0:
                raise ValueError
        except ValueError:
            self.LabelsTable.blockSignals(True)
            item.setText(str(old_class_id))
            self.LabelsTable.blockSignals(False)
            return
        self.presenter.set_instance_class(row, class_id)
        if self.model.session.loaded:
            io._save_sets(self)

    def toggle_saving(self):
        self.saveResults.setEnabled(self.ncells() > 0)

    def toggle_removals(self):
        if self.ncells() > 0:
            self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
            if hasattr(self, "RectSelectButton"):
                self.RectSelectButton.setEnabled(True)
            if hasattr(self, "DeleteSelectedButton"):
                self.DeleteSelectedButton.setEnabled(True)
            if hasattr(self, "EditSelectedButton"):
                self.EditSelectedButton.setEnabled(True)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)
            if hasattr(self, "RectSelectButton"):
                self.RectSelectButton.setEnabled(False)
                if self.RectSelectButton.isChecked():
                    self.RectSelectButton.setChecked(False)
            if hasattr(self, "DeleteSelectedButton"):
                self.DeleteSelectedButton.setEnabled(False)
            if hasattr(self, "EditSelectedButton"):
                self.EditSelectedButton.setEnabled(False)

    def remove_action(self):
        self.delete_selected_cells()

    def _selected_cell_indices(self):
        cells = list(self.model.selection.selected_cells)
        if not cells and self.model.selection.selected > 0:
            cells = [self.model.selection.selected]
        return sorted({int(idx) for idx in cells if int(idx) > 0})

    def delete_selected_cells(self):
        cells = self._selected_cell_indices()
        if cells:
            self.remove_cell(cells)

    def edit_selected_cells(self):
        cells = self._selected_cell_indices()
        if not cells:
            QMessageBox.information(self, "Edit class", "Select one or more cells first.")
            return

        self._ensure_instance_classes()
        rows = [idx - 1 for idx in cells]
        current_classes = [
            int(self.instance_classes[row])
            for row in rows
            if 0 <= row < len(self.instance_classes)
        ]
        initial = (
            str(current_classes[0])
            if current_classes and len(set(current_classes)) == 1
            else ""
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit class")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Class ID for selected cell(s):"))
        class_edit = QLineEdit(initial)
        class_edit.setValidator(
            QtGui.QIntValidator(0, np.iinfo(np.int32).max, dialog)
        )
        layout.addWidget(class_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        class_edit.returnPressed.connect(dialog.accept)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            class_id = int(class_edit.text())
            if class_id < 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Edit class", "Enter a non-negative integer class ID.")
            return

        for idx in cells:
            row = idx - 1
            if row >= 0:
                self.presenter.set_instance_class(row, class_id)
        if self.model.session.loaded:
            io._save_sets(self)

    def undo_action(self):
        if len(self.model.drawing.strokes) > 0 and self.model.drawing.strokes[-1][0][0] == self.model.session.current_z:
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells() > 0:
                self.remove_cell(self.ncells())

    def undo_remove_action(self):
        self.undo_remove_cell()

    def set_series_navigation_state(self, dataset=None, record_index=None):
        nav = self.presenter._series_nav_state(dataset, record_index)
        self.set_series_navigation(nav)

    def _commit_series_slider(self, axis_name=None):
        if (
            self._updating_series_navigation
            or self.model.series_state.dataset is None
            or self.model.series_state.record_index is None
        ):
            return
        if axis_name is None:
            return
        control = self.series_nav_controls.get(axis_name)
        if control is None:
            return
        self.navigate_series_from_sliders(axis_name)

    def navigate_series_from_sliders(self, axis_name=None, delta=0):
        self.presenter.navigate_series_from_sliders(axis_name, delta)

    def get_files(self):
        return self.presenter.get_files()

    def get_prev_image(self):
        self.presenter.get_prev_image()

    def get_next_image(self, load_seg=True):
        self.presenter.get_next_image(load_seg=load_seg)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == ".cellpose":
            io._load_seg(self, filename=files[0])
        else:
            io._load_image(self, filename=files[0], load_seg=True)

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
        self.model.session.ly, self.model.session.lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)
        self.selection_box = None
        self.rect_select_preview = None

    def reset(self):
        self.presenter.reset_session()

    def clear_all(self):
        self.presenter.clear_all()

    def draw_layer(self):
        self.presenter.refresh_mask_layer()

    def update_layer(self):
        self.presenter.refresh_mask_layer()

    def compute_saturation(self):
        self.presenter.compute_saturation()

    def get_segmentation_parameters(self):
        return self.presenter.segmentation_parameters_dict()

    def get_normalize_params(self):
        return self.presenter.get_normalize_params()

    def set_normalize_params(self, normalize_params):
        self.presenter.set_normalize_params(normalize_params)

    def initialize_model(self, model_name=None, custom=False):
        self.presenter.initialize_model(model_name=model_name, custom=custom)

    def add_model(self):
        self.presenter.add_model()

    def remove_model(self):
        self.presenter.remove_model()

    def new_model(self):
        self.presenter.train_new_model()

    def delete_restore(self):
        self.model.discard_filtered_stack()

    def clear_restore(self):
        print("GUI_INFO: clearing restored image")
        self.set_view_mode(0, restored_enabled=False)
        self.delete_restore()
        self.model.session.restore = None
        self.model.session.ratio = 1.0
        self.set_normalize_params(self.get_normalize_params())

    def _cell_bounds(self, idx, margin=2):
        cellpix = self.model.session.cellpix[self.model.session.current_z]
        mask = cellpix == idx
        if not np.any(mask):
            return None
        ar, ac = np.nonzero(mask)
        y0 = max(0, int(ar.min()) - margin)
        y1 = min(self.model.session.ly - 1, int(ar.max()) + margin)
        x0 = max(0, int(ac.min()) - margin)
        x1 = min(self.model.session.lx - 1, int(ac.max()) + margin)
        return x0, y0, x1, y1

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

    def _update_selection_boxes(self):
        if not self.model.session.loaded:
            self._clear_selection_boxes()
            return

        indices = self.model.selection.selected_cells if self.model.selection.selected_cells else (
            [self.model.selection.selected] if self.model.selection.selected > 0 else []
        )
        if not indices:
            self._clear_selection_boxes()
            return

        xs = []
        ys = []
        for idx in indices:
            bounds = self._cell_bounds(idx)
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
            xs.extend([x0, x1, x1, x0, x0, np.nan])
            ys.extend([y0, y0, y1, y1, y0, np.nan])

        if not xs:
            self._clear_selection_boxes()
            return

        self._ensure_selection_box()
        self.selection_box.setData(
            np.array(xs[:-1], dtype=float),
            np.array(ys[:-1], dtype=float),
            connect="finite",
        )

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
        x0 = max(0, min(self.model.session.lx - 1, x0))
        x1 = max(0, min(self.model.session.lx, x1))
        y0 = max(0, min(self.model.session.ly - 1, y0))
        y1 = max(0, min(self.model.session.ly, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _set_rect_preview(self, x0, y0, x1, y1):
        bounds = self._normalize_rect(x0, y0, x1, y1)
        if bounds is None:
            self._clear_rect_select_preview()
            return
        x0, y0, x1, y1 = bounds
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        self._ensure_rect_select_preview()
        self.rect_select_preview.setData(xs, ys)

    def _cells_fully_in_rect(self, x0, y0, x1, y1):
        return self.model.cells_in_rect(
            x0,
            y0,
            x1,
            y1,
            filter_class_id=self.labels_class_filter(),
        )

    def _sync_labels_table_selection(self, idx):
        if not hasattr(self, "LabelsTable") or idx - 1 >= self.LabelsTable.rowCount():
            return
        self._sync_labels_table_selection_multi([idx])

    def _sync_labels_table_selection_multi(self, indices):
        if not hasattr(self, "LabelsTable"):
            return
        selection_model = self.LabelsTable.selectionModel()
        if selection_model is None:
            return

        self._syncing_labels_table_selection = True
        selection_model.blockSignals(True)
        selection_model.clearSelection()
        for idx in indices:
            row = int(idx) - 1
            if 0 <= row < self.LabelsTable.rowCount():
                index = self.LabelsTable.model().index(row, 0)
                selection_model.select(
                    index,
                    QtCore.QItemSelectionModel.SelectionFlag.Select
                    | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                )
        selection_model.blockSignals(False)
        self._syncing_labels_table_selection = False

        if indices:
            first_row = int(indices[0]) - 1
            if 0 <= first_row < self.LabelsTable.rowCount():
                item = self.LabelsTable.item(first_row, 0)
                if item is not None:
                    self.LabelsTable.scrollToItem(
                        item,
                        QAbstractItemView.ScrollHint.EnsureVisible,
                    )

    def _apply_cell_selection(self, cells):
        self.model.selection.selected_cells = cells
        self.model.selection.selected = cells[0] if cells else 0
        self.model.selection.prev_selected = self.model.selection.selected
        self._sync_labels_table_selection_multi(cells)
        self._update_selection_boxes()

    def _update_canvas_cursor(self):
        if not hasattr(self, "p0"):
            return
        if not self.model.session.loaded:
            cursor = QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor)
        elif self.model.drawing.brush_mode:
            cursor = brush_cursor()
        elif self.rect_select_mode:
            cursor = select_cursor()
        else:
            cursor = QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.p0.setCursor(cursor)
        self.win.setCursor(cursor)

    def toggle_brush_mode(self, enabled):
        if enabled and self.rect_select_mode:
            self.RectSelectButton.setChecked(False)
        self.model.drawing.brush_mode = enabled
        self._update_canvas_cursor()
        if enabled or not self.model.drawing.in_stroke:
            return

        if hasattr(self.layer, "scatter"):
            self.p0.removeItem(self.layer.scatter)
        drawing = self.model.drawing
        drawing.in_stroke = False
        drawing.current_stroke = []
        drawing.stroke_appended = True
        self.presenter.refresh_mask_layer()

    def toggle_rect_select_mode(self, enabled):
        if enabled and self.model.drawing.brush_mode:
            self.BrushButton.setChecked(False)
        self.rect_select_mode = enabled
        if not enabled:
            self._rect_select_start = None
            self._rect_select_dragging = False
            self._clear_rect_select_preview()
        self._update_canvas_cursor()

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
        if (
            obj is self._canvas_viewport()
            and self.rect_select_mode
            and self.model.session.loaded
            and not self.model.selection.removing_region
            and not self.model.selection.deleting_multiple
        ):
            event_type = event.type()
            if (
                event_type == QtCore.QEvent.Type.MouseButtonPress
                and event.button() == QtCore.Qt.MouseButton.LeftButton
            ):
                pos = self._view_pos_from_viewport_event(event)
                if pos is not None:
                    self.begin_rect_select(pos)
                return True
            if (
                event_type == QtCore.QEvent.Type.MouseMove
                and event.buttons() & QtCore.Qt.MouseButton.LeftButton
                and self._rect_select_start is not None
            ):
                pos = self._view_pos_from_viewport_event(event)
                if pos is not None:
                    self.update_rect_select(pos)
                return True
            if (
                event_type == QtCore.QEvent.Type.MouseButtonRelease
                and event.button() == QtCore.Qt.MouseButton.LeftButton
                and self._rect_select_start is not None
            ):
                pos = self._view_pos_from_viewport_event(event)
                if pos is not None:
                    self.finish_rect_select(pos, event.modifiers())
                return True
        return super().eventFilter(obj, event)

    def begin_rect_select(self, pos):
        self._rect_select_start = (int(pos.y()), int(pos.x()))
        self._rect_select_dragging = False

    def update_rect_select(self, pos):
        if self._rect_select_start is None:
            return
        y0, x0 = self._rect_select_start
        y1, x1 = int(pos.y()), int(pos.x())
        if not self._rect_select_dragging:
            if (
                max(abs(y1 - y0), abs(x1 - x0))
                < self._rect_select_drag_threshold
            ):
                return
            self._rect_select_dragging = True
        self._set_rect_preview(x0, y0, x1, y1)

    def finish_rect_select(self, pos, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
        if self._rect_select_start is None:
            return
        y0, x0 = self._rect_select_start
        y1, x1 = int(pos.y()), int(pos.x())
        dragging = self._rect_select_dragging
        additive = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._rect_select_start = None
        self._rect_select_dragging = False
        self._clear_rect_select_preview()
        if dragging:
            cells = self._cells_fully_in_rect(x0, y0, x1, y1)
            if additive:
                cells = sorted(
                    set(self.model.selection.selected_cells) | set(cells)
                )
            self._apply_cell_selection(cells)
        else:
            self._select_cell_at_click(y1, x1, additive=additive)

    def _select_cell_at_click(self, y, x, additive=False):
        session = self.model.session
        if y < 0 or y >= session.ly or x < 0 or x >= session.lx:
            return
        idx = int(session.cellpix[session.current_z, y, x])
        if idx > 0:
            if additive:
                cells = list(self.model.selection.selected_cells)
                if idx not in cells:
                    cells.append(idx)
                self._apply_cell_selection(sorted(cells))
            else:
                self.select_cell(idx)
        elif not additive:
            self.unselect_cell()

    def select_cell(self, idx):
        self.model.selection.prev_selected = self.model.selection.selected
        self.model.selection.selected = idx
        self.model.selection.selected_cells = [idx] if idx > 0 else []
        if self.model.selection.selected > 0:
            self._sync_labels_table_selection(idx)
            self._update_selection_boxes()
        else:
            self._clear_selection_boxes()

    def select_cell_from_labels_table(self):
        if self._refreshing_labels_table or self._syncing_labels_table_selection:
            return
        selected_rows = self.LabelsTable.selectionModel().selectedRows()
        if not selected_rows:
            if self.model.selection.selected > 0 or self.model.selection.selected_cells:
                self.model.selection.selected = 0
                self.model.selection.selected_cells = []
                self._clear_selection_boxes()
            return
        cells = sorted({row.row() + 1 for row in selected_rows})
        if cells == self.model.selection.selected_cells:
            self._update_selection_boxes()
            return
        self.model.selection.prev_selected = self.model.selection.selected
        self.model.selection.selected_cells = cells
        self.model.selection.selected = cells[0] if cells else 0
        self._update_selection_boxes()

    def select_cell_multi(self, idx):
        if idx > 0:
            z = self.model.session.current_z
            self.model.session.layerz[self.model.session.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.model.session.opacity]
            )
            self.update_layer()

    def unselect_cell(self):
        self.model.selection.selected = 0
        self.model.selection.selected_cells = []
        self._clear_selection_boxes()
        if hasattr(self, "LabelsTable"):
            selection_model = self.LabelsTable.selectionModel()
            if selection_model is not None:
                self._syncing_labels_table_selection = True
                selection_model.blockSignals(True)
                selection_model.clearSelection()
                selection_model.blockSignals(False)
                self._syncing_labels_table_selection = False

    def unselect_cell_multi(self, idx):
        z = self.model.session.current_z
        self.model.session.layerz[self.model.session.cellpix[z] == idx] = np.append(
            self.model.session.cellcolors[idx], self.model.session.opacity
        )
        self.model.session.layerz[self.model.session.outpix[z] == idx] = np.array(self.model.session.outcolor).astype(np.uint8)
        self.update_layer()

    def remove_cell(self, idx):
        self.presenter.remove_cell(idx)

    def remove_single_cell(self, idx):
        self.presenter.remove_cell(idx)

    def remove_region_cells(self):
        if self.model.selection.removing_cells_list:
            for idx in self.model.selection.removing_cells_list:
                self.unselect_cell_multi(idx)
            self.model.selection.removing_cells_list.clear()
        self.disable_buttons_removeROIs()
        self.model.selection.removing_region = True

        self.clear_multi_selected_cells()

        # make roi region here in center of view, making ROI half the size of the view
        roi_width = self.p0.viewRect().width() / 2
        x_loc = self.p0.viewRect().x() + (roi_width / 2)
        roi_height = self.p0.viewRect().height() / 2
        y_loc = self.p0.viewRect().y() + (roi_height / 2)

        pos = [x_loc, y_loc]
        roi = pg.RectROI(
            pos, [roi_width, roi_height], pen=pg.mkPen("y", width=2), removable=True
        )
        roi.sigRemoveRequested.connect(self.remove_roi)
        roi.sigRegionChangeFinished.connect(self.roi_changed)
        self.p0.addItem(roi)
        self.remove_roi_obj = roi
        self.roi_changed(roi)

    def delete_multiple_cells(self):
        self.unselect_cell()
        self.disable_buttons_removeROIs()
        self.model.selection.deleting_multiple = True

    def done_remove_multiple_cells(self):
        self.model.selection.deleting_multiple = False
        self.model.selection.removing_region = False

        if self.model.selection.removing_cells_list:
            self.model.selection.removing_cells_list = list(set(self.model.selection.removing_cells_list))
            display_remove_list = [i - 1 for i in self.model.selection.removing_cells_list]
            print(f"GUI_INFO: removing cells: {display_remove_list}")
            self.remove_cell(self.model.selection.removing_cells_list)
            self.model.selection.removing_cells_list.clear()
            self.unselect_cell()
        self.enable_buttons()

        if self.remove_roi_obj is not None:
            self.remove_roi(self.remove_roi_obj)

    def merge_cells(self, idx):
        self.presenter.merge_cells(idx)

    def undo_remove_cell(self):
        self.presenter.undo_remove_cell()

    def remove_stroke(self, delete_points=True, stroke_ind=-1):
        self.presenter.remove_stroke(delete_points=delete_points, stroke_ind=stroke_ind)

    def plot_clicked(self, event):
        if (
            event.button() == QtCore.Qt.LeftButton
            and not event.modifiers()
            & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)
            and not self.model.selection.removing_region
        ):
            if event.double():
                try:
                    self.p0.setYRange(0, self.model.session.ly + self.pr)
                except:
                    self.p0.setYRange(0, self.model.session.ly)
                self.p0.setXRange(0, self.model.session.lx)

    def cancel_remove_multiple(self):
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self):
        # unselect all previously selected cells:
        for idx in self.model.selection.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.model.selection.removing_cells_list.clear()

    def add_roi(self, roi):
        self.p0.addItem(roi)
        self.remove_roi_obj = roi

    def remove_roi(self, roi):
        self.clear_multi_selected_cells()
        assert roi == self.remove_roi_obj
        self.remove_roi_obj = None
        self.p0.removeItem(roi)
        self.model.selection.removing_region = False

    def roi_changed(self, roi):
        # find the overlapping cells and make them selected
        pos = roi.pos()
        size = roi.size()
        x0 = int(pos.x())
        y0 = int(pos.y())
        x1 = int(pos.x() + size.x())
        y1 = int(pos.y() + size.y())
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > self.model.session.lx:
            x1 = self.model.session.lx
        if y1 > self.model.session.ly:
            y1 = self.model.session.ly

        # find cells in that region
        cell_idxs = np.unique(self.model.session.cellpix[self.model.session.current_z, y0:y1, x0:x1])
        cell_idxs = np.trim_zeros(cell_idxs)
        # deselect cells not in region by deselecting all and then selecting the ones in the region
        self.clear_multi_selected_cells()

        for idx in cell_idxs:
            self.select_cell_multi(idx)
            self.model.selection.removing_cells_list.append(idx)

        self.update_layer()

    def mouse_moved(self, pos):
        items = self.win.scene().items(pos)

    def update_plot(self):
        self.view = self.ViewDropDown.currentIndex()
        self.model.session.ly, self.model.session.lx, _ = self.model.session.stack[self.model.session.current_z].shape

        if self.view == 0 or self.view == self.ViewDropDown.count() - 1:
            image = (
                self.model.session.stack[self.model.session.current_z]
                if self.view == 0
                else self.model.session.stack_filtered[self.model.session.current_z]
            )
            self.img.setImage(as_gray_image(image), autoLevels=False, lut=None)
            self.img.setLevels(self.model.session.saturation[0][self.model.session.current_z])
        else:
            image = np.zeros((self.model.session.ly, self.model.session.lx), np.uint8)
            if len(self.model.session.flows) >= self.view - 1 and len(self.model.session.flows[self.view - 1]) > 0:
                image = self.model.session.flows[self.view - 1][self.model.session.current_z]
            if self.view > 1:
                self.img.setImage(as_gray_image(image), autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(as_gray_image(image), autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])

        self.sliders[0].setValue(
            [
                self.model.session.saturation[0][self.model.session.current_z][0],
                self.model.session.saturation[0][self.model.session.current_z][1],
            ]
        )
        self.win.show()
        self.show()

    def update_layer(self):
        self.layer.setImage(self.model.session.layerz, autoLevels=False)
        self._update_selection_boxes()
        self.win.show()
        self.show()

    def add_set(self):
        self.presenter.add_set()

    def add_mask(self, points=None, color=(100, 200, 50), dense=True):
        from .presenter_masks import add_mask_from_points

        return add_mask_from_points(self.model, points, color)

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        from .presenter_masks import paint_mask_at

        paint_mask_at(self.model, z, ar, ac, vr, vc, color, idx=idx)

    def compute_scale(self):
        self.presenter.refresh_scale_from_model()

    def update_scale(self):
        self.presenter.refresh_scale_from_model()

    def get_training_image_files(self, folder, nested=True):
        if not folder:
            return []
        return get_image_files(folder, "_masks", look_one_level_down=nested)

    def _get_train_dataset(self, folder, nested=True):
        image_names = self.get_training_image_files(folder, nested=nested)
        return io._get_train_set(image_names)

    def compute_cprob(self):
        self.presenter.compute_cprob()

    def compute_segmentation(self, custom=False, model_name=None, load_model=True):
        self.presenter.compute_segmentation(
            custom=custom, model_name=model_name, load_model=load_model
        )


# Backward-compatible alias used by older scripts and docs.
MainW = MainView
