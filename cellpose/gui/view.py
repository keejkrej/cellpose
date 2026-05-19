"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import copy
import datetime
import os
import pathlib
import sys
import time
import warnings

import cv2
import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication,
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

from .. import dynamics, models, train, version
from ..io import get_image_files
from ..models import normalize_default
from ..plot import disk
from ..transforms import normalize99, normalize99_tile, resize_image, smooth_sharpen_img
from ..utils import download_url_to_file
from . import io, menus, series
from .dialogs import TrainWindow
from .viewmodel import MainViewModel, ObservableVariable
from .widgets import (
    ImageDraw,
    SeriesAxisSlider,
    Slider,
    ViewBoxNoRightDrag,
    as_gray_image,
    make_bwr,
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
    MainW(image=image, logger=logger)
    ret = app.exec()
    sys.exit(ret)


class MainW(QMainWindow):
    def __init__(self, image=None, logger=None):
        super(MainW, self).__init__()

        self.logger = logger
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

        self.view_model = MainViewModel(
            model_save_folder=os.fspath(models.MODEL_DIR.joinpath("custom")),
            parent=self,
        )

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)

        self.loaded = False
        self.recompute_masks = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.lmain = QHBoxLayout()
        self.cwidget.setLayout(self.lmain)
        self.setCentralWidget(self.cwidget)
        self.lmain.setContentsMargins(0, 0, 0, 10)

        self.imask = 0
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
        self.NZ = 1
        self.restore = None
        self.ratio = 1.0
        self._sync_series_state()
        self.last_series_subfolder_template = ""
        self.last_series_filename_template = ""
        self.reset()

        self.load_3D = False

        # if called with image, load it
        if image is not None:
            self.filename = image
            io._load_image(self, self.filename)

        self.stitch_threshold = 0.0
        self.flow3D_smooth = 0.0
        self.anisotropy = 1.0
        self.min_size = 15

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    @property
    def training_params(self):
        return self.view_model.training_params

    @training_params.setter
    def training_params(self, params):
        self.view_model.set_training_parameters(params)

    def _sync_series_state(self):
        state = self.view_model.series_state
        self.series_dataset = state.dataset
        self.series_index = state.record_index
        self.output_filename = state.output_filename
        self.display_filename = state.display_filename
        if state.filename is not None:
            self.filename = state.filename

    def set_series_state(self, dataset=None, record_index=None):
        self.view_model.set_series(dataset=dataset, record_index=record_index)
        self._sync_series_state()
        self.set_series_navigation_state(self.series_dataset, self.series_index)

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
            slider.keyboardRelease.connect(
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

        self.norm3DCheckBox = QCheckBox("norm3D")
        self.norm3DCheckBox.setChecked(True)
        self.satBoxV.addWidget(self.norm3DCheckBox)

        self.autoSaturationButton = QPushButton("auto saturation")
        self.autoSaturationButton.setEnabled(False)
        self.autoSaturationButton.clicked.connect(self.compute_saturation)
        self.satBoxV.addWidget(self.autoSaturationButton)

        self.sliders = []
        gray_slider_layout = QHBoxLayout()
        label = QLabel("gray:")
        gray_slider_layout.addWidget(label)
        self.sliders.append(Slider(self, "gray", [100, 100, 100]))
        self.sliders[-1].setMinimum(-0.1)
        self.sliders[-1].setMaximum(255.1)
        self.sliders[-1].setValue([0, 255])
        gray_slider_layout.addWidget(self.sliders[-1], 1)
        self.satBoxV.addLayout(gray_slider_layout)

        b += 1
        self.drawBox = QGroupBox("Drawing")
        self.drawBoxV = QVBoxLayout()
        self.drawBox.setLayout(self.drawBoxV)
        self.right_sidebar.addWidget(self.drawBox, 1, 0, 1, 1)
        self.autosave = True

        self.brush_size = 1
        draw_checks = QHBoxLayout()
        self.drawBoxV.addLayout(draw_checks)

        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox("masks")
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        draw_checks.addWidget(self.MCheckBox)

        # turn off outlines
        self.outlinesOn = True
        self.OCheckBox = QCheckBox("outlines")
        draw_checks.addWidget(self.OCheckBox)
        self.OCheckBox.setChecked(True)
        self.OCheckBox.toggled.connect(self.toggle_masks)

        self.SCheckBox = QCheckBox("auto")
        self.SCheckBox.setChecked(True)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.SCheckBox.setEnabled(True)
        draw_checks.addWidget(self.SCheckBox)

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

        self.ncells = ObservableVariable(0)
        self.ncells.valueChanged.connect(lambda *_: self.refresh_instance_table())

        self.instanceBox = QGroupBox("Instances")
        self.instanceBoxV = QVBoxLayout()
        self.instanceBox.setLayout(self.instanceBoxV)
        self.right_sidebar.addWidget(self.instanceBox, 2, 0, 1, 1)

        instance_filter_layout = QHBoxLayout()
        instance_filter_layout.addWidget(QLabel("class filter:"))
        self.InstanceClassFilter = QLineEdit()
        self.InstanceClassFilter.setPlaceholderText("all")
        self.InstanceClassFilter.returnPressed.connect(self.instance_filter_changed)
        instance_filter_layout.addWidget(self.InstanceClassFilter)
        self.instanceBoxV.addLayout(instance_filter_layout)

        self._refreshing_instance_table = False
        self.InstanceTable = QTableWidget(0, 2)
        self.InstanceTable.setHorizontalHeaderLabels(["ROI", "Class ID"])
        self.InstanceTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.InstanceTable.verticalHeader().setVisible(False)
        self.InstanceTable.itemChanged.connect(self.set_instance_class_from_table)
        self.instanceBoxV.addWidget(self.InstanceTable)

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

        self.filterBox = QGroupBox("Preprocessing")
        self.filterBoxV = QVBoxLayout()
        self.filterBox.setLayout(self.filterBoxV)
        self.left_sidebar.addWidget(self.filterBox, b, 0, 1, 9)

        self.restore = None
        self.ratio = 1.0
        filter_buttons_layout = QHBoxLayout()
        self.reset_filter_button = QPushButton("reset")
        self.reset_filter_button.setEnabled(False)
        self.reset_filter_button.clicked.connect(self.clear_restore)
        filter_buttons_layout.addWidget(self.reset_filter_button)
        self.apply_filter_button = QPushButton("apply")
        self.apply_filter_button.setEnabled(False)
        self.apply_filter_button.clicked.connect(self.apply_filter)
        filter_buttons_layout.addWidget(self.apply_filter_button)

        self.save_norm = QCheckBox("save restored/filtered image")
        self.save_norm.setChecked(True)

        self.preprocessing_param_root = Parameter.create(
            name="preprocessing",
            type="group",
            children=[
                {
                    "name": "sharpen_radius",
                    "title": "sharpen radius",
                    "type": "float",
                    "value": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "smooth_radius",
                    "title": "smooth radius",
                    "type": "float",
                    "value": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "tile_norm_blocksize",
                    "title": "tile norm blocksize",
                    "type": "float",
                    "value": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "tile_norm_smooth3D",
                    "title": "tile norm smooth3D",
                    "type": "float",
                    "value": 0.0,
                    "step": 1.0,
                },
            ],
        )
        self.preprocessing_params_tree = ParameterTree(showHeader=False)
        self.preprocessing_params_tree.setParameters(
            self.preprocessing_param_root, showTop=False
        )
        self.filterBoxV.addWidget(self.preprocessing_params_tree)
        self.filterBoxV.addLayout(filter_buttons_layout)

        return b

    def validate_normalization_range(self):
        try:
            self.get_segmentation_parameters()
        except ValueError:
            print("GUI_ERROR: normalization percentile lower must be less than upper")
            return

    def get_segmentation_parameters(self):
        params = self.view_model.get_segmentation_parameters(
            diameter=float(self.seg_param_root.param("diameter").value()),
            flow_threshold=float(self.seg_param_root.param("flow_threshold").value()),
            cellprob_threshold=float(
                self.seg_param_root.param("cellprob_threshold").value()
            ),
            percentile_low=float(
                self.seg_param_root.param("norm_percentile_low").value()
            ),
            percentile_high=float(
                self.seg_param_root.param("norm_percentile_high").value()
            ),
            niter=int(self.seg_param_root.param("niter").value()),
        )
        low, high = params.percentile

        if self.seg_param_root.param("norm_percentile_low").value() != low:
            self.seg_param_root.param("norm_percentile_low").setValue(low)
        if self.seg_param_root.param("norm_percentile_high").value() != high:
            self.seg_param_root.param("norm_percentile_high").setValue(high)
        if self.seg_param_root.param("niter").value() != params.niter:
            self.seg_param_root.param("niter").setValue(params.niter)
        return params.to_dict()

    def get_preprocessing_parameters(self):
        params = self.view_model.get_preprocessing_parameters(
            sharpen_radius=float(
                self.preprocessing_param_root.param("sharpen_radius").value()
            ),
            smooth_radius=float(
                self.preprocessing_param_root.param("smooth_radius").value()
            ),
            tile_norm_blocksize=float(
                self.preprocessing_param_root.param("tile_norm_blocksize").value()
            ),
            tile_norm_smooth3D=float(
                self.preprocessing_param_root.param("tile_norm_smooth3D").value()
            ),
            norm3D=self.norm3DCheckBox.isChecked(),
            image_shape=(self.Ly, self.Lx),
            invert=False,
        )
        return params.to_dict()

    def set_preprocessing_parameters(self, params):
        self.preprocessing_param_root.param("sharpen_radius").setValue(
            params["sharpen_radius"]
        )
        self.preprocessing_param_root.param("smooth_radius").setValue(
            params["smooth_radius"]
        )
        self.preprocessing_param_root.param("tile_norm_blocksize").setValue(
            params["tile_norm_blocksize"]
        )
        self.preprocessing_param_root.param("tile_norm_smooth3D").setValue(
            params["tile_norm_smooth3D"]
        )
        self.norm3DCheckBox.setChecked(bool(params["norm3D"]))

    def level_change(self, r):
        if self.loaded:
            sval = self.sliders[0].value()
            self.saturation[0][self.currentZ] = sval
            self.update_plot()

    def keyPressEvent(self, event):
        if self.loaded:
            if not (
                event.modifiers()
                & (
                    QtCore.Qt.ControlModifier
                    | QtCore.Qt.ShiftModifier
                    | QtCore.Qt.AltModifier
                )
                or self.in_stroke
            ):
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                else:
                    nviews = self.ViewDropDown.count() - 1
                    nviews += int(
                        self.ViewDropDown.model()
                        .item(self.ViewDropDown.count() - 1)
                        .isEnabled()
                    )
                    if (
                        event.key() == QtCore.Qt.Key_Left
                        or event.key() == QtCore.Qt.Key_A
                    ):
                        self.get_prev_image()
                    elif (
                        event.key() == QtCore.Qt.Key_Right
                        or event.key() == QtCore.Qt.Key_D
                    ):
                        self.get_next_image()
                    elif event.key() == QtCore.Qt.Key_PageDown:
                        self.view = (self.view + 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)
                    elif event.key() == QtCore.Qt.Key_PageUp:
                        self.view = (self.view - 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)

                if not updated:
                    self.update_plot()
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def autosave_on(self):
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def default_class_id(self):
        text = self.DefaultClassEdit.text().strip()
        return int(text) if text else 0

    def _selected_segmentation_model(self):
        model_name = self.ModelChooseC.currentText().strip()
        is_cpsam = model_name.lower() == "cpsam"
        return "cpsam" if is_cpsam else model_name, not is_cpsam

    def run_selected_model(self):
        model_name, custom = self._selected_segmentation_model()
        if custom:
            self.compute_segmentation(custom=True)
        else:
            self.compute_segmentation(model_name=model_name)

    def apply_filter(self):
        self.restore = "filter"
        normalize_params = self.get_normalize_params()
        if (
            normalize_params["sharpen_radius"] == 0
            and normalize_params["smooth_radius"] == 0
            and normalize_params["tile_norm_blocksize"] == 0
        ):
            print("GUI_ERROR: no filtering settings on (use custom filter settings)")
            self.restore = None
            return
        self.compute_saturation(apply_preprocessing=True)

    def model_choose(self, custom=False):
        if custom:
            model_name, is_custom = self._selected_segmentation_model()
            if model_name:
                print(f"GUI_INFO: selected model {model_name}, loading now")
                self.initialize_model(model_name=model_name, custom=is_custom)
            return

        index = self.ModelChooseB.currentIndex()
        if index > 0:
            model_name = self.net_names[index - 1]
            print(f"GUI_INFO: selected model {model_name}, loading now")
            self.initialize_model(model_name=model_name, custom=custom)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def enable_buttons(self):
        self.ModelButtonC.setEnabled(True)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(True)

        self.reset_filter_button.setEnabled(not self.load_3D)
        self.apply_filter_button.setEnabled(True)
        self.autoSaturationButton.setEnabled(True)

        self.newmodel.setEnabled(True)
        self.loadMasks.setEnabled(True)

        self.sliders[0].setEnabled(True)

        self.toggle_mask_ops()

        self.update_plot()
        self.setWindowTitle(self.filename)

    def disable_buttons_removeROIs(self):
        self.ModelButtonC.setEnabled(False)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(False)
        self.newmodel.setEnabled(False)
        self.loadMasks.setEnabled(False)
        self.saveSet.setEnabled(False)
        self.savePNG.setEnabled(False)
        self.saveFlows.setEnabled(False)
        self.saveOutlines.setEnabled(False)
        self.saveROIs.setEnabled(False)

    def toggle_mask_ops(self):
        self.update_layer()
        self.toggle_saving()
        self.toggle_removals()
        self.refresh_instance_table()

    def _ensure_instance_classes(self):
        ncells = self.ncells.get()
        current_values = getattr(self, "instance_classes", None)
        self.instance_classes = self.view_model.ensure_instance_classes(
            ncells, current_values=current_values
        )

    def set_instance_classes(self, instance_classes=None):
        self.instance_classes = self.view_model.set_instance_classes(
            self.ncells.get(), instance_classes
        )
        self.refresh_instance_table()

    def instance_class_filter(self):
        if not hasattr(self, "InstanceClassFilter"):
            return None
        return self.view_model.instance_class_filter(self.InstanceClassFilter.text())

    def instance_filter_changed(self):
        self.refresh_instance_table()
        self.draw_layer()
        self.update_layer()

    def visible_cell_pixels(self, cellpix):
        filter_class_id = self.instance_class_filter()
        self._ensure_instance_classes()
        return self.view_model.visible_cell_pixels(cellpix, filter_class_id)

    def refresh_instance_table(self):
        if not hasattr(self, "InstanceTable"):
            return
        self._ensure_instance_classes()
        filter_class_id = self.instance_class_filter()
        self._refreshing_instance_table = True
        self.InstanceTable.blockSignals(True)
        self.InstanceTable.setRowCount(self.ncells.get())
        for row in range(self.ncells.get()):
            roi_item = QTableWidgetItem(str(row + 1))
            roi_item.setFlags(roi_item.flags() & ~QtCore.Qt.ItemIsEditable)
            class_item = QTableWidgetItem(str(int(self.instance_classes[row])))
            self.InstanceTable.setItem(row, 0, roi_item)
            self.InstanceTable.setItem(row, 1, class_item)
            self.InstanceTable.setRowHidden(
                row,
                filter_class_id is not None
                and int(self.instance_classes[row]) != filter_class_id,
            )
        self.InstanceTable.blockSignals(False)
        self._refreshing_instance_table = False

    def set_instance_class_from_table(self, item):
        if self._refreshing_instance_table or item.column() != 1:
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
            self.InstanceTable.blockSignals(True)
            item.setText(str(old_class_id))
            self.InstanceTable.blockSignals(False)
            return
        self.instance_classes = self.view_model.set_instance_class(row, class_id)
        self.refresh_instance_table()
        self.draw_layer()
        self.update_layer()
        if self.loaded:
            io._save_sets_with_check(self)

    def toggle_saving(self):
        if self.ncells > 0:
            self.saveSet.setEnabled(True)
            self.savePNG.setEnabled(True)
            self.saveFlows.setEnabled(True)
            self.saveOutlines.setEnabled(True)
            self.saveROIs.setEnabled(True)
        else:
            self.saveSet.setEnabled(False)
            self.savePNG.setEnabled(False)
            self.saveFlows.setEnabled(False)
            self.saveOutlines.setEnabled(False)
            self.saveROIs.setEnabled(False)

    def toggle_removals(self):
        if self.ncells > 0:
            self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)

    def remove_action(self):
        if self.selected > 0:
            self.remove_cell(self.selected)

    def undo_action(self):
        if len(self.strokes) > 0 and self.strokes[-1][0][0] == self.currentZ:
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells > 0:
                self.remove_cell(self.ncells.get())

    def undo_remove_action(self):
        self.undo_remove_cell()

    def set_series_navigation_state(self, dataset=None, record_index=None):
        self._updating_series_navigation = True
        try:
            enabled = dataset is not None and record_index is not None
            self.navBox.setEnabled(enabled)
            for axis_name, control in self.series_nav_controls.items():
                slider = control["slider"]
                prev_btn = control["prev_btn"]
                next_btn = control["next_btn"]
                if not enabled:
                    slider.setRange(0, 0)
                    slider.setValue(0)
                    slider.setEnabled(False)
                    prev_btn.setEnabled(False)
                    next_btn.setEnabled(False)
                    continue

                axis_values = dataset["axes"][axis_name]
                slider.setEnabled(True)
                slider.setRange(0, max(0, len(axis_values) - 1))
                slider.setValue(
                    dataset["axis_index"][axis_name][
                        dataset["records"][record_index][axis_name]
                    ]
                )
                prev_btn.setEnabled(slider.maximum() > 0)
                next_btn.setEnabled(slider.maximum() > 0)
        finally:
            self._updating_series_navigation = False

    def _commit_series_slider(self, axis_name=None):
        if (
            self._updating_series_navigation
            or self.series_dataset is None
            or self.series_index is None
        ):
            return
        if axis_name is None:
            return
        control = self.series_nav_controls.get(axis_name)
        if control is None:
            return
        self.navigate_series_from_sliders(axis_name)

    def navigate_series_from_sliders(self, axis_name=None, delta=0):
        if delta != 0:
            control = self.series_nav_controls.get(axis_name)
            if control is None:
                return
            slider = control["slider"]
            if not slider.isEnabled():
                return
            value = max(0, min(slider.maximum(), slider.value() + delta))
            old_updating_state = self._updating_series_navigation
            self._updating_series_navigation = True
            try:
                slider.setValue(value)
            finally:
                self._updating_series_navigation = old_updating_state

        if (
            self._updating_series_navigation
            or self.series_dataset is None
            or self.series_index is None
        ):
            return

        if axis_name is None:
            return

        try:
            record_index = series.resolve_series_record_index(
                self.series_dataset,
                position=self.series_dataset["axes"]["position"][
                    self.series_nav_controls["position"]["slider"].value()
                ],  # Use value() with tracking enabled.
                time=self.series_dataset["axes"]["time"][
                    self.series_nav_controls["time"]["slider"].value()
                ],  # Use value() with tracking enabled.
                channel=self.series_dataset["axes"]["channel"][
                    self.series_nav_controls["channel"]["slider"].value()
                ],  # Use value() with tracking enabled.
                z=self.series_dataset["axes"]["z"][
                    self.series_nav_controls["z"]["slider"].value()
                ],  # Use value() with tracking enabled.
            )
        except Exception as e:
            self.set_series_navigation_state(self.series_dataset, self.series_index)
            QMessageBox.warning(self, "Load folder with pattern", str(e))
            return

        if record_index == self.series_index:
            return

        try:
            io._load_series_item(
                self,
                self.series_dataset,
                record_index,
                load_3D=self.load_3D,
            )
        except Exception as e:
            self.set_series_navigation_state(self.series_dataset, self.series_index)
            print(f"ERROR: {e}")
            QMessageBox.warning(self, "Load folder with pattern", str(e))

    def get_files(self):
        if self.series_dataset is not None and self.series_index is not None:
            return (
                [record["path"] for record in self.series_dataset["records"]],
                self.series_index,
            )
        folder = os.path.dirname(self.filename)
        mask_filter = "_masks"
        images = get_image_files(folder, mask_filter)
        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]
        idx = np.nonzero(np.array(fnames) == f0)[0][0]
        return images, idx

    def get_prev_image(self):
        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        if self.series_dataset is not None:
            try:
                io._load_series_item(
                    self, self.series_dataset, idx, load_3D=self.load_3D
                )
            except Exception as e:
                print(f"ERROR: {e}")
                QMessageBox.warning(self, "Load folder with pattern", str(e))
        else:
            io._load_image(self, filename=images[idx])

    def get_next_image(self, load_seg=True):
        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        if self.series_dataset is not None:
            try:
                io._load_series_item(
                    self,
                    self.series_dataset,
                    idx,
                    load_seg=load_seg,
                    load_3D=self.load_3D,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                QMessageBox.warning(self, "Load folder with pattern", str(e))
        else:
            io._load_image(self, filename=images[idx], load_seg=load_seg)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == ".npy":
            io._load_seg(self, filename=files[0], load_3D=self.load_3D)
        else:
            io._load_image(self, filename=files[0], load_seg=True, load_3D=self.load_3D)

    def toggle_masks(self):
        if self.MCheckBox.isChecked():
            self.masksOn = True
        else:
            self.masksOn = False
        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False
        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.draw_layer()
            self.update_layer()
        if self.loaded:
            self.update_plot()
            self.update_layer()

    def make_viewbox(self):
        self.p0 = ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True,
        )
        self.p0.setCursor(QtCore.Qt.CrossCursor)
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
        self.Ly, self.Lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.nchan = 3
        self.loaded = False
        self.channel = [0, 1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.resize = False
        self.ncells.reset()
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]

        # -- zero out image stack -- #
        self.opacity = 128  # how opaque masks should be
        self.outcolor = [200, 200, 255, 200]
        self.NZ, self.Ly, self.Lx = 1, 256, 256
        self.saturation = [[[0, 255] for n in range(self.NZ)]]

        self.sliders[0].setValue([0, 255])
        self.sliders[0].setEnabled(False)
        self.sliders[0].show()
        self.currentZ = 0
        self.flows = [[], [], [], [], [[]]]
        # masks matrix
        # image matrix with a scale disk
        self.stack = np.zeros((1, self.Ly, self.Lx, 3))
        self.Lyr, self.Lxr = self.Ly, self.Lx
        self.Ly0, self.Lx0 = self.Ly, self.Lx
        self.radii = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.cellpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.outpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.ismanual = np.zeros(0, "bool")
        self.instance_classes = np.zeros(0, dtype=np.int32)

        # -- set menus to default -- #
        self.view = 0
        self.ViewDropDown.setCurrentIndex(0)
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        self.delete_restore()

        self.clear_all()

        self.filename = []
        self.view_model.reset_series()
        self._sync_series_state()
        self.loaded = False
        self.recompute_masks = False

        self.deleting_multiple = False
        self.removing_cells_list = []
        self.removing_region = False
        self.remove_roi_obj = None
        self.autoSaturationButton.setEnabled(False)

    def delete_restore(self):
        """delete restored imgs but don't reset settings"""
        if hasattr(self, "stack_filtered"):
            del self.stack_filtered
        if hasattr(self, "cellpix_orig"):
            self.cellpix = self.cellpix_orig.copy()
            self.outpix = self.outpix_orig.copy()
            del self.outpix_orig, self.outpix_resize
            del self.cellpix_orig, self.cellpix_resize

    def clear_restore(self):
        """delete restored imgs and reset settings"""
        print("GUI_INFO: clearing restored image")
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        if self.ViewDropDown.currentIndex() == self.ViewDropDown.count() - 1:
            self.ViewDropDown.setCurrentIndex(0)
        self.delete_restore()
        self.restore = None
        self.ratio = 1.0
        self.set_normalize_params(self.get_normalize_params())

    def clear_all(self):
        self.prev_selected = 0
        self.selected = 0
        if self.restore and "upsample" in self.restore:
            self.layerz = 0 * np.ones((self.Lyr, self.Lxr, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.cellpix_resize = self.cellpix.copy()
            self.outpix_resize = self.outpix.copy()
            self.cellpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
            self.outpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
        else:
            self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)

        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]
        self.instance_classes = np.zeros(0, dtype=np.int32)
        self.ncells.reset()
        self.toggle_removals()
        self.update_scale()
        self.update_layer()
        self.refresh_instance_table()

    def select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity]
            )
            if (
                hasattr(self, "InstanceTable")
                and idx - 1 < self.InstanceTable.rowCount()
            ):
                self.InstanceTable.selectRow(idx - 1)
            self.update_layer()

    def select_cell_multi(self, idx):
        if idx > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity]
            )
            self.update_layer()

    def unselect_cell(self):
        if self.selected > 0:
            idx = self.selected
            if idx < (self.ncells.get() + 1):
                z = self.currentZ
                self.layerz[self.cellpix[z] == idx] = np.append(
                    self.cellcolors[idx], self.opacity
                )
                if self.outlinesOn:
                    self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                        np.uint8
                    )
                    # [0,0,0,self.opacity])
                self.update_layer()
        self.selected = 0

    def unselect_cell_multi(self, idx):
        z = self.currentZ
        self.layerz[self.cellpix[z] == idx] = np.append(
            self.cellcolors[idx], self.opacity
        )
        if self.outlinesOn:
            self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                np.uint8
            )
            # [0,0,0,self.opacity])
        self.update_layer()

    def remove_cell(self, idx):
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        # because the function remove_single_cell updates the state of the cellpix and outpix arrays
        # by reindexing cells to avoid gaps in the indices, we need to remove the cells in reverse order
        # so that the indices are correct
        idx.sort(reverse=True)
        for i in idx:
            self.remove_single_cell(i)
        self.ncells -= len(idx)  # _save_sets uses ncells
        self.update_layer()

        if self.ncells == 0:
            self.ClearButton.setEnabled(False)
        if self.NZ == 1:
            io._save_sets_with_check(self)

    def remove_single_cell(self, idx):
        # remove from manual array
        self.selected = 0
        removed_class_id = 0
        self._ensure_instance_classes()
        if idx - 1 < len(self.instance_classes):
            removed_class_id = int(self.instance_classes[idx - 1])
        if self.NZ > 1:
            zextent = ((self.cellpix == idx).sum(axis=(1, 2)) > 0).nonzero()[0]
        else:
            zextent = [0]
        for z in zextent:
            cp = self.cellpix[z] == idx
            op = self.outpix[z] == idx
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0
            if z == self.currentZ:
                # remove from mask layer
                self.layerz[cp] = np.array([0, 0, 0, 0])

        # reduce other pixels by -1
        self.cellpix[self.cellpix > idx] -= 1
        self.outpix[self.outpix > idx] -= 1

        if self.NZ == 1:
            self.removed_cell = [
                self.ismanual[idx - 1],
                self.cellcolors[idx],
                np.nonzero(cp),
                np.nonzero(op),
                removed_class_id,
            ]
            self.redo.setEnabled(True)
            ar, ac = self.removed_cell[2]
            d = datetime.datetime.now()
            self.track_changes.append(
                [d.strftime("%m/%d/%Y, %H:%M:%S"), "removed mask", [ar, ac]]
            )
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx - 1)
        self.cellcolors = np.delete(self.cellcolors, [idx], axis=0)
        if idx - 1 < len(self.instance_classes):
            self.instance_classes = np.delete(self.instance_classes, idx - 1)
        del self.zdraw[idx - 1]
        print("GUI_INFO: removed cell %d" % (idx - 1))

    def remove_region_cells(self):
        if self.removing_cells_list:
            for idx in self.removing_cells_list:
                self.unselect_cell_multi(idx)
            self.removing_cells_list.clear()
        self.disable_buttons_removeROIs()
        self.removing_region = True

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
        self.deleting_multiple = True

    def done_remove_multiple_cells(self):
        self.deleting_multiple = False
        self.removing_region = False

        if self.removing_cells_list:
            self.removing_cells_list = list(set(self.removing_cells_list))
            display_remove_list = [i - 1 for i in self.removing_cells_list]
            print(f"GUI_INFO: removing cells: {display_remove_list}")
            self.remove_cell(self.removing_cells_list)
            self.removing_cells_list.clear()
            self.unselect_cell()
        self.enable_buttons()

        if self.remove_roi_obj is not None:
            self.remove_roi(self.remove_roi_obj)

    def merge_cells(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z] == self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z] == self.selected)
                touching = np.logical_and(
                    (ar0[:, np.newaxis] - ar1) < 3, (ac0[:, np.newaxis] - ac1) < 3
                ).sum()
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                vr0, vc0 = np.nonzero(self.outpix[z] == self.prev_selected)
                vr1, vc1 = np.nonzero(self.outpix[z] == self.selected)
                self.outpix[z, vr0, vc0] = 0
                self.outpix[z, vr1, vc1] = 0
                if touching > 0:
                    mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                    mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                    contours = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    pvc, pvr = contours[-2][0].squeeze().T
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2

                else:
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print("GUI_INFO: merged two cells")
            self.update_layer()
            io._save_sets_with_check(self)
            self.undo.setEnabled(False)
            self.redo.setEnabled(False)

    def undo_remove_cell(self):
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :], axis=0)
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            class_id = self.removed_cell[4] if len(self.removed_cell) > 4 else 0
            self.instance_classes = np.append(self.instance_classes, class_id)
            self.ncells += 1
            self.zdraw.append([])
            print(">>> added back removed cell")
            self.update_layer()
            io._save_sets_with_check(self)
            self.removed_cell = []
            self.redo.setEnabled(False)

    def remove_stroke(self, delete_points=True, stroke_ind=-1):
        stroke = np.array(self.strokes[stroke_ind])
        cZ = self.currentZ
        inZ = stroke[0, 0] == cZ
        if inZ:
            outpix = self.outpix[cZ, stroke[:, 1], stroke[:, 2]] > 0
            self.layerz[stroke[~outpix, 1], stroke[~outpix, 2]] = np.array([0, 0, 0, 0])
            cellpix = self.cellpix[cZ, stroke[:, 1], stroke[:, 2]]
            ccol = self.cellcolors.copy()
            if self.selected > 0:
                ccol[self.selected] = np.array([255, 255, 255])
            col2mask = ccol[cellpix]
            if self.masksOn:
                col2mask = np.concatenate(
                    (col2mask, self.opacity * (cellpix[:, np.newaxis] > 0)), axis=-1
                )
            else:
                col2mask = np.concatenate(
                    (col2mask, 0 * (cellpix[:, np.newaxis] > 0)), axis=-1
                )
            self.layerz[stroke[:, 1], stroke[:, 2], :] = col2mask
            if self.outlinesOn:
                self.layerz[stroke[outpix, 1], stroke[outpix, 2]] = np.array(
                    self.outcolor
                )
            if delete_points:
                del self.current_point_set[stroke_ind]
            self.update_layer()

        del self.strokes[stroke_ind]

    def plot_clicked(self, event):
        if (
            event.button() == QtCore.Qt.LeftButton
            and not event.modifiers()
            & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)
            and not self.removing_region
        ):
            if event.double():
                try:
                    self.p0.setYRange(0, self.Ly + self.pr)
                except:
                    self.p0.setYRange(0, self.Ly)
                self.p0.setXRange(0, self.Lx)

    def cancel_remove_multiple(self):
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self):
        # unselect all previously selected cells:
        for idx in self.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.removing_cells_list.clear()

    def add_roi(self, roi):
        self.p0.addItem(roi)
        self.remove_roi_obj = roi

    def remove_roi(self, roi):
        self.clear_multi_selected_cells()
        assert roi == self.remove_roi_obj
        self.remove_roi_obj = None
        self.p0.removeItem(roi)
        self.removing_region = False

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
        if x1 > self.Lx:
            x1 = self.Lx
        if y1 > self.Ly:
            y1 = self.Ly

        # find cells in that region
        cell_idxs = np.unique(self.cellpix[self.currentZ, y0:y1, x0:x1])
        cell_idxs = np.trim_zeros(cell_idxs)
        # deselect cells not in region by deselecting all and then selecting the ones in the region
        self.clear_multi_selected_cells()

        for idx in cell_idxs:
            self.select_cell_multi(idx)
            self.removing_cells_list.append(idx)

        self.update_layer()

    def mouse_moved(self, pos):
        items = self.win.scene().items(pos)

    def update_plot(self):
        self.view = self.ViewDropDown.currentIndex()
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape

        if self.view == 0 or self.view == self.ViewDropDown.count() - 1:
            image = (
                self.stack[self.currentZ]
                if self.view == 0
                else self.stack_filtered[self.currentZ]
            )
            self.img.setImage(as_gray_image(image), autoLevels=False, lut=None)
            self.img.setLevels(self.saturation[0][self.currentZ])
        else:
            image = np.zeros((self.Ly, self.Lx), np.uint8)
            if len(self.flows) >= self.view - 1 and len(self.flows[self.view - 1]) > 0:
                image = self.flows[self.view - 1][self.currentZ]
            if self.view > 1:
                self.img.setImage(as_gray_image(image), autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(as_gray_image(image), autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])

        self.sliders[0].setValue(
            [
                self.saturation[0][self.currentZ][0],
                self.saturation[0][self.currentZ][1],
            ]
        )
        self.win.show()
        self.show()

    def update_layer(self):
        if self.masksOn or self.outlinesOn:
            self.layer.setImage(self.layerz, autoLevels=False)
        self.win.show()
        self.show()

    def add_set(self):
        if len(self.current_point_set) > 0:
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set[0]) > 8:
                color = self.colormap[self.ncells.get(), :3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors = np.append(
                        self.cellcolors, color[np.newaxis, :], axis=0
                    )
                    self.ismanual = np.append(self.ismanual, True)
                    self.instance_classes = np.append(
                        self.instance_classes, self.default_class_id()
                    )
                    self.ncells += 1
                    self.draw_layer()
                    if self.NZ == 1:
                        # only save after each cell if single image
                        io._save_sets_with_check(self)
            else:
                print("GUI_ERROR: cell too small, not drawn")
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_layer()

    def add_mask(self, points=None, color=(100, 200, 50), dense=True):
        # points is list of strokes
        points_all = np.concatenate(points, axis=0)

        # loop over z values
        median = []
        zdraw = np.unique(points_all[:, 0])
        z = 0
        ars, acs, vrs, vcs = (
            np.zeros(0, "int"),
            np.zeros(0, "int"),
            np.zeros(0, "int"),
            np.zeros(0, "int"),
        )
        for stroke in points:
            stroke = np.concatenate(stroke, axis=0).reshape(-1, 4)
            vr = stroke[:, 1]
            vc = stroke[:, 2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2), axis=-1)[
                :, np.newaxis, :
            ]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0][:, 0].T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 10:
                print("GUI_ERROR: cell < 10 pixels without overlaps, not drawn")
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
                mask[ar - vr.min() + 2, ac - vc.min() + 2] = 1
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                pvc, pvr = contours[-2][0][:, 0].T
                vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            ars = np.concatenate((ars, ar), axis=0)
            acs = np.concatenate((acs, ac), axis=0)
            vrs = np.concatenate((vrs, vr), axis=0)
            vcs = np.concatenate((vcs, vc), axis=0)

        self.draw_mask(z, ars, acs, vrs, vcs, color)
        median.append(np.array([np.median(ars), np.median(acs)]))

        self.zdraw.append(zdraw)
        d = datetime.datetime.now()
        self.track_changes.append(
            [d.strftime("%m/%d/%Y, %H:%M:%S"), "added mask", [ar, ac]]
        )
        return median

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        """draw single mask using outlines and area"""
        if idx is None:
            idx = self.ncells + 1
        self.cellpix[z, vr, vc] = idx
        self.cellpix[z, ar, ac] = idx
        self.outpix[z, vr, vc] = idx
        if self.restore and "upsample" in self.restore:
            if self.resize:
                self.cellpix_resize[z, vr, vc] = idx
                self.cellpix_resize[z, ar, ac] = idx
                self.outpix_resize[z, vr, vc] = idx
                self.cellpix_orig[
                    z, (vr / self.ratio).astype(int), (vc / self.ratio).astype(int)
                ] = idx
                self.cellpix_orig[
                    z, (ar / self.ratio).astype(int), (ac / self.ratio).astype(int)
                ] = idx
                self.outpix_orig[
                    z, (vr / self.ratio).astype(int), (vc / self.ratio).astype(int)
                ] = idx
            else:
                self.cellpix_orig[z, vr, vc] = idx
                self.cellpix_orig[z, ar, ac] = idx
                self.outpix_orig[z, vr, vc] = idx

                # get upsampled mask
                vrr = (vr.copy() * self.ratio).astype(int)
                vcr = (vc.copy() * self.ratio).astype(int)
                mask = np.zeros((np.ptp(vrr) + 4, np.ptp(vcr) + 4), np.uint8)
                pts = np.stack((vcr - vcr.min() + 2, vrr - vrr.min() + 2), axis=-1)[
                    :, np.newaxis, :
                ]
                mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
                arr, acr = np.nonzero(mask)
                arr, acr = arr + vrr.min() - 2, acr + vcr.min() - 2
                # get dense outline
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                pvc, pvr = contours[-2][0].squeeze().T
                vrr, vcr = pvr + vrr.min() - 2, pvc + vcr.min() - 2
                # concatenate all points
                arr, acr = np.hstack((np.vstack((vrr, vcr)), np.vstack((arr, acr))))
                self.cellpix_resize[z, vrr, vcr] = idx
                self.cellpix_resize[z, arr, acr] = idx
                self.outpix_resize[z, vrr, vcr] = idx

        if z == self.currentZ:
            self.layerz[ar, ac, :3] = color
            if self.masksOn:
                self.layerz[ar, ac, -1] = self.opacity
            if self.outlinesOn:
                self.layerz[vr, vc] = np.array(self.outcolor)

    def compute_scale(self):
        # get diameter from gui
        diameter = self.get_segmentation_parameters()["diameter"]
        if not diameter:
            diameter = 30

        self.pr = int(diameter)
        self.radii_padding = int(self.pr * 1.25)
        self.radii = np.zeros((self.Ly + self.radii_padding, self.Lx, 4), np.uint8)
        yy, xx = disk(
            [self.Ly + self.radii_padding / 2 - 1, self.pr / 2 + 1],
            self.pr / 2,
            self.Ly + self.radii_padding,
            self.Lx,
        )
        # rgb(150,50,150)
        self.radii[yy, xx, 0] = 150
        self.radii[yy, xx, 1] = 50
        self.radii[yy, xx, 2] = 150
        self.radii[yy, xx, 3] = 255
        self.p0.setYRange(0, self.Ly + self.radii_padding)
        self.p0.setXRange(0, self.Lx)

    def update_scale(self):
        self.compute_scale()
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])
        self.win.show()
        self.show()

    def draw_layer(self):
        if self.resize:
            self.Ly, self.Lx = self.Lyr, self.Lxr
        else:
            self.Ly, self.Lx = self.Ly0, self.Lx0

        if self.masksOn or self.outlinesOn:
            if self.restore and "upsample" in self.restore:
                if self.resize:
                    self.cellpix = self.cellpix_resize.copy()
                    self.outpix = self.outpix_resize.copy()
                else:
                    self.cellpix = self.cellpix_orig.copy()
                    self.outpix = self.outpix_orig.copy()

        self.layerz = np.zeros((self.Ly, self.Lx, 4), np.uint8)
        cellpix = self.cellpix[self.currentZ]
        visible_pixels = self.visible_cell_pixels(cellpix)
        if self.masksOn:
            self.layerz[..., :3] = self.cellcolors[cellpix, :]
            self.layerz[..., 3] = self.opacity * visible_pixels.astype(np.uint8)
            if self.selected > 0 and visible_pixels[cellpix == self.selected].any():
                self.layerz[self.cellpix[self.currentZ] == self.selected] = np.array(
                    [255, 255, 255, self.opacity]
                )
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:, 1], stroke[:, 2]] = np.array(
                        [255, 0, 255, 100]
                    )
        else:
            self.layerz[..., 3] = 0

        if self.outlinesOn:
            self.layerz[(self.outpix[self.currentZ] > 0) & visible_pixels] = np.array(
                self.outcolor
            ).astype(np.uint8)

    def set_normalize_params(self, normalize_params):
        from cellpose.models import normalize_default

        if self.restore != "filter":
            keys = list(normalize_params.keys()).copy()
            for key in keys:
                if key != "percentile":
                    normalize_params[key] = normalize_default[key]
        normalize_params = {**normalize_default, **normalize_params}
        self.set_preprocessing_parameters(
            {
                "sharpen_radius": max(0, normalize_params["sharpen_radius"]),
                "smooth_radius": max(0, normalize_params["smooth_radius"]),
                "tile_norm_blocksize": max(0, normalize_params["tile_norm_blocksize"]),
                "tile_norm_smooth3D": max(0, normalize_params["tile_norm_smooth3D"]),
                "norm3D": bool(normalize_params["norm3D"]),
                "invert": bool(normalize_params["invert"]),
            }
        )
        params = self.get_preprocessing_parameters()
        self.set_preprocessing_parameters(params)

    def get_normalize_params(self):
        segmentation_params = self.get_segmentation_parameters()
        preprocessing_params = self.get_preprocessing_parameters()
        self.set_preprocessing_parameters(preprocessing_params)
        return {
            **normalize_default,
            "percentile": segmentation_params["percentile"],
            **preprocessing_params,
        }

    def compute_saturation(self, apply_preprocessing=False):
        norm = self.get_normalize_params()
        print(norm)
        sharpen, smooth = norm["sharpen_radius"], norm["smooth_radius"]
        percentile = norm["percentile"]
        tile_norm = norm["tile_norm_blocksize"]
        invert = norm["invert"]
        norm3D = norm["norm3D"]
        smooth3D = norm["tile_norm_smooth3D"]
        tile_norm = norm["tile_norm_blocksize"]

        should_apply_preprocessing = (
            apply_preprocessing and (sharpen > 0 or smooth > 0 or tile_norm > 0)
        )

        if should_apply_preprocessing:
            img_norm = self.stack.copy()
        else:
            restored_view_index = self.ViewDropDown.count() - 1
            if (
                self.ViewDropDown.currentIndex() == restored_view_index
                and hasattr(self, "stack_filtered")
            ):
                img_norm = self.stack_filtered
            else:
                img_norm = self.stack

        if should_apply_preprocessing:
            self.restore = "filter"
            print(
                "GUI_INFO: computing filtered image because sharpen > 0 or tile_norm > 0"
            )
            print(
                "GUI_WARNING: will use memory to create filtered image -- make sure to have RAM for this"
            )
            img_norm = self.stack.copy()
            if sharpen > 0 or smooth > 0:
                img_norm = smooth_sharpen_img(
                    self.stack, sharpen_radius=sharpen, smooth_radius=smooth
                )

            if tile_norm > 0:
                img_norm = normalize99_tile(
                    img_norm,
                    blocksize=tile_norm,
                    lower=percentile[0],
                    upper=percentile[1],
                    smooth3D=smooth3D,
                    norm3D=norm3D,
                )
            # convert to 0->255
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= img_norm_max - img_norm_min
            img_norm *= 255
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(
                True
            )
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)
        img_gray = as_gray_image(img_norm)
        self.saturation = [[]]
        if np.ptp(img_gray) > 1e-3:
            if norm3D:
                x01 = np.percentile(img_gray, percentile[0])
                x99 = np.percentile(img_gray, percentile[1])
                if invert:
                    x01, x99 = 255.0 - x99, 255.0 - x01
                for n in range(self.NZ):
                    self.saturation[0].append([x01, x99])
            else:
                for z in range(self.NZ):
                    if self.NZ > 1:
                        plane = img_gray[z]
                    else:
                        plane = img_gray
                    x01 = np.percentile(plane, percentile[0])
                    x99 = np.percentile(plane, percentile[1])
                    if invert:
                        x01, x99 = 255.0 - x99, 255.0 - x01
                    self.saturation[0].append([x01, x99])
        else:
            for n in range(self.NZ):
                self.saturation[0].append([0, 255.0])

        self.update_plot()

    def get_model_path(self, custom=False):
        if custom:
            self.current_model = self.ModelChooseC.currentText()
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath("custom", self.current_model)
            )
        else:
            self.current_model = "cpsam"
            self.current_model_path = models.model_path(self.current_model)

    def initialize_model(self, model_name=None, custom=False):
        if model_name is None or custom:
            self.get_model_path(custom=custom)
            if not os.path.exists(self.current_model_path):
                raise ValueError(
                    "Model file not found: need to specify model (use dropdown)"
                )

        if custom or model_name is None or not isinstance(model_name, str):
            self.model = models.CellposeModel(
                gpu=True, pretrained_model=self.current_model_path
            )
        else:
            self.current_model = model_name
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model)
            )

            self.model = models.CellposeModel(
                gpu=True, pretrained_model=self.current_model
            )

    def add_model(self):
        io._add_model(self)
        return

    def remove_model(self):
        io._remove_model(self)
        return

    def new_model(self):
        if self.NZ != 1:
            print("ERROR: cannot train model on 3D data")
            return

        current_train_data_folder = self.training_params.get("train_data_folder", "")
        if not current_train_data_folder:
            current_train_data_folder = (
                os.path.dirname(self.filename) if self.filename else ""
            )
            self.training_params["train_data_folder"] = current_train_data_folder

        (
            self.train_data,
            self.train_labels,
            self.train_files,
            restore,
            normalize_params,
        ) = ([], [], [], None, copy.deepcopy(normalize_default))
        if current_train_data_folder:
            try:
                (
                    self.train_data,
                    self.train_labels,
                    self.train_files,
                    restore,
                    normalize_params,
                ) = self._get_train_dataset(current_train_data_folder, nested=True)
            except ValueError as e:
                self.logger.info(str(e))
                self.train_files = []
        else:
            restore = None

        # train model
        TW = TrainWindow(self, models.MODEL_NAMES)
        train = TW.exec()
        if train:
            train_data_folder = self.training_params.get("train_data_folder", "")
            if not train_data_folder:
                QMessageBox.warning(self, "Train", "No training folder specified.")
                return
            try:
                (
                    self.train_data,
                    self.train_labels,
                    self.train_files,
                    restore,
                    normalize_params,
                ) = self._get_train_dataset(train_data_folder, nested=True)
            except ValueError as e:
                self.logger.info(str(e))
                QMessageBox.warning(self, "Train", str(e))
                return

            if len(self.train_files) == 0:
                QMessageBox.warning(
                    self,
                    "Train",
                    "No valid training images with _seg.npy found in folder.",
                )
                return
            self.logger.info(
                f"training with {[os.path.split(f)[1] for f in self.train_files]}"
            )
            self.train_model(restore=restore, normalize_params=normalize_params)
        else:
            print("GUI_INFO: training cancelled")

    def train_model(self, restore=None, normalize_params=None):
        from cellpose.models import normalize_default

        if normalize_params is None:
            normalize_params = copy.deepcopy(normalize_default)
        model_type = models.MODEL_NAMES[self.training_params["model_index"]]
        self.logger.info(f"training new model starting at model {model_type}")
        self.current_model = model_type

        self.model = models.CellposeModel(gpu=True, model_type=model_type)
        save_path = os.fspath(models.MODEL_DIR.joinpath("custom"))
        os.makedirs(save_path, exist_ok=True)

        print("GUI_INFO: name of new model: " + self.training_params["model_name"])
        self.new_model_path, train_losses = train.train_seg(
            self.model.net,
            train_data=self.train_data,
            train_labels=self.train_labels,
            normalize=normalize_params,
            min_train_masks=0,
            save_path=save_path,
            nimg_per_epoch=max(2, len(self.train_data)),
            learning_rate=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
            n_epochs=self.training_params["n_epochs"],
            model_name=self.training_params["model_name"],
            save_to_models_dir=False,
        )[:2]
        # save train losses
        np.save(str(self.new_model_path) + "_train_losses.npy", train_losses)
        # run model on next image
        io._add_model(self, self.new_model_path)
        diam_labels = self.model.net.diam_labels.item()  # .copy()
        self.new_model_ind = len(self.model_strings)
        self.autorun = True
        self.clear_all()
        self.restore = restore
        self.set_normalize_params(normalize_params)
        self.get_next_image(load_seg=False)

        self.compute_segmentation(custom=True)
        self.logger.info(
            f"!!! computed masks for {os.path.split(self.filename)[1]} from new model !!!"
        )

    def get_training_image_files(self, folder, nested=True):
        if not folder:
            return []
        return get_image_files(folder, "_masks", look_one_level_down=nested)

    def _get_train_dataset(self, folder, nested=True):
        image_names = self.get_training_image_files(folder, nested=nested)
        return io._get_train_set(image_names)

    def compute_cprob(self):
        if getattr(self, "recompute_masks", False):
            segmentation_params = self.get_segmentation_parameters()
            min_size = (
                int(self.min_size.text())
                if not isinstance(self.min_size, int)
                else self.min_size
            )

            self.logger.info(
                "computing masks with cell prob=%0.3f, flow error threshold=%0.3f"
                % (
                    segmentation_params["cellprob_threshold"],
                    segmentation_params["flow_threshold"],
                )
            )

            try:
                dP = self.flows[2].squeeze()
                cellprob = self.flows[3].squeeze()
            except IndexError:
                self.logger.error("Flows don't exist, try running model again.")
                return

            maski = dynamics.resize_and_compute_masks(
                dP=dP,
                cellprob=cellprob,
                niter=segmentation_params["niter"],
                do_3D=self.load_3D,
                min_size=min_size,
                # max_size_fraction=min_size_fraction, # Leave as default
                cellprob_threshold=segmentation_params["cellprob_threshold"],
                flow_threshold=segmentation_params["flow_threshold"],
            )

            self.masksOn = True
            if not self.OCheckBox.isChecked():
                self.MCheckBox.setChecked(True)
            if maski.ndim < 3:
                maski = maski[np.newaxis, ...]
            self.logger.info("%d cells found" % (len(np.unique(maski)[1:])))
            io._masks_to_gui(self, maski, outlines=None)
            self.show()

    def compute_segmentation(self, custom=False, model_name=None, load_model=True):
        self.progress.setValue(0)
        try:
            tic = time.time()
            self.clear_all()
            self.flows = [[], [], []]
            if load_model:
                self.initialize_model(model_name=model_name, custom=custom)
            self.progress.setValue(10)
            do_3D = self.load_3D
            stitch_threshold = (
                float(self.stitch_threshold.text())
                if not isinstance(self.stitch_threshold, float)
                else self.stitch_threshold
            )
            anisotropy = (
                float(self.anisotropy.text())
                if not isinstance(self.anisotropy, float)
                else self.anisotropy
            )
            flow3D_smooth = (
                float(self.flow3D_smooth.text())
                if not isinstance(self.flow3D_smooth, float)
                else self.flow3D_smooth
            )
            min_size = (
                int(self.min_size.text())
                if not isinstance(self.min_size, int)
                else self.min_size
            )

            do_3D = False if stitch_threshold > 0.0 else do_3D

            if self.restore == "filter":
                data = self.stack_filtered.copy().squeeze()
            else:
                data = self.stack.copy().squeeze()

            segmentation_params = self.get_segmentation_parameters()

            normalize_params = self.get_normalize_params()
            print(normalize_params)
            try:
                masks, flows = self.model.eval(
                    data,
                    diameter=segmentation_params["diameter"],
                    cellprob_threshold=segmentation_params["cellprob_threshold"],
                    flow_threshold=segmentation_params["flow_threshold"],
                    do_3D=do_3D,
                    niter=segmentation_params["niter"],
                    normalize=normalize_params,
                    stitch_threshold=stitch_threshold,
                    anisotropy=anisotropy,
                    flow3D_smooth=flow3D_smooth,
                    min_size=min_size,
                    channel_axis=-1,
                    progress=self.progress,
                    z_axis=0 if self.NZ > 1 else None,
                )[:2]
            except Exception as e:
                print("NET ERROR: %s" % e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)

            # convert flows to uint8 and resize to original image size
            flows_new = []
            flows_new.append(flows[0].copy())  # RGB flow
            flows_new.append(
                (np.clip(normalize99(flows[2].copy()), 0, 1) * 255).astype("uint8")
            )  # cellprob
            flows_new.append(flows[1].copy())  # XY flows
            flows_new.append(flows[2].copy())  # original cellprob

            if self.load_3D:
                if stitch_threshold == 0.0:
                    flows_new.append((flows[1][0] / 10 * 127 + 127).astype("uint8"))
                else:
                    flows_new.append(np.zeros(flows[1][0].shape, dtype="uint8"))

            if not self.load_3D:
                if self.restore and "upsample" in self.restore:
                    self.Ly, self.Lx = self.Lyr, self.Lxr

                if flows_new[0].shape[-3:-1] != (self.Ly, self.Lx):
                    self.flows = []
                    for j in range(len(flows_new)):
                        self.flows.append(
                            resize_image(
                                flows_new[j],
                                Ly=self.Ly,
                                Lx=self.Lx,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )
                else:
                    self.flows = flows_new
            else:
                self.flows = []
                Lz, Ly, Lx = self.NZ, self.Ly, self.Lx
                Lz0, Ly0, Lx0 = flows_new[0].shape[:3]
                print("GUI_INFO: resizing flows to original image size")
                for j in range(len(flows_new)):
                    flow0 = flows_new[j]
                    if Ly0 != Ly:
                        flow0 = resize_image(
                            flow0,
                            Ly=Ly,
                            Lx=Lx,
                            no_channels=flow0.ndim == 3,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    if Lz0 != Lz:
                        flow0 = np.swapaxes(
                            resize_image(
                                np.swapaxes(flow0, 0, 1),
                                Ly=Lz,
                                Lx=Lx,
                                no_channels=flow0.ndim == 3,
                                interpolation=cv2.INTER_NEAREST,
                            ),
                            0,
                            1,
                        )
                    self.flows.append(flow0)

            # add first axis
            if self.NZ == 1:
                masks = masks[np.newaxis, ...]
                self.flows = [
                    self.flows[n][np.newaxis, ...] for n in range(len(self.flows))
                ]

            self.logger.info(
                "%d cells found with model in %0.3f sec"
                % (len(np.unique(masks)[1:]), time.time() - tic)
            )
            self.progress.setValue(80)
            z = 0

            io._masks_to_gui(self, masks, outlines=None)
            self.masksOn = True
            self.MCheckBox.setChecked(True)
            self.progress.setValue(100)
            if not do_3D and not stitch_threshold > 0:
                self.recompute_masks = True
            else:
                self.recompute_masks = False
        except Exception as e:
            print("ERROR: %s" % e)
