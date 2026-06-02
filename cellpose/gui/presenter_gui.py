"""
GUI orchestration mixin for the Cellpose MVP presenter stack.

PresenterGuiMixin wires view signals, handles IO/plot/selection/ROI flows, and
keeps MainView passive (widgets + rendering only).
"""

from __future__ import annotations

import gc
import os
import shutil
from typing import Any

import cv2
import numpy as np
from .qt import QtCore, QtGui
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

import pyqtgraph as pg

from ..io import imread_2D
from ..models import get_user_models, normalize_default
from . import io, menus, series
from .widgets import brush_cursor, select_cursor


class PresenterGuiMixin:
    """Mixin with GUI wiring and orchestration for MainPresenter."""

    rect_select_mode: bool
    _rect_select_start: tuple[int, int] | None
    _rect_select_dragging: bool
    _rect_select_drag_threshold: int

    def start(self, image=None) -> None:
        self.rect_select_mode = False
        self._rect_select_start = None
        self._rect_select_dragging = False
        self._rect_select_drag_threshold = 4

        self._wire_menus()
        self._wire_buttons()
        self._wire_series_nav()
        self._wire_labels_table()
        self._wire_seg_params()
        self._wire_canvas()

        self.init_model_list()
        self.view.set_canvas_controller(self)
        self.init_canvas_drawing()
        self.reset_session()

        if image is not None:
            self.model.filename = image
            self.load_image(filename=image)

        self.view.setAcceptDrops(True)
        self.view.show_window()

    # ---- wiring ----

    def _wire_menus(self) -> None:
        menus.mainmenu(self.view, self)
        menus.editmenu(self.view, self)
        menus.modelmenu(self.view, self)

    def _wire_buttons(self) -> None:
        view = self.view
        view.ViewDropDown.currentIndexChanged.connect(self.on_view_mode_changed)
        view.autoSaturationButton.clicked.connect(self.compute_saturation)
        view.BrushButton.toggled.connect(self.toggle_brush_mode)
        view.RectSelectButton.toggled.connect(self.toggle_rect_select_mode)
        view.DeleteSelectedButton.clicked.connect(self.remove_selected_cells)
        view.EditSelectedButton.clicked.connect(self.edit_selected_cells)
        view.ModelChooseC.activated.connect(lambda: self.model_choose(custom=True))
        view.ModelButtonC.clicked.connect(self.run_selected_model)
        view.ncells_counter.valueChanged.connect(lambda *_: self.refresh_labels_table())
        view.LabelsClassFilter.returnPressed.connect(self.on_labels_filter_changed)

    def _wire_series_nav(self) -> None:
        for axis_name, control in self.view.series_nav_controls.items():
            control["prev_btn"].clicked.connect(
                lambda _checked=False, axis_name=axis_name: self.navigate_series_from_sliders(
                    axis_name, -1
                )
            )
            control["slider"].sliderReleased.connect(
                lambda axis_name=axis_name: self._commit_series_slider(axis_name)
            )
            control["next_btn"].clicked.connect(
                lambda _checked=False, axis_name=axis_name: self.navigate_series_from_sliders(
                    axis_name, +1
                )
            )

    def _wire_labels_table(self) -> None:
        view = self.view
        view._visibility_header.checkboxClicked.connect(self.toggle_all_visibility)
        view.LabelsTable.itemChanged.connect(self.on_labels_table_item_changed)
        view.LabelsTable.itemSelectionChanged.connect(
            self.on_labels_table_selection_changed
        )

    def _wire_seg_params(self) -> None:
        root = self.view.seg_param_root
        root.param("diameter").sigValueChanged.connect(
            lambda *_: self.refresh_scale_from_model()
        )
        root.param("flow_threshold").sigValueChanged.connect(
            lambda *_: self.compute_cprob()
        )
        root.param("cellprob_threshold").sigValueChanged.connect(
            lambda *_: self.compute_cprob()
        )
        root.param("niter").sigValueChanged.connect(lambda *_: self.compute_cprob())
        root.param("norm_percentile_low").sigValueChanged.connect(
            lambda *_: self.validate_normalization_range()
        )
        root.param("norm_percentile_high").sigValueChanged.connect(
            lambda *_: self.validate_normalization_range()
        )

    def _wire_canvas(self) -> None:
        view = self.view
        view.saturation_changed.connect(lambda _name: self.on_saturation_changed())
        view.files_dropped.connect(self.handle_drop)
        view.win.scene().sigMouseClicked.connect(self._handle_plot_click)
        view.rect_select_press.connect(self._on_rect_select_press)
        view.rect_select_move.connect(self._on_rect_select_move)
        view.rect_select_release.connect(self._on_rect_select_release)

    def _handle_plot_click(self, event) -> None:
        self.plot_double_clicked(event)

    def _rect_select_allowed(self) -> bool:
        return (
            self.rect_select_mode
            and self.session.loaded
            and not self.model.selection.removing_region
            and not self.model.selection.deleting_multiple
        )

    def _on_rect_select_press(self, pos, modifiers) -> None:
        if not self._rect_select_allowed():
            return
        self.begin_rect_select(pos)

    def _on_rect_select_move(self, pos) -> None:
        if not self._rect_select_allowed() or self._rect_select_start is None:
            return
        self.update_rect_select(pos)

    def _on_rect_select_release(self, pos, modifiers) -> None:
        if not self._rect_select_allowed() or self._rect_select_start is None:
            return
        self.finish_rect_select(pos, modifiers)

    def _commit_series_slider(self, axis_name: str | None = None) -> None:
        if (
            self.view.is_updating_series_navigation()
            or self.model.series_state.dataset is None
            or self.model.series_state.record_index is None
            or axis_name is None
        ):
            return
        self.navigate_series_from_sliders(axis_name)

    # ---- IO ----

    def init_model_list(self) -> None:
        io._get_custom_model_dir()
        self.model.model_strings = get_user_models()
        self.view.set_model_list(self.model.model_strings)

    def load_image(self, filename=None, load_seg: bool = True) -> None:
        if filename is None:
            name = QFileDialog.getOpenFileName(self.view, "Load image")
            filename = name[0]
            if filename == "":
                return

        self.reset_series()
        manual_file = os.path.splitext(filename)[0] + "_seg.npy"
        if load_seg and os.path.isfile(manual_file):
            image = imread_2D(filename)
            self.load_seg(manual_file, image=image, image_file=filename)
            return

        try:
            print(f"GUI_INFO: loading image: {filename}")
            image = imread_2D(filename)
        except Exception as e:
            print("ERROR: images not compatible")
            print(f"ERROR: {e}")
            return

        self.reset_session()
        self.on_initialize_images(image)
        self.on_image_loaded(filename)

    def load_seg(
        self,
        filename,
        image=None,
        image_file=None,
    ) -> None:
        from cellpose.gui.session_format import read_session

        if not filename:
            return

        try:
            session_data = read_session(filename)
        except Exception as exc:
            self.model.session.loaded = False
            print(f"ERROR: not a valid _seg.npy session: {exc}")
            return

        if image is None:
            image_path = session_data.source_image
            if not os.path.isfile(image_path):
                self.model.session.loaded = False
                print(f"ERROR: cannot find image file: {image_path}")
                return
            try:
                print(f"GUI_INFO: loading image: {image_path}")
                image = imread_2D(image_path)
            except Exception as exc:
                self.model.session.loaded = False
                print(f"ERROR: cannot load image: {exc}")
                return
            self.model.filename = image_path
        else:
            self.model.filename = image_file or session_data.source_image

        self.reset_session()
        self.reset_series()
        self.model.series_state.output_filename = None
        self.model.series_state.display_filename = str(self.model.filename)
        self.model.session.restore = None
        self.model.session.ratio = 1.0

        self.on_initialize_images(image)

        masks = np.asarray(session_data.masks)
        if masks.min() == -1:
            masks = masks + 1
        colors = session_data.colors
        if colors is None and int(masks.max()) > 0:
            colors = self.view.colormap[: int(masks.max()), :3]

        self.apply_masks_from_io(masks, colors=colors)
        self.view.apply_segmentation_metadata_widgets(session_data.segmentation)

        ismanual = np.zeros(self.ncells(), bool)
        if (
            session_data.ismanual is not None
            and len(session_data.ismanual) == self.ncells()
        ):
            ismanual = session_data.ismanual

        flows = None
        recompute_masks = False
        if session_data.flows:
            flows = session_data.flows
            try:
                if flows[0].shape[-3] != masks.shape[-2]:
                    ly, lx = masks.shape[-2:]
                    resized = []
                    for flow in flows:
                        resized.append(
                            cv2.resize(
                                np.asarray(flow).squeeze(),
                                (lx, ly),
                                interpolation=cv2.INTER_NEAREST,
                            )[np.newaxis, ...]
                        )
                    flows = resized
            except Exception:
                pass
            recompute_masks = session_data.recompute_masks

        self.on_load_seg_session(
            session_data,
            ismanual=ismanual,
            flows=flows,
            recompute_masks=recompute_masks,
            instance_classes=session_data.instance_classes,
        )
        gc.collect()

    def load_image_series(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.view, "Load image folder")
        if folder == "":
            return

        templates = io._prompt_series_templates(
            self.view,
            folder,
            self.model.last_series_subfolder_template,
            self.model.last_series_filename_template,
        )
        if templates is None:
            return

        subfolder_template, filename_template = templates
        if filename_template == "":
            return

        self.model.last_series_subfolder_template = subfolder_template
        self.model.last_series_filename_template = filename_template

        try:
            dataset = series.build_series_dataset(
                folder,
                subfolder_template=subfolder_template,
                filename_template=filename_template,
            )
            self.load_series_item(dataset, 0, load_seg=True)
        except Exception as e:
            print(f"ERROR: {e}")
            QMessageBox.warning(self.view, "Load folder with pattern", str(e))

    def load_series_item(
        self,
        dataset,
        item_index,
        load_seg: bool = True,
    ) -> None:
        output_filename = series.get_output_filename(dataset, item_index)
        seg_filename = os.path.splitext(output_filename)[0] + "_seg.npy"
        if load_seg and os.path.isfile(seg_filename):
            self.load_seg(filename=seg_filename)
            if self.model.series_state.dataset is None:
                self.set_series(dataset=dataset, record_index=item_index)
            title = self.model.series_state.display_filename or self.model.filename
            self.view.set_window_title(str(title))
            return

        self.load_image(filename=output_filename, load_seg=False)
        self.set_series(dataset=dataset, record_index=item_index)
        title = self.model.series_state.display_filename or self.model.filename
        self.view.set_window_title(str(title))

    def add_model_file(self, filename=None, load_model: bool = True) -> None:
        if filename is None:
            name = QFileDialog.getOpenFileName(self.view, "Add model to GUI")
            filename = name[0]
        if filename == "":
            return

        fname = os.path.split(filename)[-1]
        target = io._get_custom_model_dir().joinpath(fname)
        try:
            shutil.copyfile(filename, os.fspath(target))
        except shutil.SameFileError:
            pass

        model_strings = list(self.model.model_strings)
        for ind, model_string in enumerate(model_strings):
            if model_string == fname:
                self.remove_model_file(ind=ind + 1, verbose=False)
                model_strings = list(self.model.model_strings)
                break
        model_strings.append(fname)
        self.model.model_strings = model_strings
        io._write_model_list(model_strings)
        self.view.set_model_list(model_strings, current=fname)
        if load_model:
            self.model_choose(custom=True)

    def remove_model_file(self, ind=None, verbose: bool = True) -> None:
        if ind is None:
            ind = self.view.ModelChooseC.currentIndex()
        if ind > 0:
            ind -= 1
            modelstr = self.model.model_strings[ind]
            model_strings = list(self.model.model_strings)
            del model_strings[ind]
            self.model.model_strings = model_strings
            io._write_model_list(model_strings)
            model_path = io._get_custom_model_dir().joinpath(modelstr)
            if model_path.exists():
                os.remove(os.fspath(model_path))
            if model_strings:
                self.view.set_model_list(model_strings, current=model_strings[-1])
            else:
                self.view.set_model_list([], current=None)
        elif verbose:
            print("ERROR: no model selected to delete")

    def handle_drop(self, files: list[str]) -> None:
        if not files:
            return
        if files[0].endswith("_seg.npy"):
            self.load_seg(filename=files[0])
        else:
            self.load_image(filename=files[0], load_seg=True)

    def add_model(self, filename=None) -> None:
        self.add_model_file(filename=filename)

    def remove_model(self) -> None:
        self.remove_model_file()

    def model_choose(self, custom: bool = False) -> None:
        if not custom:
            return
        model_name, is_custom = self.view.read_selected_model()
        if model_name:
            print(f"GUI_INFO: selected model {model_name}, loading now")
            self.initialize_model(model_name=model_name, custom=is_custom)

    # ---- plot ----

    def refresh_plot(self) -> None:
        view_index = self.view.read_view_mode_index()
        self.view.view = view_index
        session = self.session
        session.ly, session.lx, _ = session.stack[session.current_z].shape

        restored_view_index = self.view.ViewDropDown.count() - 1
        if view_index == 0 or view_index == restored_view_index:
            image = (
                session.stack[session.current_z]
                if view_index == 0
                else session.stack_filtered[session.current_z]
            )
            low, high = session.saturation[0][session.current_z]
            self.view.render_image_plane(image, [low, high], lut=None)
        else:
            image = np.zeros((session.ly, session.lx), np.uint8)
            if (
                len(session.flows) >= view_index - 1
                and len(session.flows[view_index - 1]) > 0
            ):
                image = session.flows[view_index - 1][session.current_z]
            if view_index > 1:
                self.view.render_image_plane(image, [0.0, 255.0], lut=self.view.bwr)
            else:
                self.view.render_image_plane(image, [0.0, 255.0], lut=None)

        low, high = session.saturation[0][session.current_z]
        self.view.sync_saturation_slider(low, high)
        self.view.show_window()

    def on_saturation_changed(self) -> None:
        if self.session.loaded:
            sval = self.view.sliders[0].value()
            self.session.saturation[0][self.session.current_z] = sval
            self.refresh_plot()

    def on_view_mode_changed(self) -> None:
        self.refresh_plot()

    def plot_double_clicked(self, event) -> None:
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and not event.modifiers()
            & (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.AltModifier)
            and not self.model.selection.removing_region
        ):
            if event.double():
                try:
                    params = self.segmentation_parameters_dict()
                    pr = int(params["diameter"] or 30)
                    self.view.p0.setYRange(0, self.session.ly + pr)
                except Exception:
                    self.view.p0.setYRange(0, self.session.ly)
                self.view.p0.setXRange(0, self.session.lx)

    def mouse_moved(self, pos) -> None:
        pass

    def handle_rect_select_event(self, obj, event) -> bool:
        """Optional direct event-filter path; view MVP uses rect_select_* signals."""
        return False

    def validate_normalization_range(self) -> None:
        try:
            self.segmentation_parameters_dict()
        except ValueError:
            print("GUI_ERROR: normalization percentile lower must be less than upper")

    # ---- selection ----

    def refresh_selection_boxes(self) -> None:
        self.view.render_selection_boxes(self.selection_bounds())

    def _selected_cell_indices(self) -> list[int]:
        cells = list(self.model.selection.selected_cells)
        if not cells and self.model.selection.selected > 0:
            cells = [self.model.selection.selected]
        return sorted({int(idx) for idx in cells if int(idx) > 0})

    def remove_selected_cells(self) -> None:
        cells = self._selected_cell_indices()
        if cells:
            self.remove_cell(cells)

    def _sync_labels_table_selection_multi(self, indices: list[int]) -> None:
        self.view._sync_labels_table_selection_multi(indices)

    def _apply_cell_selection(self, cells: list[int]) -> None:
        self.model.selection.selected_cells = cells
        self.model.selection.selected = cells[0] if cells else 0
        self.model.selection.prev_selected = self.model.selection.selected
        self._sync_labels_table_selection_multi(cells)
        self.refresh_selection_boxes()

    def select_cell(self, idx: int) -> None:
        self.model.selection.prev_selected = self.model.selection.selected
        self.model.selection.selected = idx
        self.model.selection.selected_cells = [idx] if idx > 0 else []
        if self.model.selection.selected > 0:
            self._sync_labels_table_selection_multi([idx])
            self.refresh_selection_boxes()
        else:
            self.view.render_selection_boxes([])

    def unselect_cell(self) -> None:
        self.model.selection.selected = 0
        self.model.selection.selected_cells = []
        self.view.render_selection_boxes([])
        self._sync_labels_table_selection_multi([])

    def on_labels_table_selection_changed(self) -> None:
        view = self.view
        if view._refreshing_labels_table or view._syncing_labels_table_selection:
            return
        selected_rows = view.LabelsTable.selectionModel().selectedRows()
        if not selected_rows:
            if self.model.selection.selected > 0 or self.model.selection.selected_cells:
                self.model.selection.selected = 0
                self.model.selection.selected_cells = []
                self.view.render_selection_boxes([])
            return
        cells = sorted({row.row() + 1 for row in selected_rows})
        if cells == self.model.selection.selected_cells:
            self.refresh_selection_boxes()
            return
        self.model.selection.prev_selected = self.model.selection.selected
        self.model.selection.selected_cells = cells
        self.model.selection.selected = cells[0] if cells else 0
        self.refresh_selection_boxes()

    def _normalize_rect(self, x0, y0, x1, y1):
        x0, x1 = sorted([int(x0), int(x1)])
        y0, y1 = sorted([int(y0), int(y1)])
        x0 = max(0, min(self.session.lx - 1, x0))
        x1 = max(0, min(self.session.lx, x1))
        y0 = max(0, min(self.session.ly - 1, y0))
        y1 = max(0, min(self.session.ly, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _set_rect_preview(self, x0, y0, x1, y1) -> None:
        bounds = self._normalize_rect(x0, y0, x1, y1)
        self.view.render_rect_select_preview(bounds)

    def _cells_fully_in_rect(self, x0, y0, x1, y1) -> list[int]:
        return self.model.cells_in_rect(
            x0,
            y0,
            x1,
            y1,
            filter_class_id=self.labels_class_filter(),
        )

    def begin_rect_select(self, pos) -> None:
        self._rect_select_start = (int(pos.y()), int(pos.x()))
        self._rect_select_dragging = False

    def update_rect_select(self, pos) -> None:
        if self._rect_select_start is None:
            return
        y0, x0 = self._rect_select_start
        y1, x1 = int(pos.y()), int(pos.x())
        if not self._rect_select_dragging:
            if max(abs(y1 - y0), abs(x1 - x0)) < self._rect_select_drag_threshold:
                return
            self._rect_select_dragging = True
        self._set_rect_preview(x0, y0, x1, y1)

    def finish_rect_select(
        self, pos, modifiers=QtCore.Qt.KeyboardModifier.NoModifier
    ) -> None:
        if self._rect_select_start is None:
            return
        y0, x0 = self._rect_select_start
        y1, x1 = int(pos.y()), int(pos.x())
        dragging = self._rect_select_dragging
        additive = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._rect_select_start = None
        self._rect_select_dragging = False
        self.view.render_rect_select_preview(None)
        self.view._rect_select_start = None
        self.view._rect_select_dragging = False
        if dragging:
            cells = self._cells_fully_in_rect(x0, y0, x1, y1)
            if additive:
                cells = sorted(set(self.model.selection.selected_cells) | set(cells))
            self._apply_cell_selection(cells)
        else:
            self._select_cell_at_click(y1, x1, additive=additive)

    def _select_cell_at_click(self, y, x, additive: bool = False) -> None:
        session = self.session
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

    def select_cell_multi(self, idx: int) -> None:
        if idx > 0:
            z = self.session.current_z
            self.session.layerz[self.session.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.session.opacity]
            )
            self.refresh_mask_layer()

    def unselect_cell_multi(self, idx: int) -> None:
        z = self.session.current_z
        self.session.layerz[self.session.cellpix[z] == idx] = np.append(
            self.session.cellcolors[idx], self.session.opacity
        )
        self.session.layerz[self.session.outpix[z] == idx] = np.array(
            self.session.outcolor
        ).astype(np.uint8)
        self.refresh_mask_layer()

    # ---- ROI bulk delete ----

    def remove_region_cells(self) -> None:
        if self.model.selection.removing_cells_list:
            for idx in self.model.selection.removing_cells_list:
                self.unselect_cell_multi(idx)
            self.model.selection.removing_cells_list.clear()
        self.view.disable_buttons_removeROIs()
        self.model.selection.removing_region = True

        self.clear_multi_selected_cells()

        roi_width = self.view.p0.viewRect().width() / 2
        x_loc = self.view.p0.viewRect().x() + (roi_width / 2)
        roi_height = self.view.p0.viewRect().height() / 2
        y_loc = self.view.p0.viewRect().y() + (roi_height / 2)

        pos = [x_loc, y_loc]
        roi = pg.RectROI(
            pos, [roi_width, roi_height], pen=pg.mkPen("y", width=2), removable=True
        )
        roi.sigRemoveRequested.connect(self.remove_roi)
        roi.sigRegionChangeFinished.connect(self.roi_changed)
        self.view.p0.addItem(roi)
        self.view.remove_roi_obj = roi
        self.roi_changed(roi)

    def delete_multiple_cells(self) -> None:
        self.unselect_cell()
        self.view.disable_buttons_removeROIs()
        self.model.selection.deleting_multiple = True

    def done_remove_multiple_cells(self) -> None:
        self.model.selection.deleting_multiple = False
        self.model.selection.removing_region = False

        if self.model.selection.removing_cells_list:
            self.model.selection.removing_cells_list = list(
                set(self.model.selection.removing_cells_list)
            )
            display_remove_list = [
                i - 1 for i in self.model.selection.removing_cells_list
            ]
            print(f"GUI_INFO: removing cells: {display_remove_list}")
            self.remove_cell(self.model.selection.removing_cells_list)
            self.model.selection.removing_cells_list.clear()
            self.unselect_cell()
        self.view.set_loaded_chrome(True, ncells=self.ncells())
        self.update_canvas_cursor()

        if self.view.remove_roi_obj is not None:
            self.remove_roi(self.view.remove_roi_obj)

    def cancel_remove_multiple(self) -> None:
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self) -> None:
        for idx in self.model.selection.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.model.selection.removing_cells_list.clear()

    def remove_roi(self, roi) -> None:
        self.clear_multi_selected_cells()
        assert roi == self.view.remove_roi_obj
        self.view.remove_roi_obj = None
        self.view.p0.removeItem(roi)
        self.model.selection.removing_region = False

    def roi_changed(self, roi) -> None:
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
        if x1 > self.session.lx:
            x1 = self.session.lx
        if y1 > self.session.ly:
            y1 = self.session.ly

        cell_idxs = np.unique(
            self.session.cellpix[self.session.current_z, y0:y1, x0:x1]
        )
        cell_idxs = np.trim_zeros(cell_idxs)
        self.clear_multi_selected_cells()

        for idx in cell_idxs:
            self.select_cell_multi(idx)
            self.model.selection.removing_cells_list.append(idx)

        self.refresh_mask_layer()

    # ---- labels table ----

    def sync_visibility_header_checkbox(self) -> None:
        view = self.view
        if not hasattr(view, "_visibility_header"):
            return
        ncells = self.ncells()
        if ncells == 0:
            view._visibility_header.set_check_state(QtCore.Qt.CheckState.Unchecked)
            return
        self.ensure_instance_visible(ncells)
        visible_count = int(self.instance_visible[:ncells].sum())
        if visible_count == 0:
            state = QtCore.Qt.CheckState.Unchecked
        elif visible_count == ncells:
            state = QtCore.Qt.CheckState.Checked
        else:
            state = QtCore.Qt.CheckState.PartiallyChecked
        view._visibility_header.set_check_state(state)

    def toggle_all_visibility(self, state) -> None:
        view = self.view
        if view._refreshing_labels_table:
            return
        ncells = self.ncells()
        if ncells == 0:
            return
        visible = state != QtCore.Qt.CheckState.Unchecked
        view._refreshing_labels_table = True
        view.LabelsTable.blockSignals(True)
        try:
            self.set_all_instance_visible(visible, ncells)
            for row in range(ncells):
                item = view.LabelsTable.item(row, 0)
                if item is not None:
                    item.setCheckState(
                        QtCore.Qt.CheckState.Checked
                        if visible
                        else QtCore.Qt.CheckState.Unchecked
                    )
        finally:
            view.LabelsTable.blockSignals(False)
            view._refreshing_labels_table = False
        self.sync_visibility_header_checkbox()

    def on_labels_table_item_changed(self, item) -> None:
        column = item.column()
        if column == 0:
            self._set_instance_visible_from_table(item)
        elif column == 2:
            self._set_instance_class_from_labels_table(item)

    def _set_instance_visible_from_table(self, item) -> None:
        view = self.view
        if view._refreshing_labels_table:
            return
        row = item.row()
        self.ensure_instance_visible()
        if row >= len(self.instance_visible):
            return
        visible = item.checkState() == QtCore.Qt.CheckState.Checked
        self.set_instance_visible_row(row, visible)
        self.sync_visibility_header_checkbox()

    def _set_instance_class_from_labels_table(self, item) -> None:
        view = self.view
        if view._refreshing_labels_table or item.column() != 2:
            return
        row = item.row()
        self.ensure_instance_classes()
        if row >= len(self.instance_classes):
            return
        old_class_id = int(self.instance_classes[row])
        try:
            class_id = int(item.text())
            if class_id < 0:
                raise ValueError
        except ValueError:
            view.LabelsTable.blockSignals(True)
            item.setText(str(old_class_id))
            view.LabelsTable.blockSignals(False)
            return
        self.set_instance_class(row, class_id)
        if self.session.loaded:
            self.save_sets()

    def edit_selected_cells(self) -> None:
        cells = self._selected_cell_indices()
        if not cells:
            QMessageBox.information(
                self.view, "Edit class", "Select one or more cells first."
            )
            return

        self.ensure_instance_classes()
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

        dialog = QDialog(self.view)
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
            QMessageBox.warning(
                self.view, "Edit class", "Enter a non-negative integer class ID."
            )
            return

        for idx in cells:
            row = idx - 1
            if row >= 0:
                self.model.set_instance_class(row, class_id)
        self.refresh_labels_table()
        self.refresh_mask_layer()
        if self.session.loaded:
            self.save_sets()

    def undo_action(self) -> None:
        if (
            len(self.model.drawing.strokes) > 0
            and self.model.drawing.strokes[-1][0][0] == self.session.current_z
        ):
            self.remove_stroke()
        elif self.ncells() > 0:
            self.remove_cell(self.ncells())

    # ---- canvas input (ImageDraw controller) ----

    def init_canvas_drawing(self) -> None:
        self.model.drawing.current_stroke = []
        self.model.drawing.in_stroke = False

    def canvas_rect_select_mode(self) -> bool:
        return self.rect_select_mode

    # ---- brush / rect mode ----

    def update_canvas_cursor(self) -> None:
        view = self.view
        if not hasattr(view, "p0"):
            return
        if not self.session.loaded:
            cursor = QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor)
        elif self.model.drawing.brush_mode:
            cursor = brush_cursor()
        elif self.rect_select_mode:
            cursor = select_cursor()
        else:
            cursor = QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor)
        view.p0.setCursor(cursor)
        view.win.setCursor(cursor)

    def toggle_brush_mode(self, enabled: bool) -> None:
        view = self.view
        if enabled and self.rect_select_mode:
            view.RectSelectButton.setChecked(False)
        self.model.drawing.brush_mode = enabled
        self.update_canvas_cursor()
        view._update_canvas_cursor()
        if enabled or not self.model.drawing.in_stroke:
            return

        if hasattr(view.layer, "scatter"):
            view.p0.removeItem(view.layer.scatter)
        drawing = self.model.drawing
        drawing.in_stroke = False
        drawing.current_stroke = []
        drawing.stroke_appended = True
        self.refresh_mask_layer()

    def toggle_rect_select_mode(self, enabled: bool) -> None:
        view = self.view
        if enabled and self.model.drawing.brush_mode:
            view.BrushButton.setChecked(False)
        self.rect_select_mode = enabled
        view.rect_select_mode = enabled
        if not enabled:
            self._rect_select_start = None
            self._rect_select_dragging = False
            view.render_rect_select_preview(None)
        self.update_canvas_cursor()
        view._update_canvas_cursor()
