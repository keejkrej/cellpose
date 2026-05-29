"""
Presenter for the Cellpose GUI.

MainPresenter connects MainView (widgets + rendering) to MainModel (state).
"""

from __future__ import annotations

import copy
import datetime
import os
import time
from typing import Any

import cv2
import numpy as np
from .qt import QtCore  # noqa: F401 — configure QT_API before qtpy
from qtpy.QtWidgets import QMessageBox

from .. import dynamics, models, train
from ..io import get_image_files
from ..models import normalize_default
from ..plot import disk
from ..transforms import normalize99, resize_image
from . import io, series
from .model import InstanceClasses, MainModel, SegmentationParameters, SeriesState
from .presenter_gui import PresenterGuiMixin
from .presenter_masks import add_mask_from_points, paint_mask_at
from .view_protocol import LabelRow, SeriesNavViewState


class MainPresenter(PresenterGuiMixin):
    def __init__(self, view, model: MainModel):
        self.view = view
        self.model = model
        self.cp_model = None
        self.current_model = "cpsam"
        self.current_model_path = None

    @property
    def session(self):
        return self.model.session

    def training_params(self) -> dict[str, Any]:
        return self.model.training_params

    @property
    def instance_classes(self) -> np.ndarray:
        return self.model.instance_classes

    @property
    def instance_visible(self) -> np.ndarray:
        return self.model.instance_visible

    # ---- training ----

    def reset_training_parameters(self) -> dict[str, Any]:
        return self.model.reset_training_parameters()

    def set_training_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        return self.model.set_training_parameters(values)

    # ---- series ----

    def _series_nav_state(
        self, dataset=None, record_index=None
    ) -> SeriesNavViewState:
        if dataset is None or record_index is None:
            return SeriesNavViewState(enabled=False, axis_ranges={}, axis_values={})
        axis_ranges = {}
        axis_values = {}
        for axis_name in series.SERIES_AXES:
            axis_values_list = dataset["axes"][axis_name]
            axis_ranges[axis_name] = (0, max(0, len(axis_values_list) - 1))
            axis_values[axis_name] = dataset["axis_index"][axis_name][
                dataset["records"][record_index][axis_name]
            ]
        return SeriesNavViewState(
            enabled=True, axis_ranges=axis_ranges, axis_values=axis_values
        )

    def reset_series(self) -> SeriesState:
        state = self.model.reset_series()
        self.view.apply_series_labels(state)
        self.view.set_series_navigation(self._series_nav_state())
        return state

    def set_series(
        self, dataset: dict[str, Any] | None = None, record_index: int | None = None
    ) -> SeriesState:
        state = self.model.set_series(dataset=dataset, record_index=record_index)
        self.view.apply_series_labels(state)
        self.view.set_series_navigation(self._series_nav_state(dataset, record_index))
        return state

    def output_filename(self, fallback_filename: str) -> str:
        return self.model.output_filename(fallback_filename)

    # ---- parameters ----

    def segmentation_parameters(self) -> SegmentationParameters:
        widgets = self.view.read_segmentation_widgets()
        params = SegmentationParameters.from_values(
            diameter=float(widgets["diameter"]),
            flow_threshold=float(widgets["flow_threshold"]),
            cellprob_threshold=float(widgets["cellprob_threshold"]),
            percentile_low=float(widgets["percentile_low"]),
            percentile_high=float(widgets["percentile_high"]),
            niter=int(widgets["niter"]),
        )
        self.view.apply_segmentation_widgets(params)
        return params

    def segmentation_parameters_dict(self) -> dict[str, Any]:
        return self.segmentation_parameters().to_dict()

    def get_normalize_params(self) -> dict[str, Any]:
        segmentation_params = self.segmentation_parameters_dict()
        stored = dict(self.model.preprocessing_params or {})
        params = {**normalize_default, **stored}
        params["percentile"] = segmentation_params["percentile"]
        return params

    def set_normalize_params(self, normalize_params: dict[str, Any]) -> None:
        merged = {**normalize_default, **normalize_params}
        if self.session.restore != "filter":
            for key in merged:
                if key != "percentile":
                    merged[key] = normalize_default[key]
        self.model.preprocessing_params = merged

    # ---- instances ----

    def ncells(self) -> int:
        return self.model.ncells

    def ensure_instance_classes(
        self, ncells: int | None = None, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.ncells()
        return self.model.ensure_instance_classes(ncells, current_values=current_values)

    def set_instance_classes(
        self, values: np.ndarray | list[int] | None = None, ncells: int | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.ncells()
        result = self.model.set_instance_classes(ncells, values)
        self.refresh_labels_table()
        return result

    def set_instance_class(self, row: int, class_id: int) -> np.ndarray:
        result = self.model.set_instance_class(row, class_id)
        self.refresh_labels_table()
        self.refresh_mask_layer()
        return result

    def ensure_instance_visible(
        self, ncells: int | None = None, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.ncells()
        return self.model.ensure_instance_visible(ncells, current_values=current_values)

    def set_instance_visible(
        self, values: np.ndarray | list[bool] | None = None, ncells: int | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.ncells()
        result = self.model.set_instance_visible(ncells, values)
        self.refresh_labels_table()
        return result

    def set_instance_visible_row(self, row: int, visible: bool) -> np.ndarray:
        result = self.model.set_instance_visible_row(row, visible)
        self.refresh_mask_layer()
        return result

    def set_all_instance_visible(self, visible: bool, ncells: int | None = None) -> None:
        if ncells is None:
            ncells = self.ncells()
        self.model.set_all_instance_visible(visible, ncells)
        self.refresh_mask_layer()

    def remove_instance_metadata(self, row: int) -> tuple[int, bool]:
        return self.model.remove_instance_metadata(row)

    def append_instance_metadata(
        self, class_id: int = 0, visible: bool = True
    ) -> None:
        self.model.append_instance_metadata(class_id, visible)

    def reset_instance_metadata(self) -> None:
        self.model.reset_instance_metadata()

    def labels_class_filter(self) -> int | None:
        return InstanceClasses.parse_filter(self.view.read_labels_class_filter())

    def visible_cell_pixels(self, cellpix: np.ndarray) -> np.ndarray:
        self.ensure_instance_classes()
        self.ensure_instance_visible()
        return self.model.visible_cell_pixels(cellpix, self.labels_class_filter())

    def build_labels_table_rows(self) -> list[LabelRow]:
        ncells = self.ncells()
        self.ensure_instance_classes(ncells)
        self.ensure_instance_visible(ncells)
        filter_class_id = self.labels_class_filter()
        selected = set(self.model.selection.selected_cells)
        if not selected and self.model.selection.selected > 0:
            selected = {self.model.selection.selected}
        rows = []
        for row in range(ncells):
            class_id = int(self.instance_classes[row])
            rows.append(
                LabelRow(
                    roi=row + 1,
                    class_id=class_id,
                    visible=bool(self.instance_visible[row]),
                    hidden_by_filter=(
                        filter_class_id is not None and class_id != filter_class_id
                    ),
                    selected=(row + 1) in selected,
                )
            )
        return rows

    def labels_table_header_state(self) -> str:
        ncells = self.ncells()
        if ncells == 0:
            return "unchecked"
        self.ensure_instance_visible(ncells)
        visible_count = int(self.instance_visible[:ncells].sum())
        if visible_count == 0:
            return "unchecked"
        if visible_count == ncells:
            return "checked"
        return "partial"

    def refresh_labels_table(self) -> None:
        self.view.refresh_labels_table(
            self.build_labels_table_rows(),
            self.labels_table_header_state(),
        )

    def refresh_mask_layer(self) -> None:
        session = self.session
        if session.resize:
            session.ly, session.lx = session.lyr, session.lxr
        else:
            session.ly, session.lx = session.ly0, session.lx0
        if session.restore and "upsample" in session.restore:
            if session.resize:
                session.cellpix = session.cellpix_resize.copy()
                session.outpix = session.outpix_resize.copy()
            else:
                session.cellpix = session.cellpix_orig.copy()
                session.outpix = session.outpix_orig.copy()
        layerz = self.model.build_layer_rgba(filter_class_id=self.labels_class_filter())
        stroke_z = np.array([s[0][0] for s in self.model.drawing.strokes])
        in_z = np.nonzero(stroke_z == session.current_z)[0]
        for i in in_z:
            stroke = np.array(self.model.drawing.strokes[i])
            layerz[stroke[:, 1], stroke[:, 2]] = np.array([255, 0, 255, 100])
        self.view.render_mask_overlay(layerz)
        self.refresh_selection_boxes()
        self.view.show_window()

    def selection_bounds(self) -> list[tuple[int, int, int, int]]:
        indices = (
            self.model.selection.selected_cells
            if self.model.selection.selected_cells
            else (
                [self.model.selection.selected]
                if self.model.selection.selected > 0
                else []
            )
        )
        bounds = []
        for idx in indices:
            rect = self.model.cell_bounds(idx)
            if rect is not None:
                bounds.append(rect)
        return bounds

    def on_labels_filter_changed(self) -> None:
        self.refresh_labels_table()
        self.refresh_mask_layer()

    # ---- navigation ----

    def get_files(self) -> tuple[list[str], int]:
        dataset = self.model.series_state.dataset
        record_index = self.model.series_state.record_index
        if dataset is not None and record_index is not None:
            return (
                [record["path"] for record in dataset["records"]],
                record_index,
            )
        folder = os.path.dirname(str(self.model.filename))
        mask_filter = "_masks"
        images = get_image_files(folder, mask_filter)
        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.model.filename)[-1]
        idx = int(np.nonzero(np.array(fnames) == f0)[0][0])
        return images, idx

    def run_selected_model(self) -> None:
        model_name, custom = self.view.read_selected_model()
        if custom:
            self.compute_segmentation(custom=True)
        else:
            self.compute_segmentation(model_name=model_name)

    def get_prev_image(self) -> None:
        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        if self.model.series_state.dataset is not None:
            try:
                self.load_series_item(
                    self.model.series_state.dataset,
                    idx,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                self.view.show_message("Load folder with pattern", str(e))
        else:
            self.load_image(filename=images[idx])

    def get_next_image(self, load_seg: bool = True) -> None:
        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        if self.model.series_state.dataset is not None:
            try:
                self.load_series_item(
                    self.model.series_state.dataset,
                    idx,
                    load_seg=load_seg,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                self.view.show_message("Load folder with pattern", str(e))
        else:
            self.load_image(filename=images[idx], load_seg=load_seg)

    def navigate_series_from_sliders(
        self, axis_name: str | None = None, delta: int = 0
    ) -> None:
        if delta != 0 and axis_name is not None:
            axes = self.view.read_series_slider_axes()
            current = axes.get(axis_name, 0)
            dataset = self.model.series_state.dataset
            if dataset is None:
                return
            nav = self._series_nav_state(dataset, self.model.series_state.record_index)
            low, high = nav.axis_ranges.get(axis_name, (0, 0))
            value = max(low, min(high, current + delta))
            self.view.set_updating_series_navigation(True)
            try:
                self.view.set_series_slider_value(axis_name, value)
            finally:
                self.view.set_updating_series_navigation(False)

        if (
            self.view.is_updating_series_navigation()
            or self.model.series_state.dataset is None
            or self.model.series_state.record_index is None
            or axis_name is None
        ):
            return

        slider_axes = self.view.read_series_slider_axes()
        dataset = self.model.series_state.dataset
        try:
            record_index = series.resolve_series_record_index(
                dataset,
                position=dataset["axes"]["position"][slider_axes["position"]],
                time=dataset["axes"]["time"][slider_axes["time"]],
                channel=dataset["axes"]["channel"][slider_axes["channel"]],
                z=dataset["axes"]["z"][slider_axes["z"]],
            )
        except Exception as e:
            self.view.set_series_navigation(
                self._series_nav_state(dataset, self.model.series_state.record_index)
            )
            self.view.show_message("Load folder with pattern", str(e))
            return

        if record_index == self.model.series_state.record_index:
            return

        try:
            self.load_series_item(dataset, record_index)
        except Exception as e:
            self.view.set_series_navigation(
                self._series_nav_state(dataset, self.model.series_state.record_index)
            )
            print(f"ERROR: {e}")
            self.view.show_message("Load folder with pattern", str(e))

    # ---- session lifecycle ----

    def reset_session(self) -> None:
        self.model.reset_session()
        self.view.set_ncells_count(0)
        if hasattr(self.view, "BrushButton"):
            self.view.BrushButton.setChecked(False)
        self.view.sliders[0].setValue([0, 255])
        self.view.sliders[0].setEnabled(False)
        self.view.set_view_mode(0, restored_enabled=False)
        self.model.discard_filtered_stack()
        self.clear_all()
        self.reset_series()
        self.view.autoSaturationButton.setEnabled(False)
        self.update_canvas_cursor()

    def clear_all(self) -> None:
        self.model.clear_masks()
        self.view.set_ncells_count(0)
        self.view.set_mask_action_enabled(False)
        self.refresh_scale_from_model()
        self.refresh_mask_layer()
        self.refresh_labels_table()
        self.view.render_selection_boxes([])
        self.view.render_rect_select_preview(None)

    def on_initialize_images(self, image: np.ndarray) -> None:
        self.model.load_image_stack(image)
        self.clear_all()
        self.view.sliders[0].setValue([0, 255])
        self.refresh_scale_from_model()

    def on_image_loaded(self, filename: str, display_filename: str | None = None) -> None:
        self.model.filename = filename
        self.model.series_state.display_filename = display_filename or filename
        self.model.series_state.output_filename = None
        self.model.session.loaded = True
        self.view.set_loaded_chrome(True, ncells=self.ncells())
        self.refresh_plot()
        self.update_canvas_cursor()

    def refresh_scale_from_model(self) -> None:
        params = self.segmentation_parameters_dict()
        diameter = params["diameter"] or 30
        pr = int(diameter)
        radii_padding = int(pr * 1.25)
        session = self.session
        radii = np.zeros((session.ly + radii_padding, session.lx, 4), np.uint8)
        yy, xx = disk(
            [session.ly + radii_padding / 2 - 1, pr / 2 + 1],
            pr / 2,
            session.ly + radii_padding,
            session.lx,
        )
        radii[yy, xx, 0] = 150
        radii[yy, xx, 1] = 50
        radii[yy, xx, 2] = 150
        radii[yy, xx, 3] = 255
        self.view.p0.setYRange(0, session.ly + radii_padding)
        self.view.p0.setXRange(0, session.lx)
        self.view.radii = radii
        self.view.render_diameter_scale(radii)
        self.view.show_window()

    def compute_saturation(self) -> None:
        params = self.segmentation_parameters_dict()
        percentile = params["percentile"]
        restored_view_index = 3
        if (
            self.view.read_view_mode_index() == restored_view_index
            and self.session.stack_filtered is not None
        ):
            img_norm = self.session.stack_filtered
        else:
            img_norm = self.session.stack
        from .widgets import as_gray_image

        img_gray = as_gray_image(img_norm)
        self.session.saturation = [[]]
        if np.ptp(img_gray) > 1e-3:
            for z in range(self.session.nz):
                plane = img_gray[z]
                x01 = np.percentile(plane, percentile[0])
                x99 = np.percentile(plane, percentile[1])
                self.session.saturation[0].append([x01, x99])
        else:
            for _ in range(self.session.nz):
                self.session.saturation[0].append([0, 255.0])
        self.refresh_plot()

    def remove_cell(self, idx) -> None:
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        idx = sorted({int(i) for i in idx}, reverse=True)
        selection = self.model.selection
        selection.selected = 0
        selection.selected_cells = []
        if len(idx) == 1:
            i = idx[0]
            cp = self.session.cellpix[0] == i
            op = self.session.outpix[0] == i
            row = i - 1
            removed_class_id = (
                int(self.instance_classes[row]) if row < len(self.instance_classes) else 0
            )
            removed_visible = (
                bool(self.instance_visible[row])
                if row < len(self.instance_visible)
                else True
            )
            selection.removed_cell = [
                self.session.ismanual[row] if row < len(self.session.ismanual) else False,
                self.session.cellcolors[i],
                np.nonzero(cp),
                np.nonzero(op),
                removed_class_id,
                removed_visible,
            ]
            self.view.set_redo_enabled(True)
        self.model.remove_cells(idx)
        for i in idx:
            print("GUI_INFO: removed cell %d" % (i - 1))
        self.view.set_ncells_count(self.ncells())
        self.refresh_labels_table()
        self.refresh_mask_layer()
        if self.ncells() == 0:
            self.view.set_mask_action_enabled(False)
        self.save_sets()

    def _remove_single_cell(self, idx: int) -> None:
        self.remove_cell(idx)

    def add_set(self) -> None:
        drawing = self.model.drawing
        if len(drawing.current_point_set) > 0:
            while len(drawing.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(drawing.current_point_set[0]) > 8:
                color = self.view.colormap[self.ncells(), :3]
                median = add_mask_from_points(
                    self.model, drawing.current_point_set, color
                )
                if median is not None:
                    self.model.selection.removed_cell = []
                    self.session.cellcolors = np.append(
                        self.session.cellcolors, color[np.newaxis, :], axis=0
                    )
                    self.session.ismanual = np.append(self.session.ismanual, True)
                    self.append_instance_metadata(self.view.read_default_class_id(), True)
                    self.view.set_ncells_count(self.ncells())
                    self.view.set_mask_action_enabled(True)
                    self.refresh_labels_table()
                    self.refresh_mask_layer()
                    self.save_sets()
            else:
                print("GUI_ERROR: cell too small, not drawn")
            drawing.current_stroke = []
            drawing.strokes = []
            drawing.current_point_set = []
            self.refresh_mask_layer()

    def remove_stroke(self, delete_points=True, stroke_ind=-1) -> None:
        stroke = np.array(self.model.drawing.strokes[stroke_ind])
        c_z = self.session.current_z
        in_z = stroke[0, 0] == c_z
        if in_z:
            outpix = self.session.outpix[c_z, stroke[:, 1], stroke[:, 2]] > 0
            self.session.layerz[stroke[~outpix, 1], stroke[~outpix, 2]] = np.array(
                [0, 0, 0, 0]
            )
            cellpix = self.session.cellpix[c_z, stroke[:, 1], stroke[:, 2]]
            ccol = self.session.cellcolors.copy()
            if self.model.selection.selected > 0:
                ccol[self.model.selection.selected] = np.array([255, 255, 255])
            col2mask = ccol[cellpix]
            col2mask = np.concatenate(
                (col2mask, self.session.opacity * (cellpix[:, np.newaxis] > 0)),
                axis=-1,
            )
            self.session.layerz[stroke[:, 1], stroke[:, 2], :] = col2mask
            self.session.layerz[stroke[outpix, 1], stroke[outpix, 2]] = np.array(
                self.session.outcolor
            )
            if delete_points:
                del self.model.drawing.current_point_set[stroke_ind]
            self.refresh_mask_layer()
        del self.model.drawing.strokes[stroke_ind]

    def merge_cells(self, idx: int) -> None:
        selection = self.model.selection
        selection.prev_selected = selection.selected
        selection.selected = idx
        if selection.selected != selection.prev_selected:
            for z in range(self.session.nz):
                ar0, ac0 = np.nonzero(
                    self.session.cellpix[z] == selection.prev_selected
                )
                ar1, ac1 = np.nonzero(self.session.cellpix[z] == selection.selected)
                touching = np.logical_and(
                    (ar0[:, np.newaxis] - ar1) < 3, (ac0[:, np.newaxis] - ac1) < 3
                ).sum()
                vr0, vc0 = np.nonzero(self.session.outpix[z] == selection.prev_selected)
                vr1, vc1 = np.nonzero(self.session.outpix[z] == selection.selected)
                self.session.outpix[z, vr0, vc0] = 0
                self.session.outpix[z, vr1, vc1] = 0
                if touching > 0:
                    ar = np.hstack((ar0, ar1))
                    ac = np.hstack((ac0, ac1))
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
                    ar = np.hstack((ar0, ar1))
                    ac = np.hstack((ac0, ac1))
                color = self.session.cellcolors[selection.prev_selected]
                paint_mask_at(
                    self.model,
                    z,
                    ar,
                    ac,
                    vr,
                    vc,
                    color,
                    idx=selection.prev_selected,
                )
            self.remove_cell(selection.selected)
            print("GUI_INFO: merged two cells")
            self.view.set_undo_enabled(False)
            self.view.set_redo_enabled(False)

    def undo_remove_cell(self) -> None:
        selection = self.model.selection
        if len(selection.removed_cell) > 0:
            z = 0
            ar, ac = selection.removed_cell[2]
            vr, vc = selection.removed_cell[3]
            color = selection.removed_cell[1]
            paint_mask_at(self.model, z, ar, ac, vr, vc, color)
            self.session.cellcolors = np.append(
                self.session.cellcolors, color[np.newaxis, :], axis=0
            )
            self.session.ismanual = np.append(
                self.session.ismanual, selection.removed_cell[0]
            )
            class_id = selection.removed_cell[4] if len(selection.removed_cell) > 4 else 0
            visible = selection.removed_cell[5] if len(selection.removed_cell) > 5 else True
            self.append_instance_metadata(class_id, visible)
            self.view.set_ncells_count(self.ncells())
            self.session.zdraw.append([])
            self.view.set_mask_action_enabled(True)
            self.refresh_mask_layer()
            self.save_sets()
            selection.removed_cell = []
            self.view.set_redo_enabled(False)

    def _apply_masks_to_view(self, masks: np.ndarray) -> None:
        ncells = self.model.apply_masks(
            masks, outlines=None, colormap=self.view.colormap
        )
        self.view.set_ncells_count(ncells)
        print(f"GUI_INFO: {ncells} masks found")
        if ncells > 0:
            self.refresh_mask_layer()
            self.view.set_mask_action_enabled(True)
        restored = (
            self.session.restore == "filter" or self.session.stack_filtered is not None
        )
        self.view.set_view_mode(3 if restored else 0, restored_enabled=restored)
        if restored:
            print("set denoised/filtered view")

    def compute_cprob(self) -> None:
        if not self.session.recompute_masks:
            return
        segmentation_params = self.segmentation_parameters_dict()
        opts = self.view.read_inference_options()
        self.view.logger.info(
            "computing masks with cell prob=%0.3f, flow error threshold=%0.3f"
            % (
                segmentation_params["cellprob_threshold"],
                segmentation_params["flow_threshold"],
            )
        )
        try:
            d_p = self.session.flows[2].squeeze()
            cellprob = self.session.flows[3].squeeze()
        except IndexError:
            self.view.logger.error("Flows don't exist, try running model again.")
            return
        maski = dynamics.resize_and_compute_masks(
            dP=d_p,
            cellprob=cellprob,
            niter=segmentation_params["niter"],
            do_3D=False,
            min_size=opts["min_size"],
            cellprob_threshold=segmentation_params["cellprob_threshold"],
            flow_threshold=segmentation_params["flow_threshold"],
        )
        if maski.ndim < 3:
            maski = maski[np.newaxis, ...]
        self.view.logger.info("%d cells found" % (len(np.unique(maski)[1:])))
        self._apply_masks_to_view(maski)
        self.view.show_window()
        self.save_sets()

    def initialize_model(self, model_name=None, custom=False) -> None:
        if model_name is None or custom:
            model_name, _ = self.view.read_selected_model()
            self.current_model = model_name
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath("custom", self.current_model)
            )
            if not os.path.exists(self.current_model_path):
                raise ValueError(
                    "Model file not found: need to specify model (use dropdown)"
                )
            self.cp_model = models.CellposeModel(
                gpu=True, pretrained_model=self.current_model_path
            )
        else:
            self.current_model = model_name
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model)
            )
            self.cp_model = models.CellposeModel(
                gpu=True, pretrained_model=self.current_model
            )

    def compute_segmentation(
        self, custom: bool = False, model_name: str | None = None, load_model: bool = True
    ) -> None:
        self.view.set_progress(0)
        try:
            tic = time.time()
            self.clear_all()
            self.session.flows = [[], [], []]
            if load_model:
                self.initialize_model(model_name=model_name, custom=custom)
            self.view.set_progress(10)
            opts = self.view.read_inference_options()
            min_size = opts["min_size"]
            if self.session.restore == "filter":
                data = self.session.stack_filtered.copy().squeeze()
            else:
                data = self.session.stack.copy().squeeze()
            segmentation_params = self.segmentation_parameters_dict()
            normalize_params = self.get_normalize_params()
            try:
                masks, flows = self.cp_model.eval(
                    data,
                    diameter=segmentation_params["diameter"],
                    cellprob_threshold=segmentation_params["cellprob_threshold"],
                    flow_threshold=segmentation_params["flow_threshold"],
                    do_3D=False,
                    niter=segmentation_params["niter"],
                    normalize=normalize_params,
                    stitch_threshold=0.0,
                    anisotropy=1.0,
                    flow3D_smooth=0.0,
                    min_size=min_size,
                    channel_axis=-1,
                    progress=self.view.progress_widget(),
                )[:2]
            except Exception as e:
                print("NET ERROR: %s" % e)
                self.view.set_progress(0)
                return
            self.view.set_progress(75)
            flows_new = [
                flows[0].copy(),
                (np.clip(normalize99(flows[2].copy()), 0, 1) * 255).astype("uint8"),
                flows[1].copy(),
                flows[2].copy(),
            ]
            if self.session.restore and "upsample" in self.session.restore:
                self.session.ly, self.session.lx = self.session.lyr, self.session.lxr
            if flows_new[0].shape[-3:-1] != (self.session.ly, self.session.lx):
                self.session.flows = []
                for flow_item in flows_new:
                    self.session.flows.append(
                        resize_image(
                            flow_item,
                            Ly=self.session.ly,
                            Lx=self.session.lx,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )
            else:
                self.session.flows = flows_new
            masks = masks[np.newaxis, ...]
            self.session.flows = [
                self.session.flows[n][np.newaxis, ...]
                for n in range(len(self.session.flows))
            ]
            self.view.logger.info(
                "%d cells found with model in %0.3f sec"
                % (len(np.unique(masks)[1:]), time.time() - tic)
            )
            self.view.set_progress(80)
            self._apply_masks_to_view(masks)
            self.view.set_progress(100)
            self.session.recompute_masks = True
            self.save_sets()
        except Exception as e:
            print("ERROR: %s" % e)

    def apply_masks_from_io(
        self,
        masks: np.ndarray,
        outlines: np.ndarray | None = None,
        colors: np.ndarray | None = None,
    ) -> None:
        ncells = self.model.apply_masks(
            masks,
            outlines=outlines,
            colors=colors,
            colormap=self.view.colormap,
        )
        self.view.set_ncells_count(ncells)
        print(f"GUI_INFO: {ncells} masks found")
        if ncells > 0:
            self.refresh_mask_layer()
            self.view.set_mask_action_enabled(True)
        restored = self.session.stack_filtered is not None
        self.view.set_view_mode(3 if restored else 0, restored_enabled=restored)

    def on_load_seg_session(
        self,
        session_data,
        ismanual: np.ndarray | None = None,
        flows=None,
        recompute_masks: bool = False,
        instance_classes=None,
    ) -> None:
        if instance_classes is not None:
            self.set_instance_classes(instance_classes)
        if ismanual is not None and len(ismanual) == self.ncells():
            self.session.ismanual = ismanual
        if flows:
            self.session.flows = flows
            self.session.recompute_masks = recompute_masks
        else:
            self.session.recompute_masks = False
        self.model.session.loaded = True
        self.view.set_loaded_chrome(True, ncells=self.ncells())
        self.update_canvas_cursor()
        self.refresh_mask_layer()
        self.refresh_plot()

    def save_sets(self) -> None:
        from cellpose.gui.session_format import write_session

        filename = self.output_filename(str(self.model.filename))
        base = os.path.splitext(filename)[0]
        path = base + "_seg.npy"
        segmentation_params = self.segmentation_parameters_dict()
        normalize_params = self.get_normalize_params()
        self.model.segmentation_params = segmentation_params
        self.model.preprocessing_params = normalize_params
        model_name, _ = self.view.read_selected_model()
        dat = self.model.to_seg_dict(
            current_model_path=model_name,
            normalize_params=normalize_params,
            segmentation_params=segmentation_params,
        )
        try:
            written = write_session(path, dat)
            print("GUI_INFO: %d ROIs saved to %s" % (self.ncells(), written))
        except Exception as e:
            print(f"ERROR: {e}")

    def train_new_model(self) -> None:
        from .dialogs import TrainWindow

        current_train_data_folder = self.training_params().get("train_data_folder", "")
        if not current_train_data_folder:
            current_train_data_folder = (
                os.path.dirname(str(self.model.filename))
                if self.model.filename
                else ""
            )
            self.set_training_parameters(
                {"train_data_folder": current_train_data_folder}
            )
        train_data, train_labels, train_files, restore, normalize_params = (
            [],
            [],
            [],
            None,
            copy.deepcopy(normalize_default),
        )
        if current_train_data_folder:
            try:
                (
                    train_data,
                    train_labels,
                    train_files,
                    restore,
                    normalize_params,
                ) = io._get_train_set(
                    get_image_files(current_train_data_folder, "_masks", look_one_level_down=True)
                )
            except ValueError as e:
                self.view.logger.info(str(e))
                train_files = []
        tw = TrainWindow(self.view, self, models.MODEL_NAMES)
        if tw.exec():
            train_data_folder = self.training_params().get("train_data_folder", "")
            if not train_data_folder:
                self.view.show_message("Train", "No training folder specified.")
                return
            try:
                (
                    train_data,
                    train_labels,
                    train_files,
                    restore,
                    normalize_params,
                ) = io._get_train_set(
                    get_image_files(train_data_folder, "_masks", look_one_level_down=True)
                )
            except ValueError as e:
                self.view.logger.info(str(e))
                self.view.show_message("Train", str(e))
                return
            if len(train_files) == 0:
                self.view.show_message(
                    "Train",
                    "No valid training images with _seg.npy found in folder.",
                )
                return
            self.view.logger.info(
                f"training with {[os.path.split(f)[1] for f in train_files]}"
            )
            self._train_model(
                train_data,
                train_labels,
                restore=restore,
                normalize_params=normalize_params,
            )
        else:
            print("GUI_INFO: training cancelled")

    def _train_model(self, train_data, train_labels, restore=None, normalize_params=None):
        if normalize_params is None:
            normalize_params = copy.deepcopy(normalize_default)
        model_type = models.MODEL_NAMES[self.training_params()["model_index"]]
        self.view.logger.info(f"training new model starting at model {model_type}")
        self.current_model = model_type
        self.cp_model = models.CellposeModel(gpu=True, model_type=model_type)
        save_path = os.fspath(models.MODEL_DIR.joinpath("custom"))
        os.makedirs(save_path, exist_ok=True)
        print("GUI_INFO: name of new model: " + self.training_params()["model_name"])
        new_model_path, train_losses = train.train_seg(
            self.cp_model.net,
            train_data=train_data,
            train_labels=train_labels,
            normalize=normalize_params,
            min_train_masks=0,
            save_path=save_path,
            nimg_per_epoch=max(2, len(train_data)),
            learning_rate=self.training_params()["learning_rate"],
            weight_decay=self.training_params()["weight_decay"],
            n_epochs=self.training_params()["n_epochs"],
            model_name=self.training_params()["model_name"],
            save_to_models_dir=False,
        )[:2]
        np.save(str(new_model_path) + "_train_losses.npy", train_losses)
        self.add_model_file(new_model_path)
        self.session.restore = restore
        self.set_normalize_params(normalize_params)
        self.clear_all()
        self.get_next_image(load_seg=False)
        self.compute_segmentation(custom=True)
        self.view.logger.info(
            f"!!! computed masks for {os.path.split(self.model.filename)[1]} from new model !!!"
        )

