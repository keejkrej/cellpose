"""
Presenter for the Cellpose GUI.

MainPresenter connects MainView (widgets + rendering) to MainModel (state).
"""

from __future__ import annotations

import copy
import datetime
import gc
import os
import shutil
import time
from typing import Any

import cv2
import numpy as np
import pyqtgraph as pg
from cellpose.app_core.train import get_train_set
from .qt import QtCore, QtGui  # noqa: F401 — configure QT_API before qtpy
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from .. import dynamics, models, train
from ..io import get_image_files, imread_2D
from ..models import MODEL_DIR, MODEL_LIST_PATH, get_user_models, normalize_default
from ..plot import disk
from ..transforms import normalize99, resize_image
from ..utils import get_mask_ellipse_diameters
from cellpose.app_core import series
from .ui import menus
from .ui.dialogs import TrainWindow, prompt_series_templates
from .model import InstanceClasses, MainModel, SegmentationParameters, SeriesState
from .sync import SyncRequest, SyncScope, create_sync_notifier
from .view import LabelRow, SeriesNavViewState
from .ui.widgets import brush_cursor, select_cursor


def add_mask_from_points(
    model: MainModel,
    points: list,
    color: np.ndarray,
) -> list | None:
    """Create a mask from drawn stroke points; mutates model.session."""
    session = model.session
    points_all = np.concatenate(points, axis=0)
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
        mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
        pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2), axis=-1)[:, np.newaxis, :]
        mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
        ar, ac = np.nonzero(mask)
        ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pvc, pvr = contours[-2][0][:, 0].T
        vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
        ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
        ioverlap = session.cellpix[z][ar, ac] > 0
        if (~ioverlap).sum() < 10:
            print("GUI_ERROR: cell < 10 pixels without overlaps, not drawn")
            return None
        if ioverlap.sum() > 0:
            ar, ac = ar[~ioverlap], ac[~ioverlap]
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            mask[ar - vr.min() + 2, ac - vc.min() + 2] = 1
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0][:, 0].T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
        ars = np.concatenate((ars, ar), axis=0)
        acs = np.concatenate((acs, ac), axis=0)
        vrs = np.concatenate((vrs, vr), axis=0)
        vcs = np.concatenate((vcs, vc), axis=0)

    idx = int(session.cellpix.max()) + 1
    _paint_mask(session, z, ars, acs, vrs, vcs, color, idx)
    session.zdraw.append(zdraw)
    d = datetime.datetime.now()
    session.track_changes.append(
        [d.strftime("%m/%d/%Y, %H:%M:%S"), "added mask", [ars, acs]]
    )
    return [np.array([np.median(ars), np.median(acs)])]


def _paint_mask(session, z, ar, ac, vr, vc, color, idx):
    session.cellpix[z, vr, vc] = idx
    session.cellpix[z, ar, ac] = idx
    session.outpix[z, vr, vc] = idx
    if session.restore and "upsample" in session.restore:
        if session.resize:
            session.cellpix_resize[z, vr, vc] = idx
            session.cellpix_resize[z, ar, ac] = idx
            session.outpix_resize[z, vr, vc] = idx
            session.cellpix_orig[
                z, (vr / session.ratio).astype(int), (vc / session.ratio).astype(int)
            ] = idx
            session.cellpix_orig[
                z, (ar / session.ratio).astype(int), (ac / session.ratio).astype(int)
            ] = idx
            session.outpix_orig[
                z, (vr / session.ratio).astype(int), (vc / session.ratio).astype(int)
            ] = idx
        else:
            session.cellpix_orig[z, vr, vc] = idx
            session.cellpix_orig[z, ar, ac] = idx
            session.outpix_orig[z, vr, vc] = idx
            vrr = (vr.copy() * session.ratio).astype(int)
            vcr = (vc.copy() * session.ratio).astype(int)
            mask = np.zeros((np.ptp(vrr) + 4, np.ptp(vcr) + 4), np.uint8)
            pts = np.stack((vcr - vcr.min() + 2, vrr - vrr.min() + 2), axis=-1)[
                :, np.newaxis, :
            ]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            arr, acr = np.nonzero(mask)
            arr, acr = arr + vrr.min() - 2, acr + vcr.min() - 2
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0].squeeze().T
            vrr, vcr = pvr + vrr.min() - 2, pvc + vcr.min() - 2
            arr, acr = np.hstack((np.vstack((vrr, vcr)), np.vstack((arr, acr))))
            session.cellpix_resize[z, vrr, vcr] = idx
            session.cellpix_resize[z, arr, acr] = idx
            session.outpix_resize[z, vrr, vcr] = idx

    if z == session.current_z:
        session.mask_rgb[ar, ac] = color
        session.mask_rgb[vr, vc] = np.array(session.outcolor, dtype=np.uint8)


def paint_mask_at(
    model: MainModel, z, ar, ac, vr, vc, color, idx=None
) -> None:
    session = model.session
    if idx is None:
        idx = int(session.cellpix.max()) + 1
    _paint_mask(session, z, ar, ac, vr, vc, color, idx)


class MainPresenter:
    def __init__(self, view, model: MainModel):
        self.view = view
        self.model = model
        self.cp_model = None
        self.current_model = "cpsam"
        self.current_model_path = None
        self._sync_notifier = create_sync_notifier()
        self._sync_notifier.sync_requested.connect(self._apply_sync_request)

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
        self.request_sync(SyncScope.LABELS_TABLE)
        return result

    def set_instance_class(self, row: int, class_id: int) -> np.ndarray:
        result = self.model.set_instance_class(row, class_id)
        self.request_sync(SyncScope.INSTANCE_EDIT)
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
        self.request_sync(SyncScope.VISIBILITY_EDIT)
        return result

    def set_instance_visible_row(self, row: int, visible: bool) -> np.ndarray:
        result = self.model.set_instance_visible_row(row, visible)
        self.request_sync(SyncScope.VISIBILITY_EDIT)
        return result

    def set_all_instance_visible(self, visible: bool, ncells: int | None = None) -> None:
        if ncells is None:
            ncells = self.ncells()
        self.model.set_all_instance_visible(visible, ncells)
        self.request_sync(SyncScope.VISIBILITY_EDIT)

    def remove_instance_metadata(self, row: int) -> tuple[int, bool]:
        return self.model.remove_instance_metadata(row)

    def append_instance_metadata(
        self, class_id: int = 0, visible: bool = True
    ) -> None:
        self.model.append_instance_metadata(class_id, visible)

    def reset_instance_metadata(self) -> None:
        self.model.reset_instance_metadata()

    def request_sync(
        self, scope: SyncScope, *, prune_selection: bool = True
    ) -> None:
        self._sync_notifier.sync_requested.emit(
            SyncRequest(scope, prune_selection=prune_selection)
        )

    def _apply_sync_request(self, request: SyncRequest) -> None:
        self._sync(request.scope, prune_selection=request.prune_selection)

    def _sync(self, scope: SyncScope, *, prune_selection: bool = True) -> None:
        if not scope:
            return

        if SyncScope.CANVAS_SELECTION in scope and prune_selection:
            cells = self._filter_selectable_cells(self._selected_cell_indices())
            if cells != self._selected_cell_indices():
                self._set_selection_state(cells)

        if SyncScope.LABELS_TABLE in scope:
            self.refresh_labels_table()

        if SyncScope.VISIBILITY_HEADER in scope:
            self.sync_visibility_header_checkbox()

        if SyncScope.CANVAS_IMAGE in scope:
            self.refresh_plot()

        if SyncScope.CANVAS_MASK in scope:
            self._push_mask_overlay()

        if SyncScope.CANVAS_SELECTION in scope:
            self._sync_selection_view()

        if SyncScope.SCALE in scope:
            self.refresh_scale_from_model()

        if SyncScope.CANVAS_MASK in scope or SyncScope.CANVAS_IMAGE in scope:
            self.view.show_window()

    def labels_class_filter(self) -> int | None:
        return InstanceClasses.parse_filter(self.view.read_labels_class_filter())

    def _filter_selectable_cells(self, cells: list[int]) -> list[int]:
        filter_class_id = self.labels_class_filter()
        if filter_class_id is None:
            return sorted({int(c) for c in cells if int(c) > 0})
        self.ensure_instance_classes()
        return sorted(
            {
                int(c)
                for c in cells
                if self.model.instances.label_matches_filter(int(c), filter_class_id)
            }
        )

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
        major_diameters = minor_diameters = None
        if ncells > 0:
            plane = self.session.cellpix[self.session.current_z]
            major_diameters, minor_diameters = get_mask_ellipse_diameters(plane)
        rows = []
        for row in range(ncells):
            class_id = int(self.instance_classes[row])
            major = (
                None
                if major_diameters is None or np.isnan(major_diameters[row])
                else float(major_diameters[row])
            )
            minor = (
                None
                if minor_diameters is None or np.isnan(minor_diameters[row])
                else float(minor_diameters[row])
            )
            rows.append(
                LabelRow(
                    roi=row + 1,
                    class_id=class_id,
                    major_diameter=major,
                    minor_diameter=minor,
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

    def _push_mask_overlay(self) -> None:
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
        mask_rgb, visible = self.model.build_layer_rgb(
            filter_class_id=self.labels_class_filter()
        )
        stroke_z = np.array([s[0][0] for s in self.model.drawing.strokes])
        in_z = np.nonzero(stroke_z == session.current_z)[0]
        for i in in_z:
            stroke = np.array(self.model.drawing.strokes[i])
            mask_rgb[stroke[:, 1], stroke[:, 2]] = np.array([255, 0, 255], dtype=np.uint8)
            visible[stroke[:, 1], stroke[:, 2]] = True
        blend = session.mask_blend
        self.view.render_mask_overlay(mask_rgb, visible, blend)
        self.view.img.setOpacity(max(0.0, 1.0 - blend))

    def refresh_mask_layer(self) -> None:
        """Backward-compatible alias: mask + selection overlay."""
        self.request_sync(SyncScope.CANVAS_MASK | SyncScope.CANVAS_SELECTION)

    def _sync_selection_view(self) -> None:
        cells = self._selected_cell_indices()
        self._sync_labels_table_selection_multi(cells)
        self.refresh_selection_boxes()

    def _set_selection_state(self, cells: list[int]) -> None:
        self.model.selection.selected_cells = cells
        self.model.selection.selected = cells[0] if cells else 0
        self.model.selection.prev_selected = self.model.selection.selected

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
        self.request_sync(SyncScope.FILTER_CHANGE)

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
        self.session.mask_blend = 0.5
        if hasattr(self.view, "mask_blend_slider"):
            self.view.sync_mask_blend_slider(0.5)
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
        self.request_sync(
            SyncScope.SCALE
            | SyncScope.CANVAS_MASK
            | SyncScope.LABELS_TABLE
            | SyncScope.CANVAS_SELECTION
        )
        self.view.render_rect_select_preview(None)

    def on_initialize_images(self, image: np.ndarray) -> None:
        self.model.load_image_stack(image)
        self.clear_all()
        self.view.sliders[0].setValue([0, 255])
        self.request_sync(SyncScope.SCALE)

    def on_image_loaded(self, filename: str, display_filename: str | None = None) -> None:
        self.model.filename = filename
        self.model.series_state.display_filename = display_filename or filename
        self.model.series_state.output_filename = None
        self.model.session.loaded = True
        self.view.set_loaded_chrome(True, ncells=self.ncells())
        self.request_sync(SyncScope.CANVAS_IMAGE | SyncScope.CANVAS_MASK)
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
        from .ui.widgets import as_gray_image

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
        self.request_sync(SyncScope.CANVAS_IMAGE)

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
        self.request_sync(SyncScope.INSTANCE_EDIT, prune_selection=False)
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
                    self.request_sync(SyncScope.INSTANCE_EDIT)
                    self.save_sets()
            else:
                print("GUI_ERROR: cell too small, not drawn")
            drawing.current_stroke = []
            drawing.strokes = []
            drawing.current_point_set = []
            self.request_sync(SyncScope.CANVAS_MASK)

    def remove_stroke(self, delete_points=True, stroke_ind=-1) -> None:
        stroke = np.array(self.model.drawing.strokes[stroke_ind])
        c_z = self.session.current_z
        in_z = stroke[0, 0] == c_z
        if in_z:
            outpix = self.session.outpix[c_z, stroke[:, 1], stroke[:, 2]] > 0
            self.session.mask_rgb[stroke[~outpix, 1], stroke[~outpix, 2]] = 0
            cellpix = self.session.cellpix[c_z, stroke[:, 1], stroke[:, 2]]
            ccol = self.session.cellcolors.copy()
            if self.model.selection.selected > 0:
                ccol[self.model.selection.selected] = np.array([255, 255, 255])
            self.session.mask_rgb[stroke[:, 1], stroke[:, 2]] = ccol[cellpix]
            self.session.mask_rgb[stroke[outpix, 1], stroke[outpix, 2]] = np.array(
                self.session.outcolor, dtype=np.uint8
            )
            if delete_points:
                del self.model.drawing.current_point_set[stroke_ind]
            self.request_sync(SyncScope.CANVAS_MASK)
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
            self.request_sync(SyncScope.INSTANCE_EDIT)
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
            self.request_sync(SyncScope.INSTANCE_EDIT)
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
            self.request_sync(SyncScope.INSTANCE_EDIT)
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
        self.request_sync(
            SyncScope.CANVAS_IMAGE | SyncScope.CANVAS_MASK | SyncScope.LABELS_TABLE
        )

    def save_sets(self) -> None:
        from cellpose.app_core import write_session

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
                ) = get_train_set(
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
                ) = get_train_set(
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
        view.ncells_counter.valueChanged.connect(
            lambda *_: self.request_sync(SyncScope.LABELS_TABLE)
        )
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
            lambda *_: self.request_sync(SyncScope.SCALE)
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
        view.mask_blend_changed.connect(self.on_mask_blend_changed)
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

    def _custom_model_dir(self):
        custom_dir = MODEL_DIR.joinpath("custom")
        custom_dir.mkdir(parents=True, exist_ok=True)
        return custom_dir

    def _write_model_list(self, model_strings) -> None:
        with open(MODEL_LIST_PATH, "w") as textfile:
            for model_string in model_strings:
                textfile.write(model_string + "\n")

    def init_model_list(self) -> None:
        self._custom_model_dir()
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
        from cellpose.app_core import read_session

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

        templates = prompt_series_templates(
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
        target = self._custom_model_dir().joinpath(fname)
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
        self._write_model_list(model_strings)
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
            self._write_model_list(model_strings)
            model_path = self._custom_model_dir().joinpath(modelstr)
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
            self.view.render_image_plane(
                image,
                [low, high],
                lut=None,
                opacity=max(0.0, 1.0 - session.mask_blend),
            )
        else:
            image = np.zeros((session.ly, session.lx), np.uint8)
            if (
                len(session.flows) >= view_index - 1
                and len(session.flows[view_index - 1]) > 0
            ):
                image = session.flows[view_index - 1][session.current_z]
            image_opacity = max(0.0, 1.0 - session.mask_blend)
            if view_index > 1:
                self.view.render_image_plane(
                    image, [0.0, 255.0], lut=self.view.bwr, opacity=image_opacity
                )
            else:
                self.view.render_image_plane(
                    image, [0.0, 255.0], lut=None, opacity=image_opacity
                )

        low, high = session.saturation[0][session.current_z]
        self.view.sync_saturation_slider(low, high)
        self.view.show_window()

    def on_mask_blend_changed(self) -> None:
        if not self.session.loaded:
            return
        self.session.mask_blend = self.view.read_mask_blend()
        self.request_sync(SyncScope.CANVAS_IMAGE | SyncScope.CANVAS_MASK)

    def on_saturation_changed(self) -> None:
        if self.session.loaded:
            sval = self.view.sliders[0].value()
            self.session.saturation[0][self.session.current_z] = sval
            self.request_sync(SyncScope.CANVAS_IMAGE)

    def on_view_mode_changed(self) -> None:
        self.request_sync(SyncScope.CANVAS_IMAGE)

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
        cells = self._filter_selectable_cells(cells)
        self._set_selection_state(cells)
        self.request_sync(SyncScope.SELECTION_CHANGE, prune_selection=False)

    def select_cell(self, idx: int) -> None:
        if idx > 0 and idx not in self._filter_selectable_cells([idx]):
            idx = 0
        self.model.selection.prev_selected = self.model.selection.selected
        self._set_selection_state([idx] if idx > 0 else [])
        self.request_sync(SyncScope.SELECTION_CHANGE, prune_selection=False)

    def unselect_cell(self) -> None:
        self._set_selection_state([])
        self.request_sync(SyncScope.SELECTION_CHANGE, prune_selection=False)

    def on_labels_table_selection_changed(self) -> None:
        view = self.view
        if view._refreshing_labels_table or view._syncing_labels_table_selection:
            return
        selected_rows = view.LabelsTable.selectionModel().selectedRows()
        if not selected_rows:
            if self.model.selection.selected > 0 or self.model.selection.selected_cells:
                self._set_selection_state([])
                self.request_sync(SyncScope.SELECTION_CHANGE, prune_selection=False)
            return
        cells = self._filter_selectable_cells(
            sorted({row.row() + 1 for row in selected_rows})
        )
        if cells == self.model.selection.selected_cells:
            self.request_sync(SyncScope.SELECTION_CHANGE, prune_selection=False)
            return
        self.model.selection.prev_selected = self.model.selection.selected
        self._set_selection_state(cells)
        self.request_sync(SyncScope.SELECTION_CHANGE, prune_selection=False)

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
        if idx > 0 and idx not in self._filter_selectable_cells([idx]):
            idx = 0
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
        if idx > 0 and idx not in self._filter_selectable_cells([idx]):
            return
        if idx > 0:
            z = self.session.current_z
            self.session.mask_rgb[self.session.cellpix[z] == idx] = np.array(
                [255, 255, 255], dtype=np.uint8
            )
            self.request_sync(SyncScope.CANVAS_MASK)

    def unselect_cell_multi(self, idx: int) -> None:
        z = self.session.current_z
        self.session.mask_rgb[self.session.cellpix[z] == idx] = self.session.cellcolors[
            idx
        ]
        self.session.mask_rgb[self.session.outpix[z] == idx] = np.array(
            self.session.outcolor, dtype=np.uint8
        )
        self.request_sync(SyncScope.CANVAS_MASK)

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

        self.request_sync(SyncScope.CANVAS_MASK)

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
        self.request_sync(SyncScope.INSTANCE_EDIT)
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
        self.request_sync(SyncScope.CANVAS_MASK)

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