"""
Presenter for the Cellpose GUI.

MainPresenter connects MainView (widgets + rendering) to MainModel (state).
"""

from __future__ import annotations

import os
import time
from typing import Any

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox

from .. import dynamics
from ..transforms import normalize99, resize_image
from . import io, series
from .model import (
    InstanceClasses,
    MainModel,
    PreprocessingParameters,
    SegmentationParameters,
    SeriesState,
)


class MainPresenter:
    def __init__(self, view, model: MainModel):
        self.view = view
        self.model = model

    def training_params(self) -> dict[str, Any]:
        return self.model.training_params

    def instance_classes(self) -> np.ndarray:
        return self.model.instance_classes

    def instance_visible(self) -> np.ndarray:
        return self.model.instance_visible

    # ---- training ----

    def reset_training_parameters(self) -> dict[str, Any]:
        return self.model.reset_training_parameters()

    def set_training_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        return self.model.set_training_parameters(values)

    # ---- series ----

    def reset_series(self) -> SeriesState:
        state = self.model.reset_series()
        self.view.sync_series_state(state)
        self.view.set_series_navigation_state()
        return state

    def set_series(
        self, dataset: dict[str, Any] | None = None, record_index: int | None = None
    ) -> SeriesState:
        state = self.model.set_series(dataset=dataset, record_index=record_index)
        self.view.sync_series_state(state)
        if dataset is not None and record_index is not None:
            self.view.set_series_navigation_state(dataset, record_index)
        else:
            self.view.set_series_navigation_state()
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

    def preprocessing_parameters(self) -> PreprocessingParameters:
        widgets = self.view.read_preprocessing_widgets()
        return PreprocessingParameters.from_values(
            sharpen_radius=float(widgets["sharpen_radius"]),
            smooth_radius=float(widgets["smooth_radius"]),
            tile_norm_blocksize=float(widgets["tile_norm_blocksize"]),
            tile_norm_smooth3D=float(widgets["tile_norm_smooth3D"]),
            norm3D=bool(widgets["norm3D"]),
            image_shape=(self.view.Ly, self.view.Lx),
            invert=bool(widgets.get("invert", False)),
        )

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
        self.view.apply_preprocessing_widgets(model_params)

    # ---- instances ----

    def ensure_instance_classes(
        self, ncells: int | None = None, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.view.ncells()
        return self.model.ensure_instance_classes(ncells, current_values=current_values)

    def set_instance_classes(
        self, values: np.ndarray | list[int] | None = None, ncells: int | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.view.ncells()
        result = self.model.set_instance_classes(ncells, values)
        self.view.refresh_instance_table()
        return result

    def set_instance_class(self, row: int, class_id: int) -> np.ndarray:
        result = self.model.set_instance_class(row, class_id)
        self.view.refresh_instance_table()
        self.view.draw_layer()
        self.view.update_layer()
        return result

    def ensure_instance_visible(
        self, ncells: int | None = None, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.view.ncells()
        return self.model.ensure_instance_visible(ncells, current_values=current_values)

    def set_instance_visible(
        self, values: np.ndarray | list[bool] | None = None, ncells: int | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.view.ncells()
        result = self.model.set_instance_visible(ncells, values)
        self.view.refresh_instance_table()
        return result

    def set_instance_visible_row(self, row: int, visible: bool) -> np.ndarray:
        result = self.model.set_instance_visible_row(row, visible)
        self.view.draw_layer()
        self.view.update_layer()
        return result

    def set_all_instance_visible(self, visible: bool, ncells: int | None = None) -> None:
        if ncells is None:
            ncells = self.view.ncells()
        self.model.set_all_instance_visible(visible, ncells)
        self.view.draw_layer()
        self.view.update_layer()

    def remove_instance_metadata(self, row: int) -> tuple[int, bool]:
        return self.model.remove_instance_metadata(row)

    def append_instance_metadata(
        self, class_id: int = 0, visible: bool = True
    ) -> None:
        self.model.append_instance_metadata(class_id, visible)

    def reset_instance_metadata(self) -> None:
        self.model.reset_instance_metadata()

    def instance_class_filter(self) -> int | None:
        return InstanceClasses.parse_filter(self.view.instance_class_filter_text())

    def visible_cell_pixels(self, cellpix: np.ndarray) -> np.ndarray:
        self.ensure_instance_classes()
        self.ensure_instance_visible()
        return self.model.visible_cell_pixels(cellpix, self.instance_class_filter())

    def on_instance_filter_changed(self) -> None:
        self.view.refresh_instance_table()
        self.view.draw_layer()
        self.view.update_layer()

    def run_selected_model(self) -> None:
        model_name, custom = self.view._selected_segmentation_model()
        if custom:
            self.compute_segmentation(custom=True)
        else:
            self.compute_segmentation(model_name=model_name)

    def apply_filter(self) -> None:
        self.view.restore = "filter"
        normalize_params = self.view.get_normalize_params()
        if (
            normalize_params["sharpen_radius"] == 0
            and normalize_params["smooth_radius"] == 0
            and normalize_params["tile_norm_blocksize"] == 0
        ):
            print("GUI_ERROR: no filtering settings on (use custom filter settings)")
            self.view.restore = None
            return
        self.view.compute_saturation(apply_preprocessing=True)

    def get_prev_image(self) -> None:
        images, idx = self.view.get_files()
        idx = (idx - 1) % len(images)
        if self.view.series_dataset is not None:
            try:
                io._load_series_item(
                    self.view, self.view.series_dataset, idx, load_3D=self.view.load_3D
                )
            except Exception as e:
                print(f"ERROR: {e}")
                QMessageBox.warning(self.view, "Load folder with pattern", str(e))
        else:
            io._load_image(self.view, filename=images[idx])

    def get_next_image(self, load_seg: bool = True) -> None:
        images, idx = self.view.get_files()
        idx = (idx + 1) % len(images)
        if self.view.series_dataset is not None:
            try:
                io._load_series_item(
                    self.view,
                    self.view.series_dataset,
                    idx,
                    load_seg=load_seg,
                    load_3D=self.view.load_3D,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                QMessageBox.warning(self.view, "Load folder with pattern", str(e))
        else:
            io._load_image(self.view, filename=images[idx], load_seg=load_seg)

    def navigate_series_from_sliders(
        self, axis_name: str | None = None, delta: int = 0
    ) -> None:
        if delta != 0:
            control = self.view.series_nav_controls.get(axis_name)
            if control is None:
                return
            slider = control["slider"]
            if not slider.isEnabled():
                return
            value = max(0, min(slider.maximum(), slider.value() + delta))
            old_updating_state = self.view._updating_series_navigation
            self.view._updating_series_navigation = True
            try:
                slider.setValue(value)
            finally:
                self.view._updating_series_navigation = old_updating_state

        if (
            self.view._updating_series_navigation
            or self.view.series_dataset is None
            or self.view.series_index is None
        ):
            return

        if axis_name is None:
            return

        try:
            record_index = series.resolve_series_record_index(
                self.view.series_dataset,
                position=self.view.series_dataset["axes"]["position"][
                    self.view.series_nav_controls["position"]["slider"].value()
                ],
                time=self.view.series_dataset["axes"]["time"][
                    self.view.series_nav_controls["time"]["slider"].value()
                ],
                channel=self.view.series_dataset["axes"]["channel"][
                    self.view.series_nav_controls["channel"]["slider"].value()
                ],
                z=self.view.series_dataset["axes"]["z"][
                    self.view.series_nav_controls["z"]["slider"].value()
                ],
            )
        except Exception as e:
            self.view.set_series_navigation_state(
                self.view.series_dataset, self.view.series_index
            )
            QMessageBox.warning(self.view, "Load folder with pattern", str(e))
            return

        if record_index == self.view.series_index:
            return

        try:
            io._load_series_item(
                self.view,
                self.view.series_dataset,
                record_index,
                load_3D=self.view.load_3D,
            )
        except Exception as e:
            self.view.set_series_navigation_state(
                self.view.series_dataset, self.view.series_index
            )
            print(f"ERROR: {e}")
            QMessageBox.warning(self.view, "Load folder with pattern", str(e))

    def remove_cell(self, idx) -> None:
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        idx.sort(reverse=True)
        for i in idx:
            self.view.remove_single_cell(i)
        self.view._sync_ncells_counter()
        self.view.update_layer()

        if self.view.ncells() == 0:
            self.view.ClearButton.setEnabled(False)
        if self.view.NZ == 1:
            io._save_sets_with_check(self.view)

    def add_set(self) -> None:
        if len(self.view.current_point_set) > 0:
            while len(self.view.strokes) > 0:
                self.view.remove_stroke(delete_points=False)
            if len(self.view.current_point_set[0]) > 8:
                color = self.view.colormap[self.view.ncells(), :3]
                median = self.view.add_mask(
                    points=self.view.current_point_set, color=color
                )
                if median is not None:
                    self.view.removed_cell = []
                    self.view.toggle_mask_ops()
                    self.view.cellcolors = np.append(
                        self.view.cellcolors, color[np.newaxis, :], axis=0
                    )
                    self.view.ismanual = np.append(self.view.ismanual, True)
                    self.append_instance_metadata(self.view.default_class_id(), True)
                    self.view._sync_ncells_counter()
                    self.view.draw_layer()
                    if self.view.NZ == 1:
                        io._save_sets_with_check(self.view)
            else:
                print("GUI_ERROR: cell too small, not drawn")
            self.view.current_stroke = []
            self.view.strokes = []
            self.view.current_point_set = []
            self.view.update_layer()

    def _apply_masks_to_view(self, masks: np.ndarray) -> None:
        ncells = self.model.apply_masks(
            masks, outlines=None, colormap=self.view.colormap
        )
        self.view._sync_ncells_counter()
        print(f"GUI_INFO: {ncells} masks found")
        if ncells > 0:
            self.view.draw_layer()
            self.view.toggle_mask_ops()
        if self.view.restore == "filter" or self.view.stack_filtered is not None:
            self.view.ViewDropDown.setCurrentIndex(self.view.ViewDropDown.count() - 1)
            print("set denoised/filtered view")
        else:
            self.view.ViewDropDown.setCurrentIndex(0)

    def compute_cprob(self) -> None:
        if getattr(self.view, "recompute_masks", False):
            segmentation_params = self.view.get_segmentation_parameters()
            min_size = (
                int(self.view.min_size.text())
                if not isinstance(self.view.min_size, int)
                else self.view.min_size
            )

            self.view.logger.info(
                "computing masks with cell prob=%0.3f, flow error threshold=%0.3f"
                % (
                    segmentation_params["cellprob_threshold"],
                    segmentation_params["flow_threshold"],
                )
            )

            try:
                dP = self.view.flows[2].squeeze()
                cellprob = self.view.flows[3].squeeze()
            except IndexError:
                self.view.logger.error("Flows don't exist, try running model again.")
                return

            maski = dynamics.resize_and_compute_masks(
                dP=dP,
                cellprob=cellprob,
                niter=segmentation_params["niter"],
                do_3D=self.view.load_3D,
                min_size=min_size,
                cellprob_threshold=segmentation_params["cellprob_threshold"],
                flow_threshold=segmentation_params["flow_threshold"],
            )

            if maski.ndim < 3:
                maski = maski[np.newaxis, ...]
            self.view.logger.info("%d cells found" % (len(np.unique(maski)[1:])))
            self._apply_masks_to_view(maski)
            self.view.show()

    def compute_segmentation(
        self, custom: bool = False, model_name: str | None = None, load_model: bool = True
    ) -> None:
        self.view.progress.setValue(0)
        try:
            tic = time.time()
            self.view.clear_all()
            self.view.flows = [[], [], []]
            if load_model:
                self.view.initialize_model(model_name=model_name, custom=custom)
            self.view.progress.setValue(10)
            do_3D = self.view.load_3D
            stitch_threshold = (
                float(self.view.stitch_threshold.text())
                if not isinstance(self.view.stitch_threshold, float)
                else self.view.stitch_threshold
            )
            anisotropy = (
                float(self.view.anisotropy.text())
                if not isinstance(self.view.anisotropy, float)
                else self.view.anisotropy
            )
            flow3D_smooth = (
                float(self.view.flow3D_smooth.text())
                if not isinstance(self.view.flow3D_smooth, float)
                else self.view.flow3D_smooth
            )
            min_size = (
                int(self.view.min_size.text())
                if not isinstance(self.view.min_size, int)
                else self.view.min_size
            )

            do_3D = False if stitch_threshold > 0.0 else do_3D

            if self.view.restore == "filter":
                data = self.view.stack_filtered.copy().squeeze()
            else:
                data = self.view.stack.copy().squeeze()

            segmentation_params = self.view.get_segmentation_parameters()
            normalize_params = self.view.get_normalize_params()
            print(normalize_params)
            try:
                masks, flows = self.view.cp_model.eval(
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
                    progress=self.view.progress,
                    z_axis=0 if self.view.NZ > 1 else None,
                )[:2]
            except Exception as e:
                print("NET ERROR: %s" % e)
                self.view.progress.setValue(0)
                return

            self.view.progress.setValue(75)

            flows_new = []
            flows_new.append(flows[0].copy())
            flows_new.append(
                (np.clip(normalize99(flows[2].copy()), 0, 1) * 255).astype("uint8")
            )
            flows_new.append(flows[1].copy())
            flows_new.append(flows[2].copy())

            if self.view.load_3D:
                if stitch_threshold == 0.0:
                    flows_new.append((flows[1][0] / 10 * 127 + 127).astype("uint8"))
                else:
                    flows_new.append(np.zeros(flows[1][0].shape, dtype="uint8"))

            if not self.view.load_3D:
                if self.view.restore and "upsample" in self.view.restore:
                    self.view.Ly, self.view.Lx = self.view.Lyr, self.view.Lxr

                if flows_new[0].shape[-3:-1] != (self.view.Ly, self.view.Lx):
                    self.view.flows = []
                    for j in range(len(flows_new)):
                        self.view.flows.append(
                            resize_image(
                                flows_new[j],
                                Ly=self.view.Ly,
                                Lx=self.view.Lx,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )
                else:
                    self.view.flows = flows_new
            else:
                self.view.flows = []
                Lz, Ly, Lx = self.view.NZ, self.view.Ly, self.view.Lx
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
                    self.view.flows.append(flow0)

            if self.view.NZ == 1:
                masks = masks[np.newaxis, ...]
                self.view.flows = [
                    self.view.flows[n][np.newaxis, ...]
                    for n in range(len(self.view.flows))
                ]

            self.view.logger.info(
                "%d cells found with model in %0.3f sec"
                % (len(np.unique(masks)[1:]), time.time() - tic)
            )
            self.view.progress.setValue(80)
            self._apply_masks_to_view(masks)
            self.view.progress.setValue(100)
            if not do_3D and not stitch_threshold > 0:
                self.view.recompute_masks = True
            else:
                self.view.recompute_masks = False
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
        self.view._sync_ncells_counter()
        print(f"GUI_INFO: {ncells} masks found")
        if ncells > 0:
            self.view.draw_layer()
            self.view.toggle_mask_ops()
        if hasattr(self.view, "stack_filtered") and self.view.stack_filtered is not None:
            self.view.ViewDropDown.setCurrentIndex(self.view.ViewDropDown.count() - 1)
        else:
            self.view.ViewDropDown.setCurrentIndex(0)

    def save_sets(self) -> None:
        import os

        filename = self.output_filename(str(self.view.filename))
        base = os.path.splitext(filename)[0]
        segmentation_params = self.segmentation_parameters_dict()
        normalize_params = self.view.get_normalize_params()
        self.model.segmentation_params = segmentation_params
        self.model.preprocessing_params = normalize_params
        dat = self.model.to_seg_dict(
            current_model_path=getattr(self.view, "current_model_path", 0),
            normalize_params=normalize_params,
            segmentation_params=segmentation_params,
        )
        if (
            getattr(self.view, "series_dataset", None) is not None
            and self.view.series_index is not None
        ):
            dat["image_series"] = series.build_series_metadata(
                self.view.series_dataset, self.view.series_index
            )
        try:
            np.save(base + "_seg.npy", dat)
            print(
                "GUI_INFO: %d ROIs saved to %s"
                % (self.view.ncells(), base + "_seg.npy")
            )
        except Exception as e:
            print(f"ERROR: {e}")
