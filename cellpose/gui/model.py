"""
Model layer for the Cellpose GUI (MVP).

MainModel aggregates mutable GUI state. Value types and mask/session dataclasses
hold domain logic without Qt dependencies.
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import series


@dataclass
class SeriesState:
    dataset: dict[str, Any] | None = None
    record_index: int | None = None
    output_filename: str | None = None
    display_filename: str | None = None
    filename: str | None = None

    @classmethod
    def empty(cls) -> "SeriesState":
        return cls()

    @classmethod
    def from_record(cls, dataset: dict[str, Any], record_index: int) -> "SeriesState":
        record = dataset["records"][record_index]
        return cls(
            dataset=dataset,
            record_index=record_index,
            output_filename=series.get_output_filename(dataset, record_index),
            display_filename=record["label"],
            filename=record["path"],
        )


@dataclass
class TrainingParameters:
    model_index: int = 0
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    n_epochs: int = 100
    model_name: str = ""
    train_data_folder: str = ""
    model_save_folder: str = ""

    @classmethod
    def create_default(
        cls, model_save_folder: str, now: _datetime.datetime | None = None
    ) -> "TrainingParameters":
        now = now or _datetime.datetime.now()
        return cls(
            model_name="cpsam" + now.strftime("_%Y%m%d_%H%M%S"),
            model_save_folder=model_save_folder,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_index": int(self.model_index),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "n_epochs": int(self.n_epochs),
            "model_name": str(self.model_name),
            "train_data_folder": str(self.train_data_folder),
            "model_save_folder": str(self.model_save_folder),
        }


@dataclass(frozen=True)
class SegmentationParameters:
    diameter: float | None
    flow_threshold: float
    cellprob_threshold: float
    percentile: tuple[float, float]
    niter: int

    @classmethod
    def from_values(
        cls,
        diameter: float,
        flow_threshold: float,
        cellprob_threshold: float,
        percentile_low: float,
        percentile_high: float,
        niter: int,
    ) -> "SegmentationParameters":
        diameter_value = float(diameter)
        low = float(percentile_low)
        high = float(percentile_high)
        if not 0 <= low <= 100 or not 0 <= high <= 100 or low >= high:
            raise ValueError(
                "normalization percentile range must be 0 <= low < high <= 100"
            )
        niter_value = int(niter)
        if niter_value < 1:
            niter_value = 200
        return cls(
            diameter=diameter_value if diameter_value > 0 else None,
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
            percentile=(low, high),
            niter=niter_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "percentile": self.percentile,
            "niter": self.niter,
        }


@dataclass
class InstanceClasses:
    values: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )

    def ensure_size(
        self, ncells: int, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        values = self.values if current_values is None else current_values
        values = np.asarray(values, dtype=np.int32).ravel()
        values = np.maximum(values, 0)
        if len(values) < ncells:
            pad = np.zeros(ncells - len(values), dtype=np.int32)
            values = np.concatenate((values, pad))
        elif len(values) > ncells:
            values = values[:ncells]
        self.values = values
        return self.values

    def replace(
        self, ncells: int, values: np.ndarray | list[int] | None = None
    ) -> np.ndarray:
        result = np.zeros(ncells, dtype=np.int32)
        if values is not None:
            loaded = np.asarray(values, dtype=np.int32).ravel()
            loaded = np.maximum(loaded, 0)
            n = min(ncells, len(loaded))
            result[:n] = loaded[:n]
        self.values = result
        return self.values

    def set_class(self, row: int, class_id: int) -> np.ndarray:
        if class_id < 0:
            raise ValueError("class_id must be non-negative")
        if row >= len(self.values):
            return self.values
        self.values[row] = int(class_id)
        return self.values

    @staticmethod
    def parse_filter(text: str) -> int | None:
        text = text.strip()
        if text == "":
            return None
        try:
            class_id = int(text)
        except ValueError:
            return None
        return class_id if class_id >= 0 else None

    def visible_cell_pixels(
        self,
        cellpix: np.ndarray,
        filter_class_id: int | None,
        visibility: np.ndarray | None = None,
    ) -> np.ndarray:
        max_label = int(cellpix.max())
        if max_label == 0:
            return np.zeros(cellpix.shape, dtype=bool)

        visible_labels = np.zeros(max_label + 1, dtype=bool)
        nlabels = min(max_label, len(self.values))
        if nlabels > 0:
            if filter_class_id is None:
                visible_labels[1 : nlabels + 1] = True
            else:
                visible_labels[1 : nlabels + 1] = (
                    self.values[:nlabels] == filter_class_id
                )
        if visibility is not None and len(visibility) > 0:
            nvisible = min(max_label, len(visibility))
            visible_labels[1 : nvisible + 1] &= np.asarray(
                visibility[:nvisible], dtype=bool
            )
        return visible_labels[cellpix]


@dataclass
class InstanceVisibility:
    values: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )

    def ensure_size(
        self, ncells: int, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        values = self.values if current_values is None else current_values
        values = np.asarray(values, dtype=bool).ravel()
        if len(values) < ncells:
            pad = np.ones(ncells - len(values), dtype=bool)
            values = np.concatenate((values, pad))
        elif len(values) > ncells:
            values = values[:ncells]
        self.values = values
        return self.values

    def replace(
        self, ncells: int, values: np.ndarray | list[bool] | None = None
    ) -> np.ndarray:
        result = np.ones(ncells, dtype=bool)
        if values is not None:
            loaded = np.asarray(values, dtype=bool).ravel()
            n = min(ncells, len(loaded))
            result[:n] = loaded[:n]
        self.values = result
        return self.values

    def set_visible(self, row: int, visible: bool) -> np.ndarray:
        if row >= len(self.values):
            return self.values
        self.values[row] = bool(visible)
        return self.values


@dataclass
class ImageSession:
    stack: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 256, 256, 3), dtype=np.float32)
    )
    cellpix: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 256, 256), dtype=np.uint16)
    )
    outpix: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 256, 256), dtype=np.uint16)
    )
    flows: list[Any] = field(default_factory=lambda: [[], [], [], [], [[]]])
    cellcolors: np.ndarray = field(
        default_factory=lambda: np.array([[255, 255, 255]], dtype=np.uint8)
    )
    ismanual: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    zdraw: list[Any] = field(default_factory=list)
    stack_filtered: np.ndarray | None = None
    restore: str | None = None
    ratio: float = 1.0
    loaded: bool = False
    current_z: int = 0
    nz: int = 1
    ly: int = 256
    lx: int = 256
    ly0: int = 256
    lx0: int = 256
    lyr: int = 256
    lxr: int = 256
    saturation: list[list[list[float]]] = field(
        default_factory=lambda: [[[0, 255]]]
    )
    track_changes: list[Any] = field(default_factory=list)
    recompute_masks: bool = False
    layerz: np.ndarray = field(
        default_factory=lambda: np.zeros((256, 256, 4), dtype=np.uint8)
    )
    cellpix_orig: np.ndarray | None = None
    cellpix_resize: np.ndarray | None = None
    outpix_orig: np.ndarray | None = None
    outpix_resize: np.ndarray | None = None
    opacity: int = 128
    outcolor: list[int] = field(default_factory=lambda: [200, 200, 255, 200])
    resize: bool = False

    @classmethod
    def empty(cls, ly: int = 256, lx: int = 256) -> "ImageSession":
        session = cls()
        session.ly = ly
        session.lx = lx
        session.ly0 = ly
        session.lx0 = lx
        session.lyr = ly
        session.lxr = lx
        session.nz = 1
        session.stack = np.zeros((1, ly, lx, 3), dtype=np.float32)
        session.cellpix = np.zeros((1, ly, lx), dtype=np.uint16)
        session.outpix = np.zeros((1, ly, lx), dtype=np.uint16)
        session.layerz = np.zeros((ly, lx, 4), dtype=np.uint8)
        session.saturation = [[[0, 255] for _ in range(session.nz)]]
        return session


@dataclass
class SelectionState:
    selected: int = 0
    prev_selected: int = 0
    selected_cells: list[int] = field(default_factory=list)
    removed_cell: list[Any] = field(default_factory=list)
    removing_cells_list: list[int] = field(default_factory=list)
    deleting_multiple: bool = False
    removing_region: bool = False

    def reset(self) -> None:
        self.selected = 0
        self.prev_selected = 0
        self.selected_cells = []
        self.removed_cell = []
        self.removing_cells_list = []
        self.deleting_multiple = False
        self.removing_region = False


@dataclass
class DrawingState:
    strokes: list[Any] = field(default_factory=list)
    current_point_set: list[Any] = field(default_factory=list)
    current_stroke: list[Any] = field(default_factory=list)
    in_stroke: bool = False
    stroke_appended: bool = True
    brush_mode: bool = False

    def reset(self) -> None:
        self.strokes = []
        self.current_point_set = []
        self.current_stroke = []
        self.in_stroke = False
        self.stroke_appended = True
        self.brush_mode = False


class MainModel:
    """Aggregate GUI model state for the MVP stack."""

    def __init__(self, model_save_folder: str):
        self.model_save_folder = model_save_folder
        self.series_state = SeriesState.empty()
        self.session = ImageSession.empty()
        self.selection = SelectionState()
        self.drawing = DrawingState()
        self.instances = InstanceClasses()
        self.instance_visibility = InstanceVisibility()
        self.training_params: dict[str, Any] = {}
        self.segmentation_params: dict[str, Any] | None = None
        self.preprocessing_params: dict[str, Any] | None = None
        self.reset_training_parameters()

    @property
    def ncells(self) -> int:
        if not self.session.loaded:
            return 0
        return int(self.session.cellpix.max())

    @property
    def filename(self) -> Any:
        return self.series_state.filename

    @filename.setter
    def filename(self, value: Any) -> None:
        self.series_state.filename = value

    @property
    def display_filename(self) -> str | None:
        return self.series_state.display_filename

    @display_filename.setter
    def display_filename(self, value: str | None) -> None:
        self.series_state.display_filename = value

    @property
    def output_filename(self) -> str | None:
        return self.series_state.output_filename

    @output_filename.setter
    def output_filename(self, value: str | None) -> None:
        self.series_state.output_filename = value

    @property
    def instance_classes(self) -> np.ndarray:
        return self.instances.values

    @property
    def instance_visible(self) -> np.ndarray:
        return self.instance_visibility.values

    def reset_training_parameters(self) -> dict[str, Any]:
        defaults = TrainingParameters.create_default(self.model_save_folder)
        self.training_params.clear()
        self.training_params.update(defaults.to_dict())
        return self.training_params

    def set_training_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        self.training_params.clear()
        self.training_params.update(dict(values))
        return self.training_params

    def reset_series(self) -> SeriesState:
        self.series_state = SeriesState.empty()
        return self.series_state

    def set_series(
        self, dataset: dict[str, Any] | None = None, record_index: int | None = None
    ) -> SeriesState:
        if dataset is None or record_index is None:
            return self.reset_series()
        self.series_state = SeriesState.from_record(dataset, record_index)
        return self.series_state

    def output_filename(self, fallback_filename: str) -> str:
        return self.series_state.output_filename or fallback_filename

    def ensure_instance_classes(
        self, ncells: int | None = None, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.ncells
        return self.instances.ensure_size(ncells, current_values=current_values)

    def set_instance_classes(
        self, ncells: int | None = None, values: np.ndarray | list[int] | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.ncells
        return self.instances.replace(ncells, values)

    def set_instance_class(self, row: int, class_id: int) -> np.ndarray:
        return self.instances.set_class(row, class_id)

    def ensure_instance_visible(
        self, ncells: int | None = None, current_values: np.ndarray | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.ncells
        return self.instance_visibility.ensure_size(
            ncells, current_values=current_values
        )

    def set_instance_visible(
        self, ncells: int | None = None, values: np.ndarray | list[bool] | None = None
    ) -> np.ndarray:
        if ncells is None:
            ncells = self.ncells
        return self.instance_visibility.replace(ncells, values)

    def set_instance_visible_row(self, row: int, visible: bool) -> np.ndarray:
        return self.instance_visibility.set_visible(row, visible)

    def set_all_instance_visible(self, visible: bool, ncells: int | None = None) -> None:
        if ncells is None:
            ncells = self.ncells
        if ncells == 0:
            return
        self.ensure_instance_visible(ncells)
        self.instance_visibility.values[:ncells] = visible

    def remove_instance_metadata(self, row: int) -> tuple[int, bool]:
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

    def append_instance_metadata(self, class_id: int = 0, visible: bool = True) -> None:
        self.instances.values = np.append(self.instances.values, np.int32(class_id))
        self.instance_visibility.values = np.append(
            self.instance_visibility.values, visible
        )

    def reset_instance_metadata(self) -> None:
        self.instances.values = np.zeros(0, dtype=np.int32)
        self.instance_visibility.values = np.zeros(0, dtype=bool)

    def visible_cell_pixels(
        self, cellpix: np.ndarray | None = None, filter_class_id: int | None = None
    ) -> np.ndarray:
        if cellpix is None:
            cellpix = self.session.cellpix[self.session.current_z]
        if filter_class_id is None:
            filter_class_id = InstanceClasses.parse_filter("")
        return self.instances.visible_cell_pixels(
            cellpix, filter_class_id, self.instance_visibility.values
        )

    def load_image_stack(self, image: np.ndarray, load_3d: bool = False) -> None:
        session = self.session
        session.stack = image
        if load_3d:
            session.nz = len(session.stack)
        else:
            session.nz = 1
            session.stack = session.stack[np.newaxis, ...]

        img_min = session.stack.min()
        img_max = session.stack.max()
        session.stack = session.stack.astype(np.float32)
        session.stack -= img_min
        if img_max > img_min + 1e-3:
            session.stack /= img_max - img_min
        session.stack *= 255

        session.ly, session.lx = session.stack.shape[-3:-1]
        session.ly0, session.lx0 = session.ly, session.lx
        session.layerz = 255 * np.ones((session.ly, session.lx, 4), dtype=np.uint8)
        if session.stack_filtered is not None:
            session.lyr, session.lxr = session.stack_filtered.shape[-3:-1]
        elif session.restore and "upsample" in session.restore:
            session.lyr = int(session.ly * session.ratio)
            session.lxr = int(session.lx * session.ratio)
        else:
            session.lyr, session.lxr = session.ly, session.lx
        session.saturation = [[[0, 255] for _ in range(session.nz)]]
        session.track_changes = []
        session.loaded = True
        if load_3d:
            session.current_z = int(np.floor(session.nz / 2))
        else:
            session.current_z = 0

    def clear_masks(self) -> None:
        session = self.session
        if session.restore and "upsample" in session.restore:
            session.layerz = np.zeros((session.lyr, session.lxr, 4), dtype=np.uint8)
            session.cellpix = np.zeros((session.nz, session.lyr, session.lxr), np.uint16)
            session.outpix = np.zeros((session.nz, session.lyr, session.lxr), np.uint16)
            session.cellpix_resize = session.cellpix.copy()
            session.outpix_resize = session.outpix.copy()
            session.cellpix_orig = np.zeros((session.nz, session.ly0, session.lx0), np.uint16)
            session.outpix_orig = np.zeros((session.nz, session.ly0, session.lx0), np.uint16)
        else:
            session.layerz = np.zeros((session.ly, session.lx, 4), dtype=np.uint8)
            session.cellpix = np.zeros((session.nz, session.ly, session.lx), np.uint16)
            session.outpix = np.zeros((session.nz, session.ly, session.lx), np.uint16)
        session.cellcolors = np.array([[255, 255, 255]], dtype=np.uint8)
        self.reset_instance_metadata()
        self.selection.selected = 0
        self.selection.prev_selected = 0
        self.selection.selected_cells = []

    def reset_session(self) -> None:
        self.session = ImageSession.empty()
        self.selection.reset()
        self.drawing.reset()
        self.reset_instance_metadata()
        self.reset_series()

    def discard_filtered_stack(self) -> None:
        self.session.stack_filtered = None
        if self.session.cellpix_orig is not None:
            self.session.cellpix = self.session.cellpix_orig.copy()
            self.session.outpix = self.session.outpix_orig.copy()
            self.session.cellpix_orig = None
            self.session.cellpix_resize = None
            self.session.outpix_orig = None
            self.session.outpix_resize = None

    def apply_masks(
        self,
        masks: np.ndarray,
        outlines: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        colormap: np.ndarray | None = None,
    ) -> int:
        from . import mask_ops

        masks = mask_ops.renumber_masks(masks)
        masks = mask_ops.normalize_mask_dtype(masks)
        session = self.session
        if session.restore and "upsample" in session.restore:
            import cv2

            session.cellpix_resize = masks.copy()
            session.cellpix = session.cellpix_resize.copy()
            session.cellpix_orig = cv2.resize(
                masks.squeeze(),
                (session.lx0, session.ly0),
                interpolation=cv2.INTER_NEAREST,
            )[np.newaxis, :, :]
            session.resize = True
        else:
            session.cellpix = masks
        session.cellpix = mask_ops.ensure_3d_masks(session.cellpix)
        if outlines is None:
            session.outpix = mask_ops.compute_outlines(session.cellpix)
            if session.restore and "upsample" in session.restore:
                session.outpix_orig = mask_ops.compute_outlines(session.cellpix_orig)
                session.outpix_resize = session.outpix.copy()
        else:
            session.outpix = mask_ops.ensure_3d_masks(outlines)
        ncells = int(session.cellpix.max())
        if colors is None and colormap is not None and ncells > 0:
            colors = colormap[:ncells, :3]
        if colors is None:
            colors = np.array([[100, 200, 50]], dtype=np.uint8)
        session.cellcolors = np.concatenate(
            (np.array([[255, 255, 255]], dtype=np.uint8), colors), axis=0
        ).astype(np.uint8)
        session.ismanual = np.zeros(ncells, dtype=bool)
        session.zdraw = list(-1 * np.ones(ncells, dtype=np.int16))
        self.set_instance_classes(ncells)
        self.set_instance_visible(ncells)
        return ncells

    def cell_bounds(
        self, idx: int, z: int | None = None, margin: int = 2
    ) -> tuple[int, int, int, int] | None:
        from . import mask_ops

        if z is None:
            z = self.session.current_z
        return mask_ops.cell_bounds(
            self.session.cellpix[z], idx, self.session.ly, self.session.lx, margin
        )

    def normalize_rect(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> tuple[int, int, int, int] | None:
        from . import mask_ops

        return mask_ops.normalize_rect(
            x0, y0, x1, y1, self.session.ly, self.session.lx
        )

    def cells_in_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        filter_class_id: int | None = None,
    ) -> list[int]:
        from . import mask_ops

        return mask_ops.cells_fully_in_rect(
            self.session.cellpix[self.session.current_z],
            x0,
            y0,
            x1,
            y1,
            self.session.ly,
            self.session.lx,
            filter_class_id,
            self.instances.values,
        )

    def remove_cells(self, indices: list[int]) -> None:
        from . import mask_ops

        session = self.session
        session.cellpix, session.outpix = mask_ops.remove_cells_from_arrays(
            session.cellpix, session.outpix, indices
        )
        for idx in sorted(indices, reverse=True):
            if idx - 1 >= 0:
                self.remove_instance_metadata(idx - 1)
            if idx < len(session.cellcolors):
                session.cellcolors = np.delete(session.cellcolors, idx, axis=0)
            if idx - 1 < len(session.ismanual):
                session.ismanual = np.delete(session.ismanual, idx - 1)

    def build_layer_rgba(
        self,
        filter_class_id: int | None = None,
        stroke_overlay: np.ndarray | None = None,
    ) -> np.ndarray:
        session = self.session
        if session.resize:
            ly, lx = session.lyr, session.lxr
        else:
            ly, lx = session.ly0, session.lx0
        if session.restore and "upsample" in session.restore:
            cellpix = (
                session.cellpix_resize.copy()
                if session.resize
                else session.cellpix_orig.copy()
            )
            outpix = (
                session.outpix_resize.copy()
                if session.resize
                else session.outpix_orig.copy()
            )
        else:
            cellpix = session.cellpix
            outpix = session.outpix

        layerz = np.zeros((ly, lx, 4), dtype=np.uint8)
        plane = cellpix[session.current_z]
        visible_pixels = self.visible_cell_pixels(plane, filter_class_id)
        layerz[..., :3] = session.cellcolors[plane, :]
        layerz[..., 3] = session.opacity * visible_pixels.astype(np.uint8)
        layerz[(outpix[session.current_z] > 0) & visible_pixels] = np.array(
            session.outcolor, dtype=np.uint8
        )
        if stroke_overlay is not None:
            layerz = stroke_overlay
        session.layerz = layerz
        return layerz

    def to_seg_dict(
        self,
        *,
        current_model_path: Any = 0,
        normalize_params: dict[str, Any] | None = None,
        segmentation_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.session
        segmentation_params = segmentation_params or self.segmentation_params or {}
        normalize_params = normalize_params or self.preprocessing_params or {}
        filename = self.series_state.filename or ""
        if session.nz > 1:
            dat: dict[str, Any] = {
                "outlines": session.outpix,
                "colors": session.cellcolors[1:],
                "masks": session.cellpix,
                "filename": filename,
                "flows": session.flows,
                "zdraw": session.zdraw,
                "model_path": current_model_path,
                "flow_threshold": segmentation_params.get("flow_threshold", 0.4),
                "cellprob_threshold": segmentation_params.get("cellprob_threshold", 0.0),
                "normalize_params": normalize_params,
                "restore": session.restore,
                "ratio": session.ratio,
                "diameter": segmentation_params.get("diameter"),
            }
            if session.restore is not None and session.stack_filtered is not None:
                dat["img_restore"] = session.stack_filtered
        else:
            use_resize = session.restore is not None and "upsample" in session.restore
            dat = {
                "outlines": session.outpix_resize.squeeze()
                if use_resize
                else session.outpix.squeeze(),
                "colors": session.cellcolors[1:],
                "masks": session.cellpix_resize.squeeze()
                if use_resize
                else session.cellpix.squeeze(),
                "filename": filename,
                "flows": session.flows,
                "ismanual": session.ismanual,
                "zdraw": session.zdraw,
                "model_path": current_model_path,
                "flow_threshold": segmentation_params.get("flow_threshold", 0.4),
                "cellprob_threshold": segmentation_params.get("cellprob_threshold", 0.0),
                "normalize_params": normalize_params,
                "restore": session.restore,
                "ratio": session.ratio,
                "diameter": segmentation_params.get("diameter"),
            }
        if len(self.instances.values) > 0:
            dat["instance_classes"] = self.instances.values
        return dat

    def to_session_data(
        self,
        *,
        source_image: str,
        model: str = "cpsam",
        segmentation_params: dict[str, Any] | None = None,
        recompute_masks: bool = False,
    ) -> "SessionData":
        from cellpose.gui.session_format.models import SessionData, SegmentationMetadata

        session = self.session
        segmentation_params = segmentation_params or self.segmentation_params or {}
        use_resize = session.restore is not None and "upsample" in session.restore
        if session.nz > 1:
            masks = np.asarray(session.cellpix)
        else:
            masks = np.asarray(
                session.cellpix_resize.squeeze()
                if use_resize
                else session.cellpix.squeeze()
            )

        flows_list = None
        if session.flows:
            try:
                flows_list = [np.asarray(flow) for flow in session.flows]
            except TypeError:
                flows_list = None

        colors = None
        if len(session.cellcolors) > 1:
            colors = np.asarray(session.cellcolors[1:], dtype=np.uint8)

        instance_classes = None
        if len(self.instances.values) > 0:
            instance_classes = np.asarray(self.instances.values, dtype=np.int32)

        ismanual = None
        if session.nz == 1 and len(session.ismanual) > 0:
            ismanual = np.asarray(session.ismanual, dtype=bool)

        seg = SegmentationMetadata(
            flow_threshold=float(segmentation_params.get("flow_threshold", 0.4)),
            cellprob_threshold=float(segmentation_params.get("cellprob_threshold", 0.0)),
            diameter=segmentation_params.get("diameter"),
            niter=int(segmentation_params.get("niter", 200)),
            min_size=int(segmentation_params.get("min_size", 15)),
        )

        return SessionData(
            source_image=source_image,
            masks=masks,
            flows=flows_list,
            colors=colors,
            instance_classes=instance_classes,
            ismanual=ismanual,
            model=model,
            recompute_masks=bool(recompute_masks),
            segmentation=seg,
        )
