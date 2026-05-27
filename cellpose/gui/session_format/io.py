"""Read and write `.cellpose` zip session archives."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .models import SESSION_FORMAT_VERSION, SessionData, SegmentationMetadata

_ARRAYS_DIR = "arrays"
_MANIFEST_NAME = "manifest.json"

_DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
}


def default_session_path(image_path: str) -> str:
    base = os.path.splitext(image_path)[0]
    return base + "_seg.cellpose"


def _dtype_from_name(name: str) -> np.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {name}")
    return np.dtype(_DTYPE_MAP[name])


def _dtype_to_name(dtype: np.dtype) -> str:
    for name, mapped in _DTYPE_MAP.items():
        if np.dtype(mapped) == dtype:
            return name
    raise ValueError(f"Unsupported dtype for session export: {dtype}")


def _write_array_entry(
    zf: zipfile.ZipFile,
    name: str,
    arr: np.ndarray,
) -> dict[str, Any]:
    arr = np.ascontiguousarray(arr)
    rel_path = f"{_ARRAYS_DIR}/{name}.{_dtype_to_name(arr.dtype)}"
    zf.writestr(rel_path, arr.tobytes(), compress_type=zipfile.ZIP_DEFLATED)
    return {
        "file": rel_path,
        "dtype": _dtype_to_name(arr.dtype),
        "shape": list(arr.shape),
    }


def _read_array_entry(zf: zipfile.ZipFile, entry: dict[str, Any]) -> np.ndarray:
    raw = zf.read(entry["file"])
    dtype = _dtype_from_name(entry["dtype"])
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(tuple(entry["shape"]))


def _default_colors(ncells: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(50, 255, size=(max(ncells, 1), 3), dtype=np.uint8)


def write_session(path: str | os.PathLike[str], session: SessionData) -> str:
    """Write session data to a `.cellpose` zip archive."""
    path = Path(path)
    if path.suffix != ".cellpose":
        path = path.with_suffix(".cellpose")

    arrays: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        arrays["masks"] = _write_array_entry(zf, "masks", session.masks.squeeze())

        if session.flows:
            for index, flow in enumerate(session.flows):
                arrays[f"flows_{index}"] = _write_array_entry(
                    zf, f"flows_{index}", flow
                )

        colors = session.colors
        if colors is None and session.ncells > 0:
            colors = _default_colors(session.ncells)
        if colors is not None:
            arrays["colors"] = _write_array_entry(zf, "colors", colors)

        if session.instance_classes is not None:
            arrays["instance_classes"] = _write_array_entry(
                zf, "instance_classes", session.instance_classes.astype(np.int32)
            )

        if session.ismanual is not None:
            arrays["ismanual"] = _write_array_entry(
                zf, "ismanual", session.ismanual.astype(np.bool_)
            )

        manifest = {
            "version": SESSION_FORMAT_VERSION,
            "source_image": session.source_image,
            "segmentation": session.segmentation.to_dict(),
            "model": session.model,
            "recompute_masks": session.recompute_masks,
            "arrays": arrays,
        }
        zf.writestr(
            _MANIFEST_NAME,
            json.dumps(manifest, indent=2),
            compress_type=zipfile.ZIP_DEFLATED,
        )

    return str(path)


def read_session(path: str | os.PathLike[str]) -> SessionData:
    """Read session data from a `.cellpose` zip archive."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read(_MANIFEST_NAME).decode("utf-8"))
        if manifest.get("version") != SESSION_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported session format version: {manifest.get('version')}"
            )

        arrays = manifest.get("arrays", {})
        masks = _read_array_entry(zf, arrays["masks"])

        flows: list[np.ndarray] | None = None
        flow_keys = sorted(
            (key for key in arrays if key.startswith("flows_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if flow_keys:
            flows = [_read_array_entry(zf, arrays[key]) for key in flow_keys]

        colors = (
            _read_array_entry(zf, arrays["colors"]) if "colors" in arrays else None
        )
        instance_classes = (
            _read_array_entry(zf, arrays["instance_classes"])
            if "instance_classes" in arrays
            else None
        )
        ismanual = (
            _read_array_entry(zf, arrays["ismanual"]) if "ismanual" in arrays else None
        )

    source_image = manifest["source_image"]
    if not os.path.isabs(source_image):
        source_image = str((path.parent / source_image).resolve())

    return SessionData(
        source_image=source_image,
        masks=masks,
        flows=flows,
        colors=colors,
        instance_classes=instance_classes,
        ismanual=ismanual,
        model=str(manifest.get("model", "cpsam")),
        recompute_masks=bool(manifest.get("recompute_masks", False)),
        segmentation=SegmentationMetadata.from_dict(
            manifest.get("segmentation", {})
        ),
    )
