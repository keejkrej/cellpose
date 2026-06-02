"""Array serialization helpers for sidecar HTTP payloads."""

from __future__ import annotations

import base64
import zlib
from typing import Any

import numpy as np


def encode_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(arr)
    compressed = zlib.compress(arr.tobytes(), level=3)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(compressed).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(payload["data_b64"]))
    arr = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    return arr.reshape(tuple(payload["shape"]))


def encode_optional(arr: np.ndarray | None) -> dict[str, Any] | None:
    if arr is None:
        return None
    return encode_array(arr)
