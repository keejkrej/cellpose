"""Tests for `.cellpose` session archive read/write."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from cellpose.gui.session_format import SessionData, read_session, write_session
from cellpose.gui.session_format.io import default_session_path


def test_roundtrip(tmp_path):
    masks = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)
    flows = [
        np.ones((1, 2, 3), dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.uint8),
        np.zeros((1, 2, 3, 2), dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.float32),
    ]
    colors = np.array([[100, 150, 200], [50, 60, 70]], dtype=np.uint8)
    session = SessionData(
        source_image=str(tmp_path / "image.tif"),
        masks=masks,
        flows=flows,
        colors=colors,
        model="cpsam",
        recompute_masks=True,
    )

    out_path = tmp_path / "image_seg.cellpose"
    write_session(out_path, session)
    loaded = read_session(out_path)

    assert loaded.source_image == str(tmp_path / "image.tif")
    assert np.array_equal(loaded.masks, masks)
    assert loaded.ncells == 2
    assert loaded.recompute_masks is True
    assert loaded.model == "cpsam"
    assert len(loaded.flows) == 4
    assert np.array_equal(loaded.colors, colors)


def test_default_session_path():
    assert default_session_path("/data/cells.tif") == "/data/cells_seg.cellpose"


def test_manifest_version(tmp_path):
    masks = np.zeros((4, 4), dtype=np.uint16)
    path = tmp_path / "test_seg.cellpose"
    write_session(
        path,
        SessionData(source_image="img.png", masks=masks),
    )
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["version"] == 1
    assert "masks" in manifest["arrays"]


def test_unsupported_version(tmp_path):
    path = tmp_path / "bad.cellpose"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"version": 99}))
    with pytest.raises(ValueError, match="Unsupported session format version"):
        read_session(path)
