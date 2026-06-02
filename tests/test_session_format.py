"""Tests for pickled `_seg.npy` session read/write."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cellpose.app_core import (
    SessionData,
    default_session_path,
    read_session,
    session_to_pickle_dict,
    write_session,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "seg_npy"


@pytest.fixture()
def seg_fixtures():
    if not FIXTURES.is_dir():
        pytest.skip("Run scripts/generate_seg_fixtures.py to create fixtures")
    return FIXTURES


def test_roundtrip(tmp_path):
    masks = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)
    flows = [
        np.ones((1, 2, 3), dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.uint8),
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

    out_path = tmp_path / "image_seg.npy"
    write_session(out_path, session)
    loaded = read_session(out_path)

    assert loaded.source_image == str(tmp_path / "image.tif")
    assert np.array_equal(loaded.masks, masks)
    assert loaded.ncells == 2
    assert loaded.model == "cpsam"
    assert len(loaded.flows) == 2
    assert np.array_equal(loaded.colors, colors)


def test_default_session_path():
    assert default_session_path("/data/cells.tif") == "/data/cells_seg.npy"


def test_resolve_ignores_absolute_stored_path(tmp_path):
    masks = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"\xff\xd8\xff")
    dat = {
        "outlines": masks.copy(),
        "masks": masks,
        "filename": "/home/jack/data/rcc_jb4/Pos27/img.jpg",
        "flows": [],
    }
    seg_path = tmp_path / "img_seg.npy"
    np.save(seg_path, dat)

    loaded = read_session(seg_path)
    assert loaded.source_image == str(image_path.resolve())
    assert "/home/jack" not in loaded.source_image


def test_resolve_stem_extension_fallback(tmp_path):
    masks = np.array([[0, 1]], dtype=np.uint16)
    image_path = tmp_path / "img_000000000_Durchlicht_000.jpg"
    image_path.write_bytes(b"\xff\xd8\xff")
    dat = {
        "outlines": masks.copy(),
        "masks": masks,
        "filename": "/other/machine/wrong_name.jpg",
        "flows": [],
    }
    seg_path = tmp_path / "img_000000000_Durchlicht_000_seg.npy"
    np.save(seg_path, dat)

    loaded = read_session(seg_path)
    assert loaded.source_image == str(image_path.resolve())
    assert os.path.isfile(loaded.source_image)


def test_resolve_empty_filename_uses_stem(tmp_path):
    masks = np.array([[0, 1]], dtype=np.uint16)
    image_path = tmp_path / "cells.tif"
    image_path.write_bytes(b"II*\x00\x08\x00\x00\x00")
    dat = {
        "outlines": masks.copy(),
        "masks": masks,
        "filename": "",
        "flows": [],
    }
    seg_path = tmp_path / "cells_seg.npy"
    np.save(seg_path, dat)

    loaded = read_session(seg_path)
    assert loaded.source_image == str(image_path.resolve())


def test_read_legacy_pickle_dict(tmp_path):
    masks = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)
    dat = {
        "outlines": masks.copy(),
        "colors": np.array([[100, 150, 200], [50, 60, 70]], dtype=np.uint8),
        "masks": masks,
        "filename": str(tmp_path / "image.tif"),
        "flows": [],
        "ismanual": np.zeros(2, bool),
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "diameter": None,
    }
    path = tmp_path / "legacy_seg.npy"
    np.save(path, dat)

    loaded = read_session(path)
    assert loaded.source_image == str(tmp_path / "image.tif")
    assert np.array_equal(loaded.masks, masks)
    assert loaded.ncells == 2


def test_write_matches_original_format(tmp_path):
    session = SessionData(
        source_image=str(tmp_path / "image.tif"),
        masks=np.array([[0, 1]], dtype=np.uint16),
    )
    path = tmp_path / "image_seg.npy"
    write_session(path, session)
    loaded = np.load(path, allow_pickle=True).item()
    assert isinstance(loaded, dict)
    assert "outlines" in loaded
    assert "masks" in loaded
    assert np.array_equal(loaded["masks"], session_to_pickle_dict(session)["masks"])


@pytest.mark.parametrize(
    ("fixture_name", "expected_ncells", "expected_source"),
    [
        ("minimal_seg.npy", 2, "image.tif"),
        ("legacy_gui_seg.npy", 2, "legacy_image.tif"),
        ("with_instance_classes_seg.npy", 2, "image.tif"),
    ],
)
def test_read_committed_fixtures(
    seg_fixtures, fixture_name, expected_ncells, expected_source
):
    path = seg_fixtures / fixture_name
    assert path.is_file(), f"missing fixture: {path}"
    loaded = read_session(path)
    assert loaded.ncells == expected_ncells
    assert loaded.source_image.endswith(expected_source)
    assert loaded.masks.squeeze().max() == expected_ncells


def test_minimal_fixture_has_flows(seg_fixtures):
    loaded = read_session(seg_fixtures / "minimal_seg.npy")
    assert loaded.flows is not None
    assert len(loaded.flows) == 2


def test_instance_classes_fixture(seg_fixtures):
    loaded = read_session(seg_fixtures / "with_instance_classes_seg.npy")
    assert loaded.instance_classes is not None
    assert np.array_equal(loaded.instance_classes, np.array([1, 2], dtype=np.int32))
