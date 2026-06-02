"""Generate committed _seg.npy fixtures for session I/O tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cellpose.gui.core.session import SessionData, session_to_pickle_dict, write_session

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "seg_npy"


def _minimal_session() -> SessionData:
    masks = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)
    colors = np.array([[100, 150, 200], [50, 60, 70]], dtype=np.uint8)
    flows = [
        np.ones((1, 2, 3), dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.uint8),
    ]
    return SessionData(
        source_image="image.tif",
        masks=masks,
        flows=flows,
        colors=colors,
        model="cpsam",
        recompute_masks=True,
    )


def _legacy_gui_dict() -> dict:
    masks = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)
    return {
        "outlines": masks.copy(),
        "colors": np.array([[100, 150, 200], [50, 60, 70]], dtype=np.uint8),
        "masks": masks,
        "filename": "legacy_image.tif",
        "flows": [],
        "ismanual": np.zeros(2, bool),
        "manual_changes": [],
        "zdraw": [None, None],
        "model_path": 0,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "normalize_params": {"lowhigh": None, "percentile": None, "normalize": True},
        "restore": None,
        "ratio": 1.0,
        "diameter": 30.0,
    }


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    write_session(FIXTURES / "minimal_seg.npy", _minimal_session())

    legacy = _legacy_gui_dict()
    np.save(FIXTURES / "legacy_gui_seg.npy", legacy)

    session = _minimal_session()
    session.instance_classes = np.array([1, 2], dtype=np.int32)
    write_session(
        FIXTURES / "with_instance_classes_seg.npy",
        session_to_pickle_dict(session) | {"instance_classes": session.instance_classes},
    )

    print(f"Wrote fixtures to {FIXTURES}")


if __name__ == "__main__":
    main()
