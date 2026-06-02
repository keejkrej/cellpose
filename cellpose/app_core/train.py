"""Training data loading from labeled `_seg.npy` sessions."""

from __future__ import annotations

import os

import fastremap
import numpy as np

from cellpose.io import imread
from cellpose.models import normalize_default

from .session import read_session


def get_train_set(image_names):
    """Return training images, masks, and file paths for a folder listing."""
    train_data, train_labels, train_files = [], [], []
    for image_name_full in image_names:
        image_name = os.path.splitext(image_name_full)[0]
        session_path = image_name + "_seg.npy"
        if not os.path.isfile(session_path):
            continue
        try:
            session = read_session(session_path)
        except Exception as exc:
            print(f"GUI_INFO: failed to read {session_path}: {exc}")
            continue
        masks = np.asarray(session.masks).squeeze()
        if masks.ndim != 2:
            print(f"GUI_INFO: {session_path} masks.ndim!=2")
            continue
        fastremap.renumber(masks, in_place=True)
        if os.path.isfile(session.source_image):
            data = imread(session.source_image)
        elif os.path.isfile(image_name_full):
            data = imread(image_name_full)
        else:
            print(f"GUI_INFO: no image found for {session_path}")
            continue
        train_files.append(image_name_full)
        train_data.append(data)
        train_labels.append(masks)
    return train_data, train_labels, train_files, None, normalize_default
