"""
File and session helpers for the Cellpose GUI.

Orchestration lives on MainPresenter; this module keeps pure I/O utilities.
"""

import os
import shutil

import fastremap
import numpy as np

from ...io import imread_2D
from ...models import MODEL_DIR, MODEL_LIST_PATH, get_user_models, normalize_default
from . import series

try:
    from .qt import QtCore  # noqa: F401 — configure QT_API before qtpy widgets
    from qtpy.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QLabel,
        QLineEdit,
        QVBoxLayout,
    )
except ImportError:
    pass


def _get_custom_model_dir():
    custom_dir = MODEL_DIR.joinpath("custom")
    custom_dir.mkdir(parents=True, exist_ok=True)
    return custom_dir


def _write_model_list(model_strings):
    with open(MODEL_LIST_PATH, "w") as textfile:
        for model_string in model_strings:
            textfile.write(model_string + "\n")


def _get_train_set(image_names):
    """Get training data and labels for images in current folder image_names."""
    from cellpose.gui.core.session import read_session
    from cellpose.io import imread

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


def _prompt_series_templates(
    parent, folder, subfolder_text="", filename_text=""
):
    suggestion = series.suggest_series_templates(folder)
    if not subfolder_text:
        subfolder_text = suggestion["subfolder_template"]
    if not filename_text:
        filename_text = suggestion["filename_template"]

    dialog = QDialog(parent)
    dialog.setWindowTitle("Load folder with pattern")
    dialog.setMinimumWidth(500)

    layout = QVBoxLayout(dialog)
    info_label = QLabel(
        "Use placeholders {t}, {p}, {c}, {z}. Subfolder matching is case-insensitive."
    )
    info_label.setWordWrap(True)
    layout.addWidget(info_label)

    form = QFormLayout()
    subfolder_edit = QLineEdit(subfolder_text)
    filename_edit = QLineEdit(filename_text)
    form.addRow("Subfolder template:", subfolder_edit)
    form.addRow("Filename template:", filename_edit)
    layout.addLayout(form)

    button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)

    if dialog.exec() != QDialog.Accepted:
        return None

    return subfolder_edit.text().strip(), filename_edit.text().strip()


def get_user_model_strings():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return get_user_models()
