"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os
import gc
import numpy as np
import cv2
import fastremap
import shutil

from ..io import imread, imread_2D, imread_3D
from ..models import normalize_default, MODEL_DIR, MODEL_LIST_PATH, get_user_models
from ..utils import masks_to_outlines

try:
    from PySide6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QVBoxLayout,
    )
    GUI = True
except:
    GUI = False

try:
    MATPLOTLIB = True
except:
    MATPLOTLIB = False


from . import series


def _init_model_list(parent):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parent.model_list_path = MODEL_LIST_PATH
    parent.model_strings = get_user_models()


def _get_custom_model_dir():
    custom_dir = MODEL_DIR.joinpath("custom")
    custom_dir.mkdir(parents=True, exist_ok=True)
    return custom_dir


def _write_model_list(model_strings):
    with open(MODEL_LIST_PATH, "w") as textfile:
        for model_string in model_strings:
            textfile.write(model_string + "\n")


def _add_model(parent, filename=None, load_model=True):
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Add model to GUI")
        filename = name[0]
    if filename == "":
        return
    fname = os.path.split(filename)[-1]
    target = _get_custom_model_dir().joinpath(fname)
    try:
        shutil.copyfile(filename, os.fspath(target))
    except shutil.SameFileError:
        pass
    parent.model_strings.append(fname)
    parent.ModelChooseC.addItems([fname])

    for ind, model_string in enumerate(parent.model_strings[:-1]):
        if model_string == fname:
            _remove_model(parent, ind=ind + 1, verbose=False)
    _write_model_list(parent.model_strings)

    parent.ModelChooseC.setCurrentIndex(len(parent.model_strings))
    if load_model:
        parent.model_choose(custom=True)


def _remove_model(parent, ind=None, verbose=True):
    if ind is None:
        ind = parent.ModelChooseC.currentIndex()
    if ind > 0:
        ind -= 1
        modelstr = parent.model_strings[ind]
        parent.ModelChooseC.removeItem(ind + 1)
        del parent.model_strings[ind]
        _write_model_list(parent.model_strings)
        model_path = _get_custom_model_dir().joinpath(modelstr)
        if model_path.exists():
            os.remove(os.fspath(model_path))
        if len(parent.model_strings) > 0:
            parent.ModelChooseC.setCurrentIndex(len(parent.model_strings))
        else:
            parent.ModelChooseC.setCurrentIndex(0)
    else:
        print("ERROR: no model selected to delete")


def _get_train_set(image_names):
    """Get training data and labels for images in current folder image_names."""
    from cellpose.gui.session_format import read_session
    from cellpose.io import imread

    train_data, train_labels, train_files = [], [], []
    for image_name_full in image_names:
        image_name = os.path.splitext(image_name_full)[0]
        session_path = image_name + "_seg.cellpose"
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


def _clear_series_state(parent):
    parent.presenter.reset_series()


def _set_series_state(parent, dataset=None, item_index=None):
    if dataset is None or item_index is None:
        parent.presenter.reset_series()
        return
    parent.presenter.set_series(dataset=dataset, record_index=item_index)


def _get_output_filename(parent):
    return parent.presenter.output_filename(str(parent.model.filename))


def _prompt_series_templates(parent, folder):
    suggestion = series.suggest_series_templates(folder)
    subfolder_text = getattr(parent, "last_series_subfolder_template", "")
    filename_text = getattr(parent, "last_series_filename_template", "")
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


def _load_image_series(parent):
    folder = QFileDialog.getExistingDirectory(parent, "Load image folder")
    if folder == "":
        return

    templates = _prompt_series_templates(parent, folder)
    if templates is None:
        return

    subfolder_template, filename_template = templates
    if filename_template == "":
        return

    parent.last_series_subfolder_template = subfolder_template
    parent.last_series_filename_template = filename_template

    try:
        dataset = series.build_series_dataset(
            folder,
            subfolder_template=subfolder_template,
            filename_template=filename_template,
        )
        _load_series_item(parent, dataset, 0, load_seg=True, load_3D=parent.load_3D)
    except Exception as e:
        print(f"ERROR: {e}")
        QMessageBox.warning(parent, "Load folder with pattern", str(e))


def _load_series_item(parent, dataset, item_index, load_seg=True, load_3D=False):
    output_filename = series.get_output_filename(dataset, item_index)
    seg_filename = os.path.splitext(output_filename)[0] + "_seg.cellpose"
    if load_seg and os.path.isfile(seg_filename):
        _load_seg(parent, filename=seg_filename, load_3D=load_3D)
        if parent.model.series_state.dataset is None:
            _set_series_state(parent, dataset=dataset, item_index=item_index)
        title = parent.model.series_state.display_filename or parent.model.filename
        parent.set_window_title(str(title))
        return

    _load_image(parent, filename=output_filename, load_seg=False, load_3D=load_3D)
    _set_series_state(parent, dataset=dataset, item_index=item_index)
    title = parent.model.series_state.display_filename or parent.model.filename
    parent.set_window_title(str(title))


def _load_image(parent, filename=None, load_seg=True, load_3D=False):
    """Load image with filename; if None, open QFileDialog."""
    if parent.load_3D:
        load_3D = True

    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Load image")
        filename = name[0]
        if filename == "":
            return
    _clear_series_state(parent)
    manual_file = os.path.splitext(filename)[0] + "_seg.cellpose"
    if load_seg and os.path.isfile(manual_file):
        image = imread_2D(filename) if not load_3D else imread_3D(filename)
        _load_seg(parent, manual_file, image=image, image_file=filename, load_3D=load_3D)
        return
    try:
        print(f"GUI_INFO: loading image: {filename}")
        if not load_3D:
            image = imread_2D(filename)
        else:
            image = imread_3D(filename)
    except Exception as e:
        print("ERROR: images not compatible")
        print(f"ERROR: {e}")
        return

    parent.presenter.reset_session()
    parent.presenter.on_initialize_images(image, load_3d=load_3D)
    parent.presenter.on_image_loaded(filename)


def _initialize_images(parent, image, load_3D=False):
    parent.presenter.on_initialize_images(image, load_3d=load_3D)


def _apply_cellpose_segmentation_widgets(parent, segmentation) -> None:
    if not hasattr(parent, "seg_param_root"):
        return

    parent.seg_param_root.param("flow_threshold").setValue(
        float(segmentation.flow_threshold)
    )
    parent.seg_param_root.param("cellprob_threshold").setValue(
        float(segmentation.cellprob_threshold)
    )
    parent.seg_param_root.param("niter").setValue(int(segmentation.niter))
    if segmentation.diameter is not None:
        parent.seg_param_root.param("diameter").setValue(float(segmentation.diameter))
    if hasattr(parent, "min_size"):
        parent.min_size = int(segmentation.min_size)


def _load_seg(parent, filename, image=None, image_file=None, load_3D=False):
    """Load a *_seg.cellpose session archive."""
    from cellpose.gui.session_format import read_session

    if not filename:
        return

    try:
        session_data = read_session(filename)
    except Exception as exc:
        parent.model.session.loaded = False
        print(f"ERROR: not a valid .cellpose session: {exc}")
        return

    if image is None:
        image_path = session_data.source_image
        if not os.path.isfile(image_path):
            parent.model.session.loaded = False
            print(f"ERROR: cannot find image file: {image_path}")
            return
        try:
            print(f"GUI_INFO: loading image: {image_path}")
            image = imread_2D(image_path) if not load_3D else imread_3D(image_path)
        except Exception as exc:
            parent.model.session.loaded = False
            print(f"ERROR: cannot load image: {exc}")
            return
        parent.model.filename = image_path
    else:
        parent.model.filename = image_file or session_data.source_image

    parent.presenter.reset_session()
    _clear_series_state(parent)
    parent.model.series_state.output_filename = None
    parent.model.series_state.display_filename = str(parent.model.filename)
    parent.model.session.restore = None
    parent.model.session.ratio = 1.0

    parent.presenter.on_initialize_images(image, load_3d=load_3D)

    masks = np.asarray(session_data.masks)
    if masks.min() == -1:
        masks = masks + 1
    colors = session_data.colors
    if colors is None and int(masks.max()) > 0:
        colors = parent.colormap[: int(masks.max()), :3]

    parent.presenter.apply_masks_from_io(masks, colors=colors)
    parent.apply_segmentation_metadata_widgets(session_data.segmentation)

    ismanual = np.zeros(parent.presenter.ncells(), bool)
    if (
        session_data.ismanual is not None
        and len(session_data.ismanual) == parent.presenter.ncells()
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

    parent.presenter.on_load_seg_session(
        session_data,
        ismanual=ismanual,
        flows=flows,
        recompute_masks=recompute_masks,
        instance_classes=session_data.instance_classes,
    )
    gc.collect()


def _masks_to_gui(parent, masks, outlines=None, colors=None):
    parent.presenter.apply_masks_from_io(masks, outlines=outlines, colors=colors)


def _save_sets(parent):
    parent.presenter.save_sets()
