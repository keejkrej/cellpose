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
    if hasattr(parent, "presenter"):
        parent.presenter.reset_series()
    else:
        parent.series_dataset = None
        parent.series_index = None
        parent.output_filename = None
        parent.display_filename = None
    if hasattr(parent, "set_series_navigation_state"):
        parent.set_series_navigation_state()


def _set_series_state(parent, dataset=None, item_index=None):
    if dataset is None or item_index is None:
        if hasattr(parent, "presenter"):
            parent.presenter.reset_series()
        else:
            parent.series_dataset = None
            parent.series_index = None
            parent.output_filename = None
        if hasattr(parent, "set_series_navigation_state"):
            parent.set_series_navigation_state()
        return

    if hasattr(parent, "presenter"):
        parent.presenter.set_series(dataset=dataset, record_index=item_index)
    else:
        item = dataset["records"][item_index]
        parent.series_dataset = dataset
        parent.series_index = item_index
        parent.output_filename = series.get_output_filename(dataset, item_index)
        parent.display_filename = item["label"]
        parent.filename = item["path"]
    if hasattr(parent, "set_series_navigation_state"):
        parent.set_series_navigation_state(dataset, item_index)


def _get_output_filename(parent):
    if hasattr(parent, "presenter"):
        return parent.presenter.output_filename(parent.filename)
    return parent.output_filename if getattr(parent, "output_filename", None) else parent.filename


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
        if getattr(parent, "series_dataset", None) is None:
            _set_series_state(parent, dataset=dataset, item_index=item_index)
        parent.setWindowTitle(parent.display_filename or parent.filename)
        return

    _load_image(parent, filename=output_filename, load_seg=False, load_3D=load_3D)
    _set_series_state(parent, dataset=dataset, item_index=item_index)
    parent.setWindowTitle(parent.display_filename or parent.filename)


def _load_image(parent, filename=None, load_seg=True, load_3D=False):
    """ load image with filename; if None, open QFileDialog
    if image is grey change view to default to grey scale 
    """

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
        _load_seg(parent, manual_file, image=image, image_file=filename,
                  load_3D=load_3D)
        return
    try:
        print(f"GUI_INFO: loading image: {filename}")
        if not load_3D:
            image = imread_2D(filename)
        else:
            image = imread_3D(filename)
        parent.loaded = True
    except Exception as e:
        print("ERROR: images not compatible")
        print(f"ERROR: {e}")
        return

    if parent.loaded:
        parent.reset()
        parent.filename = filename
        parent.display_filename = filename
        parent.output_filename = None
        filename = os.path.split(parent.filename)[-1]
        _initialize_images(parent, image, load_3D=load_3D)
        parent.loaded = True
        parent.enable_buttons()


def _initialize_images(parent, image, load_3D=False):
    """ format image for GUI

    assumes image is Z x W x H x C

    """
    load_3D = parent.load_3D if load_3D is False else load_3D

    if hasattr(parent, "model"):
        parent.model.load_image_stack(image, load_3d=load_3D)
        parent.imask = 0
        if load_3D:
            parent.scroll.setMaximum(parent.model.session.nz - 1)
            parent.scroll.setValue(parent.model.session.current_z)
            parent.zpos.setText(str(parent.model.session.current_z))
        parent.clear_all()
        parent.sliders[0].setValue([0, 255])
        if not hasattr(parent, "stack_filtered") and parent.restore:
            print("GUI_INFO: no 'img_restore' found, applying current settings")
            parent.compute_restore()
        parent.compute_scale()
        del image
        gc.collect()
        return

    parent.stack = image
    print(f"GUI_INFO: image shape: {image.shape}")
    if load_3D:
        parent.NZ = len(parent.stack)
        parent.scroll.setMaximum(parent.NZ - 1)
    else:
        parent.NZ = 1
        parent.stack = parent.stack[np.newaxis, ...]

    img_min = image.min()
    img_max = image.max()
    parent.stack = parent.stack.astype(np.float32)
    parent.stack -= img_min
    if img_max > img_min + 1e-3:
        parent.stack /= (img_max - img_min)
    parent.stack *= 255

    if load_3D:
        print("GUI_INFO: converted to float and normalized values to 0.0->255.0")

    del image
    gc.collect()

    parent.imask = 0
    parent.Ly, parent.Lx = parent.stack.shape[-3:-1]
    parent.Ly0, parent.Lx0 = parent.stack.shape[-3:-1]
    parent.layerz = 255 * np.ones((parent.Ly, parent.Lx, 4), "uint8")
    if hasattr(parent, "stack_filtered"):
        parent.Lyr, parent.Lxr = parent.stack_filtered.shape[-3:-1]
    elif parent.restore and "upsample" in parent.restore:
        parent.Lyr, parent.Lxr = int(parent.Ly * parent.ratio), int(parent.Lx *
                                                                    parent.ratio)
    else:
        parent.Lyr, parent.Lxr = parent.Ly, parent.Lx
    parent.clear_all()
    parent.saturation = [[[0, 255] for n in range(parent.NZ)]]
    parent.sliders[0].setValue([0, 255])

    if not hasattr(parent, "stack_filtered") and parent.restore:
        print("GUI_INFO: no 'img_restore' found, applying current settings")
        parent.compute_restore()

    parent.compute_scale()
    parent.track_changes = []

    if load_3D:
        parent.currentZ = int(np.floor(parent.NZ / 2))
        parent.scroll.setValue(parent.currentZ)
        parent.zpos.setText(str(parent.currentZ))
    else:
        parent.currentZ = 0


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


def _load_seg(parent, filename, image=None, image_file=None, load_3D=False):
    """Load a *_seg.cellpose session archive."""
    from cellpose.gui.session_format import read_session

    if not filename:
        return

    try:
        session_data = read_session(filename)
    except Exception as exc:
        parent.loaded = False
        print(f"ERROR: not a valid .cellpose session: {exc}")
        return

    if image is None:
        image_path = session_data.source_image
        if not os.path.isfile(image_path):
            parent.loaded = False
            print(f"ERROR: cannot find image file: {image_path}")
            return
        try:
            print(f"GUI_INFO: loading image: {image_path}")
            image = imread_2D(image_path) if not load_3D else imread_3D(image_path)
        except Exception as exc:
            parent.loaded = False
            print(f"ERROR: cannot load image: {exc}")
            return
        parent.filename = image_path
    else:
        parent.filename = image_file or session_data.source_image

    parent.reset()
    _clear_series_state(parent)
    parent.output_filename = None
    parent.display_filename = parent.filename
    parent.restore = None
    parent.ratio = 1.0

    _initialize_images(parent, image, load_3D=load_3D)

    masks = np.asarray(session_data.masks)
    if masks.min() == -1:
        masks = masks + 1
    ncells = int(masks.max())
    colors = session_data.colors
    if colors is None and ncells > 0:
        colors = parent.colormap[:ncells, :3]

    _masks_to_gui(parent, masks, colors=colors)
    _apply_cellpose_segmentation_widgets(parent, session_data.segmentation)

    parent.ismanual = np.zeros(parent.ncells(), bool)
    if (
        session_data.ismanual is not None
        and len(session_data.ismanual) == parent.ncells()
    ):
        parent.ismanual = session_data.ismanual

    if hasattr(parent, "set_instance_classes"):
        parent.set_instance_classes(session_data.instance_classes)

    if session_data.flows:
        parent.flows = session_data.flows
        try:
            if parent.flows[0].shape[-3] != masks.shape[-2]:
                Ly, Lx = masks.shape[-2:]
                for i in range(len(parent.flows)):
                    parent.flows[i] = cv2.resize(
                        parent.flows[i].squeeze(),
                        (Lx, Ly),
                        interpolation=cv2.INTER_NEAREST,
                    )[np.newaxis, ...]
        except Exception:
            pass
        parent.recompute_masks = session_data.recompute_masks
    else:
        parent.recompute_masks = False

    parent.loaded = True
    parent.enable_buttons()
    parent.update_layer()
    gc.collect()


def _masks_to_gui(parent, masks, outlines=None, colors=None):
    """ masks loaded into GUI """
    if hasattr(parent, "presenter"):
        parent.presenter.apply_masks_from_io(masks, outlines=outlines, colors=colors)
        return
    # get unique values
    shape = masks.shape
    if len(fastremap.unique(masks)) != masks.max() + 1:
        print("GUI_INFO: renumbering masks")
        fastremap.renumber(masks, in_place=True)
        outlines = None
        masks = masks.reshape(shape)
    if masks.ndim == 2:
        outlines = None
    masks = masks.astype(np.uint16) if masks.max() < 2**16 - 1 else masks.astype(
        np.uint32)
    if parent.restore and "upsample" in parent.restore:
        parent.cellpix_resize = masks.copy()
        parent.cellpix = parent.cellpix_resize.copy()
        parent.cellpix_orig = cv2.resize(
            masks.squeeze(), (parent.Lx0, parent.Ly0),
            interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
        parent.resize = True
    else:
        parent.cellpix = masks
    if parent.cellpix.ndim == 2:
        parent.cellpix = parent.cellpix[np.newaxis, :, :]
        if parent.restore and "upsample" in parent.restore:
            if parent.cellpix_resize.ndim == 2:
                parent.cellpix_resize = parent.cellpix_resize[np.newaxis, :, :]
            if parent.cellpix_orig.ndim == 2:
                parent.cellpix_orig = parent.cellpix_orig[np.newaxis, :, :]

    print(f"GUI_INFO: {masks.max()} masks found")

    # get outlines
    if outlines is None:  # parent.outlinesOn
        parent.outpix = np.zeros_like(parent.cellpix)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_orig = np.zeros_like(parent.cellpix_orig)
        for z in range(parent.NZ):
            outlines = masks_to_outlines(parent.cellpix[z])
            parent.outpix[z] = outlines * parent.cellpix[z]
            if parent.restore and "upsample" in parent.restore:
                outlines = masks_to_outlines(parent.cellpix_orig[z])
                parent.outpix_orig[z] = outlines * parent.cellpix_orig[z]
            if z % 50 == 0 and parent.NZ > 1:
                print("GUI_INFO: plane %d outlines processed" % z)
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_resize = parent.outpix.copy()
    else:
        parent.outpix = outlines
        if parent.restore and "upsample" in parent.restore:
            parent.outpix_resize = parent.outpix.copy()
            parent.outpix_orig = np.zeros_like(parent.cellpix_orig)
            for z in range(parent.NZ):
                outlines = masks_to_outlines(parent.cellpix_orig[z])
                parent.outpix_orig[z] = outlines * parent.cellpix_orig[z]
                if z % 50 == 0 and parent.NZ > 1:
                    print("GUI_INFO: plane %d outlines processed" % z)

    if parent.outpix.ndim == 2:
        parent.outpix = parent.outpix[np.newaxis, :, :]
        if parent.restore and "upsample" in parent.restore:
            if parent.outpix_resize.ndim == 2:
                parent.outpix_resize = parent.outpix_resize[np.newaxis, :, :]
            if parent.outpix_orig.ndim == 2:
                parent.outpix_orig = parent.outpix_orig[np.newaxis, :, :]

    ncells = int(parent.cellpix.max())
    if hasattr(parent, "ncells_counter"):
        parent.ncells_counter.set(ncells)
    colors = parent.colormap[:ncells, :3] if colors is None else colors
    print("GUI_INFO: creating cellcolors and drawing masks")
    parent.cellcolors = np.concatenate((np.array([[255, 255, 255]]), colors),
                                       axis=0).astype(np.uint8)
    if ncells > 0:
        parent.draw_layer()
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(ncells, bool)
    parent.zdraw = list(-1 * np.ones(ncells, np.int16))
    if hasattr(parent, "set_instance_classes"):
        parent.set_instance_classes()

    if hasattr(parent, "set_instance_visible"):
        parent.set_instance_visible()

    if hasattr(parent, "stack_filtered"):
        parent.ViewDropDown.setCurrentIndex(parent.ViewDropDown.count() - 1)
        print("set denoised/filtered view")
    else:
        parent.ViewDropDown.setCurrentIndex(0)


def _save_sets(parent):
    """Save masks to *_seg.cellpose."""
    if hasattr(parent, "presenter"):
        parent.presenter.save_sets()
        return

    from cellpose.gui.session_format import write_session

    filename = _get_output_filename(parent)
    base = os.path.splitext(filename)[0]
    path = base + "_seg.cellpose"
    segmentation_params = parent.get_segmentation_parameters()

    if hasattr(parent, "model"):
        model_name = "cpsam"
        if hasattr(parent, "ModelChooseC"):
            model_name = parent.ModelChooseC.currentText().lower()
        session_data = parent.model.to_session_data(
            source_image=str(parent.filename),
            model=model_name,
            segmentation_params=segmentation_params,
            recompute_masks=bool(getattr(parent, "recompute_masks", False)),
        )
    else:
        print("ERROR: cannot save session without model state")
        return

    try:
        written = write_session(path, session_data)
        print("GUI_INFO: %d ROIs saved to %s" % (parent.ncells(), written))
    except Exception as e:
        print(f"ERROR: {e}")
