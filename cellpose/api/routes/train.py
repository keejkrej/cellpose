"""Model training and management routes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from cellpose.app_core.train import get_train_set
from cellpose.models import MODEL_DIR, MODEL_LIST_PATH, get_user_models
from cellpose.train import train_seg

from ..schemas import AddModelRequest, RemoveModelRequest, TrainRequest
from ..segmentation import get_model

router = APIRouter(tags=["train"])


@router.post("/train")
def train(request: TrainRequest) -> dict:
    folder = os.path.expanduser(request.train_data_folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    image_names = sorted(
        [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
        ]
    )
    train_data, train_labels, train_files, _, _ = get_train_set(image_names)
    if len(train_data) == 0:
        raise HTTPException(
            status_code=400,
            detail="No training data with *_seg.npy labels found",
        )

    save_folder = request.model_save_folder or str(MODEL_DIR / "custom")
    os.makedirs(save_folder, exist_ok=True)
    model = get_model()
    try:
        model_path, train_losses, test_losses = train_seg(
            model.net,
            train_data=train_data,
            train_labels=train_labels,
            train_files=train_files,
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            n_epochs=request.n_epochs,
            model_name=request.model_name,
            save_path=save_folder,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    custom_name = Path(model_path).name
    custom_dir = MODEL_DIR / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    target = custom_dir / custom_name
    if str(Path(model_path).resolve()) != str(target.resolve()):
        shutil.copy2(model_path, target)

    model_strings = get_user_models()
    if custom_name not in model_strings:
        model_strings.append(custom_name)
        with open(MODEL_LIST_PATH, "w") as textfile:
            for model_string in model_strings:
                textfile.write(model_string + "\n")

    return {
        "model_path": str(target),
        "model_name": custom_name,
        "train_losses": [float(x) for x in train_losses[-5:]] if train_losses is not None else [],
        "test_losses": [float(x) for x in test_losses[-5:]] if test_losses is not None else [],
    }


@router.post("/models/add")
def add_model(request: AddModelRequest) -> dict:
    path = os.path.expanduser(request.path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    fname = os.path.basename(path)
    target = MODEL_DIR / "custom" / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    model_strings = get_user_models()
    if fname not in model_strings:
        model_strings.append(fname)
        with open(MODEL_LIST_PATH, "w") as textfile:
            for model_string in model_strings:
                textfile.write(model_string + "\n")
    return {"model_name": fname}


@router.post("/models/remove")
def remove_model(request: RemoveModelRequest) -> dict:
    model_strings = get_user_models()
    if request.model_name not in model_strings:
        raise HTTPException(status_code=404, detail="Model not found")
    model_strings.remove(request.model_name)
    with open(MODEL_LIST_PATH, "w") as textfile:
        for model_string in model_strings:
            textfile.write(model_string + "\n")
    model_path = MODEL_DIR / "custom" / request.model_name
    if model_path.exists():
        os.remove(model_path)
    return {"removed": request.model_name}
