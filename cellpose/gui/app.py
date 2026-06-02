"""
Application bootstrap for the Cellpose Qt GUI.

Creates QApplication, MainModel, MainView, and MainPresenter; presenter owns
composition and startup.
"""

from __future__ import annotations

import os
import pathlib
import sys
import warnings

from .qt import QtCore, QtGui
from qtpy.QtWidgets import QApplication

from .. import models
from ..utils import download_url_to_file
from .model import MainModel
from .presenter import MainPresenter
from .ui.theme import install_system_theme_sync
from .view import MainView


def _setup_app_icon(app: QApplication) -> None:
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath(".cellpose")
        cp_dir.mkdir(exist_ok=True)
        print("downloading logo")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_transparent.png",
            icon_path,
            progress=True,
        )
    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 256):
        app_icon.addFile(icon_path, QtCore.QSize(size, size))
    app.setWindowIcon(app_icon)


def run(image=None) -> None:
    from ..io import logger_setup

    logger, _log_file = logger_setup()
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    _setup_app_icon(app)
    app.setStyle("Fusion")
    install_system_theme_sync(app)

    model = MainModel(
        model_save_folder=os.fspath(models.MODEL_DIR.joinpath("custom")),
    )
    view = MainView(logger=logger)
    presenter = MainPresenter(view, model)
    presenter.start(image=image)

    ret = app.exec()
    sys.exit(ret)
