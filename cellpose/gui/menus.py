"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
from PySide6.QtGui import QAction
from . import io


def mainmenu(parent):
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    loadImg = QAction("&Load image", parent)
    loadImg.triggered.connect(lambda: io._load_image(parent))
    file_menu.addAction(loadImg)

    loadFolderPattern = QAction("Load &folder", parent)
    loadFolderPattern.triggered.connect(lambda: io._load_image_series(parent))
    file_menu.addAction(loadFolderPattern)

    parent.saveResults = QAction("&Save results", parent)
    parent.saveResults.triggered.connect(lambda: io._save_sets(parent))
    file_menu.addAction(parent.saveResults)
    parent.saveResults.setEnabled(False)


def editmenu(parent):
    main_menu = parent.menuBar()
    edit_menu = main_menu.addMenu("&Edit")
    parent.undo = QAction("Undo previous mask/trace", parent)
    parent.undo.triggered.connect(parent.undo_action)
    parent.undo.setEnabled(False)
    edit_menu.addAction(parent.undo)

    parent.redo = QAction("Undo remove mask", parent)
    parent.redo.triggered.connect(parent.undo_remove_action)
    parent.redo.setEnabled(False)
    edit_menu.addAction(parent.redo)

    parent.ClearButton = QAction("Clear all masks", parent)
    parent.ClearButton.triggered.connect(parent.clear_all)
    parent.ClearButton.setEnabled(False)
    edit_menu.addAction(parent.ClearButton)

    parent.remcell = QAction("Remove selected cell (Ctrl+CLICK)", parent)
    parent.remcell.triggered.connect(parent.remove_action)
    parent.remcell.setEnabled(False)
    edit_menu.addAction(parent.remcell)

    parent.mergecell = QAction("FYI: Merge cells by Alt+Click", parent)
    parent.mergecell.setEnabled(False)
    edit_menu.addAction(parent.mergecell)


def modelmenu(parent):
    main_menu = parent.menuBar()
    io._init_model_list(parent)
    model_menu = main_menu.addMenu("&Models")
    parent.addmodel = QAction("Add custom torch model to GUI", parent)
    #parent.addmodel.setShortcut("Ctrl+A")
    parent.addmodel.triggered.connect(parent.add_model)
    parent.addmodel.setEnabled(True)
    model_menu.addAction(parent.addmodel)

    parent.removemodel = QAction("Remove selected custom model from GUI", parent)
    #parent.removemodel.setShortcut("Ctrl+R")
    parent.removemodel.triggered.connect(parent.remove_model)
    parent.removemodel.setEnabled(True)
    model_menu.addAction(parent.removemodel)

    parent.newmodel = QAction("&Train new model with image+masks in folder", parent)
    parent.newmodel.triggered.connect(parent.new_model)
    parent.newmodel.setEnabled(False)
    model_menu.addAction(parent.newmodel)
