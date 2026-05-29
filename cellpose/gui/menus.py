"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
from .qt import QtCore  # noqa: F401 — configure QT_API before qtpy
from qtpy.QtGui import QAction


def mainmenu(view, presenter):
    main_menu = view.menuBar()
    file_menu = main_menu.addMenu("&File")
    loadImg = QAction("&Load image", view)
    loadImg.triggered.connect(presenter.load_image)
    file_menu.addAction(loadImg)

    loadFolderPattern = QAction("Load &folder", view)
    loadFolderPattern.triggered.connect(presenter.load_image_series)
    file_menu.addAction(loadFolderPattern)

    view.saveResults = QAction("&Save results", view)
    view.saveResults.triggered.connect(presenter.save_sets)
    file_menu.addAction(view.saveResults)
    view.saveResults.setEnabled(False)


def editmenu(view, presenter):
    main_menu = view.menuBar()
    edit_menu = main_menu.addMenu("&Edit")
    view.undo = QAction("Undo previous mask/trace", view)
    view.undo.triggered.connect(presenter.undo_action)
    view.undo.setEnabled(False)
    edit_menu.addAction(view.undo)

    view.redo = QAction("Undo remove mask", view)
    view.redo.triggered.connect(presenter.undo_remove_cell)
    view.redo.setEnabled(False)
    edit_menu.addAction(view.redo)

    view.ClearButton = QAction("Clear all masks", view)
    view.ClearButton.triggered.connect(presenter.clear_all)
    view.ClearButton.setEnabled(False)
    edit_menu.addAction(view.ClearButton)

    view.remcell = QAction("Remove selected cell (Ctrl+CLICK)", view)
    view.remcell.triggered.connect(presenter.remove_selected_cells)
    view.remcell.setEnabled(False)
    edit_menu.addAction(view.remcell)

    view.mergecell = QAction("FYI: Merge cells by Alt+Click", view)
    view.mergecell.setEnabled(False)
    edit_menu.addAction(view.mergecell)


def modelmenu(view, presenter):
    main_menu = view.menuBar()
    model_menu = main_menu.addMenu("&Models")
    view.addmodel = QAction("Add custom torch model to GUI", view)
    view.addmodel.triggered.connect(presenter.add_model)
    view.addmodel.setEnabled(True)
    model_menu.addAction(view.addmodel)

    view.removemodel = QAction("Remove selected custom model from GUI", view)
    view.removemodel.triggered.connect(presenter.remove_model)
    view.removemodel.setEnabled(True)
    model_menu.addAction(view.removemodel)

    view.newmodel = QAction("&Train new model with image+masks in folder", view)
    view.newmodel.triggered.connect(presenter.train_new_model)
    view.newmodel.setEnabled(False)
    model_menu.addAction(view.newmodel)
