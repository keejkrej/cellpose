"""Qt bindings via qtpy (default backend: PySide6)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")
_pyqtgraph_by_qt_api = {
    "pyqt5": "PyQt5",
    "pyside2": "PySide2",
    "pyqt6": "PyQt6",
    "pyside6": "PySide6",
}
os.environ.setdefault(
    "PYQTGRAPH_QT_LIB",
    _pyqtgraph_by_qt_api.get(os.environ.get("QT_API", "pyside6").lower(), "PySide6"),
)

from qtpy import QtCore, QtGui, QtWidgets

__all__ = ["QtCore", "QtGui", "QtWidgets"]
