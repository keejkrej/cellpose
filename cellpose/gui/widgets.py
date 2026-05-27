"""
Custom graphics widgets for the Cellpose GUI view layer.
"""

import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

Horizontal = QtCore.Qt.Orientation.Horizontal


class TristateHeaderCheckBox(QCheckBox):
    """Tristate for sync display; user clicks toggle only checked/unchecked."""

    def nextCheckState(self):
        if self.checkState() == QtCore.Qt.CheckState.Checked:
            self.setCheckState(QtCore.Qt.CheckState.Unchecked)
        else:
            self.setCheckState(QtCore.Qt.CheckState.Checked)


class CheckBoxHeader(QHeaderView):
    checkboxClicked = QtCore.Signal(int)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._checkbox = TristateHeaderCheckBox(self)
        self._checkbox.setTristate(True)
        self._checkbox.setToolTip("Show or hide all cell masks and outlines")
        self._checkbox.stateChanged.connect(self.checkboxClicked.emit)
        self.sectionResized.connect(self._update_checkbox_geometry)
        self.geometriesChanged.connect(self._update_checkbox_geometry)

    def showEvent(self, event):
        super().showEvent(event)
        self._update_checkbox_geometry()

    def _update_checkbox_geometry(self, *_args):
        if self.count() == 0:
            return
        x = self.sectionPosition(0)
        width = self.sectionSize(0)
        height = self.height()
        size = min(20, max(width - 4, 0), max(height - 4, 0))
        self._checkbox.setGeometry(
            int(x + (width - size) / 2),
            int((height - size) / 2),
            size,
            size,
        )

    def set_check_state(self, state):
        self._checkbox.blockSignals(True)
        self._checkbox.setCheckState(state)
        self._checkbox.blockSignals(False)


class Slider(QWidget):
    valueChanged = QtCore.Signal()

    def __init__(self, parent, name, color):
        super().__init__(parent)
        self._scale = 10
        self._value = [0.0, 99.0]
        self.name = name

        self.setEnabled(False)
        if parent is not None:
            self.valueChanged.connect(lambda: self.levelChanged(parent))

        self.lowerSlider = QSlider(Horizontal)
        self.upperSlider = QSlider(Horizontal)
        self.lowerSlider.valueChanged.connect(self._update_value)
        self.upperSlider.valueChanged.connect(self._update_value)

        min_row = QHBoxLayout()
        min_row.setContentsMargins(0, 0, 0, 0)
        min_row.setSpacing(6)
        min_row.addWidget(QLabel("min"))
        min_row.addWidget(self.lowerSlider, 1)

        max_row = QHBoxLayout()
        max_row.setContentsMargins(0, 0, 0, 0)
        max_row.setSpacing(6)
        max_row.addWidget(QLabel("max"))
        max_row.addWidget(self.upperSlider, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(min_row)
        layout.addLayout(max_row)
        self.show()

    def setMinimum(self, value):
        self.lowerSlider.setMinimum(int(round(value * self._scale)))
        self.upperSlider.setMinimum(int(round(value * self._scale)))

    def setMaximum(self, value):
        self.lowerSlider.setMaximum(int(round(value * self._scale)))
        self.upperSlider.setMaximum(int(round(value * self._scale)))

    def setValue(self, value):
        self.lowerSlider.blockSignals(True)
        self.upperSlider.blockSignals(True)
        self.lowerSlider.setValue(int(round(value[0] * self._scale)))
        self.upperSlider.setValue(int(round(value[1] * self._scale)))
        self.lowerSlider.blockSignals(False)
        self.upperSlider.blockSignals(False)
        self._update_value(emit=False)

    def value(self):
        return self._value

    def _update_value(self, emit=True):
        self._value = sorted(
            [
                self.lowerSlider.value() / self._scale,
                self.upperSlider.value() / self._scale,
            ]
        )
        if emit:
            self.valueChanged.emit()

    def levelChanged(self, parent):
        parent.level_change(self.name)


class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setLineWidth(8)


class SeriesAxisSlider(QSlider):
    pass


def make_bwr():
    b = np.append(255 * np.ones(128), np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255 * np.ones(128))[:, np.newaxis]
    g = np.append(np.linspace(0, 255, 128), np.linspace(0, 255, 128)[::-1])[
        :, np.newaxis
    ]
    color = np.concatenate((r, g, b), axis=-1).astype(np.uint8)
    return pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)


def as_gray_image(image):
    if image.ndim > 2:
        return image[..., 0]
    return image


def brush_cursor() -> QtGui.QCursor:
    return QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor)


def select_cursor() -> QtGui.QCursor:
    return QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor)


class ViewBoxNoRightDrag(pg.ViewBox):
    def __init__(
        self,
        parent=None,
        border=None,
        lockAspect=False,
        enableMouse=True,
        invertY=False,
        enableMenu=True,
        name=None,
        invertX=False,
    ):
        pg.ViewBox.__init__(
            self, None, border, lockAspect, enableMouse, invertY, enableMenu, name, invertX
        )
        self.parent = parent
        self.axHistoryPointer = -1


class ImageDraw(pg.ImageItem):
    sigImageChanged = QtCore.Signal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()
        self.levels = np.array([0, 255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = "row-major"
        self.removable = False

        self.parent = parent
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.model.drawing.current_stroke = []
        self.parent.model.drawing.in_stroke = False

    def mouseClickEvent(self, ev):
        if self.parent.rect_select_mode:
            ev.accept()
            return
        session = self.parent.model.session
        selection = self.parent.model.selection
        drawing = self.parent.model.drawing
        if session.loaded and not selection.removing_region:
            if (
                drawing.brush_mode
                and ev.button() == QtCore.Qt.LeftButton
                and not ev.double()
                and not selection.deleting_multiple
                and not self.parent.rect_select_mode
            ):
                if not drawing.in_stroke:
                    ev.accept()
                    self.create_start(ev.pos())
                    drawing.stroke_appended = False
                    drawing.in_stroke = True
                    self.drawAt(ev.pos(), ev)
                else:
                    # Second click closes the stroke; hover only extends preview.
                    ev.accept()
                    self.drawAt(ev.pos(), ev)
                    self.end_stroke()
                    drawing.in_stroke = False
            elif not drawing.in_stroke:
                y, x = int(ev.pos().y()), int(ev.pos().x())
                if y >= 0 and y < session.ly and x >= 0 and x < session.lx:
                    if ev.button() == QtCore.Qt.LeftButton and not ev.double():
                        idx = session.cellpix[session.current_z][y, x]
                        if idx > 0:
                            if ev.modifiers() & QtCore.Qt.ControlModifier:
                                self.parent.remove_cell(idx)
                            elif ev.modifiers() & QtCore.Qt.AltModifier:
                                self.parent.merge_cells(idx)
                            elif (
                                not selection.deleting_multiple
                                and not self.parent.rect_select_mode
                            ):
                                self.parent.unselect_cell()
                                self.parent.select_cell(idx)
                            elif selection.deleting_multiple:
                                if idx in selection.removing_cells_list:
                                    self.parent.unselect_cell_multi(idx)
                                    selection.removing_cells_list.remove(idx)
                                else:
                                    self.parent.select_cell_multi(idx)
                                    selection.removing_cells_list.append(idx)

                        elif not selection.deleting_multiple:
                            self.parent.unselect_cell()

    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        # Extend preview while drawing; never auto-close on proximity to start.
        if self.parent.model.drawing.in_stroke:
            self.drawAt(ev.pos())

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem(
            [pos.x()],
            [pos.y()],
            pxMode=False,
            pen=pg.mkPen(color=(255, 0, 0), width=self.parent.brush_size),
            size=max(3 * 2, self.parent.brush_size * 1.8 * 2),
            brush=None,
        )
        self.parent.p0.addItem(self.scatter)

    def end_stroke(self):
        drawing = self.parent.model.drawing
        self.parent.p0.removeItem(self.scatter)
        if not drawing.stroke_appended:
            drawing.strokes.append(drawing.current_stroke)
            drawing.stroke_appended = True
            drawing.current_stroke = np.array(drawing.current_stroke)
            ioutline = drawing.current_stroke[:, 3] == 1
            drawing.current_point_set.append(list(drawing.current_stroke[ioutline]))
            drawing.current_stroke = []
            self.parent.add_set()
        if len(drawing.current_point_set) and len(drawing.current_point_set[0]) > 0:
            self.parent.add_set()
        drawing.in_stroke = False

    def tabletEvent(self, ev):
        pass

    def drawAt(self, pos, ev=None):
        session = self.parent.model.session
        drawing = self.parent.model.drawing
        mask = self.strokemask
        stroke = drawing.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0, dk.shape[0]]
        sy = [0, dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0] + dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1] + dk.shape[1]]
        kcent = kc.copy()
        if tx[0] <= 0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx = sx
            kcent[0] = 0
        if ty[0] <= 0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty = sy
            kcent[1] = 0
        if tx[1] >= session.ly - 1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = session.ly - kc[0] - 1
            tx[1] = session.ly
            kcent[0] = tx[1] - tx[0] - 1
        if ty[1] >= session.lx - 1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = session.lx - kc[1] - 1
            ty[1] = session.lx
            kcent[1] = ty[1] - ty[0] - 1

        ts = (slice(tx[0], tx[1]), slice(ty[0], ty[1]))
        ss = (slice(sx[0], sx[1]), slice(sy[0], sy[1]))
        self.image[ts] = mask[ss]

        for ky, y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx, x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx == kcent[0], ky == kcent[1])
                stroke.append([session.current_z, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs, bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [
            int(np.floor(kernel.shape[0] / 2)),
            int(np.floor(kernel.shape[1] / 2)),
        ]
        onmask = 255 * kernel[:, :, np.newaxis]
        offmask = np.zeros((bs, bs, 1))
        opamask = 100 * kernel[:, :, np.newaxis]
        self.redmask = np.concatenate((onmask, offmask, offmask, onmask), axis=-1)
        self.strokemask = np.concatenate((onmask, offmask, onmask, opamask), axis=-1)


class ObservableVariable(QtCore.QObject):
    valueChanged = QtCore.Signal(object)

    def __init__(self, initial=None):
        super().__init__()
        self._value = initial

    def set(self, new_value):
        if new_value != self._value:
            self._value = new_value
            self.valueChanged.emit(new_value)

    def get(self):
        return self._value

    def __call__(self):
        return self._value

    def reset(self):
        self.set(0)

    def __iadd__(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Value must be numeric.")
        self.set(self._value + amount)
        return self

    def __radd__(self, other):
        return other + self._value

    def __add__(self, other):
        return other + self._value

    def __isub__(self, amount):
        if not isinstance(amount, (int, float)):
            raise TypeError("Value must be numeric.")
        self.set(self._value - amount)
        return self

    def __str__(self):
        return str(self._value)

    def __lt__(self, x):
        return self._value < x

    def __gt__(self, x):
        return self._value > x

    def __eq__(self, x):
        return self._value == x
