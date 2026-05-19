"""
Custom graphics widgets for the Cellpose GUI view layer.
"""

import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6 import QtCore
from PySide6.QtWidgets import QFrame, QSlider, QVBoxLayout, QWidget
import pyqtgraph as pg

Horizontal = QtCore.Qt.Orientation.Horizontal


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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.lowerSlider)
        layout.addWidget(self.upperSlider)
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
    keyboardRelease = QtCore.Signal()

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        if event.isAutoRepeat():
            return
        if event.key() in [
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_PageUp,
            QtCore.Qt.Key_PageDown,
            QtCore.Qt.Key_Home,
            QtCore.Qt.Key_End,
        ]:
            self.keyboardRelease.emit()


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

    def keyPressEvent(self, ev):
        ev.accept()
        if ev.text() == "-":
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ["+", "="]:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()


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
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        if (
            self.parent.masksOn or self.parent.outlinesOn
        ) and not self.parent.removing_region:
            if (
                self.parent.loaded
                and ev.modifiers() & QtCore.Qt.ShiftModifier
                and not ev.double()
                and not self.parent.deleting_multiple
            ):
                if not self.parent.in_stroke:
                    ev.accept()
                    self.create_start(ev.pos())
                    self.parent.stroke_appended = False
                    self.parent.in_stroke = True
                    self.drawAt(ev.pos(), ev)
                else:
                    ev.accept()
                    self.end_stroke()
                    self.parent.in_stroke = False
            elif not self.parent.in_stroke:
                y, x = int(ev.pos().y()), int(ev.pos().x())
                if y >= 0 and y < self.parent.Ly and x >= 0 and x < self.parent.Lx:
                    if ev.button() == QtCore.Qt.LeftButton and not ev.double():
                        idx = self.parent.cellpix[self.parent.currentZ][y, x]
                        if idx > 0:
                            if ev.modifiers() & QtCore.Qt.ControlModifier:
                                self.parent.remove_cell(idx)
                            elif ev.modifiers() & QtCore.Qt.AltModifier:
                                self.parent.merge_cells(idx)
                            elif (
                                self.parent.masksOn
                                and not self.parent.deleting_multiple
                            ):
                                self.parent.unselect_cell()
                                self.parent.select_cell(idx)
                            elif self.parent.deleting_multiple:
                                if idx in self.parent.removing_cells_list:
                                    self.parent.unselect_cell_multi(idx)
                                    self.parent.removing_cells_list.remove(idx)
                                else:
                                    self.parent.select_cell_multi(idx)
                                    self.parent.removing_cells_list.append(idx)

                        elif self.parent.masksOn and not self.parent.deleting_multiple:
                            self.parent.unselect_cell()

    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        if self.parent.in_stroke:
            self.drawAt(ev.pos())
            if self.is_at_start(ev.pos()):
                self.end_stroke()

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

    def is_at_start(self, pos):
        thresh_out = max(6, self.parent.brush_size * 3)
        thresh_in = max(3, self.parent.brush_size * 1.8)
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (
                ((stroke[1:, 1:] - stroke[:1, 1:][np.newaxis, :, :]) ** 2).sum(axis=-1)
            ) ** 0.5
            dist = dist.flatten()
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4, first_left + 1) :] < thresh_in).sum()
                return has_returned > 0
            return False
        return False

    def end_stroke(self):
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.strokes.append(self.parent.current_stroke)
            self.parent.stroke_appended = True
            self.parent.current_stroke = np.array(self.parent.current_stroke)
            ioutline = self.parent.current_stroke[:, 3] == 1
            self.parent.current_point_set.append(
                list(self.parent.current_stroke[ioutline])
            )
            self.parent.current_stroke = []
            if self.parent.autosave:
                self.parent.add_set()
        if (
            len(self.parent.current_point_set)
            and len(self.parent.current_point_set[0]) > 0
            and self.parent.autosave
        ):
            self.parent.add_set()
        self.parent.in_stroke = False

    def tabletEvent(self, ev):
        pass

    def drawAt(self, pos, ev=None):
        mask = self.strokemask
        stroke = self.parent.current_stroke
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
        if tx[1] >= self.parent.Ly - 1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1] - tx[0] - 1
        if ty[1] >= self.parent.Lx - 1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1] - ty[0] - 1

        ts = (slice(tx[0], tx[1]), slice(ty[0], ty[1]))
        ss = (slice(sx[0], sx[1]), slice(sy[0], sy[1]))
        self.image[ts] = mask[ss]

        for ky, y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx, x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx == kcent[0], ky == kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
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
