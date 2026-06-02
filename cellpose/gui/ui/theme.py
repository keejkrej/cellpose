"""
System light/dark theme sync for the Cellpose Qt GUI.

Uses Qt 6.5+ color scheme hints when available; on Linux falls back to the
freedesktop appearance portal and gsettings, then watches for runtime changes.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Callable

from ..qt import QtCore, QtGui
from qtpy.QtWidgets import QApplication


def _fusion_dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(
        QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black
    )
    return palette


def _fusion_light_palette(app: QApplication) -> QtGui.QPalette:
    style = app.style()
    if style is not None:
        return style.standardPalette()
    return QtGui.QPalette()


def _qt_color_scheme(app: QApplication) -> QtCore.Qt.ColorScheme | None:
    scheme = app.styleHints().colorScheme()
    if scheme == QtCore.Qt.ColorScheme.Unknown:
        return None
    return scheme


def _linux_portal_color_scheme() -> int | None:
    if shutil.which("dbus-send") is None:
        return None
    try:
        result = subprocess.run(
            [
                "dbus-send",
                "--session",
                "--print-reply",
                "--dest=org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
                "org.freedesktop.portal.Settings.Read",
                "string:org.freedesktop.appearance",
                "string:color-scheme",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"uint32\s+(\d+)", result.stdout)
    if match is None:
        return None
    return int(match.group(1))


def _gsettings_get(key: str) -> str | None:
    if shutil.which("gsettings") is None:
        return None
    try:
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", key],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if value in ("", "''"):
        return None
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def _linux_desktop_prefers_dark() -> bool | None:
    portal = _linux_portal_color_scheme()
    if portal == 1:
        return True
    if portal == 2:
        return False

    color_scheme = _gsettings_get("color-scheme")
    if color_scheme == "prefer-dark":
        return True
    if color_scheme == "prefer-light":
        return False

    gtk_theme = _gsettings_get("gtk-theme")
    if gtk_theme is not None and "dark" in gtk_theme.lower():
        return True
    if gtk_theme is not None:
        return False
    return None


def _palette_prefers_dark(app: QApplication) -> bool:
    palette = app.palette()
    text = palette.color(QtGui.QPalette.ColorRole.WindowText)
    window = palette.color(QtGui.QPalette.ColorRole.Window)
    return text.lightness() > window.lightness()


def is_dark_mode(app: QApplication | None = None) -> bool:
    if app is None:
        app = QApplication.instance()
    if app is None:
        return False

    scheme = _qt_color_scheme(app)
    if scheme == QtCore.Qt.ColorScheme.Dark:
        return True
    if scheme == QtCore.Qt.ColorScheme.Light:
        return False

    if sys.platform.startswith("linux"):
        desktop = _linux_desktop_prefers_dark()
        if desktop is not None:
            return desktop

    return _palette_prefers_dark(app)


def apply_app_theme(app: QApplication) -> None:
    if is_dark_mode(app):
        app.setPalette(_fusion_dark_palette())
    else:
        app.setPalette(_fusion_light_palette(app))


def apply_pyqtgraph_theme(dark: bool) -> None:
    import pyqtgraph as pg

    if dark:
        pg.setConfigOptions(foreground="w", background="#353535")
    else:
        pg.setConfigOptions(foreground="k", background="w")


def _refresh_views(app: QApplication) -> None:
    from ..view import MainView

    dark = is_dark_mode(app)
    apply_pyqtgraph_theme(dark)
    for widget in app.allWidgets():
        if isinstance(widget, MainView):
            widget.apply_theme(dark)


def _apply_theme(app: QApplication) -> None:
    apply_app_theme(app)
    _refresh_views(app)


class _GsettingsThemeMonitor(QtCore.QObject):
    def __init__(self, on_change: Callable[[], None], parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._on_change = on_change
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._restart)
        self._restart()

    def _restart(self) -> None:
        if self._process.state() != QtCore.QProcess.ProcessState.NotRunning:
            return
        self._process.start(
            "gsettings",
            ["monitor", "org.gnome.desktop.interface"],
        )

    def _on_output(self) -> None:
        while self._process.canReadLine():
            line = bytes(self._process.readLine()).decode("utf-8", errors="ignore")
            if "color-scheme" in line or "gtk-theme" in line:
                self._on_change()
                return


def install_system_theme_sync(app: QApplication) -> None:
    """Apply the system theme and subscribe to light/dark changes."""
    app.styleHints().unsetColorScheme()
    _apply_theme(app)
    app.styleHints().colorSchemeChanged.connect(lambda *_args: _apply_theme(app))

    if sys.platform.startswith("linux") and shutil.which("gsettings") is not None:
        _GsettingsThemeMonitor(lambda: _apply_theme(app), parent=app)
