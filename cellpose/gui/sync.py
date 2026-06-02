"""View sync scopes for presenter-driven model–view updates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto


class SyncScope(IntFlag):
    """Bit flags for granular canvas and sidebar updates."""

    LABELS_TABLE = auto()
    VISIBILITY_HEADER = auto()
    CANVAS_IMAGE = auto()
    CANVAS_MASK = auto()
    CANVAS_SELECTION = auto()
    SCALE = auto()

    LABELS = LABELS_TABLE | VISIBILITY_HEADER
    SELECTION_CHANGE = CANVAS_SELECTION
    INSTANCE_EDIT = LABELS_TABLE | CANVAS_MASK | CANVAS_SELECTION
    FILTER_CHANGE = INSTANCE_EDIT
    VISIBILITY_EDIT = LABELS_TABLE | CANVAS_MASK | VISIBILITY_HEADER


@dataclass(frozen=True, slots=True)
class SyncRequest:
    scope: SyncScope
    prune_selection: bool = True


def create_sync_notifier():
    """Build a QObject that emits :class:`SyncRequest` payloads."""

    from .qt import QtCore

    class GuiStateNotifier(QtCore.QObject):
        sync_requested = QtCore.Signal(object)

    return GuiStateNotifier()
