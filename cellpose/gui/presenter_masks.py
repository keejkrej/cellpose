"""Mask drawing helpers used by MainPresenter (no Qt)."""

from __future__ import annotations

import datetime

import cv2
import numpy as np

from .model import MainModel


def add_mask_from_points(
    model: MainModel,
    points: list,
    color: np.ndarray,
) -> list | None:
    """Create a mask from drawn stroke points; mutates model.session."""
    session = model.session
    points_all = np.concatenate(points, axis=0)
    zdraw = np.unique(points_all[:, 0])
    z = 0
    ars, acs, vrs, vcs = (
        np.zeros(0, "int"),
        np.zeros(0, "int"),
        np.zeros(0, "int"),
        np.zeros(0, "int"),
    )
    for stroke in points:
        stroke = np.concatenate(stroke, axis=0).reshape(-1, 4)
        vr = stroke[:, 1]
        vc = stroke[:, 2]
        mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
        pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2), axis=-1)[:, np.newaxis, :]
        mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
        ar, ac = np.nonzero(mask)
        ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pvc, pvr = contours[-2][0][:, 0].T
        vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
        ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
        ioverlap = session.cellpix[z][ar, ac] > 0
        if (~ioverlap).sum() < 10:
            print("GUI_ERROR: cell < 10 pixels without overlaps, not drawn")
            return None
        if ioverlap.sum() > 0:
            ar, ac = ar[~ioverlap], ac[~ioverlap]
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            mask[ar - vr.min() + 2, ac - vc.min() + 2] = 1
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0][:, 0].T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
        ars = np.concatenate((ars, ar), axis=0)
        acs = np.concatenate((acs, ac), axis=0)
        vrs = np.concatenate((vrs, vr), axis=0)
        vcs = np.concatenate((vcs, vc), axis=0)

    idx = int(session.cellpix.max()) + 1
    _paint_mask(session, z, ars, acs, vrs, vcs, color, idx)
    session.zdraw.append(zdraw)
    d = datetime.datetime.now()
    session.track_changes.append(
        [d.strftime("%m/%d/%Y, %H:%M:%S"), "added mask", [ars, acs]]
    )
    return [np.array([np.median(ars), np.median(acs)])]


def _paint_mask(session, z, ar, ac, vr, vc, color, idx):
    session.cellpix[z, vr, vc] = idx
    session.cellpix[z, ar, ac] = idx
    session.outpix[z, vr, vc] = idx
    if session.restore and "upsample" in session.restore:
        if session.resize:
            session.cellpix_resize[z, vr, vc] = idx
            session.cellpix_resize[z, ar, ac] = idx
            session.outpix_resize[z, vr, vc] = idx
            session.cellpix_orig[
                z, (vr / session.ratio).astype(int), (vc / session.ratio).astype(int)
            ] = idx
            session.cellpix_orig[
                z, (ar / session.ratio).astype(int), (ac / session.ratio).astype(int)
            ] = idx
            session.outpix_orig[
                z, (vr / session.ratio).astype(int), (vc / session.ratio).astype(int)
            ] = idx
        else:
            session.cellpix_orig[z, vr, vc] = idx
            session.cellpix_orig[z, ar, ac] = idx
            session.outpix_orig[z, vr, vc] = idx
            vrr = (vr.copy() * session.ratio).astype(int)
            vcr = (vc.copy() * session.ratio).astype(int)
            mask = np.zeros((np.ptp(vrr) + 4, np.ptp(vcr) + 4), np.uint8)
            pts = np.stack((vcr - vcr.min() + 2, vrr - vrr.min() + 2), axis=-1)[
                :, np.newaxis, :
            ]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            arr, acr = np.nonzero(mask)
            arr, acr = arr + vrr.min() - 2, acr + vcr.min() - 2
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0].squeeze().T
            vrr, vcr = pvr + vrr.min() - 2, pvc + vcr.min() - 2
            arr, acr = np.hstack((np.vstack((vrr, vcr)), np.vstack((arr, acr))))
            session.cellpix_resize[z, vrr, vcr] = idx
            session.cellpix_resize[z, arr, acr] = idx
            session.outpix_resize[z, vrr, vcr] = idx

    if z == session.current_z:
        session.layerz[ar, ac, :3] = color
        session.layerz[ar, ac, -1] = session.opacity
        session.layerz[vr, vc] = np.array(session.outcolor)


def paint_mask_at(
    model: MainModel, z, ar, ac, vr, vc, color, idx=None
) -> None:
    session = model.session
    if idx is None:
        idx = int(session.cellpix.max()) + 1
    _paint_mask(session, z, ar, ac, vr, vc, color, idx)
