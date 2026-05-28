"""Build API session payloads from in-memory sidecar sessions."""

from __future__ import annotations

from .arrays import encode_array, encode_optional
from .schemas import SessionResponse
from .segmentation import display_image_from_stack


def session_response(session) -> SessionResponse:
    image = session.stack_filtered if session.stack_filtered is not None else session.image
    return SessionResponse(
        session_id=session.session_id,
        filename=session.filename,
        shape=list(session.image.shape),
        ncells=session.ncells,
        masks=encode_optional(session.masks.squeeze() if session.masks is not None else None),
        outlines=encode_optional(
            session.outlines.squeeze() if session.outlines is not None else None
        ),
        display_image=encode_array(display_image_from_stack(image)),
        colors=encode_optional(session.colors),
        instance_classes=encode_optional(session.instance_classes),
        flows=[encode_array(flow) for flow in session.flows] if session.flows else None,
        recompute_masks=session.recompute_masks,
        metadata={
            "normalize_params": session.normalize_params,
            "segmentation_params": session.segmentation_params,
            "restore": session.restore,
            "ratio": session.ratio,
        },
    )
