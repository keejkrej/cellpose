#!/usr/bin/env python3
"""Merge single-channel grayscale series files into multichannel RGB images."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from cellpose import io
from cellpose.gui.series import (
    PLACEHOLDER_ALIASES,
    _sort_key,
    build_series_dataset,
    normalize_series_templates,
)

MAX_CHANNELS = 3

logger = logging.getLogger(__name__)

_RENDER_ALIASES = {
    "position": ("p", "position"),
    "time": ("t", "time"),
    "channel": ("c", "channel"),
    "z": ("z",),
}


def render_series_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for canonical, aliases in _RENDER_ALIASES.items():
        if canonical not in values:
            continue
        for alias in aliases:
            rendered = rendered.replace(f"{{{alias}}}", values[canonical])
    return rendered


def _template_placeholders(template: str) -> set[str]:
    return set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", template))


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return np.mean(image[..., :3], axis=-1)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        return np.mean(image[:3], axis=0)
    raise ValueError(f"expected a grayscale image, got shape {image.shape}")


def merge_channel_group(records: list[dict]) -> np.ndarray:
    ordered = sorted(records, key=lambda record: _sort_key(record["channel"]))
    if len(ordered) > MAX_CHANNELS:
        logger.warning(
            "found %d channels for position=%s time=%s z=%s; using the first %d",
            len(ordered),
            ordered[0]["position"],
            ordered[0]["time"],
            ordered[0]["z"],
            MAX_CHANNELS,
        )
        ordered = ordered[:MAX_CHANNELS]

    planes = []
    for record in ordered:
        image = io.imread(record["path"])
        if image is None:
            raise ValueError(f"could not read image file {record['path']}")
        planes.append(_as_grayscale(image))

    shape = planes[0].shape
    for record, plane in zip(ordered, planes):
        if plane.shape != shape:
            raise ValueError(
                f"channel images must share shape for position={record['position']} "
                f"time={record['time']} z={record['z']}; "
                f"expected {shape}, got {plane.shape} in {record['path']}"
            )

    rgb = np.zeros((*shape, MAX_CHANNELS), dtype=planes[0].dtype)
    for index, plane in enumerate(planes):
        rgb[..., index] = plane
    return rgb


def merge_series_folder(
    folder: str | Path,
    *,
    template: str,
    output_template: str,
) -> list[str]:
    folder_path = Path(folder)
    subfolder_template, filename_template, _ = normalize_series_templates(template)
    if "channel" not in {
        PLACEHOLDER_ALIASES.get(name, name)
        for name in _template_placeholders(filename_template)
    }:
        raise ValueError(
            f"input template '{template}' must include a channel placeholder "
            f"({{c}} or {{channel}})."
        )

    output_filename_template = output_template.strip()
    if "channel" in {
        PLACEHOLDER_ALIASES.get(name, name)
        for name in _template_placeholders(output_filename_template)
    }:
        raise ValueError(
            f"output template '{output_filename_template}' must not include a "
            "channel placeholder."
        )

    dataset = build_series_dataset(
        folder_path,
        subfolder_template=subfolder_template,
        filename_template=filename_template,
    )
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in dataset["records"]:
        groups[(record["position"], record["time"], record["z"])].append(record)

    written_paths: list[str] = []
    for (position, time, z_value), records in sorted(
        groups.items(),
        key=lambda item: (
            _sort_key(item[0][0]),
            _sort_key(item[0][1]),
            _sort_key(item[0][2]),
        ),
    ):
        output_name = render_series_template(
            output_filename_template,
            {"position": position, "time": time, "z": z_value},
        )
        if subfolder_template:
            subfolder_name = render_series_template(
                subfolder_template, {"position": position}
            )
            output_path = folder_path / subfolder_name / output_name
        else:
            output_path = folder_path / output_name

        rgb = merge_channel_group(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(str(output_path), rgb)
        source_paths = ", ".join(record["relative_path"] for record in records)
        logger.info("wrote %s from [%s]", output_path, source_paths)
        written_paths.append(str(output_path))

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge single-channel grayscale series files into RGB images. "
            "Uses the first 3 channels only."
        )
    )
    parser.add_argument(
        "dir",
        type=str,
        help="folder containing single-channel series files",
    )
    parser.add_argument(
        "--template",
        required=True,
        type=str,
        help='input template, e.g. Pos{p}/img_{t}_{c}_{z}.jpg',
    )
    parser.add_argument(
        "--output-template",
        required=True,
        type=str,
        help="output filename template without a channel placeholder, e.g. img_{t}_{z}.jpg",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    written = merge_series_folder(
        args.dir,
        template=args.template,
        output_template=args.output_template,
    )
    logger.info("wrote %d merged image(s)", len(written))


if __name__ == "__main__":
    main()
