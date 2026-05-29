from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_PLACEHOLDERS = ("t", "p", "c", "z")
PLACEHOLDER_ALIASES = {
    "t": "time",
    "time": "time",
    "p": "position",
    "position": "position",
    "c": "channel",
    "channel": "channel",
    "z": "z",
}
SERIES_AXES = ("position", "time", "channel", "z")
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
SUPPORTED_SUBFOLDER_TEMPLATES = (
    "Pos{p}",
    "Position{p}",
    "Pos_{p}",
    "Position_{p}",
)
SUPPORTED_FILENAME_TEMPLATES = (
    "img_{t}_{c}_{z}.jpg",
    "img_channel{c}_position{p}_time{t}_z{z}.tif",
)
SUPPORTED_SERIES_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def compile_series_template(
    template: str,
    *,
    case_sensitive: bool = True,
    allow_empty: bool = False,
) -> tuple[re.Pattern[str], tuple[str, ...]]:
    """Convert a placeholder template into a regex."""
    if not template:
        if allow_empty:
            return re.compile(r"^$"), ()
        raise ValueError("Filename template cannot be empty.")

    parts: list[str] = ["^"]
    placeholders: list[str] = []
    cursor = 0
    for match in _PLACEHOLDER_RE.finditer(template):
        parts.append(re.escape(template[cursor : match.start()]))
        name = match.group(1)
        canonical_name = PLACEHOLDER_ALIASES.get(name)
        if canonical_name is None:
            supported = ", ".join(f"{{{field}}}" for field in SUPPORTED_PLACEHOLDERS)
            raise ValueError(
                f"Unsupported placeholder {{{name}}}. Supported placeholders: {supported}."
            )
        if canonical_name in placeholders:
            raise ValueError(f"Placeholder {{{name}}} appears more than once.")
        placeholders.append(canonical_name)
        parts.append(f"(?P<{canonical_name}>.+?)")
        cursor = match.end()

    parts.append(re.escape(template[cursor:]))
    parts.append("$")
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile("".join(parts), flags), tuple(placeholders)


def join_series_template(subfolder_template: str, filename_template: str) -> str:
    if not filename_template:
        raise ValueError("Filename template cannot be empty.")
    return (
        f"{subfolder_template}/{filename_template}"
        if subfolder_template
        else filename_template
    )


def normalize_series_templates(
    template: str | None = None,
    *,
    subfolder_template: str | None = None,
    filename_template: str | None = None,
) -> tuple[str, str, str]:
    if template is not None and (
        subfolder_template is not None or filename_template is not None
    ):
        raise ValueError(
            "Provide either template or subfolder_template/filename_template, not both."
        )

    if template is not None:
        normalized_template = template.strip().replace("\\", "/")
        if normalized_template.count("/") > 1:
            raise ValueError(
                "Series template must contain at most one folder separator."
            )
        if "/" in normalized_template:
            subfolder_template, filename_template = normalized_template.rsplit("/", 1)
        else:
            subfolder_template, filename_template = "", normalized_template
    else:
        subfolder_template = (subfolder_template or "").strip()
        filename_template = (filename_template or "").strip()

    if "/" in subfolder_template or "\\" in subfolder_template:
        raise ValueError("Subfolder template cannot contain path separators.")
    if "/" in filename_template or "\\" in filename_template:
        raise ValueError("Filename template cannot contain path separators.")

    if not filename_template:
        raise ValueError("Filename template cannot be empty.")

    return (
        subfolder_template,
        filename_template,
        join_series_template(subfolder_template, filename_template),
    )


def _sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if re.fullmatch(r"-?\d+", value) else (1, value)


def _is_supported_series_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SERIES_SUFFIXES and not path.name.endswith(
        "_seg.npy"
    )


def _values_are_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    if re.fullmatch(r"-?\d+", left) and re.fullmatch(r"-?\d+", right):
        return int(left) == int(right)
    return False


def _merge_placeholder_values(
    subfolder_values: dict[str, str],
    filename_values: dict[str, str],
    relative_path: str,
) -> dict[str, str]:
    values = {
        key: value for key, value in subfolder_values.items() if value not in (None, "")
    }
    for key, value in filename_values.items():
        if value in (None, ""):
            continue
        existing = values.get(key)
        if existing is None:
            values[key] = value
            continue
        if not _values_are_compatible(existing, value):
            raise ValueError(
                f"Conflicting placeholder values for {relative_path}: "
                f"{key}={existing!r} and {value!r}."
            )
    return values


def _record_lookup_key(
    position: str, time: str, channel: str, z: str
) -> str:
    return f"{position}_{time}_{channel}_{z}"


def _collect_series_matches(
    folder_path: Path, subfolder_template: str, filename_template: str
) -> tuple[list[dict], tuple[str, ...]]:
    subfolder_pattern, subfolder_placeholders = compile_series_template(
        subfolder_template, case_sensitive=False, allow_empty=True
    )
    filename_pattern, filename_placeholders = compile_series_template(filename_template)
    placeholders = tuple(dict.fromkeys(subfolder_placeholders + filename_placeholders))

    matches = []
    candidate_folders = (
        [folder_path] if not subfolder_template else sorted(folder_path.iterdir())
    )
    for candidate_folder in candidate_folders:
        if subfolder_template:
            if not candidate_folder.is_dir():
                continue
            subfolder_name = candidate_folder.name
        else:
            if not candidate_folder.exists():
                continue
            subfolder_name = ""

        subfolder_match = subfolder_pattern.fullmatch(subfolder_name)
        if subfolder_match is None:
            continue

        for path in sorted(candidate_folder.iterdir()):
            if not path.is_file() or not _is_supported_series_file(path):
                continue
            filename_match = filename_pattern.fullmatch(path.name)
            if filename_match is None:
                continue

            relative_path = (
                f"{subfolder_name}/{path.name}" if subfolder_name else path.name
            )
            values = _merge_placeholder_values(
                subfolder_match.groupdict(), filename_match.groupdict(), relative_path
            )
            matches.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "position": values.get("position") or "0",
                    "time": values.get("time") or "0",
                    "channel": values.get("channel") or "0",
                    "z": values.get("z") or "0",
                }
            )

    return matches, placeholders


def suggest_series_templates(folder: str | Path) -> dict:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not a directory.")

    best_match = None
    for subfolder_template in SUPPORTED_SUBFOLDER_TEMPLATES:
        for filename_template in SUPPORTED_FILENAME_TEMPLATES:
            matches, placeholders = _collect_series_matches(
                folder_path, subfolder_template, filename_template
            )
            if not matches:
                continue
            score = (len(matches), len(placeholders))
            if best_match is None or score > best_match["score"]:
                best_match = {
                    "subfolder_template": subfolder_template,
                    "filename_template": filename_template,
                    "template": join_series_template(
                        subfolder_template, filename_template
                    ),
                    "matched_files": len(matches),
                    "score": score,
                    "detected": True,
                }

    if best_match is not None:
        del best_match["score"]
        return best_match

    first_subfolder = ""
    first_filename = ""
    for candidate_folder in sorted(folder_path.iterdir()):
        if not candidate_folder.is_dir():
            continue
        for path in sorted(candidate_folder.iterdir()):
            if path.is_file() and _is_supported_series_file(path):
                first_subfolder = candidate_folder.name
                first_filename = path.name
                break
        if first_filename:
            break

    if not first_filename:
        for path in sorted(folder_path.iterdir()):
            if path.is_file() and _is_supported_series_file(path):
                first_filename = path.name
                break

    return {
        "subfolder_template": first_subfolder,
        "filename_template": first_filename,
        "template": join_series_template(first_subfolder, first_filename)
        if first_filename
        else "",
        "matched_files": 0,
        "detected": False,
    }


def build_series_dataset(
    folder: str | Path,
    template: str | None = None,
    *,
    subfolder_template: str | None = None,
    filename_template: str | None = None,
) -> dict:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not a directory.")

    subfolder_template, filename_template, combined_template = (
        normalize_series_templates(
            template,
            subfolder_template=subfolder_template,
            filename_template=filename_template,
        )
    )
    matches, placeholders = _collect_series_matches(
        folder_path, subfolder_template, filename_template
    )

    if not matches:
        raise ValueError(
            f"No files in {folder_path} matched template '{combined_template}'."
        )

    records = sorted(
        matches,
        key=lambda record: (
            _sort_key(record["position"]),
            _sort_key(record["time"]),
            _sort_key(record["channel"]),
            _sort_key(record["z"]),
            record["relative_path"],
        ),
    )
    lookup: dict[str, int] = {}
    for index, record in enumerate(records):
        key = _record_lookup_key(
            record["position"], record["time"], record["channel"], record["z"]
        )
        if key in lookup:
            raise ValueError(
                "Duplicate file match for "
                f"position={record['position']}, time={record['time']}, "
                f"channel={record['channel']}, z={record['z']}."
            )
        record["label"] = record["relative_path"]
        lookup[key] = index

    axes = {
        axis: sorted({record[axis] for record in records}, key=_sort_key)
        for axis in SERIES_AXES
    }
    axis_index = {
        axis: {value: index for index, value in enumerate(values)}
        for axis, values in axes.items()
    }

    return {
        "folder": str(folder_path),
        "template": combined_template,
        "subfolder_template": subfolder_template,
        "filename_template": filename_template,
        "placeholders": list(placeholders),
        "records": records,
        "axes": axes,
        "axis_index": axis_index,
        "lookup": lookup,
    }


def resolve_series_record_index(
    dataset: dict, *, position: str, time: str, channel: str, z: str
) -> int:
    try:
        return dataset["lookup"][_record_lookup_key(position, time, channel, z)]
    except KeyError as exc:
        raise ValueError(
            "No file matched the selected coordinates "
            f"(position={position}, time={time}, channel={channel}, z={z})."
        ) from exc


def build_series_metadata(dataset: dict, record_index: int) -> dict:
    record = dataset["records"][record_index]
    return {
        "folder": dataset["folder"],
        "template": dataset["template"],
        "subfolder_template": dataset.get("subfolder_template", ""),
        "filename_template": dataset.get("filename_template", dataset["template"]),
        "path": record["path"],
        "position": record["position"],
        "time": record["time"],
        "channel": record["channel"],
        "z": record["z"],
    }


def resolve_series_metadata(metadata: dict) -> tuple[dict, int]:
    if "filename_template" in metadata:
        dataset = build_series_dataset(
            metadata["folder"],
            subfolder_template=metadata.get("subfolder_template", ""),
            filename_template=metadata["filename_template"],
        )
    else:
        dataset = build_series_dataset(metadata["folder"], metadata["template"])

    if "channel" in metadata and "z" in metadata:
        return (
            dataset,
            resolve_series_record_index(
                dataset,
                position=metadata["position"],
                time=metadata["time"],
                channel=metadata["channel"],
                z=metadata["z"],
            ),
        )

    for index, record in enumerate(dataset["records"]):
        if (
            record["position"] == metadata["position"]
            and record["time"] == metadata["time"]
        ):
            return dataset, index

    path = metadata.get("path")
    if path is not None:
        for index, record in enumerate(dataset["records"]):
            if record["path"] == path:
                return dataset, index

    raise ValueError(
        "Saved series metadata refers to a file that is no longer present in the source folder."
    )


def get_output_filename(dataset: dict, record_index: int) -> str:
    return dataset["records"][record_index]["path"]
