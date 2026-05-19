# Cellpose SwiftUI GUI (macOS)

Native macOS SwiftUI front-end for Cellpose, backed by a local Python sidecar that runs the existing PyTorch segmentation stack.

## Requirements

- macOS 14+
- Xcode 15+
- [uv](https://github.com/astral-sh/uv) with the cellpose repo environment

## Development

### 1. Install dependencies

```bash
cd /path/to/cellpose
uv sync
```

### 2. Run the sidecar (optional — the app launches it automatically)

```bash
uv run python -m cellpose.gui.sidecar --port 8787
```

### 3. Open and run the SwiftUI app

```bash
open macos/CellposeGUI.xcodeproj
```

Set environment variables in the Xcode scheme if needed:

| Variable | Purpose |
|----------|---------|
| `SIDECAR_URL` | Existing sidecar URL (default `http://127.0.0.1:8787`) |
| `CELLPOSE_ROOT` | Repo root for auto-launching sidecar via `uv run` |

## Architecture

- **SwiftUI app** (`macos/CellposeGUI/`): 3-column layout, image canvas, menus, editing
- **SegmentationEngine protocol**: abstraction for future Core ML backend
- **SidecarSegmentationEngine**: HTTP client to Python sidecar
- **Python sidecar** (`cellpose/gui/sidecar/`): FastAPI server wrapping `CellposeModel.eval`, I/O, mask editing, training

## Features

- Load images (TIFF, PNG, JPG) and `_seg.npy` state files
- Run CPSAM / custom model segmentation
- Mask overlay with select (click), remove (⌃+click), merge (⌥+click)
- Shift+drag drawing with Enter/Commit to add cells
- Save `_seg.npy`, export masks PNG/TIF
- Folder series discovery and navigation
- Preprocessing panel and live mask recompute from cached flows
- Train new model dialog and custom model management
- Drag-and-drop file loading

## Sidecar API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /models` | List models |
| `POST /segment` | Run segmentation |
| `POST /recompute-masks` | Update masks from cached flows |
| `POST /io/load-image` | Load image file |
| `POST /io/load-seg` | Load `_seg.npy` |
| `POST /io/save-seg` | Save `_seg.npy` |
| `POST /masks/remove` | Remove cells |
| `POST /masks/add` | Add drawn cell |
| `POST /series/discover` | Discover folder series |
| `POST /train` | Train custom model |

Sidecar binds to `127.0.0.1` only.
