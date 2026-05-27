# Cellpose SwiftUI GUI (macOS)

Native macOS SwiftUI front-end for Cellpose. Image I/O, mask editing, series navigation, and `*_seg.cellpose` session files are handled locally in Swift. ML inference, recompute, training, and model management go through a local Python sidecar.

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

```
SwiftUI app (local):  image load, masks, series, .cellpose I/O
        ↓ HTTP (ML only)
Python sidecar:       /infer, /recompute, /train, /models
```

- **SwiftUI app** (`macos/CellposeGUI/`): 3-column layout, image canvas, menus, editing
- **Local services**: `ImageLoaderService`, `CellposeSessionStore`, `MaskEditService`, `SeriesDiscoveryService`
- **MlInferenceEngine protocol**: narrow ML abstraction (`SidecarMlEngine` today)
- **Python sidecar** (`cellpose/gui/sidecar/`): stateless FastAPI server for inference, recompute, training

## Session format

Segmentation state is saved as `{stem}_seg.cellpose` — a ZIP archive containing `manifest.json` and raw binary arrays under `arrays/` (masks, flows, colors). This matches the WinUI app and Python `cellpose/gui/session_format/` package.

## Features

- Load images (TIFF, PNG, JPG) and `*_seg.cellpose` session files
- Run CPSAM / custom model segmentation
- Mask overlay with select (click), remove (⌃+click), merge (⌥+click)
- Shift+drag drawing with Enter/Commit to add cells
- Save `*_seg.cellpose`, export masks PNG/TIF, outlines, flows, ImageJ ROIs
- Folder series discovery and navigation
- Live mask recompute from cached flows (threshold sliders)
- Train new model dialog and custom model management
- Drag-and-drop file loading

## Sidecar API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /models` | List models |
| `POST /infer` | Run segmentation on image path |
| `POST /recompute` | Update masks from cached flows |
| `POST /train` | Train custom model |
| `POST /models/add` | Register custom model |
| `POST /models/remove` | Remove custom model |

Sidecar binds to `127.0.0.1` only.
