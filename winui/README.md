# Cellpose WinUI GUI (Windows)

Native Windows WinUI 3 front-end for Cellpose. The app owns image I/O, mask editing, series navigation, and session files locally. A slim Python sidecar handles ML only (inference, flow-based recompute, training, model registry).

## Requirements

- Windows 10 1809+ (Windows 11 recommended)
- [.NET 10 SDK](https://dotnet.microsoft.com/download) (or the SDK version targeted by the project)
- [Windows App SDK](https://learn.microsoft.com/windows/apps/windows-app-sdk/) (restored via NuGet when building)
- [uv](https://github.com/astral-sh/uv) with the cellpose repo environment

## Development

### 1. Install dependencies

```powershell
cd C:\path\to\cellpose
uv sync
```

### 2. Run the sidecar (optional — the app launches it automatically)

```powershell
uv run python -m cellpose.gui.sidecar --port 8787
```

### 3. Build and run the WinUI app

```powershell
cd winui\CellposeGUI
dotnet run
```

Set environment variables if needed:

| Variable | Purpose |
|----------|---------|
| `SIDECAR_URL` | Existing sidecar URL (default `http://127.0.0.1:8787`) |
| `CELLPOSE_ROOT` | Repo root for auto-launching sidecar via `uv run` |

## Architecture

- **WinUI app** (`winui/CellposeGUI/`): UI, canvas, local session state, mask editing, series discovery
- **Local services**: `ImageLoaderService`, `CellposeSessionStore`, `MaskEditService`, `SeriesDiscoveryService`
- **IMlInferenceEngine** / **SidecarMlEngine**: HTTP client for ML-only sidecar endpoints
- **Python sidecar** (`cellpose/gui/sidecar/`): stateless `/infer`, `/recompute`, `/train`, `/models`
- **Session format** (`cellpose/gui/session_format/`): portable `{stem}_seg.cellpose` zip archives (manifest + raw arrays)

## Sidecar API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /models` | List models |
| `POST /infer` | Run segmentation on image path or array |
| `POST /recompute` | Recompute masks from cached flows |
| `POST /train` | Train custom model from `*_seg.cellpose` labels |
| `POST /models/add` | Install custom model |
| `POST /models/remove` | Remove custom model |

Sidecar binds to `127.0.0.1` only.

## Features

- Load images (TIFF, PNG, JPG) locally
- Save/load `{stem}_seg.cellpose` session files
- Run CPSAM / custom model segmentation via sidecar
- Mask overlay with select (click), remove (Ctrl+click), merge (Alt+click)
- Shift+drag drawing to add cells (local mask edit)
- Live mask recompute from cached flows when thresholds change
- Export masks PNG/TIF, outlines, flows, ROIs
- Folder series discovery and navigation
- Train new model dialog and custom model management
- Drag-and-drop file loading
