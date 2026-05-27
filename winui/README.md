# Cellpose WinUI GUI (Windows)

Native Windows WinUI 3 front-end for Cellpose, backed by the same local Python sidecar used by the macOS SwiftUI app.

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

- **WinUI app** (`winui/CellposeGUI/`): 3-column layout, image canvas, menus, editing
- **ISegmentationEngine**: abstraction for future on-device backend
- **SidecarSegmentationEngine**: HTTP client to Python sidecar
- **Python sidecar** (`cellpose/gui/sidecar/`): FastAPI server wrapping `CellposeModel.eval`, I/O, mask editing, training

The sidecar is shared with the macOS app — no duplicate Python backend.

## Features

- Load images (TIFF, PNG, JPG) and `_seg.npy` state files
- Run CPSAM / custom model segmentation
- Mask overlay with select (click), remove (Ctrl+click), merge (Alt+click)
- Shift+drag drawing to add cells
- Save `_seg.npy`, export masks PNG/TIF
- Folder series discovery and navigation
- Preprocessing panel and live mask recompute from cached flows
- Train new model dialog and custom model management
- Drag-and-drop file loading

## Sidecar API

See `macos/README.md` for the full HTTP API table. The sidecar binds to `127.0.0.1` only.
