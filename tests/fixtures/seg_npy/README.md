# `_seg.npy` session fixtures

Binary fixtures for session read/write tests. Regenerate after format changes:

```bash
uv run python scripts/generate_seg_fixtures.py
```

Files:

- `minimal_seg.npy` — written via `write_session` (current Python GUI path)
- `legacy_gui_seg.npy` — pickled dict matching original Cellpose GUI fields
- `with_instance_classes_seg.npy` — includes `instance_classes`

Use these for WinUI/Swift `SegNpyIO` manual parity checks as well.
