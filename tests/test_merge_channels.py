import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest
from cellpose import io

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_channels.py"
_spec = importlib.util.spec_from_file_location("merge_channels", _SCRIPT)
assert _spec and _spec.loader
merge_channels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_channels)


@pytest.fixture
def channel_series_dir(tmp_path):
    folder = tmp_path / "series"
    folder.mkdir()
    for channel in range(3):
        for time in range(2):
            for z in range(2):
                image = np.full((8, 10), fill_value=30 * channel + 10 * time + z, dtype=np.uint8)
                io.imsave(str(folder / f"img_{channel}_{time}_{z}.tif"), image)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def test_merge_channel_series(channel_series_dir):
    written = merge_channels.merge_series_folder(
        channel_series_dir,
        template="img_{c}_{t}_{z}.tif",
        output_template="img_{t}_{z}.tif",
    )
    assert len(written) == 4

    rgb = io.imread(str(channel_series_dir / "img_0_0.tif"))
    assert rgb.shape == (8, 10, 3)
    assert rgb[0, 0, 0] == 0
    assert rgb[0, 0, 1] == 30
    assert rgb[0, 0, 2] == 60

    rgb = io.imread(str(channel_series_dir / "img_1_1.tif"))
    assert rgb[0, 0, 0] == 11
    assert rgb[0, 0, 1] == 41
    assert rgb[0, 0, 2] == 71


def test_merge_with_subfolder_template(tmp_path):
    folder = tmp_path / "series"
    pos_dir = folder / "Pos0"
    pos_dir.mkdir(parents=True)
    for time in range(2):
        for channel in range(3):
            for z in range(2):
                image = np.full((6, 6), fill_value=100 + channel, dtype=np.uint8)
                io.imsave(str(pos_dir / f"img_{time}_{channel}_{z}.tif"), image)

    written = merge_channels.merge_series_folder(
        folder,
        template="Pos{p}/img_{t}_{c}_{z}.tif",
        output_template="img_{t}_{z}.tif",
    )
    assert len(written) == 4

    rgb = io.imread(str(pos_dir / "img_0_0.tif"))
    assert rgb.shape == (6, 6, 3)
    assert rgb[0, 0, 0] == 100
    assert rgb[0, 0, 1] == 101
    assert rgb[0, 0, 2] == 102


def test_merge_uses_first_three_channels_only(tmp_path):
    folder = tmp_path / "many_channels"
    folder.mkdir()
    for channel in range(5):
        image = np.full((4, 4), fill_value=10 * channel, dtype=np.uint8)
        io.imsave(str(folder / f"img_{channel}_0_0.tif"), image)

    merge_channels.merge_series_folder(
        folder,
        template="img_{c}_{t}_{z}.tif",
        output_template="img_{t}_{z}.tif",
    )

    rgb = io.imread(str(folder / "img_0_0.tif"))
    assert rgb.shape == (4, 4, 3)
    assert rgb[0, 0, 0] == 0
    assert rgb[0, 0, 1] == 10
    assert rgb[0, 0, 2] == 20
