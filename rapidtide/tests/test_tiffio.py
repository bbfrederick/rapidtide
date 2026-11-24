"""Float32 TIFF fidelity, geometry, sample metadata, and BigTIFF selection."""

import json

import numpy as np
import pytest

from rapidtide.tiffio import needs_bigtiff, open_map, save_map
from rapidtide.zarrio import blood_header

tifffile = pytest.importorskip("tifffile")


@pytest.mark.parametrize("compression", [None, "deflate"])
@pytest.mark.parametrize(
    "axis,frames", [(None, None), ("time", 1), ("time", 7), ("lag", 7), ("frequency", 7)]
)
def test_tiff_roundtrip(tmp_path, compression, axis, frames):
    shape = (3, 5) if frames is None else (3, 5, frames)
    native = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    native.reshape(-1)[:5] = [np.nan, np.inf, -np.inf, -0.0, -1.25]
    internal = native.T[..., None] if frames is None else native.transpose(1, 0, 2)[:, :, None, :]
    path = tmp_path / "result.tif"
    header = blood_header((3, 5, frames or 1), 0.125, -2)
    info = save_map(
        path,
        internal,
        header,
        {"SourceShape": np.array([3, 5]), "Units": "test"},
        axis or "time",
        compression,
    )
    group = open_map(path)
    result = group["data"][:]
    assert result.shape == native.shape and result.dtype == np.float32
    np.testing.assert_array_equal(result.view(np.uint32), native.view(np.uint32))
    with tifffile.TiffFile(path) as file:
        assert len(file.pages) == (frames or 1)
        assert all(page.dtype == np.float32 and page.shape == (3, 5) for page in file.pages)
        assert file.pages[0].photometric.name == "MINISBLACK"
        assert (
            file.pages[0].compression.name in ("DEFLATE", "ADOBE_DEFLATE")
            if compression
            else file.pages[0].compression.name == "NONE"
        )
        assert file.series[0].axes == (
            "YX" if frames is None else "TYX" if axis == "time" else "QYX"
        )
        assert json.loads(file.pages[0].description)["rapidtide"] == info
    assert info["SourceShape"] == [3, 5]
    if frames is not None:
        coordinates = -2 + np.arange(frames) * 0.125
        name = "timestamps" if axis == "time" else axis
        np.testing.assert_array_equal(group[name], coordinates)
        np.testing.assert_array_equal(group["data"][..., 0], native[..., 0])
        np.testing.assert_array_equal(group["data"][..., 1:4], native[..., 1:4])
        assert info["sample_units"] == ("Hz" if axis == "frequency" else "seconds")


def test_bigtiff_without_large_allocation(tmp_path, monkeypatch):
    import rapidtide.tiffio as module

    assert needs_bigtiff((2048, 1024, 1024))
    assert not needs_bigtiff((10, 20, 30))
    monkeypatch.setattr(module, "needs_bigtiff", lambda shape: True)
    path = tmp_path / "big.tif"
    save_map(path, np.ones((5, 3, 1)), blood_header((3, 5, 1), 1, 0), {})
    with tifffile.TiffFile(path) as file:
        assert file.is_bigtiff


def test_no_lossy_compression_or_incomplete_replacement(tmp_path, monkeypatch):
    path = tmp_path / "result.tif"
    path.write_bytes(b"previous result")
    data = np.ones((5, 3, 1))
    header = blood_header((3, 5, 1), 1, 0)
    with pytest.raises(ValueError, match="lossless"):
        save_map(path, data, header, {}, compression="jpeg")

    def fail(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(tifffile, "imwrite", fail)
    with pytest.raises(OSError, match="write failed"):
        save_map(path, data, header, {})
    assert path.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [path]
