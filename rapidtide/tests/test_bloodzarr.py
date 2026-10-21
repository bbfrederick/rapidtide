"""Blood Zarr schema, geometry, timing, and workflow integration tests."""

import numpy as np
import pytest

import rapidtide.io as tide_io
from rapidtide.voxelData import VoxelData
from rapidtide.zarrio import is_zarr, open_blood_zarr

zarr = pytest.importorskip("zarr")


@pytest.fixture
def blood_store(tmp_path):
    path = tmp_path / "blood.zarr"
    times = 7.0 + np.arange(480) * 0.5
    data = np.empty((3, 5, len(times)), dtype=np.float32)
    delays = np.arange(15).reshape(3, 5) / 10
    for y, x in np.ndindex(3, 5):
        t = times - times[0] - delays[y, x]
        data[y, x] = np.sin(2 * np.pi * 0.043 * t) + 0.4 * np.sin(2 * np.pi * 0.081 * t)
        data[y, x] -= data[y, x].mean()
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = False
    data[0, 0] = np.nan
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.attrs.update(
        {
            "schema": "rapidtide.video_to_blood",
            "schema_version": 1,
            "axes": ["y", "x", "time"],
            "sampling_interval_seconds": 0.5,
            "start_time_seconds": 7.0,
            "mean_removed": True,
        }
    )
    group.create_array("blood", data=data, chunks=(2, 3, 16))
    group.create_array("timestamps", data=times)
    group.create_array("valid_mask", data=mask)
    return path, data, times, mask


def test_voxel_geometry_and_timing(blood_store):
    path, native, times, mask = blood_store
    data = VoxelData(str(path), validstart=5, validend=24)
    assert data.filetype == "zarr"
    assert data.getdims() == (5, 3, 1, 480)
    assert data.getsizes() == (1, 1, 1, 0.5)
    assert data.nativespaceshape == (5, 3, 1)
    assert data.nativefmrishape == (5, 3, 1, 20)
    assert data.byvol().dtype == np.float32
    np.testing.assert_array_equal(data.timestamps, times)
    np.testing.assert_array_equal(data.spatial_valid_mask, mask.T.reshape(-1))
    np.testing.assert_array_equal(data.byvol()[4, 2, 0], native[2, 4])
    np.testing.assert_array_equal(data.byvoxel()[4 * 3 + 2], native[2, 4, 5:25])
    assert data.copyheader()["toffset"] == 9.5
    assert data.copyheader()["pixdim"][4] == 0.5
    assert data.smooth(-1) == 0
    data.reload()
    np.testing.assert_array_equal(data.byvoxel()[4 * 3 + 2], native[2, 4, 5:25])


def test_result_geometry(blood_store, tmp_path):
    path, native, times, mask = blood_store
    data = VoxelData(str(path), validstart=5, validend=24)
    native_map = np.arange(15, dtype=np.float32).reshape(3, 5)
    internal_map = native_map.T.reshape(-1)
    tide_io.savemaplist(
        str(tmp_path / "result"),
        [(internal_map, "test", "map", None, None)],
        None,
        data.nativespaceshape,
        data.copyheader(numtimepoints=1),
        {},
        filetype="zarr",
    )
    result = zarr.open_group(str(tmp_path / "result_desc-test_map.zarr"), mode="r")
    np.testing.assert_array_equal(result["data"][:], native_map)
    assert result.attrs["axes"] == ["y", "x"]
    tide_io.savemaplist(
        str(tmp_path / "result"),
        [(data.byvoxel(), "test", "bold", None, None)],
        None,
        data.nativefmrishape,
        data.copyheader(),
        {},
        filetype="zarr",
    )
    result = zarr.open_group(str(tmp_path / "result_desc-test_bold.zarr"), mode="r")
    np.testing.assert_array_equal(result["data"][2, 4], native[2, 4, 5:25])
    np.testing.assert_array_equal(result["timestamps"][:], times[5:25])


@pytest.mark.parametrize(
    "problem", ["schema", "axes", "shape", "mask", "times", "irregular", "step"]
)
def test_malformed_store(blood_store, problem):
    path, _, times, mask = blood_store
    group = zarr.open_group(str(path), mode="a")
    if problem == "schema":
        group.attrs["schema_version"] = 99
    elif problem == "axes":
        group.attrs["axes"] = ["x", "y", "time"]
    elif problem == "shape":
        del group["timestamps"]
        group.create_array("timestamps", data=times[:-1])
    elif problem == "mask":
        group["valid_mask"][:] = False
    elif problem == "times":
        group["timestamps"][1] = times[0]
    elif problem == "irregular":
        group["timestamps"][30] += 0.1
    else:
        group.attrs["sampling_interval_seconds"] = 0.6
    with pytest.raises(ValueError):
        open_blood_zarr(str(path))


def test_sampling_override_and_nonfinite(blood_store):
    path, _, _, mask = blood_store
    with pytest.raises(ValueError, match="override disagrees"):
        VoxelData(str(path), timestep=1)
    group = zarr.open_group(str(path), mode="a")
    group["blood"][1, 3, 3] = np.inf
    data = VoxelData(str(path))
    assert not data.spatial_valid_mask.reshape(5, 3)[3, 1]
    assert np.isfinite(data.byvol()).all()


def test_parser_sampling(blood_store, tmp_path):
    from rapidtide.workflows.rapidtide_parser import process_args

    path, _, _, _ = blood_store
    assert is_zarr(str(path))
    args, _ = process_args([str(path), str(tmp_path / "result")])
    assert args["realtr"] == 0.5
    assert args["inputfreq"] == 2
    assert args["dataiszeromean"]


@pytest.mark.parametrize("refinement", [False, True])
def test_full_blood_workflow(blood_store, tmp_path, monkeypatch, refinement):
    """Run correlation, map reconstruction, and time-series output on an asymmetric grid."""
    from rapidtide.workflows.rapidtide import rapidtide_main
    from rapidtide.workflows.rapidtide_parser import process_args
    import rapidtide.util as tide_util

    configure = tide_util.configurepyfftw

    def configure_local_wisdom(*args, **kwargs):
        configure(*args, **kwargs)
        return str(tmp_path / "fftw_wisdom.txt")

    monkeypatch.setattr(tide_util, "configurepyfftw", configure_local_wisdom)

    path, _, times, mask = blood_store
    output = str(tmp_path / "analysis")
    args = [
        str(path),
        output,
        "--passes",
        "2" if refinement else "1",
        "--numnull",
        "0",
        "--nprocs",
        "1",
        "--spatialfilt",
        "0",
        "--despecklepasses",
        "0",
        "--searchrange",
        "-3",
        "3",
        "--noprogressbar",
        "--outputlevel",
        "more",
        "--nonumba",
        "--nothresh",
    ]
    if not refinement:
        args += ["--norefinedelay", "--norefineoffset"]
    options, prefilter = process_args(args)
    rapidtide_main((options, prefilter))
    processed = zarr.open_group(output + "_desc-processed_mask.zarr", mode="r")
    np.testing.assert_array_equal(processed["data"][:], mask)
    initial = zarr.open_group(output + "_desc-initregressor_mask.zarr", mode="r")
    assert not initial["data"][0, 0]
    lags = zarr.open_group(output + "_desc-maxtime_map.zarr", mode="r")
    assert lags["data"].shape == (3, 5)
    assert np.isfinite(lags["data"][:]).all()
    assert lags["data"][2, 4] - lags["data"][0, 1] == pytest.approx(1.3, abs=0.3)
    cleaned = zarr.open_group(output + "_desc-lfofilterCleaned_bold.zarr", mode="r")
    assert cleaned["data"].shape == (3, 5, len(times))
    np.testing.assert_allclose(cleaned["timestamps"][:], times)
