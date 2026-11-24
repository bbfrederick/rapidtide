"""Native geometry, colormap, and end-to-end rendering checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from rapidtide.maprender import _lag_slice, _map_figure, _panels, render_rapidtide


@pytest.fixture
def result_root(tmp_path):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "example"
    mask = np.ones((6, 4), dtype=np.int16)
    mask[0, 0] = 0
    arrays = {
        "corrfit_mask": mask,
        "processed_mask": mask,
        "maxtime_map": np.linspace(-0.2, 0.2, 24).reshape(6, 4),
        "maxcorr_map": np.full((6, 4), 0.7),
        "maxtimerefined_map": np.linspace(-0.3, 0.3, 24).reshape(6, 4),
        "maxcorrrefined_map": np.full((6, 4), -0.2),
        "corrout_info": np.broadcast_to(np.linspace(-0.8, 0.8, 5), (6, 4, 5)).copy(),
    }
    for descriptor, data in arrays.items():
        group = zarr.open_group(f"{root}_desc-{descriptor}.zarr", mode="w", zarr_format=2)
        group.attrs["axes"] = ["y", "x"] + (["lag"] if data.ndim == 3 else [])
        group.create_array("data", data=data, chunks=tuple(min(n, 2) for n in data.shape))
        if data.ndim == 3:
            group.create_array("lag", data=np.linspace(-0.2, 0.2, 5))
    (tmp_path / "example_desc-runoptions_info.json").write_text(
        json.dumps({"currentstage": "done", "lagmin": -0.2, "lagmax": 0.2, "fmrifreq": 30})
    )
    return root


def test_map_colormaps_geometry_and_limits(result_root: Path) -> None:
    """Check panel colormaps, display limits, native geometry, and gray masking.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis.

    Returns
    -------
    None
    """
    panels, mask = _panels(result_root)
    figure = _map_figure(result_root, panels, {}, -0.35, 0.35)
    images = [ax.images[0] for ax in figure.axes if ax.images]
    assert [image.cmap.name for image in images] == ["viridis", "hot", "viridis", "hot"]
    assert [image.get_clim() for image in images] == [
        (-0.35, 0.35),
        (-1, 1),
        (-0.35, 0.35),
        (-1, 1),
    ]
    for image in images:
        assert image.origin == "upper"
        assert image.get_array().shape == (6, 4)
        assert image.get_array().mask[0, 0]
        np.testing.assert_allclose(image.cmap.get_bad(), [189 / 255] * 3 + [1])
    assert mask.sum() == 23
    figure.clear()


def test_render_all_and_overwrite(result_root: Path) -> None:
    """Render every artifact twice, checking the second run replaces the first.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis.

    Returns
    -------
    None
    """
    av = pytest.importorskip("av")
    for _ in range(2):
        paths = render_rapidtide(result_root, fps=10, showprogress=False)
        assert set(paths) == {
            "static-maps.png",
            "static-maps.pdf",
            "corrout-lag-sweep.mp4",
            "corrout-zero-lag.png",
            "render-info.json",
        }
        assert all(path.stat().st_size > 0 for path in paths.values())
        metadata = json.loads(paths["render-info.json"].read_text())
        assert metadata["shape_yx"] == [6, 4]
        assert metadata["correlation_colormap"] == "hot"
        assert metadata["correlation_display_limits"] == [-1, 1]
        assert metadata["movie"]["colormap"] == "hot"
        assert metadata["movie"]["frames"] == 5
        assert metadata["movie"]["display_limits"] == [-1, 1]
        np.testing.assert_allclose(metadata["movie"]["lag_seconds"], np.linspace(-0.2, 0.2, 5))
        with av.open(str(paths["corrout-lag-sweep.mp4"])) as container:
            assert container.streams.video[0].average_rate == 10
            assert "signed hot scale [-1, 1]" in container.metadata["comment"]
            frames = list(container.decode(video=0))
            assert len(frames) == 5
            np.testing.assert_allclose([float(frame.time) for frame in frames], np.arange(5) / 10)
            assert frames[0].width % 2 == frames[0].height % 2 == 0
            # Verify encoded colors, so a correct label cannot hide a wrong colormap.
            from matplotlib import colormaps

            rgb = frames[2].to_ndarray(format="rgb24")
            expected_color = np.asarray(colormaps["hot"](0.5)[:3]) * 255
            np.testing.assert_allclose(rgb[400, 200], expected_color, atol=8)


def test_video_lag_range_selects_matching_frames(result_root):
    av = pytest.importorskip("av")
    full = render_rapidtide(result_root, fps=10, showprogress=False)
    with av.open(str(full["corrout-lag-sweep.mp4"])) as container:
        original = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    paths = render_rapidtide(result_root, fps=10, showprogress=False, videolagrange=(-0.1, 0.1))
    info = json.loads(paths["render-info.json"].read_text())
    assert info["movie"]["requested_lag_range_seconds"] == [-0.1, 0.1]
    np.testing.assert_allclose(info["movie"]["lag_seconds"], [-0.1, 0, 0.1], atol=1e-12)
    assert info["movie"]["frames"] == 3
    assert info["movie"]["duration_seconds"] == 0.3
    with av.open(str(paths["corrout-lag-sweep.mp4"])) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 3
        np.testing.assert_allclose([float(frame.time) for frame in frames], [0, 0.1, 0.2])
        # A region inside the image must match the corresponding original lag,
        # including selection across source chunks and the final partial chunk.
        for frame, expected in zip(frames, original[1:4]):
            rgb = frame.to_ndarray(format="rgb24")
            np.testing.assert_allclose(rgb[400:410, 200:210], expected[400:410, 200:210], atol=5)


@pytest.mark.parametrize(
    "bounds,expected",
    [
        (None, [0, 1, 2, 3, 4]),
        ((-1, 0), [0, 1, 2]),
        ((0.1, 1), [3, 4]),
        ((0, 0), [2]),
        ((0, 1e20), [2, 3, 4]),
    ],
)
def test_lag_selection_boundaries(bounds, expected):
    lag = np.linspace(-0.2, 0.2, 5)
    np.testing.assert_array_equal(np.arange(len(lag))[_lag_slice(lag, bounds)], expected)


@pytest.mark.parametrize("bounds", [(1, 2), (-0.15, -0.14)])
def test_empty_video_range_preserves_outputs(result_root, tmp_path, bounds):
    output = tmp_path / "figures"
    output.mkdir()
    previous = output / "corrout-lag-sweep.mp4"
    previous.write_bytes(b"previous movie")
    with pytest.raises(ValueError, match="No saved corrout lags"):
        render_rapidtide(result_root, outdir=output, videolagrange=bounds)
    assert previous.read_bytes() == b"previous movie"
    assert list(output.iterdir()) == [previous]


@pytest.mark.parametrize("bounds", [(0.1, -0.1), (np.nan, 0.1), (0, np.inf), (0,), (0, 1, 2)])
def test_invalid_video_range(bounds):
    with pytest.raises(ValueError, match="videolagrange"):
        render_rapidtide("missing", videolagrange=bounds)


def test_missing_optional_outputs(result_root: Path, tmp_path: Path) -> None:
    """Render only the required maps when refined maps and corrout are absent.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis.
    tmp_path : pathlib.Path
        Pytest temporary directory for the figure output.

    Returns
    -------
    None
    """
    import shutil

    for descriptor in ["maxtimerefined_map", "maxcorrrefined_map", "corrout_info"]:
        shutil.rmtree(f"{result_root}_desc-{descriptor}.zarr")
    with pytest.warns(UserWarning, match="--savecorrout"):
        paths = render_rapidtide(result_root, outdir=tmp_path / "figures")
    assert set(paths) == {
        "static-maps.png",
        "static-maps.pdf",
        "render-info.json",
    }
    assert json.loads(paths["render-info.json"].read_text())["panels"] == [
        "Peak-fit lag",
        "Peak correlation",
    ]


@pytest.mark.parametrize("failure", ["geometry", "axes", "lag", "unfinished"])
def test_invalid_results_leave_existing_artifacts(
    result_root: Path, tmp_path: Path, failure: str
) -> None:
    """Leave a previous rendering in place when the analysis results are invalid.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis.
    tmp_path : pathlib.Path
        Pytest temporary directory holding the analysis outputs.
    failure : str
        Which defect to introduce: bad geometry, wrong axes, bad lag
        coordinates, or an unfinished analysis.

    Returns
    -------
    None
    """
    import zarr

    if failure == "unfinished":
        (tmp_path / "example_desc-runoptions_info.json").write_text(
            json.dumps({"currentstage": "prerefine_pass1"})
        )
    else:
        descriptor = "corrout_info" if failure == "lag" else "maxtime_map"
        group = zarr.open_group(f"{result_root}_desc-{descriptor}.zarr", mode="a")
        if failure == "geometry":
            del group["data"]
            group.create_array("data", data=np.zeros((4, 6)))
        elif failure == "axes":
            group.attrs["axes"] = ["x", "y"]
        else:
            group["lag"][:] = [0, 0.1, 0.1, 0.3, 0.4]
    output = tmp_path / "figures"
    output.mkdir()
    previous = output / "static-maps.png"
    previous.write_bytes(b"previous figure")
    with pytest.raises(ValueError):
        render_rapidtide(result_root, outdir=output, showprogress=False)
    assert previous.read_bytes() == b"previous figure"
    assert list(output.iterdir()) == [previous]


def test_nonfinite_map_pixels_are_excluded(result_root):
    import zarr

    group = zarr.open_group(f"{result_root}_desc-maxtime_map.zarr", mode="a")
    group["data"][2, 1] = np.nan
    with pytest.warns(UserWarning, match="Nonfinite"):
        panels, _ = _panels(result_root)
    assert not panels[0][2][2, 1]


def test_encoding_failure_preserves_previous_artifacts(
    result_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave previous artifacts untouched when movie encoding fails.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis.
    tmp_path : pathlib.Path
        Pytest temporary directory for the figure output.
    monkeypatch : pytest.MonkeyPatch
        Used to make movie encoding fail.

    Returns
    -------
    None
    """
    pytest.importorskip("av")
    import rapidtide.maprender as renderer

    output = tmp_path / "figures"
    output.mkdir()
    for name in ["static-maps.png", "corrout-lag-sweep.mp4"]:
        (output / name).write_bytes(b"previous artifact")

    def fail_encoding(*args, **kwargs):
        raise OSError("encoding failed")

    monkeypatch.setattr(renderer, "_movie", fail_encoding)
    with pytest.raises(OSError, match="encoding failed"):
        render_rapidtide(result_root, outdir=output, showprogress=False)
    assert len(list(output.iterdir())) == 2
    assert all(path.read_bytes() == b"previous artifact" for path in output.iterdir())


def test_output_directory_cannot_modify_analysis_store(result_root):
    output = f"{result_root}_desc-maxtime_map.zarr"
    with pytest.raises(ValueError, match="outside Zarr"):
        render_rapidtide(result_root, outdir=output, movie=False)


@pytest.mark.parametrize(
    "keyword,value",
    [("fps", 0), ("fps", np.nan), ("minlag", np.nan), ("maxlag", np.inf)],
)
def test_invalid_display_options(keyword: str, value: float) -> None:
    """Reject invalid display options before reading any analysis outputs.

    Parameters
    ----------
    keyword : str
        Name of the ``render_rapidtide`` option to set.
    value : float
        Invalid value for that option.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match=keyword):
        render_rapidtide("missing", **{keyword: value})


def test_cli(result_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Run the command-line workflow with explicit delay and correlation limits.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis.
    tmp_path : pathlib.Path
        Pytest temporary directory for the figure output.
    capsys : pytest.CaptureFixture
        Captures the printed artifact paths.

    Returns
    -------
    None
    """
    from rapidtide.workflows import parser_funcs as pf
    from rapidtide.workflows import showrapidtide as workflow

    pf.generic_init(
        workflow._get_parser,
        workflow.main,
        [
            str(result_root),
            "--outdir",
            str(tmp_path / "plots"),
            "--nomovie",
            "--minlag",
            "-0.5",
            "--maxlag",
            "0.5",
            "--mincorr",
            "0",
            "--maxcorr",
            "0.6",
        ],
    )
    assert "static-maps.png" in capsys.readouterr().out
    info = json.loads((tmp_path / "plots/render-info.json").read_text())
    assert info["lag_display_limits_seconds"] == [-0.5, 0.5]
    assert info["correlation_display_limits"] == [0, 0.6]
    assert "movie" not in info
    with pytest.raises(SystemExit):
        workflow._get_parser().parse_args([str(result_root), "--fps", "nan"])


def test_cli_video_range(result_root, tmp_path, capsys):
    pytest.importorskip("av")
    from rapidtide.workflows import parser_funcs as pf
    from rapidtide.workflows import showrapidtide as workflow

    pf.generic_init(
        workflow._get_parser,
        workflow.main,
        [
            str(result_root),
            "--outdir",
            str(tmp_path / "plots"),
            "--videolagrange",
            "0",
            "0",
            "--noprogressbar",
            "--mincorr",
            "-0.4",
            "--maxcorr",
            "0.8",
        ],
    )
    assert "corrout-lag-sweep.mp4" in capsys.readouterr().out
    info = json.loads((tmp_path / "plots/render-info.json").read_text())
    assert info["movie"]["frames"] == 1
    assert info["movie"]["requested_lag_range_seconds"] == [0, 0]
    assert info["movie"]["display_limits"] == [-0.4, 0.8]


@pytest.mark.parametrize("keep_legacy", [True, False])
def test_tiff_maps_and_movie(
    result_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_legacy: bool
) -> None:
    """Prefer new TIFF outputs over legacy stores, and stream selected lag pages.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis.
    tmp_path : pathlib.Path
        Pytest temporary directory holding the analysis outputs.
    monkeypatch : pytest.MonkeyPatch
        Used to capture every image the renderer draws.
    keep_legacy : bool
        Whether the legacy Zarr stores remain beside the new TIFF maps.

    Returns
    -------
    None
    """
    pytest.importorskip("av")
    pytest.importorskip("tifffile")
    import zarr
    import shutil
    import rapidtide.maprender as renderer
    from rapidtide.tiffio import save_map
    from rapidtide.zarrio import blood_header

    for path in tmp_path.glob("example_desc-*.zarr"):
        group = zarr.open_group(str(path), mode="r")
        native = group["data"][:]
        if native.ndim == 2:
            internal = native.T[..., None]
            axis, start, step = "time", 0, 1
        else:
            internal = native.transpose(1, 0, 2)[:, :, None, :]
            axis, start, step = "lag", float(group["lag"][0]), 0.1
        save_map(
            path.with_suffix(".tif"),
            internal,
            blood_header((6, 4, native.shape[-1] if native.ndim == 3 else 1), step, start),
            {},
            axis,
        )
    # A stale Zarr map must not override the new TIFF result.
    old = zarr.open_group(f"{result_root}_desc-maxtime_map.zarr", mode="a")
    old["data"][:] = 99
    if not keep_legacy:
        for path in tmp_path.glob("example_desc-*.zarr"):
            shutil.rmtree(path)
    images = []
    original_image = renderer._image

    def capture_image(*args: object, **kwargs: object) -> tuple:
        """Record the image drawn by ``_image`` and pass its result through.

        Parameters
        ----------
        *args, **kwargs : object
            Arguments forwarded to the original ``_image``.

        Returns
        -------
        tuple
            ``(image, minval, maxval)`` from the original ``_image``.
        """
        result = original_image(*args, **kwargs)
        images.append(result[0])
        return result

    monkeypatch.setattr(renderer, "_image", capture_image)
    paths = render_rapidtide(
        result_root, showprogress=False, videolagrange=(-0.1, 0.1), mincorr=0, maxcorr=0.6
    )
    info = json.loads(paths["render-info.json"].read_text())
    assert info["lag_display_limits_seconds"][1] < 1
    assert info["movie"]["frames"] == 3
    assert info["correlation_display_limits"] == info["movie"]["display_limits"] == [0, 0.6]
    assert [
        (image.cmap.name, image.get_clim()) for image in images if image.cmap.name != "viridis"
    ] == [("hot", (0, 0.6)), ("hot", (0, 0.6)), ("hot", (0, 0.6))]
    np.testing.assert_allclose(info["movie"]["lag_seconds"], [-0.1, 0, 0.1], atol=1e-12)


def test_custom_correlation_limits_leave_delays_and_values_unchanged(result_root: Path) -> None:
    """Apply correlation limits without changing delay limits or stored values.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis.

    Returns
    -------
    None
    """
    panels, _ = _panels(result_root)
    figure = _map_figure(result_root, panels, {}, -0.35, 0.35, mincorr=0.1, maxcorr=0.5)
    images = [ax.images[0] for ax in figure.axes if ax.images]
    assert [image.get_clim() for image in images] == [
        (-0.35, 0.35),
        (0.1, 0.5),
        (-0.35, 0.35),
        (0.1, 0.5),
    ]
    for image, (_, values, mask, *_) in zip(images, panels):
        np.testing.assert_array_equal(image.get_array().compressed(), values[mask])
    figure.clear()


@pytest.mark.parametrize(
    "minlag,maxlag,expected",
    [(None, None, [-0.3, 0.3]), (None, 1.0, [-0.3, 1.0]), (-1.0, None, [-1.0, 0.3])],
)
def test_default_lag_limits(
    result_root: Path,
    tmp_path: Path,
    minlag: float | None,
    maxlag: float | None,
    expected: list[float],
) -> None:
    """Fill any delay limit not given with the symmetric default covering all delays.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis; delays span +/-0.3 s.
    tmp_path : pathlib.Path
        Pytest temporary directory for the figure output.
    minlag, maxlag : float or None
        Requested delay limits; None selects the default.
    expected : list of float
        Expected ``lag_display_limits_seconds`` in the manifest.

    Returns
    -------
    None
    """
    paths = render_rapidtide(
        result_root, outdir=tmp_path / "figures", minlag=minlag, maxlag=maxlag, movie=False
    )
    info = json.loads(paths["render-info.json"].read_text())
    np.testing.assert_allclose(info["lag_display_limits_seconds"], expected)


@pytest.mark.parametrize("minlag,maxlag", [(0.5, None), (None, -0.5)])
def test_one_sided_lag_limit_must_exceed_default(
    result_root: Path, minlag: float | None, maxlag: float | None
) -> None:
    """Reject a single delay limit that crosses the default for the other side.

    Parameters
    ----------
    result_root : pathlib.Path
        Output prefix of the synthetic legacy Zarr analysis; delays span +/-0.3 s.
    minlag, maxlag : float or None
        One requested delay limit; the other takes the default.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="minlag must be less than maxlag"):
        render_rapidtide(result_root, minlag=minlag, maxlag=maxlag, movie=False)


@pytest.mark.parametrize(
    "minimum,maximum", [(0, 0), (1, -1), (np.nan, 1), (-1, np.inf), (-np.inf, 1), (-1, np.nan)]
)
def test_invalid_correlation_limits(minimum, maximum):
    with pytest.raises(ValueError, match="mincorr < maxcorr"):
        render_rapidtide("missing", mincorr=minimum, maxcorr=maximum)
