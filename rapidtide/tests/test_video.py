"""Scientific and file-level validation of offline RGB-to-blood extraction."""

from fractions import Fraction
import subprocess
import sys
import textwrap
import numpy as np
import pytest

from rapidtide.video import RGBBloodProjector, _timing, _video_frames, video_to_blood


def _rgb_series():
    """Distinct skin colors, channel covariances, spatial phases, and slow variation."""
    time = np.arange(180, dtype=float) / 30
    phase = np.arange(6).reshape(2, 3) * 0.17
    slow = np.sin(2 * np.pi * time[:, None, None] / 6 + phase)
    pulse = np.sin(2 * np.pi * 1.2 * time[:, None, None] + phase)
    rgb = np.empty((len(time), 2, 3, 3))
    rgb[..., 0] = (100 + 2 * slow + pulse) * np.linspace(0.7, 1.4, 6).reshape(2, 3)
    rgb[..., 1] = 140 + 5 * slow + 3 * pulse
    rgb[..., 2] = 80 + slow - 2 * pulse
    return rgb


def _teleplethy_reference(samples, method):
    """Direct whole-window formulas, independent of the streaming implementation."""
    mean = samples.mean(axis=0)
    normalized = samples / mean
    r, g, b = normalized.T
    if method == "grgb":
        signal = g / r + g / b
        return 100 * (signal - signal.mean())
    if method == "chrom":
        x, y = 3 * r - 2 * g, 1.5 * r + g - 1.5 * b
        alpha = x.std() / y.std()
        return -100 * ((x - x.mean()) - alpha * (y - y.mean())) / np.hypot(1, alpha)
    normal = mean / np.linalg.norm(mean)
    seed = np.array([0, 1, -1]) / np.sqrt(2)
    first = seed - np.dot(seed, normal) * normal
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    x, y = normalized @ first, normalized @ second
    alpha = x.std() / y.std()
    return 100 * ((x - x.mean()) + alpha * (y - y.mean())) / np.hypot(1, alpha)


@pytest.mark.parametrize("method", ["grgb", "pos", "chrom"])
def test_teleplethy_agreement(method):
    rgb = _rgb_series()
    projector = RGBBloodProjector(method)
    for frame in rgb:
        projector.update(frame)
    projector.finalize()
    result = np.stack([projector.transform(frame) for frame in rgb])
    assert result.dtype == np.float32
    assert projector.valid.all()
    np.testing.assert_allclose(result.mean(axis=0, dtype=np.float64), 0, atol=2e-7)
    for y, x in np.ndindex(2, 3):
        np.testing.assert_allclose(
            result[:, y, x], _teleplethy_reference(rgb[:, y, x], method), rtol=2e-6, atol=2e-7
        )


@pytest.mark.parametrize("method", ["grgb", "pos", "chrom"])
def test_no_short_window_or_causal_effect(method):
    """Reversing time must reverse the waveform, without changing its values."""
    rgb = _rgb_series()
    results = []
    for frames in (rgb, rgb[::-1]):
        projector = RGBBloodProjector(method)
        for frame in frames:
            projector.update(frame)
        projector.finalize()
        results.append(np.stack([projector.transform(frame) for frame in frames]))
    np.testing.assert_allclose(results[0], results[1][::-1], atol=2e-7)


@pytest.mark.parametrize("method", ["grgb", "pos", "chrom"])
def test_invalid_pixels(method):
    rgb = _rgb_series()
    rgb[:, 0, 0] = 0  # Black for the entire recording.
    rgb[:, 0, 1] = [100, 100, 100]  # Constant: valid zero GRGB, undefined POS/CHROM.
    rgb[5, 1, 0, 0] = np.nan
    rgb[6, 1, 1, 0] = -1
    if method == "grgb":
        rgb[7, 0, 2, 2] = 0  # One zero denominator invalidates the whole time course.
    projector = RGBBloodProjector(method)
    for frame in rgb:
        projector.update(frame)
    projector.finalize()
    result = np.stack([projector.transform(frame) for frame in rgb])
    assert not projector.valid[0, 0]
    assert not projector.valid[1, 0]
    assert not projector.valid[1, 1]
    assert projector.valid[1, 2]
    assert projector.valid[0, 1] == (method == "grgb")
    if method == "grgb":
        assert not projector.valid[0, 2]
        np.testing.assert_array_equal(result[:, 0, 1], 0)
    assert np.isnan(result[:, ~projector.valid]).all()
    assert np.isfinite(result[:, projector.valid]).all()


def test_projector_validation():
    with pytest.raises(ValueError, match="method"):
        RGBBloodProjector("green")
    projector = RGBBloodProjector()
    with pytest.raises(ValueError, match="At least two"):
        projector.finalize()
    with pytest.raises(ValueError, match="Finalize"):
        projector.transform(np.ones((2, 3, 3)))
    with pytest.raises(ValueError, match="shape"):
        projector.update(np.ones((3, 3)))
    projector.update(np.ones((2, 3, 3)))
    with pytest.raises(ValueError, match="dimensions changed"):
        projector.update(np.ones((3, 3, 3)))
    projector.update(np.ones((2, 3, 3)))
    projector.finalize()
    with pytest.raises(ValueError, match="finalized"):
        projector.update(np.ones((2, 3, 3)))


def test_slow_signal_and_spatial_delays():
    """A 0.05 Hz signal retains its amplitude and known spatial phase offsets."""
    times = np.arange(1800) / 30
    delays = np.array([[0, 1, 3]])
    wave = np.sin(2 * np.pi * 0.05 * (times[:, None, None] - delays))
    rgb = np.empty((len(times), 1, 3, 3))
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 100, 140 * (1 + 0.01 * wave), 80
    projector = RGBBloodProjector("grgb")
    for frame in rgb:
        projector.update(frame)
    projector.finalize()
    result = np.stack([projector.transform(frame) for frame in rgb])
    np.testing.assert_allclose(result, 2 * wave, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("method", ["pos", "chrom"])
def test_small_signal_covariance(method):
    """Weak fluctuations must survive covariance accumulation against a large DC."""
    rng = np.random.default_rng(123)
    rgb = np.array([100, 140, 80]) + rng.normal(size=(600, 1, 1, 3)) * 0.0001
    projector = RGBBloodProjector(method)
    for frame in rgb:
        projector.update(frame)
    projector.finalize()
    assert projector.valid.all()
    signal = np.array([projector.transform(frame)[0, 0] for frame in rgb])
    np.testing.assert_allclose(
        signal, _teleplethy_reference(rgb[:, 0, 0], method), rtol=2e-5, atol=2e-10
    )


def test_timing():
    assert _timing(np.array([2, 2.1, 2.2]))["uniform_sampling"]
    irregular = _timing(np.array([2, 2.1, 2.3]))
    assert not irregular["uniform_sampling"]
    assert irregular["sampling_interval_seconds"] is None
    for times in ([0, 0], [1, 0], [0, np.nan]):
        with pytest.raises(ValueError, match="timestamps"):
            _timing(np.array(times))


def _write_video(filename, rgb):
    """Lossless RGB FFV1 fixture; no RGB/YUV conversion in the encoder."""
    av = pytest.importorskip("av")
    with av.open(str(filename), "w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height = rgb.shape[2], rgb.shape[1]
        stream.pix_fmt = "bgr0"
        for index, frame in enumerate(rgb):
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = index
            video_frame.time_base = Fraction(1, 30)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("method", ["grgb", "pos", "chrom"])
def test_video_zarr_roundtrip(tmp_path, method):
    zarr = pytest.importorskip("zarr")
    rgb = np.rint(_rgb_series()[:35]).astype(np.uint8)
    rgb[:, 0, 0] = 0
    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    _write_video(source, rgb)
    decoded = list(_video_frames(source))
    # Verify decoder color order, orientation, and intensity ratios.
    decoded_rgb = np.stack([frame for frame, _ in decoded])
    np.testing.assert_allclose(decoded_rgb / 257, rgb, atol=1)
    video_to_blood(source, output, method=method, chunks=(1, 2, 8), showprogress=False)
    group = zarr.open_group(str(output), mode="r")
    blood = group["blood"][:]
    times = group["timestamps"][:]
    assert blood.shape == (2, 3, len(times))
    assert blood.dtype == np.float32
    assert group["blood"].chunks == (1, 2, 8)
    assert not group["valid_mask"][0, 0]
    assert np.isnan(blood[0, 0]).all()
    np.testing.assert_allclose(blood[1, 2].mean(), 0, atol=2e-7)
    interpolated = np.column_stack(
        [
            np.interp(times, [time for _, time in decoded], decoded_rgb[:, 1, 2, channel])
            for channel in range(3)
        ]
    )
    np.testing.assert_allclose(
        blood[1, 2], _teleplethy_reference(interpolated, method), atol=2e-7, rtol=2e-6
    )
    np.testing.assert_array_equal(group["source_timestamps"][:], [time for _, time in decoded])
    np.testing.assert_allclose(np.diff(times), 1 / 30)
    assert group.attrs["method"] == method
    assert group.attrs["statistics"] == "whole_recording_per_pixel"
    assert group.attrs["temporal_filter"] is None
    assert group.attrs["axes"] == ["y", "x", "time"]
    assert group.attrs["timestamp_source"] == "regular_grid_from_presentation"
    with pytest.raises(FileExistsError):
        video_to_blood(source, output)


def test_framerate_override(tmp_path):
    zarr = pytest.importorskip("zarr")
    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    _write_video(source, np.rint(_rgb_series()[:17]).astype(np.uint8))
    video_to_blood(source, output, framerate=20, showprogress=False)
    group = zarr.open_group(str(output), mode="r")
    np.testing.assert_allclose(group["timestamps"][:], np.arange(11) / 20)
    assert group["source_timestamps"].shape == (17,)
    assert group.attrs["resampled"] is True
    assert group.attrs["requested_framerate_hz"] == 20
    assert group.attrs["sampling_interval_seconds"] == pytest.approx(0.05)


def test_failure_cleanup(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    pytest.importorskip("av")
    import rapidtide.video as video

    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    source.touch()
    frames = _rgb_series()[:9]
    passes = []

    def changing_video(*args):
        passes.append(True)
        length = len(frames) if len(passes) <= 2 else len(frames) - 1
        yield from zip(frames[:length], np.arange(length) / 30)

    monkeypatch.setattr(video, "_video_frames", changing_video)
    with pytest.raises(ValueError, match="frame count changed"):
        video_to_blood(source, output, framerate=30, showprogress=False)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("chunks", [(0, 1, 1), (1, -1, 1), (1, 1), (1, 1, 1.5)])
def test_bad_chunks(chunks):
    with pytest.raises(ValueError, match="chunks"):
        video_to_blood("missing.mp4", "blood.zarr", chunks=chunks)


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_bad_framerate(rate):
    with pytest.raises(ValueError, match="framerate"):
        video_to_blood("missing.mp4", "blood.zarr", framerate=rate)


@pytest.mark.parametrize("method", ["grgb", "pos", "chrom"])
def test_missing_frames_regularized_before_projection(tmp_path, monkeypatch, method):
    """Recover missing RGB frames on a known grid before nonlinear projection/statistics."""
    pytest.importorskip("av")
    zarr = pytest.importorskip("zarr")
    import rapidtide.video as video
    from rapidtide.zarrio import open_blood_zarr

    source, output = tmp_path / "input.mp4", tmp_path / "regular.zarr"
    source.touch()
    source_times = np.array([4.0, 4.1, 4.3, 4.5, 4.6])
    rgb = _rgb_series()[[0, 3, 9, 15, 18]]
    monkeypatch.setattr(video, "_video_frames", lambda *args: iter(zip(rgb, source_times)))
    video_to_blood(source, output, method=method, framerate=10, showprogress=False)
    group, target_times, mask, step = open_blood_zarr(str(output))
    np.testing.assert_allclose(target_times, 4 + np.arange(7) / 10)
    assert step == pytest.approx(0.1)
    expected_rgb = np.column_stack(
        [np.interp(target_times, source_times, rgb[:, 1, 2, c]) for c in range(3)]
    )
    np.testing.assert_allclose(
        group["blood"][1, 2], _teleplethy_reference(expected_rgb, method), rtol=2e-6, atol=2e-7
    )
    assert group.attrs["resampled"]
    assert group.attrs["interpolation"] == "linear_rgb_before_projection"
    np.testing.assert_array_equal(group["source_timestamps"][:], source_times)


def test_cli(tmp_path, capsys):
    from rapidtide.workflows import parser_funcs as pf
    from rapidtide.workflows import videotoblood as workflow

    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    _write_video(source, np.rint(_rgb_series()[:17]).astype(np.uint8))
    pytest.importorskip("zarr")
    pf.generic_init(
        workflow._get_parser,
        workflow.main,
        [str(source), str(output), "--method", "CHROM", "--noprogressbar"],
    )
    assert output.is_dir()
    assert "Saved CHROM" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        workflow._get_parser().parse_args([str(source), "other.zarr", "--chunks", "0", "1", "1"])


@pytest.mark.parametrize("program", ["videotoblood", "rapidtide"])
def test_cli_without_installed_tests(program):
    """Start without test helpers or a package-level version (namespace installs)."""
    code = textwrap.dedent("""
        import runpy
        import sys

        # Simulate a runtime installation that does not include the test package.
        sys.modules['rapidtide.tests'] = None
        sys.modules['rapidtide.tests.utils'] = None
        import rapidtide
        # Some installed layouts expose a namespace package with no __version__.
        if hasattr(rapidtide, '__version__'):
            del rapidtide.__version__
        import numpy as np
        import rapidtide.io as tide_io

        first = np.array([1.0, 2.0, 3.0])
        second = first + np.array([0.1, 0.2, 0.3])
        assert tide_io.checkdatamatch(first, second, absthresh=0.31, msethresh=0.04) == (
            False, True
        )
        assert tide_io.checkdatamatch(first, second, absthresh=0.31, msethresh=0.05) == (
            True, True
        )
        sys.argv = ['PROGRAM', '--help']
        runpy.run_module('rapidtide.scripts.PROGRAM', run_name='__main__')
        """)
    code = code.replace("PROGRAM", program)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert f"usage: {program}" in result.stdout
