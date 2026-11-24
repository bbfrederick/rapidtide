"""Scientific and file-level validation of offline RGB-to-blood extraction."""

from fractions import Fraction
import subprocess
import sys
import textwrap
import numpy as np
import pytest

from rapidtide.video import (
    RGBBloodProjector,
    _spatial_frame,
    _timing,
    _video_frames,
    video_to_blood,
)


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


def _spatial_reference(frame, downsamp, lpf):
    """Independent plane filtering and explicit per-block means."""
    from scipy.ndimage import gaussian_filter

    frame = frame.astype(np.float64)
    if lpf:
        frame = np.stack(
            [gaussian_filter(frame[..., c], sigma=lpf, mode="reflect") for c in range(3)],
            axis=-1,
        )
    height, width = frame.shape[0] // downsamp, frame.shape[1] // downsamp
    ystart = (frame.shape[0] - height * downsamp) // 2
    xstart = (frame.shape[1] - width * downsamp) // 2
    result = np.empty((height, width, 3))
    for y, x in np.ndindex(height, width):
        result[y, x] = frame[
            ystart + y * downsamp : ystart + (y + 1) * downsamp,
            xstart + x * downsamp : xstart + (x + 1) * downsamp,
        ].mean(axis=(0, 1))
    return result


@pytest.mark.parametrize(
    "shape,factor,start,retained",
    [
        ((8, 12), 4, (0, 0), (8, 12)),  # No cropping.
        ((5, 7), 2, (0, 0), (4, 6)),  # One pixel, high side only.
        ((10, 14), 4, (1, 1), (8, 12)),  # Even remainders, both sides equal.
        ((11, 14), 4, (1, 1), (8, 12)),  # Odd/even remainders on different axes.
        ((7, 9), 4, (1, 0), (4, 8)),  # Unequal remainders on the two axes.
    ],
)
def test_spatial_frame_block_means_and_truncation(shape, factor, start, retained):
    frame = np.arange(shape[0] * shape[1] * 3, dtype=np.uint16).reshape(*shape, 3)
    original = frame.copy()
    result = _spatial_frame(frame, downsamp=factor)
    assert result.shape == (retained[0] // factor, retained[1] // factor, 3)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, _spatial_reference(frame, factor, 0))
    np.testing.assert_array_equal(
        result[0, 0],
        frame[start[0] : start[0] + factor, start[1] : start[1] + factor].mean(axis=(0, 1)),
    )
    cropped_out = np.ones(shape, dtype=bool)
    cropped_out[start[0] : start[0] + retained[0], start[1] : start[1] + retained[1]] = False
    changed_edges = frame.copy()
    changed_edges[cropped_out] = 60000
    np.testing.assert_array_equal(_spatial_frame(changed_edges, downsamp=factor), result)
    np.testing.assert_array_equal(frame, original)
    np.testing.assert_array_equal(_spatial_frame(frame), frame)


@pytest.mark.parametrize("edge", [(0, 0), (-1, -1)])
def test_spatial_frame_gaussian_before_crop_and_average(edge):
    """An edge impulse must spread before cropping, without mixing RGB planes."""
    frame = np.zeros((7, 11, 3), dtype=np.uint16)
    frame[edge[0], edge[1], 0] = 1000  # Discarded by centered cropping, after filtering.
    frame[..., 1] = 100
    result = _spatial_frame(frame, downsamp=4, lpf=0.8)
    np.testing.assert_allclose(result, _spatial_reference(frame, 4, 0.8))
    assert result[..., 0].max() > 0
    assert np.any(result[..., 0] != np.floor(result[..., 0]))
    np.testing.assert_allclose(result[..., 1], 100)
    np.testing.assert_array_equal(result[..., 2], 0)
    np.testing.assert_allclose(_spatial_frame(frame, lpf=0.8), _spatial_reference(frame, 1, 0.8))


@pytest.mark.parametrize("downsamp", [0, -1, 1.5, 2.0, True, np.nan, np.inf])
def test_bad_downsamp(downsamp):
    with pytest.raises(ValueError, match="downsamp"):
        video_to_blood("missing.mp4", "blood.zarr", downsamp=downsamp)


@pytest.mark.parametrize("lpf", [-1, np.nan, np.inf])
def test_bad_lpf(lpf):
    with pytest.raises(ValueError, match="lpf"):
        video_to_blood("missing.mp4", "blood.zarr", lpf=lpf)


def test_oversized_downsamp_fails_early(tmp_path, monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("zarr")
    import rapidtide.video as video

    source, output = tmp_path / "rgb.mp4", tmp_path / "blood.zarr"
    source.touch()
    monkeypatch.setattr(video, "_video_frames", lambda *args: iter([(np.ones((2, 3, 3)), 0)]))
    with pytest.raises(ValueError, match="either input frame dimension"):
        video_to_blood(source, output, downsamp=3, showprogress=False)
    assert list(tmp_path.iterdir()) == [source]


def test_spatial_preprocessing_rejects_hidden_dimension_changes(tmp_path, monkeypatch):
    """Cropping must not hide source-size changes that produce equal output sizes."""
    pytest.importorskip("av")
    pytest.importorskip("zarr")
    import rapidtide.video as video

    source, output = tmp_path / "rgb.mp4", tmp_path / "blood.zarr"
    source.touch()
    frames = [(np.ones((4, 6, 3)), 0), (np.ones((5, 7, 3)), 0.1)]
    monkeypatch.setattr(video, "_video_frames", lambda *args: iter(frames))
    with pytest.raises(ValueError, match="dimensions changed"):
        video_to_blood(source, output, downsamp=2, framerate=10, showprogress=False)
    assert list(tmp_path.iterdir()) == [source]


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
@pytest.mark.parametrize("downsamp,lpf", [(1, 0), (1, 0.8), (2, 0), (2, 0.8), (3, 0), (4, 0.8)])
def test_spatial_preprocessing_zarr_roundtrip(tmp_path, method, downsamp, lpf):
    """Validate spatial RGB preprocessing against direct blood formulas after decoding."""
    av = pytest.importorskip("av")
    zarr = pytest.importorskip("zarr")
    from rapidtide.zarrio import open_blood_zarr

    rng = np.random.default_rng(834)
    rgb = rng.integers(40, 200, size=(18, 5, 7, 3), dtype=np.uint8)
    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    _write_video(source, rgb)
    decoded = list(_video_frames(source))
    source_times = [time for _, time in decoded]
    expected_rgb = np.stack([_spatial_reference(frame, downsamp, lpf) for frame, _ in decoded])
    movie = tmp_path / "blood.mp4" if (method, downsamp, lpf) == ("grgb", 2, 0.8) else None
    video_to_blood(
        source,
        output,
        method=method,
        downsamp=downsamp,
        lpf=lpf,
        showprogress=False,
        bloodvideo=movie,
    )
    group, times, mask, step = open_blood_zarr(str(output))
    result = group["blood"][:]
    assert result.shape == (5 // downsamp, 7 // downsamp, len(times))
    assert result.dtype == np.float32
    assert mask.all()
    assert step == pytest.approx(1 / 30)
    np.testing.assert_array_equal(group["source_timestamps"][:], source_times)
    for y, x in np.ndindex(result.shape[:2]):
        interpolated = np.column_stack(
            [np.interp(times, source_times, expected_rgb[:, y, x, c]) for c in range(3)]
        )
        np.testing.assert_allclose(
            result[y, x], _teleplethy_reference(interpolated, method), rtol=3e-6, atol=3e-6
        )
    preprocessing = group.attrs["spatial_preprocessing"]
    assert preprocessing["source_shape"] == [5, 7]
    assert preprocessing["cropped_source_shape"] == [5 - 5 % downsamp, 7 - 7 % downsamp]
    assert preprocessing["crop_low_side_source_pixels"] == [
        (5 % downsamp) // 2,
        (7 % downsamp) // 2,
    ]
    assert preprocessing["crop_high_side_source_pixels"] == [
        5 % downsamp - (5 % downsamp) // 2,
        7 % downsamp - (7 % downsamp) // 2,
    ]
    assert preprocessing["gaussian_sigma_source_pixels"] == lpf
    assert preprocessing["gaussian_mode"] == "reflect"
    assert preprocessing["downsample_factor"] == downsamp
    assert preprocessing["downsample_method"] == "block_mean"
    assert preprocessing["output_pixel_size_source_pixels"] == [downsamp, downsamp]
    assert (
        preprocessing["output_pixel_origin_source_pixels"]
        == {1: [0, 0], 2: [0.5, 0.5], 3: [2, 1], 4: [1.5, 2.5]}[downsamp]
    )
    assert preprocessing["order"] == ["gaussian_lowpass", "crop", "block_mean"]
    if movie is not None:
        with av.open(str(movie)) as container:
            assert (container.streams.video[0].height, container.streams.video[0].width) == (2, 4)
            assert len(list(container.decode(video=0))) == len(times)


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
    (output / "obsolete.txt").write_text("old output")
    video_to_blood(source, output, method=method, showprogress=False)
    assert not (output / "obsolete.txt").exists()
    np.testing.assert_allclose(zarr.open_group(str(output), mode="r")["blood"][:], blood)


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


@pytest.mark.parametrize("existing", [False, True])
def test_failure_cleanup(tmp_path, monkeypatch, existing):
    pytest.importorskip("zarr")
    pytest.importorskip("av")
    import rapidtide.video as video

    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    source.touch()
    if existing:
        output.mkdir()
        (output / "previous.txt").write_text("previous output")
    frames = _rgb_series()[:9]
    passes = []

    def changing_video(*args):
        passes.append(True)
        length = len(frames) if len(passes) <= 2 else len(frames) - 1
        yield from zip(frames[:length], np.arange(length) / 30)

    monkeypatch.setattr(video, "_video_frames", changing_video)
    with pytest.raises(ValueError, match="frame count changed"):
        video_to_blood(source, output, framerate=30, showprogress=False)
    if existing:
        assert (output / "previous.txt").read_text() == "previous output"
        assert set(tmp_path.iterdir()) == {source, output}
    else:
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


def test_gray_frame_mapping():
    from rapidtide.video import _gray_frame

    frame = np.array([[0.0, 2.0, -2.0], [5.0, -5.0, np.nan], [1.0, -1.0, 0.5]])
    gray = _gray_frame(frame, 2.0)
    assert gray.dtype == np.uint8
    assert gray.shape == (4, 4)
    np.testing.assert_array_equal(gray[:3, :3], [[128, 255, 1], [255, 1, 0], [192, 64, 160]])
    assert not gray[3].any() and not gray[:, 3].any()


def test_blood_video(tmp_path):
    av = pytest.importorskip("av")
    zarr = pytest.importorskip("zarr")
    from rapidtide.video import write_blood_video

    rgb = np.rint(_rgb_series()[:35]).astype(np.uint8)
    rgb[:, 0, 0] = 0
    source, output, movie = tmp_path / "rgb.mkv", tmp_path / "blood.zarr", tmp_path / "blood.mp4"
    _write_video(source, rgb)
    video_to_blood(source, output, chunks=(1, 2, 8), showprogress=False, bloodvideo=str(movie))
    blood = zarr.open_group(str(output), mode="r")["blood"][:]
    with av.open(str(movie)) as container:
        stream = container.streams.video[0]
        assert (stream.height, stream.width) == (2, 4)
        assert stream.average_rate == 30
        assert "gray 128 = 0" in container.metadata["comment"]
        frames = np.stack([f.to_ndarray(format="gray") for f in container.decode(video=0)])
    assert frames.shape == (blood.shape[-1], 2, 4)
    assert (frames[:, 0, 0] < 8).all()
    valid = frames[:, 1, 2].astype(float)
    assert np.corrcoef(valid, blood[1, 2])[0, 1] > 0.95
    movie.write_bytes(b"obsolete movie")
    write_blood_video(output, movie, showprogress=False)
    with av.open(str(movie)) as container:
        assert len(list(container.decode(video=0))) == blood.shape[-1]
    with pytest.raises(ValueError, match="limit"):
        write_blood_video(output, tmp_path / "other.mp4", limit=0)
    assert not (tmp_path / "other.mp4").exists()


def test_existing_outputs_are_replaced(tmp_path):
    av = pytest.importorskip("av")
    zarr = pytest.importorskip("zarr")
    source, output, movie = tmp_path / "rgb.mkv", tmp_path / "blood.zarr", tmp_path / "blood.mp4"
    _write_video(source, np.rint(_rgb_series()[:35]).astype(np.uint8))
    video_to_blood(source, output, bloodvideo=str(movie), showprogress=False)
    (output / "obsolete.txt").write_text("old output")
    # Change recording length and projection to detect stale chunks or metadata.
    _write_video(source, np.rint(_rgb_series()[:17]).astype(np.uint8))
    video_to_blood(source, output, method="chrom", bloodvideo=str(movie), showprogress=False)
    group = zarr.open_group(str(output), mode="r")
    assert group.attrs["method"] == "chrom"
    assert not (output / "obsolete.txt").exists()
    with av.open(str(movie)) as container:
        assert len(list(container.decode(video=0))) == group["blood"].shape[-1] < 35


def test_blood_video_failure_preserves_existing(tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    pytest.importorskip("zarr")
    from rapidtide.video import write_blood_video

    source, output, movie = tmp_path / "rgb.mkv", tmp_path / "blood.zarr", tmp_path / "blood.mp4"
    _write_video(source, np.rint(_rgb_series()[:17]).astype(np.uint8))
    video_to_blood(source, output, showprogress=False)
    movie.write_bytes(b"previous movie")

    def failed_open(*args, **kwargs):
        raise OSError("encoding failed")

    monkeypatch.setattr(av, "open", failed_open)
    with pytest.raises(OSError, match="encoding failed"):
        write_blood_video(output, movie, showprogress=False)
    assert movie.read_bytes() == b"previous movie"
    assert set(tmp_path.iterdir()) == {source, output, movie}


@pytest.mark.parametrize("collision", ["input_store", "input_movie", "store_movie", "nested"])
def test_output_paths_must_not_overlap(tmp_path, collision):
    source, output, movie = tmp_path / "rgb.mkv", tmp_path / "blood.zarr", tmp_path / "blood.mp4"
    source.write_bytes(b"input")
    if collision == "input_store":
        output = source
    elif collision == "input_movie":
        movie = source
    elif collision == "store_movie":
        movie = output
    else:
        movie = output / "blood.mp4"
    with pytest.raises(ValueError, match="output"):
        video_to_blood(source, output, bloodvideo=str(movie), showprogress=False)
    assert source.read_bytes() == b"input"


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


def test_cli_spatial_options(tmp_path, capsys):
    from rapidtide.workflows import parser_funcs as pf
    from rapidtide.workflows import videotoblood as workflow

    zarr = pytest.importorskip("zarr")
    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    _write_video(source, np.rint(_rgb_series()[:17]).astype(np.uint8))
    pf.generic_init(
        workflow._get_parser,
        workflow.main,
        [str(source), str(output), "--downsamp", "2", "--lpf", "0.8", "--noprogressbar"],
    )
    assert zarr.open_group(str(output), mode="r")["blood"].shape[:2] == (1, 1)
    assert "Saved GRGB" in capsys.readouterr().out
    for option, value in [
        ("--downsamp", "0"),
        ("--downsamp", "1.5"),
        ("--lpf", "-1"),
        ("--lpf", "nan"),
        ("--lpf", "inf"),
    ]:
        with pytest.raises(SystemExit):
            workflow._get_parser().parse_args([str(source), "other.zarr", option, value])


def _fake_skin_segmenter(monkeypatch, mask_function, model_path):
    import rapidtide.skinmask as skinmask

    passes = []
    metadata = {
        "backend": "test_multiclass",
        "included_classes": ["face-skin", "body-skin"],
        "landmark_refinement": False,
    }
    monkeypatch.setattr(skinmask, "prepare_skin_model", lambda filename: (model_path, metadata))

    class Segmenter:
        def __enter__(self):
            self.times = []
            passes.append(self)
            self.closed = False
            return self

        def __init__(self, model):
            assert model == model_path

        def __exit__(self, *args):
            self.closed = True

        def mask(self, rgb, timestamp):
            self.times.append(timestamp)
            return mask_function(rgb, timestamp)

    monkeypatch.setattr(skinmask, "SkinSegmenter", Segmenter)
    return passes


@pytest.mark.parametrize("method", ["grgb", "pos", "chrom"])
@pytest.mark.parametrize("downsamp,lpf", [(1, 0), (1, 0.8), (2, 0), (2, 0.8)])
def test_skinmask_after_filter_before_downsampling(tmp_path, monkeypatch, method, downsamp, lpf):
    zarr = pytest.importorskip("zarr")
    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    rng = np.random.default_rng(293)
    _write_video(source, rng.integers(40, 200, size=(18, 5, 7, 3), dtype=np.uint8))
    decoded = list(_video_frames(source))
    selected = np.zeros((5, 7), dtype=bool)
    selected[1:4, 1:3] = True  # body skin
    selected[:3, 4:6] = True  # face skin
    original_frames = {time: rgb for rgb, time in decoded}

    def infer_mask(rgb, timestamp):
        np.testing.assert_array_equal(rgb, original_frames[timestamp])
        assert rgb.dtype == original_frames[timestamp].dtype
        return selected

    passes = _fake_skin_segmenter(monkeypatch, infer_mask, tmp_path / "model.tflite")
    video_to_blood(
        source,
        output,
        method=method,
        skinmask=True,
        downsamp=downsamp,
        lpf=lpf,
        showprogress=False,
    )
    group = zarr.open_group(str(output), mode="r")
    times, valid, blood = group["timestamps"][:], group["valid_mask"][:], group["blood"][:]
    expected_frames = np.stack(
        [
            _spatial_reference(
                np.where(selected[..., None], _spatial_reference(rgb, 1, lpf), 0), downsamp, 0
            )
            for rgb, _ in decoded
        ]
    )
    source_times = [time for _, time in decoded]
    for y, x in np.ndindex(blood.shape[:2]):
        rgb = np.column_stack(
            [np.interp(times, source_times, expected_frames[:, y, x, c]) for c in range(3)]
        )
        if not rgb.any():
            assert not valid[y, x]
            assert np.isnan(blood[y, x]).all()
        else:
            assert valid[y, x]
            np.testing.assert_allclose(
                blood[y, x], _teleplethy_reference(rgb, method), rtol=3e-6, atol=3e-6
            )
    assert len(passes) == 2 and all(p.closed for p in passes)
    assert passes[0].times == passes[1].times == source_times
    assert group.attrs["skin_masking"]["included_classes"] == ["face-skin", "body-skin"]
    assert group.attrs["skin_masking"]["landmark_refinement"] is False
    assert group.attrs["spatial_preprocessing"]["order"] == [
        "skin_mask_inference",
        "gaussian_lowpass",
        "skin_mask",
        "crop",
        "block_mean",
    ]


@pytest.mark.parametrize("method", ["grgb", "pos", "chrom"])
def test_skinmask_before_missing_frame_interpolation(tmp_path, monkeypatch, method):
    pytest.importorskip("av")
    zarr = pytest.importorskip("zarr")
    import rapidtide.video as video

    source, output = tmp_path / "rgb.mp4", tmp_path / "blood.zarr"
    source.touch()
    source_times = np.array([4.0, 4.1, 4.3, 4.5, 4.6])
    frames = _rgb_series()[[0, 3, 9, 15, 18]]
    monkeypatch.setattr(video, "_video_frames", lambda *args: iter(zip(frames, source_times)))

    def selected(rgb, timestamp):
        mask = np.ones((2, 3), dtype=bool)
        mask[0, 0] = timestamp < 4.2
        mask[0, 1] = timestamp > 4.2
        return mask

    _fake_skin_segmenter(monkeypatch, selected, tmp_path / "model.tflite")
    video_to_blood(
        source, output, skinmask=True, method=method, lpf=0.8, framerate=10, showprogress=False
    )
    group = zarr.open_group(str(output), mode="r")
    times = group["timestamps"][:]
    preprocessed = np.stack(
        [
            np.where(selected(rgb, timestamp)[..., None], _spatial_reference(rgb, 1, 0.8), 0)
            for rgb, timestamp in zip(frames, source_times)
        ]
    )
    for y, x in np.ndindex(2, 3):
        expected = np.column_stack(
            [np.interp(times, source_times, preprocessed[:, y, x, c]) for c in range(3)]
        )
        if method == "grgb" and np.any(expected == 0):
            assert not group["valid_mask"][y, x]
            assert np.isnan(group["blood"][y, x]).all()
        else:
            np.testing.assert_allclose(
                group["blood"][y, x], _teleplethy_reference(expected, method), rtol=3e-6, atol=3e-6
            )


def test_skinmask_is_optional(tmp_path, monkeypatch):
    import rapidtide.skinmask as skinmask

    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    _write_video(source, np.rint(_rgb_series()[:17]).astype(np.uint8))

    def unexpected_model(*args):
        raise AssertionError("MediaPipe must not load without --skinmask")

    monkeypatch.setattr(skinmask, "prepare_skin_model", unexpected_model)
    video_to_blood(source, output, showprogress=False)
    with pytest.raises(ValueError, match="requires skinmask=True"):
        video_to_blood(source, tmp_path / "other.zarr", skinmask_model="local.tflite")


def test_empty_skin_mask_does_not_fall_back(tmp_path, monkeypatch):
    source, output = tmp_path / "rgb.mkv", tmp_path / "blood.zarr"
    _write_video(source, np.rint(_rgb_series()[:17]).astype(np.uint8))
    passes = _fake_skin_segmenter(
        monkeypatch, lambda *args: np.zeros((2, 3), dtype=bool), tmp_path / "model.tflite"
    )
    with pytest.raises(ValueError, match="after skin masking"):
        video_to_blood(source, output, skinmask=True, showprogress=False)
    assert not output.exists()
    assert len(passes) == 1 and passes[0].closed


def test_cli_skinmask(tmp_path, monkeypatch):
    from rapidtide.workflows import parser_funcs as pf
    from rapidtide.workflows import videotoblood as workflow

    zarr = pytest.importorskip("zarr")
    source, output, model = (
        tmp_path / "rgb.mkv",
        tmp_path / "blood.zarr",
        tmp_path / "model.tflite",
    )
    model.touch()
    _write_video(source, np.rint(_rgb_series()[:17]).astype(np.uint8))
    mask = np.array([[False, True, True], [False, True, True]])
    passes = _fake_skin_segmenter(monkeypatch, lambda *args: mask, model)
    pf.generic_init(
        workflow._get_parser,
        workflow.main,
        [str(source), str(output), "--skinmask", "--skinmaskmodel", str(model), "--noprogressbar"],
    )
    group = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(group["valid_mask"][:], mask)
    assert np.isnan(group["blood"][0, 0]).all()
    assert len(passes) == 2


@pytest.mark.parametrize("program", ["videotoblood", "rapidtide"])
def test_cli_without_installed_tests(program):
    """Start without test helpers or a package-level version (namespace installs)."""
    code = textwrap.dedent("""
        import runpy
        import sys

        # Simulate a runtime installation that does not include the test package.
        sys.modules['rapidtide.tests'] = None
        sys.modules['rapidtide.tests.utils'] = None
        sys.modules['mediapipe'] = None
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


@pytest.mark.parametrize("downsamp", [1, 2])
def test_skinmask_filter_boundary_order(downsamp):
    """Filter uses the complete RGB image, then masks before block averaging."""
    rgb = np.full((5, 7, 3), [200, 120, 80], dtype=np.uint16)
    rgb[2, 2] = [20, 40, 60]
    selected = np.zeros((5, 7), dtype=bool)
    selected[2, 2] = True
    filtered = _spatial_reference(rgb, 1, 0.8)
    expected = _spatial_reference(np.where(selected[..., None], filtered, 0), downsamp, 0)
    result = _spatial_frame(rgb, downsamp=downsamp, lpf=0.8, skin_mask=selected)
    np.testing.assert_allclose(result, expected)
    old_order = _spatial_reference(np.where(selected[..., None], rgb, 0), downsamp, 0.8)
    assert not np.allclose(result, old_order)
    if downsamp == 1:
        assert not result[~selected].any()
        np.testing.assert_allclose(result[selected], filtered[selected])
