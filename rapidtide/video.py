"""Offline RGB-to-blood conversion using TelePlethy's GRGB, POS, and CHROM projections.

Statistics are computed independently at every pixel over the entire recording.
Frames are linearly interpolated onto a regular grid before projection. No
bandpass filtering, detrending, or spatial averaging is performed.
"""

from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import rapidtide._version as tide_versioneer


class RGBBloodProjector:
    """Accumulate whole-recording RGB statistics, then project individual frames.

    Parameters
    ----------
    method : {'grgb', 'pos', 'chrom'}
        TelePlethy projection, including its adaptive POS basis and CHROM polarity.

    Notes
    -----
    Call :meth:`update` on every RGB frame, :meth:`finalize` once, then
    :meth:`transform` on a second pass through the same recording. Float64
    Welford covariance accumulation avoids subtracting nearly equal raw moments.
    Memory use depends on frame dimensions, not recording duration.
    """

    def __init__(self, method: str = "grgb") -> None:
        if method not in ("grgb", "pos", "chrom"):
            raise ValueError("method must be grgb, pos, or chrom")
        self.method = method
        self.count = 0
        self.ready = False

    def update(self, rgb: NDArray) -> None:
        """Accumulate one frame in native RGB order, with shape (height, width, 3)."""
        if self.ready:
            raise ValueError("Cannot update a finalized projector")
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or 0 in rgb.shape:
            raise ValueError("RGB frames must have shape (height, width, 3)")
        if self.count == 0:
            self.mean = np.zeros_like(rgb)
            self.valid = np.ones(rgb.shape[:2], dtype=bool)
            if self.method == "grgb":
                self.ratio_mean = np.zeros(rgb.shape[:2] + (2,), dtype=np.float64)
                self.minimum = np.full_like(rgb, np.inf)
            else:
                self.m2 = np.zeros(rgb.shape[:2] + (3, 3), dtype=np.float64)
        elif rgb.shape != self.mean.shape:
            raise ValueError("Video frame dimensions changed during decoding")
        self.valid &= np.all(np.isfinite(rgb) & (rgb >= 0), axis=-1)
        rgb = np.where(np.isfinite(rgb) & (rgb >= 0), rgb, 0.0)
        self.count += 1
        delta = rgb - self.mean
        self.mean += delta / self.count
        if self.method == "grgb":
            self.minimum = np.minimum(self.minimum, rgb)
            ratios = rgb[..., 1, None] / np.where(rgb[..., [0, 2]] > 0, rgb[..., [0, 2]], 1)
            self.ratio_mean += (ratios - self.ratio_mean) / self.count
        else:
            self.m2 += delta[..., :, None] * (rgb - self.mean)[..., None, :]

    def finalize(self) -> None:
        """Fix projection parameters; undefined pixels are marked in ``valid``."""
        if self.ready:
            raise ValueError("Projector is already finalized")
        if self.count < 2:
            raise ValueError("At least two video frames are required")
        self.valid &= np.all(self.mean > 0, axis=-1)
        self.safe_mean = np.where(self.mean > 0, self.mean, 1.0)
        epsilon = np.finfo(np.float32).eps
        if self.method == "grgb":
            self.valid &= np.all(
                self.minimum[..., [0, 2]] / self.safe_mean[..., [0, 2]] > epsilon, axis=-1
            )
            self.ratio_scale = self.safe_mean[..., [0, 2]] / self.safe_mean[..., 1, None]
            self.offset = np.sum(self.ratio_scale * self.ratio_mean, axis=-1)
        else:
            if self.method == "chrom":
                first = np.broadcast_to([3.0, -2.0, 0.0], self.mean.shape)
                second = np.broadcast_to([1.5, 1.0, -1.5], self.mean.shape)
            else:
                normal = self.mean / np.maximum(
                    np.linalg.norm(self.mean, axis=-1, keepdims=True), np.finfo(float).tiny
                )
                seed = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
                first = seed - np.sum(normal * seed, axis=-1, keepdims=True) * normal
                norm = np.linalg.norm(first, axis=-1, keepdims=True)
                self.valid &= norm[..., 0] > 1e-6
                first /= np.maximum(norm, np.finfo(float).tiny)
                second = np.cross(normal, first)
            covariance = self.m2 / self.count
            covariance /= self.safe_mean[..., :, None] * self.safe_mean[..., None, :]
            stds = [
                np.sqrt(np.maximum(np.einsum("...i,...ij,...j->...", v, covariance, v), 0))
                for v in (first, second)
            ]
            self.valid &= stds[1] > epsilon
            alpha = stds[0] / np.where(stds[1] > epsilon, stds[1], 1.0)
            if self.method == "chrom":
                self.weights = -(first - alpha[..., None] * second)
            else:
                self.weights = first + alpha[..., None] * second
            self.weights /= np.hypot(1.0, alpha)[..., None]
            del self.m2
        self.ready = True

    def transform(self, rgb: NDArray) -> NDArray:
        """Return a mean-removed float32 blood frame; invalid pixels contain NaN."""
        if not self.ready:
            raise ValueError("Finalize whole-recording statistics before transforming")
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.shape != self.mean.shape:
            raise ValueError("Video frame dimensions changed during decoding")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.method == "grgb":
                ratios = rgb[..., 1, None] / rgb[..., [0, 2]]
                signal = np.sum(self.ratio_scale * ratios, axis=-1) - self.offset
            else:
                # Channel means are exactly one after normalization. Centering here
                # equals subtracting each projected waveform's whole-recording mean.
                signal = np.sum((rgb / self.safe_mean - 1.0) * self.weights, axis=-1)
            return np.where(self.valid, 100.0 * signal, np.nan).astype(np.float32)


def _video_frames(filename: str):
    """Decode the first video stream in presentation order, retaining 16-bit RGB."""
    import av

    with av.open(str(filename)) as container:
        if not container.streams.video:
            raise ValueError("Input contains no video stream")
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError("Video has no frame timestamps; missing frames cannot be located")
            yield frame.to_ndarray(format="rgb48le"), float(frame.time)


def _timing(timestamps: NDArray) -> dict:
    """Describe timing without interpolating, duplicating, or dropping samples."""
    if len(timestamps) < 2:
        raise ValueError("At least two video frames are required")
    intervals = np.diff(timestamps)
    if not np.all(np.isfinite(timestamps)) or np.any(intervals <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing")
    interval = float(np.median(intervals))
    uniform = bool(np.allclose(intervals, interval, rtol=0.01, atol=1e-6))
    return {
        "uniform_sampling": uniform,
        "sampling_interval_seconds": interval if uniform else None,
        "start_time_seconds": float(timestamps[0]),
    }


def _regular_times(filename: str, source_times: NDArray, framerate: float | None) -> NDArray:
    """Choose a fixed acquisition grid within the observed interval, without extrapolation."""
    import av

    _timing(source_times)
    if framerate is None:
        with av.open(str(filename)) as container:
            rate = container.streams.video[0].guessed_rate
        framerate = float(rate) if rate is not None and rate > 0 else None
    if framerate is None:
        framerate = 1.0 / np.median(np.diff(source_times))
    count = int(np.floor((source_times[-1] - source_times[0]) * framerate + 1e-8)) + 1
    if count < 2:
        raise ValueError("Recording is too short for the requested output frame rate")
    return source_times[0] + np.arange(count, dtype=np.float64) / framerate


def _regular_frames(filename: str, source_times: NDArray, target_times: NDArray):
    """Stream linear RGB interpolation using only the two neighboring source frames."""
    previous = None
    output_index = 0
    count = 0
    for rgb, timestamp in _video_frames(filename):
        if count >= len(source_times) or timestamp != source_times[count]:
            raise ValueError("Video timestamps or frame count changed between decoding passes")
        if previous is not None:
            if rgb.shape != previous.shape:
                raise ValueError("Video frame dimensions changed during decoding")
            while (
                output_index < len(target_times)
                and target_times[output_index] <= timestamp + 1e-10
            ):
                target = target_times[output_index]
                weight = np.clip((target - previous_time) / (timestamp - previous_time), 0, 1)
                # Float64 interpolation preserves fractional intensities for the
                # subsequent whole-recording statistics and nonlinear projections.
                frame = previous.astype(np.float64) * (1.0 - weight) + rgb * weight
                yield frame, target
                output_index += 1
        previous, previous_time = rgb, timestamp
        count += 1
    if count != len(source_times) or output_index != len(target_times):
        raise ValueError("Video frame count changed between decoding passes")


def _display_limit(blood, percentile: float = 99.5, pixels: int = 65536) -> float:
    """Find a fixed symmetric display range for a whole blood recording.

    Parameters
    ----------
    blood : zarr.Array
        Blood signal with axes (y, x, time). Read one temporal chunk at a time.
    percentile : float, optional
        Percentile of absolute finite values that maps to full scale. Default 99.5.
    pixels : int, optional
        Approximate number of pixels per frame used for the estimate; larger
        frames are spatially subsampled, smaller ones are used in full. Default 65536.

    Returns
    -------
    float
        Positive display limit; 1.0 if the recording contains no nonzero finite values.
    """
    stride = max(1, int(np.sqrt(blood.shape[0] * blood.shape[1] / pixels)))
    step = blood.chunks[2]
    samples = [
        np.abs(blood[::stride, ::stride, start : start + step]).ravel()
        for start in range(0, blood.shape[2], step)
    ]
    values = np.concatenate(samples)
    values = values[np.isfinite(values)]
    limit = float(np.percentile(values, percentile)) if values.size else 0.0
    return limit if limit > 0 else 1.0


def _gray_frame(frame: NDArray, limit: float) -> NDArray:
    """Map one blood frame to 8-bit gray, padded to even dimensions for H.264.

    Parameters
    ----------
    frame : ndarray
        Blood frame with shape (height, width); invalid pixels are NaN.
    limit : float
        Signal magnitude mapped to full white (+limit) or near black (-limit).

    Returns
    -------
    ndarray
        uint8 image with even height and width. Zero signal is 128, the
        signal range maps to 1-255, and NaN (invalid) or padded pixels are 0.
    """
    height, width = frame.shape
    gray = np.zeros((height + height % 2, width + width % 2), dtype=np.uint8)
    scaled = np.clip(np.rint(128.0 + 127.0 * frame / limit), 1, 255)
    gray[:height, :width] = np.where(np.isfinite(frame), scaled, 0)
    return gray


def write_blood_video(
    zarrfilename: str,
    videofilename: str,
    limit: float | None = None,
    showprogress: bool = True,
) -> tuple[Path, float]:
    """Write the blood signal in a videotoblood Zarr store as a monochrome video.

    This is intended for visual debugging and data validation only. The video is
    8-bit H.264 (lossy), so quantitative analysis should use the Zarr store.

    Parameters
    ----------
    zarrfilename : str
        Completed videotoblood Zarr store.
    videofilename : str
        New output video file; the container is chosen from the extension
        (e.g. .mp4, .mkv, .mov). Existing files are never replaced.
    limit : float, optional
        Signal magnitude mapped to full scale. Defaults to the 99.5th percentile
        of absolute finite values over the whole recording, so the gray mapping
        is fixed across frames.
    showprogress : bool, optional
        Display a progress bar while encoding. Default True.

    Returns
    -------
    pathlib.Path
        Completed video file.
    float
        Display limit used. Gray level 128 is zero signal, 255 is +limit,
        1 is -limit, and 0 marks invalid pixels.

    Raises
    ------
    FileExistsError
        If ``videofilename`` already exists.
    ValueError
        If ``limit`` is not finite and positive.
    """
    import av

    from rapidtide.zarrio import open_blood_zarr

    output = Path(videofilename).absolute()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    group, times, _, step = open_blood_zarr(str(zarrfilename))
    blood = group["blood"]
    if limit is None:
        limit = _display_limit(blood)
    elif not np.isfinite(limit) or limit <= 0:
        raise ValueError("Display limit must be finite and positive")
    rate = Fraction(1.0 / step).limit_denominator(1001000)
    chunk = blood.chunks[2]
    try:
        with av.open(str(output), "w") as container:
            container.metadata["comment"] = (
                f"rapidtide videotoblood {group.attrs.get('method')} signal: gray 128 = 0, "
                f"255 = +{limit:.6g}, 1 = -{limit:.6g}, 0 = invalid"
            )
            stream = container.add_stream("libx264", rate=rate)
            stream.height = blood.shape[0] + blood.shape[0] % 2
            stream.width = blood.shape[1] + blood.shape[1] % 2
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "12"}
            with tqdm(total=len(times), desc="Blood video", disable=not showprogress) as bar:
                for start in range(0, len(times), chunk):
                    frames = blood[..., start : start + chunk]
                    for offset in range(frames.shape[-1]):
                        image = av.VideoFrame.from_ndarray(
                            _gray_frame(frames[..., offset], limit), format="gray"
                        )
                        image.pts = start + offset
                        image.time_base = 1 / rate
                        for packet in stream.encode(image):
                            container.mux(packet)
                        bar.update()
            for packet in stream.encode():
                container.mux(packet)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output, limit


def video_to_blood(
    infilename: str,
    outfilename: str,
    method: str = "grgb",
    chunks: tuple[int, int, int] = (64, 64, 16),
    framerate: float | None = None,
    showprogress: bool = True,
    commandline: str | None = None,
    bloodvideo: str | None = None,
) -> Path:
    """Regularize video sampling and convert RGB to a local blood-signal Zarr group.

    Parameters
    ----------
    infilename, outfilename : str
        Input video and new output Zarr directory. Existing output is never replaced.
    method : {'grgb', 'pos', 'chrom'}, optional
        Projection method, default 'grgb'. Statistics are independent per pixel.
    chunks : tuple of int, optional
        Chunk dimensions (y, x, time). The temporal chunk also sets the number
        of output frames buffered in RAM. Default (64, 64, 16).
    framerate : float, optional
        Output sampling rate in Hz. Defaults to the decoder's nominal frame rate,
        or the reciprocal median source interval if that rate is unavailable.
        RGB frames are linearly interpolated to this grid before projection.
    showprogress : bool, optional
        Display a progress bar for each pass.
    commandline : str, optional
        Command invocation recorded as provenance.
    bloodvideo : str, optional
        If given, also write the blood signal as a monochrome 8-bit video to
        this new file after the Zarr store is complete (see
        :func:`write_blood_video`). For debugging and validation only.

    Returns
    -------
    pathlib.Path
        Completed Zarr directory containing blood (y, x, time), timestamps
        (seconds), and valid_mask (y, x). Undefined pixels are NaN throughout.

    Notes
    -----
    Sampling is regularized before estimating whole-recording statistics. The
    original timestamps are retained as provenance. No temporal bandpass filter
    is applied. Image row/column orientation is retained.
    An incomplete conversion is removed automatically on failure.
    """
    if len(chunks) != 3 or any(int(c) != c or c < 1 for c in chunks):
        raise ValueError("chunks must contain three positive integers (y, x, time)")
    if framerate is not None and (not np.isfinite(framerate) or framerate <= 0):
        raise ValueError("framerate must be finite and positive")
    projector = RGBBloodProjector(method)
    output = Path(outfilename).absolute()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if bloodvideo is not None and Path(bloodvideo).exists():
        raise FileExistsError(f"Output already exists: {Path(bloodvideo).absolute()}")
    if not Path(infilename).is_file():
        raise FileNotFoundError(f"Video not found: {infilename}")
    try:
        import av
        import zarr
    except ImportError as exc:
        raise ImportError("Video conversion requires pip install 'rapidtide[video]'") from exc

    source_times = np.asarray(
        [
            time
            for _, time in tqdm(
                _video_frames(infilename), desc="Frame timing", disable=not showprogress
            )
        ],
        dtype=np.float64,
    )
    timestamps = _regular_times(infilename, source_times, framerate)
    for rgb, timestamp in tqdm(
        _regular_frames(infilename, source_times, timestamps),
        total=len(timestamps),
        desc="RGB statistics",
        disable=not showprogress,
    ):
        projector.update(rgb)
    projector.finalize()
    timing = _timing(timestamps)
    if not np.any(projector.valid):
        raise ValueError("No pixels have a valid blood projection for this method")
    shape = projector.mean.shape[:2] + (projector.count,)
    chunks = tuple(min(int(c), s) for c, s in zip(chunks, shape))
    with TemporaryDirectory(prefix=".video-to-blood-", dir=output.parent) as temporary:
        storepath = Path(temporary) / "blood.zarr"
        group = zarr.open_group(str(storepath), mode="w", zarr_format=2)
        group.attrs.update(
            {
                "schema": "rapidtide.video_to_blood",
                "schema_version": 1,
                "rapidtide_version": tide_versioneer.get_versions()["version"],
                "pyav_version": av.__version__,
                "zarr_version": zarr.__version__,
                "source": str(Path(infilename).absolute()),
                "method": method,
                "projection_implementation": "TelePlethy (adaptive POS, normalized CHROM)",
                "statistics": "whole_recording_per_pixel",
                "mean_removed": True,
                "temporal_filter": None,
                "resampled": True,
                "interpolation": "linear_rgb_before_projection",
                "source_frame_count": len(source_times),
                "max_source_gap_seconds": float(np.max(np.diff(source_times))),
                "signal_scale": 100.0,
                "units": "relative RGB projection (not calibrated blood volume)",
                "axes": ["y", "x", "time"],
                "spatial_units": "pixels",
                "rgb_decode_format": "rgb48le",
                "timestamp_source": "regular_grid_from_presentation",
                "requested_framerate_hz": framerate,
                "commandline": commandline,
                **timing,
            }
        )
        blood = group.create_array("blood", shape=shape, chunks=chunks, dtype="float32")
        blood.attrs["_ARRAY_DIMENSIONS"] = ["y", "x", "time"]
        times = group.create_array("timestamps", data=timestamps)
        times.attrs.update({"units": "seconds", "_ARRAY_DIMENSIONS": ["time"]})
        original_times = group.create_array("source_timestamps", data=source_times)
        original_times.attrs.update({"units": "seconds", "_ARRAY_DIMENSIONS": ["source_frame"]})
        mask = group.create_array("valid_mask", data=projector.valid, chunks=chunks[:2])
        mask.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
        buffer = np.empty(shape[:2] + (chunks[2],), dtype=np.float32)
        count = 0
        for rgb, timestamp in tqdm(
            _regular_frames(infilename, source_times, timestamps),
            total=projector.count,
            desc="Blood conversion",
            disable=not showprogress,
        ):
            if count >= projector.count or timestamp != timestamps[count]:
                raise ValueError("Video timestamps or frame count changed between decoding passes")
            slot = count % chunks[2]
            buffer[..., slot] = projector.transform(rgb)
            count += 1
            if slot == chunks[2] - 1 or count == projector.count:
                blood[..., count - slot - 1 : count] = buffer[..., : slot + 1]
        if count != projector.count:
            raise ValueError("Video frame count changed between decoding passes")
        # Publish only a complete store, and refuse to replace existing output.
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        storepath.rename(output)
    if bloodvideo is not None:
        write_blood_video(output, bloodvideo, showprogress=showprogress)
    return output
