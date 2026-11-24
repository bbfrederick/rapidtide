"""Render native 2D Rapidtide TIFF/legacy Zarr maps and correlation lag sweeps."""

import json
import warnings
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tqdm import tqdm

from rapidtide.stats import getfracvals
from rapidtide.zarrio import is_zarr


def _map_path(root, descriptor):
    for suffix in (".tif", ".tiff", ".zarr"):
        path = Path(f"{root}_desc-{descriptor}{suffix}")
        if path.exists():
            return path
    return None


def _open_map(root, descriptor, axes):
    path = _map_path(root, descriptor)
    if path is None:
        raise FileNotFoundError(
            f"Required 2D TIFF/Zarr output not found: {root}_desc-{descriptor}"
        )
    if path.suffix in (".tif", ".tiff"):
        from rapidtide.tiffio import open_map

        group = open_map(path)
    else:
        import zarr

        group = zarr.open_group(str(path), mode="r")
    if group.attrs.get("axes") != axes or "data" not in group:
        raise ValueError(f"{path.name} must contain data with axes {axes}")
    array = group["data"]
    if array.ndim != len(axes) or min(array.shape) < 1:
        raise ValueError(f"Invalid dimensions in {path.name}")
    return group


def _panels(root):
    fit = _open_map(root, "corrfit_mask", ["y", "x"])["data"][:] > 0
    processed = _open_map(root, "processed_mask", ["y", "x"])["data"][:] > 0
    if fit.shape != processed.shape or not processed.any():
        raise ValueError("Analysis masks must have matching geometry and processed pixels")
    specifications = [
        ("maxtime_map", "Peak-fit lag", fit, "delay", "viridis"),
        ("maxcorr_map", "Peak correlation", fit, "corr", "hot"),
    ]
    for descriptor, title, imtype, colormap in [
        ("maxtimerefined_map", "Refined lag", "delay", "viridis"),
        ("maxcorrrefined_map", "Refined correlation", "corr", "hot"),
        ("CoV_map", "Coefficient of variation", None, "hot"),
        ("maxcorrsq_map", "Squared peak correlation", None, "hot"),
        ("maxwidth_map", "Peak width", None, "rainbow"),
        ("mean_map", "Image", None, "gray"),
        ("MTT_map", "Mean transit time", None, "rainbow"),
        ("neglog10p_map", "Significance", None, "hot"),
        ("std_map", "Standard deviation", None, "hot"),
        ("timepercentile_map", "Time percentile", None, "viridis"),
        ("unfiltmean_map", "Unfiltered image", None, "gray"),
    ]:
        print(f"{descriptor}, {title}, {imtype}, {colormap}")
        if _map_path(root, descriptor) is not None:
            specifications.append((descriptor, title, processed, imtype, colormap))
    panels = []
    for descriptor, title, mask, imtype, colormap in specifications:
        values = _open_map(root, descriptor, ["y", "x"])["data"][:]
        if values.shape != processed.shape:
            raise ValueError(f"Geometry of {descriptor} disagrees with analysis masks")
        valid = mask & np.isfinite(values)
        if np.any(mask & ~valid):
            warnings.warn(
                f"Nonfinite values in {descriptor} are excluded from display", stacklevel=2
            )
        if not valid.any():
            raise ValueError(f"No finite included pixels in {descriptor}")
        panels.append((title, values, valid, imtype, colormap))
    return panels, processed


def _figure(figsize):
    figure = Figure(figsize=figsize, dpi=120, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _image(
    ax,
    values,
    valid,
    imtype=None,
    minlag=-1.0,
    maxlag: float | int = 1.0,
    input_cmap="hot",
    mincorr=-1.0,
    maxcorr=1.0,
):
    from matplotlib import colormaps

    thevaluearray = np.nan_to_num(np.ma.array(values, mask=~valid))
    if imtype is None:
        minval, maxval = getfracvals(values[valid], [0.02, 0.98], nozero=True)
    elif imtype == "corr":
        minval = mincorr
        maxval = maxcorr
    elif imtype == "delay":
        minval = minlag
        maxval = maxlag
    else:
        minval, maxval = getfracvals(values[valid], [0.02, 0.98], nozero=True)
    print(f"{minval=}, {maxval=}")
    cmap = colormaps[input_cmap].copy()
    print(f"cmap set to {input_cmap}")
    cmap.set_bad("#bdbdbd")
    return (
        ax.imshow(
            thevaluearray,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,
            vmin=minval,
            vmax=maxval,
        ),
        minval,
        maxval,
    )


def _geometry(ax, shape, ylabel=True):
    height, width = shape
    ax.set_xlabel("x (output pixels)")
    ax.set_xticks([0, width // 2, width - 1])
    ax.set_yticks(np.linspace(0, height - 1, 5).astype(int))
    if ylabel:
        ax.set_ylabel("y (output pixels)")
    else:
        ax.tick_params(labelleft=False)


def _map_figure(root, panels, options, minlag, maxlag, mincorr=-1.0, maxcorr=1.0):
    figure = _figure((4 * len(panels), 9.2))
    axes = figure.subplots(1, len(panels), squeeze=False)[0]
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.27, top=0.89, wspace=0.18)
    subtitle = []
    if options.get("filterband"):
        subtitle.append(f"{options['filterband'].capitalize()} band")
    if options.get("fmrifreq"):
        subtitle.append(f"{options['fmrifreq']:g} Hz")
    figure.suptitle(f"{root.name}\n{' · '.join(subtitle)}", fontsize=15, y=0.975)
    for index, (ax, (title, values, valid, imtype, colormap)) in enumerate(zip(axes, panels)):
        image, minval, maxval = _image(
            ax,
            values,
            valid,
            imtype=imtype,
            minlag=minlag,
            maxlag=maxlag,
            input_cmap=colormap,
            mincorr=mincorr,
            maxcorr=maxcorr,
        )
        ax.set_title(title, pad=10)
        _geometry(ax, values.shape, ylabel=index == 0)
        position = ax.get_position()
        cax = figure.add_axes([position.x0, 0.125, position.width, 0.024])
        bar = figure.colorbar(image, cax=cax, orientation="horizontal")
        if imtype is None:
            barlabel = "Arbitrary units"
        elif imtype == "delay":
            barlabel = "Lag (s)"
        elif imtype == "corr":
            barlabel = "Correlation"
        else:
            barlabel = "Arbitrary units"
        bar.set_label(barlabel)
        bar.set_ticks(np.linspace(image.norm.vmin, image.norm.vmax, 5))
        figure.text(
            (position.x0 + position.x1) / 2,
            0.185,
            f"{np.count_nonzero(valid):,} included pixels",
            ha="center",
            va="top",
            fontsize=10,
        )
    height, width = panels[0][1].shape
    footer = f"Gray: excluded pixels. Native {width} × {height} geometry."
    if "lagmin" in options and "lagmax" in options:
        footer += f"\nLag search: {options['lagmin']:g} to {options['lagmax']:g} seconds."
    figure.text(0.5, 0.025, footer, ha="center", va="bottom", fontsize=10)
    return figure


def _corrout(root, shape):
    group = _open_map(root, "corrout_info", ["y", "x", "lag"])
    if group["data"].shape[:2] != shape or "lag" not in group:
        raise ValueError("Correlation output must match the maps and contain a lag coordinate")
    lag = np.asarray(group["lag"][:], dtype=float)
    if (
        lag.shape != (group["data"].shape[-1],)
        or not np.isfinite(lag).all()
        or np.any(np.diff(lag) <= 0)
    ):
        raise ValueError("Correlation lag coordinates must be finite and strictly increasing")
    return group, lag


def _lag_limits(
    panels: list[tuple], minlag: float | None, maxlag: float | None
) -> tuple[float, float]:
    """Resolve the delay display limits, filling in any limit not given.

    The default is a symmetric range covering every included finite delay,
    rounded up to a multiple of 0.05 s (0.5 s once delays exceed 1 s), and
    never narrower than +/-0.25 s.

    Parameters
    ----------
    panels : list of tuple
        Panels from ``_panels``: ``(title, values, valid, imtype, colormap)``.
        Only panels with ``imtype == "delay"`` contribute to the default.
    minlag : float or None
        Requested lower delay limit in seconds, or None for the default.
    maxlag : float or None
        Requested upper delay limit in seconds, or None for the default.

    Returns
    -------
    tuple of float
        ``(minlag, maxlag)`` in seconds.

    Raises
    ------
    ValueError
        If the resolved limits do not satisfy ``minlag < maxlag``.
    """
    if minlag is None or maxlag is None:
        largest = max(
            np.abs(values[valid]).max()
            for _, values, valid, imtype, _ in panels
            if imtype == "delay"
        )
        step = 0.5 if largest > 1 else 0.05
        limit = float(max(0.25, np.ceil(largest / step) * step))
        minlag = -limit if minlag is None else float(minlag)
        maxlag = limit if maxlag is None else float(maxlag)
    if minlag >= maxlag:
        raise ValueError("minlag must be less than maxlag")
    return minlag, maxlag


def _lag_slice(lag, bounds):
    """Select saved coordinates inclusively, allowing floating-point roundoff."""
    if bounds is None:
        return slice(0, len(lag))
    tolerance = (
        16 * np.finfo(float).eps * np.maximum(max(1.0, np.max(np.abs(lag))), np.abs(bounds))
    )
    start = int(np.searchsorted(lag, bounds[0] - tolerance[0], side="left"))
    stop = int(np.searchsorted(lag, bounds[1] + tolerance[1], side="right"))
    if start >= stop:
        raise ValueError(
            f"No saved corrout lags in videolagrange {bounds[0]:g} to {bounds[1]:g} seconds; "
            f"available lags span {lag[0]:g} to {lag[-1]:g} seconds"
        )
    return slice(start, stop)


def _movie(
    root,
    group,
    lag,
    mask,
    directory,
    fps,
    showprogress,
    first_index=0,
    minlag=-1.0,
    maxlag=1.0,
    input_cmap="hot",
    imtype="corr",
    mincorr=-1.0,
    maxcorr=1.0,
):
    import av

    figure = _figure((5.2, 8.4))
    ax = figure.add_axes([0.11, 0.22, 0.66, 0.68])
    image, minval, maxval = _image(
        ax,
        np.zeros(mask.shape),
        mask,
        imtype=imtype,
        minlag=minlag,
        maxlag=maxlag,
        input_cmap=input_cmap,
        mincorr=mincorr,
        maxcorr=maxcorr,
    )
    _geometry(ax, mask.shape)
    cax = figure.add_axes([0.82, 0.22, 0.035, 0.68])
    bar = figure.colorbar(image, cax=cax, ticks=np.linspace(minval, maxval, 5))
    bar.set_label("Signed correlation")
    figure.suptitle(f"{root.name}\nCorrelation versus lag", y=0.98, fontsize=13)
    label = figure.text(0.5, 0.92, "", ha="center", fontsize=13)
    figure.text(
        0.5,
        0.075,
        f"Fixed hot scale: {minval:g} to {maxval:g}\nGray: excluded pixels\n"
        "Playback sweeps correlation lag, not acquisition time",
        ha="center",
        va="center",
        fontsize=10,
    )
    path = directory / "corrout-lag-sweep.mp4"
    rate = Fraction(float(fps)).limit_denominator(1001000)
    data = group["data"]
    zero_index = int(np.argmin(np.abs(lag)))
    with av.open(str(path), "w") as container:
        container.metadata["comment"] = (
            f"Rapidtide correlation lag sweep: {lag[0]:.9g} to {lag[-1]:.9g} seconds; "
            f"signed hot scale [{minval:g}, {maxval:g}]"
        )
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = figure.canvas.get_width_height()
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        # Buffer one temporal chunk, rather than materializing the correlation volume.
        chunk = data.chunks[-1]
        with tqdm(total=len(lag), desc="Correlation video", disable=not showprogress) as progress:
            for start in range(0, len(lag), chunk):
                block = data[..., first_index + start : first_index + min(start + chunk, len(lag))]
                for offset in range(block.shape[-1]):
                    index = start + offset
                    values = block[..., offset]
                    image.set_data(np.ma.array(values, mask=~(mask & np.isfinite(values))))
                    seconds = 0.0 if abs(lag[index]) < 1e-9 else lag[index]
                    label.set_text(f"Lag = {seconds:+.3f} s")
                    figure.canvas.draw()
                    rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    frame.pts, frame.time_base = index, 1 / rate
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    if index == zero_index:
                        figure.savefig(directory / "corrout-zero-lag.png", dpi=120)
                    progress.update()
        for packet in stream.encode():
            container.mux(packet)
    figure.clear()


def render_rapidtide(
    outputroot: str,
    outdir: str | None = None,
    fps: float = 30.0,
    minlag: float | None = None,
    maxlag: float | None = None,
    movie: bool = True,
    showprogress: bool = True,
    videolagrange: tuple[float, float] | None = None,
    mincorr: float = -1.0,
    maxcorr: float = 1.0,
) -> dict[str, Path]:
    """Render a completed 2D TIFF or legacy Zarr analysis without modifying its outputs.

    Parameters
    ----------
    outputroot : str
        Rapidtide output prefix, excluding ``_desc-*`` suffixes.
    outdir : str, optional
        Figure directory, default ``<outputroot>_figures``. Existing artifacts
        are replaced after successful rendering.
    fps : float, optional
        Movie playback rate in frames per second, default 30. Each movie frame
        represents one saved correlation lag, not one acquired video frame.
    minlag : float, optional
        Minimum display limit in seconds for all delay panels. By default,
        the negative of a symmetric limit covering every finite included delay.
    maxlag : float, optional
        Maximum display limit in seconds for all delay panels. By default,
        a symmetric limit covering every finite included delay. Each limit
        may be given alone; the resolved limits must satisfy minlag < maxlag.
    movie : bool, optional
        Generate the lag sweep when corrout exists, default True.
    showprogress : bool, optional
        Show video encoding progress, default True.
    videolagrange : tuple of float, optional
        Inclusive minimum and maximum saved correlation lags to display, in
        seconds. Defaults to all saved lags. No temporal interpolation is done;
        an interval containing no saved lag raises ValueError. The preview uses
        the selected lag nearest zero. Static maps are unaffected.
    mincorr, maxcorr : float, optional
        Fixed correlation color limits, default -1 and 1. Applied to static hot
        correlation panels, the hot movie, and its preview. Values outside
        the display range saturate at the endpoint colors. Must be finite with
        mincorr < maxcorr. Delay color limits are controlled by minlag and maxlag.

    Returns
    -------
    dict of str to pathlib.Path
        Paths of the generated PNG/PDF, optional MP4 and zero-lag preview, and
        JSON manifest. Missing corrout emits a warning and still renders maps.

    Notes
    -----
    Supports native (y, x) TIFF and legacy Zarr outputs from videotoblood inputs.
    Correlation maps and movies use hot metal; delay maps use viridis.
    All correlation displays use mincorr to maxcorr, independent of
    the included data values.
    Masked and nonfinite pixels are gray. Input video and source Zarr are not needed.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if minlag is not None and (not np.isfinite(minlag)):
        raise ValueError("minlag must be finite")
    if maxlag is not None and (not np.isfinite(maxlag)):
        raise ValueError("maxlag must be finite")
    if minlag is not None and maxlag is not None and minlag >= maxlag:
        raise ValueError("minlag must be less than maxlag")
    if not np.isfinite(mincorr) or not np.isfinite(maxcorr) or mincorr >= maxcorr:
        raise ValueError("mincorr and maxcorr must be finite with mincorr < maxcorr")
    if videolagrange is not None:
        bounds = np.asarray(videolagrange, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("videolagrange must contain finite MIN <= MAX in seconds")
        videolagrange = tuple(float(value) for value in bounds)
    root = Path(outputroot).absolute()
    options_path = Path(f"{root}_desc-runoptions_info.json")
    options = json.loads(options_path.read_text()) if options_path.is_file() else {}
    if options.get("currentstage", "done") != "done":
        raise ValueError(f"Analysis has not completed (stage: {options['currentstage']})")
    panels, processed = _panels(root)
    minlag, maxlag = _lag_limits(panels, minlag, maxlag)
    correlation = None
    if movie:
        if _map_path(root, "corrout_info") is not None:
            group, lag = _corrout(root, processed.shape)
            selected = _lag_slice(lag, videolagrange)
            correlation = group, lag[selected], selected.start
            try:
                import av  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Movie encoding requires pip install 'rapidtide[video]'"
                ) from exc
        else:
            warnings.warn(
                "No corrout output found; rendering maps only. Run rapidtide with "
                "--savecorrout to save correlation functions.",
                stacklevel=2,
            )
    destination = Path(outdir).absolute() if outdir else Path(f"{root}_figures")
    if any(is_zarr(path) for path in (destination, *destination.parents)):
        raise ValueError("Figure output directory must be outside Zarr stores")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "outputroot": str(root),
        "delay_colormap": "viridis",
        "correlation_colormap": "hot",
        "correlation_display_limits": [mincorr, maxcorr],
        "lag_display_limits_seconds": [minlag, maxlag],
        "shape_yx": list(processed.shape),
        "panels": [title for title, *_ in panels],
    }
    with TemporaryDirectory(prefix=".showrapidtide-", dir=destination) as temporary:
        staging = Path(temporary)
        figure = _map_figure(root, panels, options, minlag, maxlag, mincorr, maxcorr)
        for extension in ("png", "pdf"):
            figure.savefig(staging / f"static-maps.{extension}", dpi=200)
        figure.clear()
        if correlation is not None:
            group, lag, first_index = correlation
            _movie(
                root,
                group,
                lag,
                processed,
                staging,
                fps,
                showprogress,
                first_index=first_index,
                mincorr=mincorr,
                maxcorr=maxcorr,
            )
            manifest["movie"] = {
                "colormap": "hot",
                "frames": len(lag),
                "fps": fps,
                "duration_seconds": len(lag) / fps,
                "lag_seconds": lag.tolist(),
                "display_limits": [mincorr, maxcorr],
                "requested_lag_range_seconds": videolagrange,
            }
        (staging / "render-info.json").write_text(json.dumps(manifest, indent=2) + "\n")
        paths = {}
        for path in sorted(staging.iterdir()):
            target = destination / path.name
            path.replace(target)
            paths[path.name] = target
    return paths
