"""Lossless float32 TIFF analysis maps and page-wise sequence access.

TIFF stores sequences as (sample, y, x). The reader presents the native video
geometry (y, x[, sample]) used by Rapidtide's analysis and visualization code.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def _tifffile():
    """
    Import the optional TIFF dependency.

    Returns
    -------
    module
        Imported tifffile module.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("TIFF output requires pip install 'rapidtide[video]'") from exc
    return tifffile


def _json_value(value):
    """
    Convert metadata to JSON-compatible Python values.

    Parameters
    ----------
    value : object
        Metadata value to convert recursively.

    Returns
    -------
    object
        JSON-compatible representation of the metadata value.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def needs_bigtiff(shape):
    """
    Reserve space for float32 pixels and per-page TIFF directories below 4 GiB.

    Parameters
    ----------
    shape : tuple of int
        Array dimensions in the order expected by the routine.

    Returns
    -------
    bool
        Whether the estimated image size calls for BigTIFF.
    """
    pages = shape[0] if len(shape) == 3 else 1
    return int(np.prod(shape, dtype=object)) * 4 + pages * 4096 >= 2**32 - 2**25


def save_map(filename, data, header, metadata, sample_axis="time", compression="deflate"):
    """
    Write an internal (x, y, 1[, sample]) array as float32 TIFF.

    Scalar maps use one grayscale page. Sequences use one page per sample,
    with exact coordinates embedded in JSON metadata. DEFLATE is lossless;
    passing None disables compression. Other codecs are rejected. Large stacks
    automatically use BigTIFF. Returns the embedded Rapidtide metadata dictionary.

    Parameters
    ----------
    filename : path-like
        Path to the input or output file.
    data : array-like
        Input data to filter.
    header : object
        Image header supplying spatial and temporal metadata.
    metadata : dict
        Rapidtide metadata to embed in the TIFF header.
    sample_axis : str, optional
        Name of the optional sample dimension.
    compression : str or None, optional
        Compression codec for the TIFF image.

    Returns
    -------
    object
        The info.
    """
    tifffile = _tifffile()
    if compression not in (None, "deflate"):
        raise ValueError("Analysis TIFF compression must be lossless DEFLATE or None")
    if data.ndim not in (3, 4) or data.shape[2] != 1 or min(data.shape) < 1:
        raise ValueError("Video TIFF output requires a single internal image plane")
    native = np.swapaxes(np.squeeze(data, axis=2), 0, 1)
    axes = ["y", "x"] + ([sample_axis] if native.ndim == 3 else [])
    physical_shape = native.shape if native.ndim == 2 else (native.shape[-1], *native.shape[:2])
    info = {
        **metadata,
        "schema": "rapidtide.analysis_map",
        "schema_version": 1,
        "axes": axes,
        "spatial_units": "pixels",
        "shape": list(native.shape),
        "dtype": "float32",
        "compression": compression or "none",
        "bigtiff": needs_bigtiff(physical_shape),
    }
    if native.ndim == 3:
        step, start = float(header["pixdim"][4]), float(header["toffset"])
        info.update(
            {
                "sampling_interval": step,
                "sample_start": start,
                "sample_coordinates": (start + np.arange(native.shape[-1]) * step).tolist(),
                "sample_units": "Hz" if sample_axis == "frequency" else "seconds",
            }
        )
    info = json.loads(json.dumps(info, default=_json_value))
    # Q is a generic sequence axis; lag and frequency must not be mislabeled as time.
    storage_axes = "YX" if native.ndim == 2 else ("TYX" if sample_axis == "time" else "QYX")
    path = Path(filename)
    with TemporaryDirectory(prefix=".analysis-tiff-", dir=path.parent) as temporary:
        staging = Path(temporary) / "output.tif"
        if native.ndim == 2:
            frames = np.asarray(native, dtype=np.float32)
        else:
            frames = (
                np.ascontiguousarray(native[..., i], dtype=np.float32)
                for i in range(native.shape[-1])
            )
        tifffile.imwrite(
            staging,
            frames,
            shape=physical_shape,
            dtype=np.float32,
            photometric="minisblack",
            compression=compression,
            bigtiff=info["bigtiff"],
            metadata={"axes": storage_axes, "rapidtide": info},
        )
        staging.replace(path)
    return info


class TiffArray:
    """Present TIFF pages as a native video array, loading only requested samples."""

    def __init__(self, path, shape):
        """
        Initialize `TiffArray`.

        Parameters
        ----------
        path : path-like
            Path to the file.
        shape : tuple of int
            Array dimensions in the order expected by the routine.
        """
        self.path = path
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = np.dtype("float32")
        self.chunks = self.shape[:2] + ((min(16, self.shape[-1]),) if self.ndim == 3 else ())

    def __getitem__(self, key):
        """
        Retrieve the requested data.

        Parameters
        ----------
        key : object
            Array index or slice selecting the requested data.

        Returns
        -------
        numpy.ndarray
            Array values at the requested indices.
        """
        tifffile = _tifffile()
        with tifffile.TiffFile(self.path) as file:
            if self.ndim == 2:
                return file.pages[0].asarray()[key]
            if isinstance(key, tuple) and len(key) == 2 and key[0] is Ellipsis:
                selection = key[1]
            elif key is Ellipsis or (isinstance(key, slice) and key == slice(None)):
                selection = slice(None)
            else:
                values = np.stack([page.asarray() for page in file.pages], axis=-1)
                return values[key]
            if isinstance(selection, (int, np.integer)):
                return file.pages[int(selection)].asarray()
            indices = range(*selection.indices(self.shape[-1]))
            if not indices:
                return np.empty(self.shape[:2] + (0,), dtype=np.float32)
            return np.stack([file.pages[index].asarray() for index in indices], axis=-1)


class TiffMap:
    """Read TIFF analysis metadata without materializing a sequence."""

    def __init__(self, filename):
        """
        Initialize `TiffMap`.

        Parameters
        ----------
        filename : path-like
            Path to the input or output file.
        """
        self.path = Path(filename)
        tifffile = _tifffile()
        with tifffile.TiffFile(self.path) as file:
            description = json.loads(file.pages[0].description or "{}")
            self.attrs = description.get("rapidtide", {})
            shape = tuple(self.attrs.get("shape", ()))
            axes = self.attrs.get("axes", [])
            if (
                self.attrs.get("schema") != "rapidtide.analysis_map"
                or self.attrs.get("schema_version") != 1
                or len(shape) not in (2, 3)
                or len(axes) != len(shape)
                or axes[:2] != ["y", "x"]
                or min(shape) < 1
            ):
                raise ValueError(f"Unsupported Rapidtide analysis TIFF: {self.path}")
            count = shape[-1] if len(shape) == 3 else 1
            if (
                len(file.pages) != count
                or file.pages[0].shape != shape[:2]
                or file.pages[0].dtype != np.dtype("float32")
            ):
                raise ValueError(f"TIFF pages disagree with metadata: {self.path}")
        self.arrays = {"data": TiffArray(self.path, shape)}
        if len(shape) == 3:
            coordinates = np.asarray(self.attrs.get("sample_coordinates"), dtype=float)
            if coordinates.shape != (shape[-1],) or not np.isfinite(coordinates).all():
                raise ValueError(f"Invalid TIFF sample coordinates: {self.path}")
            name = "timestamps" if axes[-1] == "time" else axes[-1]
            self.arrays[name] = coordinates

    def __getitem__(self, name):
        """
        Retrieve the requested data.

        Parameters
        ----------
        name : str
            Name of the requested dataset.

        Returns
        -------
        object
            Requested lazy TIFF array or coordinate dataset.
        """
        return self.arrays[name]

    def __contains__(self, name):
        """
        Check whether the named dataset is available.

        Parameters
        ----------
        name : str
            Name of the requested dataset.

        Returns
        -------
        bool
            Whether the named dataset exists.
        """
        return name in self.arrays


def open_map(filename):
    """
    Open a Rapidtide TIFF map or sequence, retaining native array conventions.

    Parameters
    ----------
    filename : path-like
        Path to the input or output file.

    Returns
    -------
    TiffMap
        Opened TIFF map with lazy array access.
    """
    return TiffMap(filename)
