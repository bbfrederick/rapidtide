"""Blood Zarr input validation and geometry-preserving analysis output.

Zarr uses image order (y, x, time); the existing analysis uses (x, y, z, time).
The singleton z plane below is an internal adapter, not a spatial resampling.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np


def is_zarr(filename: str) -> bool:
    """Recognize a local Zarr group without importing optional dependencies."""
    path = Path(filename)
    return path.is_dir() and any((path / marker).is_file() for marker in (".zgroup", "zarr.json"))


def open_blood_zarr(filename: str):
    """Open and validate a uniformly sampled videotoblood store without loading blood data.

    Returns the group, timestamps, validity mask, and sampling interval. Timing
    is validated against the actual timestamps, independently of metadata flags.
    """
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("Blood Zarr input requires pip install 'rapidtide[video]'") from exc
    group = zarr.open_group(str(filename), mode="r")
    if (
        group.attrs.get("schema") != "rapidtide.video_to_blood"
        or group.attrs.get("schema_version") != 1
    ):
        raise ValueError("Input must be a supported videotoblood Zarr store (schema version 1)")
    if group.attrs.get("axes") != ["y", "x", "time"]:
        raise ValueError("Blood Zarr axes must be (y, x, time)")
    if any(name not in group for name in ("blood", "timestamps", "valid_mask")):
        raise ValueError("Blood Zarr requires blood, timestamps, and valid_mask arrays")
    blood = group["blood"]
    if blood.ndim != 3 or min(blood.shape) < 1 or blood.shape[-1] < 2 or blood.dtype != np.float32:
        raise ValueError("Blood Zarr must contain float32 data with shape (y, x, time >= 2)")
    if group["timestamps"].shape != (blood.shape[-1],):
        raise ValueError("Blood Zarr timestamps must match the number of frames")
    times = np.asarray(group["timestamps"][:], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Blood Zarr timestamps must be finite and strictly increasing")
    step = float((times[-1] - times[0]) / (len(times) - 1))
    regular = times[0] + np.arange(len(times)) * step
    if not np.allclose(times, regular, rtol=0, atol=max(1e-8, step * 1e-5)):
        raise ValueError(
            "Blood Zarr sampling is irregular. Reconvert the source video with videotoblood "
            "to interpolate missing frames onto a regular sampling grid."
        )
    declared_step = group.attrs.get("sampling_interval_seconds")
    if declared_step is not None and (
        not np.isfinite(declared_step) or not np.isclose(declared_step, step, rtol=1e-5)
    ):
        raise ValueError("Blood Zarr sampling metadata disagrees with timestamps")
    if group["valid_mask"].shape != blood.shape[:2] or group["valid_mask"].dtype != np.bool_:
        raise ValueError("Blood Zarr valid_mask must be boolean and match the (y, x) geometry")
    mask = group["valid_mask"][:]
    if not mask.any():
        raise ValueError("Blood Zarr valid_mask contains no valid pixels")
    return group, times, mask, step


def blood_header(shape: tuple, step: float, start: float):
    """Make a NIfTI-2 metadata adapter with unit pixel spacing and time in seconds."""
    height, width, frames = shape
    header = nib.Nifti2Header()
    header.set_data_shape((width, height, 1, frames))
    header.set_zooms((1.0, 1.0, 1.0, step))
    header.set_xyzt_units(xyz="unknown", t="sec")
    header["toffset"] = start
    return header


def save_map(filename: str, data: np.ndarray, header, metadata: dict, sample_axis: str = "time"):
    """Write an analysis map in native (y, x[, sample]) geometry to a Zarr group."""
    import zarr

    # Zarr attributes use standard JSON, unlike NumPy arrays and scalars produced
    # by the analysis. Normalize nested metadata just as for JSON sidecars.
    def numpy_json(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported metadata type: {type(value).__name__}")

    metadata = json.loads(json.dumps(metadata, default=numpy_json))
    if data.ndim not in (3, 4) or data.shape[2] != 1:
        raise ValueError("Blood Zarr output requires a single internal image plane")
    native = np.swapaxes(np.squeeze(data, axis=2), 0, 1)
    axes = ["y", "x"] + ([sample_axis] if native.ndim == 3 else [])
    chunks = tuple(min(size, 64 if index < 2 else 16) for index, size in enumerate(native.shape))
    group = zarr.open_group(str(filename), mode="w", zarr_format=2)
    group.attrs.update(
        {
            **metadata,
            "schema": "rapidtide.analysis_map",
            "schema_version": 1,
            "axes": axes,
            "spatial_units": "pixels",
        }
    )
    array = group.create_array("data", data=native, chunks=chunks)
    array.attrs["_ARRAY_DIMENSIONS"] = axes
    if native.ndim == 3:
        step, start = float(header["pixdim"][4]), float(header["toffset"])
        coordinates = start + np.arange(native.shape[-1]) * step
        name = "timestamps" if sample_axis == "time" else sample_axis
        coord = group.create_array(name, data=coordinates)
        coord.attrs.update({"units": "Hz" if sample_axis == "frequency" else "seconds"})
        group.attrs.update({"sampling_interval": step, "sample_start": start})
