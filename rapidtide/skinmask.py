"""Optional, seed-free MediaPipe face/body skin masking of decoded RGB frames."""

import hashlib
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

MODEL_NAME = "selfie_multiclass_256x256.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/1/" + MODEL_NAME
)
MODEL_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
SKIN_LABELS = ("face-skin", "body-skin")


def _mediapipe():
    """
    Import the optional MediaPipe dependency.

    Returns
    -------
    module
        Imported MediaPipe module.
    """
    try:
        import mediapipe
    except ImportError as exc:
        raise ImportError("Skin masking requires pip install 'rapidtide[skinmask]'") from exc
    return mediapipe


def _sha256(path: Path) -> str:
    """
    Compute the hexadecimal SHA-256 digest of a file.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    str
        Hexadecimal digest of the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_skin_model(filename: str | None = None) -> tuple[Path, dict]:
    """
    Resolve a supplied model or download Google's versioned model into a user cache.

    The default download is verified and published atomically. A local model
    override supports offline use; its skin-class labels are checked at inference.
    No video data is sent to the model host.

    Parameters
    ----------
    filename : str | None, optional
        Path to the input or output file.

    Returns
    -------
    tuple
        Local model path and model provenance metadata.
    """
    mp = _mediapipe()
    if filename is not None:
        path = Path(filename).expanduser().absolute()
        if not path.is_file():
            raise FileNotFoundError(f"Skin segmentation model not found: {path}")
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache = cache / "rapidtide" / "models"
        path = cache / MODEL_NAME
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".skin-model-", dir=cache) as temporary:
                downloaded = Path(temporary) / MODEL_NAME
                try:
                    with (
                        urlopen(MODEL_URL, timeout=60) as response,
                        downloaded.open("wb") as stream,
                    ):
                        shutil.copyfileobj(response, stream)
                except (URLError, OSError) as exc:
                    raise OSError(
                        "Cannot download the MediaPipe skin model. Use --skinmaskmodel FILE "
                        "to supply a local copy."
                    ) from exc
                if _sha256(downloaded) != MODEL_SHA256:
                    raise ValueError("Downloaded MediaPipe skin model checksum does not match")
                downloaded.replace(path)
        if not path.is_file() or _sha256(path) != MODEL_SHA256:
            raise ValueError(f"Cached skin model checksum does not match; remove {path} and retry")
    return path, {
        "backend": "mediapipe_selfie_multiclass",
        "mediapipe_version": mp.__version__,
        "model": str(path),
        "model_sha256": _sha256(path),
        "model_url": MODEL_URL if filename is None else None,
        "included_classes": list(SKIN_LABELS),
        "selection": "winning_category_union",
        "non_skin_rgb_value": 0,
        "landmark_refinement": False,
        "calculated_before": ["gaussian_lowpass", "crop", "block_mean", "temporal_interpolation"],
        "applied_after": ["gaussian_lowpass"],
        "applied_before": ["crop", "block_mean", "temporal_interpolation"],
        "inference_passes": 2,
    }


class SkinSegmenter:
    """Context-managed multiclass segmenter retaining face-skin and body-skin only.

    A fresh video-mode segmenter is used for each decoding pass, so temporal
    state starts identically for whole-recording statistics and blood extraction.
    Inference uses an 8-bit RGB copy; retained blood-extraction RGB is unmodified.
    """

    def __init__(self, model: str | Path):
        """
        Initialize `SkinSegmenter`.

        Parameters
        ----------
        model : str | Path
            Model instance or model file used by this routine.
        """
        self.mp = _mediapipe()
        options = self.mp.tasks.vision.ImageSegmenterOptions(
            base_options=self.mp.tasks.BaseOptions(
                model_asset_path=str(model), delegate=self.mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=self.mp.tasks.vision.RunningMode.VIDEO,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        try:
            self.segmenter = self.mp.tasks.vision.ImageSegmenter.create_from_options(options)
        except RuntimeError as exc:
            raise ValueError(f"Cannot initialize MediaPipe skin segmentation: {exc}") from exc
        try:
            labels = list(self.segmenter.labels)
            self.skin_ids = tuple(labels.index(name) for name in SKIN_LABELS)
        except ValueError as exc:
            self.segmenter.close()
            raise ValueError(
                "Skin model must contain face-skin and body-skin class labels"
            ) from exc
        self.last_timestamp_ms = -1
        self.timestamp_origin = None

    def __enter__(self):
        """
        Return the detector for use in a context manager.

        Returns
        -------
        SkinSegmenter
            Active detector instance.
        """
        return self

    def __exit__(self, *args):
        """
        Close the detector when leaving the context manager.

        Parameters
        ----------
        *args : object
            Additional positional arguments.
        """
        self.segmenter.close()

    def mask(self, rgb: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Return a native (y, x) skin mask without changing source RGB values.

        Parameters
        ----------
        rgb : np.ndarray
            Image or samples with red, green, and blue channels.
        timestamp : float
            Frame time in seconds.

        Returns
        -------
        numpy.ndarray of bool
            Pixels assigned to any of the selected skin classes.
        """
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape) == 0:
            raise ValueError("Skin segmentation requires RGB frames with shape (height, width, 3)")
        if rgb.dtype == np.uint16:
            image = ((rgb.astype(np.uint32) + 128) // 257).astype(np.uint8)
        elif rgb.dtype == np.uint8:
            image = np.ascontiguousarray(rgb)
        else:
            raise ValueError("Skin segmentation requires decoded uint8 or uint16 RGB")
        if not np.isfinite(timestamp):
            raise ValueError("Skin segmentation requires finite frame timestamps")
        if self.timestamp_origin is None:
            self.timestamp_origin = min(0.0, float(timestamp))
        # MediaPipe requires unique integer milliseconds. Preserve actual source
        # times outside inference even if two frames round to the same millisecond.
        timestamp_ms = max(
            int(round((timestamp - self.timestamp_origin) * 1000)), self.last_timestamp_ms + 1
        )
        result = self.segmenter.segment_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image), timestamp_ms
        )
        self.last_timestamp_ms = timestamp_ms
        if result.category_mask is None:
            raise ValueError("MediaPipe did not return a skin segmentation category mask")
        category = result.category_mask.numpy_view()
        if category.ndim == 3 and category.shape[-1] == 1:
            category = category[..., 0]
        if category.shape != rgb.shape[:2]:
            raise ValueError("Skin segmentation mask does not match the source frame dimensions")
        return np.isin(category, self.skin_ids)


def skin_segmented_frames(frames, model: str | Path):
    """
    Yield original RGB frames, timestamps, and skin masks without applying them.

    Parameters
    ----------
    frames : iterable
        Input video frames and their timing information.
    model : str | Path
        Model instance or model file used by this routine.

    Yields
    ------
    object
        Next item produced by the iterator.
    """
    with SkinSegmenter(model) as segmenter:
        for rgb, timestamp in frames:
            mask = segmenter.mask(rgb, timestamp)
            yield rgb, timestamp, mask


def skinmasked_frames(frames, model: str | Path):
    """
    Apply skin masks directly to unfiltered frames (convenience iterator).

    Parameters
    ----------
    frames : iterable
        Input video frames and their timing information.
    model : str | Path
        Model instance or model file used by this routine.

    Yields
    ------
    object
        Next item produced by the iterator.
    """
    for rgb, timestamp, mask in skin_segmented_frames(frames, model):
        yield np.where(mask[..., None], rgb, 0), timestamp
