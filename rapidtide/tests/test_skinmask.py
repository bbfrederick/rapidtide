"""Validate optional model setup and RGB masking without downloading a model in CI."""

import hashlib
from io import BytesIO
from types import SimpleNamespace
import sys
from urllib.error import URLError

import numpy as np
import pytest

import rapidtide.skinmask as skinmask


def _fake_mediapipe(monkeypatch, category, labels=None):
    records = SimpleNamespace(images=[], times=[], options=[], closed=0)
    if labels is None:
        labels = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]

    class BaseOptions(SimpleNamespace):
        Delegate = SimpleNamespace(CPU="cpu")

    class Segmenter:
        def __init__(self, options):
            records.options.append(options)
            self.labels = labels

        def segment_for_video(self, image, timestamp_ms):
            records.images.append(image.data.copy())
            records.times.append(timestamp_ms)
            mask = None if category is None else SimpleNamespace(numpy_view=lambda: category)
            return SimpleNamespace(category_mask=mask)

        def close(self):
            records.closed += 1

    vision = SimpleNamespace(
        ImageSegmenterOptions=SimpleNamespace,
        ImageSegmenter=SimpleNamespace(create_from_options=Segmenter),
        RunningMode=SimpleNamespace(VIDEO="video"),
    )
    mp = SimpleNamespace(
        __version__="test",
        tasks=SimpleNamespace(BaseOptions=BaseOptions, vision=vision),
        Image=SimpleNamespace,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    monkeypatch.setattr(skinmask, "_mediapipe", lambda: mp)
    return records


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("singleton_channel", [False, True])
def test_skin_union_and_preserved_rgb(monkeypatch, dtype, singleton_channel):
    category = np.arange(6, dtype=np.uint8).reshape(2, 3)
    if singleton_channel:
        category = category[..., None]
    records = _fake_mediapipe(monkeypatch, category)
    rgb8 = (np.arange(18).reshape(2, 3, 3) + 80).astype(np.uint8)
    rgb = rgb8 if dtype == np.uint8 else rgb8.astype(np.uint16) * 257 + 1
    original = rgb.copy()
    times = [0.0, 0.0004, 0.001]
    frames = list(skinmask.skinmasked_frames(((rgb, time) for time in times), "model.tflite"))
    selected = np.array([[False, False, True], [True, False, False]])
    for (frame, timestamp), expected_time in zip(frames, times):
        assert timestamp == expected_time
        assert frame.dtype == rgb.dtype
        np.testing.assert_array_equal(frame[selected], rgb[selected])
        assert not frame[~selected].any()
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_array_equal(records.images[0], rgb8)
    assert records.times == [0, 1, 2]
    assert records.closed == 1
    assert records.options[0].output_category_mask
    assert not records.options[0].output_confidence_masks


def test_skin_labels_not_hardcoded_indices(monkeypatch):
    category = np.array([[0, 1, 2]], dtype=np.uint8)
    records = _fake_mediapipe(monkeypatch, category, ["face-skin", "other", "body-skin"])
    with skinmask.SkinSegmenter("model.tflite") as segmenter:
        np.testing.assert_array_equal(
            segmenter.mask(np.ones((1, 3, 3), dtype=np.uint8), 0), [[True, False, True]]
        )
    assert records.closed == 1


def test_negative_source_times_preserve_model_cadence(monkeypatch):
    records = _fake_mediapipe(monkeypatch, np.full((2, 3), 2, dtype=np.uint8))
    times = [-1.0, -0.9, -0.8]
    rgb = np.ones((2, 3, 3), dtype=np.uint8)
    frames = list(skinmask.skinmasked_frames(((rgb, t) for t in times), "model.tflite"))
    assert [t for _, t in frames] == times
    assert records.times == [0, 100, 200]


def test_model_labels_checked(monkeypatch):
    records = _fake_mediapipe(monkeypatch, None, ["background", "person"])
    with pytest.raises(ValueError, match="face-skin and body-skin"):
        skinmask.SkinSegmenter("model.tflite")
    assert records.closed == 1


@pytest.mark.parametrize("category", [None, np.zeros((3, 3), dtype=np.uint8)])
def test_bad_category_mask_fails_closed(monkeypatch, category):
    records = _fake_mediapipe(monkeypatch, category)
    with pytest.raises(ValueError, match="mask"):
        list(skinmask.skinmasked_frames([(np.ones((2, 3, 3), dtype=np.uint8), 0)], "model.tflite"))
    assert records.closed == 1


def test_segmenter_closed_on_decode_error(monkeypatch):
    records = _fake_mediapipe(monkeypatch, np.zeros((2, 3), dtype=np.uint8))

    def broken_frames():
        yield np.ones((2, 3, 3), dtype=np.uint8), 0
        raise ValueError("decode failed")

    with pytest.raises(ValueError, match="decode failed"):
        list(skinmask.skinmasked_frames(broken_frames(), "model.tflite"))
    assert records.closed == 1


def test_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    with pytest.raises(ImportError, match=r"rapidtide\[skinmask\]"):
        skinmask._mediapipe()


def _model_download(monkeypatch, tmp_path, data=b"skin weights"):
    _fake_mediapipe(monkeypatch, None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(skinmask, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return BytesIO(data)

    monkeypatch.setattr(skinmask, "urlopen", download)
    return calls


def test_model_download_and_cache(monkeypatch, tmp_path):
    calls = _model_download(monkeypatch, tmp_path)
    path, metadata = skinmask.prepare_skin_model()
    assert path.read_bytes() == b"skin weights"
    assert metadata["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["included_classes"] == ["face-skin", "body-skin"]
    assert metadata["landmark_refinement"] is False
    assert metadata["non_skin_rgb_value"] == 0
    assert skinmask.prepare_skin_model()[0] == path
    assert calls == [(skinmask.MODEL_URL, 60)]
    assert list(path.parent.iterdir()) == [path]
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Cached skin model checksum"):
        skinmask.prepare_skin_model()


def test_unverified_download_not_cached(monkeypatch, tmp_path):
    _model_download(monkeypatch, tmp_path)
    monkeypatch.setattr(skinmask, "MODEL_SHA256", "not the checksum")
    with pytest.raises(ValueError, match="Downloaded MediaPipe skin model checksum"):
        skinmask.prepare_skin_model()
    assert not list(tmp_path.rglob("*.tflite"))
    assert not list(tmp_path.rglob(".skin-model-*"))


def test_failed_download_cleanup(monkeypatch, tmp_path):
    _model_download(monkeypatch, tmp_path)

    def failed(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(skinmask, "urlopen", failed)
    with pytest.raises(OSError, match="--skinmaskmodel"):
        skinmask.prepare_skin_model()
    assert not list(tmp_path.rglob("*.tflite"))
    assert not list(tmp_path.rglob(".skin-model-*"))


def test_local_model_offline(monkeypatch, tmp_path):
    calls = _model_download(monkeypatch, tmp_path)
    model = tmp_path / "local.tflite"
    model.write_bytes(b"local model")
    path, metadata = skinmask.prepare_skin_model(model)
    assert path == model
    assert metadata["model_url"] is None
    assert metadata["model_sha256"] == hashlib.sha256(b"local model").hexdigest()
    assert "gaussian_lowpass" in metadata["calculated_before"]
    assert metadata["applied_after"] == ["gaussian_lowpass"]
    assert "gaussian_lowpass" not in metadata["applied_before"]
    assert not calls
    with pytest.raises(FileNotFoundError, match="Skin segmentation model"):
        skinmask.prepare_skin_model(tmp_path / "missing.tflite")
