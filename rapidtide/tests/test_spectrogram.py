#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
import argparse
import io
import tempfile
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rapidtide.workflows.spectrogram import (
    _get_parser,
    calcspecgram,
    make_legend_axes,
    ndplot,
    showspecgram,
    spectrogram,
)

# ---- helpers ----


def _make_test_signal(samplerate=100.0, duration=5.0, freq=10.0):
    """Create a simple sinusoidal test signal.

    Returns (signal, time_vector).
    """
    npts = int(samplerate * duration)
    t = np.arange(npts) / samplerate
    x = np.sin(2.0 * np.pi * freq * t)
    return x, t


def _make_chirp_signal(samplerate=200.0, duration=5.0, f0=5.0, f1=50.0):
    """Create a chirp signal sweeping from f0 to f1.

    Returns (signal, time_vector).
    """
    npts = int(samplerate * duration)
    t = np.arange(npts) / samplerate
    x = np.sin(2.0 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2.0 * duration)))
    return x, t


def _make_default_args(**overrides):
    """Create default args Namespace for spectrogram workflow."""
    defaults = dict(
        inputfile="input.txt",
        nperseg=128,
        samplerate=100.0,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---- _get_parser tests ----


def parser_basic(debug=False):
    """Test that _get_parser returns a valid parser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "spectrogram"

    if debug:
        print("parser_basic passed")


def parser_required_args(debug=False):
    """Test that parser requires inputfile."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    if debug:
        print("parser_required_args passed")


def parser_defaults(debug=False):
    """Test default values from the parser."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        args = parser.parse_args([f.name])

    assert args.nperseg == 128
    assert args.samplerate is None
    assert args.debug is False

    if debug:
        print("parser_defaults passed")


def parser_nperseg(debug=False):
    """Test --nperseg option."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        args = parser.parse_args([f.name, "--nperseg", "256"])
    assert args.nperseg == 256

    if debug:
        print("parser_nperseg passed")


def parser_samplerate(debug=False):
    """Test --samplerate option."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        args = parser.parse_args([f.name, "--samplerate", "44100.0"])
    assert np.isclose(args.samplerate, 44100.0)

    if debug:
        print("parser_samplerate passed")


def parser_debug(debug=False):
    """Test --debug flag."""
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        args = parser.parse_args([f.name, "--debug"])
    assert args.debug is True

    if debug:
        print("parser_debug passed")


def parser_inputfile_validation(debug=False):
    """Test that inputfile must exist (is_valid_file check)."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["/nonexistent/file.txt"])

    if debug:
        print("parser_inputfile_validation passed")


# ---- calcspecgram tests ----


def calcspecgram_basic(debug=False):
    """Test calcspecgram returns correct output types and shapes."""
    x, t = _make_test_signal(samplerate=100.0, duration=5.0, freq=10.0)

    freq, segtimes, thestft, isinvertable = calcspecgram(x, t, nperseg=32)

    assert isinstance(freq, np.ndarray)
    assert isinstance(segtimes, np.ndarray)
    assert isinstance(thestft, np.ndarray)
    assert isinstance(isinvertable, (bool, np.bool_))

    # freq should be 1D
    assert freq.ndim == 1
    # segtimes should be 1D
    assert segtimes.ndim == 1
    # thestft should be 2D (freq x time)
    assert thestft.ndim == 2
    # first dim of stft should match freq length
    assert thestft.shape[0] == len(freq)
    # second dim of stft should match segtimes length
    assert thestft.shape[1] == len(segtimes)

    if debug:
        print(f"calcspecgram_basic: freq shape={freq.shape}, "
              f"segtimes shape={segtimes.shape}, stft shape={thestft.shape}")
        print("calcspecgram_basic passed")


def calcspecgram_frequency_range(debug=False):
    """Test that frequency range is correct for given sample rate."""
    samplerate = 200.0
    x, t = _make_test_signal(samplerate=samplerate, duration=5.0, freq=10.0)

    freq, segtimes, thestft, isinvertable = calcspecgram(x, t, nperseg=64)

    # onesided FFT: max freq should be Nyquist = samplerate/2
    assert freq[0] >= 0.0
    assert freq[-1] <= samplerate / 2.0

    if debug:
        print(f"calcspecgram_frequency_range: freq range [{freq[0]:.2f}, {freq[-1]:.2f}]")
        print("calcspecgram_frequency_range passed")


def calcspecgram_invertible(debug=False):
    """Test that the NOLA condition is satisfied with default settings."""
    x, t = _make_test_signal(samplerate=100.0, duration=5.0, freq=10.0)

    freq, segtimes, thestft, isinvertable = calcspecgram(x, t, nperseg=32)

    assert bool(isinvertable), "Spectrogram with hann window should be invertible"

    if debug:
        print("calcspecgram_invertible passed")


def calcspecgram_nperseg_affects_resolution(debug=False):
    """Test that different nperseg values produce different frequency resolutions."""
    x, t = _make_test_signal(samplerate=100.0, duration=5.0, freq=10.0)

    freq_small, _, stft_small, _ = calcspecgram(x, t, nperseg=16)
    freq_large, _, stft_large, _ = calcspecgram(x, t, nperseg=64)

    # larger nperseg should give more frequency bins
    assert len(freq_large) > len(freq_small)

    if debug:
        print(f"calcspecgram_nperseg: small={len(freq_small)} bins, "
              f"large={len(freq_large)} bins")
        print("calcspecgram_nperseg_affects_resolution passed")


def calcspecgram_stft_is_complex(debug=False):
    """Test that the STFT output is complex."""
    x, t = _make_test_signal(samplerate=100.0, duration=5.0, freq=10.0)

    freq, segtimes, thestft, isinvertable = calcspecgram(x, t, nperseg=32)

    assert np.iscomplexobj(thestft), "STFT output should be complex"

    if debug:
        print("calcspecgram_stft_is_complex passed")


def calcspecgram_peak_at_signal_freq(debug=False):
    """Test that the STFT has a peak near the signal frequency."""
    signal_freq = 10.0
    x, t = _make_test_signal(samplerate=100.0, duration=5.0, freq=signal_freq)

    freq, segtimes, thestft, isinvertable = calcspecgram(x, t, nperseg=64)

    # average magnitude across time to find dominant frequency
    avg_mag = np.mean(np.abs(thestft), axis=1)
    peak_freq = freq[np.argmax(avg_mag)]

    # peak should be near the signal frequency (within frequency resolution)
    freq_resolution = freq[1] - freq[0]
    assert abs(peak_freq - signal_freq) <= freq_resolution, (
        f"Peak at {peak_freq:.2f} Hz, expected near {signal_freq:.2f} Hz"
    )

    if debug:
        print(f"calcspecgram_peak_at_signal_freq: peak at {peak_freq:.2f} Hz")
        print("calcspecgram_peak_at_signal_freq passed")


def calcspecgram_different_window(debug=False):
    """Test calcspecgram with a different window type."""
    x, t = _make_test_signal(samplerate=100.0, duration=5.0, freq=10.0)

    freq, segtimes, thestft, isinvertable = calcspecgram(
        x, t, nperseg=32, windowtype="hamming"
    )

    assert len(freq) > 0
    assert thestft.shape[0] == len(freq)
    assert thestft.shape[1] == len(segtimes)

    if debug:
        print("calcspecgram_different_window passed")


def calcspecgram_short_signal(debug=False):
    """Test calcspecgram with a signal only slightly longer than nperseg."""
    samplerate = 100.0
    npts = 40  # just above nperseg=32
    t = np.arange(npts) / samplerate
    x = np.sin(2.0 * np.pi * 10.0 * t)

    freq, segtimes, thestft, isinvertable = calcspecgram(x, t, nperseg=32)

    assert len(freq) > 0
    assert len(segtimes) > 0
    assert thestft.shape[0] == len(freq)

    if debug:
        print("calcspecgram_short_signal passed")


# ---- make_legend_axes tests ----


def make_legend_axes_basic(debug=False):
    """Test that make_legend_axes returns an axes object."""
    fig, ax = plt.subplots()
    legend_ax = make_legend_axes(ax)

    assert legend_ax is not None
    # the returned object should be a matplotlib axes
    assert hasattr(legend_ax, "set_visible")

    plt.close(fig)

    if debug:
        print("make_legend_axes_basic passed")


def make_legend_axes_separate(debug=False):
    """Test that make_legend_axes creates a distinct axes from the original."""
    fig, ax = plt.subplots()
    legend_ax = make_legend_axes(ax)

    assert legend_ax is not ax

    plt.close(fig)

    if debug:
        print("make_legend_axes_separate passed")


# ---- showspecgram tests ----


def showspecgram_mag(debug=False):
    """Test showspecgram in magnitude mode."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    freq, segtimes, thestft, _ = calcspecgram(x, t, nperseg=32)

    fig, ax = plt.subplots()
    im, cbar = showspecgram(thestft, segtimes, freq, ax, fig, mode="mag")

    assert im is not None
    assert cbar is not None

    plt.close(fig)

    if debug:
        print("showspecgram_mag passed")


def showspecgram_phase(debug=False):
    """Test showspecgram in phase mode."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    freq, segtimes, thestft, _ = calcspecgram(x, t, nperseg=32)

    fig, ax = plt.subplots()
    im, cbar = showspecgram(thestft, segtimes, freq, ax, fig, mode="phase")

    assert im is not None
    assert cbar is not None

    plt.close(fig)

    if debug:
        print("showspecgram_phase passed")


def showspecgram_real(debug=False):
    """Test showspecgram in real mode."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    freq, segtimes, thestft, _ = calcspecgram(x, t, nperseg=32)

    fig, ax = plt.subplots()
    im, cbar = showspecgram(thestft, segtimes, freq, ax, fig, mode="real")

    assert im is not None
    assert cbar is not None

    plt.close(fig)

    if debug:
        print("showspecgram_real passed")


def showspecgram_imag(debug=False):
    """Test showspecgram in imaginary mode."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    freq, segtimes, thestft, _ = calcspecgram(x, t, nperseg=32)

    fig, ax = plt.subplots()
    im, cbar = showspecgram(thestft, segtimes, freq, ax, fig, mode="imag")

    assert im is not None
    assert cbar is not None

    plt.close(fig)

    if debug:
        print("showspecgram_imag passed")


def showspecgram_invalid_mode(debug=False):
    """Test showspecgram exits with invalid mode."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    freq, segtimes, thestft, _ = calcspecgram(x, t, nperseg=32)

    fig, ax = plt.subplots()
    with pytest.raises(SystemExit):
        showspecgram(thestft, segtimes, freq, ax, fig, mode="invalid")

    plt.close(fig)

    if debug:
        print("showspecgram_invalid_mode passed")


def showspecgram_ylim(debug=False):
    """Test that showspecgram sets ylim from freq[1] to freq.max()."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    freq, segtimes, thestft, _ = calcspecgram(x, t, nperseg=32)

    fig, ax = plt.subplots()
    showspecgram(thestft, segtimes, freq, ax, fig, mode="mag")

    ymin, ymax = ax.get_ylim()
    assert np.isclose(ymin, freq[1]), f"ymin should be freq[1]={freq[1]}, got {ymin}"
    assert np.isclose(ymax, freq.max()), f"ymax should be freq.max()={freq.max()}, got {ymax}"

    plt.close(fig)

    if debug:
        print("showspecgram_ylim passed")


def showspecgram_prints_dimensions(debug=False):
    """Test that showspecgram prints time dimension info."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    freq, segtimes, thestft, _ = calcspecgram(x, t, nperseg=32)

    fig, ax = plt.subplots()
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        showspecgram(thestft, segtimes, freq, ax, fig, mode="mag")

    output = captured.getvalue()
    # showspecgram prints len(t), len(time)
    assert str(len(segtimes)) in output

    plt.close(fig)

    if debug:
        print("showspecgram_prints_dimensions passed")


# ---- ndplot tests ----


def ndplot_basic(debug=False):
    """Test that ndplot runs without error."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        ndplot(x, t, "Test Signal", nperseg=32)

    output = captured.getvalue()
    assert "arrived in ndplot" in output
    assert "invertable" in output

    plt.close("all")

    if debug:
        print("ndplot_basic passed")


def ndplot_creates_subplots(debug=False):
    """Test that ndplot creates the expected number of subplots."""
    x, t = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)

    ndplot(x, t, "Test Signal", nperseg=32)

    # ndplot creates a figure with 3 subplots (311, 312, 313)
    fig = plt.gcf()
    # should have at least 3 axes (plus legend axes)
    assert len(fig.get_axes()) >= 3

    plt.close("all")

    if debug:
        print("ndplot_creates_subplots passed")


def ndplot_different_nperseg(debug=False):
    """Test ndplot with different nperseg values."""
    x, t = _make_test_signal(samplerate=100.0, duration=3.0, freq=10.0)

    # should run without error for various nperseg
    for nperseg in [16, 32, 64]:
        ndplot(x, t, f"nperseg={nperseg}", nperseg=nperseg)
        plt.close("all")

    if debug:
        print("ndplot_different_nperseg passed")


# ---- spectrogram workflow tests ----


def spectrogram_basic(debug=False):
    """Test spectrogram workflow with mocked I/O and parser."""
    x, t = _make_test_signal(samplerate=100.0, duration=3.0, freq=10.0)
    args = _make_default_args(samplerate=100.0, nperseg=32)

    def mock_readvectorsfromtextfile(fname, onecol=True):
        return (100.0, 0.0, None, x, False, "text")

    with (
        patch(
            "rapidtide.workflows.spectrogram._get_parser",
            return_value=MagicMock(
                parse_args=MagicMock(return_value=args),
                print_help=MagicMock(),
            ),
        ),
        patch(
            "rapidtide.workflows.spectrogram.tide_io.readvectorsfromtextfile",
            side_effect=mock_readvectorsfromtextfile,
        ),
        patch("rapidtide.workflows.spectrogram.plt.show"),
    ):
        spectrogram(args)

    plt.close("all")

    if debug:
        print("spectrogram_basic passed")


def spectrogram_samplerate_from_args(debug=False):
    """Test that --samplerate overrides file sample rate."""
    x, t = _make_test_signal(samplerate=100.0, duration=3.0, freq=10.0)
    args = _make_default_args(samplerate=200.0, nperseg=32)

    captured_xvec = {}

    original_ndplot = ndplot

    def mock_ndplot(yvec, xvec, label, nperseg=32):
        captured_xvec["xvec"] = xvec.copy()
        captured_xvec["yvec"] = yvec.copy()

    def mock_readvectorsfromtextfile(fname, onecol=True):
        # file says 100Hz but args say 200Hz
        return (100.0, 0.0, None, x, False, "text")

    with (
        patch(
            "rapidtide.workflows.spectrogram._get_parser",
            return_value=MagicMock(
                parse_args=MagicMock(return_value=args),
                print_help=MagicMock(),
            ),
        ),
        patch(
            "rapidtide.workflows.spectrogram.tide_io.readvectorsfromtextfile",
            side_effect=mock_readvectorsfromtextfile,
        ),
        patch("rapidtide.workflows.spectrogram.ndplot", side_effect=mock_ndplot),
        patch("rapidtide.workflows.spectrogram.plt.show"),
    ):
        spectrogram(args)

    # timestep = 1/200 = 0.005, xvec[1] - xvec[0] should be 0.005
    xvec = captured_xvec["xvec"]
    timestep = xvec[1] - xvec[0]
    assert np.isclose(timestep, 1.0 / 200.0), (
        f"Expected timestep=0.005, got {timestep}"
    )

    if debug:
        print("spectrogram_samplerate_from_args passed")


def spectrogram_samplerate_from_file(debug=False):
    """Test that file sample rate is used when --samplerate is not set."""
    x, _ = _make_test_signal(samplerate=50.0, duration=3.0, freq=10.0)
    args = _make_default_args(samplerate=None, nperseg=32)

    captured_xvec = {}

    def mock_ndplot(yvec, xvec, label, nperseg=32):
        captured_xvec["xvec"] = xvec.copy()

    def mock_readvectorsfromtextfile(fname, onecol=True):
        return (50.0, 0.0, None, x, False, "text")

    with (
        patch(
            "rapidtide.workflows.spectrogram._get_parser",
            return_value=MagicMock(
                parse_args=MagicMock(return_value=args),
                print_help=MagicMock(),
            ),
        ),
        patch(
            "rapidtide.workflows.spectrogram.tide_io.readvectorsfromtextfile",
            side_effect=mock_readvectorsfromtextfile,
        ),
        patch("rapidtide.workflows.spectrogram.ndplot", side_effect=mock_ndplot),
        patch("rapidtide.workflows.spectrogram.plt.show"),
    ):
        spectrogram(args)

    xvec = captured_xvec["xvec"]
    timestep = xvec[1] - xvec[0]
    assert np.isclose(timestep, 1.0 / 50.0), (
        f"Expected timestep=0.02, got {timestep}"
    )

    if debug:
        print("spectrogram_samplerate_from_file passed")


def spectrogram_no_samplerate_exits(debug=False):
    """Test that spectrogram exits when no sample rate is available."""
    x, _ = _make_test_signal(samplerate=100.0, duration=3.0, freq=10.0)
    args = _make_default_args(samplerate=None, nperseg=32)

    def mock_readvectorsfromtextfile(fname, onecol=True):
        return (None, 0.0, None, x, False, "text")

    with pytest.raises(SystemExit):
        with (
            patch(
                "rapidtide.workflows.spectrogram._get_parser",
                return_value=MagicMock(
                    parse_args=MagicMock(return_value=args),
                    print_help=MagicMock(),
                ),
            ),
            patch(
                "rapidtide.workflows.spectrogram.tide_io.readvectorsfromtextfile",
                side_effect=mock_readvectorsfromtextfile,
            ),
            patch("rapidtide.workflows.spectrogram.plt.show"),
        ):
            spectrogram(args)

    if debug:
        print("spectrogram_no_samplerate_exits passed")


def spectrogram_starttime(debug=False):
    """Test that starttime from file offsets the time vector."""
    x, _ = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    args = _make_default_args(samplerate=100.0, nperseg=32)

    captured_xvec = {}

    def mock_ndplot(yvec, xvec, label, nperseg=32):
        captured_xvec["xvec"] = xvec.copy()

    def mock_readvectorsfromtextfile(fname, onecol=True):
        return (100.0, 5.0, None, x, False, "text")

    with (
        patch(
            "rapidtide.workflows.spectrogram._get_parser",
            return_value=MagicMock(
                parse_args=MagicMock(return_value=args),
                print_help=MagicMock(),
            ),
        ),
        patch(
            "rapidtide.workflows.spectrogram.tide_io.readvectorsfromtextfile",
            side_effect=mock_readvectorsfromtextfile,
        ),
        patch("rapidtide.workflows.spectrogram.ndplot", side_effect=mock_ndplot),
        patch("rapidtide.workflows.spectrogram.plt.show"),
    ):
        spectrogram(args)

    xvec = captured_xvec["xvec"]
    assert np.isclose(xvec[0], 5.0), f"Expected xvec[0]=5.0, got {xvec[0]}"

    if debug:
        print("spectrogram_starttime passed")


def spectrogram_debug_output(debug=False):
    """Test that debug mode prints args."""
    x, _ = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    args = _make_default_args(samplerate=100.0, nperseg=32, debug=True)

    def mock_readvectorsfromtextfile(fname, onecol=True):
        return (100.0, 0.0, None, x, False, "text")

    captured = io.StringIO()
    with (
        patch(
            "rapidtide.workflows.spectrogram._get_parser",
            return_value=MagicMock(
                parse_args=MagicMock(return_value=args),
                print_help=MagicMock(),
            ),
        ),
        patch(
            "rapidtide.workflows.spectrogram.tide_io.readvectorsfromtextfile",
            side_effect=mock_readvectorsfromtextfile,
        ),
        patch("rapidtide.workflows.spectrogram.plt.show"),
        patch("sys.stdout", captured),
    ):
        spectrogram(args)

    output = captured.getvalue()
    # debug prints args object
    assert "nperseg" in output or "samplerate" in output

    plt.close("all")

    if debug:
        print("spectrogram_debug_output passed")


def spectrogram_uses_inputfile_as_label(debug=False):
    """Test that spectrogram uses inputfile name as the plot label."""
    x, _ = _make_test_signal(samplerate=100.0, duration=2.0, freq=10.0)
    args = _make_default_args(
        inputfile="my_data.txt", samplerate=100.0, nperseg=32,
    )

    captured_label = {}

    def mock_ndplot(yvec, xvec, label, nperseg=32):
        captured_label["label"] = label

    def mock_readvectorsfromtextfile(fname, onecol=True):
        return (100.0, 0.0, None, x, False, "text")

    with (
        patch(
            "rapidtide.workflows.spectrogram._get_parser",
            return_value=MagicMock(
                parse_args=MagicMock(return_value=args),
                print_help=MagicMock(),
            ),
        ),
        patch(
            "rapidtide.workflows.spectrogram.tide_io.readvectorsfromtextfile",
            side_effect=mock_readvectorsfromtextfile,
        ),
        patch("rapidtide.workflows.spectrogram.ndplot", side_effect=mock_ndplot),
        patch("rapidtide.workflows.spectrogram.plt.show"),
    ):
        spectrogram(args)

    assert captured_label["label"] == "my_data.txt"

    if debug:
        print("spectrogram_uses_inputfile_as_label passed")


# ---- main test function ----


def test_spectrogram(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_nperseg(debug=debug)
    parser_samplerate(debug=debug)
    parser_debug(debug=debug)
    parser_inputfile_validation(debug=debug)

    # calcspecgram tests
    calcspecgram_basic(debug=debug)
    calcspecgram_frequency_range(debug=debug)
    calcspecgram_invertible(debug=debug)
    calcspecgram_nperseg_affects_resolution(debug=debug)
    calcspecgram_stft_is_complex(debug=debug)
    calcspecgram_peak_at_signal_freq(debug=debug)
    calcspecgram_different_window(debug=debug)
    calcspecgram_short_signal(debug=debug)

    # make_legend_axes tests
    make_legend_axes_basic(debug=debug)
    make_legend_axes_separate(debug=debug)

    # showspecgram tests
    showspecgram_mag(debug=debug)
    showspecgram_phase(debug=debug)
    showspecgram_real(debug=debug)
    showspecgram_imag(debug=debug)
    showspecgram_invalid_mode(debug=debug)
    showspecgram_ylim(debug=debug)
    showspecgram_prints_dimensions(debug=debug)

    # ndplot tests
    ndplot_basic(debug=debug)
    ndplot_creates_subplots(debug=debug)
    ndplot_different_nperseg(debug=debug)

    # spectrogram workflow tests
    spectrogram_basic(debug=debug)
    spectrogram_samplerate_from_args(debug=debug)
    spectrogram_samplerate_from_file(debug=debug)
    spectrogram_no_samplerate_exits(debug=debug)
    spectrogram_starttime(debug=debug)
    spectrogram_debug_output(debug=debug)
    spectrogram_uses_inputfile_as_label(debug=debug)


if __name__ == "__main__":
    test_spectrogram(debug=True)
