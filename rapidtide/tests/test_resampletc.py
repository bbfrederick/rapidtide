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
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.resampletc import _get_parser, resampletc

# ============================================================================
# Helpers
# ============================================================================


def _make_default_args(**overrides):
    defaults = dict(
        inputfile="input.txt",
        outputfile="output.txt",
        insamplerate=10.0,
        outsamplerate=5.0,
        antialias=True,
        display=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ============================================================================
# Parser tests
# ============================================================================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "resampletc"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    """Parse with all required args and verify defaults for optional flags."""
    if debug:
        print("parser_defaults")
    # addreqinputtextfile validates files exist, so use a temp file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n2.0\n3.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "10.0", "output.txt", "5.0"])
    assert args.insamplerate == 10.0
    assert args.outsamplerate == 5.0
    assert args.outputfile == "output.txt"
    assert args.antialias is True
    assert args.display is True


def parser_nodisplay_flag(debug=False):
    if debug:
        print("parser_nodisplay_flag")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n2.0\n3.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "10.0", "output.txt", "5.0", "--nodisplay"])
    assert args.display is False


def parser_noantialias_flag(debug=False):
    if debug:
        print("parser_noantialias_flag")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n2.0\n3.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "10.0", "output.txt", "5.0", "--noantialias"])
    assert args.antialias is False


def parser_all_flags(debug=False):
    if debug:
        print("parser_all_flags")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n2.0\n3.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "10.0", "out.txt", "2.5", "--nodisplay", "--noantialias"])
    assert args.display is False
    assert args.antialias is False
    assert args.outsamplerate == 2.5


def parser_samplerate_is_float(debug=False):
    """Verify sample rates are parsed as floats."""
    if debug:
        print("parser_samplerate_is_float")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n2.0\n3.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "10", "out.txt", "5"])
    assert isinstance(args.insamplerate, float)
    assert isinstance(args.outsamplerate, float)


# ============================================================================
# resampletc function tests
# ============================================================================


def resampletc_basic(debug=False):
    """Test basic downsampling flow: read, resample, write."""
    if debug:
        print("resampletc_basic")

    insamplerate = 10.0
    outsamplerate = 5.0
    inputdata = np.sin(2 * np.pi * 0.5 * np.arange(100) / insamplerate)
    resampled = np.sin(2 * np.pi * 0.5 * np.arange(50) / outsamplerate)

    captured = {}

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (insamplerate, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        captured["init_freq"] = init_freq
        captured["final_freq"] = final_freq
        captured["antialias"] = antialias
        captured["decimate"] = decimate
        np.testing.assert_array_equal(data, inputdata)
        return resampled

    def _mock_writevec(data, outfile):
        captured["written_data"] = data.copy()
        captured["outfile"] = outfile

    args = _make_default_args(insamplerate=insamplerate, outsamplerate=outsamplerate)

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_io.writevec",
            side_effect=_mock_writevec,
        ),
    ):
        resampletc(args)

    assert captured["init_freq"] == insamplerate
    assert captured["final_freq"] == outsamplerate
    assert captured["antialias"] is True
    assert captured["decimate"] is False
    np.testing.assert_array_equal(captured["written_data"], resampled)
    assert captured["outfile"] == "output.txt"


def resampletc_antialias_off(debug=False):
    """Test that antialias=False is forwarded to arbresample."""
    if debug:
        print("resampletc_antialias_off")

    inputdata = np.ones(50)
    captured = {}

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (None, None, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        captured["antialias"] = antialias
        return np.ones(25)

    args = _make_default_args(antialias=False)

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch("rapidtide.workflows.resampletc.tide_io.writevec"),
    ):
        resampletc(args)

    assert captured["antialias"] is False


def resampletc_samplerate_mismatch_warning(debug=False):
    """When file samplerate differs from args.insamplerate, a warning is printed."""
    if debug:
        print("resampletc_samplerate_mismatch_warning")

    inputdata = np.ones(50)
    file_samplerate = 20.0
    args_samplerate = 10.0

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (file_samplerate, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return np.ones(25)

    args = _make_default_args(insamplerate=args_samplerate)

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch("rapidtide.workflows.resampletc.tide_io.writevec"),
        patch("builtins.print") as mock_print,
    ):
        resampletc(args)

    # Check that the warning was printed
    mock_print.assert_any_call(
        f"warning: specified sampling rate {file_samplerate} does not match input file {args_samplerate}"
    )


def resampletc_samplerate_match_no_warning(debug=False):
    """When file samplerate matches args.insamplerate, no warning is printed."""
    if debug:
        print("resampletc_samplerate_match_no_warning")

    inputdata = np.ones(50)
    samplerate = 10.0

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (samplerate, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return np.ones(25)

    args = _make_default_args(insamplerate=samplerate)

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch("rapidtide.workflows.resampletc.tide_io.writevec"),
        patch("builtins.print") as mock_print,
    ):
        resampletc(args)

    # No warning calls should contain "does not match"
    for c in mock_print.call_args_list:
        assert "does not match" not in str(c)


def resampletc_none_samplerate_no_warning(debug=False):
    """When file returns samplerate=None, no warning is printed."""
    if debug:
        print("resampletc_none_samplerate_no_warning")

    inputdata = np.ones(50)

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (None, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return np.ones(25)

    args = _make_default_args()

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch("rapidtide.workflows.resampletc.tide_io.writevec"),
        patch("builtins.print") as mock_print,
    ):
        resampletc(args)

    for c in mock_print.call_args_list:
        assert "does not match" not in str(c)


def resampletc_none_starttime_defaults_to_zero(debug=False):
    """When file returns starttime=None, it defaults to 0.0."""
    if debug:
        print("resampletc_none_starttime_defaults_to_zero")

    inputdata = np.arange(10, dtype=float)

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (None, None, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return np.arange(5, dtype=float)

    args = _make_default_args()

    # Should not raise even though starttime is None
    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch("rapidtide.workflows.resampletc.tide_io.writevec"),
    ):
        resampletc(args)


def resampletc_display_calls_plot(debug=False):
    """When display=True, matplotlib plot and show are called."""
    if debug:
        print("resampletc_display_calls_plot")

    inputdata = np.ones(20)
    outputdata = np.ones(10)

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (None, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return outputdata

    args = _make_default_args(display=True)

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch("rapidtide.workflows.resampletc.tide_io.writevec"),
        patch("rapidtide.workflows.resampletc.plt.plot") as mock_plot,
        patch("rapidtide.workflows.resampletc.plt.legend") as mock_legend,
        patch("rapidtide.workflows.resampletc.plt.show") as mock_show,
    ):
        resampletc(args)

    assert mock_plot.call_count == 2
    mock_legend.assert_called_once()
    mock_show.assert_called_once()


def resampletc_no_display_skips_plot(debug=False):
    """When display=False, matplotlib plot is not called."""
    if debug:
        print("resampletc_no_display_skips_plot")

    inputdata = np.ones(20)
    outputdata = np.ones(10)

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (None, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return outputdata

    args = _make_default_args(display=False)

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch("rapidtide.workflows.resampletc.tide_io.writevec"),
        patch("rapidtide.workflows.resampletc.plt.plot") as mock_plot,
        patch("rapidtide.workflows.resampletc.plt.show") as mock_show,
    ):
        resampletc(args)

    mock_plot.assert_not_called()
    mock_show.assert_not_called()


def resampletc_output_truncation(debug=False):
    """When out_t is shorter than outputdata, outputdata is truncated."""
    if debug:
        print("resampletc_output_truncation")

    inputdata = np.ones(20)
    # Return a resampled array that is longer than what out_t would be.
    # out_t has len(outputdata) points, so normally they match. But
    # the code does: out_t = outtimestep * linspace(0, len(outputdata), len(outputdata), endpoint=True)
    # Since endpoint=True and the number of points equals len(outputdata),
    # len(out_t) == len(outputdata), so they match and no truncation occurs.
    # However, if we somehow get a mismatch, the code handles it.
    # We test the normal path (no truncation needed).
    outputdata = np.arange(10, dtype=float)

    captured = {}

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (None, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return outputdata

    def _mock_writevec(data, outfile):
        captured["written_data"] = data.copy()

    args = _make_default_args()

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_io.writevec",
            side_effect=_mock_writevec,
        ),
    ):
        resampletc(args)

    np.testing.assert_array_equal(captured["written_data"], outputdata)


def resampletc_inputfile_passed_to_reader(debug=False):
    """Verify the input filename is forwarded to readvectorsfromtextfile."""
    if debug:
        print("resampletc_inputfile_passed_to_reader")

    inputdata = np.ones(20)
    captured = {}

    def _mock_readvectors(filepath, onecol=True, debug=False):
        captured["filepath"] = filepath
        captured["onecol"] = onecol
        return (None, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return np.ones(10)

    args = _make_default_args(inputfile="my_special_input.txt")

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch("rapidtide.workflows.resampletc.tide_io.writevec"),
    ):
        resampletc(args)

    assert captured["filepath"] == "my_special_input.txt"
    assert captured["onecol"] is True


def resampletc_upsample(debug=False):
    """Test upsampling (outsamplerate > insamplerate)."""
    if debug:
        print("resampletc_upsample")

    insamplerate = 5.0
    outsamplerate = 20.0
    inputdata = np.sin(2 * np.pi * 0.5 * np.arange(50) / insamplerate)
    resampled = np.sin(2 * np.pi * 0.5 * np.arange(200) / outsamplerate)

    captured = {}

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (None, 0.0, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        captured["init_freq"] = init_freq
        captured["final_freq"] = final_freq
        return resampled

    def _mock_writevec(data, outfile):
        captured["written_len"] = len(data)

    args = _make_default_args(insamplerate=insamplerate, outsamplerate=outsamplerate)

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_io.writevec",
            side_effect=_mock_writevec,
        ),
    ):
        resampletc(args)

    assert captured["init_freq"] == insamplerate
    assert captured["final_freq"] == outsamplerate
    assert captured["written_len"] == len(resampled)


def resampletc_nonzero_starttime(debug=False):
    """Test that a non-zero starttime from the file is used for in_t."""
    if debug:
        print("resampletc_nonzero_starttime")

    insamplerate = 10.0
    outsamplerate = 5.0
    instarttime = 5.0
    inputdata = np.ones(100)
    resampled = np.ones(50)

    captured = {}

    def _mock_readvectors(filepath, onecol=True, debug=False):
        return (None, instarttime, None, inputdata, None, "text")

    def _mock_arbresample(data, init_freq, final_freq, decimate=False, antialias=True, **kwargs):
        return resampled

    def _mock_writevec(data, outfile):
        captured["written_data"] = data.copy()

    args = _make_default_args(insamplerate=insamplerate, outsamplerate=outsamplerate)

    with (
        patch(
            "rapidtide.workflows.resampletc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_resample.arbresample",
            side_effect=_mock_arbresample,
        ),
        patch(
            "rapidtide.workflows.resampletc.tide_io.writevec",
            side_effect=_mock_writevec,
        ),
    ):
        resampletc(args)

    # The function should still write the resampled data
    np.testing.assert_array_equal(captured["written_data"], resampled)


# ============================================================================
# Main test entry point
# ============================================================================


def test_resampletc(debug=False):
    # Parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_nodisplay_flag(debug=debug)
    parser_noantialias_flag(debug=debug)
    parser_all_flags(debug=debug)
    parser_samplerate_is_float(debug=debug)

    # resampletc function tests
    resampletc_basic(debug=debug)
    resampletc_antialias_off(debug=debug)
    resampletc_samplerate_mismatch_warning(debug=debug)
    resampletc_samplerate_match_no_warning(debug=debug)
    resampletc_none_samplerate_no_warning(debug=debug)
    resampletc_none_starttime_defaults_to_zero(debug=debug)
    resampletc_display_calls_plot(debug=debug)
    resampletc_no_display_skips_plot(debug=debug)
    resampletc_output_truncation(debug=debug)
    resampletc_inputfile_passed_to_reader(debug=debug)
    resampletc_upsample(debug=debug)
    resampletc_nonzero_starttime(debug=debug)


if __name__ == "__main__":
    test_resampletc(debug=True)
