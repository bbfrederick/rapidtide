#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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

from rapidtide.workflows.filttc import _get_parser, filttc

# ==================== Helpers ====================


def _make_default_args(inputfile, outputfile, **overrides):
    defaults = dict(
        inputfile=inputfile,
        outputfile=outputfile,
        samplerate="auto",
        normfirst=False,
        normmethod="None",
        demean=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _mock_readvectorsfromtextfile(samplerate, invecs, starttime=0.0, colnames=None):
    if colnames is None:
        colnames = ["tc0"]
    compressed = False
    filetype = "text"
    return (samplerate, starttime, colnames, invecs, compressed, filetype)


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "filttc"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".txt") as infile, tempfile.NamedTemporaryFile(
        suffix=".txt"
    ) as outfile:
        args = parser.parse_args(["--inputfile", infile.name, "--outputfile", outfile.name])
    assert args.samplerate == "auto"
    assert args.normfirst is False
    assert args.demean is False
    assert args.debug is False
    assert args.normmethod == "None"


# ==================== filttc tests ====================


def filttc_samplerate_from_args_when_file_none(debug=False):
    if debug:
        print("filttc_samplerate_from_args_when_file_none")
    invecs = np.array([[1.0, 2.0, 3.0, 4.0]])
    args = _make_default_args("in.txt", "out.txt", samplerate=2.0)

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, x: x)

    captured = {}

    def mock_writevectorstotextfile(outvecs, outputfile, **kwargs):
        captured["outvecs"] = outvecs.copy()
        captured["kwargs"] = kwargs

    with (
        patch(
            "rapidtide.workflows.filttc.pf.postprocessfilteropts",
            return_value=(args, mock_filter),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.readvectorsfromtextfile",
            return_value=_mock_readvectorsfromtextfile(None, invecs),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.writevectorstotextfile",
            side_effect=mock_writevectorstotextfile,
        ),
    ):
        filttc(args)

    assert np.allclose(captured["outvecs"], invecs)
    assert captured["kwargs"]["samplerate"] == 2.0


def filttc_samplerate_overrides_file(debug=False):
    if debug:
        print("filttc_samplerate_overrides_file")
    invecs = np.array([[1.0, 2.0, 3.0, 4.0]])
    args = _make_default_args("in.txt", "out.txt", samplerate=5.0)

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, x: x)

    captured = {}

    def mock_writevectorstotextfile(outvecs, outputfile, **kwargs):
        captured["kwargs"] = kwargs

    with (
        patch(
            "rapidtide.workflows.filttc.pf.postprocessfilteropts",
            return_value=(args, mock_filter),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.readvectorsfromtextfile",
            return_value=_mock_readvectorsfromtextfile(1.25, invecs),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.writevectorstotextfile",
            side_effect=mock_writevectorstotextfile,
        ),
    ):
        filttc(args)

    assert captured["kwargs"]["samplerate"] == 5.0


def filttc_samplerate_missing_auto_exit(debug=False):
    if debug:
        print("filttc_samplerate_missing_auto_exit")
    invecs = np.array([[1.0, 2.0, 3.0, 4.0]])
    args = _make_default_args("in.txt", "out.txt", samplerate="auto")

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, x: x)

    with (
        patch(
            "rapidtide.workflows.filttc.pf.postprocessfilteropts",
            return_value=(args, mock_filter),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.readvectorsfromtextfile",
            return_value=_mock_readvectorsfromtextfile(None, invecs),
        ),
    ):
        with pytest.raises(SystemExit):
            filttc(args)


def filttc_normfirst_true(debug=False):
    if debug:
        print("filttc_normfirst_true")
    invecs = np.array([[1.0, 2.0, 3.0, 4.0]])
    args = _make_default_args(
        "in.txt",
        "out.txt",
        samplerate=2.0,
        normfirst=True,
        normmethod="zscore",
        demean=False,
    )

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, x: x + 1.0)

    captured = {}

    def mock_writevectorstotextfile(outvecs, outputfile, **kwargs):
        captured["outvecs"] = outvecs.copy()

    with (
        patch(
            "rapidtide.workflows.filttc.pf.postprocessfilteropts",
            return_value=(args, mock_filter),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.readvectorsfromtextfile",
            return_value=_mock_readvectorsfromtextfile(2.0, invecs),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.writevectorstotextfile",
            side_effect=mock_writevectorstotextfile,
        ),
        patch(
            "rapidtide.workflows.filttc.tide_math.normalize",
            side_effect=lambda x, method=None: x * 2.0,
        ),
    ):
        filttc(args)

    expected = (invecs * 2.0) + 1.0
    np.testing.assert_allclose(captured["outvecs"], expected)


def filttc_demean_true(debug=False):
    if debug:
        print("filttc_demean_true")
    invecs = np.array([[1.0, 2.0, 3.0, 4.0]])
    args = _make_default_args(
        "in.txt",
        "out.txt",
        samplerate=2.0,
        normfirst=False,
        normmethod="None",
        demean=True,
    )

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, x: x)

    captured = {}

    def mock_writevectorstotextfile(outvecs, outputfile, **kwargs):
        captured["outvecs"] = outvecs.copy()

    with (
        patch(
            "rapidtide.workflows.filttc.pf.postprocessfilteropts",
            return_value=(args, mock_filter),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.readvectorsfromtextfile",
            return_value=_mock_readvectorsfromtextfile(2.0, invecs),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.writevectorstotextfile",
            side_effect=mock_writevectorstotextfile,
        ),
    ):
        filttc(args)

    expected = invecs - np.mean(invecs, axis=1, keepdims=True)
    np.testing.assert_allclose(captured["outvecs"], expected)


def filttc_normfirst_and_demean(debug=False):
    if debug:
        print("filttc_normfirst_and_demean")
    invecs = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    args = _make_default_args(
        "in.txt",
        "out.txt",
        samplerate=2.0,
        normfirst=True,
        normmethod="zscore",
        demean=True,
    )

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, x: x + 1.0)

    captured = {}

    def mock_writevectorstotextfile(outvecs, outputfile, **kwargs):
        captured["outvecs"] = outvecs.copy()

    with (
        patch(
            "rapidtide.workflows.filttc.pf.postprocessfilteropts",
            return_value=(args, mock_filter),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.readvectorsfromtextfile",
            return_value=_mock_readvectorsfromtextfile(2.0, invecs, colnames=["a", "b"]),
        ),
        patch(
            "rapidtide.workflows.filttc.tide_io.writevectorstotextfile",
            side_effect=mock_writevectorstotextfile,
        ),
        patch(
            "rapidtide.workflows.filttc.tide_math.normalize",
            side_effect=lambda x, method=None: x * 2.0,
        ),
    ):
        filttc(args)

    expected = (invecs * 2.0) + 1.0
    expected = expected - np.mean(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(captured["outvecs"], expected)


# ==================== Main test function ====================


def test_filttc(debug=False):
    # _get_parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)

    # filttc tests
    filttc_samplerate_from_args_when_file_none(debug=debug)
    filttc_samplerate_overrides_file(debug=debug)
    filttc_samplerate_missing_auto_exit(debug=debug)
    filttc_normfirst_true(debug=debug)
    filttc_demean_true(debug=debug)
    filttc_normfirst_and_demean(debug=debug)


if __name__ == "__main__":
    test_filttc(debug=True)
