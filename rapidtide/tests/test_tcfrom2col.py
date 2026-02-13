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
from unittest.mock import patch

import numpy as np
import pytest

from rapidtide.workflows.tcfrom2col import _get_parser, tcfrom2col


def _make_default_args(**overrides):
    defaults = dict(
        infilename="input_2col.txt",
        timestep=0.5,
        numpoints=6,
        outfilename="output_tc.txt",
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "tcfrom2col"


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
    args = parser.parse_args(["in.txt", "0.25", "10", "out.txt"])
    assert args.infilename == "in.txt"
    assert args.timestep == 0.25
    assert args.numpoints == 10
    assert args.outfilename == "out.txt"
    assert args.debug is False


def parser_debug_flag(debug=False):
    if debug:
        print("parser_debug_flag")
    parser = _get_parser()
    args = parser.parse_args(["in.txt", "0.25", "10", "out.txt", "--debug"])
    assert args.debug is True


def tcfrom2col_basic(debug=False):
    if debug:
        print("tcfrom2col_basic")
    args = _make_default_args(debug=False)
    inputdata = np.array(
        [
            [0.0, 10.0],
            [0.5, 20.0],
            [1.0, 30.0],
        ]
    )
    expected_timeaxis = np.arange(0.0, args.numpoints * args.timestep, args.timestep)
    expected_output = expected_timeaxis + 100.0

    captured = {}

    def _mock_maketcfrom2col(indata, timeaxis, outdata, debug=False):
        captured["indata"] = np.array(indata, copy=True)
        captured["timeaxis"] = np.array(timeaxis, copy=True)
        captured["outdata_initial"] = np.array(outdata, copy=True)
        captured["debug"] = debug
        return expected_output

    def _mock_writenpvecs(outvec, outname):
        captured["outvec"] = np.array(outvec, copy=True)
        captured["outname"] = outname

    with (
        patch("rapidtide.workflows.tcfrom2col.tide_io.readvecs", return_value=inputdata),
        patch(
            "rapidtide.workflows.tcfrom2col.tide_util.maketcfrom2col",
            side_effect=_mock_maketcfrom2col,
        ),
        patch("rapidtide.workflows.tcfrom2col.tide_io.writenpvecs", side_effect=_mock_writenpvecs),
    ):
        tcfrom2col(args)

    np.testing.assert_allclose(captured["indata"], inputdata)
    np.testing.assert_allclose(captured["timeaxis"], expected_timeaxis)
    np.testing.assert_allclose(captured["outdata_initial"], np.zeros_like(expected_timeaxis))
    assert captured["debug"] is False
    np.testing.assert_allclose(captured["outvec"], expected_output)
    assert captured["outname"] == args.outfilename


def tcfrom2col_debug_propagates(debug=False):
    if debug:
        print("tcfrom2col_debug_propagates")
    args = _make_default_args(debug=True)
    inputdata = np.array([[0.0, 10.0]])
    captured = {}

    def _mock_maketcfrom2col(indata, timeaxis, outdata, debug=False):
        captured["debug"] = debug
        return np.array(timeaxis, copy=True)

    with (
        patch("rapidtide.workflows.tcfrom2col.tide_io.readvecs", return_value=inputdata),
        patch(
            "rapidtide.workflows.tcfrom2col.tide_util.maketcfrom2col",
            side_effect=_mock_maketcfrom2col,
        ),
        patch("rapidtide.workflows.tcfrom2col.tide_io.writenpvecs"),
    ):
        tcfrom2col(args)

    assert captured["debug"] is True


def test_tcfrom2col(debug=False):
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_debug_flag(debug=debug)

    tcfrom2col_basic(debug=debug)
    tcfrom2col_debug_propagates(debug=debug)


if __name__ == "__main__":
    test_tcfrom2col(debug=True)
