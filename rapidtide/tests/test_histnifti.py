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

from rapidtide.workflows.histnifti import _get_parser, histnifti


def _default_args(**overrides):
    args = dict(
        inputfile="input.nii.gz",
        outputroot="/tmp/histnifti_test",
        histlen=None,
        minval=None,
        maxval=None,
        robustrange=False,
        transform=False,
        nozero=False,
        nozerothresh=0.01,
        normhist=False,
        maskfile=None,
        display=True,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _dims_for_data(data):
    if data.ndim == 4:
        x, y, z, t = data.shape
    else:
        x, y, z = data.shape
        t = 1
    return np.array([4, x, y, z, t, 1, 1, 1], dtype=int)


def _mock_nifti_hdr(data):
    dims = _dims_for_data(data).tolist()
    return {
        "dim": dims,
        "pixdim": [1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0],
        "toffset": 0.0,
    }


def _run_histnifti_with_mocks(args, input_data, mask_data=None, mask_match=True):
    input_hdr = _mock_nifti_hdr(input_data)
    input_dims = _dims_for_data(input_data)
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0], dtype=float)

    saved = {}
    hist_calls = []

    def _mock_readfromnifti(fname, **kwargs):
        if args.maskfile is not None and fname == args.maskfile:
            this_mask = (
                np.array(mask_data, dtype=float)
                if mask_data is not None
                else np.ones(input_data.shape[:3], dtype=float)
            )
            mask_hdr = _mock_nifti_hdr(this_mask)
            mask_dims = _dims_for_data(this_mask)
            return (MagicMock(), this_mask, mask_hdr, mask_dims, sizes)
        return (MagicMock(), np.array(input_data), input_hdr, input_dims, sizes)

    def _mock_savetonifti(arr, hdr, outname, **kwargs):
        saved[outname] = np.array(arr, copy=True)

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        hist_calls.append(
            {
                "indata": np.array(indata, copy=True),
                "histlen": histlen,
                "endtrim": endtrim,
                "outname": outname,
                "kwargs": kwargs,
            }
        )

    with (
        patch("rapidtide.workflows.histnifti.tide_io.readfromnifti", side_effect=_mock_readfromnifti),
        patch("rapidtide.workflows.histnifti.tide_io.savetonifti", side_effect=_mock_savetonifti),
        patch("rapidtide.workflows.histnifti.tide_stats.makeandsavehistogram", side_effect=_mock_makeandsavehistogram),
        patch("rapidtide.workflows.histnifti.tide_io.checkspacematch", return_value=mask_match),
    ):
        histnifti(args)

    return saved, hist_calls


def test_get_parser_required_args():
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_get_parser_defaults():
    parser = _get_parser()
    args = parser.parse_args(["in.nii.gz", "out"])
    assert args.histlen is None
    assert args.minval is None
    assert args.maxval is None
    assert args.robustrange is False
    assert args.transform is False
    assert args.nozero is False
    assert args.nozerothresh == 0.01
    assert args.normhist is False
    assert args.maskfile is None
    assert args.display is True


def test_get_parser_maskfile_accepts_existing_file():
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        args = parser.parse_args(["in.nii.gz", "out", "--maskfile", f.name])
    assert args.maskfile == f.name


def test_histnifti_4d_outputs_all_expected_files():
    rng = np.random.RandomState(0)
    data = rng.randn(3, 3, 2, 20) * 5.0 + 100.0
    args = _default_args()

    saved, hist_calls = _run_histnifti_with_mocks(args, data)

    assert args.outputroot + "_sorted" in saved
    assert args.outputroot + "_pcts" in saved
    assert args.outputroot + "_hists" in saved
    assert len(hist_calls) == 0
    assert saved[args.outputroot + "_sorted"].shape == data.shape
    assert saved[args.outputroot + "_pcts"].shape == (3, 3, 2, 5)
    assert saved[args.outputroot + "_hists"].shape == (3, 3, 2, 9)


def test_histnifti_4d_nozero_uses_fractional_percentiles():
    data = np.linspace(0.0, 1.0, 3 * 3 * 2 * 20, dtype=float).reshape((3, 3, 2, 20))
    data[0, 0, 0, :4] = 0.0
    args = _default_args(nozero=True)
    call_counter = {"count": 0}

    def _counting_getfracvals(*gargs, **gkwargs):
        call_counter["count"] += 1
        return original_getfracvals(*gargs, **gkwargs)

    from rapidtide import stats as tide_stats

    original_getfracvals = tide_stats.getfracvals
    with patch("rapidtide.workflows.histnifti.tide_stats.getfracvals", side_effect=_counting_getfracvals):
        saved, _ = _run_histnifti_with_mocks(args, data)

    assert args.outputroot + "_pcts" in saved
    assert call_counter["count"] >= np.prod(data.shape[:3])


def test_histnifti_4d_mask_mismatch_exits():
    rng = np.random.RandomState(1)
    data = rng.randn(3, 3, 2, 20)
    args = _default_args(maskfile="mask.nii.gz")

    with pytest.raises(SystemExit):
        _run_histnifti_with_mocks(args, data, mask_match=False)


def test_histnifti_3d_histogram_call_and_flags():
    data = np.arange(27, dtype=float).reshape((3, 3, 3))
    args = _default_args(normhist=True, display=False, histlen=11)

    saved, hist_calls = _run_histnifti_with_mocks(args, data)

    assert len(saved) == 0
    assert len(hist_calls) == 1
    call = hist_calls[0]
    assert call["histlen"] == 11
    assert call["outname"] == args.outputroot + "_hist"
    assert call["kwargs"]["normalize"] is True
    assert call["kwargs"]["displayplots"] is False
    assert call["kwargs"]["refine"] is False


def test_histnifti_3d_transform_saves_transformed_volume():
    data = np.linspace(10.0, 90.0, 27, dtype=float).reshape((3, 3, 3))
    args = _default_args(transform=True)

    saved, hist_calls = _run_histnifti_with_mocks(args, data)

    transformed_name = args.outputroot + "_transformed"
    assert transformed_name in saved
    transformed = saved[transformed_name]
    assert transformed.shape == data.shape
    assert np.min(transformed) >= 0.0
    assert np.max(transformed) <= 100.0
    assert len(hist_calls) == 1


def test_histnifti_3d_nozero_filters_small_values():
    data = np.array(
        [
            [[0.0, 0.004], [0.02, 1.0]],
            [[-0.001, -0.5], [0.2, 0.3]],
        ],
        dtype=float,
    )
    args = _default_args(nozero=True, nozerothresh=0.01)

    _, hist_calls = _run_histnifti_with_mocks(args, data)

    assert len(hist_calls) == 1
    indata = hist_calls[0]["indata"]
    assert np.all(np.fabs(indata) >= 0.01)


def test_histnifti(debug=False):
    if debug:
        print("Running parser tests")
    test_get_parser_required_args()
    test_get_parser_defaults()
    test_get_parser_maskfile_accepts_existing_file()

    if debug:
        print("Running 4D workflow tests")
    test_histnifti_4d_outputs_all_expected_files()
    test_histnifti_4d_nozero_uses_fractional_percentiles()
    test_histnifti_4d_mask_mismatch_exits()

    if debug:
        print("Running 3D workflow tests")
    test_histnifti_3d_histogram_call_and_flags()
    test_histnifti_3d_transform_saves_transformed_volume()
    test_histnifti_3d_nozero_filters_small_values()


if __name__ == "__main__":
    test_histnifti(debug=True)
