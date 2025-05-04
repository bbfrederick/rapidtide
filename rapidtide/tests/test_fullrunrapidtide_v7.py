#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
import os

import matplotlib as mpl
import numpy as np

import rapidtide.io as tide_io
import rapidtide.workflows.rapidtide as rapidtide_workflow
import rapidtide.workflows.rapidtide_parser as rapidtide_parser
import rapidtide.workflows.retroregress as rapidtide_retroregress
from rapidtide.tests.utils import get_examples_path, get_test_temp_path, mse


def test_fullrunrapidtide_v7(debug=False, local=False, displayplots=False):
    # set input and output directories
    if local:
        exampleroot = "../data/examples/src"
        testtemproot = "./tmp"
    else:
        exampleroot = get_examples_path()
        testtemproot = get_test_temp_path()

    # test anatomic masks
    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST_seg"),
        "--nprocs",
        "-1",
        "--passes",
        "3",
        "--brainmask",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_brainmask.nii.gz"),
        "--graymattermask",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_synthseg.nii.gz:SSEG_GRAY"),
        "--whitemattermask",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_synthseg.nii.gz:SSEG_WHITE"),
        "--csfmask",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_synthseg.nii.gz:SSEG_CSF"),
    ]
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))

    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST_seg"),
        "--alternateoutput",
        os.path.join(testtemproot, "segtest"),
        "--nprocs",
        "-1",
        "--outputlevel",
        "max",
    ]
    rapidtide_retroregress.retroregress(rapidtide_retroregress.process_args(inputargs=inputargs))

    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST_seg"),
        "--alternateoutput",
        os.path.join(testtemproot, "regressoronly"),
        "--nprocs",
        "-1",
        "--outputlevel",
        "onlyregressors",
    ]
    rapidtide_retroregress.retroregress(rapidtide_retroregress.process_args(inputargs=inputargs))

    # check to see that rapidtide and retroregress output match
    msethresh = 1e-6
    aethresh = 2
    tclist = ["brain", "GM", "WM", "CSF"]
    for timecourse in tclist:
        infilespec = os.path.join(
            testtemproot,
            f"sub-RAPIDTIDETEST_seg_desc-regionalpostfilter_timeseries.json:{timecourse}",
        )
        insamplerate, instarttime, incolumns, indata, incompressed, infiletype = (
            tide_io.readvectorsfromtextfile(infilespec, onecol=True, debug=debug)
        )
        outfilespec = os.path.join(
            testtemproot, f"segtest_desc-regionalpostfilter_timeseries.json:{timecourse}"
        )
        outsamplerate, outstarttime, outcolumns, outdata, outcompressed, outfiletype = (
            tide_io.readvectorsfromtextfile(outfilespec, onecol=True, debug=debug)
        )
        assert insamplerate == outsamplerate
        assert instarttime == outstarttime
        assert incompressed == outcompressed
        assert infiletype == outfiletype
        assert incolumns == outcolumns
        assert indata.shape == outdata.shape
        assert indata.shape[0] == indata.shape[0]
        assert mse(indata, outdata) < msethresh
        np.testing.assert_almost_equal(indata, outdata, aethresh)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunrapidtide_v7(debug=True, local=True, displayplots=True)
