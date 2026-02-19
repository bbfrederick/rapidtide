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
import os

import matplotlib as mpl
import pytest

from rapidtide.tests.utils import (assert_text_vectors_match,
                                   get_example_and_temp_roots, run_rapidtide,
                                   run_retroregress)

pytestmark = pytest.mark.slow

def test_fullrunrapidtide_v7(debug=False, local=False, displayplots=False):
    # set input and output directories
    exampleroot, testtemproot = get_example_and_temp_roots(local)

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
    run_rapidtide(inputargs)

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
    run_retroregress(inputargs)

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
    run_retroregress(inputargs)

    # check to see that rapidtide and retroregress output match
    tclist = ["brain", "GM", "WM", "CSF"]
    for timecourse in tclist:
        assert_text_vectors_match(
            infile_spec=os.path.join(
                testtemproot,
                f"sub-RAPIDTIDETEST_seg_desc-regionalpostfilter_timeseries.json:{timecourse}",
            ),
            outfile_spec=os.path.join(
                testtemproot, f"segtest_desc-regionalpostfilter_timeseries.json:{timecourse}"
            ),
            debug=debug,
        )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunrapidtide_v7(debug=True, local=True, displayplots=True)
