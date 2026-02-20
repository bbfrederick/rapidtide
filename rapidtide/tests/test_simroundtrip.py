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
try:
    from rapidtide.tests._mplsetup import configure_matplotlib_env
except Exception:
    import os
    import sys

    _TESTSDIR = os.path.dirname(os.path.abspath(__file__))
    if _TESTSDIR not in sys.path:
        sys.path.insert(0, _TESTSDIR)
    from _mplsetup import configure_matplotlib_env

configure_matplotlib_env()

import os

import matplotlib as mpl
import pytest

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf
import rapidtide.workflows.simdata as rapidtide_simdata
from rapidtide.tests.utils import get_example_and_temp_roots, run_rapidtide

pytestmark = pytest.mark.slow

def test_simroundtrip(debug=False, local=False, displayplots=False):
    # set input and output directories
    exampleroot, testtemproot = get_example_and_temp_roots(local)

    # run initial rapidtide
    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETESTSIM"),
        "--corrmask",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_restrictedmask.nii.gz"),
        "--globalmeaninclude",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_brainmask.nii.gz"),
        "--spatialfilt",
        "2",
        "--simcalcrange",
        "4",
        "-1",
        "--nprocs",
        "-1",
        "--passes",
        "2",
        "--despecklepasses",
        "3",
        "--delaypatchthresh",
        "4.0",
    ]
    run_rapidtide(inputargs)

    print("initial rapidtide run complete")

    # now simulate data from maps
    print(testtemproot)
    inputargs = [
        "1.5",
        "260",
        os.path.join(testtemproot, "sub-RAPIDTIDETESTSIM_desc-unfiltmean_map.nii.gz"),
        os.path.join(testtemproot, "simulatedfmri_vn05"),
        "--lfopctfile",
        os.path.join(testtemproot, "sub-RAPIDTIDETESTSIM_desc-maxcorr_map.nii.gz"),
        "--lfolagfile",
        os.path.join(testtemproot, "sub-RAPIDTIDETESTSIM_desc-maxtimerefined_map.nii.gz"),
        "--lforegressor",
        os.path.join(
            testtemproot, "sub-RAPIDTIDETESTSIM_desc-movingregressor_timeseries.json:pass2"
        ),
        "--voxelnoiselevel",
        "5.0",
    ]

    pf.generic_init(rapidtide_simdata._get_parser, rapidtide_simdata.simdata, inputargs=inputargs)
    print("simulated dataset generated")

    # run repeat rapidtide
    inputargs = [
        os.path.join(testtemproot, "simulatedfmri_vn05.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETESTSIMRERUN"),
        "--spatialfilt",
        "2",
        "--simcalcrange",
        "4",
        "-1",
        "--nprocs",
        "-1",
        "--passes",
        "2",
        "--despecklepasses",
        "3",
        "--delaypatchthresh",
        "4.0",
    ]
    run_rapidtide(inputargs)
    print("repeat rapidtide completed")

    absthresh = 1e-10
    msethresh = 1e-12
    spacetolerance = 1e-3
    """for map in [
        "maxtime",
        "maxtimerefined",
        "lfofilterCoeff",
    ]:
        print(f"Testing map={map}")
        filename1 = os.path.join(testtemproot, f"sub-RAPIDTIDETESTSIM_desc-{map}_map.nii.gz")
        filename2 = os.path.join(testtemproot, f"sub-RAPIDTIDETESTSIMRERUN_desc-{map}_map.nii.gz")
        assert tide_io.checkniftifilematch(
            filename1,
            filename2,
            absthresh=absthresh,
            msethresh=msethresh,
            spacetolerance=spacetolerance,
            debug=debug,
        )"""


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_simroundtrip(debug=True, local=True, displayplots=True)
