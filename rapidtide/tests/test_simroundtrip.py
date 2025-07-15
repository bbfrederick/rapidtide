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

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf
import rapidtide.workflows.rapidtide as rapidtide_workflow
import rapidtide.workflows.rapidtide_parser as rapidtide_parser
import rapidtide.workflows.simdata as rapidtide_simdata
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def test_simroundtrip(debug=False, local=False, displayplots=False):
    # set input and output directories
    if local:
        exampleroot = "../data/examples/src"
        testtemproot = "./tmp"
    else:
        exampleroot = get_examples_path()
        testtemproot = get_test_temp_path()


    # run initial rapidtide
    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETESTSIM"),
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
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))

    # now simulate data from maps
    print(testtemproot)
    inputargs = [
        "1.5",
        "260",
        os.path.join( testtemproot, "sub-RAPIDTIDETESTSIM_desc-unfiltmean_map.nii.gz"),
        os.path.join( testtemproot, "simulatedfmri_vn05"),
        "--lfopctfile",
        os.path.join( testtemproot, "sub-RAPIDTIDETESTSIM_desc-maxcorr_map.nii.gz"),
        "--lfolagfile",
        os.path.join( testtemproot, "sub-RAPIDTIDETESTSIM_desc-maxtimerefined_map.nii.gz"),
        "--lforegressor",
        os.path.join(testtemproot, "sub-RAPIDTIDETESTSIM_desc-movingregressor_timeseries.json:pass2"),
        "--voxelnoiselevel",
        "5.0"
    ]

    pf.generic_init(rapidtide_simdata._get_parser, rapidtide_simdata.simdata, inputargs=inputargs)

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
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))

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
