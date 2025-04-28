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
import rapidtide.workflows.rapidtide as rapidtide_workflow
import rapidtide.workflows.rapidtide_parser as rapidtide_parser
import rapidtide.workflows.retroregress as rapidtide_retroregress
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def test_fullrunrapidtide_v6(debug=False, local=False, displayplots=False):
    # set input and output directories
    if local:
        exampleroot = "../data/examples/src"
        testtemproot = "./tmp"
    else:
        exampleroot = get_examples_path()
        testtemproot = get_test_temp_path()

    # run rapidtide
    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST6"),
        "--spatialfilt",
        "2",
        "--simcalcrange",
        "4",
        "-1",
        "--nprocs",
        "-1",
        "--passes",
        "1",
        "--despecklepasses",
        "3",
        "--regressderivs",
        "0",
        "--delaypatchthresh",
        "4.0",
        "--outputlevel",
        "max",
    ]
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))

    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST6"),
        "--alternateoutput",
        os.path.join(testtemproot, "2deriv"),
        "--nprocs",
        "-1",
        "--regressderivs",
        "2",
        "--makepseudofile",
        "--outputlevel",
        "max",
    ]
    rapidtide_retroregress.retroregress(rapidtide_retroregress.process_args(inputargs=inputargs))

    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST6"),
        "--alternateoutput",
        os.path.join(testtemproot, "1deriv_refined_corrected"),
        "--nprocs",
        "1",
        "--regressderivs",
        "1",
        "--makepseudofile",
        "--outputlevel",
        "max",
        "--nofilterwithrefineddelay",
    ]
    rapidtide_retroregress.retroregress(rapidtide_retroregress.process_args(inputargs=inputargs))

    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST6"),
        "--alternateoutput",
        os.path.join(testtemproot, "concordance"),
        "--nprocs",
        "-1",
        "--regressderivs",
        "0",
        "--delaypatchthresh",
        "4.0",
        "--outputlevel",
        "max",
    ]
    rapidtide_retroregress.retroregress(rapidtide_retroregress.process_args(inputargs=inputargs))

    absthresh = 1e-10
    msethresh = 1e-12
    spacetolerance = 1e-3
    for map in [
        "regressderivratios",
        "medfiltregressderivratios",
        "filteredregressderivratios",
        "maxtimerefined",
        "lfofilterInbandVarianceBefore",
        "lfofilterInbandVarianceAfter",
        "lfofilterInbandVarianceChange",
        "lfofilterCoeff",
        "lfofilterMean",
        "lfofilterNorm",
        "lfofilterR2",
        "lfofilterR",
    ]:
        print(f"Testing map={map}")
        filename1 = os.path.join(testtemproot, f"sub-RAPIDTIDETEST6_desc-{map}_map.nii.gz")
        filename2 = os.path.join(testtemproot, f"concordance_desc-{map}_map.nii.gz")
        assert tide_io.checkniftifilematch(
            filename1,
            filename2,
            absthresh=absthresh,
            msethresh=msethresh,
            spacetolerance=spacetolerance,
            debug=debug,
        )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunrapidtide_v6(debug=True, local=True, displayplots=True)
