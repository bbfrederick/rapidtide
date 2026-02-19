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

from rapidtide.tests.utils import (
    assert_output_maps_match,
    get_example_and_temp_roots,
    run_rapidtide,
    run_retroregress,
)

pytestmark = pytest.mark.slow

def test_fullrunrapidtide_v6(debug=False, local=False, displayplots=False):
    # set input and output directories
    exampleroot, testtemproot = get_example_and_temp_roots(local)

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
    run_rapidtide(inputargs)

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
    run_retroregress(inputargs)

    #inputargs = [
        #os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        #os.path.join(testtemproot, "sub-RAPIDTIDETEST6"),
        #"--alternateoutput",
        #os.path.join(testtemproot, "1deriv_refined_corrected"),
        #"--nprocs",
        #"1",
        #"--regressderivs",
        #"1",
        #"--makepseudofile",
        #"--outputlevel",
        #"max",
        #"--nofilterwithrefineddelay",
    #]
    #rapidtide_retroregress.retroregress(rapidtide_retroregress.process_args(inputargs=inputargs))

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
    run_retroregress(inputargs)

    assert_output_maps_match(
        [
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
        ],
        output_root_1="sub-RAPIDTIDETEST6",
        output_root_2="concordance",
        temp_root=testtemproot,
        debug=debug,
    )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunrapidtide_v6(debug=True, local=True, displayplots=True)
