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

import rapidtide.workflows.rapidtide as rapidtide_workflow
import rapidtide.workflows.rapidtide_parser as rapidtide_parser
import rapidtide.workflows.retroregress as rapidtide_retroregress
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def test_fullrunrapidtide_v3(debug=False, local=False, displayplots=False):
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
        os.path.join(testtemproot, "sub-RAPIDTIDETEST3"),
        "--corrmask",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_restrictedmask.nii.gz:1"),
        "--maxpasses",
        "2",
        "--numnull",
        "100",
        "--refinetype",
        "pca",
        "--convergencethresh",
        "0.5",
        "--nodenoise",
        "--nprocs",
        "-1",
        "--similaritymetric",
        "mutualinfo",
        "--norefinedelay",
        "--dpoutput",
        "--spcalculation",
        "--simcalcrange",
        "100",
        "-1",
        "--globalsignalmethod",
        "pca",
        "--calccoherence",
        "--refineprenorm",
        "std",
        "--refineweighting",
        "R",
        "--regressor",
        os.path.join(
            exampleroot,
            "sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json:pass3",
        ),
    ]
    #    "--psdfilter",
    #    "--cleanrefined",
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))

    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST3"),
        "--alternateoutput",
        os.path.join(testtemproot, "onlyregressors"),
        "--outputlevel",
        "onlyregressors",
    ]
    rapidtide_retroregress.retroregress(rapidtide_retroregress.process_args(inputargs=inputargs))


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunrapidtide_v3(debug=True, local=True, displayplots=True)
