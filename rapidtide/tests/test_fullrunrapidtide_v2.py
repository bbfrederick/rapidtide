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
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def test_fullrunrapidtide_v2(debug=False, local=False, displayplots=False):
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
        os.path.join(testtemproot, "sub-RAPIDTIDETEST2"),
        "--tincludemask",
        os.path.join(exampleroot, "tmask3.txt"),
        "--corrmask",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_restrictedmask.nii.gz"),
        "--globalmeaninclude",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_brainmask.nii.gz"),
        "--globalmeanexclude",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_nullmask.nii.gz"),
        "--refineinclude",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_brainmask.nii.gz"),
        "--refineexclude",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_nullmask.nii.gz"),
        "--offsetinclude",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_brainmask.nii.gz"),
        "--offsetexclude",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_nullmask.nii.gz"),
        "--spatialfilt",
        "-1",
        "--savelags",
        "--autosync",
        "--checkpoint",
        "--timerange",
        "11",
        "-1",
        "--saveintermediatemaps",
        "--bipolar",
        "--norefinedelay",
        "--outputlevel",
        "max",
        "--calccoherence",
        "--cleanrefined",
        "--dispersioncalc",
        "--nprocs",
        "1",
        "--passes",
        "2",
        "--numnull",
        "0",
        "--similaritymetric",
        "hybrid",
        "--globalsignalmethod",
        "meanscale",
        "--refineprenorm",
        "var",
        "--motionfile",
        os.path.join(exampleroot, "fakemotion.par"),
        "--denoisesourcefile",
        os.path.join(exampleroot, "sub-RAPIDTIDETEST.nii.gz"),
    ]
    if debug:
        print(inputargs)
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunrapidtide_v2(debug=True, local=True, displayplots=True)
