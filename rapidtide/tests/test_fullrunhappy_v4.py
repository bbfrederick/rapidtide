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

import rapidtide.workflows.happy as happy_workflow
import rapidtide.workflows.happy_parser as happy_parser
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def test_fullrunhappy_v4(debug=False, local=False, displayplots=False):
    # set input and output directories
    if local:
        exampleroot = "../data/examples/src"
        testtemproot = "./tmp"
    else:
        exampleroot = get_examples_path()
        testtemproot = get_test_temp_path()

    # run happy
    inputargs = [
        os.path.join(exampleroot, "sub-HAPPYTEST.nii.gz"),
        os.path.join(exampleroot, "sub-HAPPYTEST.json"),
        os.path.join(testtemproot, "happyout4"),
        "--estweights",
        os.path.join(exampleroot, "sub-HAPPYTEST_smallmask.nii.gz"),
        "--projmask",
        os.path.join(exampleroot, "sub-HAPPYTEST_smallmask.nii.gz"),
        "--mklthreads",
        "-1",
        "--model",
        "model_revised_tf2",
        "--motionfile",
        os.path.join(exampleroot, "sub-HAPPYTEST_mcf.par"),
        "--aliasedcorrelation",
    ]
    happy_workflow.happy_main(happy_parser.process_args(inputargs=inputargs))


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunhappy_v4(debug=True, local=True, displayplots=True)
