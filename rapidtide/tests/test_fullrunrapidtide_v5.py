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


def test_fullrunrapidtide_v5(debug=False, local=False, displayplots=False):
    # set input and output directories
    if local:
        exampleroot = "../data/examples/src"
        testtemproot = "./tmp"
    else:
        exampleroot = get_examples_path()
        testtemproot = get_test_temp_path()

    # run rapidtide
    inputargs = [
        os.path.join(exampleroot, "sub-RAPIDTIDETEST_cifti.ptseries.nii"),
        os.path.join(testtemproot, "sub-RAPIDTIDETEST5"),
        "--nprocs",
        "-1",
        "--passes",
        "2",
        "--numnull",
        "10000",
        "--pickleft",
        "--norefinedelay",
        "--refineprenorm",
        "None",
    ]
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunrapidtide_v5(debug=True, local=True, displayplots=True)
