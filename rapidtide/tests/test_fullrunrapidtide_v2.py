#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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


def test_fullrunrapidtide_v2(debug=False, display=False):
    # run rapidtide
    inputargs = [
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST"),
        "--tmask",
        os.path.join(get_examples_path(), "tmask3.txt"),
        "--corrmask",
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST_restrictedmask.nii.gz"),
        "--globalmeaninclude",
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST_brainmask.nii.gz"),
        "--globalmeanexclude",
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST_nullmask.nii.gz"),
        "--refineinclude",
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST_brainmask.nii.gz"),
        "--refineexclude",
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST_nullmask.nii.gz"),
        "--spatialfilt",
        "-1",
        "--savelags",
        "--checkpoint",
        "--saveintermediatemaps",
        "--nolimitoutput",
        "--calccoherence",
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
    ]
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))


def main():
    test_fullrunrapidtide_v2(debug=True, display=True)


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
