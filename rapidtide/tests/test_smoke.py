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
import argparse
import os

import matplotlib as mpl

import rapidtide.workflows.happy as happy_workflow
import rapidtide.workflows.happy_parser as happy_parser
import rapidtide.workflows.rapidtide as rapidtide_workflow
import rapidtide.workflows.rapidtide_parser as rapidtide_parser

# import rapidtide.workflows.aligntcs as aligntcs
from rapidtide.tests.utils import create_dir, get_examples_path, get_test_temp_path, mse


def test_smoke(debug=False, display=False):
    """
    aligntcsargs = argparse.Namespace
    aligntcsargs.arbvec = None
    aligntcsargs.display = False
    aligntcsargs.filterband = "lfo"
    aligntcsargs.infile1 = os.path.join(get_examples_path(), "timecourse1.txt")
    aligntcsargs.infile2 = os.path.join(get_examples_path(), "timecourse2.txt")
    aligntcsargs.insamplerate1 = 12.5
    aligntcsargs.insamplerate2 = 12.5
    aligntcsargs.lag_extrema = (-30.0, 30.0)
    aligntcsargs.outputfile = os.path.join(get_test_temp_path(), "hoot")
    aligntcsargs.verbose = False
    aligntcs.main(aligntcsargs)
    """

    # run happy
    inputargs = [
        os.path.join(get_examples_path(), "sub-HAPPYTEST.nii.gz"),
        os.path.join(get_examples_path(), "sub-HAPPYTEST.json"),
        os.path.join(get_test_temp_path(), "happyout"),
        "--mklthreads",
        "8",
        "--model",
        "model_revised",
        "--aliasedcorrelation",
    ]
    happy_workflow.happy_main(happy_parser.process_args(inputargs=inputargs))

    # run rapidtide
    inputargs = [
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST.nii.gz"),
        os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST"),
        "--spatialfilt",
        "2",
        "--nprocs",
        "-1",
        "--passes",
        "3",
    ]
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))


def main():
    test_smoke(debug=True, display=True)


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
