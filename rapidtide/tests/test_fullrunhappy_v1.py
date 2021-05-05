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
from rapidtide.tests.utils import create_dir, get_examples_path, get_test_temp_path, mse


def test_fullrunhappy_v1(debug=False, display=False):
    # run happy
    inputargs = [
        os.path.join(get_examples_path(), "sub-HAPPYTEST.nii.gz"),
        os.path.join(get_examples_path(), "sub-HAPPYTEST.json"),
        os.path.join(get_test_temp_path(), "happyout"),
        "--mklthreads",
        "-1",
        "--spatialglm",
        "--model",
        "model_revised",
        "--aliasedcorrelation",
    ]
    happy_workflow.happy_main(happy_parser.process_args(inputargs=inputargs))


def main():
    test_fullrunhappy_v1(debug=True, display=True)


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
