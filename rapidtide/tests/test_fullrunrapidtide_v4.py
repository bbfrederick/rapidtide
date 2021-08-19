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


def test_fullrunrapidtide_v4(debug=False, display=False):
    # run rapidtide
    inputargs = [
        os.path.join(get_examples_path(), "sub-NIRSRAPIDTIDETEST.txt"),
        os.path.join(get_test_temp_path(), "sub-NIRSRAPIDTIDETEST"),
        "--globalmeaninclude",
        os.path.join(get_examples_path(), "sub-NIRSRAPIDTIDETEST_mask.txt"),
        "--nirs",
        "--datatstep",
        "0.2560",
        "--globalmaskmethod",
        "variance",
        "--despecklepasses",
        "0",
        "--numnull",
        "1000",
        "--respdelete",
        "--echocancel",
    ]
    rapidtide_workflow.rapidtide_main(rapidtide_parser.process_args(inputargs=inputargs))


def main():
    test_fullrunrapidtide_v4(debug=True, display=True)


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
