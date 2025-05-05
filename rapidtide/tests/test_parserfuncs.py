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
import argparse
import rapidtide.workflows.parser_funcs as pf
from rapidtide.tests.utils import get_examples_path, get_test_temp_path

def _get_parser():
    """
    Argument parser for adjust offset
    """
    parser = argparse.ArgumentParser(
        prog="dummy",
        description="dummy",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputmap",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The name of the rapidtide maxtime map.",
    )

    return parser


def test_parserfuncs(debug=False, local=False, displayplots=False):
    # set input and output directories
    if local:
        exampleroot = "../data/examples/src"
        testtemproot = "./tmp"
    else:
        exampleroot = get_examples_path()
        testtemproot = get_test_temp_path()


    theparser = _get_parser()

    filename = os.path.join(exampleroot,"sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json")
    retval = pf.is_valid_file(theparser, filename)
    print(filename, retval)

    #filename = os.path.join(exampleroot,"sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseriesxyz.json")
    #retval = pf.is_valid_file(theparser, filename)
    #print(filename, retval)

    filename = os.path.join(exampleroot,"sub-RAPIDTIDETEST_desc-oversampledmovingregressor_timeseries.json:acolname")
    retval = pf.is_valid_file(theparser, filename)
    print(filename, retval)


if __name__ == "__main__":
    test_parserfuncs(debug=True, local=True, displayplots=True)
