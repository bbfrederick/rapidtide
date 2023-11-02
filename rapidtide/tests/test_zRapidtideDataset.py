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
"""
A simple GUI for looking at the results of a rapidtide analysis
"""

import os

from rapidtide.RapidtideDataset import RapidtideDataset
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def main(runninglocally=False):
    # initialize default values
    if runninglocally:
        datafileroot = "../data/examples/dst/sub-RAPIDTIDETEST_"
    else:
        print(f"get_test_temp_path={get_test_temp_path()}")
        datafileroot = (os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST1_"),)
    anatname = None
    geommaskname = None
    userise = False
    usecorrout = True
    useatlas = False
    forcetr = False
    forceoffset = False
    offsettime = 0.0
    verbose = 2

    # read in the dataset
    thesubject = RapidtideDataset(
        "main",
        datafileroot,
        anatname=anatname,
        geommaskname=geommaskname,
        userise=userise,
        usecorrout=usecorrout,
        useatlas=useatlas,
        forcetr=forcetr,
        forceoffset=forceoffset,
        offsettime=offsettime,
        init_LUT=False,
        verbose=verbose,
    )

    theoverlays = thesubject.getoverlays()
    theregressors = thesubject.getregressors()

    assert thesubject.focusregressor == "prefilt"
    thesubject.setfocusregressor("pass3")
    assert thesubject.focusregressor == "pass3"

    if debug:
        print(thesubject.regressorfilterlimits)
    assert thesubject.regressorfilterlimits == (0.01, 0.15)


if __name__ == "__main__":
    main(runninglocally=True)
