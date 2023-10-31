#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
import os
from types import SimpleNamespace

import matplotlib as mpl

import rapidtide.workflows.aligntcs as aligntcs_workflow
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def test_aligntcs(displayplots=False, localrun=False):
    args = SimpleNamespace()
    if localrun:
        inputdatapath = "../data/examples/src"
        outputdatapath = "./tmp"
    else:
        inputdatapath = get_examples_path()
        outputdatapath = get_test_temp_path()
    args.infile1 = os.path.join(inputdatapath, "lforegressor.txt")
    args.infile2 = os.path.join(inputdatapath, "lforegressor.txt")
    args.insamplerate1 = 1.0
    args.insamplerate2 = 1.0
    args.outputfile = os.path.join(outputdatapath, "aligntcout")
    args.displayplots = displayplots
    args.filterband = "lfo"
    args.filterfreqs = None
    args.stopvec = None
    args.passvec = None
    args.lag_extrema = (-10.0, 10.0)

    aligntcs_workflow.doaligntcs(args)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_aligntcs(displayplots=True, localrun=True)
