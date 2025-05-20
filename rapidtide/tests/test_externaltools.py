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

import rapidtide.externaltools as tide_exttools
from rapidtide.tests.utils import get_examples_path, get_test_temp_path, mse


def test_externaltools(debug=False, local=False):
    # set input and output directories
    if local:
        exampleroot = "../data/examples/src"
        testtemproot = "./tmp"
    else:
        exampleroot = get_examples_path()
        testtemproot = get_test_temp_path()

    thefsldir = tide_exttools.fslinfo()
    if debug:
        print(f"{thefsldir=}")

    if not local:
        os.environ["FSLDIR"] = "/plausible_FSLDIR"

    thefsldir = tide_exttools.fslinfo()
    if debug:
        print(f"{thefsldir=}")

    fslexists, c3dexists, antsexists = tide_exttools.whatexists()
    if debug:
        print(f"{fslexists=}, {c3dexists=}, {antsexists=}")

    fslsubcmd, flirtcmd, applywarpcmd = tide_exttools.getfslcmds()
    if debug:
        print(f"{fslsubcmd=}, {flirtcmd=}, {applywarpcmd=}")

    tide_exttools.runflirt(
        "inputname", "targetname", "xform", "outputname", warpfile="thewarp", fake=True
    )
    tide_exttools.runflirt("inputname", "targetname", "xform", "outputname", fake=True)

    tide_exttools.n4correct("inputname", "outputdir", fake=True)

    tide_exttools.antsapply(
        "inputname", "targetname", "outputroot", ["transform1", "transform2"], fake=True
    )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_externaltools(debug=True, local=True)
