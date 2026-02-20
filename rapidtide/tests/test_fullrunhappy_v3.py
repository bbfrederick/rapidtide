#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
try:
    from rapidtide.tests._mplsetup import configure_matplotlib_env
except Exception:
    import os
    import sys

    _TESTSDIR = os.path.dirname(os.path.abspath(__file__))
    if _TESTSDIR not in sys.path:
        sys.path.insert(0, _TESTSDIR)
    from _mplsetup import configure_matplotlib_env

configure_matplotlib_env()

import os

import matplotlib as mpl
import pytest

from rapidtide.tests.utils import get_example_and_temp_roots, run_happy

try:
    import tensorflow as tf

    tensorflowexists = True
except ImportError:
    tensorflowexists = False


pytestmark = pytest.mark.slow

def test_fullrunhappy_v3(debug=False, local=False, displayplots=False):
    # set input and output directories
    exampleroot, testtemproot = get_example_and_temp_roots(local)

    # run happy
    inputargs = [
        os.path.join(exampleroot, "sub-HAPPYTEST.nii.gz"),
        os.path.join(exampleroot, "sub-HAPPYTEST.json"),
        os.path.join(testtemproot, "happyout3"),
        "--estweights",
        os.path.join(exampleroot, "sub-HAPPYTEST_smallmask.nii.gz"),
        "--projmask",
        os.path.join(exampleroot, "sub-HAPPYTEST_smallmask.nii.gz"),
        "--mklthreads",
        "-1",
        "--cardcalconly",
    ]
    if tensorflowexists:
        inputargs.append("--usetensorflow")
        inputargs.append("--model")
        inputargs.append("model_revised_tf2")
    run_happy(inputargs)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunhappy_v3(debug=True, local=True, displayplots=True)
