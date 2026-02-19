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
import os

import matplotlib as mpl
import pytest

from rapidtide.tests.utils import get_example_and_temp_roots, run_happy

pytestmark = pytest.mark.slow

def test_fullrunhappy_v1(debug=False, local=False, displayplots=False):
    # set input and output directories
    exampleroot, testtemproot = get_example_and_temp_roots(local)

    # run happy
    inputargs = [
        os.path.join(exampleroot, "sub-HAPPYTEST.nii.gz"),
        os.path.join(exampleroot, "sub-HAPPYTEST.json"),
        os.path.join(testtemproot, "happyout1"),
        "--mklthreads",
        "-1",
        "--spatialregression",
    ]
    run_happy(inputargs)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunhappy_v1(debug=True, local=True, displayplots=True)
