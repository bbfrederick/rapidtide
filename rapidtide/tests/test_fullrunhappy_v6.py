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
"""Exercise the happy options that the other full run tests leave untouched.

v1 through v5 between them cover the regression types, the cardiac and motion file
paths, the deep learning filter and the aliased correlation.  This one covers global
mean signal filtering and Wright's vessel mask; the respiration file option was the
second test here and now lives in v7, so the slow shard can put the two on different
containers.

Both options ride on a single happy run on purpose.  Measured on this dataset a bare
run is about 38 s, ``--gmsfilt`` adds about 12 s and ``--wrightiterations 2`` about
13 s, so giving them a run each would add a whole base run to buy back a fraction of
that.  They stay together.

Deliberately not covered: ``--estimateflow``.  Measured against the full size dataset
it added about 18 minutes to a roughly 2 minute run, in exchange for around a dozen
lines - by far the worst ratio of any option here.  Both halves of that ratio shrink
with the cropped dataset this test now uses, but not the ratio itself, so the
conclusion is unchanged: if the optical flow code needs test coverage it wants a small
synthetic fixture aimed at ``calc_3d_optical_flow`` directly, not a full happy run.
"""

import os

import matplotlib as mpl
import pytest

from rapidtide.tests.utils import get_example_and_temp_roots, run_happy

pytestmark = pytest.mark.slow


def test_fullrunhappy_v6(debug=False, local=False, displayplots=False):
    """Run happy with global mean signal filtering and Wright's vessel mask."""
    exampleroot, testtemproot = get_example_and_temp_roots(local)

    inputargs = [
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL.nii.gz"),
        os.path.join(exampleroot, "sub-HAPPYTESTSMALL.json"),
        os.path.join(testtemproot, "happyout6"),
        "--mklthreads",
        "-1",
        "--gmsfilt",
        "--wrightiterations",
        "2",
    ]
    run_happy(inputargs)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_fullrunhappy_v6(debug=True, local=True, displayplots=True)
