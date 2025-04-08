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
import numpy as np

import rapidtide.util as tide_util


def test_numpy2shared(debug=False):
    vectorlen = 1000
    for intype in [np.float32, np.float64]:
        sourcevector = np.random.normal(size=vectorlen).astype(intype)
        if debug:
            print(f"{intype=}, {sourcevector.size=}, {sourcevector.dtype=}")
        for outtype in [np.float32, np.float64]:

            destvector, shm = tide_util.numpy2shared(sourcevector, outtype)
            if debug:
                print(f"\t{outtype=}, {destvector.size=}, {destvector.dtype=}")

            # check everything
            assert destvector.dtype == outtype
            assert destvector.size == sourcevector.size
            np.testing.assert_almost_equal(sourcevector, destvector, 3)

            # clean up
            tide_util.cleanup_shm(shm)


def test_allocshared(debug=False):
    datashape = (10, 10, 10)
    for outtype in [np.float32, np.float64]:
        destarray, shm = tide_util.allocshared(datashape, outtype)
        if debug:
            print(f"{outtype=}, {destarray.size=}, {destarray.dtype=}")

        # check everything
        assert destarray.dtype == outtype
        assert destarray.size == np.prod(datashape)

        # clean up if needed
        tide_util.cleanup_shm(shm)


if __name__ == "__main__":
    test_numpy2shared(debug=True)
    test_allocshared(debug=True)
