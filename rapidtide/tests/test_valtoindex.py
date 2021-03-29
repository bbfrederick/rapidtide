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
import numpy as np

from rapidtide.util import valtoindex


def test_valtoindex(debug=False):
    tr = 1.0
    testtr = 0.7
    xaxislen = 100
    shiftdist = 30
    xaxis = np.arange(0.0, tr * xaxislen, tr)
    minx = np.min(xaxis)
    maxx = np.max(xaxis)
    testvec = np.arange(-1.0, 1.1 * maxx, testtr)
    for i in range(len(testvec)):
        testval = testvec[i]
        indclosest = valtoindex(xaxis, testval)
        print(testval, xaxis[indclosest])


def main():
    test_valtoindex(debug=True)


if __name__ == "__main__":
    main()
