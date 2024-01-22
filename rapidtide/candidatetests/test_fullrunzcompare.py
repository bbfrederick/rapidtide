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

import numpy as np

import rapidtide.util as tide_util
from rapidtide.tests.utils import create_dir, get_examples_path, get_test_temp_path, mse


def test_fullrunzcompare(debug=False):
    # comparerapidtideruns smoke test
    rapidtideroot1 = os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST3")
    rapidtideroot2 = os.path.join(get_test_temp_path(), "sub-RAPIDTIDETEST2")

    therapidtideresults = tide_util.comparerapidtideruns(
        rapidtideroot1, rapidtideroot2, debug=debug
    )

    # comparehappyruns smoke test
    happyroot1 = os.path.join(get_test_temp_path(), "happyout2")
    happyroot2 = os.path.join(get_test_temp_path(), "happyout4")

    # thehappyresults = tide_util.comparehappyruns(happyroot1, happyroot2)


if __name__ == "__main__":
    test_fullrunzcompare(debug=True)
