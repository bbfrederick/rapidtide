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

import matplotlib.pyplot as plt
import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.util as tide_util
import rapidtide.workflows.happy as happy_workflow
from rapidtide.tests.utils import (
    create_dir,
    get_examples_path,
    get_rapidtide_root,
    get_scripts_path,
    get_test_data_path,
    get_test_target_path,
    get_test_temp_path,
)


def test_happy_phase1(debug=False):
    recalculate = True
    if recalculate:
        # create outputdir if it doesn't exist
        create_dir(get_test_temp_path())

        # and launch the processing
        theargs = ["happy"]
        theargs += [os.path.join(get_examples_path(), "happyfmri.nii.gz")]
        theargs += [os.path.join(get_examples_path(), "happyfmri.json")]
        theargs += [os.path.join(get_test_temp_path(), "happy_output")]
        theargs += ["--dodlfilter"]
        theargs += ["--saveinfoasjson"]
        theargs += ["--glm"]
        theargs += ["--numskip=0"]
        theargs += ["--gridbins=2.0"]
        theargs += ["--gridkernel=kaiser"]
        theargs += ["--model=model_revised"]
        print(" ".join(theargs))
        happy_workflow.happy_main(theargs)

    diffmaps = tide_util.comparehappyruns(
        os.path.join(get_test_temp_path(), "happy_output"),
        os.path.join(get_test_target_path(), "happy_target"),
        debug=debug,
    )

    for mapname, maps in diffmaps.items():
        print("checking", mapname)
        print("\trelmindiff", maps["relmindiff"])
        print("\trelmaxdiff", maps["relmaxdiff"])
        print("\trelmeandiff", maps["relmeandiff"])
        print("\trelmse", maps["relmse"])
        assert maps["relmindiff"] < 1e2
        assert maps["relmaxdiff"] < 1e2
        assert maps["relmeandiff"] < 1e-2
        assert maps["relmse"] < 1e2


def main():
    test_happy_phase1(debug=True)


if __name__ == "__main__":
    main()
