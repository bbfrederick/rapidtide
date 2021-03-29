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
import rapidtide.workflows.rapidtide2x as rapidtide2x_workflow
from rapidtide.tests.utils import (
    create_dir,
    get_examples_path,
    get_test_target_path,
    get_test_temp_path,
)


def test_rapidtide2x_phase3(debug=False):
    recalculate = True
    if recalculate:
        # create outputdir if it doesn't exist
        create_dir(get_test_temp_path())

        # trigger the usage function
        rapidtide2x_workflow.usage()

        # and launch the processing
        theargs = ["rapidtide2x"]
        theargs += [os.path.join(get_examples_path(), "fmri.nii.gz")]
        theargs += [os.path.join(get_test_temp_path(), "rapidtide2x_phase3output")]
        theargs += ["--nowindow"]
        theargs += ["--windowfunc=hamming"]
        theargs += ["--liang", "--eckart", "--phat"]
        theargs += ["--usesp"]
        theargs += ["--preservefiltering"]
        theargs += ["--corrmaskthresh=0.25"]
        theargs += ["-I", "-B", "-a", "-M", "-m"]
        theargs += ["-C", "-R", "-L", "-V", "-F", "0.01,0.08"]
        theargs += ["-v", "--debug"]
        theargs += ["--globalmaskmethod=mean"]
        theargs += ["--mklthreads=1"]
        theargs += ["--nosharedmem"]
        theargs += ["-S"]
        rapidtide2x_workflow.rapidtide_main(theargs)
    assert True


def main():
    test_rapidtide2x_phase3(debug=True)


if __name__ == "__main__":
    main()
