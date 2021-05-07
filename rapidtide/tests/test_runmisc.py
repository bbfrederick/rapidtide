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
import argparse
import os

import matplotlib as mpl

import rapidtide.workflows.parser_funcs as pf
import rapidtide.workflows.showtc as showtc
from rapidtide.tests.utils import create_dir, get_examples_path, get_test_temp_path, mse


def test_runmisc(debug=False, display=False):
    # run showtc
    inputargs = [
        os.path.join(
            get_examples_path(),
            "sub-HAPPYTEST_desc-slicerescardfromfmri_timeseries.json:cardiacfromfmri,cardiacfromfmri_dlfiltered",
        ),
        "--sampletime",
        "12.5",
        "--tofile",
        "showtcout1.jpg",
        "--starttime",
        "100",
        "--endtime",
        "800",
    ]
    pf.generic_init(showtc._get_parser, showtc.showtc, inputargs=inputargs)

    inputargs = [
        os.path.join(
            get_examples_path(),
            "sub-HAPPYTEST_desc-slicerescardfromfmri_timeseries.json:cardiacfromfmri,cardiacfromfmri_dlfiltered",
        ),
        "--format",
        "separate",
        "--sampletime",
        "12.5",
        "--displaytype",
        "power",
        "--tofile",
        "showtcout2.jpg",
        "--noxax",
        "--noyax",
        "--nolegend",
        "--linewidth",
        "0.75",
        "--saveres",
        "400",
        "--title",
        "thetitle",
        "--xlabel",
        "thexlabel",
        "--ylabel",
        "theylabel",
        "--legends",
        "lf_HbO,rf_HbO",
        "--legendloc",
        "5",
        "--colors",
        "red,green",
    ]
    pf.generic_init(showtc._get_parser, showtc.showtc, inputargs=inputargs)

    inputargs = [
        os.path.join(
            get_examples_path(),
            "sub-HAPPYTEST_desc-slicerescardfromfmri_timeseries.json:cardiacfromfmri,cardiacfromfmri_dlfiltered",
        ),
        "--format",
        "separatelinked",
        "--sampletime",
        "12.5",
        "--displaytype",
        "phase",
        "--tofile",
        "showtcout3.jpg",
    ]
    pf.generic_init(showtc._get_parser, showtc.showtc, inputargs=inputargs)


def main():
    test_runmisc(debug=True, display=True)


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
