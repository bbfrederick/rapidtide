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
import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.helper_classes as tide_classes
import rapidtide.resample as tide_resample
import rapidtide.workflows.cleanregressor as tide_cleanregressor
from rapidtide.tests.utils import get_examples_path, get_test_temp_path


def test_cleanregressor(debug=False, local=False, displayplots=False):
    # set input and output directories
    if local:
        exampleroot = "../data/examples/src"
        testtemproot = "./tmp"
    else:
        exampleroot = get_examples_path()
        testtemproot = get_test_temp_path()

    outputname = os.path.join(testtemproot, "cleanregressortest")
    thepass = 1
    fmrifreq = 1.0
    oversampfac = 2
    oversampfreq = oversampfac * fmrifreq
    theprefilter = tide_filt.NoncausalFilter("lfo")
    lagmin = -5
    lagmax = 5
    lagmininpts = int((lagmin * oversampfreq) - 0.5)
    lagmaxinpts = int((lagmax * oversampfreq) + 0.5)
    lagmod = 1000.0

    tclen = 500
    osvalidsimcalcstart = 0
    osvalidsimcalcend = tclen * oversampfac

    theCorrelator = tide_classes.Correlator(
        Fs=oversampfreq,
        ncprefilter=theprefilter,
        detrendorder=1,
        windowfunc="hamming",
        corrweighting="phat",
    )
    theFitter = tide_classes.SimilarityFunctionFitter(
        lagmod=lagmod,
        lagmin=lagmin,
        lagmax=lagmax,
        debug=debug,
        zerooutbadfit=False,
    )

    # make a reference timecourse
    rng = np.random.default_rng(seed=1234)
    resampnonosref_y = theprefilter.apply(oversampfreq, rng.normal(loc=0.0, scale=1.0, size=tclen))
    resampref_y = tide_resample.upsample(resampnonosref_y, fmrifreq, oversampfreq)
    theCorrelator.setreftc(resampnonosref_y)
    referencetc = resampref_y

    resampref_y = tide_resample.upsample(resampnonosref_y, fmrifreq, oversampfreq)

    (
        cleaned_resampref_y,
        cleaned_referencetc,
        cleaned_nonosreferencetc,
        despeckle_thresh,
        sidelobeamp,
        sidelobetime,
        lagmod,
        acwidth,
        absmaxsigma,
    ) = tide_cleanregressor.cleanregressor(
        outputname,
        thepass,
        referencetc,
        resampref_y,
        resampnonosref_y,
        fmrifreq,
        oversampfreq,
        osvalidsimcalcstart,
        osvalidsimcalcend,
        lagmininpts,
        lagmaxinpts,
        theFitter,
        theCorrelator,
        lagmin,
        lagmax,
        LGR=None,
        check_autocorrelation=True,
        fix_autocorrelation=True,
        despeckle_thresh=5.0,
        lthreshval=0.0,
        fixdelay=False,
        detrendorder=3,
        windowfunc="hamming",
        respdelete=False,
        debug=False,
        rt_floattype="float64",
        rt_floatset=np.float64,
    )

    print(f"{despeckle_thresh=}")
    print(f"{sidelobeamp=}")
    print(f"{sidelobetime=}")
    print(f"{lagmod=}")
    print(f"{acwidth=}")
    print(f"{absmaxsigma=}")


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_cleanregressor(debug=True, local=True, displayplots=True)
