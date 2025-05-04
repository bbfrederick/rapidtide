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
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.workflows.cleanregressor as tide_cleanregressor
from rapidtide.tests.utils import get_examples_path, get_test_temp_path, mse


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
    padtrs = 30
    fmrifreq = 1.0
    oversampfac = 2
    oversampfreq = oversampfac * fmrifreq
    theprefilter = tide_filt.NoncausalFilter("lfo")
    lagmin = -30
    lagmax = 30
    lagmininpts = int((lagmin * oversampfreq) - 0.5)
    lagmaxinpts = int((lagmax * oversampfreq) + 0.5)
    lagmod = 1000.0
    noiseamp = 0.25
    detrendorder = 3
    windowfunc = "hamming"

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
        allowhighfitamps=True,
        enforcethresh=False,
        zerooutbadfit=False,
    )

    # make a reference timecourse
    rng = np.random.default_rng(seed=1234)
    basewave = theprefilter.apply(fmrifreq, rng.normal(loc=0.0, scale=1.0, size=tclen))
    noisewave = rng.normal(loc=0.0, scale=noiseamp, size=tclen)
    theparamsets = [
        [2.0, 0.0, False, 0],
        [2.5, 0.8, True, 0],
        [5.0, 0.5, True, 0],
        [7.5, 0.25, True, 100],
        [10.0, 0.1, True, 0],
    ]
    for paramset in theparamsets:
        echotime = paramset[0]
        echoamp = paramset[1]
        check_autocorrelation = paramset[2]
        osvalidsimcalcstart = paramset[3]
        if debug:
            print(
                "**********Start******************************************************************"
            )
            print(f"{echotime=}, {echoamp=}, {check_autocorrelation=}, {osvalidsimcalcstart=}")
            print(
                "*********************************************************************************"
            )
        theechotc, dummy, dummy, dummy = tide_resample.timeshift(
            basewave, echotime * oversampfreq, padtrs, doplot=displayplots, debug=debug
        )
        resampnonosref_y = basewave + echoamp * theechotc + noisewave
        resampref_y = tide_resample.upsample(resampnonosref_y, fmrifreq, oversampfreq)
        theCorrelator.setreftc(resampnonosref_y)
        referencetc = tide_math.corrnormalize(
            resampref_y[osvalidsimcalcstart:],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )

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
            check_autocorrelation=check_autocorrelation,
            fix_autocorrelation=True,
            despeckle_thresh=5.0,
            lthreshval=0.0,
            fixdelay=False,
            detrendorder=detrendorder,
            windowfunc=windowfunc,
            respdelete=False,
            displayplots=displayplots,
            debug=debug,
            rt_floattype="float64",
            rt_floatset=np.float64,
        )
        print(f"\t{len(referencetc)=}")
        print(f"\t{len(resampref_y)=}")
        print(f"\t{len(resampnonosref_y)=}")
        print(f"\t{len(cleaned_resampref_y)=}")
        print(f"\t{len(cleaned_referencetc)=}")
        print(f"\t{len(cleaned_nonosreferencetc)=}")
        print(f"\t{check_autocorrelation=}")
        print(f"\t{despeckle_thresh=}")
        print(f"\t{sidelobeamp=}")
        print(f"\t{sidelobetime=}")
        print(f"\t{lagmod=}")
        print(f"\t{acwidth=}")
        print(f"\t{absmaxsigma=}")
        assert len(referencetc) == len(cleaned_referencetc)
        assert len(resampref_y) == len(cleaned_resampref_y)
        assert len(resampnonosref_y) == len(cleaned_nonosreferencetc)

        if debug:
            print(
                "*********************************************************************************"
            )
            print(f"{echotime=}, {echoamp=}, {check_autocorrelation=}, {osvalidsimcalcstart=}")
            print(
                "**************End****************************************************************"
            )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_cleanregressor(debug=True, local=True, displayplots=True)
