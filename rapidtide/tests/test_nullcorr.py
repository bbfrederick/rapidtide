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
from __future__ import print_function, division

import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.correlate as tide_corr
import rapidtide.stats as tide_stats
import rapidtide.io as tide_io
import rapidtide.nullcorrpass as tide_nullcorr
import rapidtide.nullcorrpassx as tide_nullcorrx

import matplotlib.pyplot as plt
from rapidtide.tests.utils import get_test_data_path, get_test_target_path, get_test_temp_path, get_examples_path, get_rapidtide_root, get_scripts_path, create_dir
import os


def test_nullcorr(debug=False, display=False):
    # make the lfo filter
    lfofilter = tide_filt.noncausalfilter(filtertype='lfo')

    # make the starting regressor
    timestep = 1.5
    Fs = 1.0 / timestep
    #sourcelen = 1200
    #sourcedata = lfofilter.apply(Fs, np.random.rand(sourcelen))
    sourcedata = tide_io.readvecs(os.path.join(get_test_data_path(), 'fmri_globalmean.txt'))[0]
    sourcelen = len(sourcedata)
    numpasses = 3

    if display:
        plt.figure()
        plt.plot(sourcedata)
        plt.show()

    thexcorr = tide_corr.fastcorrelate(sourcedata, sourcedata)
    xcorrlen = len(thexcorr)
    xcorr_x = np.linspace(0.0, xcorrlen, xcorrlen, endpoint=False) * timestep - (xcorrlen * timestep) / 2.0 + timestep / 2.0

    if display:
        plt.figure()
        plt.plot(xcorr_x, thexcorr)
        plt.show()


    corrzero = xcorrlen // 2
    lagmin = -10
    lagmax = 10
    lagmininpts = int((-lagmin / timestep) - 0.5)
    lagmaxinpts = int((lagmax / timestep) + 0.5)

    searchstart = int(np.round(corrzero + lagmin / timestep))
    searchend = int(np.round(corrzero + lagmax / timestep))

    optiondict = {
        'numestreps':        10000,
        'showprogressbar':   debug,
        'usewindowfunc':     True,
        'detrendorder':      3,
        'windowfunc':        'hamming',
        'corrweighting':     'none',
        'nprocs':            1,
        'widthlimit':        1000.0,
        'bipolar':           False,
        'fixdelay':          False,
        'findmaxtype':       'gauss',
        'lagmin':            lagmin,
        'lagmax':            lagmax,
        'absmaxsigma':       25.0,
        'edgebufferfrac':    0.0,
        'lthreshval':        0.0,
        'uthreshval':        1.0,
        'debug':             False,
        'gaussrefine':       True,
        'fastgauss':         False,
        'enforcethresh':     True,
        'lagmod':            1000.0,
        'searchfrac':        0.5,
        'permutationmethod': 'shuffle',
        'hardlimit':         True
    }

    if debug:
        print(optiondict)

    for nullfunction in [tide_nullcorrx.getNullDistributionDatax, tide_nullcorr.getNullDistributionData]:
        for i in range(numpasses):
            if nullfunction == tide_nullcorr.getNullDistributionData:
                corrlist = nullfunction(sourcedata,
                                        xcorr_x,
                                        lfofilter,
                                        Fs,
                                        corrzero,
                                        lagmininpts,
                                        lagmaxinpts,
                                        optiondict)
            else:
                break
    
            tide_io.writenpvecs(corrlist, os.path.join(get_test_temp_path(), 'corrdistdata.txt'))
    
            # calculate percentiles for the crosscorrelation from the distribution data
            histlen = 250
            thepercentiles = [0.95, 0.99, 0.995]
    
            pcts, pcts_fit, histfit = tide_stats.sigFromDistributionData(corrlist, histlen, thepercentiles)
            if debug:
                tide_stats.printthresholds(pcts, thepercentiles, 'Crosscorrelation significance thresholds from data:')
                tide_stats.printthresholds(pcts_fit, thepercentiles, 'Crosscorrelation significance thresholds from fit:')

            tide_stats.makeandsavehistogram(corrlist, histlen, 0,
                                            os.path.join(get_test_temp_path(), 'correlationhist'),
                                            displaytitle='Null correlation histogram',
                                            displayplots=display, refine=False)

            assert True


if __name__ == '__main__':
    test_nullcorr(debug=True, display=True)
