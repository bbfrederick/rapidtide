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

import rapidtide.correlate as tide_corr
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.workflows.parser_funcs as pf


def main(args):
    print(args)
    if args.display:
        import matplotlib as mpl

        mpl.use("TkAgg")
        import matplotlib.pyplot as plt

    args = pf.postprocesssearchrangeopts(args)
    args, theprefilter = pf.postprocessfilteropts(args)

    intimestep1 = 1.0 / args.insamplerate1
    intimestep2 = 1.0 / args.insamplerate2

    inputdata1 = tide_io.readcolfromtextfile(args.infile1)
    inputdata2 = tide_io.readcolfromtextfile(args.infile2)

    # determine waveform lengths
    time1 = args.insamplerate1 * (len(inputdata1) - 1)
    time2 = args.insamplerate2 * (len(inputdata2) - 1)

    fulltime = np.max([time1, time2])
    # pad waveform1 if it's shorter than waveform2
    if time1 < fulltime:
        paddeddata1 = np.zeros(int(np.ceil(fulltime // intimestep1)), dtype=float)
        paddeddata1[0 : len(inputdata1) + 1] = tide_math.corrnormalize(
            theprefilter.apply(args.insamplerate1, inputdata1)
        )
    else:
        paddeddata1 = tide_math.corrnormalize(theprefilter.apply(args.insamplerate1, inputdata1))

    timeaxisfull = np.linspace(
        0.0, intimestep1 * len(paddeddata1), num=len(paddeddata1), endpoint=False
    )
    timeaxis1 = np.linspace(
        0.0, intimestep1 * len(inputdata1), num=len(inputdata1), endpoint=False
    )
    timeaxis2 = np.linspace(
        0.0, intimestep2 * len(inputdata2), num=len(inputdata2), endpoint=False
    )
    paddeddata2 = tide_resample.doresample(
        timeaxis2,
        tide_math.corrnormalize(theprefilter.apply(args.insamplerate2, inputdata2)),
        timeaxisfull,
    )

    # now paddeddata1 and 2 are on the same timescales
    thexcorr = tide_corr.fastcorrelate(paddeddata1, paddeddata2)
    xcorrlen = len(thexcorr)
    xcorr_x = (
        np.r_[0.0:xcorrlen] * intimestep1 - (xcorrlen * intimestep1) / 2.0 + intimestep1 / 2.0
    )

    (
        maxindex,
        maxdelay,
        maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    ) = tide_fit.findmaxlag_gauss(
        xcorr_x,
        thexcorr,
        args.lagmin,
        args.lagmax,
        1000.0,
        refine=True,
        useguess=False,
        fastgauss=False,
        displayplots=False,
    )

    print("Crosscorrelation_Rmax:\t", maxval)
    print("Crosscorrelation_maxdelay:\t", maxdelay)

    # now align the second timecourse to the first

    aligneddata2 = tide_resample.doresample(timeaxis2, inputdata2, timeaxis1 - maxdelay)
    tide_io.writevec(aligneddata2, args.outputfile)

    if args.display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_title('GCC')
        plt.plot(xcorr_x, thexcorr, "k")
        plt.show()
        fig = plt.figure()
        plt.plot(timeaxis1, inputdata1)
        plt.plot(timeaxis1, aligneddata2)
        plt.plot()
        plt.show()
