#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2020-2025 Blaise Frederick
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

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, welch

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="physiofreq",
        description="Finds the dominant frequency in a cardiac or respiratory waveform.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "textfilename", help="A text input files, with optional column specification"
    )
    parser.add_argument(
        "--display",
        dest="displayplots",
        action="store_true",
        help="display the fit spectrum",
    )
    parser.add_argument(
        "--samplerate",
        dest="samplerate",
        type=float,
        default=1.0,
        help="sample rate of the waveform in Hz",
    )
    parser.add_argument(
        "--lowestbpm",
        dest="lowestbpm",
        type=float,
        default=6.0,
        help="Lowest allowable frequency in cycles per minute",
    )
    parser.add_argument(
        "--highestbpm",
        dest="highestbpm",
        type=float,
        default=20.0,
        help="Highest allowable frequency in cycles per minute",
    )
    parser.add_argument(
        "--disablesmoothing",
        dest="nosmooth",
        action="store_true",
        help="Do not apply Savitsky-Golay filter to spectrum",
    )
    return parser


def getwavefreq(
    thewaveform,
    thesamplerate,
    minpermin=40.0,
    maxpermin=140.0,
    smooth=True,
    smoothlen=101,
    debug=False,
    displayplots=False,
):
    if len(thewaveform) % 2 == 1:
        thewaveform = thewaveform[:-1]
    if len(thewaveform) > 1024:
        thex, they = welch(thewaveform, thesamplerate, nperseg=1024)
    else:
        thex, they = welch(thewaveform, thesamplerate)
    initpeakfreq = np.round(thex[np.argmax(they)] * 60.0, 2)
    if displayplots:
        plt.figure()
        plt.plot(thex, they, "k")
        plt.plot(
            [initpeakfreq / 60.0, initpeakfreq / 60.0],
            [np.min(they), np.max(they)],
            "r-",
            lw=2,
        )
        plt.show()
    if initpeakfreq > maxpermin:
        initpeakfreq = maxpermin
    if initpeakfreq < minpermin:
        initpeakfreq = minpermin
    if debug:
        print("initpeakfreq:", initpeakfreq, "BPM")

    # calculate the power spectrum
    normwave = tide_filt.hamming(len(thewaveform)) * tide_fit.detrend(
        thewaveform, order=1, demean=True
    )
    freqaxis, spectrum = tide_filt.spectrum(normwave, Fs=thesamplerate, mode="power")

    # Constrain fit to valid range
    binsize = freqaxis[1] - freqaxis[0]
    minbin = int(minpermin // (60.0 * binsize))
    maxbin = int(maxpermin // (60.0 * binsize))
    spectrum[:minbin] = 0.0
    spectrum[maxbin:] = 0.0

    # find the max
    if smooth:
        ampspec = tide_filt.savgolsmooth(spectrum, smoothlen=smoothlen)
    else:
        ampspec = spectrum
    peakfreq = freqaxis[np.argmax(ampspec)]
    if displayplots:
        plt.figure()
        plt.plot(freqaxis, ampspec, "k")
        plt.plot([peakfreq, peakfreq], [np.min(ampspec), np.max(ampspec)], "r-", lw=2)
        plt.xlim([0.0, 1.1 * maxpermin / 60.0])
        plt.show()
    if debug:
        print("the fundamental frequency is", np.round(peakfreq * 60.0, 2), "BPM")
    normfac = np.sqrt(2.0) * tide_math.rms(thewaveform)
    if debug:
        print("normfac:", normfac)
    return peakfreq


def physiofreq(args):
    textfileinfo, textfilecolspec = tide_io.parsefilespec(args.textfilename)
    filebase, extension = os.path.splitext(textfileinfo[0])
    if extension == ".json":
        (
            thissamplerate,
            thisstartoffset,
            colnames,
            invec,
            compressed,
            fakecolumns,
        ) = tide_io.readbidstsv(textfileinfo[0])
    else:
        invec = tide_io.readvecs(textfileinfo[0])[0]
        thissamplerate = args.samplerate
        thisstartoffset = 0.0
        colnames = None
    peakfreq = getwavefreq(
        invec,
        thissamplerate,
        minpermin=args.lowestbpm,
        maxpermin=args.highestbpm,
        smooth=(not args.nosmooth),
        displayplots=args.displayplots,
    )
    print(
        textfileinfo[0]
        + ":\t"
        + "%.2f" % peakfreq
        + " Hz, "
        + "%.2f" % (peakfreq * 60.0)
        + " per minute, period is "
        + "%.2f" % (1.0 / peakfreq),
        "seconds",
    )
