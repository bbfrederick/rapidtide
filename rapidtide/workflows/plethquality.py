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
import argparse
import sys

import numpy as np
from scipy.stats import skew

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for plethquality
    """
    parser = argparse.ArgumentParser(
        prog="plethquality",
        description=("Calculate quality metrics from a cardiac text file."),
        allow_abbrev=False,
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "infilename", onecol=True)
    parser.add_argument("outfilename", help="The name of the output text file.")

    # add optional arguments
    parser.add_argument(
        "--samplerate",
        type=lambda x: pf.is_float(parser, x),
        help="Sample rate of timecourse, in Hz (if not specified in input file).",
        default=None,
    )
    parser.add_argument(
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Do not plot the data (for noninteractive use)."),
        default=True,
    )
    return parser


def plethquality(waveform, Fs, S_windowsecs=5.0, debug=False):
    """

    Parameters
    ----------
    waveform: array-like
        The cardiac waveform to be assessed
    Fs: float
        The sample rate of the data
    S_windowsecs: float
        Window duration in seconds.  Defaults to 2.0 (optimal according to Elgendi
    debug: boolean
        Turn on extended output

    Returns
    -------
    S_sqi_mean: float
        The mean value of the quality index over all time
    S_std_mean: float
        The standard deviation of the quality index over all time
    S_waveform: array
        The quality metric over all timepoints


    Calculates the windowed skewness quality metrics described in Elgendi, M. "Optimal Signal Quality Index for
    Photoplethysmogram Signals". Bioengineering 2016, Vol. 3, Page 21 3, 21 (2016).
    """
    # calculate S_sqi over a sliding window.  Window size should be an odd number of points.
    S_windowpts = int(np.round(S_windowsecs * Fs, 0))
    S_windowpts += 1 - S_windowpts % 2
    S_waveform = waveform * 0.0
    if debug:
        print("S_windowsecs, S_windowpts:", S_windowsecs, S_windowpts)
    for i in range(0, len(waveform)):
        startpt = np.max([0, i - S_windowpts // 2])
        endpt = np.min([i + S_windowpts // 2, len(waveform)])
        S_waveform[i] = skew(waveform[startpt : endpt + 1], nan_policy="omit")
        if debug:
            print(i, startpt, endpt, endpt - startpt + 1, S_waveform[i])

    S_sqi_mean = np.mean(S_waveform)
    S_sqi_std = np.std(S_waveform)

    return S_sqi_mean, S_sqi_std, S_waveform


def plethquality(args):
    if args.display:
        import matplotlib as mpl

        mpl.use("TkAgg")
        import matplotlib.pyplot as plt

    Fs, starttime, dummy, plethdata, dummy, dummy = tide_io.readvectorsfromtextfile(
        args.infilename, onecol=True
    )

    if args.samplerate is not None:
        Fs = args.samplerate
    elif Fs is None:
        print("no samplerate found in file - must be set with the --samplerate option")
        sys.exit()

    # calculate the quality score
    s_mean, s_std, quality = plethquality(plethdata, Fs)
    print(args.infilename, s_mean, "+/-", s_std)
    tide_io.writevec(quality, args.outfilename)

    if args.display:
        plt.figure()
        plt.plot(quality)
        plt.show()
