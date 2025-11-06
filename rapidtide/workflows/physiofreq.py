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
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter, welch

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math


def _get_parser() -> Any:
    """
    Create and configure argument parser for physiofreq command line tool.

    This function initializes an ArgumentParser object with all necessary
    command line arguments for the physiofreq tool, which is designed to
    find the dominant frequency in cardiac or respiratory waveforms.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required arguments
        for the physiofreq tool.

    Notes
    -----
    The parser includes arguments for input file specification, display options,
    sampling rate, frequency range constraints, and smoothing settings.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.txt', '--display', '--samplerate', '2.0'])
    >>> print(args.textfilename)
    'input.txt'
    """
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
    thewaveform: Any,
    thesamplerate: Any,
    minpermin: float = 40.0,
    maxpermin: float = 140.0,
    smooth: bool = True,
    smoothlen: int = 101,
    debug: bool = False,
    displayplots: bool = False,
) -> None:
    """
    Compute the fundamental frequency of a waveform using Welch's method and spectral analysis.

    This function estimates the fundamental frequency of a given waveform by computing
    its power spectrum using Welch's method, applying filtering and smoothing, and
    identifying the peak within a specified frequency range. The result is returned in
    beats per minute (BPM).

    Parameters
    ----------
    thewaveform : array-like
        Input waveform data to analyze.
    thesamplerate : float
        Sampling rate of the waveform in Hz.
    minpermin : float, optional
        Minimum allowed frequency in BPM. Default is 40.0.
    maxpermin : float, optional
        Maximum allowed frequency in BPM. Default is 140.0.
    smooth : bool, optional
        If True, apply Savitzky-Golay smoothing to the spectrum. Default is True.
    smoothlen : int, optional
        Length of the smoothing window for Savitzky-Golay filter. Default is 101.
    debug : bool, optional
        If True, print debug information during computation. Default is False.
    displayplots : bool, optional
        If True, display intermediate plots of the power spectrum. Default is False.

    Returns
    -------
    float
        Estimated fundamental frequency in BPM.

    Notes
    -----
    - The function internally uses `scipy.signal.welch` for power spectral density estimation.
    - A Hamming window and detrending are applied before spectral analysis.
    - The frequency range is constrained to the interval [minpermin, maxpermin].
    - If `displayplots` is True, two plots will be shown:
        1. Initial power spectrum with peak marked.
        2. Smoothed spectrum with final peak marked.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import chirp
    >>> t = np.linspace(0, 5, 1000, endpoint=False)
    >>> signal = chirp(t, f0=60, f1=120, t1=5, method='linear')
    >>> freq = getwavefreq(signal, thesamplerate=1000, debug=True)
    >>> print(f"Estimated frequency: {freq} BPM")
    """
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


def physiofreq(args: Any) -> None:
    """
    Calculate and display the dominant frequency of physiological data.

    This function reads physiological time series data from a file and determines
    the peak frequency using wavelet analysis. It supports both JSON and standard
    text file formats, and displays the results in Hz, BPM, and period in seconds.

    Parameters
    ----------
    args : Any
        An object containing command line arguments with the following attributes:
        - textfilename : str
            Path to the input file containing physiological data
        - samplerate : float, optional
            Sampling rate of the data (used when file is not in JSON format)
        - lowestbpm : float, optional
            Minimum allowed heart rate in beats per minute (default: 30)
        - highestbpm : float, optional
            Maximum allowed heart rate in beats per minute (default: 200)
        - nosmooth : bool, optional
            If True, disables smoothing of the frequency spectrum
        - displayplots : bool, optional
            If True, displays frequency plots

    Returns
    -------
    None
        This function prints the frequency analysis results to stdout and does not return a value.

    Notes
    -----
    The function automatically detects the file format based on the extension:
    - JSON files are processed using `tide_io.readbidstsv()`
    - Other formats are processed using `tide_io.readvecs()`

    Examples
    --------
    >>> args = type('Args', (), {
    ...     'textfilename': 'data.txt',
    ...     'samplerate': 100.0,
    ...     'lowestbpm': 40,
    ...     'highestbpm': 180,
    ...     'nosmooth': False,
    ...     'displayplots': False
    ... })()
    >>> physiofreq(args)
    data.txt:	0.83 Hz, 49.80 per minute, period is 1.20 seconds
    """
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
