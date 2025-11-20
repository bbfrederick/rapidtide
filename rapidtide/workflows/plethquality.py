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
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import skew

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf


def _get_parser() -> Any:
    """
    Argument parser for plethquality.

    This function creates and configures an argument parser for the plethquality
    command-line tool that calculates quality metrics from cardiac text files.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional
        arguments for plethquality functionality.

    Notes
    -----
    The parser includes both required and optional arguments for processing
    cardiac text files and generating quality metrics.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.txt', 'output.txt'])
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


def plethquality(waveform: Any, Fs: Any, S_windowsecs: float = 5.0, debug: bool = False) -> None:
    """
    Calculate the windowed skewness quality metrics for a photoplethysmogram (PPG) signal.

    This function computes the signal quality index based on the skewness of the PPG waveform
    over a sliding window, as described in Elgendi, M. "Optimal Signal Quality Index for
    Photoplethysmogram Signals". Bioengineering 2016, Vol. 3, Page 21 (2016).

    Parameters
    ----------
    waveform : array-like
        The cardiac waveform to be assessed.
    Fs : float
        The sample rate of the data in Hz.
    S_windowsecs : float, optional
        Window duration in seconds. Defaults to 5.0.
    debug : bool, optional
        Turn on extended output for debugging purposes. Defaults to False.

    Returns
    -------
    S_sqi_mean : float
        The mean value of the quality index over all time.
    S_sqi_std : float
        The standard deviation of the quality index over all time.
    S_waveform : array
        The quality metric computed over all timepoints.

    Notes
    -----
    The window size is rounded to the nearest odd number of samples to ensure symmetric
    sliding windows around each point. The skewness is calculated using `scipy.stats.skew`
    with `nan_policy="omit"` to ignore NaN values.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import skew
    >>> waveform = np.random.randn(1000)
    >>> Fs = 100.0
    >>> mean_sqi, std_sqi, sqi_waveform = plethquality(waveform, Fs)
    >>> print(f"Mean SQI: {mean_sqi:.3f}")
    Mean SQI: 0.000

    """
    # calculate S_sqi over a sliding window.  Window size should be an odd number of points.
    S_windowpts = int(np.round(S_windowsecs * Fs, 0))
    S_windowpts += 1 - S_windowpts % 2
    S_waveform = np.zeros_like(waveform)
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


def plethquality(args: Any) -> None:
    """
    Calculate plethysmography quality score and optionally display results.

    This function reads plethysmography data from a text file, calculates a quality
    score based on the signal characteristics, and writes the quality scores to an
    output file. Optionally displays the quality score plot.

    Parameters
    ----------
    args : Any
        An object containing command line arguments with the following attributes:
        - infilename : str
            Input filename containing plethysmography data
        - outfilename : str
            Output filename for quality scores
        - samplerate : float, optional
            Sampling rate of the data (if not specified, will be read from file)
        - display : bool
            Whether to display the quality score plot

    Returns
    -------
    None
        This function does not return a value but writes results to files and
        optionally displays plots.

    Notes
    -----
    The function uses `tide_io.readvectorsfromtextfile` to read data and
    `tide_io.writevec` to write quality scores. Quality scores are calculated
    using an internal `plethquality` function that analyzes signal characteristics.

    Examples
    --------
    >>> args = argparse.Namespace(
    ...     infilename='pleth_data.txt',
    ...     outfilename='quality_scores.txt',
    ...     samplerate=100.0,
    ...     display=True
    ... )
    >>> plethquality(args)
    """
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
