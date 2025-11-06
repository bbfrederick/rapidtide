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
import bisect
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import invert_float, is_float


def phase(mcv: Any) -> None:
    """
    Compute the phase angle of a complex number.

    This function calculates the phase angle (also known as the argument) of a complex number
    using the arctan2 function, which correctly handles all quadrants and special cases.

    Parameters
    ----------
    mcv : Any
        A complex number or array of complex numbers for which to compute the phase angle.
        Can be a scalar complex number or an array-like object containing complex numbers.

    Returns
    -------
    ndarray or scalar
        The phase angle in radians. The return type matches the input type, returning
        a scalar for scalar input or an array for array input. The phase angle is
        in the range [-π, π].

    Notes
    -----
    This function uses `np.arctan2(mcv.imag, mcv.real)` which is preferred over
    `np.arctan(mcv.imag/mcv.real)` because it correctly handles the quadrant
    and avoids division by zero errors.

    Examples
    --------
    >>> import numpy as np
    >>> phase(1+1j)
    0.7853981633974483

    >>> phase(-1-1j)
    -2.3561944901923448

    >>> phase(np.array([1+1j, -1-1j]))
    array([ 0.78539816, -2.35619449])
    """
    return np.arctan2(mcv.imag, mcv.real)


def _get_parser() -> Any:
    """
    Create and configure an argument parser for the endtidalproc command-line tool.

    This function sets up an `argparse.ArgumentParser` with various options to process
    a gas trace and generate an endtidal waveform. It supports input and output file
    specifications, sample rate configuration, time range settings, threshold for
    peak detection, and debugging options.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional arguments.

    Notes
    -----
    The parser is designed for use with the `endtidalproc` program and includes
    mutually exclusive group for specifying sample rate either as frequency or
    time step. The default sample rate is 1 Hz.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.txt', 'output.txt'])
    >>> print(args.infilename)
    'input.txt'
    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="endtidalproc",
        description="Process a gas trace to generate the endtidal waveform.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "infilename",
        help="a text file containing a single gas trace, one timepoint per line",
    )
    parser.add_argument("outfilename", help="a text file for the interpolated data")
    parser.add_argument(
        "--isoxygen",
        dest="isoxygen",
        action="store_true",
        help="Assume the trace is oxygen, fits the bottom of the waveform, not the top.",
        default=False,
    )
    samp_group = parser.add_mutually_exclusive_group()
    samp_group.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        type=lambda x: is_float(parser, x),
        metavar="FREQ",
        help=("The sample rate of the input data is FREQ Hz (default is 1Hz)."),
        default=None,
    )
    samp_group.add_argument(
        "--sampletime",
        dest="samplerate",
        action="store",
        type=lambda x: invert_float(parser, x),
        metavar="TSTEP",
        help=("The sample rate of the input data is 1/TSTEP Hz (default is 1Hz)."),
        default=None,
    )
    parser.add_argument(
        "--starttime",
        dest="thestarttime",
        metavar="START",
        type=float,
        help="Start plot at START seconds.",
        default=-1000000.0,
    )
    parser.add_argument(
        "--endtime",
        dest="theendtime",
        metavar="END",
        type=float,
        help="Finish plot at END seconds.",
        default=1000000.0,
    )
    parser.add_argument(
        "--thresh",
        dest="thresh",
        metavar="PCT",
        type=float,
        help="Amount of fall (or rise) needed, in percent, to recognize a peak (or through).",
        default=1.0,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Print additional internal information.",
        default=False,
    )

    return parser


def process_args(args: Any) -> None:
    """
    Process and validate input arguments for audio processing.

    This function sets default values for samplerate if not provided and
    optionally prints debug information when debug mode is enabled.

    Parameters
    ----------
    args : Any
        An object containing audio processing arguments. Expected to have
        attributes 'samplerate' and 'debug'. The 'samplerate' attribute
        will be set to 1.0 if None, and 'debug' controls whether to print
        argument information.

    Returns
    -------
    None
        This function modifies the args object in-place and does not return
        a value.

    Notes
    -----
    - The function modifies the input args object in-place
    - Default samplerate is set to 1.0 when None is provided
    - Debug output is printed only when args.debug is True

    Examples
    --------
    >>> class Args:
    ...     def __init__(self):
    ...         self.samplerate = None
    ...         self.debug = True
    ...
    >>> args = Args()
    >>> process_args(args)
    >>> print(args.samplerate)
    1.0
    """
    if args.samplerate is None:
        args.samplerate = 1.0

    if args.debug:
        print()
        print("args:")
        print(args)

    return args


def endtidalproc() -> None:
    """
    Process tidal data to detect peaks and interpolate values for output.

    This function reads a time series from an input file, detects either maximum or
    minimum peaks depending on whether the data is treated as oxygen or CO2, and
    interpolates the peak values over the full time range. The resulting interpolated
    data is written to an output file.

    Parameters
    ----------
    None
        This function does not take any direct parameters. It reads from command-line
        arguments and system input files.

    Returns
    -------
    None
        This function does not return any value. It performs file I/O operations
        and writes results to disk.

    Notes
    -----
    - The function uses `bisect` to find the start and end indices in the time vector
      based on the provided start and end times.
    - Peak detection is performed using `tide_fit.peakdetect`, which requires a
      `lookahead` parameter and a `delta` threshold.
    - If `isoxygen` is True, the function detects minimum peaks; otherwise, it detects
      maximum peaks.
    - The interpolation is linear between detected peaks.

    Examples
    --------
    >>> endtidalproc()
    Fitting trace as CO2
    endtime must be greater then starttime;
    """
    args = process_args(_get_parser().parse_args(sys.argv[1:]))

    if args.isoxygen:
        print("Fitting trace as oxygen")
    else:
        print("Fitting trace as CO2")

    # check range
    if args.thestarttime >= args.theendtime:
        print("endtime must be greater then starttime;")
        sys.exit()

    # read in the data
    yvec = tide_io.readvec(args.infilename)
    xvec = np.arange(0.0, len(yvec), 1.0) / args.samplerate

    thestartpoint = np.max([0, bisect.bisect_right(xvec, args.thestarttime)])
    theendpoint = np.min([len(xvec) - 1, bisect.bisect_left(xvec, args.theendtime)])
    args.thestarttime = xvec[thestartpoint]
    args.theendtime = xvec[theendpoint]

    # set parameters - maxtime is the longest to look ahead for a peak (or through) in seconds
    # lookahead should be '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    maxtime = 1.0
    f = 2.0
    lookaheadval = int((args.samplerate * maxtime) / f)
    maxpeaks, minpeaks = tide_fit.peakdetect(yvec, lookahead=lookaheadval, delta=args.thethresh)

    if args.isoxygen:
        peaklist = minpeaks
    else:
        peaklist = maxpeaks

    peakinterp = 0.0 * yvec
    curpos = 0
    curval = peaklist[0][1]
    for thepeak in peaklist:
        slope = (thepeak[1] - curval) / (thepeak[0] - curpos)
        for theindex in range(curpos, thepeak[0]):
            peakinterp[theindex] = curval + slope * (theindex - curpos)
        curpos = thepeak[0] + 0
        curval = thepeak[1] + 0.0
    if curpos < len(peakinterp):
        peakinterp[curpos:] = curval

    tide_io.writevec(peakinterp, args.outfilename)


if __name__ == "__main__":
    endtidalproc()
