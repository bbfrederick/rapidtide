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
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats
import rapidtide.workflows.parser_funcs as pf


def _get_parser() -> Any:
    """
    Argument parser for histtc.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    for parsing command-line arguments for the `histtc` tool, which generates histograms
    from timecourse data.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for histtc.

    Notes
    -----
    The parser includes both required and optional arguments for specifying input/output
    files, histogram bin settings, and various data filtering options.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['--inputfilename', 'data.txt', '--outputroot', 'output'])
    """
    parser = argparse.ArgumentParser(
        prog="histtc",
        description=("Generate a histogram of the values in a timecourse"),
        allow_abbrev=False,
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "inputfilename", onecol=True)
    pf.addreqoutputtextfile(parser, "outputroot")

    # add optional arguments
    parser.add_argument(
        "--numbins",
        dest="histlen",
        action="store",
        type=int,
        metavar="BINS",
        help=("Number of histogram bins (default is 101)"),
        default=101,
    )
    parser.add_argument(
        "--minval",
        dest="minval",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        metavar="MINVAL",
        help="Minimum bin value in histogram.",
        default=None,
    )
    parser.add_argument(
        "--maxval",
        dest="maxval",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        metavar="MAXVAL",
        help="Maximum bin value in histogram.",
        default=None,
    )
    parser.add_argument(
        "--robustrange",
        dest="robustrange",
        action="store_true",
        help=("Set histogram limits to the data's robust range (2nd to 98th percentile)."),
        default=False,
    )
    parser.add_argument(
        "--nozero",
        dest="nozero",
        action="store_true",
        help=("Do not include zero values in the histogram."),
        default=False,
    )
    parser.add_argument(
        "--nozerothresh",
        dest="nozerothresh",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        metavar="THRESH",
        help="Absolute values less than this are considered zero.  Default is 0.01.",
        default=0.01,
    )
    parser.add_argument(
        "--normhist",
        dest="normhist",
        action="store_true",
        help=("Return a probability density instead of raw counts."),
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Print additional debugging information."),
        default=False,
    )

    return parser


def histtc(args: Any) -> None:
    """
    Compute and save a histogram of input data, optionally filtered and normalized.

    This function reads input data from a text file, computes specified percentiles,
    and generates a histogram with optional normalization and range settings.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the following attributes:
        - inputfilename : str
            Path to the input text file containing the data.
        - debug : bool
            If True, print debug information during execution.
        - nozero : bool
            If True, exclude values with absolute magnitude less than or equal to `nozerothresh`.
        - nozerothresh : float
            Threshold for excluding zero-like values when `nozero` is True.
        - histlen : int
            Number of bins for the histogram.
        - outputroot : str
            Root name for output files.
        - robustrange : bool
            If True, use robust range (2nd and 98th percentiles) for histogram.
        - minval : float, optional
            Minimum value for histogram range. If None, use data minimum.
        - maxval : float, optional
            Maximum value for histogram range. If None, use data maximum.
        - normhist : bool
            If True, normalize the histogram.

    Returns
    -------
    None
        This function does not return a value but saves a histogram to disk.

    Notes
    -----
    The function uses `tide_io.readvectorsfromtextfile` to read input data and
    `tide_stats.getfracvals` to compute percentiles. It then calls
    `tide_stats.makeandsavehistogram` to generate and save the histogram.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfilename='data.txt',
    ...     debug=False,
    ...     nozero=True,
    ...     nozerothresh=1e-6,
    ...     histlen=50,
    ...     outputroot='output',
    ...     robustrange=False,
    ...     minval=None,
    ...     maxval=None,
    ...     normhist=True
    ... )
    >>> histtc(args)
    """
    # set default variable values
    thepercentiles = [0.2, 0.25, 0.5, 0.75, 0.98]
    thepvalnames = []
    for thispercentile in thepercentiles:
        thepvalnames.append(str(1.0 - thispercentile).replace(".", "p"))
    pcts_data = np.zeros((len(thepercentiles)), dtype="float")

    if args.debug:
        print(args)

    dummy, dummy, colnames, inputdata, compressed, filetype = tide_io.readvectorsfromtextfile(
        args.inputfilename
    )

    inputdata = inputdata[0]

    if args.debug:
        print(inputdata)

    if args.nozero:
        inputdata = inputdata[np.where(np.fabs(inputdata) > args.nozerothresh)]

    if args.debug:
        print(inputdata)

    pcts_data[:] = tide_stats.getfracvals(inputdata, thepercentiles)
    for idx, thispercentile in enumerate(thepercentiles):
        print(f"percentile {thispercentile} is {pcts_data[idx]}")

    if args.robustrange:
        histmin, histmax = tide_stats.getfracvals(inputdata, [0.02, 0.98])
    else:
        if args.minval is None:
            histmin = np.min(inputdata)
        else:
            histmin = args.minval
        if args.maxval is None:
            histmax = np.max(inputdata)
        else:
            histmax = args.maxval
    therange = (histmin, histmax)
    print("the range is ", therange, flush=True)

    tide_stats.makeandsavehistogram(
        inputdata,
        args.histlen,
        0,
        args.outputroot,
        therange=therange,
        normalize=args.normhist,
        debug=args.debug,
    )
