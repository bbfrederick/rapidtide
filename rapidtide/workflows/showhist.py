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
from matplotlib.pyplot import bar, legend, plot, savefig, show, title, xlabel, ylabel
from numpy.typing import NDArray

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats


def _get_parser() -> Any:
    """
    Create and configure an argument parser for the showhist command-line tool.

    This function sets up an `argparse.ArgumentParser` with specific arguments
    to control the behavior of the histogram plotting script. It defines
    required and optional parameters for input file, axis labels, title,
    output file, plot style, and debugging options.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with defined arguments for
        controlling histogram plotting behavior.

    Notes
    -----
    The parser is configured with `allow_abbrev=False` to prevent
    abbreviated argument names from being accepted.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['data.txt'])
    >>> print(args.infilename)
    'data.txt'
    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="showhist",
        description="Plots xy histogram data in text file.",
        allow_abbrev=False,
    )
    parser.add_argument("infilename", help="a text file containing histogram data")
    parser.add_argument(
        "--xlabel",
        dest="thexlabel",
        type=str,
        metavar="XLABEL",
        help="Use XLABEL on the x axis.",
        default=None,
    )
    parser.add_argument(
        "--ylabel",
        dest="theylabel",
        type=str,
        metavar="YLABEL",
        help="Use YLABEL on the y axis.",
        default=None,
    )
    parser.add_argument(
        "--title",
        dest="thetitle",
        type=str,
        metavar="TITLE",
        help="Use TITLE at the top of the graph.",
        default=None,
    )
    parser.add_argument(
        "--outputfile",
        dest="outputfile",
        type=str,
        metavar="FILENAME",
        help="Save plot to FILENAME rather than displaying on the screen.",
        default=None,
    )
    parser.add_argument(
        "--dobars",
        dest="dobars",
        action="store_true",
        help="Plot bars rather than lines.",
        default=False,
    )
    parser.add_argument(
        "--calcdist",
        dest="calcdist",
        action="store_true",
        help="Make a histogram out of the data.",
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Print additional internal information.",
        default=False,
    )

    return parser


def showhist(args: Any) -> None:
    """
    Display or save a histogram or line plot based on input data and arguments.

    This function reads data from a specified input file, computes a histogram if
    requested, and visualizes the results using matplotlib. The plot can be displayed
    on screen or saved to a file depending on the provided arguments.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - infilename : str
            Path to the input file containing data to be plotted.
        - debug : bool
            If True, prints the args object for debugging purposes.
        - calcdist : bool
            If True, computes a histogram from the input data.
        - dobars : bool
            If True, displays the histogram as bars; otherwise, as a line plot.
        - thetitle : str, optional
            Title for the plot.
        - thexlabel : str, optional
            Label for the x-axis.
        - theylabel : str, optional
            Label for the y-axis.
        - outputfile : str, optional
            Path to save the plot. If None, the plot is displayed on screen.

    Returns
    -------
    None
        This function does not return any value. It either displays the plot
        or saves it to a file.

    Notes
    -----
    - The function uses `tide_io.readvecs` to read input data.
    - If `calcdist` is True, `tide_stats.makehistogram` is used to compute the histogram.
    - The histogram is displayed as either bars or a line, depending on the `dobars` flag.
    - If `outputfile` is provided, the plot is saved using `savefig`; otherwise, `show()` is called.

    Examples
    --------
    >>> args = type('Args', (), {
    ...     'infilename': 'data.txt',
    ...     'debug': False,
    ...     'calcdist': True,
    ...     'dobars': True,
    ...     'thetitle': 'Sample Histogram',
    ...     'thexlabel': 'Values',
    ...     'theylabel': 'Frequency',
    ...     'outputfile': 'output.png'
    ... })()
    >>> showhist(args)
    """
    if args.debug:
        print(args)
    if args.calcdist:
        histlen = 101
        inlist = tide_io.readvecs(args.infilename)
        (
            thehist,
            peakheight,
            peakloc,
            peakwidth,
            centerofmass,
            peakpercentile,
        ) = tide_stats.makehistogram(inlist, histlen)
        indata = np.zeros((2, len(thehist[0])), dtype="float64")
        indata[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
        indata[1, :] = thehist[0][-histlen:]
    else:
        indata = tide_io.readvecs(args.infilename)
    xvecs = indata[0, :]
    yvecs = indata[1, :]
    if args.dobars:
        bar(xvecs, yvecs, width=(0.8 * (xvecs[1] - xvecs[0])), color="g")
    else:
        plot(xvecs, yvecs, "r")

    if args.thetitle is not None:
        title(args.thetitle)

    if args.thexlabel is not None:
        xlabel(args.thexlabel, fontsize=16, fontweight="bold")

    if args.theylabel is not None:
        ylabel(args.theylabel, fontsize=16, fontweight="bold")

    if args.outputfile is None:
        show()
    else:
        savefig(args.outputfile, bbox_inches="tight")
