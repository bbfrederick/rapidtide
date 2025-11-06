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

import matplotlib.cm as cm
import numpy as np
from matplotlib.pyplot import (
    annotate,
    axhline,
    bar,
    legend,
    plot,
    savefig,
    scatter,
    show,
    title,
    xlabel,
    xlim,
    ylabel,
    ylim,
)
from numpy.typing import NDArray
from scipy.stats import linregress

import rapidtide.io as tide_io


def _get_parser() -> Any:
    """
    Create and configure an argument parser for plotting xy data from text files.

    This function sets up an `argparse.ArgumentParser` with a variety of options to
    control how xy data is read from text files and plotted. It supports features such as
    axis labels, range limits, title, output file saving, font scaling, legend handling,
    color specification, and special plot types like Bland-Altman.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object ready to parse command-line arguments.

    Notes
    -----
    The parser expects one or more text file names as positional arguments. Each file
    should contain whitespace-separated x and y data, one point per line.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['data.txt'])
    >>> print(args.textfilenames)
    ['data.txt']
    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="showxy",
        description="Plots xy data in text files.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "textfilenames",
        type=str,
        nargs="+",
        help="One or more text files containing whitespace separated x y data, one point per line.",
    )

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
        "--xrange",
        dest="xrange",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        help=("Limit x display range to XMIN to XMAX. "),
        default=None,
    )
    parser.add_argument(
        "--yrange",
        dest="yrange",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help=("Limit y display range to YMIN to YMAX. "),
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
        "--fontscalefac",
        dest="fontscalefac",
        metavar="FAC",
        type=float,
        action="store",
        help="Scaling factor for annotation fonts (default is 1.0).",
        default=1.0,
    )
    parser.add_argument(
        "--saveres",
        dest="saveres",
        metavar="DPI",
        type=int,
        action="store",
        help="Write figure to file at DPI dots per inch (default is 1000).",
        default=1000,
    )
    parser.add_argument(
        "--legends",
        dest="legends",
        metavar="LEGEND[,LEGEND[,LEGEND...]]",
        type=str,
        action="store",
        help="Comma separated list of legends for each timecourse.",
        default=None,
    )
    parser.add_argument(
        "--legendloc",
        dest="legendloc",
        metavar="LOC",
        type=int,
        action="store",
        help=(
            "Integer from 0 to 10 inclusive specifying legend location.  Legal values are: "
            "0: best, 1: upper right, 2: upper left, 3: lower left, 4: lower right, "
            "5: right, 6: center left, 7: center right, 8: lower center, 9: upper center, "
            "10: center.  Default is 2."
        ),
        default=2,
    )
    parser.add_argument(
        "--colors",
        dest="colors",
        metavar="COLOR[,COLOR[,COLOR...]]",
        type=str,
        action="store",
        help="Comma separated list of colors for each timecourse.",
        default=None,
    )
    parser.add_argument(
        "--blandaltman",
        dest="blandaltman",
        action="store_true",
        help="Make a Bland-Altman plot.",
        default=False,
    )
    parser.add_argument(
        "--usex",
        dest="usex",
        action="store_true",
        help="Use x instead of (y + x)/2 in Bland-Altman plot.",
        default=False,
    )
    parser.add_argument(
        "--noannotate",
        dest="doannotate",
        action="store_false",
        help="Hide annotation on Bland-Altman plots.",
        default=True,
    )
    parser.add_argument(
        "--usepoints",
        dest="usepoints",
        action="store_true",
        help="Plot as individual values (do not connect points)",
        default=False,
    )
    parser.add_argument(
        "--dobars",
        dest="dobars",
        action="store_true",
        help="Plot bars rather than lines.",
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


def bland_altman_plot(
    data1: Any,
    data2: Any,
    usex: bool = False,
    identifier: Optional[Any] = None,
    fontscalefac: float = 1.0,
    xrange: Optional[Any] = None,
    yrange: Optional[Any] = None,
    doannotate: bool = True,
    debug: bool = False,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Create a Bland-Altman plot to compare two measurement methods.

    This function generates a Bland-Altman plot, which is used to compare two
    measurement methods by plotting the difference between the measurements
    against their mean. It also computes and displays key statistics such as
    mean difference, standard deviation of differences, and regression slopes.

    Parameters
    ----------
    data1 : array-like
        First set of measurements (X-axis data if `usex=True`).
    data2 : array-like
        Second set of measurements (Y-axis data).
    usex : bool, optional
        If True, use `data1` as the X-axis values; otherwise, use the mean of
        `data1` and `data2` as the X-axis values. Default is False.
    identifier : optional
        Identifier for the dataset, printed for reference. Default is None.
    fontscalefac : float, optional
        Scaling factor for font size in annotations. Default is 1.0.
    xrange : array-like, optional
        X-axis limits for the plot. Default is None.
    yrange : array-like, optional
        Y-axis limits for the plot. Default is None.
    doannotate : bool, optional
        If True, annotate the plot with mean difference and standard deviation.
        Default is True.
    debug : bool, optional
        If True, print debug information. Default is False.
    *args : tuple
        Additional arguments passed to the scatter plot.
    **kwargs : dict
        Additional keyword arguments passed to the scatter plot.

    Returns
    -------
    None
        This function does not return a value but displays a plot.

    Notes
    -----
    The Bland-Altman plot is useful for assessing the agreement between two
    measurement methods. The plot includes horizontal lines indicating the mean
    difference and ±2 standard deviations from the mean.

    Examples
    --------
    >>> import numpy as np
    >>> data1 = np.array([1, 2, 3, 4, 5])
    >>> data2 = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    >>> bland_altman_plot(data1, data2)
    """
    # data1 is X, data2 is Y
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    slope, intercept, r_value, p_value, std_err = linregress(data1, data2)
    if usex:
        if debug:
            print("using X as the X axis")
        diff_slope, diff_intercept, diff_r_value, diff_p_value, diff_std_err = linregress(
            data1, data2 - data1
        )
        mean = data1
    else:
        if debug:
            print("using (Y + X)/2 as the X axis")
        diff_slope, diff_intercept, diff_r_value, diff_p_value, diff_std_err = linregress(
            (data2 + data1) / 2.0, data2 - data1
        )
        mean = np.mean([data1, data2], axis=0)
    diff = data2 - data1  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference
    print()
    if identifier is not None:
        print("id:", identifier)
    print("slope:", slope)
    print("r_value:", r_value)
    print("p_value:", p_value)
    print("diff_slope:", diff_slope)
    print("diff_r_value:", diff_r_value)
    print("diff_p_value:", diff_p_value)
    print("mean difference:", md)
    print("std difference:", sd)
    mdstring = "Mean diff: " + format(md, ".3f")
    sdstring = "Std diff: " + format(sd, ".3f")
    if doannotate:
        annotate(
            mdstring + "\n" + sdstring,
            xy=(0.45, 0.75),
            xycoords="figure fraction",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=fontscalefac * 16,
        )

    scatter(mean, diff, facecolors="none", edgecolors="b", *args, **kwargs)
    axhline(md, color="gray", linestyle="--")
    axhline(md + 2 * sd, color="gray", linestyle="--")
    axhline(md - 2 * sd, color="gray", linestyle="--")
    if xrange is not None:
        xlim(xrange)
    if yrange is not None:
        ylim(yrange)


def stringtorange(thestring: Any) -> None:
    """
    Convert a comma-separated string to a tuple of floats representing a range.

    Parameters
    ----------
    thestring : Any
        A string containing two comma-separated float values representing
        the minimum and maximum values of a range.

    Returns
    -------
    tuple of float
        A tuple containing (min_value, max_value) as floats.

    Notes
    -----
    This function expects exactly two comma-separated float values. If the
    input string does not contain exactly two values or if the values cannot
    be converted to floats, the program will exit with an error message.

    Examples
    --------
    >>> stringtorange("1.5,10.0")
    (1.5, 10.0)

    >>> stringtorange("0,-5.5")
    (0.0, -5.5)
    """
    thelist = thestring.split(",")
    if len(thelist) != 2:
        print("range setting requires two comma separated floats - exiting")
        sys.exit()
    try:
        themin = float(thelist[0])
    except ValueError:
        print("range setting requires two comma separated floats - exiting")
        sys.exit()
    try:
        themax = float(thelist[1])
    except ValueError:
        print("range setting requires two comma separated floats - exiting")
        sys.exit()
    return (themin, themax)


def showxy(args: Any) -> None:
    """
    Display or save xy data plots based on command-line arguments.

    This function reads data from text files and plots the corresponding x and y vectors.
    It supports multiple plot types including line plots, bar plots, and Bland-Altman plots.
    The function also handles legend, title, axis labels, and output file saving.

    Parameters
    ----------
    args : Any
        An object containing various arguments for plotting, including:
        - `textfilenames`: List of text files containing data vectors.
        - `colors`: Comma-separated string of color names for plotting.
        - `legends`: Comma-separated string of legend labels.
        - `legendloc`: Location of the legend.
        - `dobars`: Boolean indicating whether to plot bar charts.
        - `blandaltman`: Boolean indicating whether to use Bland-Altman plot.
        - `usex`: Used in Bland-Altman plot.
        - `fontscalefac`: Scaling factor for font sizes.
        - `xrange`: X-axis range.
        - `yrange`: Y-axis range.
        - `doannotate`: Boolean indicating whether to annotate points.
        - `debug`: Boolean for debugging output.
        - `usepoints`: Boolean for plotting points instead of lines.
        - `thetitle`: Title of the plot.
        - `thexlabel`: X-axis label.
        - `theylabel`: Y-axis label.
        - `outputfile`: Output file name for saving the plot.
        - `saveres`: Resolution for saved figure.

    Returns
    -------
    None
        This function does not return any value. It either displays the plot or saves it to a file.

    Notes
    -----
    - If `dobars` is True, bar plots are displayed using light gray bars.
    - If `blandaltman` is True, a Bland-Altman plot is generated.
    - If neither `dobars` nor `blandaltman` is True, line plots are displayed.
    - If `colors` is not provided, colors are selected from the `nipy_spectral` colormap.
    - If `outputfile` is None, the plot is displayed using `matplotlib.pyplot.show()`.

    Examples
    --------
    >>> args = argparse.Namespace(
    ...     textfilenames=['data1.txt', 'data2.txt'],
    ...     colors='red,blue',
    ...     legends='Dataset1, Dataset2',
    ...     legendloc='upper right',
    ...     dobars=False,
    ...     blandaltman=False,
    ...     usepoints=False,
    ...     thetitle='My Plot',
    ...     thexlabel='X-axis',
    ...     theylabel='Y-axis',
    ...     outputfile='output.png',
    ...     saveres=300
    ... )
    >>> showxy(args)
    """
    # process the file list
    textfilename = []
    for thefile in args.textfilenames:
        textfilename.append(thefile)

    # set various cosmetic aspects of the plots
    if args.colors is not None:
        colornames = args.colors.split(",")
    else:
        colornames = []

    if args.legends is not None:
        args.legends = args.legends.split(",")

    xvecs = []
    yvecs = []
    for i in range(0, len(textfilename)):
        if args.debug:
            print("reading data from", textfilename[i])
        indata = tide_io.readvecs(textfilename[i])
        xvecs.append(1.0 * indata[0, :])
        yvecs.append(1.0 * indata[1, :])

    numvecs = len(xvecs)
    if len(colornames) > 0:
        colorlist = [colornames[i % len(colornames)] for i in range(numvecs)]
    else:
        colorlist = [cm.nipy_spectral(float(i) / numvecs) for i in range(numvecs)]

    for i in range(0, len(textfilename)):
        if args.dobars:
            bar(
                xvecs[i],
                yvecs[i],
                width=1.5,
                color="lightgray",
                align="center",
                edgecolor=None,
            )
            if args.xrange is not None:
                xlim(args.xrange)
            if args.yrange is not None:
                ylim(args.yrange)

        elif args.blandaltman:
            bland_altman_plot(
                xvecs[i],
                yvecs[i],
                usex=args.usex,
                identifier=textfilename[i],
                fontscalefac=args.fontscalefac,
                xrange=args.xrange,
                yrange=args.yrange,
                doannotate=args.doannotate,
                debug=args.debug,
            )
        else:
            if args.usepoints:
                plot(xvecs[i], yvecs[i], color=colorlist[i], marker=".", linestyle="None")
            else:
                plot(xvecs[i], yvecs[i], color=colorlist[i])
            if args.xrange is not None:
                xlim(args.xrange)
            if args.yrange is not None:
                ylim(args.yrange)

    if args.legends is not None:
        legend(args.legends, loc=args.legendloc)
    if args.thetitle is not None:
        title(args.thetitle, fontsize=args.fontscalefac * 18, fontweight="bold")
    if args.thexlabel is not None:
        xlabel(args.thexlabel, fontsize=args.fontscalefac * 16, fontweight="bold")
    if args.theylabel is not None:
        ylabel(args.theylabel, fontsize=args.fontscalefac * 16, fontweight="bold")

    if args.outputfile is None:
        show()
    else:
        savefig(args.outputfile, bbox_inches="tight", dpi=args.saveres)
