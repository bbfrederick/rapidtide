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
from scipy.stats import linregress

import rapidtide.io as tide_io


def _get_parser():
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
    data1,
    data2,
    usex=False,
    identifier=None,
    fontscalefac=1.0,
    xrange=None,
    yrange=None,
    doannotate=True,
    debug=False,
    *args,
    **kwargs,
):
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


def stringtorange(thestring):
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


def showxy(args):
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
