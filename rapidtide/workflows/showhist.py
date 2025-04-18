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

import numpy as np
from matplotlib.pyplot import bar, legend, plot, savefig, show, title, xlabel, ylabel

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats


def _get_parser():
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


def showhist(args):
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
