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

import matplotlib as mpl
import numpy as np
from numpy.polynomial import Polynomial

import rapidtide.io as tide_io

mpl.use("Agg")
import matplotlib.pyplot as plt


def _get_parser():
    """
    Argument parser for pixelcomp
    """
    parser = argparse.ArgumentParser(
        prog="pixelcomp",
        description=("Compare two nifti files, voxel by voxel, in a contour plot"),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputfilename1", type=str, help="The name of the first input image nifti file."
    )
    parser.add_argument(
        "maskfilename1", type=str, help="The name of the first input mask nifti file."
    )
    parser.add_argument(
        "inputfilename2", type=str, help="The name of the second input image nifti file."
    )
    parser.add_argument(
        "maskfilename2", type=str, help="The name of the second input mask nifti file."
    )
    parser.add_argument("outputroot", type=str, help="The root name of the output files.")

    # add optional arguments
    parser.add_argument(
        "--scatter",
        action="store_true",
        help=("Do a scatter plot instead of a contour plot."),
        default=False,
    )
    parser.add_argument(
        "--fitonly",
        action="store_true",
        help=("Perform fit only - do not generate graph."),
        default=False,
    )
    parser.add_argument(
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Save graphs to file only - do not display."),
        default=True,
    )
    parser.add_argument(
        "--fitorder",
        action="store",
        type=int,
        metavar="ORDER",
        help=("Order of line fit - default is 1 (linear)."),
        default=1,
    )
    parser.add_argument(
        "--usex",
        dest="usex",
        action="store_true",
        help="Use x instead of (y + x)/2 in Bland-Altman plot.",
        default=False,
    )
    parser.add_argument(
        "--histbins",
        action="store",
        type=int,
        metavar="NUM",
        help=("Number of bins per dimension for the contour plot -Default is 51."),
        default=51,
    )
    return parser


def bland_altman_plot(data1, data2, usex=False, *args, **kwargs):
    # data1 is X, data2 is Y
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    if usex:
        mean = np.mean(data1)
    else:
        mean = np.mean([data1, data2], axis=0)
    diff = data2 - data1  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color="gray", linestyle="--")
    plt.axhline(md + 2 * sd, color="gray", linestyle="--")
    plt.axhline(md - 2 * sd, color="gray", linestyle="--")


def pairdata(input1_data, input2_data, totalmask):
    nonzeropoints = np.where(totalmask > 0)
    pairlist = []
    for i in range(0, len(nonzeropoints[0])):
        pairlist.append(
            [
                input1_data[nonzeropoints[0][i], nonzeropoints[1][i], nonzeropoints[2][i]],
                input2_data[nonzeropoints[0][i], nonzeropoints[1][i], nonzeropoints[2][i]],
            ]
        )

    return np.asarray(pairlist)


def pixelcomp(args):
    if args.display:
        mpl.use("TkAgg")

    input1_img, input1_data, input1_hdr, thedims1, thesizes1 = tide_io.readfromnifti(
        args.inputfilename1
    )
    (
        mask1_img,
        mask1_data,
        mask1_hdr,
        themaskdims1,
        themasksizes1,
    ) = tide_io.readfromnifti(args.maskfilename1)

    if not tide_io.checkspacedimmatch(thedims1, themaskdims1):
        print("input image 1 dimensions do not match mask")
        exit()

    input2_img, input2_data, input2_hdr, thedims2, thesizes2 = tide_io.readfromnifti(
        args.inputfilename2
    )
    (
        mask2_img,
        mask2_data,
        mask2_hdr,
        themaskdims2,
        themasksizes2,
    ) = tide_io.readfromnifti(args.maskfilename2)

    if not tide_io.checkspacedimmatch(thedims2, themaskdims2):
        print("input image 2 dimensions do not match mask")
        exit()

    if not tide_io.checkspacedimmatch(thedims1, thedims2):
        print("input images 1 and 2 dimensions do not match")
        exit()

    totalmask = mask1_data * mask2_data
    thearray = pairdata(input1_data, input2_data, totalmask)

    plt.figure()
    if args.scatter:
        plt.plot(thearray[:, 0], thearray[:, 1], "k.")
        theplotname = args.outputroot + "_scatterplot.png"
    else:
        # construct a 2d histogram
        H, xedges, yedges = np.histogram2d(
            thearray[:, 0], thearray[:, 1], bins=args.histbins, normed=True
        )
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
        plt.contour(H, extent=extent)
        theplotname = args.outputroot + "_contourplot.png"
    if args.display:
        plt.show()
    else:
        plt.savefig(theplotname, bbox_inches="tight")

    # now fit the line
    try:
        thecoffs = (
            Polynomial.fit(thearray[:, 0], thearray[:, 1], args.fitorder).convert().coef[::-1]
        )
    except np.RankWarning:
        thecoffs = [0.0, 0.0]
    print("thecoffs=", thecoffs)
    with open(f"{args.outputroot}_order_{args.fitorder}_fit", "w") as file:
        file.writelines(str(thecoffs))

    if not args.fitonly:
        with open(args.outputroot, "w") as file:
            for pair in range(thearray.shape[0]):
                file.writelines(str(thearray[pair, 0]) + "\t" + str(thearray[pair, 1]) + "\n")
        plt.figure()
        bland_altman_plot(thearray[:, 0], thearray[:, 1])
        plt.title("Bland-Altman Plot")
        if args.display:
            plt.show()
        else:
            plt.savefig(args.outputroot + "_blandaltman.png", bbox_inches="tight")
