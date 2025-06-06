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
import copy

import matplotlib.pyplot as plt
import numpy as np

import rapidtide.io as tide_io
import rapidtide.maskutil as tide_mask
import rapidtide.stats as tide_stats
import rapidtide.workflows.parser_funcs as pf

DEFAULT_LAGMIN = -10
DEFAULT_LAGMAX = 20
DEFAULT_PEAKTHRESH = 0.33
DEFAULT_HISTBINS = 151


def _get_parser():
    """
    Argument parser for adjust offset
    """
    parser = argparse.ArgumentParser(
        prog="adjustoffset",
        description="Adjust the offset of a rapidtide delay map.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputmap",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The name of the rapidtide maxtime map.",
    )
    parser.add_argument("outputroot", help="The root name for the output files.")

    maskopts = parser.add_argument_group("Masking options")
    maskopts.add_argument(
        "--includemask",
        dest="includespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Only use voxels that are also in file MASK in calculating the offset values "
            "(if VALSPEC is given, only voxels "
            "with integral values listed in VALSPEC are used). "
        ),
        default=None,
    )
    maskopts.add_argument(
        "--excludemask",
        dest="excludespec",
        metavar="MASK[:VALSPEC]",
        help=(
            "Do not use voxels that are also in file MASK in calculating the offset values "
            "(if VALSPEC is given, voxels "
            "with integral values listed in VALSPEC are excluded). "
        ),
        default=None,
    )
    maskopts.add_argument(
        "--extramask",
        dest="extramaskname",
        metavar="MASK",
        type=lambda x: pf.is_valid_file(parser, x),
        help=(
            "Additional mask to apply to select voxels for adjustment.  Zero voxels in this mask will be excluded.  "
            "If not specified, the corrfit_mask will be used."
        ),
        default=None,
    )

    parser.add_argument(
        "--searchrange",
        dest="searchrange",
        action=pf.IndicateSpecifiedAction,
        nargs=2,
        type=float,
        metavar=("LAGMIN", "LAGMAX"),
        help=(
            "Limit fit to a range of lags from LAGMIN to "
            f"LAGMAX.  Default is {DEFAULT_LAGMIN} to {DEFAULT_LAGMAX} seconds. "
        ),
        default=(DEFAULT_LAGMIN, DEFAULT_LAGMAX),
    )
    parser.add_argument(
        "--histbins",
        metavar="BINS",
        help=f"Number of bins in the entropy histogram (default is {DEFAULT_HISTBINS}).",
        type=int,
        default=DEFAULT_HISTBINS,
    )
    parser.add_argument(
        "--histonly",
        dest="histonly",
        action="store_true",
        help=("Only calculate offset histograms - do not perform adjustments."),
        default=False,
    )
    parser.add_argument(
        "--display",
        dest="display",
        action="store_true",
        help=("Show the delay histogram."),
        default=False,
    )
    parser.add_argument(
        "--pickleft",
        dest="pickleft",
        action="store_true",
        help=("Choose the leftmost peak of the histogram that exceeds the threshold."),
        default=False,
    )
    parser.add_argument(
        "--pickleftthresh",
        metavar="THRESH",
        help=f"Fraction of the maximum height that can be considered a peak.  Default is {DEFAULT_PEAKTHRESH}",
        type=float,
        default=DEFAULT_PEAKTHRESH,
    )
    parser.add_argument(
        "--setoffset",
        metavar="OFFSET",
        help=f"Directly set the offset value to OFFSET.  Overrides histogram.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--norefine",
        dest="refine",
        action="store_false",
        help=("Do not fit the histogram peak."),
        default=True,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Output debugging information."),
        default=False,
    )

    return parser


def adjustoffset(args):
    if args.debug:
        print(f"reading map file {args.inputmap}")
    (
        themap,
        themap_data,
        themap_hdr,
        themapdims,
        thetemplatesizes,
    ) = tide_io.readfromnifti(args.inputmap)
    nx, ny, nz, nummaps = tide_io.parseniftidims(themapdims)

    # process masks
    if args.includespec is not None:
        (
            includename,
            includevals,
        ) = tide_io.processnamespec(
            args.includespec, "Including voxels where ", "in offset calculation."
        )
    else:
        includename = None
        includevals = None
    if args.excludespec is not None:
        (
            excludename,
            excludevals,
        ) = tide_io.processnamespec(
            args.excludespec, "Excluding voxels where ", "from offset calculation."
        )
    else:
        excludename = None
        excludevals = None

    if args.extramaskname is None:
        args.extramaskname = (args.inputmap).replace("maxtime_map", "corrfit_mask")

    numspatiallocs = int(nx) * int(ny) * int(nz)
    includemask, excludemask, extramask = tide_mask.getmaskset(
        "anatomic",
        includename,
        includevals,
        excludename,
        excludevals,
        themap_hdr,
        numspatiallocs,
        extramask=args.extramaskname,
    )

    theflatmap = themap_data.reshape((numspatiallocs))
    theflatmask = theflatmap * 0 + 1
    if includemask is not None:
        theflatmask = theflatmask * includemask.reshape((numspatiallocs))
    if excludemask is not None:
        theflatmask = theflatmask * (1 - excludemask.reshape((numspatiallocs)))
    if extramask is not None:
        theflatextramask = extramask.reshape((numspatiallocs))
        theflatmask = theflatmask * theflatextramask

    # generate the mask
    themask_data = theflatmask.reshape((nx, ny, nz))
    maskmap = themask_data

    peakloc, peakheight, peakwidth = tide_stats.gethistprops(
        theflatmap[np.where(theflatmask > 0.0)],
        args.histbins,
        therange=args.searchrange,
        refine=args.refine,
        pickleft=args.pickleft,
        peakthresh=args.pickleftthresh,
    )
    print(f"{peakloc=}, {peakheight=}, {peakwidth=}")

    (
        thehist,
        peakheight2,
        peakloc2,
        peakwidth2,
        centerofmass2,
        peakpercentile2,
    ) = tide_stats.makehistogram(
        theflatmap[np.where(theflatmask > 0.0)],
        args.histbins,
        binsize=None,
        therange=args.searchrange,
        refine=args.refine,
        normalize=False,
    )
    # print(f"{peakloc=}, {peakheight=}, {peakwidth=}, {centerofmass=}")

    endtrim = 1
    if args.display:
        thestore = np.zeros((2, len(thehist[0])), dtype="float64")
        thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
        thestore[1, :] = thehist[0][-args.histbins :]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("histogram")
        plt.plot(thestore[0, : (-1 - endtrim)], thestore[1, : (-1 - endtrim)])
        plt.show()

    # save the maskmap as nifti
    themaskmaphdr = copy.deepcopy(themap_hdr)
    themaskmaphdr["dim"][0] = 3
    themaskmaphdr["dim"][4] = 1
    tide_io.savetonifti(maskmap, themaskmaphdr, args.outputroot + "_maskmap")

    if not args.histonly:
        # actually change the offset
        if args.setoffset is not None:
            offsetvalue = args.setoffset
        else:
            offsetvalue = peakloc
        theflatmap[np.where(theflatmask > 0.0)] += offsetvalue
        tide_io.savetonifti(
            theflatmap.reshape((nx, ny, nz)), themap_hdr, args.outputroot + "_adjustedmaxtime"
        )
