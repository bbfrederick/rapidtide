#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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
import sys

import numpy as np
from tqdm import tqdm

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="histnifti",
        description="Generates a histogram of the values in a NIFTI file.",
        allow_abbrev=False,
    )
    parser.add_argument("inputfile", help="The name of the input NIFTI file.")
    parser.add_argument("outputroot", help="The root of the output file names.")
    parser.add_argument(
        "--histlen",
        dest="histlen",
        type=int,
        metavar="LEN",
        help="Set histogram length to LEN (default is to set automatically).",
        default=None,
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
        "--transform",
        dest="transform",
        action="store_true",
        help=("Replace data value with it's percentile score."),
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
        "--maskfile",
        dest="maskfile",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="MASK",
        help="Only process voxels within the 3D mask MASK.",
        default=None,
    )
    parser.add_argument(
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Do not display histogram."),
        default=True,
    )
    return parser


def histnifti(args):
    # set default variable values
    thepercentiles = [0.2, 0.25, 0.5, 0.75, 0.98]
    thepvalnames = []
    for thispercentile in thepercentiles:
        thepvalnames.append(str(1.0 - thispercentile).replace(".", "p"))

    # load the data
    print("loading data...")
    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfile)
    print("done")
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    if timepoints > 1:
        is4D = True
    else:
        is4D = False

    if args.maskfile is not None:
        print("loading mask...")
        (
            mask_img,
            mask_data,
            mask_hdr,
            themaskdims,
            themasksizes,
        ) = tide_io.readfromnifti(args.maskfile)
        if not tide_io.checkspacematch(mask_hdr, input_hdr):
            print("Dimensions of " + args.maskfile + " mask do not match the input data - exiting")
            sys.exit()
    else:
        print("generating mask...")
        mask_data = np.ones((xsize, ysize, numslices), dtype=np.float64)
    print("done")

    print("finding valid voxels...")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    maskasmatrix = mask_data.reshape((numspatiallocs))
    validvoxels = np.where(maskasmatrix > 0)[0]
    numvalidvoxels = len(validvoxels)
    print("done")

    print("reshaping matrices...")
    if is4D:
        dataasmatrix = input_data.reshape((numspatiallocs, timepoints))
        validdata = dataasmatrix[validvoxels, :]
    else:
        dataasmatrix = input_data.reshape((numspatiallocs))
        validdata = dataasmatrix[validvoxels]
    print("dataasmatrix, validdata shapes:", dataasmatrix.shape, validdata.shape)
    print("done", flush=True)

    # set the histogram range
    print("setting histogram range...", flush=True)
    if args.robustrange:
        histmin, histmax = tide_stats.getfracvals(validdata.flatten(), [0.02, 0.98])
    else:
        if args.minval is None:
            histmin = np.min(validdata)
        else:
            histmin = args.minval
        if args.maxval is None:
            histmax = np.max(validdata)
        else:
            histmax = args.maxval
    therange = (histmin, histmax)
    print("the range is ", therange, flush=True)

    if is4D:
        if args.histlen is None:
            thehistlen = 2 * int(np.floor(np.sqrt(timepoints))) + 1
            print(f"histogram length set to {thehistlen}", flush=True)
        else:
            thehistlen = args.histlen

        print("allocating arrays", flush=True)
        outputhists = np.zeros((numvalidvoxels, thehistlen), dtype="float")
        pcts_data = np.zeros((numvalidvoxels, len(thepercentiles)), dtype="float")

        # calculate and save the sorted lists
        print("calculating sorted arrays", flush=True)
        sorteddata = np.sort(validdata, axis=1)
        outmatrix = np.zeros((numspatiallocs, timepoints), dtype="float")
        outmatrix[validvoxels, :] = sorteddata
        outputimg = outmatrix.reshape((xsize, ysize, numslices, timepoints))
        theheader = copy.deepcopy(input_hdr)
        tide_io.savetonifti(
            outputimg,
            theheader,
            args.outputroot + "_sorted",
        )

        # now pull out and save the percentiles
        print("calculating percentiles", flush=True)
        if args.nozero:
            for spatialloc in tqdm(
                range(0, numvalidvoxels),
                desc="Voxel",
                unit="voxels",
            ):
                pcts_data[spatialloc, :] = tide_stats.getfracvals(
                    sorteddata[spatialloc, :], thepercentiles
                )
            print()
        else:
            for idx, thispercentile in enumerate(thepercentiles):
                pctindex = int(np.round(timepoints * thispercentile, 0))
                print(f"percentile {thispercentile} at index {pctindex}")
                pcts_data[:, idx] = 1.0 * sorteddata[:, pctindex]
        del sorteddata
        outmatrix = np.zeros((numspatiallocs, len(thepercentiles)), dtype="float")
        outmatrix[validvoxels, :] = pcts_data
        outputimg = outmatrix.reshape((xsize, ysize, numslices, len(thepercentiles)))
        theheader = copy.deepcopy(input_hdr)
        theheader["dim"][4] = len(thepercentiles)
        theheader["pixdim"][4] = 1.0 / (len(thepercentiles) - 1)
        theheader["toffset"] = 0.0
        tide_io.savetonifti(
            outputimg,
            theheader,
            args.outputroot + "_pcts",
        )
        del pcts_data

        # cycle over all voxels
        print("calculating histograms", flush=True)
        for spatialloc in tqdm(
            range(0, numvalidvoxels),
            desc="Voxel",
            unit="voxels",
        ):
            if args.nozero:
                inputarray = validdata[
                    spatialloc, np.where(np.fabs(validdata[spatialloc, :]) > args.nozerothresh)[0]
                ]
            else:
                inputarray = validdata[spatialloc, :]
            outputhists[spatialloc, :], bins = np.histogram(
                inputarray, bins=thehistlen, range=(histmin, histmax)
            )
            # if args.normhist:
            #    totalval = np.sum(outputhists[spatialloc, :])
            #    if totalval != 0.0:
            #        outputhists[spatialloc, :] /= totalval
        print()

        # save the histogram data
        outmatrix = np.zeros((numspatiallocs, thehistlen), dtype="float")
        outmatrix[validvoxels, :] = outputhists
        outputimg = outmatrix.reshape((xsize, ysize, numslices, thehistlen))
        theheader = copy.deepcopy(input_hdr)
        theheader["dim"][4] = thehistlen
        theheader["pixdim"][4] = bins[1] - bins[0]
        theheader["toffset"] = bins[0]
        tide_io.savetonifti(
            outputimg,
            theheader,
            args.outputroot + "_hists",
        )

    else:
        if args.histlen is None:
            thehistlen = int(np.floor(np.sqrt(numvalidvoxels))) + 1
            print(f"histogram length set to {thehistlen}")
        else:
            thehistlen = args.histlen
        dataasmatrix = input_data.reshape((numspatiallocs))
        if args.nozero:
            validdata = dataasmatrix[
                np.where(np.fabs(dataasmatrix[validvoxels]) >= args.nozerothresh)[0]
            ]
        else:
            validdata = dataasmatrix[validvoxels]
        if args.transform:
            thehist, bins = np.histogram(validdata, bins=thehistlen, range=(histmin, histmax))
            npbins = np.asarray(bins[:-1], dtype=float) + (bins[1] - bins[0]) / 2.0
            transformeddataasmatrix = dataasmatrix * 0.0
            for thevoxel in validvoxels:
                transformeddataasmatrix[thevoxel] = (
                    100.0 * tide_util.valtoindex(npbins, dataasmatrix[thevoxel]) / len(npbins)
                )
            outputimg = transformeddataasmatrix.reshape((xsize, ysize, numslices))
            theheader = copy.deepcopy(input_hdr)
            theheader["dim"][0] = 3
            theheader["dim"][4] = 1
            theheader["pixdim"][4] = bins[1] - bins[0]
            theheader["toffset"] = bins[0]
            tide_io.savetonifti(
                outputimg,
                theheader,
                args.outputroot + "_transformed",
            )
        tide_stats.makeandsavehistogram(
            validdata,
            thehistlen,
            0,
            args.outputroot + "_hist",
            displaytitle="Value histogram",
            displayplots=args.display,
            normalize=args.normhist,
            refine=False,
        )
