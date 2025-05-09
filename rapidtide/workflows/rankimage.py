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
from scipy.stats import rankdata

import rapidtide.io as tide_io


def _get_parser():
    """
    Argument parser for pixelcomp
    """
    parser = argparse.ArgumentParser(
        prog="rankimage",
        description=("Convert a 3D or 4D nifti image into a percentile map."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument("inputfilename", type=str, help="The name of the input image nifti file.")
    parser.add_argument("maskfilename", type=str, help="The name of the input mask nifti file.")
    parser.add_argument("outputroot", type=str, help="The root name of the output files.")

    # add optional arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Output additional debugging information."),
        default=False,
    )

    return parser


def imtopercentile(image, mask, debug=False):
    outmaparray = image * 0.0
    nativespaceshape = image.shape
    validvoxels = np.where(mask > 0)
    numvalidspatiallocs = np.shape(validvoxels[0])[0]
    input_data_valid = image[validvoxels] + 0.0

    if debug:
        print(
            f"Processing {numvalidspatiallocs} of {nativespaceshape[0] * nativespaceshape[1] * nativespaceshape[2]} voxels"
        )
        print(f"{np.shape(input_data_valid)=}")
    percentilescore = (
        100.0 * (rankdata(input_data_valid, method="dense") - 1) / (numvalidspatiallocs - 1)
    )
    outmaparray[validvoxels] = percentilescore[:]
    return outmaparray.reshape(nativespaceshape)


def rankimage(args):
    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    (
        mask_img,
        mask_data,
        mask_hdr,
        themaskdims,
        themasksizes,
    ) = tide_io.readfromnifti(args.maskfilename)

    if not tide_io.checkspacedimmatch(thedims, themaskdims):
        print("input image 1 dimensions do not match mask")
        exit()

    # select the valid voxels
    xsize, ysize, numslices, timepoints = tide_io.parseniftisizes(thedims)
    if timepoints > 1:
        is4d = True
    else:
        is4d = False

    if is4d:
        print("processing 4D nifti file")
        percentiles = input_data * 0.0
        for i in range(timepoints):
            percentiles[:, :, :, i] = imtopercentile(
                input_data[:, :, :, i], mask_data[:, :, :, i], debug=args.debug
            )
    else:
        print("processing 3D nifti file")
        percentiles = imtopercentile(input_data, mask_data, debug=args.debug)

    savename = args.outputroot
    bidsdict = {
        "RawSources": [args.inputfilename, args.maskfilename],
        "Units": "percentile",
    }
    tide_io.writedicttojson(bidsdict, f"{savename}.json")
    tide_io.savetonifti(percentiles, input_hdr, savename)
