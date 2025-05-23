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

import nibabel as nib
import numpy as np
import pandas as pd

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for diffrois
    """
    parser = argparse.ArgumentParser(
        prog="diffrois",
        description="Create matrices showing the difference in values between ROIs in a CSV file.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "datafile",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The name of the CSV file containing the ROI data.  Assumes a 1 line header row.",
    )
    parser.add_argument("outputroot", help="The root name for the output files.")

    # Optional arguments
    parser.add_argument(
        "--keyfile",
        dest="keyfile",
        type=lambda x: pf.is_valid_file(parser, x),
        help=(
            "A file containing the region labels in the desired order. "
            "The axes of the output files will be in that order, rather than "
            "the order in which they occur in the source CSV file."
        ),
        default=None,
    )
    parser.add_argument(
        "--maxlines",
        dest="maxlines",
        action="store",
        type=lambda x: pf.is_int(parser, x),
        metavar="MAXLINES",
        help=("Only process the first MAXLINES lines of the CSV file."),
        default=None,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Output debugging information."),
        default=False,
    )

    return parser


def diffrois(args):
    df = pd.read_csv(args.datafile)

    theregions = np.array(df.columns[1:].values)
    if args.debug:
        print(theregions)

    thelabels = df["Region"].values
    numlabels = len(thelabels)
    if args.maxlines is not None:
        numlabels = np.min([args.maxlines, numlabels])
        print(f"only processing first {numlabels} lines of the input CSV.")
    if args.debug:
        print(thelabels)

    colkeys = []
    if args.keyfile is not None:
        with open(args.keyfile) as thefile:
            for region in thefile:
                colkeys.append(region.strip())
    else:
        for region in theregions:
            colkeys.append(str(region))

    if args.debug:
        print(f"{colkeys=}")

    numoutregions = len(colkeys)

    thediffs = np.zeros((numoutregions, numoutregions, 1, numlabels), dtype=float)
    thedemeaneddiffs = np.zeros((numoutregions, numoutregions, 1, numlabels), dtype=float)
    themask = np.ones((numoutregions, numoutregions, 1, numlabels), dtype=int)
    for z in range(numlabels):
        print(z)
        for i in range(numoutregions):
            for j in range(i, numoutregions):
                ival = df[colkeys[i]].values[z]
                jval = df[colkeys[j]].values[z]
                # print(z, i, j, ival, jval)
                if np.isnan(ival) or np.isnan(jval):
                    themask[i, j, 0, z] = 0
                    themask[j, i, 0, z] = 0
                else:
                    thediffs[i, j, 0, z] = float(ival) - float(jval)
                    thediffs[j, i, 0, z] = float(ival) - float(jval)

    outputaffine = np.eye(4)
    init_img = nib.Nifti1Image(thediffs, outputaffine)
    init_hdr = init_img.header
    tide_io.savetonifti(
        thediffs,
        init_hdr,
        args.outputroot + "_diffs",
    )
    tide_io.savetonifti(
        themask,
        init_hdr,
        args.outputroot + "_mask",
    )

    # make some summaries
    numvox = numoutregions * numoutregions
    thediffs_rs = thediffs.reshape((numvox, numlabels))
    themask_rs = themask.reshape((numvox, numlabels))
    themeandiffs_rs = thediffs_rs[:, 0] * 0.0
    thestddiffs_rs = thediffs_rs[:, 0] * 0.0
    for idx in range(numvox):
        inputvec = thediffs_rs[idx, :]
        inputmask = themask_rs[idx, :]
        themeandiffs_rs[idx] = np.mean(inputvec[np.where(inputmask > 0)])
        thestddiffs_rs[idx] = np.std(inputvec[np.where(inputmask > 0)])
    themeandiffs = themeandiffs_rs.reshape((numoutregions, numoutregions, 1, 1))
    thestddiffs = thestddiffs_rs.reshape((numoutregions, numoutregions, 1, 1))
    map_hdr = copy.deepcopy(init_hdr)
    map_hdr["dim"][4] = 1
    map_hdr["pixdim"][4] = 1.0

    tide_io.savetonifti(
        themeandiffs,
        map_hdr,
        args.outputroot + "_meandiffs",
    )
    tide_io.savetonifti(
        thestddiffs,
        map_hdr,
        args.outputroot + "_stddiffs",
    )

    # save a demeaned output
    thedemeaneddiffs = thediffs - themeandiffs
    thedemeaneddiffs[np.where(themask == 0)] = 0.0
    tide_io.savetonifti(
        thedemeaneddiffs,
        init_hdr,
        args.outputroot + "_demeaneddiffs",
    )
