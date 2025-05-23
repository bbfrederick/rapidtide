#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2020-2025 Blaise Frederick
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

import rapidtide.io as tide_io
from rapidtide.RapidtideDataset import RapidtideDataset


def _get_parser():
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="synthASL",
        description="Use rapidtide output to predict ASL image.",
        allow_abbrev=False,
    )
    parser.add_argument("dataset", type=str, help="The name of the rapidtide dataset.")
    parser.add_argument("outputfilename", type=str, help="The name of the output nifti file.")

    # optional parameters
    parser.add_argument(
        "--tagoffset",
        dest="tagoffset",
        metavar="SECS",
        type=float,
        help="The assumed time of tagging, relative to the peak of the lag histogram (default is 2.945).",
        default=2.945,
    )
    parser.add_argument(
        "--pld",
        dest="pld",
        metavar="SECS",
        type=float,
        help="The postlabel delay (default is 1.8).",
        default=1.8,
    )
    parser.add_argument(
        "--labelduration",
        dest="labelduration",
        metavar="SECS",
        type=float,
        help="The length of the labelling period (default is 1.8).",
        default=1.8,
    )
    parser.add_argument(
        "--bloodT1",
        dest="bloodT1",
        metavar="SECS",
        type=float,
        help="The T1 of blood at this field strength (default is 1.841, for 3T).",
        default=1.841,
    )
    return parser


def calcASL(lags, strengths, widths, mask, tagoffset=2.945, pld=1.8, TI=1.8, bloodT1=1.841):
    theaslimage = lags * 0.0

    # convert rapidtide delays to time from tagging, and only keep positive delays after pld
    offsets = lags + tagoffset
    calcmask = mask * np.where(offsets < pld, 0.0, 1.0)

    for imtime in np.linspace(pld, pld + TI, num=50, endpoint=True):
        tagdecayfac = np.exp(-(offsets + imtime) / bloodT1) * calcmask
        oxyfac = strengths * 0.0 + 1.0
        cbvfac = np.fabs(strengths) * oxyfac
        theaslimage += tagdecayfac * cbvfac
    return theaslimage, tagdecayfac, oxyfac, cbvfac, calcmask, offsets


def synthASL(args):
    # get the command line parameters
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    thedataset = RapidtideDataset("main", args.dataset, init_LUT=False)

    themask = thedataset.overlays["lagmask"].data
    thelags = thedataset.overlays["lagtimes"].data
    thewidths = thedataset.overlays["lagsigma"].data
    thestrengths = thedataset.overlays["lagstrengths"].data

    print(
        "bloodT1, tagoffset, pld, and inversion time:",
        args.bloodT1,
        args.tagoffset,
        args.pld,
        args.labelduration,
    )
    theaslimage, tagdecayfac, oxyfac, cbvfac, calcmask, offsets = calcASL(
        thelags,
        thestrengths,
        thewidths,
        themask,
        tagoffset=args.tagoffset,
        pld=args.pld,
        TI=args.labelduration,
        bloodT1=args.bloodT1,
    )

    tide_io.savetonifti(
        theaslimage,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_ASL",
    )
    tide_io.savetonifti(
        tagdecayfac,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_tagdecayfac",
    )
    tide_io.savetonifti(
        oxyfac,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_oxyfac",
    )
    tide_io.savetonifti(
        cbvfac,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_cbvfac",
    )
    tide_io.savetonifti(
        calcmask,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_calcmask",
    )
    tide_io.savetonifti(
        offsets,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_offsets",
    )
    tide_io.savetonifti(
        themask,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_lagmask",
    )
    tide_io.savetonifti(
        thelags,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_lagtimes",
    )
    tide_io.savetonifti(
        thestrengths,
        thedataset.overlays["lagstrengths"].header,
        args.outputfilename + "_lagstrengths",
    )
