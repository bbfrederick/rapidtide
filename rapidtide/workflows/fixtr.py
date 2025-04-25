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

import rapidtide.io as tide_io


def _get_parser():
    """
    Argument parser for resamplenifti
    """
    parser = argparse.ArgumentParser(
        prog="fixtr",
        description="Change the TR in a NIFTI header.",
        allow_abbrev=False,
    )

    parser.add_argument("inputfile", help="The name of the input nifti file, including extension")
    parser.add_argument("outputfile", help="The name of the output nifti file, without extension")
    parser.add_argument("outputtr", type=float, help="The new TR, in seconds")

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )
    return parser


def fixtr(args):
    # get the input TR
    inputtr, numinputtrs = tide_io.fmritimeinfo(args.inputfile)
    if args.debug:
        print("input data: ", numinputtrs, " timepoints, tr = ", inputtr)

    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfile)

    thedtype = input_hdr.get_data_dtype()

    output_hdr = input_hdr.copy()
    if input_hdr.get_xyzt_units()[1] == "msec":
        output_hdr["pixdim"][4] = args.outputtr / 1000.0
    else:
        output_hdr["pixdim"][4] = args.outputtr

    tide_io.savetonifti(input_data.astype(thedtype), output_hdr, args.outputfile)
