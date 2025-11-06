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
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import rapidtide.io as tide_io


def _get_parser() -> Any:
    """
    Argument parser for resamplenifti.

    Creates and configures an argument parser for the resamplenifti command-line tool
    that changes the TR (repetition time) in a NIFTI header.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional arguments
        for the resamplenifti tool.

    Notes
    -----
    The parser is configured with:
    - Required positional arguments: inputfile, outputfile, and outputtr
    - Optional debug flag for additional output
    - Program name set to "fixtr"
    - Description explaining the purpose of changing TR in NIFTI headers

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.nii', 'output', '2.0'])
    >>> print(args.inputfile)
    'input.nii'
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


def fixtr(args: Any) -> None:
    """
    Fix the temporal resolution (TR) of a NIfTI file.

    This function reads a NIfTI file and modifies its header to change the
    temporal resolution (TR) while preserving the original data. The output
    file is saved with the new TR value in the header.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - inputfile : str
            Path to the input NIfTI file
        - outputfile : str
            Path to the output NIfTI file
        - outputtr : float
            Desired output temporal resolution in seconds
        - debug : bool, optional
            If True, print debug information including input timepoints and TR

    Returns
    -------
    None
        This function does not return a value but saves the modified NIfTI file
        to the specified output path.

    Notes
    -----
    The function preserves the original data type and spatial dimensions of the
    input file. The temporal dimension (t) in the output header is updated based
    on the units specified in the input header. If the input header specifies
    millisecond units, the output TR is converted from seconds to milliseconds.

    Examples
    --------
    >>> class Args:
    ...     def __init__(self):
    ...         self.inputfile = "input.nii"
    ...         self.outputfile = "output.nii"
    ...         self.outputtr = 2.0
    ...         self.debug = False
    >>> args = Args()
    >>> fixtr(args)
    """
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
