#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2024-2025 Blaise Frederick
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
import rapidtide.qualitycheck as tide_quality


def _get_parser() -> Any:
    """
    Argument parser for runqualitycheck.

    This function creates and configures an argument parser for the runqualitycheck
    command-line tool. The parser handles both required and optional arguments needed
    to perform quality checks on rapidtide datasets.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional arguments
        for the runqualitycheck tool.

    Notes
    -----
    The argument parser is configured with:
    - Required input file root name
    - Optional gray matter mask specification
    - Optional white matter mask specification
    - Debug flag for additional output

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['dataset_root'])
    >>> print(args.inputfileroot)
    'dataset_root'
    """
    parser = argparse.ArgumentParser(
        prog="runqualitycheck",
        description=("Run a quality check on a rapidtide dataset."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputfileroot",
        type=str,
        help="The root of the rapidtide dataset name (without the underscore.)",
    )

    # add optional arguments
    parser.add_argument(
        "--graymaskspec",
        metavar="MASK[:VALSPEC]",
        type=str,
        help="The name of a gray matter mask that matches the input dataset. If VALSPEC is given, only voxels "
        "with integral values listed in VALSPEC are used.  If using an aparc+aseg file, set to APARC_GRAY.",
        default=None,
    )
    parser.add_argument(
        "--whitemaskspec",
        metavar="MASK[:VALSPEC]",
        type=str,
        help="The name of a white matter mask that matches the input dataset. If VALSPEC is given, only voxels "
        "with integral values listed in VALSPEC are used.  If using an aparc+aseg file, set to APARC_WHITE.",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Output additional debugging information."),
        default=False,
    )

    return parser


def runqualitycheck(args: Any) -> None:
    """
    Run quality check on input data and write results to JSON file.

    This function performs a quality check on the input data using the tide_quality
    module and writes the results to a JSON file with a standardized naming convention.

    Parameters
    ----------
    args : Any
        An object containing input arguments with the following attributes:
        - inputfileroot : str
            Root name of the input file(s)
        - graymaskspec : str, optional
            Specification for gray matter masking
        - whitemaskspec : str, optional
            Specification for white matter masking
        - debug : bool, optional
            Flag to enable debug mode

    Returns
    -------
    None
        This function does not return any value but writes results to a JSON file.

    Notes
    -----
    The output JSON file will be named as '{inputfileroot}_desc-qualitymetrics_info.json'
    where inputfileroot is the root name provided in the args object.

    Examples
    --------
    >>> class Args:
    ...     def __init__(self):
    ...         self.inputfileroot = "sub-01_task-rest"
    ...         self.graymaskspec = "gray_mask.nii.gz"
    ...         self.whitemaskspec = "white_mask.nii.gz"
    ...         self.debug = False
    ...
    >>> args = Args()
    >>> runqualitycheck(args)
    """
    resultsdict = tide_quality.qualitycheck(
        args.inputfileroot,
        graymaskspec=args.graymaskspec,
        whitemaskspec=args.whitemaskspec,
        debug=args.debug,
    )
    tide_io.writedicttojson(
        resultsdict,
        f"{args.inputfileroot}_desc-qualitymetrics_info.json",
    )
