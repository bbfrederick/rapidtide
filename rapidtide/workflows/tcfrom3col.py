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

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io
import rapidtide.util as tide_util


def _get_parser() -> Any:
    """
    Create and configure argument parser for command line interface.

    This function initializes an ArgumentParser object with specific parameters
    required for processing three-column data files and generating time course output.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional arguments

    Notes
    -----
    The parser expects exactly four positional arguments followed by an optional
    debug flag. The function is designed for use with the tcfrom3col program.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.txt', '0.1', '100', 'output.txt'])
    >>> print(args.infilename)
    'input.txt'
    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="tcfrom3col",
        description="Plots the data in text files.",
        allow_abbrev=False,
    )
    parser.add_argument("infilename", help="the name of the input three column file")
    parser.add_argument(
        "timestep",
        type=float,
        help="the time step of the output time course in seconds",
    )
    parser.add_argument("numpoints", type=int, help="the number of output time points")
    parser.add_argument("outfilename", help="the name of the output time course file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="turn on additional debugging output",
        default=False,
    )
    return parser


def tcfrom3col(args: Any) -> None:
    """
    Convert three-column data to tidal constituent data.

    This function reads three-column input data, processes it to extract tidal constituents,
    and writes the results to an output file. The input data is expected to contain time,
    latitude, and longitude columns, and the output contains tidal constituent information.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - infilename : str
            Path to the input file containing three-column data
        - outfilename : str
            Path to the output file for writing tidal constituent data
        - numpoints : int
            Number of data points in the time series
        - timestep : float
            Time step between data points
        - debug : bool
            Flag to enable debug printing

    Returns
    -------
    None
        This function does not return a value but writes output to a file.

    Notes
    -----
    The function uses `tide_io.readvecs` to read input data and `tide_util.maketcfrom3col`
    to perform the tidal constituent calculation. The time axis is generated using
    `np.arange` with the specified number of points and time step.

    Examples
    --------
    >>> class Args:
    ...     def __init__(self):
    ...         self.infilename = 'input.dat'
    ...         self.outfilename = 'output.dat'
    ...         self.numpoints = 1000
    ...         self.timestep = 3600.0
    ...         self.debug = False
    ...
    >>> args = Args()
    >>> tcfrom3col(args)
    """
    if args.debug:
        print(args)

    # now make the vector
    inputdata = tide_io.readvecs(args.infilename)
    timeaxis = np.arange(0.0, args.numpoints * args.timestep, args.timestep)
    outputdata = 0.0 * timeaxis
    outputdata = tide_util.maketcfrom3col(inputdata, timeaxis, outputdata, debug=args.debug)
    tide_io.writenpvecs(outputdata, args.outfilename)
