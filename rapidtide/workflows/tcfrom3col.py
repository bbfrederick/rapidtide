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

import rapidtide.io as tide_io
import rapidtide.util as tide_util


def _get_parser():
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


def tcfrom3col(args):
    if args.debug:
        print(args)

    # now make the vector
    inputdata = tide_io.readvecs(args.infilename)
    timeaxis = np.arange(0.0, args.numpoints * args.timestep, args.timestep)
    outputdata = 0.0 * timeaxis
    outputdata = tide_util.maketcfrom3col(inputdata, timeaxis, outputdata, debug=args.debug)
    tide_io.writenpvecs(outputdata, args.outfilename)
