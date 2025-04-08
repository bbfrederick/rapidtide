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

import matplotlib.pyplot as plt
import numpy as np

import rapidtide.io as tide_io
import rapidtide.resample as tide_resample
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for resamp1tc
    """
    parser = argparse.ArgumentParser(
        prog="resampletc",
        description=("Resample a timeseries file"),
        allow_abbrev=False,
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "inputfile")
    parser.add_argument(
        "insamplerate",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        help=("Output sample rate in Hz."),
    )
    pf.addreqoutputtextfile(parser, "outputfile")

    parser.add_argument(
        "outsamplerate",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        help=("Output sample rate in Hz."),
    )

    # add optional arguments
    parser.add_argument(
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Do not display data."),
        default=True,
    )

    parser.add_argument(
        "--noantialias",
        dest="antialias",
        action="store_false",
        help=("Enable additional debugging output."),
        default=True,
    )

    """parser.add_argument(
        "--outputstarttime",
        dest="starttime",
        type=lambda x: pf.is_float(parser, x),
        help=("Start time in seconds of the output.  If not set, go to the beginning of data (default)"),
        default=None,
    )

    parser.add_argument(
        "--outputendtime",
        dest="endtime",
        type=lambda x: pf.is_float(parser, x),
        help=("End time in seconds - if not set, go to the end of data (default)."),
        default=None,
    )"""

    # Miscellaneous options

    return parser


def resampletc(args):
    intimestep = 1.0 / args.insamplerate
    outtimestep = 1.0 / args.outsamplerate
    (
        insamplerate,
        instarttime,
        incolumns,
        inputdata,
        compressed,
        filetype,
    ) = tide_io.readvectorsfromtextfile(args.inputfile, onecol=True)

    if insamplerate is not None:
        if insamplerate != args.insamplerate:
            print(
                f"warning: specified sampling rate {insamplerate} does not match input file {args.insamplerate}"
            )
    if instarttime is None:
        instarttime = 0.0

    outputdata = tide_resample.arbresample(
        inputdata, args.insamplerate, args.outsamplerate, decimate=False, antialias=args.antialias
    )
    in_t = (
        intimestep * np.linspace(0.0, 1.0 * len(inputdata), len(inputdata), endpoint=True)
        + instarttime
    )
    """if args.starttime is None:
        outstarttime = in_t[0]
    else:
        outstarttime = args.starttime"""
    out_t = outtimestep * np.linspace(0.0, len(outputdata), len(outputdata), endpoint=True)
    """if args.endtime is None:
        endtime = in_t[-1]
    else:
        endtime = args.endtime"""
    if len(out_t) < len(outputdata):
        outputdata = outputdata[0 : len(out_t)]

    tide_io.writevec(outputdata, args.outputfile)
    if args.display:
        plt.plot(in_t, inputdata, "k")
        plt.plot(out_t, outputdata, "r")
        plt.legend(("original signal", "resampled"))
        plt.show()
