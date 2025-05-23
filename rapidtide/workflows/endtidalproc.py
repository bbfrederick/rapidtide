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
import bisect
import sys

import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import invert_float, is_float


def phase(mcv):
    return np.arctan2(mcv.imag, mcv.real)


def _get_parser():
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="endtidalproc",
        description="Process a gas trace to generate the endtidal waveform.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "infilename",
        help="a text file containing a single gas trace, one timepoint per line",
    )
    parser.add_argument("outfilename", help="a text file for the interpolated data")
    parser.add_argument(
        "--isoxygen",
        dest="isoxygen",
        action="store_true",
        help="Assume the trace is oxygen, fits the bottom of the waveform, not the top.",
        default=False,
    )
    samp_group = parser.add_mutually_exclusive_group()
    samp_group.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        type=lambda x: is_float(parser, x),
        metavar="FREQ",
        help=("The sample rate of the input data is FREQ Hz (default is 1Hz)."),
        default=None,
    )
    samp_group.add_argument(
        "--sampletime",
        dest="samplerate",
        action="store",
        type=lambda x: invert_float(parser, x),
        metavar="TSTEP",
        help=("The sample rate of the input data is 1/TSTEP Hz (default is 1Hz)."),
        default=None,
    )
    parser.add_argument(
        "--starttime",
        dest="thestarttime",
        metavar="START",
        type=float,
        help="Start plot at START seconds.",
        default=-1000000.0,
    )
    parser.add_argument(
        "--endtime",
        dest="theendtime",
        metavar="END",
        type=float,
        help="Finish plot at END seconds.",
        default=1000000.0,
    )
    parser.add_argument(
        "--thresh",
        dest="thresh",
        metavar="PCT",
        type=float,
        help="Amount of fall (or rise) needed, in percent, to recognize a peak (or through).",
        default=1.0,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Print additional internal information.",
        default=False,
    )

    return parser


def process_args(args):
    if args.samplerate is None:
        args.samplerate = 1.0

    if args.debug:
        print()
        print("args:")
        print(args)

    return args


def endtidalproc():
    args = process_args(_get_parser().parse_args(sys.argv[1:]))

    if args.isoxygen:
        print("Fitting trace as oxygen")
    else:
        print("Fitting trace as CO2")

    # check range
    if args.thestarttime >= args.theendtime:
        print("endtime must be greater then starttime;")
        sys.exit()

    # read in the data
    yvec = tide_io.readvec(args.infilename)
    xvec = np.arange(0.0, len(yvec), 1.0) / args.samplerate

    thestartpoint = np.max([0, bisect.bisect_right(xvec, args.thestarttime)])
    theendpoint = np.min([len(xvec) - 1, bisect.bisect_left(xvec, args.theendtime)])
    args.thestarttime = xvec[thestartpoint]
    args.theendtime = xvec[theendpoint]

    # set parameters - maxtime is the longest to look ahead for a peak (or through) in seconds
    # lookahead should be '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    maxtime = 1.0
    f = 2.0
    lookaheadval = int((args.samplerate * maxtime) / f)
    maxpeaks, minpeaks = tide_fit.peakdetect(yvec, lookahead=lookaheadval, delta=args.thethresh)

    if args.isoxygen:
        peaklist = minpeaks
    else:
        peaklist = maxpeaks

    peakinterp = 0.0 * yvec
    curpos = 0
    curval = peaklist[0][1]
    for thepeak in peaklist:
        slope = (thepeak[1] - curval) / (thepeak[0] - curpos)
        for theindex in range(curpos, thepeak[0]):
            peakinterp[theindex] = curval + slope * (theindex - curpos)
        curpos = thepeak[0] + 0
        curval = thepeak[1] + 0.0
    if curpos < len(peakinterp):
        peakinterp[curpos:] = curval

    tide_io.writevec(peakinterp, args.outfilename)


if __name__ == "__main__":
    endtidalproc()
