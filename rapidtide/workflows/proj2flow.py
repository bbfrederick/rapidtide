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

import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io


def _get_parser():
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="proj2flow",
        description="Convert phase projection movie to velocity map.",
        allow_abbrev=False,
    )
    parser.add_argument("inputfilename", type=str, help="The name of the input nifti file.")
    parser.add_argument("outputroot", type=str, help="The root name of the output nifti files.")

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Turn on debugging information.",
        default=False,
    )

    """parser.add_argument(
        "lowestfreq",
        type=float,
        help="The low passband frequency limit in Hz (set less than zero to disable HPF).",
    )
    parser.add_argument(
        "highestfreq",
        type=float,
        help="The high passband frequency limit in Hz (set less than zero to disable LPF)",
    )"""

    return parser


def proj2flow(args):
    # set default variable values
    displayplots = False

    # get the command line parameters
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    # get the input TR
    inputtr_fromfile, numinputtrs = tide_io.fmritimeinfo(args.inputfilename)
    print("input data: ", numinputtrs, " timepoints, tr = ", inputtr_fromfile)

    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    if input_hdr.get_xyzt_units()[1] == "msec":
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    Fs = 1.0 / tr
    print("tr from header =", tr, ", sample frequency is ", Fs)

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
    forwarddiffs = np.zeros((xsize - 2, ysize - 2, numslices - 2, 3, 3, 3), dtype="float32")
    backwarddiffs = np.zeros((xsize - 2, ysize - 2, numslices - 2, 3, 3, 3), dtype="float32")
    velocities = np.zeros((xsize, ysize, numslices, timepoints, 3), dtype="float32")
    averagevelocity = np.zeros((xsize, ysize, numslices, 3), dtype="float32")

    # cycle over all timepoints
    print("now cycling over all timepoints")
    for timepoint in range(timepoints):
        print(f"processing timepoint {timepoint}")
        # calculate forward differences
        for zoffset in [-1, 0, 1]:
            for yoffset in [-1, 0, 1]:
                for xoffset in [-1, 0, 1]:
                    if args.debug:
                        print([xoffset * xdim, yoffset * ydim, zoffset * slicethickness])
                    distance = np.linalg.norm(
                        np.array([xoffset * xdim, yoffset * ydim, zoffset * slicethickness])
                    )
                    # if distance == 0.0:
                    #    distance = 1.0
                    if args.debug:
                        print(f"{xoffset}, {yoffset}, {zoffset}")
                        print(f"\tdistance:{distance}")
                        print(
                            f"\t{input_data[1 + xoffset : xsize + xoffset - 1,1 + yoffset : ysize + yoffset - 1,1 + zoffset : numslices + zoffset - 1,(timepoint + 1) % timepoints,].shape}"
                        )
                        print(
                            f"\t{input_data[1: -1 ,1 : -1,1: -1 ,(timepoint + 1) % timepoints,].shape}"
                        )
                    forwarddiffs[:, :, :, xoffset + 1, yoffset + 1, zoffset + 1] = (
                        (
                            input_data[
                                1 + xoffset : xsize + xoffset - 1,
                                1 + yoffset : ysize + yoffset - 1,
                                1 + zoffset : numslices + zoffset - 1,
                                (timepoint + 1) % timepoints,
                            ]
                            - input_data[
                                1:-1,
                                1:-1,
                                1:-1,
                                timepoint,
                            ]
                        )
                    ) * distance
                    backwarddiffs[:, :, :, xoffset + 1, yoffset + 1, zoffset + 1] = (
                        (
                            input_data[
                                1:-1,
                                1:-1,
                                1:-1,
                                timepoint,
                            ]
                            - input_data[
                                1 + xoffset : xsize + xoffset - 1,
                                1 + yoffset : ysize + yoffset - 1,
                                1 + zoffset : numslices + zoffset - 1,
                                (timepoint - 1) % timepoints,
                            ]
                        )
                    ) * distance
                    if args.debug:
                        print(f"{np.min(forwarddiffs)=}, {np.max(forwarddiffs)=}")
                        print(f"{np.min(backwarddiffs)=}, {np.max(backwarddiffs)=}")
                        print(f"{forwarddiffs.dtype=}")
        velocities[1:-1, 1:-1, 1:-1, timepoint, 0] = np.sum(
            forwarddiffs[:, :, :, :, 1, 1], axis=3
        ) + np.sum(backwarddiffs[:, :, :, :, 1, 1], axis=3)
        velocities[1:-1, 1:-1, 1:-1, timepoint, 1] = np.sum(
            forwarddiffs[:, :, :, 1, :, 1], axis=3
        ) + np.sum(backwarddiffs[:, :, :, 1, :, 1], axis=3)
        velocities[1:-1, 1:-1, 1:-1, timepoint, 2] = np.sum(
            forwarddiffs[:, :, :, 1, 1, :], axis=3
        ) + np.sum(backwarddiffs[:, :, :, 1, 1, :], axis=3)

        # now do the ones with other numbers of time points
        output_hdr = copy.deepcopy(input_hdr)
        output_hdr.set_intent(2003)
        output_hdr["dim"][0] = 4
        output_hdr["dim"][4] = 3
        output_hdr["dim"][5] = 1
        output_hdr.datatype = 16
        if args.debug:
            print(f"{velocities.dtype=}")
        tide_io.savetonifti(
            velocities[:, :, :, timepoint, :].reshape((xsize, ysize, numslices, 3)),
            output_hdr,
            f"{args.outputroot}_{str(timepoint).zfill(2)}",
            debug=args.debug,
        )
        averagevelocity += velocities[:, :, :, timepoint, :].reshape((xsize, ysize, numslices, 3))
    tide_io.savetonifti(
        averagevelocity[:, :, :, :] / timepoints,
        output_hdr,
        f"{args.outputroot}_average",
        debug=args.debug,
    )
