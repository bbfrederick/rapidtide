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

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io


def _get_parser():
    """
    Argument parser for filtnifti

    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="filtnifti",
        description="Temporally filters a NIFTI file.",
        allow_abbrev=False,
    )
    parser.add_argument("inputfilename", type=str, help="The name of the input nifti file.")
    parser.add_argument("outputfilename", type=str, help="The name of the output nifti file.")
    parser.add_argument(
        "lowestfreq",
        type=float,
        help="The low passband frequency limit in Hz (set less than zero to disable HPF).",
    )
    parser.add_argument(
        "highestfreq",
        type=float,
        help="The high passband frequency limit in Hz (set less than zero to disable LPF)",
    )
    return parser


def filtnifti(args):
    # get the input TR
    inputtr_fromfile, numinputtrs = tide_io.fmritimeinfo(args.inputfilename)
    print("input data: ", numinputtrs, " timepoints, tr = ", inputtr_fromfile)

    # sanity check the filter frequencies
    ftype = "arb"
    if args.lowestfreq < 0.0:
        print("disabling highpass filter")
        args.lowestfreq = 0.0

    if args.highestfreq < 0.0:
        print("disabling lowpass filter")

    if ftype == "bandpass":
        print("passing frequencies between ", args.lowestfreq, " and ", args.highestfreq)

    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    if input_hdr.get_xyzt_units()[1] == "msec":
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    Fs = 1.0 / tr
    print("tr from header =", tr, ", sample frequency is ", Fs)

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    filteredtcs = np.zeros((xsize, ysize, numslices, timepoints), dtype="float")

    # cycle over all voxels
    print("now cycling over all voxels")
    theprefilter = tide_filt.NoncausalFilter()
    theprefilter.settype("arb")
    theprefilter.setfreqs(args.lowestfreq, args.lowestfreq, args.highestfreq, args.highestfreq)

    for zloc in range(numslices):
        print("processing slice ", zloc)
        for yloc in range(ysize):
            for xloc in range(xsize):
                filteredtcs[xloc, yloc, zloc, :] = theprefilter.apply(
                    Fs, input_data[xloc, yloc, zloc, :]
                )

    # now do the ones with other numbers of time points
    tide_io.savetonifti(filteredtcs, input_hdr, args.outputfilename)
