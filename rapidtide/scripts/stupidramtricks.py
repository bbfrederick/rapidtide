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
import os

import numpy as np

import rapidtide.multiproc as tide_multiproc
import rapidtide.refinedelay as tide_refinedelay
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for stupidramtricks
    """
    parser = argparse.ArgumentParser(
        prog="stupidramtricks",
        description="Make shared memory blocks, do something with them.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "--numvoxels",
        dest="numvoxels",
        action="store",
        type=int,
        metavar="NUMVOXELS",
        help=("Set blocks to have NUMVOXELS voxels."),
        default=230000,
    )
    # Required arguments
    parser.add_argument(
        "--numtimepoints",
        dest="numtimepoints",
        action="store",
        type=int,
        metavar="NUMPOINTS",
        help=("Set blocks to have NUMPOINTS timespoints."),
        default=500,
    )

    parser.add_argument(
        "--nprocs",
        dest="nprocs",
        action="store",
        type=int,
        metavar="NPROCS",
        help=(
            "Use NPROCS worker processes for multiprocessing. "
            "Setting NPROCS to less than 1 sets the number of "
            "worker processes to n_cpus."
        ),
        default=1,
    )
    parser.add_argument(
        "--usesharedmem",
        dest="usesharedmem",
        action="store_true",
        help=("Use shared memory"),
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Print debug messages"),
        default=False,
    )

    return parser


def stupidramtricks(args):
    # get the pid of the parent process
    args.pid = os.getpid()

    args.internalprecision = "single"
    args.outputprecision = "double"

    print("Start")

    if args.nprocs < 1:
        args.nprocs = tide_multiproc.maxcpus()

    if args.internalprecision == "double":
        rt_floattype = "float64"
        rt_floatset = np.float64
    else:
        rt_floattype = "float32"
        rt_floatset = np.float32

    # set the output precision
    if args.outputprecision == "double":
        rt_outfloattype = "float64"
        rt_outfloatset = np.float64
    else:
        rt_outfloattype = "float32"
        rt_outfloatset = np.float32

    # select the voxels in the mask
    print("setting sizes")
    numvalidspatiallocs = args.numvoxels
    if args.debug:
        print(f"{numvalidspatiallocs=}")
    internalvalidspaceshape = numvalidspatiallocs
    derivaxissize = 2
    internalvalidspaceshapederivs = (
        internalvalidspaceshape,
        derivaxissize,
    )
    internalvalidfmrishape = (numvalidspatiallocs, args.numtimepoints)
    if args.debug:
        print(f"validvoxels shape = {numvalidspatiallocs}")
        print(f"internalvalidfmrishape shape = {internalvalidfmrishape}")

    fmridata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)

    if args.usesharedmem:
        if args.debug:
            print("allocating shared memory")
        # first move fmridata into shared memory
        fmridata, fmridata_shm = tide_util.numpy2shared(
            fmridata, rt_floatset, name=f"fmridata_{args.pid}"
        )
        sLFOfitmean, sLFOfitmean_shm = tide_util.allocshared(
            internalvalidspaceshape, rt_outfloatset
        )
        rvalue, rvalue_shm = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        r2value, r2value_shm = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        fitNorm, fitNorm_shm = tide_util.allocshared(internalvalidspaceshapederivs, rt_outfloatset)
        fitcoeff, fitcoeff_shm = tide_util.allocshared(
            internalvalidspaceshapederivs, rt_outfloatset
        )
        movingsignal, movingsignal_shm = tide_util.allocshared(
            internalvalidfmrishape, rt_outfloatset
        )
        lagtc, lagtc_shm = tide_util.allocshared(internalvalidfmrishape, rt_floatset)
        filtereddata, filtereddata_shm = tide_util.allocshared(
            internalvalidfmrishape, rt_outfloatset
        )

        location = "in shared memory"
    else:
        if args.debug:
            print("allocating memory")
        sLFOfitmean = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        rvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        r2value = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitNorm = np.zeros(internalvalidspaceshapederivs, dtype=rt_outfloattype)
        fitcoeff = np.zeros(internalvalidspaceshapederivs, dtype=rt_outfloattype)
        movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
        lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
        filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
        location = "locally"

    totalbytes = (
        fmridata.nbytes
        + sLFOfitmean.nbytes
        + rvalue.nbytes
        + r2value.nbytes
        + fitNorm.nbytes
        + fitcoeff.nbytes
        + movingsignal.nbytes
        + lagtc.nbytes
        + filtereddata.nbytes
    )

    thesize, theunit = tide_util.format_bytes(totalbytes)
    print(f"allocated {thesize:.3f} {theunit} {location}")

    """regressderivratios, regressrvalues = tide_refinedelay.getderivratios(
        fmri_data_valid,
        validvoxels,
        initial_fmri_x,
        lagtimes_valid,
        corrmask_valid,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm[:, :2],
        fitcoeff[:, :2],
        movingsignal,
        lagtc,
        filtereddata,
        LGR,
        TimingLGR,
        therunoptions,
        debug=args.debug,
    )"""

    # clean up shared memory
    if args.usesharedmem:
        tide_util.cleanup_shm(fmridata_shm)
        tide_util.cleanup_shm(sLFOfitmean_shm)
        tide_util.cleanup_shm(rvalue_shm)
        tide_util.cleanup_shm(r2value_shm)
        tide_util.cleanup_shm(fitNorm_shm)
        tide_util.cleanup_shm(fitcoeff_shm)
        tide_util.cleanup_shm(movingsignal_shm)
        tide_util.cleanup_shm(lagtc_shm)
        tide_util.cleanup_shm(filtereddata_shm)


def process_args(inputargs=None):
    """
    Compile arguments for stupidramtrics workflow.
    """
    args, argstowrite = pf.setargs(_get_parser, inputargs=inputargs)
    return args


def entrypoint():
    pf.generic_init(_get_parser, stupidramtricks)


if __name__ == "__main__":
    entrypoint()
