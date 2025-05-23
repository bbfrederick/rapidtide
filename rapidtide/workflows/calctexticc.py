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
import platform
import sys
import time

import numpy as np
from scipy.linalg import pinv

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
from rapidtide.workflows.parser_funcs import is_valid_file


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="calcicc",
        description="Calculate per-column ICC(3,1) on a set of text files.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "datafile",
        type=str,
        help=(
            "A comma separated list of 1 or more 2 dimensional text files.  Each column is a distinct quantity.  Each "
            "line in the file is a measurement on a subject."
        ),
    )
    parser.add_argument(
        "measurementlist",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "A multicolumn value file of integers specifying how to group measurements.  "
            "Each row is a subject, each column specifies the line numbers of the repeated measurement.  "
            "Subject and measurement numbering starts at 0."
        ),
    )
    parser.add_argument(
        "outputroot",
        type=str,
        help=(
            "The root name for the output text files.  Each distinct quantity "
            "will be in a separate row corresponding to the input file(s) columns."
        ),
    )

    # Optional arguments
    parser.add_argument(
        "--demedian",
        dest="demedian",
        action="store_true",
        help=("Subtract the median value from each map prior to ICC calculation."),
        default=False,
    )
    parser.add_argument(
        "--demean",
        dest="demean",
        action="store_true",
        help=("Subtract the mean value from each map prior to ICC calculation."),
        default=False,
    )
    parser.add_argument(
        "--nocache",
        dest="nocache",
        action="store_true",
        help=(
            "Disable caching for the ICC calculation.  This is a terrible idea.  Don't do this."
        ),
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Print out additional debugging information."),
        default=False,
    )
    parser.add_argument(
        "--deepdebug",
        dest="deepdebug",
        action="store_true",
        help=("Print out insane additional debugging information."),
        default=False,
    )
    return parser


def parsetextmeasurementlist(measlist, numfiles, debug=False):
    # how do we get the number of subjects?
    nummeas, numsubjs = measlist.shape[0], measlist.shape[1]
    filesel = np.zeros((nummeas, numsubjs), dtype=int)
    volumesel = np.zeros((nummeas, numsubjs), dtype=int)
    for thesubj in range(numsubjs):
        for themeas in range(nummeas):
            thecomponents = str(measlist[themeas, thesubj]).split(",")
            if len(thecomponents) == 2:
                filesel[themeas, thesubj] = int(thecomponents[0])
                volumesel[themeas, thesubj] = int(thecomponents[1])
            elif len(thecomponents) == 1:
                filesel[themeas, thesubj] = 0
                volumesel[themeas, thesubj] = int(thecomponents[0])
            else:
                print(
                    f"Error in element {themeas, thesubj}: each table entry has a maximum of 1 comma."
                )
                sys.exit()
            if filesel[themeas, thesubj] > numfiles - 1:
                print(f"Error in element {themeas, thesubj}: illegal file number.")
                sys.exit()
            if debug:
                print(
                    f"element {themeas, thesubj}: {filesel[themeas, thesubj]}, {volumesel[themeas, thesubj]}"
                )
    return filesel, volumesel


def makdcommandlinelist(arglist, starttime, endtime, extra=None):
    # get the processing date
    dateline = (
        "# Processed on "
        + time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime(starttime))
        + "."
    )
    timeline = f"# Processing took {endtime - starttime:.3f} seconds."

    # diagnostic information about version
    (
        release_version,
        dummy,
        git_date,
        dummy,
    ) = tide_util.version()

    nodeline = "# " + " ".join(
        [
            "Using",
            platform.node(),
            "(",
            release_version + ",",
            git_date,
            ")",
        ]
    )

    # and the actual command issued
    commandline = " ".join(arglist)

    if extra is not None:
        return [dateline, timeline, nodeline, "# " + extra, commandline]
    else:
        return [dateline, timeline, nodeline, commandline]


def calctexticc(args):
    runstarttime = time.time()

    datafiles = (args.datafile).split(",")
    numfiles = len(datafiles)
    measlist = tide_io.readvecs(args.measurementlist, thedtype=str)

    print(f"measurementlist shape: {measlist.shape}")
    (nummeas, numsubjs) = measlist.shape[0], measlist.shape[1]

    print(f"numsubjs: {numsubjs}, nummeas: {nummeas}")

    filesel, volumesel = parsetextmeasurementlist(measlist, numfiles, debug=args.debug)
    if args.debug:
        for i in range(len(filesel)):
            print(filesel[i], volumesel[i])

    # check the data headers first
    print("checking headers")
    dimlist = []
    for thefile in datafiles:
        thedims = tide_io.readvecs(thefile).shape
        dimlist.append([thedims[0] + 0, thedims[1] + 0])
        if args.debug:
            print(f"dimensions for file {thefile}: {thedims[1]} rows, {thedims[0]} columns")
    numvals = dimlist[0][0]
    print(f"numvals={numvals}, numsubjs={numsubjs}")

    # now read in the data
    print("reading in data files")
    if args.debug:
        print(f"target array size is {numvals, numsubjs * nummeas}")
    datafile_data = np.zeros((numvals, numsubjs * nummeas), dtype=float)
    for thisfile in range(numfiles):
        print(f"reading datafile {thisfile + 1}")
        inputfile_data = tide_io.readvecs(datafiles[thisfile])
        thisfilelocs = np.where(filesel == thisfile)
        for i in range(len(thisfilelocs[0])):
            themeas = thisfilelocs[0][i]
            thesubject = thisfilelocs[1][i]
            datafile_data[:, thesubject * nummeas + themeas] = np.nan_to_num(
                inputfile_data[:, volumesel[themeas, thesubject]]
            )
            if args.debug:
                print(
                    f"copying file:{thisfile}, volume:{volumesel[themeas, thesubject]} (meas:{themeas}, subject:{thesubject}) to volume {thesubject * nummeas + themeas}"
                )
    print(f"Done reading in data for {nummeas} measurements on {numsubjs} subjects")
    del inputfile_data

    # now reformat to voxelnumber, measurement, subject
    print("reshaping to voxel by (numsubjs * nummeas)")
    data_in_voxacq = datafile_data.reshape((numvals, numsubjs * nummeas))

    print("finding valid voxels")
    validvoxels = range(numvals)
    valid_in_voxacq = data_in_voxacq[validvoxels, :]

    print("reshaping to validvox by numsubjects by nummeas")
    validinvms = valid_in_voxacq.reshape((numvals, numsubjs, nummeas))
    print(validinvms.shape)

    ICC_in_valid = np.zeros((numvals), dtype=float)
    r_var_in_valid = np.zeros((numvals), dtype=float)
    e_var_in_valid = np.zeros((numvals), dtype=float)
    session_effect_F_in_valid = np.zeros((numvals), dtype=float)

    # remove median from each map, if requested
    if args.demedian:
        print("removing median map values")
        for thesubj in range(numsubjs):
            for themeas in range(0, nummeas):
                validinvms[:, thesubj, themeas] -= np.median(validinvms[:, thesubj, themeas])
        print("done removing median values")

    # remove mean from each map, if requested
    if args.demean:
        print("removing mean map values")
        for thesubj in range(numsubjs):
            for themeas in range(0, nummeas):
                validinvms[:, thesubj, themeas] -= np.mean(validinvms[:, thesubj, themeas])
        print("done removing median values")

    print("calculating ICC")
    iccstarttime = time.time()
    for voxel in range(numvals):
        # get the voxel data matrix
        Y = validinvms[voxel, :, :]

        if args.deepdebug:
            print(f"shape of Y: {Y.shape}")
            for thevolume in range(Y.shape[1]):
                print(f"\tY: {Y[:, thevolume]}")

        # calculate ICC(3,1)
        (
            ICC_in_valid[voxel],
            r_var_in_valid[voxel],
            e_var_in_valid[voxel],
            session_effect_F_in_valid[voxel],
            dfc,
            dfe,
        ) = tide_stats.fast_ICC_rep_anova(Y, nocache=args.nocache, debug=args.debug)
    iccduration = time.time() - iccstarttime

    print(f"\ndfc: {dfc}, dfe: {dfe}")

    extraline = f"ICC calculation time: {1000.0 * iccduration / numvals:.3f} ms per voxel.  nocache={args.nocache}"
    print(extraline)

    outarray_in_vox = np.zeros((numvals), dtype=float)

    outarray_in_vox[validvoxels] = ICC_in_valid[:]
    tide_io.writenpvecs(outarray_in_vox, f"{args.outputroot}_ICC.txt")
    outarray_in_vox[validvoxels] = r_var_in_valid[:]
    tide_io.writenpvecs(outarray_in_vox, f"{args.outputroot}_r_var.txt")
    outarray_in_vox[validvoxels] = e_var_in_valid[:]
    tide_io.writenpvecs(outarray_in_vox, f"{args.outputroot}_e_var.txt")
    outarray_in_vox[validvoxels] = session_effect_F_in_valid[:]
    tide_io.writenpvecs(
        outarray_in_vox,
        f"{args.outputroot}_session_effect_F.txt",
    )

    runendtime = time.time()
    thecommandfilelines = makdcommandlinelist(sys.argv, runstarttime, runendtime, extra=extraline)
    tide_io.writevec(thecommandfilelines, args.outputroot + "_commandline.txt")
