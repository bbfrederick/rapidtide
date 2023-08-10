#!/usr/bin/env python
import argparse
import copy
import platform
import sys
import time

import numpy as np
from scipy.stats import ttest_ind, ttest_rel
from tqdm import tqdm

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
from rapidtide.workflows.parser_funcs import is_float, is_valid_file


def _get_parser(calctype="icc"):
    if calctype == "icc":
        parser = argparse.ArgumentParser(
            prog="calcicc",
            description="Calculate per-voxel ICC(3,1) on a set of nifti images. (workflow version)",
            allow_abbrev=False,
        )
    elif calctype == "ttest":
        parser = argparse.ArgumentParser(
            prog="calcttest",
            description="Calculate per-voxel t tests between pairs of nifti files. (workflow version)",
            allow_abbrev=False,
        )
    else:
        raise ("Illegal calculation type")

    # Required arguments
    parser.add_argument(
        "datafile",
        type=str,
        help=(
            "A comma separated list of 1 or more 4 dimensional nifti files (3 spatial dimensions and a "
            "subject dimension).  Each file is a measurement.  The subject dimension in each file must be the same."
        ),
    )
    if calctype == "icc":
        parser.add_argument(
            "measurementlist",
            type=lambda x: is_valid_file(parser, x),
            help=(
                "A multicolumn tab separated value file of integers specifying which images to compare.  Each row is a "
                "subject, each column is a measurement.  A measurement is either in the format of SUBJNUM "
                "or FILENUM,SUBJNUM. Subject, file and measurement numbering starts at 0.  If no filenum is specified, "
                "it is assumed to be 0"
            ),
        )
    parser.add_argument("outputroot", type=str, help="The root name for the output nifti files.")

    # Optional arguments
    parser.add_argument(
        "--dmask",
        dest="datamaskname",
        type=lambda x: is_valid_file(parser, x),
        action="store",
        metavar="DATAMASK",
        help=("Use DATAMASK to specify which voxels in the data to use."),
        default=None,
    )
    parser.add_argument(
        "--smooth",
        dest="sigma",
        type=lambda x: is_float(parser, x),
        action="store",
        metavar="SIGMA",
        help=("Spatially smooth the input data with a SIGMA mm kernel prior to calculation."),
        default=0.0,
    )
    if calctype == "ttest":
        parser.add_argument(
            "--paired",
            dest="paired",
            action="store_true",
            help=("Perform a paired t test (default is independent)."),
            default=False,
        )
        parser.add_argument(
            "--alternative",
            dest="alternative",
            action="store",
            type=str,
            choices=["two-sided", "less", "greater"],
            help=(
                "Defines the alternative hypothesis. The options are:‘two-sided’ - the means of the "
                "distributions underlying the samples are unequal. ‘less’: the mean of the distribution "
                "underlying the first sample is less than the mean of the distribution underlying the "
                "second sample.  ‘greater’: the mean of the distribution underlying the first sample is "
                "greater than the mean of the distribution underlying the second sample."
            ),
            default="two-sided",
        )
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
    if calctype == "icc":
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
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help=("Disable progress bar."),
        default=True,
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


def parsemeasurementlist(measlist, numfiles, debug=False):
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


def niftistats_main(calctype="icc"):
    try:
        args = _get_parser(calctype=calctype).parse_args()
    except SystemExit:
        _get_parser(calctype=calctype).print_help()
        raise

    runstarttime = time.time()

    datafiles = (args.datafile).split(",")
    numfiles = len(datafiles)

    if calctype == "icc":
        measlist = tide_io.readvecs(args.measurementlist, thedtype=str)
        print(f"measurementlist shape: {measlist.shape}")
        (nummeas, numsubjs) = measlist.shape[0], measlist.shape[1]
        if nummeas < 2:
            print(
                "ICC requires at least two measurements per subject - specify at least two measurements."
            )
            sys.exit()

        filesel, volumesel = parsemeasurementlist(measlist, numfiles, debug=args.debug)
    elif calctype == "ttest":
        nummeas = len(datafiles)
        if nummeas != 2:
            print(
                "ttest requires at exactly two measurements  - specify two nifti files of equal dimensions."
            )
            sys.exit()

    # check the data headers first
    print("checking headers")
    sizelist = []
    dimlist = []
    for thefile in datafiles:
        thesizes, thedims = tide_io.fmriheaderinfo(thefile)
        sizelist.append(thesizes[1:5].copy())
        dimlist.append(thedims[1:5].copy())
    for i in range(1, len(sizelist)):
        if not (sizelist[i] == sizelist[0]).all() or not (dimlist[i] == dimlist[0]).all():
            print(f"data file {i + 1} does not match first file")
            sys.exit()
    xsize, ysize, numslices, dummy = dimlist[0]
    if calctype == "ttest":
        numsubjs = dimlist[0][3]
    xdim, ydim, slicethickness, dummy = sizelist[0]
    print(f"{numsubjs=}, {nummeas=}")

    # now read in the data
    print("reading in data files")
    if args.debug:
        print(f"target array size is {xsize, ysize, numslices, numsubjs * nummeas}")
    if calctype == "icc":
        datafile_data = np.zeros((xsize, ysize, numslices, numsubjs * nummeas), dtype=float)
    elif calctype == "ttest":
        datafile_data = np.zeros((xsize, ysize, numslices, numsubjs, nummeas), dtype=float)
    for thisfile in range(numfiles):
        print(f"reading datafile {thisfile + 1}")
        (
            _,
            inputfile_data,
            datafile_hdr,
            _,
            _,
        ) = tide_io.readfromnifti(datafiles[thisfile])
        if calctype == "icc":
            thisfilelocs = np.where(filesel == thisfile)
            for i in range(len(thisfilelocs[0])):
                themeas = thisfilelocs[0][i]
                thesubject = thisfilelocs[1][i]
                datafile_data[:, :, :, thesubject * nummeas + themeas] = np.nan_to_num(
                    inputfile_data[:, :, :, volumesel[themeas, thesubject]]
                )
                if args.debug:
                    print(
                        f"copying file:{thisfile}, volume:{volumesel[themeas, thesubject]} (meas:{themeas}, subject:{thesubject}) to volume {thesubject * nummeas + themeas}"
                    )
        elif calctype == "ttest":
            datafile_data[:, :, :, :, thisfile] = inputfile_data[:, :, :, :] + 0.0
    print(f"Done reading in data for {nummeas} measurements on {numsubjs} subjects")
    print(f"Datafile shape is {datafile_data.shape}")
    del inputfile_data

    # smooth the data
    if args.sigma > 0.0:
        print("smoothing data")
        if calctype == "icc":
            for i in range(numsubjs * nummeas):
                datafile_data[:, :, :, i] = tide_filt.ssmooth(
                    xdim, ydim, slicethickness, args.sigma, datafile_data[:, :, :, i]
                )
        elif calctype == "ttest":
            for i in range(numsubjs):
                datafile_data[:, :, :, i, 0] = tide_filt.ssmooth(
                    xdim, ydim, slicethickness, args.sigma, datafile_data[:, :, :, i, 0]
                )
                datafile_data[:, :, :, i, 1] = tide_filt.ssmooth(
                    xdim, ydim, slicethickness, args.sigma, datafile_data[:, :, :, i, 1]
                )
        print("done smoothing data")
    if args.datamaskname is not None:
        print("reading in mask array")
        (
            datamask_img,
            datamask_data,
            datamask_hdr,
            datamaskdims,
            datamasksizes,
        ) = tide_io.readfromnifti(args.datamaskname)
        if not tide_io.checkspacematch(datafile_hdr, datamask_hdr):
            print("Data and mask dimensions do not match - exiting.")
            sys.exit()
        print("done reading in mask array")
    else:
        if calctype == "icc":
            datamask_data = datafile_data[:, :, :, 0] * 0.0 + 1.0
        elif calctype == "ttest":
            datamask_data = datafile_data[:, :, :, 0, 0] * 0.0 + 1.0

    # now reformat from x, y, z, time to voxelnumber, measurement, subject
    numvoxels = int(xsize) * int(ysize) * int(numslices)
    mask_in_vox = datamask_data.reshape((numvoxels))

    print("finding valid voxels")
    validvoxels = np.where(mask_in_vox > 0)[0]
    numvalid = int(len(validvoxels))

    if calctype == "icc":
        print("reshaping to voxel by (numsubjs * nummeas)")
        data_in_voxacq = datafile_data.reshape((numvoxels, numsubjs * nummeas))
        valid_in_voxacq = data_in_voxacq[validvoxels, :]
        print("reshaping to validvox by numsubjects by nummeas")
        validinvms = valid_in_voxacq.reshape((numvalid, numsubjs, nummeas))
    elif calctype == "ttest":
        print("reshaping to voxel by numsubjs by nummeas")
        data_in_vms = datafile_data.reshape((numvoxels, numsubjs, nummeas))
        print("reshaping to validvox by numsubjects by nummeas")
        validinvms = data_in_vms[validvoxels, :, :]
    print(validinvms.shape)

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

    if calctype == "icc":
        print("calculating ICC")
        ICC_in_valid = np.zeros((numvalid), dtype=float)
        r_var_in_valid = np.zeros((numvalid), dtype=float)
        e_var_in_valid = np.zeros((numvalid), dtype=float)
        session_effect_F_in_valid = np.zeros((numvalid), dtype=float)

        iccstarttime = time.time()
        for voxel in tqdm(
            range(0, numvalid),
            desc="Voxel",
            unit="voxels",
            disable=(not args.showprogressbar),
        ):
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

        extraline = f"ICC calculation time: {1000.0 * iccduration / numvalid:.3f} ms per voxel.  nocache={args.nocache}"
        print(extraline)
    elif calctype == "ttest":
        t_in_valid = np.zeros((numvalid), dtype=float)
        p_in_valid = np.zeros((numvalid), dtype=float)

        tteststarttime = time.time()
        for voxel in tqdm(
            range(0, numvalid),
            desc="Voxel",
            unit="voxels",
            disable=(not args.showprogressbar),
        ):
            if args.paired:
                t_in_valid[voxel], p_in_valid[voxel], df = ttest_rel(
                    validinvms[voxel, :, 0],
                    validinvms[voxel, :, 1],
                    alternative=args.alternative,
                )
            else:
                t_in_valid[voxel], p_in_valid[voxel] = ttest_ind(
                    validinvms[voxel, :, 0],
                    validinvms[voxel, :, 1],
                    alternative=args.alternative,
                )

        ttestduration = time.time() - tteststarttime

        extraline = (
            f"t test calculation time: {1000.0 * ttestduration / numvalid:.3f} ms per voxel."
        )
        print(extraline)

    outarray_in_vox = mask_in_vox * 0.0

    theheader = copy.deepcopy(datafile_hdr)
    theheader["dim"][0] = 3
    theheader["dim"][4] = 1

    if calctype == "icc":
        outarray_in_vox[validvoxels] = ICC_in_valid[:]
        tide_io.savetonifti(
            outarray_in_vox.reshape(xsize, ysize, numslices), theheader, f"{args.outputroot}_ICC"
        )
        outarray_in_vox[validvoxels] = r_var_in_valid[:]
        tide_io.savetonifti(
            outarray_in_vox.reshape(xsize, ysize, numslices), theheader, f"{args.outputroot}_r_var"
        )
        outarray_in_vox[validvoxels] = e_var_in_valid[:]
        tide_io.savetonifti(
            outarray_in_vox.reshape(xsize, ysize, numslices), theheader, f"{args.outputroot}_e_var"
        )
        outarray_in_vox[validvoxels] = session_effect_F_in_valid[:]
        tide_io.savetonifti(
            outarray_in_vox.reshape(xsize, ysize, numslices),
            theheader,
            f"{args.outputroot}_session_effect_F",
        )
    elif calctype == "ttest":
        outarray_in_vox[validvoxels] = t_in_valid[:]
        tide_io.savetonifti(
            outarray_in_vox.reshape(xsize, ysize, numslices), theheader, f"{args.outputroot}_t"
        )
        outarray_in_vox[validvoxels] = p_in_valid[:]
        tide_io.savetonifti(
            outarray_in_vox.reshape(xsize, ysize, numslices), theheader, f"{args.outputroot}_p"
        )

    runendtime = time.time()
    thecommandfilelines = makdcommandlinelist(sys.argv, runstarttime, runendtime, extra=extraline)
    tide_io.writevec(thecommandfilelines, args.outputroot + "_commandline.txt")
