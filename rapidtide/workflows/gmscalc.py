#!/usr/bin/env python
import argparse
import platform
import sys
import time

import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="gmscalc",
        description="Calculate the global mean signal, and filtered versions",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "datafile",
        type=str,
        help=(
            "The name of a 4 dimensional nifti files (3 spatial dimensions and a "
            "subject dimension)."
        ),
    )
    parser.add_argument("outputroot", type=str, help="The root name for the output nifti files.")

    # Optional arguments
    parser.add_argument(
        "--dmask",
        dest="datamaskname",
        type=lambda x: pf.is_valid_file(parser, x),
        action="store",
        metavar="DATAMASK",
        help=("Use DATAMASK to specify which voxels in the data to use."),
        default=None,
    )
    pf.addnormalizationopts(parser, normtarget="timecourses", defaultmethod="None")

    parser.add_argument(
        "--normfirst",
        dest="normfirst",
        action="store_true",
        help=("Normalize before filtering, rather than after."),
        default=False,
    )
    parser.add_argument(
        "--smooth",
        dest="sigma",
        type=lambda x: pf.is_float(parser, x),
        action="store",
        metavar="SIGMA",
        help=("Spatially smooth the input data with a SIGMA mm kernel prior to calculation."),
        default=0.0,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Print out additional debugging information."),
        default=False,
    )
    return parser


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


def gmscalc_main():
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise

    runstarttime = time.time()

    # now read in the data
    print("reading datafile")
    (
        datafile_img,
        datafile_data,
        datafile_hdr,
        thedims,
        thesizes,
    ) = tide_io.readfromnifti(args.datafile)

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
    print(f"Datafile shape is {datafile_data.shape}")

    # smooth the data
    if args.sigma > 0.0:
        print("smoothing data")
        for i in range(timepoints):
            datafile_data[:, :, :, i] = tide_filt.ssmooth(
                xdim, ydim, slicethickness, args.sigma, datafile_data[:, :, :, i]
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
        datamask_data = datafile_data[:, :, :, 0] * 0.0 + 1.0

    # now reformat from x, y, z, time to voxelnumber, measurement, subject
    numvoxels = int(xsize) * int(ysize) * int(numslices)
    mask_in_vox = datamask_data.reshape((numvoxels))

    print("finding valid voxels")
    validvoxels = np.where(mask_in_vox > 0)[0]
    numvalid = int(len(validvoxels))

    data_in_voxacq = datafile_data.reshape((numvoxels, timepoints))
    valid_in_voxacq = data_in_voxacq[validvoxels, :]

    # calculate each voxel's mean over time
    mean_in_voxacq = np.mean(valid_in_voxacq, axis=1)

    # calculate mean timecourse
    gms = np.mean(valid_in_voxacq, axis=0)

    # now filter
    lfofilter = tide_filt.NoncausalFilter("lfo")
    hffilter = tide_filt.NoncausalFilter("arb")
    lowerpass = 0.175
    lowerstop = 0.95 * lowerpass
    hffilter.setfreqs(lowerstop, lowerpass, 10.0, 10.0)
    gms_lfo = tide_math.normalize(lfofilter.apply(1.0 / tr, gms), method=args.normmethod)
    gms_hf = tide_math.normalize(hffilter.apply(1.0 / tr, gms), method=args.normmethod)

    tide_io.writevec(gms, args.outputroot + "_gms.txt")
    tide_io.writevec(gms_lfo, args.outputroot + "_gmslfo.txt")
    tide_io.writevec(gms_hf, args.outputroot + "_gmshf.txt")
    runendtime = time.time()
    thecommandfilelines = makdcommandlinelist(sys.argv, runstarttime, runendtime)
    tide_io.writevec(thecommandfilelines, args.outputroot + "_commandline.txt")
