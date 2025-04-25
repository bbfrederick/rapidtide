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
import time

import numpy as np
from tqdm import tqdm

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.workflows.parser_funcs as pf

DEFAULT_NUMSPACESTEPS = 1
DEFAULT_NPASSES = 20
DEFAULT_RADIUS = 10.5
DEFAULT_WINDOW_TYPE = "hamming"
DEFAULT_DETREND_ORDER = 3
DEFAULT_AMPTHRESH = 0.3
DEFAULT_MINLAGDIFF = 0.0


def _get_parser():
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="localflow",
        description="Calculate local sources of signal.",
        allow_abbrev=False,
    )
    parser.add_argument("inputfilename", type=str, help="The name of the input nifti file.")
    parser.add_argument("outputroot", type=str, help="The root name of the output nifti files.")

    parser.add_argument(
        "--npasses",
        dest="npasses",
        type=int,
        help=f"The number of passes for reconstruction.  Default is {DEFAULT_NPASSES}",
        default=DEFAULT_NPASSES,
    )
    parser.add_argument(
        "--radius",
        dest="radius",
        type=float,
        help=f"The radius around the voxel to check correlations.  Default is {DEFAULT_RADIUS}",
        default=DEFAULT_RADIUS,
    )
    parser.add_argument(
        "--minlagdiff",
        dest="minlagdiff",
        type=float,
        help=f"The minimum lagtime difference threshold to select which diffs to include in reconstruction.  Default is {DEFAULT_MINLAGDIFF}",
        default=DEFAULT_MINLAGDIFF,
    )
    parser.add_argument(
        "--ampthresh",
        dest="ampthresh",
        type=float,
        help=f"The correlation threshold to select which diffs to include in reconstruction.  Default is {DEFAULT_AMPTHRESH}",
        default=DEFAULT_AMPTHRESH,
    )
    parser.add_argument(
        "--gausssigma",
        dest="gausssigma",
        type=float,
        help=(
            "Spatially filter fMRI data prior to analysis "
            "using GAUSSSIGMA in mm.  Set GAUSSSIGMA negative "
            "to set it to half the mean voxel "
            "dimension (a rule of thumb for a good value)."
        ),
        default=0.0,
    )
    parser.add_argument(
        "--oversampfac",
        dest="oversampfactor",
        action="store",
        type=int,
        metavar="OVERSAMPFAC",
        help=(
            "Oversample the fMRI data by the following "
            "integral factor.  Set to -1 for automatic selection (default)."
        ),
        default=-1,
    )
    parser.add_argument(
        "--dofit",
        dest="dofit",
        action="store_true",
        help="Turn on correlation fitting.",
        default=False,
    )
    parser.add_argument(
        "--detrendorder",
        dest="detrendorder",
        action="store",
        type=int,
        metavar="ORDER",
        help=(f"Set order of trend removal (0 to disable). Default is {DEFAULT_DETREND_ORDER}."),
        default=DEFAULT_DETREND_ORDER,
    )
    parser.add_argument(
        "--nosphere",
        dest="dosphere",
        action="store_false",
        help=("Use rectangular rather than spherical reconstruction kernel."),
        default=True,
    )

    pf.addfilteropts(parser, filtertarget="data and regressors", details=True)
    pf.addwindowopts(parser, windowtype=DEFAULT_WINDOW_TYPE)

    misc = parser.add_argument_group("Miscellaneous options")
    misc.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help=("Will disable showing progress bars (helpful if stdout is going " "to a file)."),
        default=True,
    )
    misc.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Turn on debugging information.",
        default=False,
    )
    return parser


def preprocdata(
    fmridata,
    themask,
    theprefilter,
    oversamplefactor,
    Fs,
    tr,
    detrendorder=3,
    windowfunc="hamming",
    padseconds=0,
    showprogressbar=True,
):
    numspatiallocs = fmridata.shape[0] * fmridata.shape[1] * fmridata.shape[2]
    timepoints = fmridata.shape[3]

    initial_fmri_x = np.arange(0.0, timepoints) * tr

    oversamptr = tr / oversamplefactor
    oversampFs = oversamplefactor * Fs
    os_fmri_x = np.arange(0.0, timepoints * oversamplefactor - (oversamplefactor - 1))
    os_fmri_x *= oversamptr
    ostimepoints = len(os_fmri_x)
    fmridata_byvox = fmridata.reshape((numspatiallocs, timepoints))
    themask_byvox = themask.reshape((numspatiallocs))
    osfmridata = np.zeros(
        (fmridata.shape[0], fmridata.shape[1], fmridata.shape[2], ostimepoints), dtype=float
    )
    osfmridata_byvox = osfmridata.reshape((numspatiallocs, ostimepoints))

    numvoxelsprocessed = 0
    for thevoxel in tqdm(
        range(0, numspatiallocs),
        desc="Voxel",
        unit="voxels",
        disable=(not showprogressbar),
    ):
        if themask_byvox[thevoxel] > 0:
            osfmridata_byvox[thevoxel, :] = tide_math.corrnormalize(
                theprefilter.apply(
                    oversampFs,
                    tide_resample.doresample(
                        initial_fmri_x,
                        fmridata_byvox[thevoxel, :],
                        os_fmri_x,
                        padlen=int(oversampFs * padseconds),
                    ),
                ),
                detrendorder=detrendorder,
                windowfunc=windowfunc,
            )
            numvoxelsprocessed += 1
    return osfmridata_byvox, ostimepoints, oversamptr, numvoxelsprocessed


def getcorrloc(
    thedata,
    idx1,
    idx2,
    Fs,
    dofit=False,
    lagmin=-12.5,
    lagmax=12.5,
    widthmax=100.0,
    negsearch=15.0,
    possearch=15.0,
    padding=0,
    debug=False,
):
    tc1 = thedata[idx1, :]
    tc2 = thedata[idx2, :]
    if np.any(tc1) != 0.0 and np.any(tc2) != 0.0:
        if debug:
            print(f"{idx1=}, {idx2=}")
            print(f"{tc1=}")
            print(f"{tc2=}")

        thesimfunc = tide_corr.fastcorrelate(
            tc1,
            tc2,
            zeropadding=padding,
            usefft=True,
            debug=debug,
        )
        similarityfunclen = len(thesimfunc)
        similarityfuncorigin = similarityfunclen // 2 + 1

        negpoints = int(negsearch * Fs)
        pospoints = int(possearch * Fs)
        trimsimfunc = thesimfunc[
            similarityfuncorigin - negpoints : similarityfuncorigin + pospoints
        ]
        offset = 0.0
        trimtimeaxis = (
            (
                np.arange(0.0, similarityfunclen) * (1.0 / Fs)
                - ((similarityfunclen - 1) * (1.0 / Fs)) / 2.0
            )
            - offset
        )[similarityfuncorigin - negpoints : similarityfuncorigin + pospoints]
        if dofit:
            (
                maxindex,
                maxtime,
                maxcorr,
                maxsigma,
                maskval,
                failreason,
                peakstart,
                peakend,
            ) = tide_fit.simfuncpeakfit(
                trimsimfunc,
                trimtimeaxis,
                useguess=False,
                maxguess=0.0,
                displayplots=False,
                functype="correlation",
                peakfittype="gauss",
                searchfrac=0.5,
                lagmod=1000.0,
                enforcethresh=True,
                allowhighfitamps=False,
                lagmin=lagmin,
                lagmax=lagmax,
                absmaxsigma=1000.0,
                absminsigma=0.25,
                hardlimit=True,
                bipolar=False,
                lthreshval=0.0,
                uthreshval=1.0,
                zerooutbadfit=True,
                debug=False,
            )
        else:
            maxtime = trimtimeaxis[np.argmax(trimsimfunc)]
            maxcorr = np.max(trimsimfunc)
            maskval = 1
            failreason = 0
        if debug:
            print(f"{maxtime=}")
            print(f"{maxcorr=}")
            print(f"{maskval=}")
            print(f"{negsearch=}")
            print(f"{possearch=}")
            print(f"{Fs=}")
            print(f"{len(trimtimeaxis)=}")
            print(trimsimfunc, trimtimeaxis)
        return maxcorr, maxtime, maskval, failreason
    else:
        return 0.0, 0.0, 0, 0


def xyz2index(x, y, z, xsize, ysize, zsize):
    if (0 <= x < xsize) and (0 <= y < ysize) and (0 <= z < zsize):
        return int(z) + int(y) * int(zsize) + int(x) * int(zsize * ysize)
    else:
        return -1


def index2xyz(theindex, ysize, zsize):
    x = theindex // int(zsize * ysize)
    theindex -= int(x) * int(zsize * ysize)
    y = theindex // int(zsize)
    theindex -= int(y) * int(zsize)
    z = theindex
    return x, y, z


def localflow(args):
    # set default variable values
    displayplots = False

    # postprocess filter options
    theobj, theprefilter = pf.postprocessfilteropts(args)

    # save timinginfo
    eventtimes = []
    starttime = time.time()
    thistime = starttime
    eventtimes.append(["Start", 0.0, 0.0, None, None])

    # get the input TR
    inputtr_fromfile, numinputtrs = tide_io.fmritimeinfo(args.inputfilename)
    print("input data: ", numinputtrs, " timepoints, tr = ", inputtr_fromfile)

    input_img, fmridata, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    if input_hdr.get_xyzt_units()[1] == "msec":
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    Fs = 1.0 / tr
    print("tr from header =", tr, ", sample frequency is ", Fs)
    thistime = time.time() - starttime
    eventtimes.append(["Read input file", thistime, thistime, None, None])

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)

    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    fmridata_voxbytime = fmridata.reshape((numspatiallocs, timepoints))
    if args.debug:
        print(f"{fmridata.shape=}")
        print(f"{fmridata_voxbytime.shape=}")

    # make a mask
    meanim = np.mean(fmridata, axis=3)
    themask = np.uint16(tide_stats.makemask(meanim, threshpct=0.1))
    themask_byvox = themask.reshape((numspatiallocs))
    validvoxels = np.where(themask > 0)
    numvalid = len(validvoxels[0])
    print(f"{numvalid} valid")
    output_hdr = copy.deepcopy(input_hdr)
    output_hdr["dim"][4] = 1
    tide_io.savetonifti(
        themask,
        output_hdr,
        f"{args.outputroot}_mask",
        debug=args.debug,
    )
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Made and saved mask", thistime, thistime - lasttime, None, None])

    lasttime = thistime
    thistime = time.time() - starttime
    if args.gausssigma < 0.0:
        # set gausssigma automatically
        args.gausssigma = np.mean([xdim, ydim, slicethickness]) / 2.0
    if args.gausssigma > 0.0:
        eventtimes.append(["Spatial filter start", thistime, thistime - lasttime, None, None])
        print(f"applying gaussian spatial filter to fmri data " f" with sigma={args.gausssigma}")
        for i in tqdm(
            range(timepoints),
            desc="Timepoint",
            unit="timepoints",
            disable=(not args.showprogressbar),
        ):
            fmridata[:, :, :, i] = tide_filt.ssmooth(
                xdim,
                ydim,
                slicethickness,
                args.gausssigma,
                fmridata[:, :, :, i],
            )
        lasttime = thistime
        thistime = time.time() - starttime
        eventtimes.append(
            ["Spatial filter done", thistime, thistime - lasttime, timepoints, "timepoints"]
        )

    # prepare the input data
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Prepare data start", thistime, thistime - lasttime, None, None])
    if args.oversampfactor == -1:
        oversamplefactor = int(np.max([np.ceil(tr / 0.5), 1]))
    else:
        oversamplefactor = args.oversampfactor
    print(f"using an oversample factor of {oversamplefactor}")
    print("Preparing data", flush=True)
    osfmridata_voxbytime, ostimepoints, oversamptr, numvoxelsprocessed = preprocdata(
        fmridata,
        themask,
        theprefilter,
        oversamplefactor,
        Fs,
        tr,
        detrendorder=args.detrendorder,
        windowfunc=args.windowfunc,
        padseconds=args.padseconds,
        showprogressbar=args.showprogressbar,
    )
    print("...done", flush=True)
    print("\n", flush=True)
    output_hdr = copy.deepcopy(input_hdr)
    output_hdr["dim"][4] = ostimepoints
    output_hdr["pixdim"][4] = oversamptr
    tide_io.savetonifti(
        osfmridata_voxbytime.reshape((xsize, ysize, numslices, ostimepoints)),
        output_hdr,
        f"{args.outputroot}_preprocdata",
        debug=args.debug,
    )
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        ["Prepare data done", thistime, thistime - lasttime, numvoxelsprocessed, "voxels"]
    )

    # make list of neighbors
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Find neighbors start", thistime, thistime - lasttime, None, None])
    args.dosphere = True
    xsteps = int(np.ceil(args.radius / xdim))
    ysteps = int(np.ceil(args.radius / ydim))
    zsteps = int(np.ceil(args.radius / slicethickness))

    neighborlist = []
    distancelist = []
    for z in range(-zsteps, zsteps + 1):
        for y in range(-ysteps, ysteps + 1):
            for x in range(-xsteps, xsteps + 1):
                if args.dosphere:
                    distance = np.sqrt(
                        np.square(x * xdim) + np.square(y * ydim) + np.square(z * slicethickness)
                    )
                    if (x != 0 or y != 0 or z != 0) and distance <= args.radius:
                        neighborlist.append((x, y, z))
                        distancelist.append(distance)
                else:
                    if x != 0 or y != 0 or z != 0:
                        neighborlist.append((x, y, z))
    tide_io.writenpvecs(np.transpose(np.asarray(neighborlist)), f"{args.outputroot}_neighbors")
    if args.debug:
        print(f"{len(neighborlist)=}, {neighborlist=}")
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        ["Find neighbors done", thistime, thistime - lasttime, len(neighborlist), "voxels"]
    )

    corrcoeffs = np.zeros((xsize, ysize, numslices, len(neighborlist)), dtype=float)
    delays = np.zeros((xsize, ysize, numslices, len(neighborlist)), dtype=float)
    corrvalid = np.zeros((xsize, ysize, numslices, len(neighborlist)), dtype=int)
    failreason = np.zeros((xsize, ysize, numslices, len(neighborlist)), dtype=int)
    if args.debug:
        print(f"{corrcoeffs.shape=}, {delays.shape=}, {corrvalid.shape=}")
        printfirstdetails = True
    else:
        printfirstdetails = False

    corrcoeffs_byvox = corrcoeffs.reshape((numspatiallocs, len(neighborlist)))
    delays_byvox = delays.reshape((numspatiallocs, len(neighborlist)))
    corrvalid_byvox = corrvalid.reshape((numspatiallocs, len(neighborlist)))
    failreason_byvox = failreason.reshape((numspatiallocs, len(neighborlist)))

    # Find every voxel's valid neighbors
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        ["Generate correlation list start", thistime, thistime - lasttime, None, None]
    )
    print("Generate the correlation list", flush=True)
    indexlist = []
    indexpairs = []
    theindex = 0
    for index1 in tqdm(
        range(numspatiallocs),
        desc="Voxel",
        unit="voxels",
        disable=(not args.showprogressbar),
    ):
        if themask_byvox[index1] > 0:
            # voxel is in the mask
            x, y, z = index2xyz(index1, ysize, numslices)
            for idx, neighbor in enumerate(neighborlist):
                index2 = xyz2index(
                    x + neighbor[0],
                    y + neighbor[1],
                    z + neighbor[2],
                    xsize,
                    ysize,
                    numslices,
                )
                if index2 > 0:
                    # neighbor location is valid
                    if themask_byvox[index2] > 0:
                        # neighbor is in the mask
                        indexpairs.append((index1, index2, idx, theindex + 0))
            theindex += 1
            indexlist.append(index1)
    print("...done", flush=True)
    print("\n", flush=True)
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        [
            "Generate correlation list done",
            thistime,
            thistime - lasttime,
            len(indexpairs),
            "correlation pairs",
        ]
    )
    tide_io.writenpvecs(np.transpose(np.asarray(indexlist)), f"{args.outputroot}_indexlist")

    # Do the correlations
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Do correlations start", thistime, thistime - lasttime, None, None])
    print(f"Process {len(indexpairs)} correlations", flush=True)
    for index1, index2, neighboridx, theindex in tqdm(
        indexpairs,
        desc="Correlation pair",
        unit="pairs",
        disable=(not args.showprogressbar),
    ):
        # print(index1, index2, neighboridx, theindex)
        (
            corrcoeffs_byvox[index1, neighboridx],
            delays_byvox[index1, neighboridx],
            corrvalid_byvox[index1, neighboridx],
            failreason_byvox[index1, neighboridx],
        ) = getcorrloc(
            osfmridata_voxbytime,
            index1,
            index2,
            oversamplefactor * Fs,
            dofit=args.dofit,
            debug=printfirstdetails,
        )
        neighborloc = (
            (neighborlist[neighboridx])[0] + xsteps,
            (neighborlist[neighboridx])[1] + ysteps,
            (neighborlist[neighboridx])[2] + zsteps,
        )
        # print(neighborlist[neighboridx], neighborloc)
    print("...done", flush=True)
    print("\n", flush=True)
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        [
            "Do correlations done",
            thistime,
            thistime - lasttime,
            len(indexpairs),
            "correlation pairs",
        ]
    )

    output_hdr = copy.deepcopy(input_hdr)
    output_hdr["dim"][4] = len(neighborlist)
    tide_io.savetonifti(
        corrcoeffs,
        output_hdr,
        f"{args.outputroot}_corrcoeffs",
        debug=args.debug,
    )
    tide_io.savetonifti(
        delays,
        output_hdr,
        f"{args.outputroot}_delays",
        debug=args.debug,
    )
    tide_io.savetonifti(
        corrvalid,
        output_hdr,
        f"{args.outputroot}_corrvalid",
        debug=args.debug,
    )
    tide_io.savetonifti(
        failreason,
        output_hdr,
        f"{args.outputroot}_failreason",
        debug=args.debug,
    )

    # now reconstruct
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Reconstruction start", thistime, thistime - lasttime, None, None])
    print("Reconstruct", flush=True)
    gain = 0.95
    targetdelay = np.zeros((xsize, ysize, numslices, args.npasses), dtype=float)
    targetdelay_byvox = targetdelay.reshape((numspatiallocs, args.npasses))

    targetdelay[:, :, :, 0] = 0.0 * (np.random.random((xsize, ysize, numslices)) - 0.5)

    numneighbors = np.zeros((xsize, ysize, numslices), dtype=int)
    numneighbors_byvox = numneighbors.reshape((numspatiallocs))
    # loop over passes
    for thepass in tqdm(
        range(1, args.npasses), desc="Pass", unit="passes", disable=(not args.showprogressbar)
    ):
        # loop over voxels
        for thearrayindex, thecoordindex in enumerate(indexlist):
            deltasum = 0.0
            numneighbors_byvox[thecoordindex] = 0
            for whichneighbor in range(len(neighborlist)):
                if (
                    corrvalid_byvox[
                        thecoordindex,
                        whichneighbor,
                    ]
                    > 0
                    and np.fabs(
                        corrcoeffs_byvox[
                            thecoordindex,
                            whichneighbor,
                        ]
                    )
                    > args.ampthresh
                ):
                    thediff = (
                        delays_byvox[
                            thecoordindex,
                            whichneighbor,
                        ]
                        - targetdelay_byvox[thecoordindex, thepass - 1]
                    )
                    thenorm = corrcoeffs_byvox[
                        thecoordindex,
                        whichneighbor,
                    ]
                    numneighbors_byvox[thecoordindex] += 1
                    # deltasum += thediff * thenorm * thenorm / distancelist[whichneighbor]
                    deltasum += thediff * thenorm
            if numneighbors_byvox[thecoordindex] > 0:
                targetdelay_byvox[thecoordindex, thepass] = (
                    gain * targetdelay_byvox[thecoordindex, thepass - 1]
                    + deltasum / numneighbors_byvox[thecoordindex]
                )
    print("...done", flush=True)
    print("\n", flush=True)
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        ["Reconstruction done", thistime, thistime - lasttime, args.npasses - 1, "passes"]
    )

    output_hdr = copy.deepcopy(input_hdr)
    output_hdr["dim"][4] = args.npasses
    tide_io.savetonifti(
        targetdelay,
        output_hdr,
        f"{args.outputroot}_targetdelay",
        debug=args.debug,
    )
    output_hdr["dim"][4] = 1
    tide_io.savetonifti(
        numneighbors,
        output_hdr,
        f"{args.outputroot}_numneighbors",
        debug=args.debug,
    )
    output_hdr["dim"][4] = 1
    tide_io.savetonifti(
        targetdelay[:, :, :, -1],
        output_hdr,
        f"{args.outputroot}_maxtime",
        debug=args.debug,
    )
    formattedtimings = []
    for eventtime in eventtimes:
        if eventtime[3] is not None:
            formattedtimings.append(
                f"{eventtime[1]:.2f}\t{eventtime[2]:.2f}\t{eventtime[0]}\t{eventtime[3]/eventtime[2]:.2f} ({eventtime[4]}/sec)"
            )
        else:
            formattedtimings.append(f"{eventtime[1]:.2f}\t{eventtime[2]:.2f}\t{eventtime[0]}")
        print(formattedtimings[-1])
    tide_io.writevec(formattedtimings, f"{args.outputroot}_formattedruntimings.txt")
