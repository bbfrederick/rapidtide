#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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
import bisect
import copy
import gc
import logging
import multiprocessing as mp
import os
import platform
import warnings
from pathlib import Path

import numpy as np
from nilearn import masking
from scipy import ndimage
from sklearn.decomposition import PCA
from tqdm import tqdm

import rapidtide.calccoherence as tide_calccoherence
import rapidtide.calcnullsimfunc as tide_nullsimfunc
import rapidtide.calcsimfunc as tide_calcsimfunc
import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.glmpass as tide_glmpass
import rapidtide.helper_classes as tide_classes
import rapidtide.io as tide_io
import rapidtide.maskutil as tide_mask
import rapidtide.miscmath as tide_math
import rapidtide.multiproc as tide_multiproc
import rapidtide.peakeval as tide_peakeval
import rapidtide.refine as tide_refine
import rapidtide.resample as tide_resample
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
import rapidtide.wiener as tide_wiener
from rapidtide.tests.utils import mse

from .utils import setup_logger

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False

try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False

LGR = logging.getLogger("GENERAL")
TimingLGR = logging.getLogger("TIMING")


def conditionalprofile():
    def resdec(f):
        if memprofilerexists:
            return profile(f)
        return f

    return resdec


@conditionalprofile()
def memcheckpoint(message):
    print(message)


def numpy2shared(inarray, thetype, debug=False):
    thesize = inarray.size
    theshape = inarray.shape
    if debug:
        print(f"numpy2shared: {thesize=}")
        print(f"numpy2shared: {theshape=}")
    if thetype == np.float64:
        inarray_shared = mp.RawArray("d", inarray.reshape(thesize))
    else:
        inarray_shared = mp.RawArray("f", inarray.reshape(thesize))
    inarray = np.frombuffer(inarray_shared, dtype=thetype, count=thesize)
    inarray.shape = theshape
    if debug:
        print(f"numpy2shared: done")
    return inarray


def allocshared(theshape, thetype):
    thesize = int(1)
    if not isinstance(theshape, (list, tuple)):
        thesize = theshape
    else:
        for element in theshape:
            thesize *= int(element)
    if thetype == np.float64:
        outarray_shared = mp.RawArray("d", thesize)
    else:
        outarray_shared = mp.RawArray("f", thesize)
    outarray = np.frombuffer(outarray_shared, dtype=thetype, count=thesize)
    outarray.shape = theshape
    return outarray, outarray_shared, theshape


def getglobalsignal(indata, optiondict, includemask=None, excludemask=None, pcacomponents=0.8):
    # Start with all voxels
    themask = indata[:, 0] * 0 + 1

    # modify the mask if needed
    if includemask is not None:
        themask = themask * includemask
    if excludemask is not None:
        themask = themask * (1 - excludemask)

    # combine all the voxels using one of the three methods
    global rt_floatset, rt_floattype
    globalmean = rt_floatset(indata[0, :])
    thesize = np.shape(themask)
    numvoxelsused = int(np.sum(np.where(themask > 0.0, 1, 0)))
    selectedvoxels = indata[np.where(themask > 0.0), :][0]
    LGR.info(f"constructing global mean signal using {optiondict['globalsignalmethod']}")
    if optiondict["globalsignalmethod"] == "sum":
        globalmean = np.sum(selectedvoxels, axis=0)
    elif optiondict["globalsignalmethod"] == "meanscale":
        themean = np.mean(indata, axis=1)
        for vox in range(0, thesize[0]):
            if themask[vox] > 0.0:
                if themean[vox] != 0.0:
                    globalmean += indata[vox, :] / themean[vox] - 1.0
    elif optiondict["globalsignalmethod"] == "pca":
        try:
            thefit = PCA(n_components=pcacomponents).fit(selectedvoxels)
        except ValueError:
            if pcacomponents == "mle":
                LGR.warning("mle estimation failed - falling back to pcacomponents=0.8")
                thefit = PCA(n_components=0.8).fit(selectedvoxels)
            else:
                raise ValueError("unhandled math exception in PCA refinement - exiting")

        varex = 100.0 * np.cumsum(thefit.explained_variance_ratio_)[len(thefit.components_) - 1]
        LGR.info(
            f"Using {len(thefit.components_)} component(s), accounting for "
            f"{varex:.2f}% of the variance"
        )
    else:
        dummy = optiondict["globalsignalmethod"]
        raise ValueError(f"illegal globalsignalmethod: {dummy}")
    LGR.info(f"used {numvoxelsused} voxels to calculate global mean signal")
    return tide_math.stdnormalize(globalmean), themask


def addmemprofiling(thefunc, memprofile, themessage):
    if memprofile:
        return profile(thefunc, precision=2)
    else:
        tide_util.logmem(themessage)
        return thefunc


def checkforzeromean(thedataset):
    themean = np.mean(thedataset, axis=1)
    thestd = np.std(thedataset, axis=1)
    if np.mean(thestd) > np.mean(themean):
        return True
    else:
        return False


def echocancel(thetimecourse, echooffset, thetimestep, outputname, padtimepoints):
    tide_io.writebidstsv(
        f"{outputname}_desc-echocancellation_timeseries",
        thetimecourse,
        1.0 / thetimestep,
        columns=["original"],
        append=False,
    )
    shifttr = echooffset / thetimestep  # lagtime is in seconds
    echotc, dummy, dummy, dummy = tide_resample.timeshift(thetimecourse, shifttr, padtimepoints)
    echotc[0 : int(np.ceil(shifttr))] = 0.0
    echofit, echoR = tide_fit.mlregress(echotc, thetimecourse)
    fitcoeff = echofit[0, 1]
    outputtimecourse = thetimecourse - fitcoeff * echotc
    tide_io.writebidstsv(
        f"{outputname}_desc-echocancellation_timeseries",
        echotc,
        1.0 / thetimestep,
        columns=["echo"],
        append=True,
    )
    tide_io.writebidstsv(
        f"{outputname}_desc-echocancellation_timeseries",
        outputtimecourse,
        1.0 / thetimestep,
        columns=["filtered"],
        append=True,
    )
    return outputtimecourse, echofit, echoR


def rapidtide_main(argparsingfunc):
    optiondict, theprefilter = argparsingfunc

    optiondict["nodename"] = platform.node()

    # if we are running on AWS, store the instance type
    try:
        optiondict["AWS_instancetype"] = os.environ["AWS_INSTANCETYPE"]
    except KeyError:
        pass

    fmrifilename = optiondict["in_file"]
    outputname = optiondict["outputname"]
    regressorfilename = optiondict["regressorfile"]

    # create the canary file
    Path(f"{outputname}_ISRUNNING.txt").touch()

    # if we are running in a Docker container, make sure we enforce memory limits properly
    try:
        testval = os.environ["IS_DOCKER_8395080871"]
    except KeyError:
        optiondict["runningindocker"] = False
    else:
        optiondict["runningindocker"] = True
        optiondict["dockermemfree"], optiondict["dockermemswap"] = tide_util.findavailablemem()
        if optiondict["dockermemfix"]:
            tide_util.setmemlimit(optiondict["dockermemfree"])

    # Set up loggers for workflow
    setup_logger(
        logger_filename=f"{outputname}_log.txt",
        timing_filename=f"{outputname}_runtimings.tsv",
        memory_filename=f"{outputname}_memusage.tsv",
        verbose=optiondict["verbose"],
        debug=optiondict["debug"],
    )
    TimingLGR.info("Start")

    # print version
    theversion = optiondict["release_version"]
    LGR.info(f"starting rapidtide {theversion}")

    # construct the BIDS base dictionary
    outputpath = os.path.dirname(optiondict["outputname"])
    rawsources = [os.path.relpath(optiondict["in_file"], start=outputpath)]
    if optiondict["regressorfile"] is not None:
        rawsources.append(os.path.relpath(optiondict["regressorfile"], start=outputpath))
    bidsbasedict = {
        "RawSources": rawsources,
        "Units": "arbitrary",
        "CommandLineArgs": optiondict["commandlineargs"],
    }

    TimingLGR.debug("Argument parsing done")

    # don't use shared memory if there is only one process
    if (optiondict["nprocs"] == 1) and not optiondict["alwaysmultiproc"]:
        optiondict["sharedmem"] = False
        LGR.debug("running single process - disabled shared memory use")

    # disable numba now if we're going to do it (before any jits)
    if optiondict["nonumba"]:
        tide_util.disablenumba()

    # set the internal precision
    global rt_floatset, rt_floattype
    if optiondict["internalprecision"] == "double":
        LGR.debug("setting internal precision to double")
        rt_floattype = "float64"
        rt_floatset = np.float64
    else:
        LGR.debug("setting internal precision to single")
        rt_floattype = "float32"
        rt_floatset = np.float32

    # set the output precision
    if optiondict["outputprecision"] == "double":
        LGR.debug("setting output precision to double")
        rt_outfloattype = "float64"
        rt_outfloatset = np.float64
    else:
        LGR.debug("setting output precision to single")
        rt_outfloattype = "float32"
        rt_outfloatset = np.float32

    # set the number of worker processes if multiprocessing
    if optiondict["nprocs"] < 1:
        optiondict["nprocs"] = tide_multiproc.maxcpus(reservecpu=optiondict["reservecpu"])

    if optiondict["singleproc_getNullDist"]:
        optiondict["nprocs_getNullDist"] = 1
    else:
        optiondict["nprocs_getNullDist"] = optiondict["nprocs"]

    if optiondict["singleproc_calcsimilarity"]:
        optiondict["nprocs_calcsimilarity"] = 1
    else:
        optiondict["nprocs_calcsimilarity"] = optiondict["nprocs"]

    if optiondict["singleproc_peakeval"]:
        optiondict["nprocs_peakeval"] = 1
    else:
        optiondict["nprocs_peakeval"] = optiondict["nprocs"]

    if optiondict["singleproc_fitcorr"]:
        optiondict["nprocs_fitcorr"] = 1
    else:
        optiondict["nprocs_fitcorr"] = optiondict["nprocs"]

    if optiondict["singleproc_glm"]:
        optiondict["nprocs_glm"] = 1
    else:
        optiondict["nprocs_glm"] = optiondict["nprocs"]

    # set the number of MKL threads to use
    if mklexists:
        mklmaxthreads = mkl.get_max_threads()
        if not (1 <= optiondict["mklthreads"] <= mklmaxthreads):
            optiondict["mklthreads"] = mklmaxthreads
        mkl.set_num_threads(optiondict["mklthreads"])
        print(f"using {optiondict['mklthreads']} MKL threads")

    # Generate MemoryLGR output file with column names
    if not optiondict["memprofile"]:
        tide_util.logmem()

    # open the fmri datafile
    tide_util.logmem("before reading in fmri data")
    if tide_io.checkiftext(fmrifilename):
        LGR.debug("input file is text - all I/O will be to text files")
        optiondict["textio"] = True
        if optiondict["gausssigma"] > 0.0:
            optiondict["gausssigma"] = 0.0
            LGR.info("gaussian spatial filter disabled for text input files")
    else:
        optiondict["textio"] = False

    if optiondict["textio"]:
        nim_data = tide_io.readvecs(fmrifilename)
        nim_hdr = None
        theshape = np.shape(nim_data)
        xsize = theshape[0]
        ysize = 1
        numslices = 1
        fileiscifti = False
        timepoints = theshape[1]
        thesizes = [0, int(xsize), 1, 1, int(timepoints)]
        numspatiallocs = int(xsize)
    else:
        fileiscifti = tide_io.checkifcifti(fmrifilename)
        if fileiscifti:
            LGR.debug("input file is CIFTI")
            (
                cifti,
                cifti_hdr,
                nim_data,
                nim_hdr,
                thedims,
                thesizes,
                dummy,
            ) = tide_io.readfromcifti(fmrifilename)
            optiondict["isgrayordinate"] = True
            timepoints = nim_data.shape[1]
            numspatiallocs = nim_data.shape[0]
            LGR.debug(f"cifti file has {timepoints} timepoints, {numspatiallocs} numspatiallocs")
            slicesize = numspatiallocs
        else:
            LGR.debug("input file is NIFTI")
            nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
            optiondict["isgrayordinate"] = False
            xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
            numspatiallocs = int(xsize) * int(ysize) * int(numslices)
        xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
    tide_util.logmem("after reading in fmri data")

    # correct some fields if necessary
    if fileiscifti:
        fmritr = 0.72  # this is wrong and is a hack until I can parse CIFTI XML
    else:
        if optiondict["textio"]:
            if optiondict["realtr"] <= 0.0:
                raise ValueError(
                    "for text file data input, you must use the -t option to set the timestep"
                )
        else:
            if nim_hdr.get_xyzt_units()[1] == "msec":
                fmritr = thesizes[4] / 1000.0
            else:
                fmritr = thesizes[4]
    if optiondict["realtr"] > 0.0:
        fmritr = optiondict["realtr"]

    # check to see if we need to adjust the oversample factor
    if optiondict["oversampfactor"] < 0:
        optiondict["oversampfactor"] = int(np.max([np.ceil(fmritr / 0.5), 1]))
        LGR.debug(f"oversample factor set to {optiondict['oversampfactor']}")

    oversamptr = fmritr / optiondict["oversampfactor"]
    LGR.verbose(f"fmri data: {timepoints} timepoints, tr = {fmritr}, oversamptr = {oversamptr}")
    LGR.info(f"{numspatiallocs} spatial locations, {timepoints} timepoints")
    TimingLGR.info("Finish reading fmrifile")

    # if the user has specified start and stop points, limit check, then use these numbers
    validstart, validend = tide_util.startendcheck(
        timepoints, optiondict["startpoint"], optiondict["endpoint"]
    )
    if abs(optiondict["lagmin"]) > (validend - validstart + 1) * fmritr / 2.0:
        raise ValueError(
            f"magnitude of lagmin exceeds {(validend - validstart + 1) * fmritr / 2.0} - invalid"
        )

    if abs(optiondict["lagmax"]) > (validend - validstart + 1) * fmritr / 2.0:
        raise ValueError(
            f"magnitude of lagmax exceeds {(validend - validstart + 1) * fmritr / 2.0} - invalid"
        )

    # do spatial filtering if requested
    if optiondict["gausssigma"] < 0.0 and not optiondict["textio"]:
        # set gausssigma automatically
        optiondict["gausssigma"] = np.mean([xdim, ydim, slicethickness]) / 2.0
    if optiondict["gausssigma"] > 0.0:
        LGR.info(
            f"applying gaussian spatial filter to timepoints {validstart} "
            f"to {validend} with sigma={optiondict['gausssigma']}"
        )
        for i in tqdm(
            range(validstart, validend + 1),
            desc="Timpoint",
            unit="timepoints",
            disable=(not optiondict["showprogressbar"]),
        ):
            nim_data[:, :, :, i] = tide_filt.ssmooth(
                xdim,
                ydim,
                slicethickness,
                optiondict["gausssigma"],
                nim_data[:, :, :, i],
            )
        print()
        TimingLGR.info("End 3D smoothing")

    # reshape the data and trim to a time range, if specified.  Check for special case of no trimming to save RAM
    fmri_data = nim_data.reshape((numspatiallocs, timepoints))[:, validstart : validend + 1]
    validtimepoints = validend - validstart + 1

    # detect zero mean data
    optiondict["dataiszeromean"] = checkforzeromean(fmri_data)
    if optiondict["dataiszeromean"]:
        LGR.warning(
            "WARNING: dataset is zero mean - forcing variance masking and no refine prenormalization. "
            "Consider specifying a global mean and correlation mask."
        )
        optiondict["refineprenorm"] = "None"
        optiondict["globalmaskmethod"] = "variance"

    # read in the optional masks
    tide_util.logmem("before setting masks")

    internalglobalmeanincludemask, internalglobalmeanexcludemask, dummy = tide_mask.getmaskset(
        "global mean",
        optiondict["globalmeanincludename"],
        optiondict["globalmeanincludevals"],
        optiondict["globalmeanexcludename"],
        optiondict["globalmeanexcludevals"],
        nim_hdr,
        numspatiallocs,
        istext=optiondict["textio"],
        tolerance=optiondict["spatialtolerance"],
    )

    internalrefineincludemask, internalrefineexcludemask, dummy = tide_mask.getmaskset(
        "refine",
        optiondict["refineincludename"],
        optiondict["refineincludevals"],
        optiondict["refineexcludename"],
        optiondict["refineexcludevals"],
        nim_hdr,
        numspatiallocs,
        istext=optiondict["textio"],
        tolerance=optiondict["spatialtolerance"],
    )

    internaloffsetincludemask, internaloffsetexcludemask, dummy = tide_mask.getmaskset(
        "offset",
        optiondict["offsetincludename"],
        optiondict["offsetincludevals"],
        optiondict["offsetexcludename"],
        optiondict["offsetexcludevals"],
        nim_hdr,
        numspatiallocs,
        istext=optiondict["textio"],
        tolerance=optiondict["spatialtolerance"],
    )

    tide_util.logmem("after setting masks")

    # read or make a mask of where to calculate the correlations
    tide_util.logmem("before selecting valid voxels")
    threshval = tide_stats.getfracvals(fmri_data[:, :], [0.98])[0] / 25.0
    LGR.debug("constructing correlation mask")
    if optiondict["corrmaskincludename"] is not None:
        thecorrmask = tide_mask.readamask(
            optiondict["corrmaskincludename"],
            nim_hdr,
            xsize,
            istext=optiondict["textio"],
            valslist=optiondict["corrmaskincludevals"],
            maskname="correlation",
            tolerance=optiondict["spatialtolerance"],
        )

        corrmask = np.uint16(np.where(thecorrmask > 0, 1, 0).reshape(numspatiallocs))
    else:
        # check to see if the data has been demeaned
        meanim = np.mean(fmri_data, axis=1)
        stdim = np.std(fmri_data, axis=1)
        if fileiscifti:
            corrmask = np.uint(nim_data[:, 0] * 0 + 1)
        else:
            if (np.mean(stdim) < np.mean(meanim)) and not optiondict["nirs"]:
                LGR.verbose("generating correlation mask from mean image")
                corrmask = np.uint16(tide_mask.makeepimask(nim).dataobj.reshape(numspatiallocs))
            else:
                LGR.verbose("generating correlation mask from std image")
                corrmask = np.uint16(
                    tide_stats.makemask(stdim, threshpct=optiondict["corrmaskthreshpct"])
                )
    if tide_stats.getmasksize(corrmask) == 0:
        raise ValueError("ERROR: there are no voxels in the correlation mask - exiting")

    optiondict["corrmasksize"] = tide_stats.getmasksize(corrmask)
    if internalrefineincludemask is not None:
        if internalrefineexcludemask is not None:
            if (
                tide_stats.getmasksize(
                    corrmask * internalrefineincludemask * (1 - internalrefineexcludemask)
                )
                == 0
            ):
                raise ValueError(
                    "ERROR: the refine include and exclude masks not leave any voxels in the corrmask - exiting"
                )
        else:
            if tide_stats.getmasksize(corrmask * internalrefineincludemask) == 0:
                raise ValueError(
                    "ERROR: the refine include mask does not leave any voxels in the corrmask - exiting"
                )
    else:
        if internalrefineexcludemask is not None:
            if tide_stats.getmasksize(corrmask * (1 - internalrefineexcludemask)) == 0:
                raise ValueError(
                    "ERROR: the refine exclude mask does not leave any voxels in the corrmask - exiting"
                )

    if optiondict["nothresh"]:
        corrmask *= 0
        corrmask += 1
        threshval = -10000000.0
    if optiondict["savecorrmask"] and not (fileiscifti or optiondict["textio"]):
        theheader = copy.deepcopy(nim_hdr)
        theheader["dim"][0] = 3
        theheader["dim"][4] = 1
        theheader["pixdim"][4] = 1.0
        savename = f"{outputname}_desc-processed_mask"
        tide_io.savetonifti(corrmask.reshape(xsize, ysize, numslices), theheader, savename)

    LGR.verbose(f"image threshval = {threshval}")
    validvoxels = np.where(corrmask > 0)[0]
    numvalidspatiallocs = np.shape(validvoxels)[0]
    LGR.debug(f"validvoxels shape = {numvalidspatiallocs}")
    fmri_data_valid = fmri_data[validvoxels, :] + 0.0
    LGR.verbose(
        f"original size = {np.shape(fmri_data)}, trimmed size = {np.shape(fmri_data_valid)}"
    )

    if internalrefineincludemask is not None:
        internalrefineincludemask_valid = 1.0 * internalrefineincludemask[validvoxels]
        del internalrefineincludemask
        LGR.debug(
            "internalrefineincludemask_valid has size: " f"{internalrefineincludemask_valid.size}"
        )
    else:
        internalrefineincludemask_valid = None
    if internalrefineexcludemask is not None:
        internalrefineexcludemask_valid = 1.0 * internalrefineexcludemask[validvoxels]
        del internalrefineexcludemask
        LGR.debug(
            "internalrefineexcludemask_valid has size: " f"{internalrefineexcludemask_valid.size}"
        )
    else:
        internalrefineexcludemask_valid = None

    if internaloffsetincludemask is not None:
        internaloffsetincludemask_valid = 1.0 * internaloffsetincludemask[validvoxels]
        del internaloffsetincludemask
        LGR.debug(
            "internaloffsetincludemask_valid has size: " f"{internaloffsetincludemask_valid.size}"
        )
    else:
        internaloffsetincludemask_valid = None
    if internaloffsetexcludemask is not None:
        internaloffsetexcludemask_valid = 1.0 * internaloffsetexcludemask[validvoxels]
        del internaloffsetexcludemask
        LGR.debug(
            "internaloffsetexcludemask_valid has size: " f"{internaloffsetexcludemask_valid.size}"
        )
    else:
        internaloffsetexcludemask_valid = None

    tide_util.logmem("after selecting valid voxels")

    # calculate the initial bandlimited variance if we're going to filter the data
    if optiondict["doglmfilt"] or optiondict["docvrmap"]:
        initialvariance = tide_math.imagevariance(fmri_data_valid, theprefilter, 1.0 / fmritr)

    # move fmri_data_valid into shared memory
    if optiondict["sharedmem"]:
        LGR.info("moving fmri data to shared memory")
        TimingLGR.verbose("Start moving fmri_data to shared memory")
        numpy2shared_func = addmemprofiling(
            numpy2shared, optiondict["memprofile"], "before fmri data move"
        )
        fmri_data_valid = numpy2shared_func(fmri_data_valid, rt_floatset)
        TimingLGR.verbose("End moving fmri_data to shared memory")

    # get rid of memory we aren't using
    tide_util.logmem("before purging full sized fmri data")
    meanvalue = np.mean(
        nim_data.reshape((numspatiallocs, timepoints))[:, validstart : validend + 1],
        axis=1,
    )

    # calculate the global mean whether we intend to use it or not, before deleting full fmri_data array
    meanfreq = 1.0 / fmritr
    meanperiod = 1.0 * fmritr
    meanstarttime = 0.0
    meanvec, meanmask = getglobalsignal(
        fmri_data,
        optiondict,
        includemask=internalglobalmeanincludemask,
        excludemask=internalglobalmeanexcludemask,
        pcacomponents=optiondict["globalpcacomponents"],
    )

    del fmri_data
    del nim_data
    gc.collect()
    tide_util.logmem("after purging full sized fmri data")

    # filter out motion regressors here
    if optiondict["motionfilename"] is not None:
        LGR.info("regressing out motion")

        TimingLGR.verbose("Motion filtering start")
        (
            motionregressors,
            motionregressorlabels,
            fmri_data_valid,
        ) = tide_glmpass.motionregress(
            optiondict["motionfilename"],
            fmri_data_valid,
            fmritr,
            motstart=validstart,
            motend=validend + 1,
            position=optiondict["mot_pos"],
            deriv=optiondict["mot_deriv"],
            derivdelayed=optiondict["mot_delayderiv"],
        )

        TimingLGR.info(
            "Motion filtering end",
            {
                "message2": fmri_data_valid.shape[0],
                "message3": "voxels",
            },
        )
        tide_io.writebidstsv(
            f"{outputname}_desc-orthogonalizedmotion_timeseries",
            motionregressors,
            1.0 / fmritr,
            columns=motionregressorlabels,
            append=True,
        )
        if optiondict["memprofile"]:
            memcheckpoint("...done")
        else:
            tide_util.logmem("after motion glm filter")

        if optiondict["savemotionfiltered"]:
            outfmriarray = np.zeros((numspatiallocs, validtimepoints), dtype=rt_floattype)
            outfmriarray[validvoxels, :] = fmri_data_valid[:, :]
            if optiondict["textio"]:
                tide_io.writenpvecs(
                    outfmriarray.reshape((numspatiallocs, validtimepoints)),
                    f"{outputname}_motionfiltered.txt",
                )
            else:
                savename = f"{outputname}_desc-motionfiltered"
                tide_io.savetonifti(
                    outfmriarray.reshape((xsize, ysize, numslices, validtimepoints)),
                    nim_hdr,
                    savename,
                )

    # read in the timecourse to resample
    TimingLGR.info("Start of reference prep")
    if regressorfilename is None:
        LGR.info("no regressor file specified - will use the global mean regressor")
        optiondict["useglobalref"] = True
    else:
        optiondict["useglobalref"] = False

    # now set the regressor that we'll use
    if optiondict["useglobalref"]:
        LGR.verbose("using global mean as probe regressor")
        inputfreq = meanfreq
        inputperiod = meanperiod
        inputstarttime = meanstarttime
        inputvec = meanvec
        savename = f"{outputname}_desc-globalmean_mask"
        if fileiscifti:
            theheader = copy.deepcopy(nim_hdr)
            timeindex = theheader["dim"][0] - 1
            spaceindex = theheader["dim"][0]
            theheader["dim"][timeindex] = 1
            theheader["dim"][spaceindex] = numspatiallocs
            tide_io.savetocifti(
                meanmask,
                cifti_hdr,
                theheader,
                savename,
                isseries=False,
                names=["meanmask"],
            )
        elif optiondict["textio"]:
            tide_io.writenpvecs(
                meanmask,
                savename + ".txt",
            )
        else:
            theheader = copy.deepcopy(nim_hdr)
            theheader["dim"][0] = 3
            theheader["dim"][4] = 1
            theheader["pixdim"][4] = 1.0
            tide_io.savetonifti(meanmask.reshape((xsize, ysize, numslices)), theheader, savename)

        optiondict["preprocskip"] = 0
    else:
        LGR.info(f"using externally supplied probe regressor {regressorfilename}")
        (
            fileinputfreq,
            filestarttime,
            dummy,
            inputvec,
            dummy,
            dummy,
        ) = tide_io.readvectorsfromtextfile(regressorfilename, onecol=True)
        inputfreq = optiondict["inputfreq"]
        inputstarttime = optiondict["inputstarttime"]
        if inputfreq is None:
            if fileinputfreq is not None:
                inputfreq = fileinputfreq
            else:
                inputfreq = 1.0 / fmritr
            LGR.warning(f"no regressor frequency specified - defaulting to {inputfreq} (1/tr)")
        if inputstarttime is None:
            if filestarttime is not None:
                inputstarttime = filestarttime
            else:
                LGR.warning("no regressor start time specified - defaulting to 0.0")
                inputstarttime = 0.0
        inputperiod = 1.0 / inputfreq
    numreference = len(inputvec)
    optiondict["inputfreq"] = inputfreq
    optiondict["inputstarttime"] = inputstarttime
    LGR.debug(
        "Regressor start time, end time, and step: {:.3f}, {:.3f}, {:.3f}".format(
            -inputstarttime, inputstarttime + numreference * inputperiod, inputperiod
        )
    )
    LGR.verbose("Input vector")
    LGR.verbose(f"length: {len(inputvec)}")
    LGR.verbose(f"input freq: {inputfreq}")
    LGR.verbose(f"input start time: {inputstarttime:.3f}")

    if not optiondict["useglobalref"]:
        globalcorrx, globalcorry, dummy, dummy = tide_corr.arbcorr(
            meanvec, meanfreq, inputvec, inputfreq, start2=inputstarttime
        )
        synctime = globalcorrx[np.argmax(globalcorry)]
        if optiondict["autosync"]:
            optiondict["offsettime"] = -synctime
            optiondict["offsettime_total"] = synctime
    else:
        synctime = 0.0
    LGR.info(f"synctime is {synctime}")

    reference_x = np.arange(0.0, numreference) * inputperiod - (
        inputstarttime - optiondict["offsettime"]
    )
    LGR.info(f"total probe regressor offset is {inputstarttime + optiondict['offsettime']}")

    # Print out initial information
    LGR.verbose(f"there are {numreference} points in the original regressor")
    LGR.verbose(f"the timepoint spacing is {1.0 / inputfreq}")
    LGR.verbose(f"the input timecourse start time is {inputstarttime}")

    # if there is an externally specified noise regressor, read it in here
    if optiondict["noisetimecoursespec"] is not None:
        noisetimecoursespec = optiondict["noisetimecoursespec"]
        LGR.info(f"using externally supplied noise regressor {noisetimecoursespec}")
        (
            filenoisefreq,
            filenoisestarttime,
            dummy,
            noisevec,
            dummy,
            dummy,
        ) = tide_io.readvectorsfromtextfile(optiondict["noisetimecoursespec"], onecol=True)
        if optiondict["noiseinvert"]:
            noisevec = noisevec * -1.0
        noisefreq = optiondict["noisefreq"]
        noisestarttime = optiondict["noisestarttime"]
        if noisefreq is None:
            if filenoisefreq is not None:
                noisefreq = filenoisefreq
            else:
                noisefreq = 1.0 / fmritr
            LGR.warning(f"no regressor frequency specified - defaulting to {noisefreq} (1/tr)")
        if noisestarttime is None:
            if filenoisestarttime is not None:
                noisestarttime = filenoisestarttime
            else:
                LGR.warning("no regressor start time specified - defaulting to 0.0")
                noisestarttime = 0.0
        noiseperiod = 1.0 / noisefreq
        numnoise = len(noisevec)
        optiondict["noisefreq"] = noisefreq
        optiondict["noisestarttime"] = noisestarttime
        LGR.debug(
            "Noise timecourse start time, end time, and step: {:.3f}, {:.3f}, {:.3f}".format(
                -noisestarttime, noisestarttime + numnoise * noiseperiod, noiseperiod
            )
        )
        noise_x = np.arange(0.0, numnoise) * noiseperiod - noisestarttime
        noise_y = noisevec[0:numnoise] - np.mean(noisevec[0:numnoise])
        # write out the noise regressor as read
        tide_io.writebidstsv(
            f"{outputname}_desc-initialnoiseregressor_timeseries",
            noise_y,
            noisefreq,
            starttime=-noisestarttime,
            columns=["prefilt"],
            append=False,
        )
        LGR.verbose("noise vector")
        LGR.verbose(f"length: {len(noisevec)}")
        LGR.verbose(f"noise freq: {noisefreq}")
        LGR.verbose(f"noise start time: {noisestarttime:.3f}")

    # generate the time axes
    fmrifreq = 1.0 / fmritr
    optiondict["fmrifreq"] = fmrifreq
    skiptime = fmritr * (optiondict["preprocskip"])
    LGR.debug(f"first fMRI point is at {skiptime} seconds relative to time origin")
    initial_fmri_x = np.arange(0.0, validtimepoints) * fmritr + skiptime
    os_fmri_x = (
        np.arange(
            0.0,
            validtimepoints * optiondict["oversampfactor"] - (optiondict["oversampfactor"] - 1),
        )
        * oversamptr
        + skiptime
    )

    LGR.verbose(f"os_fmri_x dim-0 shape: {np.shape(os_fmri_x)[0]}")
    LGR.verbose(f"initial_fmri_x dim-0 shape: {np.shape(initial_fmri_x)[0]}")

    # generate the comparison regressor from the input timecourse
    # correct the output time points
    # check for extrapolation
    if os_fmri_x[0] < reference_x[0]:
        LGR.warning(
            f"WARNING: extrapolating {os_fmri_x[0] - reference_x[0]} "
            "seconds of data at beginning of timecourse"
        )
    if os_fmri_x[-1] > reference_x[-1]:
        LGR.warning(
            f"WARNING: extrapolating {os_fmri_x[-1] - reference_x[-1]} "
            "seconds of data at end of timecourse"
        )

    # invert the regressor if necessary
    if optiondict["invertregressor"]:
        invertfac = -1.0
    else:
        invertfac = 1.0

    # detrend the regressor if necessary
    if optiondict["detrendorder"] > 0:
        reference_y = invertfac * tide_fit.detrend(
            inputvec[0:numreference],
            order=optiondict["detrendorder"],
            demean=optiondict["dodemean"],
        )
    else:
        reference_y = invertfac * (inputvec[0:numreference] - np.mean(inputvec[0:numreference]))

    # write out the reference regressor prior to filtering
    tide_io.writebidstsv(
        f"{outputname}_desc-initialmovingregressor_timeseries",
        reference_y,
        inputfreq,
        starttime=-inputstarttime,
        columns=["prefilt"],
        append=False,
    )

    # band limit the regressor if that is needed
    LGR.info(f"filtering to {theprefilter.gettype()} band")
    (
        optiondict["lowerstop"],
        optiondict["lowerpass"],
        optiondict["upperpass"],
        optiondict["upperstop"],
    ) = theprefilter.getfreqs()
    reference_y_classfilter = theprefilter.apply(inputfreq, reference_y)
    if optiondict["negativegradregressor"]:
        reference_y = -np.gradient(reference_y_classfilter)
    else:
        reference_y = reference_y_classfilter
    if optiondict["noisetimecoursespec"] is not None:
        noise_y = theprefilter.apply(noisefreq, noise_y)

    # write out the reference regressor used
    tide_io.writebidstsv(
        f"{outputname}_desc-initialmovingregressor_timeseries",
        tide_math.stdnormalize(reference_y),
        inputfreq,
        starttime=-inputstarttime,
        columns=["postfilt"],
        append=True,
    )

    # filter the input data for antialiasing
    if optiondict["antialias"]:
        LGR.debug("applying trapezoidal antialiasing filter")
        reference_y_filt = tide_filt.dolptrapfftfilt(
            inputfreq,
            0.25 * fmrifreq,
            0.5 * fmrifreq,
            reference_y,
            padlen=int(inputfreq * optiondict["padseconds"]),
            debug=optiondict["debug"],
        )
        reference_y = rt_floatset(reference_y_filt.real)
        if optiondict["noisetimecoursespec"] is not None:
            noise_y_filt = tide_filt.dolptrapfftfilt(
                noisefreq,
                0.25 * fmrifreq,
                0.5 * fmrifreq,
                noise_y,
                padlen=int(noisefreq * optiondict["padseconds"]),
                debug=optiondict["debug"],
            )
            noise_y = rt_floatset(noise_y_filt.real)

            # write out the noise regressor after filtering
            tide_io.writebidstsv(
                f"{outputname}_desc-initialnoiseregressor_timeseries",
                noise_y,
                noisefreq,
                starttime=-noisestarttime,
                columns=["postfilt"],
                append=True,
            )

    warnings.filterwarnings("ignore", "Casting*")

    if optiondict["fakerun"]:
        return

    # generate the resampled reference regressors
    oversampfreq = optiondict["oversampfactor"] / fmritr
    if optiondict["detrendorder"] > 0:
        resampnonosref_y = tide_fit.detrend(
            tide_resample.doresample(
                reference_x,
                reference_y,
                initial_fmri_x,
                padlen=int((1.0 / fmritr) * optiondict["padseconds"]),
                method=optiondict["interptype"],
                debug=optiondict["debug"],
            ),
            order=optiondict["detrendorder"],
            demean=optiondict["dodemean"],
        )
        resampref_y = tide_fit.detrend(
            tide_resample.doresample(
                reference_x,
                reference_y,
                os_fmri_x,
                padlen=int(oversampfreq * optiondict["padseconds"]),
                method=optiondict["interptype"],
                debug=optiondict["debug"],
            ),
            order=optiondict["detrendorder"],
            demean=optiondict["dodemean"],
        )
        if optiondict["noisetimecoursespec"] is not None:
            if optiondict["detrendorder"] > 0:
                resampnonosnoise_y = tide_fit.detrend(
                    tide_resample.doresample(
                        noise_x,
                        noise_y,
                        initial_fmri_x,
                        padlen=int((1.0 / fmritr) * optiondict["padseconds"]),
                        padtype="zero",
                        method=optiondict["interptype"],
                        debug=optiondict["debug"],
                    ),
                    order=optiondict["detrendorder"],
                    demean=optiondict["dodemean"],
                )
                resampnoise_y = tide_fit.detrend(
                    tide_resample.doresample(
                        noise_x,
                        noise_y,
                        os_fmri_x,
                        padlen=int(oversampfreq * optiondict["padseconds"]),
                        padtype="zero",
                        method=optiondict["interptype"],
                        debug=optiondict["debug"],
                    ),
                    order=optiondict["detrendorder"],
                    demean=optiondict["dodemean"],
                )

    else:
        resampnonosref_y = tide_resample.doresample(
            reference_x,
            reference_y,
            initial_fmri_x,
            padlen=int((1.0 / fmritr) * optiondict["padseconds"]),
            method=optiondict["interptype"],
        )
        resampref_y = tide_resample.doresample(
            reference_x,
            reference_y,
            os_fmri_x,
            padlen=int(oversampfreq * optiondict["padseconds"]),
            method=optiondict["interptype"],
        )
        if optiondict["noisetimecoursespec"] is not None:
            resampnonosnoise_y = tide_resample.doresample(
                noise_x,
                noise_y,
                initial_fmri_x,
                padlen=int((1.0 / fmritr) * optiondict["padseconds"]),
                padtype="zero",
                method=optiondict["interptype"],
            )
            resampnoise_y = tide_resample.doresample(
                noise_x,
                noise_y,
                os_fmri_x,
                padlen=int(oversampfreq * optiondict["padseconds"]),
                padtype="zero",
                method=optiondict["interptype"],
            )

    LGR.debug(
        f"{len(os_fmri_x)} "
        f"{len(resampref_y)} "
        f"{len(initial_fmri_x)} "
        f"{len(resampnonosref_y)}"
    )
    previousnormoutputdata = resampnonosref_y + 0.0

    # save the factor used to normalize the input regressor
    optiondict["initialmovingregressornormfac"] = np.std(resampnonosref_y)

    # prepare the temporal mask
    if optiondict["tmaskname"] is not None:
        tmask_y = tide_mask.maketmask(
            optiondict["tmaskname"], reference_x, rt_floatset(reference_y)
        )
        tmaskos_y = tide_resample.doresample(
            reference_x, tmask_y, os_fmri_x, method=optiondict["interptype"]
        )
        # bidsify
        tide_io.writebidstsv(
            f"{outputname}_desc-_temporalmask_timeseries",
            tmask_y,
            1.0 / oversamptr,
            starttime=0.0,
            columns=["initial"],
            append=False,
        )
        # tide_io.writenpvecs(tmask_y, f"{outputname}_temporalmask.txt")
        resampnonosref_y *= tmask_y
        thefit, R = tide_fit.mlregress(tmask_y, resampnonosref_y)
        resampnonosref_y -= thefit[0, 1] * tmask_y
        resampref_y *= tmaskos_y
        thefit, R = tide_fit.mlregress(tmaskos_y, resampref_y)
        resampref_y -= thefit[0, 1] * tmaskos_y

    if optiondict["noisetimecoursespec"] is not None:
        tide_io.writebidstsv(
            f"{outputname}_desc-noiseregressor_timeseries",
            tide_math.stdnormalize(resampnonosref_y),
            1.0 / fmritr,
            columns=["resamplled"],
            append=False,
        )
        tide_io.writebidstsv(
            f"{outputname}_desc-oversamplednoiseregressor_timeseries",
            tide_math.stdnormalize(resampref_y),
            oversampfreq,
            columns=["oversampled"],
            append=False,
        )

    (
        optiondict["kurtosis_reference_pass1"],
        optiondict["kurtosisz_reference_pass1"],
        optiondict["kurtosisp_reference_pass1"],
    ) = tide_stats.kurtosisstats(resampref_y)
    tide_io.writebidstsv(
        f"{outputname}_desc-movingregressor_timeseries",
        tide_math.stdnormalize(resampnonosref_y),
        1.0 / fmritr,
        columns=["pass1"],
        append=False,
    )
    tide_io.writebidstsv(
        f"{outputname}_desc-oversampledmovingregressor_timeseries",
        tide_math.stdnormalize(resampref_y),
        oversampfreq,
        columns=["pass1"],
        append=False,
    )
    TimingLGR.info("End of reference prep")

    corrtr = oversamptr
    LGR.verbose(f"corrtr={corrtr}")

    # initialize the Correlator
    theCorrelator = tide_classes.Correlator(
        Fs=oversampfreq,
        ncprefilter=theprefilter,
        negativegradient=optiondict["negativegradient"],
        detrendorder=optiondict["detrendorder"],
        windowfunc=optiondict["windowfunc"],
        corrweighting=optiondict["corrweighting"],
        corrpadding=optiondict["corrpadding"],
        debug=optiondict["debug"],
    )
    theCorrelator.setreftc(
        np.zeros((optiondict["oversampfactor"] * validtimepoints), dtype=np.float64)
    )
    corrorigin = theCorrelator.similarityfuncorigin
    dummy, corrscale, dummy = theCorrelator.getfunction(trim=False)

    lagmininpts = int((-optiondict["lagmin"] / corrtr) - 0.5)
    lagmaxinpts = int((optiondict["lagmax"] / corrtr) + 0.5)

    if (lagmaxinpts + lagmininpts) < 3:
        raise ValueError(
            "correlation search range is too narrow - decrease lagmin, increase lagmax, or increase oversample factor"
        )

    theCorrelator.setlimits(lagmininpts, lagmaxinpts)
    dummy, trimmedcorrscale, dummy = theCorrelator.getfunction()

    # initialize the MutualInformationator
    theMutualInformationator = tide_classes.MutualInformationator(
        Fs=oversampfreq,
        smoothingtime=optiondict["smoothingtime"],
        ncprefilter=theprefilter,
        negativegradient=optiondict["negativegradient"],
        detrendorder=optiondict["detrendorder"],
        windowfunc=optiondict["windowfunc"],
        madnorm=False,
        lagmininpts=lagmininpts,
        lagmaxinpts=lagmaxinpts,
        debug=optiondict["debug"],
    )
    theMutualInformationator.setreftc(
        np.zeros((optiondict["oversampfactor"] * validtimepoints), dtype=np.float64)
    )
    nummilags = theMutualInformationator.similarityfunclen
    theMutualInformationator.setlimits(lagmininpts, lagmaxinpts)
    dummy, trimmedmiscale, dummy = theMutualInformationator.getfunction()

    LGR.verbose(f"trimmedcorrscale length: {len(trimmedcorrscale)}")
    LGR.verbose(f"trimmedmiscale length: {len(trimmedmiscale)} {nummilags}")
    LGR.verbose(f"corrorigin at point {corrorigin} {corrscale[corrorigin]}")
    LGR.verbose(
        f"corr range from {corrorigin - lagmininpts} ({corrscale[corrorigin - lagmininpts]}) "
        f"to {corrorigin + lagmaxinpts} ({corrscale[corrorigin + lagmaxinpts]})"
    )

    if optiondict["savecorrtimes"]:
        # bidsify
        """tide_io.writebidstsv(
            f"{outputname}_desc-corrtimes_timeseries",
            trimmedcorrscale,
            1.0,
            starttime=0.0,
            columns=["initial"],
            append=False,
        )
        tide_io.writebidstsv(
            f"{outputname}_desc-mitimes_timeseries",
            trimmedmiscale,
            1.0,
            starttime=0.0,
            columns=["initial"],
            append=False,
        )"""
        tide_io.writenpvecs(trimmedcorrscale, f"{outputname}_corrtimes.txt")
        tide_io.writenpvecs(trimmedmiscale, f"{outputname}_mitimes.txt")

    # allocate all the data arrays
    tide_util.logmem("before main array allocation")
    if optiondict["textio"]:
        nativespaceshape = xsize
    else:
        if fileiscifti:
            nativespaceshape = (1, 1, 1, 1, numspatiallocs)
        else:
            nativespaceshape = (xsize, ysize, numslices)
    internalspaceshape = numspatiallocs
    internalvalidspaceshape = numvalidspatiallocs
    meanval = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagtimes = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagstrengths = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagsigma = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    fitmask = np.zeros(internalvalidspaceshape, dtype="uint16")
    failreason = np.zeros(internalvalidspaceshape, dtype="uint32")
    R2 = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    outmaparray = np.zeros(internalspaceshape, dtype=rt_floattype)
    tide_util.logmem("after main array allocation")

    corroutlen = np.shape(trimmedcorrscale)[0]
    if optiondict["textio"]:
        nativecorrshape = (xsize, corroutlen)
    else:
        if fileiscifti:
            nativecorrshape = (1, 1, 1, corroutlen, numspatiallocs)
        else:
            nativecorrshape = (xsize, ysize, numslices, corroutlen)
    internalcorrshape = (numspatiallocs, corroutlen)
    internalvalidcorrshape = (numvalidspatiallocs, corroutlen)
    LGR.debug(
        f"allocating memory for correlation arrays {internalcorrshape} {internalvalidcorrshape}"
    )
    if optiondict["sharedmem"]:
        corrout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        gaussout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        windowout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        outcorrarray, dummy, dummy = allocshared(internalcorrshape, rt_floatset)
    else:
        corrout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        gaussout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        windowout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        outcorrarray = np.zeros(internalcorrshape, dtype=rt_floattype)
    tide_util.logmem("after correlation array allocation")

    if optiondict["textio"]:
        nativefmrishape = (xsize, np.shape(initial_fmri_x)[0])
    else:
        if fileiscifti:
            nativefmrishape = (1, 1, 1, np.shape(initial_fmri_x)[0], numspatiallocs)
        else:
            nativefmrishape = (xsize, ysize, numslices, np.shape(initial_fmri_x)[0])
    internalfmrishape = (numspatiallocs, np.shape(initial_fmri_x)[0])
    internalvalidfmrishape = (numvalidspatiallocs, np.shape(initial_fmri_x)[0])
    if optiondict["sharedmem"]:
        lagtc, dummy, dummy = allocshared(internalvalidfmrishape, rt_floatset)
    else:
        lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    tide_util.logmem("after lagtc array allocation")

    if (
        optiondict["passes"] > 1
        or optiondict["globalpreselect"]
        or optiondict["convergencethresh"] is not None
    ):
        if optiondict["sharedmem"]:
            shiftedtcs, dummy, dummy = allocshared(internalvalidfmrishape, rt_floatset)
            weights, dummy, dummy = allocshared(internalvalidfmrishape, rt_floatset)
        else:
            shiftedtcs = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
            weights = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
        tide_util.logmem("after refinement array allocation")
    if optiondict["sharedmem"]:
        outfmriarray, dummy, dummy = allocshared(internalfmrishape, rt_floatset)
    else:
        outfmriarray = np.zeros(internalfmrishape, dtype=rt_floattype)

    # prepare for fast resampling
    padtime = (
        max((-optiondict["lagmin"], optiondict["lagmax"]))
        + 30.0
        + np.abs(optiondict["offsettime"])
    )
    LGR.info(f"setting up fast resampling with padtime = {padtime}")
    numpadtrs = int(padtime // fmritr)
    padtime = fmritr * numpadtrs
    genlagtc = tide_resample.FastResampler(reference_x, reference_y, padtime=padtime)

    # cycle over all voxels
    refine = True
    LGR.verbose(f"refine is set to {refine}")
    optiondict["edgebufferfrac"] = max(
        [optiondict["edgebufferfrac"], 2.0 / np.shape(corrscale)[0]]
    )
    LGR.verbose(f"edgebufferfrac set to {optiondict['edgebufferfrac']}")

    # intitialize the correlation fitter
    thefitter = tide_classes.SimilarityFunctionFitter(
        lagmod=optiondict["lagmod"],
        lthreshval=optiondict["lthreshval"],
        uthreshval=optiondict["uthreshval"],
        bipolar=optiondict["bipolar"],
        lagmin=optiondict["lagmin"],
        lagmax=optiondict["lagmax"],
        absmaxsigma=optiondict["absmaxsigma"],
        absminsigma=optiondict["absminsigma"],
        debug=optiondict["debug"],
        peakfittype=optiondict["peakfittype"],
        searchfrac=optiondict["searchfrac"],
        enforcethresh=optiondict["enforcethresh"],
        hardlimit=optiondict["hardlimit"],
        zerooutbadfit=optiondict["zerooutbadfit"],
    )

    # Preprocessing - echo cancellation
    if optiondict["echocancel"]:
        LGR.info("\n\nEcho cancellation")
        TimingLGR.info("Echo cancellation start")
        calcsimilaritypass_func = addmemprofiling(
            tide_calcsimfunc.correlationpass,
            optiondict["memprofile"],
            "before correlationpass",
        )

        referencetc = tide_math.corrnormalize(
            resampref_y,
            detrendorder=optiondict["detrendorder"],
            windowfunc=optiondict["windowfunc"],
        )

        (
            voxelsprocessed_echo,
            theglobalmaxlist,
            trimmedcorrscale,
        ) = calcsimilaritypass_func(
            fmri_data_valid[:, :],
            referencetc,
            theCorrelator,
            initial_fmri_x,
            os_fmri_x,
            lagmininpts,
            lagmaxinpts,
            corrout,
            meanval,
            nprocs=optiondict["nprocs_calcsimilarity"],
            alwaysmultiproc=optiondict["alwaysmultiproc"],
            oversampfactor=optiondict["oversampfactor"],
            interptype=optiondict["interptype"],
            showprogressbar=optiondict["showprogressbar"],
            chunksize=optiondict["mp_chunksize"],
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
        )
        for i in range(len(theglobalmaxlist)):
            theglobalmaxlist[i] = corrscale[theglobalmaxlist[i]]
        namesuffix = "_desc-globallag_hist"
        tide_stats.makeandsavehistogram(
            np.asarray(theglobalmaxlist),
            len(corrscale),
            0,
            outputname + namesuffix,
            displaytitle="lagtime histogram",
            therange=(corrscale[0], corrscale[-1]),
            refine=False,
            dictvarname="globallaghist_preechocancel",
            append=False,
            thedict=optiondict,
        )

        # Now find and regress out the echo
        echooffset, echoratio = tide_stats.echoloc(np.asarray(theglobalmaxlist), len(corrscale))
        LGR.info(f"Echooffset, echoratio: {echooffset} {echoratio}")
        echoremovedtc, echofit, echoR = echocancel(
            resampref_y, echooffset, oversamptr, outputname, numpadtrs
        )
        optiondict["echooffset"] = echooffset
        optiondict["echoratio"] = echoratio
        optiondict["echofit"] = [echofit[0, 0], echofit[0, 1]]
        optiondict["echofitR"] = echoR
        resampref_y = echoremovedtc
        TimingLGR.info(
            "Echo cancellation calculation end",
            {
                "message2": voxelsprocessed_echo,
                "message3": "voxels",
            },
        )

    # --------------------- Main pass loop ---------------------
    # loop over all passes
    stoprefining = False
    refinestopreason = "passesreached"
    if optiondict["convergencethresh"] is None:
        numpasses = optiondict["passes"]
    else:
        numpasses = np.max([optiondict["passes"], optiondict["maxpasses"]])
    for thepass in range(1, numpasses + 1):
        if stoprefining:
            break

        # initialize the pass
        if optiondict["passes"] > 1:
            LGR.info("\n\n*********************")
            LGR.info(f"Pass number {thepass}")

        referencetc = tide_math.corrnormalize(
            resampref_y,
            detrendorder=optiondict["detrendorder"],
            windowfunc=optiondict["windowfunc"],
        )

        # Step -1 - check the regressor for periodic components in the passband
        dolagmod = True
        doreferencenotch = True
        if optiondict["respdelete"]:
            resptracker = tide_classes.FrequencyTracker(nperseg=64)
            thetimes, thefreqs = resptracker.track(resampref_y, 1.0 / oversamptr)
            tide_io.writevec(thefreqs, f"{outputname}_peakfreaks_pass{thepass}.txt")
            resampref_y = resptracker.clean(resampref_y, 1.0 / oversamptr, thetimes, thefreqs)
            tide_io.writevec(resampref_y, f"{outputname}_respfilt_pass{thepass}.txt")
            referencetc = tide_math.corrnormalize(
                resampref_y,
                detrendorder=optiondict["detrendorder"],
                windowfunc=optiondict["windowfunc"],
            )
        if optiondict["noisetimecoursespec"] is not None:
            # align the noise signal with referencetc
            (
                shiftednoise,
                optiondict[f"noisedelay_pass{thepass}"],
                optiondict[f"noisecorr_pass{thepass}"],
                thisfailreason,
            ) = tide_corr.aligntcwithref(
                resampref_y,
                resampnoise_y,
                oversampfreq,
                zerooutbadfit=False,
                verbose=True,
            )
            print(
                "Maximum correlation amplitude with noise regressor is ",
                optiondict[f"noisecorr_pass{thepass}"],
                " at ",
                optiondict[f"noisedelay_pass{thepass}"],
            )

            # regress out
            resampref_y, datatoremove, R = tide_fit.glmfilt(resampref_y, shiftednoise, debug=True)

            # save
            tide_io.writebidstsv(
                f"{outputname}_desc-regressornoiseremoval_timeseries",
                shiftednoise,
                1.0 / oversamptr,
                starttime=0.0,
                columns=[f"shiftednoise_pass{thepass}"],
                append=(thepass > 1),
            )
            tide_io.writebidstsv(
                f"{outputname}_desc-regressornoiseremoval_timeseries",
                datatoremove,
                1.0 / oversamptr,
                starttime=0.0,
                columns=[f"removed_pass{thepass}"],
                append=True,
            )
            tide_io.writebidstsv(
                f"{outputname}_desc-regressornoiseremoval_timeseries",
                resampref_y,
                1.0 / oversamptr,
                starttime=0.0,
                columns=[f"filtered_pass{thepass}"],
                append=True,
            )

        if optiondict["check_autocorrelation"]:
            LGR.info("checking reference regressor autocorrelation properties")
            optiondict["lagmod"] = 1000.0
            lagindpad = corrorigin - 2 * np.max((lagmininpts, lagmaxinpts))
            acmininpts = lagmininpts + lagindpad
            acmaxinpts = lagmaxinpts + lagindpad
            theCorrelator.setreftc(referencetc)
            theCorrelator.setlimits(acmininpts, acmaxinpts)
            thexcorr, accheckcorrscale, dummy = theCorrelator.run(resampref_y)
            thefitter.setcorrtimeaxis(accheckcorrscale)
            (
                maxindex,
                maxlag,
                maxval,
                acwidth,
                maskval,
                peakstart,
                peakend,
                thisfailreason,
            ) = tide_simfuncfit.onesimfuncfit(
                thexcorr,
                thefitter,
                despeckle_thresh=optiondict["despeckle_thresh"],
                lthreshval=optiondict["lthreshval"],
                fixdelay=optiondict["fixdelay"],
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype,
            )
            outputarray = np.asarray([accheckcorrscale, thexcorr])
            tide_io.writebidstsv(
                f"{outputname}_desc-autocorr_timeseries",
                thexcorr,
                1.0 / (accheckcorrscale[1] - accheckcorrscale[0]),
                starttime=accheckcorrscale[0],
                columns=[f"pass{thepass}"],
                append=(thepass > 1),
            )
            thelagthresh = np.max((abs(optiondict["lagmin"]), abs(optiondict["lagmax"])))
            theampthresh = 0.1
            LGR.info(
                f"searching for sidelobes with amplitude > {theampthresh} "
                f"with abs(lag) < {thelagthresh} s"
            )
            sidelobetime, sidelobeamp = tide_corr.check_autocorrelation(
                accheckcorrscale,
                thexcorr,
                acampthresh=theampthresh,
                aclagthresh=thelagthresh,
                detrendorder=optiondict["detrendorder"],
            )
            optiondict["acwidth"] = acwidth + 0.0
            optiondict["absmaxsigma"] = acwidth * 10.0
            passsuffix = "_pass" + str(thepass)
            if sidelobetime is not None:
                optiondict["acsidelobelag" + passsuffix] = sidelobetime
                optiondict["despeckle_thresh"] = np.max(
                    [optiondict["despeckle_thresh"], sidelobetime / 2.0]
                )
                optiondict["acsidelobeamp" + passsuffix] = sidelobeamp
                LGR.warning(
                    f"\n\nWARNING: check_autocorrelation found bad sidelobe at {sidelobetime} "
                    f"seconds ({1.0 / sidelobetime} Hz)..."
                )
                # bidsify
                """tide_io.writebidstsv(
                    f"{outputname}_desc-movingregressor_timeseries",
                    tide_math.stdnormalize(resampnonosref_y),
                    1.0 / fmritr,
                    columns=["pass1"],
                    append=False,
                )"""
                tide_io.writenpvecs(
                    np.array([sidelobetime]),
                    f"{outputname}_autocorr_sidelobetime" + passsuffix + ".txt",
                )
                if optiondict["fix_autocorrelation"]:
                    LGR.info("Removing sidelobe")
                    if dolagmod:
                        LGR.info("subjecting lag times to modulus")
                        optiondict["lagmod"] = sidelobetime / 2.0
                    if doreferencenotch:
                        LGR.info("removing spectral component at sidelobe frequency")
                        acstopfreq = 1.0 / sidelobetime
                        acfixfilter = tide_filt.NoncausalFilter(
                            debug=optiondict["debug"],
                        )
                        acfixfilter.settype("arb_stop")
                        acfixfilter.setfreqs(
                            acstopfreq * 0.9,
                            acstopfreq * 0.95,
                            acstopfreq * 1.05,
                            acstopfreq * 1.1,
                        )
                        cleaned_resampref_y = tide_math.corrnormalize(
                            acfixfilter.apply(1.0 / oversamptr, resampref_y),
                            windowfunc="None",
                            detrendorder=optiondict["detrendorder"],
                        )
                        cleaned_referencetc = tide_math.corrnormalize(
                            cleaned_resampref_y,
                            detrendorder=optiondict["detrendorder"],
                            windowfunc=optiondict["windowfunc"],
                        )
                        cleaned_nonosreferencetc = tide_math.stdnormalize(
                            acfixfilter.apply(fmrifreq, resampnonosref_y)
                        )
                        tide_io.writebidstsv(
                            f"{outputname}_desc-cleanedreferencefmrires_info",
                            cleaned_nonosreferencetc,
                            fmrifreq,
                            columns=[f"pass{thepass}"],
                            append=(thepass > 1),
                        )
                        tide_io.writebidstsv(
                            f"{outputname}_desc-cleanedreference_info",
                            cleaned_referencetc,
                            1.0 / oversamptr,
                            columns=[f"pass{thepass}"],
                            append=(thepass > 1),
                        )
                        tide_io.writebidstsv(
                            f"{outputname}_desc-cleanedresamprefy_info",
                            cleaned_resampref_y,
                            1.0 / oversamptr,
                            columns=[f"pass{thepass}"],
                            append=(thepass > 1),
                        )
                else:
                    cleaned_resampref_y = 1.0 * tide_math.corrnormalize(
                        resampref_y,
                        windowfunc="None",
                        detrendorder=optiondict["detrendorder"],
                    )
                    cleaned_referencetc = 1.0 * referencetc
                    cleaned_nonosreferencetc = 1.0 * resampnonosref_y
            else:
                LGR.info("no sidelobes found in range")
                cleaned_resampref_y = 1.0 * tide_math.corrnormalize(
                    resampref_y,
                    windowfunc="None",
                    detrendorder=optiondict["detrendorder"],
                )
                cleaned_referencetc = 1.0 * referencetc
                cleaned_nonosreferencetc = 1.0 * resampnonosref_y
        else:
            cleaned_resampref_y = 1.0 * tide_math.corrnormalize(
                resampref_y, windowfunc="None", detrendorder=optiondict["detrendorder"]
            )
            cleaned_referencetc = 1.0 * referencetc
            cleaned_nonosreferencetc = 1.0 * resampnonosref_y

        # Step 0 - estimate significance
        if optiondict["numestreps"] > 0:
            TimingLGR.info(f"Significance estimation start, pass {thepass}")
            LGR.info(f"\n\nSignificance estimation, pass {thepass}")
            LGR.verbose(
                "calling getNullDistributionData with args: "
                f"{oversampfreq} {fmritr} {corrorigin} {lagmininpts} {lagmaxinpts}"
            )
            getNullDistributionData_func = addmemprofiling(
                tide_nullsimfunc.getNullDistributionDatax,
                optiondict["memprofile"],
                "before getnulldistristributiondata",
            )
            if optiondict["checkpoint"]:
                # bidsify
                """tide_io.writebidstsv(
                    f"{outputname}_desc-movingregressor_timeseries",
                    tide_math.stdnormalize(resampnonosref_y),
                    1.0 / fmritr,
                    columns=["pass1"],
                    append=False,
                )"""
                tide_io.writenpvecs(
                    cleaned_referencetc,
                    f"{outputname}_cleanedreference_pass" + str(thepass) + ".txt",
                )
                tide_io.writenpvecs(
                    cleaned_resampref_y,
                    f"{outputname}_cleanedresampref_y_pass" + str(thepass) + ".txt",
                )
                tide_io.writedicttojson(
                    optiondict,
                    f"{outputname}_options_pregetnull_pass" + str(thepass) + ".json",
                )
            theCorrelator.setlimits(lagmininpts, lagmaxinpts)
            theCorrelator.setreftc(cleaned_resampref_y)
            theMutualInformationator.setlimits(lagmininpts, lagmaxinpts)
            theMutualInformationator.setreftc(cleaned_resampref_y)
            dummy, trimmedcorrscale, dummy = theCorrelator.getfunction()
            thefitter.setcorrtimeaxis(trimmedcorrscale)
            corrdistdata = getNullDistributionData_func(
                cleaned_resampref_y,
                oversampfreq,
                theCorrelator,
                thefitter,
                numestreps=optiondict["numestreps"],
                nprocs=optiondict["nprocs_getNullDist"],
                alwaysmultiproc=optiondict["alwaysmultiproc"],
                showprogressbar=optiondict["showprogressbar"],
                chunksize=optiondict["mp_chunksize"],
                permutationmethod=optiondict["permutationmethod"],
                fixdelay=optiondict["fixdelay"],
                fixeddelayvalue=optiondict["fixeddelayvalue"],
                rt_floatset=np.float64,
                rt_floattype="float64",
            )
            tide_io.writebidstsv(
                f"{outputname}_desc-corrdistdata_info",
                corrdistdata,
                1.0,
                columns=["pass" + str(thepass)],
                append=(thepass > 1),
            )

            # calculate percentiles for the crosscorrelation from the distribution data
            thepercentiles = np.array([0.95, 0.99, 0.995, 0.999])
            thepvalnames = []
            for thispercentile in thepercentiles:
                thepvalnames.append("{:.3f}".format(1.0 - thispercentile).replace(".", "p"))

            pcts, pcts_fit, sigfit = tide_stats.sigFromDistributionData(
                corrdistdata,
                optiondict["sighistlen"],
                thepercentiles,
                twotail=optiondict["bipolar"],
                nozero=optiondict["nohistzero"],
                dosighistfit=optiondict["dosighistfit"],
            )
            for i in range(len(thepvalnames)):
                optiondict[
                    "p_lt_" + thepvalnames[i] + "_pass" + str(thepass) + "_thresh.txt"
                ] = pcts[i]
                if optiondict["dosighistfit"]:
                    optiondict[
                        "p_lt_" + thepvalnames[i] + "_pass" + str(thepass) + "_fitthresh"
                    ] = pcts_fit[i]
                    optiondict["sigfit"] = sigfit
            if optiondict["ampthreshfromsig"]:
                if pcts is not None:
                    LGR.info(
                        f"setting ampthresh to the p < {1.0 - thepercentiles[0]:.3f} threshhold"
                    )
                    optiondict["ampthresh"] = pcts[0]
                    tide_stats.printthresholds(
                        pcts,
                        thepercentiles,
                        "Crosscorrelation significance thresholds from data:",
                    )
                    if optiondict["dosighistfit"]:
                        tide_stats.printthresholds(
                            pcts_fit,
                            thepercentiles,
                            "Crosscorrelation significance thresholds from fit:",
                        )
                        namesuffix = "_desc-nullsimfunc_hist"
                        tide_stats.makeandsavehistogram(
                            corrdistdata,
                            optiondict["sighistlen"],
                            0,
                            outputname + namesuffix,
                            displaytitle="Null correlation histogram, pass" + str(thepass),
                            refine=False,
                            dictvarname="nullsimfunchist_pass" + str(thepass),
                            therange=(0.0, 1.0),
                            append=(thepass > 1),
                            thedict=optiondict,
                        )
                else:
                    LGR.info("leaving ampthresh unchanged")

            del corrdistdata
            TimingLGR.info(
                f"Significance estimation end, pass {thepass}",
                {
                    "message2": optiondict["numestreps"],
                    "message3": "repetitions",
                },
            )

        # Step 1 - Correlation step
        if optiondict["similaritymetric"] == "mutualinfo":
            similaritytype = "Mutual information"
        elif optiondict["similaritymetric"] == "correlation":
            similaritytype = "Correlation"
        else:
            similaritytype = "MI enhanced correlation"
        LGR.info(f"\n\n{similaritytype} calculation, pass {thepass}")
        TimingLGR.info(f"{similaritytype} calculation start, pass {thepass}")
        calcsimilaritypass_func = addmemprofiling(
            tide_calcsimfunc.correlationpass,
            optiondict["memprofile"],
            "before correlationpass",
        )

        if optiondict["similaritymetric"] == "mutualinfo":
            theMutualInformationator.setlimits(lagmininpts, lagmaxinpts)
            (
                voxelsprocessed_cp,
                theglobalmaxlist,
                trimmedcorrscale,
            ) = calcsimilaritypass_func(
                fmri_data_valid[:, :],
                cleaned_referencetc,
                theMutualInformationator,
                initial_fmri_x,
                os_fmri_x,
                lagmininpts,
                lagmaxinpts,
                corrout,
                meanval,
                nprocs=optiondict["nprocs_calcsimilarity"],
                alwaysmultiproc=optiondict["alwaysmultiproc"],
                oversampfactor=optiondict["oversampfactor"],
                interptype=optiondict["interptype"],
                showprogressbar=optiondict["showprogressbar"],
                chunksize=optiondict["mp_chunksize"],
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype,
            )
        else:
            (
                voxelsprocessed_cp,
                theglobalmaxlist,
                trimmedcorrscale,
            ) = calcsimilaritypass_func(
                fmri_data_valid[:, :],
                cleaned_referencetc,
                theCorrelator,
                initial_fmri_x,
                os_fmri_x,
                lagmininpts,
                lagmaxinpts,
                corrout,
                meanval,
                nprocs=optiondict["nprocs_calcsimilarity"],
                alwaysmultiproc=optiondict["alwaysmultiproc"],
                oversampfactor=optiondict["oversampfactor"],
                interptype=optiondict["interptype"],
                showprogressbar=optiondict["showprogressbar"],
                chunksize=optiondict["mp_chunksize"],
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype,
            )
        for i in range(len(theglobalmaxlist)):
            theglobalmaxlist[i] = corrscale[theglobalmaxlist[i]]
        namesuffix = "_desc-globallag_hist"
        tide_stats.makeandsavehistogram(
            np.asarray(theglobalmaxlist),
            len(corrscale),
            0,
            outputname + namesuffix,
            displaytitle="lagtime histogram",
            therange=(corrscale[0], corrscale[-1]),
            refine=False,
            dictvarname="globallaghist_pass" + str(thepass),
            append=(optiondict["echocancel"] or (thepass > 1)),
            thedict=optiondict,
        )

        if optiondict["checkpoint"]:
            outcorrarray[:, :] = 0.0
            outcorrarray[validvoxels, :] = corrout[:, :]
            if optiondict["textio"]:
                tide_io.writenpvecs(
                    outcorrarray.reshape(nativecorrshape),
                    f"{outputname}_corrout_prefit_pass" + str(thepass) + ".txt",
                )
            else:
                savename = f"{outputname}_desc-corroutprefit_pass-" + str(thepass)
                tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader, savename)

        TimingLGR.info(
            f"{similaritytype} calculation end, pass {thepass}",
            {
                "message2": voxelsprocessed_cp,
                "message3": "voxels",
            },
        )

        # Step 1b.  Do a peak prefit
        if optiondict["similaritymetric"] == "hybrid":
            LGR.info(f"\n\nPeak prefit calculation, pass {thepass}")
            TimingLGR.info(f"Peak prefit calculation start, pass {thepass}")
            peakevalpass_func = addmemprofiling(
                tide_peakeval.peakevalpass,
                optiondict["memprofile"],
                "before peakevalpass",
            )

            voxelsprocessed_pe, thepeakdict = peakevalpass_func(
                fmri_data_valid[:, :],
                cleaned_referencetc,
                initial_fmri_x,
                os_fmri_x,
                theMutualInformationator,
                trimmedcorrscale,
                corrout,
                nprocs=optiondict["nprocs_peakeval"],
                alwaysmultiproc=optiondict["alwaysmultiproc"],
                bipolar=optiondict["bipolar"],
                oversampfactor=optiondict["oversampfactor"],
                interptype=optiondict["interptype"],
                showprogressbar=optiondict["showprogressbar"],
                chunksize=optiondict["mp_chunksize"],
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype,
            )

            TimingLGR.info(
                f"Peak prefit end, pass {thepass}",
                {
                    "message2": voxelsprocessed_pe,
                    "message3": "voxels",
                },
            )
            mipeaks = lagtimes * 0.0
            for i in range(numvalidspatiallocs):
                if len(thepeakdict[str(i)]) > 0:
                    mipeaks[i] = thepeakdict[str(i)][0][0]
        else:
            thepeakdict = None

        # Step 2 - similarity function fitting and time lag estimation
        LGR.info(f"\n\nTime lag estimation pass {thepass}")
        TimingLGR.info(f"Time lag estimation start, pass {thepass}")
        fitcorr_func = addmemprofiling(
            tide_simfuncfit.fitcorr, optiondict["memprofile"], "before fitcorr"
        )
        thefitter.setfunctype(optiondict["similaritymetric"])
        thefitter.setcorrtimeaxis(trimmedcorrscale)

        # use initial lags if this is a hybrid fit
        if optiondict["similaritymetric"] == "hybrid" and thepeakdict is not None:
            initlags = mipeaks
        else:
            initlags = None

        voxelsprocessed_fc = fitcorr_func(
            genlagtc,
            initial_fmri_x,
            lagtc,
            trimmedcorrscale,
            thefitter,
            corrout,
            fitmask,
            failreason,
            lagtimes,
            lagstrengths,
            lagsigma,
            gaussout,
            windowout,
            R2,
            despeckling=False,
            peakdict=thepeakdict,
            nprocs=optiondict["nprocs_fitcorr"],
            alwaysmultiproc=optiondict["alwaysmultiproc"],
            fixdelay=optiondict["fixdelay"],
            showprogressbar=optiondict["showprogressbar"],
            chunksize=optiondict["mp_chunksize"],
            despeckle_thresh=optiondict["despeckle_thresh"],
            initiallags=initlags,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
        )

        TimingLGR.info(
            f"Time lag estimation end, pass {thepass}",
            {
                "message2": voxelsprocessed_fc,
                "message3": "voxels",
            },
        )

        # Step 2b - Correlation time despeckle
        if optiondict["despeckle_passes"] > 0:
            LGR.info(f"\n\nCorrelation despeckling pass {thepass}")
            LGR.info(f"\tUsing despeckle_thresh = {optiondict['despeckle_thresh']:.3f}")
            TimingLGR.info(f"Correlation despeckle start, pass {thepass}")

            # find lags that are very different from their neighbors, and refit starting at the median lag for the point
            voxelsprocessed_fc_ds = 0
            despecklingdone = False
            for despecklepass in range(optiondict["despeckle_passes"]):
                LGR.info(f"\n\nCorrelation despeckling subpass {despecklepass + 1}")
                outmaparray *= 0.0
                outmaparray[validvoxels] = eval("lagtimes")[:]
                medianlags = ndimage.median_filter(
                    outmaparray.reshape(nativespaceshape), 3
                ).reshape(numspatiallocs)
                # voxels that we're happy with have initlags set to -1000000.0
                initlags = np.where(
                    np.abs(outmaparray - medianlags) > optiondict["despeckle_thresh"],
                    medianlags,
                    -1000000.0,
                )[validvoxels]
                if len(initlags) > 0:
                    if len(np.where(initlags != -1000000.0)[0]) > 0:
                        voxelsprocessed_thispass = fitcorr_func(
                            genlagtc,
                            initial_fmri_x,
                            lagtc,
                            trimmedcorrscale,
                            thefitter,
                            corrout,
                            fitmask,
                            failreason,
                            lagtimes,
                            lagstrengths,
                            lagsigma,
                            gaussout,
                            windowout,
                            R2,
                            despeckling=True,
                            peakdict=thepeakdict,
                            nprocs=optiondict["nprocs_fitcorr"],
                            alwaysmultiproc=optiondict["alwaysmultiproc"],
                            fixdelay=optiondict["fixdelay"],
                            showprogressbar=optiondict["showprogressbar"],
                            chunksize=optiondict["mp_chunksize"],
                            despeckle_thresh=optiondict["despeckle_thresh"],
                            initiallags=initlags,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )

                        voxelsprocessed_fc_ds += voxelsprocessed_thispass
                        optiondict[
                            "despecklemasksize_pass" + str(thepass) + "_d" + str(despecklepass + 1)
                        ] = voxelsprocessed_thispass
                        optiondict[
                            "despecklemaskpct_pass" + str(thepass) + "_d" + str(despecklepass + 1)
                        ] = (100.0 * voxelsprocessed_thispass / optiondict["corrmasksize"])
                    else:
                        despecklingdone = True
                else:
                    despecklingdone = True
                if despecklingdone:
                    LGR.info("Nothing left to do! Terminating despeckling")
                    break

            internaldespeckleincludemask = np.where(
                np.abs(outmaparray - medianlags) > optiondict["despeckle_thresh"],
                medianlags,
                0.0,
            )
            if optiondict["savedespecklemasks"]:
                despecklesavemask = np.where(
                    internaldespeckleincludemask[validvoxels] == 0.0, 0, 1
                )
                if thepass == optiondict["passes"]:
                    theheader = copy.deepcopy(nim_hdr)
                    theheader["dim"][4] = 1
                    theheader["pixdim"][4] = 1.0
                    savename = f"{outputname}_desc-despeckle_mask"
                    if not fileiscifti:
                        theheader["dim"][0] = 3
                        tide_io.savetonifti(
                            np.where(internaldespeckleincludemask == 0.0, 0, 1).reshape(
                                nativespaceshape
                            ),
                            theheader,
                            savename,
                        )
                    else:
                        timeindex = theheader["dim"][0] - 1
                        spaceindex = theheader["dim"][0]
                        theheader["dim"][timeindex] = 1
                        theheader["dim"][spaceindex] = numspatiallocs
                        tide_io.savetocifti(
                            (
                                np.where(
                                    np.abs(outmaparray - medianlags)
                                    > optiondict["despeckle_thresh"],
                                    medianlags,
                                    0.0,
                                )
                            ),
                            cifti_hdr,
                            theheader,
                            savename,
                            isseries=False,
                            names=["despecklemask"],
                        )
            LGR.info(
                f"\n\n{voxelsprocessed_fc_ds} voxels despeckled in "
                f"{optiondict['despeckle_passes']} passes"
            )
            TimingLGR.info(
                f"Correlation despeckle end, pass {thepass}",
                {
                    "message2": voxelsprocessed_fc_ds,
                    "message3": "voxels",
                },
            )

        # Step 3 - regressor refinement for next pass
        if (
            thepass < optiondict["passes"]
            or optiondict["convergencethresh"] is not None
            or optiondict["globalpreselect"]
        ):
            LGR.info(f"\n\nRegressor refinement, pass {thepass}")
            TimingLGR.info(f"Regressor refinement start, pass {thepass}")
            if optiondict["refineoffset"]:
                # check that we won't end up excluding all voxels from offset calculation before accepting mask
                offsetmask = np.uint16(fitmask)
                if internaloffsetincludemask_valid is not None:
                    offsetmask[np.where(internaloffsetincludemask_valid == 0)] = 0
                if internaloffsetexcludemask_valid is not None:
                    offsetmask[np.where(internaloffsetexcludemask_valid != 0.0)] = 0
                if tide_stats.getmasksize(offsetmask) == 0:
                    print(
                        "NB: cannot exclude voxels from offset calculation mask - including for this pass"
                    )
                    offsetmask = fitmask

                peaklag, peakheight, peakwidth = tide_stats.gethistprops(
                    lagtimes[np.where(offsetmask > 0)],
                    optiondict["histlen"],
                    pickleft=optiondict["pickleft"],
                    peakthresh=optiondict["pickleftthresh"],
                )
                optiondict["offsettime"] = peaklag
                optiondict["offsettime_total"] += peaklag
                LGR.info(
                    f"offset time set to {optiondict['offsettime']:.3f}, "
                    f"total is {optiondict['offsettime_total']:.3f}"
                )

            if optiondict["refinedespeckled"] or (optiondict["despeckle_passes"] == 0):
                # if refinedespeckled is true, or there is no despeckling, masks are unaffected
                thisinternalrefineexcludemask_valid = internalrefineexcludemask_valid
            else:
                # if refinedespeckled is false and there is despeckling, need to make a proper mask
                if internalrefineexcludemask_valid is None:
                    # if there is currently no exclude mask, set exclude mask = despeckle mask
                    thisinternalrefineexcludemask_valid = np.where(
                        internaldespeckleincludemask[validvoxels] == 0.0, 0, 1
                    )
                else:
                    # if there is a current exclude mask, add any voxels that are being despeckled
                    thisinternalrefineexcludemask_valid = np.where(
                        internalrefineexcludemask_valid > 0, 1, 0
                    )
                    thisinternalrefineexcludemask_valid[
                        np.where(internaldespeckleincludemask[validvoxels] != 0.0)
                    ] = 1

                # now check that we won't end up excluding all voxels from refinement before accepting mask
                overallmask = np.uint16(fitmask)
                if internalrefineincludemask_valid is not None:
                    overallmask[np.where(internalrefineincludemask_valid == 0)] = 0
                if thisinternalrefineexcludemask_valid is not None:
                    overallmask[np.where(thisinternalrefineexcludemask_valid != 0.0)] = 0
                if tide_stats.getmasksize(overallmask) == 0:
                    print(
                        "NB: cannot exclude despeckled voxels from refinement - including for this pass"
                    )
                    thisinternalrefineexcludemask_valid = internalrefineexcludemask_valid

            # regenerate regressor for next pass
            refineregressor_func = addmemprofiling(
                tide_refine.refineregressor,
                optiondict["memprofile"],
                "before refineregressor",
            )
            (
                voxelsprocessed_rr,
                outputdata,
                refinemask,
                locationfails,
                ampfails,
                lagfails,
                sigmafails,
            ) = refineregressor_func(
                fmri_data_valid,
                fmritr,
                shiftedtcs,
                weights,
                thepass,
                lagstrengths,
                lagtimes,
                lagsigma,
                fitmask,
                R2,
                theprefilter,
                optiondict,
                bipolar=optiondict["bipolar"],
                padtrs=numpadtrs,
                includemask=internalrefineincludemask_valid,
                excludemask=thisinternalrefineexcludemask_valid,
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype,
            )
            optiondict["refinemasksize_pass" + str(thepass)] = voxelsprocessed_rr
            optiondict["refinemaskpct_pass" + str(thepass)] = (
                100.0 * voxelsprocessed_rr / optiondict["corrmasksize"]
            )
            optiondict["refinelocationfails_pass" + str(thepass)] = locationfails
            optiondict["refineampfails_pass" + str(thepass)] = ampfails
            optiondict["refinelagfails_pass" + str(thepass)] = lagfails
            optiondict["refinesigmafails_pass" + str(thepass)] = sigmafails
            if voxelsprocessed_rr > 0:
                normoutputdata = tide_math.stdnormalize(theprefilter.apply(fmrifreq, outputdata))
                normunfilteredoutputdata = tide_math.stdnormalize(outputdata)
                tide_io.writebidstsv(
                    f"{outputname}_desc-refinedmovingregressor_timeseries",
                    normunfilteredoutputdata,
                    1.0 / fmritr,
                    columns=["unfiltered_pass" + str(thepass)],
                    append=(thepass > 1),
                )
                tide_io.writebidstsv(
                    f"{outputname}_desc-refinedmovingregressor_timeseries",
                    normoutputdata,
                    1.0 / fmritr,
                    columns=["filtered_pass" + str(thepass)],
                    append=True,
                )

                # check for convergence
                regressormse = mse(normoutputdata, previousnormoutputdata)
                optiondict["regressormse_pass" + str(thepass).zfill(2)] = regressormse
                LGR.info(f"regressor difference at end of pass {thepass:d} is {regressormse:.6f}")
                if optiondict["convergencethresh"] is not None:
                    if thepass >= optiondict["maxpasses"]:
                        LGR.info("refinement ended (maxpasses reached)")
                        stoprefining = True
                        refinestopreason = "maxpassesreached"
                    elif regressormse < optiondict["convergencethresh"]:
                        LGR.info("refinement ended (refinement has converged")
                        stoprefining = True
                        refinestopreason = "convergence"
                    else:
                        stoprefining = False
                elif thepass >= optiondict["passes"]:
                    stoprefining = True
                    refinestopreason = "passesreached"
                else:
                    stoprefining = False

                if optiondict["detrendorder"] > 0:
                    resampnonosref_y = tide_fit.detrend(
                        tide_resample.doresample(
                            initial_fmri_x,
                            normoutputdata,
                            initial_fmri_x,
                            method=optiondict["interptype"],
                        ),
                        order=optiondict["detrendorder"],
                        demean=optiondict["dodemean"],
                    )
                    resampref_y = tide_fit.detrend(
                        tide_resample.doresample(
                            initial_fmri_x,
                            normoutputdata,
                            os_fmri_x,
                            method=optiondict["interptype"],
                        ),
                        order=optiondict["detrendorder"],
                        demean=optiondict["dodemean"],
                    )
                else:
                    resampnonosref_y = tide_resample.doresample(
                        initial_fmri_x,
                        normoutputdata,
                        initial_fmri_x,
                        method=optiondict["interptype"],
                    )
                    resampref_y = tide_resample.doresample(
                        initial_fmri_x,
                        normoutputdata,
                        os_fmri_x,
                        method=optiondict["interptype"],
                    )
                if optiondict["tmaskname"] is not None:
                    resampnonosref_y *= tmask_y
                    thefit, R = tide_fit.mlregress(tmask_y, resampnonosref_y)
                    resampnonosref_y -= thefit[0, 1] * tmask_y
                    resampref_y *= tmaskos_y
                    thefit, R = tide_fit.mlregress(tmaskos_y, resampref_y)
                    resampref_y -= thefit[0, 1] * tmaskos_y

                # reinitialize lagtc for resampling
                previousnormoutputdata = normoutputdata + 0.0
                genlagtc = tide_resample.FastResampler(
                    initial_fmri_x, normoutputdata, padtime=padtime
                )
                (
                    optiondict["kurtosis_reference_pass" + str(thepass + 1)],
                    optiondict["kurtosisz_reference_pass" + str(thepass + 1)],
                    optiondict["kurtosisp_reference_pass" + str(thepass + 1)],
                ) = tide_stats.kurtosisstats(resampref_y)
                if not stoprefining:
                    tide_io.writebidstsv(
                        f"{outputname}_desc-movingregressor_timeseries",
                        tide_math.stdnormalize(resampnonosref_y),
                        1.0 / fmritr,
                        columns=["pass" + str(thepass + 1)],
                        append=True,
                    )
                    tide_io.writebidstsv(
                        f"{outputname}_desc-oversampledmovingregressor_timeseries",
                        tide_math.stdnormalize(resampref_y),
                        oversampfreq,
                        columns=["pass" + str(thepass + 1)],
                        append=True,
                    )
            else:
                LGR.warning(f"refinement failed - terminating at end of pass {thepass}")
                stoprefining = True
                refinestopreason = "emptymask"

            TimingLGR.info(
                f"Regressor refinement end, pass {thepass}",
                {
                    "message2": voxelsprocessed_rr,
                    "message3": "voxels",
                },
            )
        if optiondict["saveintermediatemaps"]:
            maplist = [
                ("lagtimes", "maxtime"),
                ("lagstrengths", "maxcorr"),
                ("lagsigma", "maxwidth"),
                ("fitmask", "fitmask"),
                ("failreason", "corrfitfailreason"),
            ]
            if optiondict["savedespecklemasks"]:
                maplist.append(("despecklesavemask", "despecklemask"))
            if thepass < optiondict["passes"]:
                maplist.append(("refinemask", "refinemask"))
            for mapname, mapsuffix in maplist:
                if optiondict["memprofile"]:
                    memcheckpoint(f"about to write {mapname} to {mapsuffix}")
                else:
                    tide_util.logmem(f"about to write {mapname} to {mapsuffix}")
                outmaparray[:] = 0.0
                outmaparray[validvoxels] = eval(mapname)[:]
                if optiondict["textio"]:
                    tide_io.writenpvecs(
                        outmaparray.reshape(nativespaceshape, 1),
                        f"{outputname}_{mapsuffix}{passsuffix}.txt",
                    )
                else:
                    bidspasssuffix = f"_intermediatedata-pass{thepass}"
                    if mapname == "fitmask":
                        savename = f"{outputname}{bidspasssuffix}_desc-corrfit_mask"
                    elif mapname == "failreason":
                        savename = f"{outputname}{bidspasssuffix}_desc-corrfitfailreason_info"
                    else:
                        savename = f"{outputname}{bidspasssuffix}_desc-{mapsuffix}_map"
                    bidsdict = bidsbasedict.copy()
                    if mapname == "lagtimes" or mapname == "lagsigma":
                        bidsdict["Units"] = "second"
                    tide_io.writedicttojson(bidsdict, f"{savename}.json")
                    tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, savename)
    # We are done with refinement.
    if optiondict["convergencethresh"] is None:
        optiondict["actual_passes"] = optiondict["passes"]
    else:
        optiondict["actual_passes"] = thepass - 1
    optiondict["refinestopreason"] = refinestopreason

    # Post refinement step -1 - Coherence calculation
    if optiondict["calccoherence"]:
        TimingLGR.info("Coherence calculation start")
        LGR.info("\n\nCoherence calculation")

        # make the Coherer
        theCoherer = tide_classes.Coherer(
            Fs=(1.0 / fmritr),
            reftc=cleaned_nonosreferencetc,
            freqmin=0.0,
            freqmax=0.2,
            ncprefilter=theprefilter,
            windowfunc=optiondict["windowfunc"],
            detrendorder=optiondict["detrendorder"],
            debug=False,
        )
        theCoherer.setreftc(cleaned_nonosreferencetc)
        (
            coherencefreqstart,
            dummy,
            coherencefreqstep,
            coherencefreqaxissize,
        ) = theCoherer.getaxisinfo()
        if optiondict["textio"]:
            nativecoherenceshape = (xsize, coherencefreqaxissize)
        else:
            if fileiscifti:
                nativecoherenceshape = (1, 1, 1, coherencefreqaxissize, numspatiallocs)
            else:
                nativecoherenceshape = (xsize, ysize, numslices, coherencefreqaxissize)

        internalvalidcoherenceshape = (numvalidspatiallocs, coherencefreqaxissize)
        internalcoherenceshape = (numspatiallocs, coherencefreqaxissize)

        # now allocate the arrays needed for the coherence calculation
        if optiondict["sharedmem"]:
            coherencefunc, dummy, dummy = allocshared(internalvalidcoherenceshape, rt_outfloatset)
            coherencepeakval, dummy, dummy = allocshared(numvalidspatiallocs, rt_outfloatset)
            coherencepeakfreq, dummy, dummy = allocshared(numvalidspatiallocs, rt_outfloatset)
        else:
            coherencefunc = np.zeros(internalvalidcoherenceshape, dtype=rt_outfloattype)
            coherencepeakval, dummy, dummy = allocshared(numvalidspatiallocs, rt_outfloatset)
            coherencepeakfreq = np.zeros(numvalidspatiallocs, dtype=rt_outfloattype)

        coherencepass_func = addmemprofiling(
            tide_calccoherence.coherencepass,
            optiondict["memprofile"],
            "before coherencepass",
        )
        voxelsprocessed_coherence = coherencepass_func(
            fmri_data_valid,
            theCoherer,
            coherencefunc,
            coherencepeakval,
            coherencepeakfreq,
            alt=True,
            showprogressbar=optiondict["showprogressbar"],
            chunksize=optiondict["mp_chunksize"],
            nprocs=1,
            alwaysmultiproc=False,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
        )

        # save the results of the calculations
        outcoherencearray = np.zeros(internalcoherenceshape, dtype=rt_floattype)
        outcoherencearray[validvoxels, :] = coherencefunc[:, :]
        theheader = copy.deepcopy(nim_hdr)
        theheader["toffset"] = coherencefreqstart
        theheader["pixdim"][4] = coherencefreqstep
        if optiondict["textio"]:
            tide_io.writenpvecs(
                outcoherencearray.reshape(nativecoherenceshape),
                f"{outputname}_coherence.txt",
            )
        else:
            savename = f"{outputname}_desc-coherence_info"
        if fileiscifti:
            timeindex = theheader["dim"][0] - 1
            spaceindex = theheader["dim"][0]
            theheader["dim"][timeindex] = coherencefreqaxissize
            theheader["dim"][spaceindex] = numspatiallocs
            tide_io.savetocifti(
                outcoherencearray,
                cifti_hdr,
                theheader,
                savename,
                isseries=True,
                names=["coherence"],
            )
        else:
            theheader["dim"][0] = 3
            theheader["dim"][4] = coherencefreqaxissize
            tide_io.savetonifti(
                outcoherencearray.reshape(nativecoherenceshape), theheader, savename
            )
        del coherencefunc
        del outcoherencearray

        TimingLGR.info(
            "Coherence calculation end",
            {
                "message2": voxelsprocessed_coherence,
                "message3": "voxels",
            },
        )

    # Post refinement step 0 - Wiener deconvolution
    if optiondict["dodeconv"]:
        TimingLGR.info("Wiener deconvolution start")
        LGR.info("\n\nWiener deconvolution")

        # now allocate the arrays needed for Wiener deconvolution
        if optiondict["sharedmem"]:
            wienerdeconv, dummy, dummy = allocshared(internalvalidspaceshape, rt_outfloatset)
            wpeak, dummy, dummy = allocshared(internalvalidspaceshape, rt_outfloatset)
        else:
            wienerdeconv = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
            wpeak = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)

        wienerpass_func = addmemprofiling(
            tide_wiener.wienerpass,
            optiondict["memprofile"],
            "before wienerpass",
        )
        voxelsprocessed_wiener = wienerpass_func(
            numspatiallocs,
            fmri_data_valid,
            threshval,
            optiondict,
            wienerdeconv,
            wpeak,
            resampref_y,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
        )
        TimingLGR.info(
            "Wiener deconvolution end",
            {
                "message2": voxelsprocessed_wiener,
                "message3": "voxels",
            },
        )

    # Post refinement step 1 - GLM fitting, either to remove moving signal, or to calculate delayed CVR
    if optiondict["doglmfilt"] or optiondict["docvrmap"]:
        if optiondict["doglmfilt"]:
            TimingLGR.info("GLM filtering start")
            LGR.info("\n\nGLM filtering")
        else:
            TimingLGR.info("CVR map generation start")
            LGR.info("\n\nCVR mapping")
        if (
            (optiondict["gausssigma"] > 0.0)
            or (optiondict["glmsourcefile"] is not None)
            or optiondict["docvrmap"]
        ):
            if optiondict["glmsourcefile"] is not None:
                LGR.info(f"reading in {optiondict['glmsourcefile']} for GLM filter, please wait")
                if fileiscifti:
                    LGR.info("input file is CIFTI")
                    (
                        cifti,
                        cifti_hdr,
                        nim_data,
                        nim_hdr,
                        thedims,
                        thesizes,
                        dummy,
                    ) = tide_io.readfromcifti(optiondict["glmsourcefile"])
                else:
                    if optiondict["textio"]:
                        nim_data = tide_io.readvecs(optiondict["glmsourcefile"])
                    else:
                        nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(
                            optiondict["glmsourcefile"]
                        )
            else:
                LGR.info(f"rereading {fmrifilename} for GLM filter, please wait")
                if fileiscifti:
                    LGR.info("input file is CIFTI")
                    (
                        cifti,
                        cifti_hdr,
                        nim_data,
                        nim_hdr,
                        thedims,
                        thesizes,
                        dummy,
                    ) = tide_io.readfromcifti(optiondict["glmsourcefile"])
                else:
                    if optiondict["textio"]:
                        nim_data = tide_io.readvecs(fmrifilename)
                    else:
                        nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(
                            fmrifilename
                        )

            fmri_data_valid = (
                nim_data.reshape((numspatiallocs, timepoints))[:, validstart : validend + 1]
            )[validvoxels, :] + 0.0

            if optiondict["docvrmap"]:
                # percent normalize the fmri data
                print("normalzing data for CVR map")
                themean = np.mean(fmri_data_valid, axis=1)
                fmri_data_valid /= themean[:, None]

            if optiondict["preservefiltering"]:
                print("reapplying temporal filters...")
                print(f"fmri_data_valid.shape: {fmri_data_valid.shape}")
                for i in range(len(validvoxels)):
                    filteredtc = theprefilter.apply(
                        optiondict["fmrifreq"], fmri_data_valid[i, :] + 0.0
                    )
                    fmri_data_valid[i, :] = filteredtc + 0.0
                print("...done")

            # move fmri_data_valid into shared memory
            if optiondict["sharedmem"]:
                LGR.info("moving fmri data to shared memory")
                TimingLGR.info("Start moving fmri_data to shared memory")
                numpy2shared_func = addmemprofiling(
                    numpy2shared,
                    optiondict["memprofile"],
                    "before movetoshared (glm)",
                )
                fmri_data_valid = numpy2shared_func(fmri_data_valid, rt_floatset)
                TimingLGR.info("End moving fmri_data to shared memory")
            del nim_data

        # now allocate the arrays needed for GLM filtering
        if optiondict["sharedmem"]:
            glmmean, dummy, dummy = allocshared(internalvalidspaceshape, rt_outfloatset)
            rvalue, dummy, dummy = allocshared(internalvalidspaceshape, rt_outfloatset)
            r2value, dummy, dummy = allocshared(internalvalidspaceshape, rt_outfloatset)
            fitNorm, dummy, dummy = allocshared(internalvalidspaceshape, rt_outfloatset)
            fitcoeff, dummy, dummy = allocshared(internalvalidspaceshape, rt_outfloatset)
            movingsignal, dummy, dummy = allocshared(internalvalidfmrishape, rt_outfloatset)
            filtereddata, dummy, dummy = allocshared(internalvalidfmrishape, rt_outfloatset)
        else:
            glmmean = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
            rvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
            r2value = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
            fitNorm = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
            fitcoeff = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
            movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
            filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)

        if optiondict["memprofile"]:
            if optiondict["doglmfilt"]:
                memcheckpoint("about to start glm noise removal...")
            else:
                memcheckpoint("about to start CVR magnitude estimation...")
        else:
            tide_util.logmem("before glm")

        # and do the filtering
        glmpass_func = addmemprofiling(
            tide_glmpass.glmpass, optiondict["memprofile"], "before glmpass"
        )
        if optiondict["docvrmap"]:
            # set the threshval to zero
            glmthreshval = 0.0
        else:
            glmthreshval = threshval

        voxelsprocessed_glm = glmpass_func(
            numvalidspatiallocs,
            fmri_data_valid,
            glmthreshval,
            lagtc,
            glmmean,
            rvalue,
            r2value,
            fitcoeff,
            fitNorm,
            movingsignal,
            filtereddata,
            nprocs=optiondict["nprocs_glm"],
            alwaysmultiproc=optiondict["alwaysmultiproc"],
            showprogressbar=optiondict["showprogressbar"],
            mp_chunksize=optiondict["mp_chunksize"],
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
        )

        if optiondict["docvrmap"]:
            # if we are doing a cvr map, multiply the fitcoeff by 100, so we are in percent
            fitcoeff *= 100.0

        # calculate the final bandlimited variance
        finalvariance = tide_math.imagevariance(filtereddata, theprefilter, 1.0 / fmritr)
        divlocs = np.where(finalvariance > 0.0)
        varchange = initialvariance * 0.0
        varchange[divlocs] = finalvariance[divlocs] / initialvariance[divlocs] - 1.0
        del fmri_data_valid

        TimingLGR.info(
            "GLM filtering end",
            {
                "message2": voxelsprocessed_glm,
                "message3": "voxels",
            },
        )
        if optiondict["memprofile"]:
            memcheckpoint("...done")
        else:
            tide_util.logmem("after glm filter")
        LGR.info("")
    else:
        # get the original data to calculate the mean
        LGR.info(f"rereading {fmrifilename} to calculate mean value, please wait")
        if optiondict["textio"]:
            nim_data = tide_io.readvecs(fmrifilename)
        else:
            nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
        """meanvalue = np.mean(
            nim_data.reshape((numspatiallocs, timepoints))[:, validstart : validend + 1], axis=1
        )"""

    # Post refinement step 2 - make and save interesting histograms
    TimingLGR.info("Start saving histograms")
    namesuffix = "_desc-maxtime_hist"
    tide_stats.makeandsavehistogram(
        lagtimes[np.where(fitmask > 0)],
        optiondict["histlen"],
        0,
        outputname + namesuffix,
        displaytitle="lagtime histogram",
        refine=False,
        dictvarname="laghist",
        thedict=optiondict,
    )
    namesuffix = "_desc-maxcorr_hist"
    tide_stats.makeandsavehistogram(
        lagstrengths[np.where(fitmask > 0)],
        optiondict["histlen"],
        0,
        outputname + namesuffix,
        displaytitle="lagstrength histogram",
        therange=(0.0, 1.0),
        dictvarname="strengthhist",
        thedict=optiondict,
    )
    namesuffix = "_desc-maxwidth_hist"
    tide_stats.makeandsavehistogram(
        lagsigma[np.where(fitmask > 0)],
        optiondict["histlen"],
        1,
        outputname + namesuffix,
        displaytitle="lagsigma histogram",
        dictvarname="widthhist",
        thedict=optiondict,
    )
    namesuffix = "_desc-maxcorrsq_hist"
    if optiondict["doglmfilt"]:
        tide_stats.makeandsavehistogram(
            r2value[np.where(fitmask > 0)],
            optiondict["histlen"],
            1,
            outputname + namesuffix,
            displaytitle="correlation R2 histogram",
            dictvarname="R2hist",
            thedict=optiondict,
        )
    TimingLGR.info("Finished saving histograms")

    # put some quality metrics into the info structure
    histpcts = [0.02, 0.25, 0.5, 0.75, 0.98]
    thetimepcts = tide_stats.getfracvals(lagtimes[np.where(fitmask > 0)], histpcts, nozero=False)
    thestrengthpcts = tide_stats.getfracvals(
        lagstrengths[np.where(fitmask > 0)], histpcts, nozero=False
    )
    thesigmapcts = tide_stats.getfracvals(lagsigma[np.where(fitmask > 0)], histpcts, nozero=False)
    for i in range(len(histpcts)):
        optiondict[
            "lagtimes_" + str(int(np.round(100 * histpcts[i], 0))).zfill(2) + "pct"
        ] = thetimepcts[i]
        optiondict[
            "lagstrengths_" + str(int(np.round(100 * histpcts[i], 0))).zfill(2) + "pct"
        ] = thestrengthpcts[i]
        optiondict[
            "lagsigma_" + str(int(np.round(100 * histpcts[i], 0))).zfill(2) + "pct"
        ] = thesigmapcts[i]
    optiondict["fitmasksize"] = np.sum(fitmask)
    optiondict["fitmaskpct"] = 100.0 * optiondict["fitmasksize"] / optiondict["corrmasksize"]

    # Post refinement step 3 - save out all the important arrays to nifti files
    # write out the options used
    tide_io.writedicttojson(optiondict, f"{outputname}_options.json")

    # do ones with one time point first
    TimingLGR.info("Start saving maps")
    if not optiondict["textio"]:
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            timeindex = theheader["dim"][0] - 1
            spaceindex = theheader["dim"][0]
            theheader["dim"][timeindex] = 1
            theheader["dim"][spaceindex] = numspatiallocs
        else:
            theheader["dim"][0] = 3
            theheader["dim"][4] = 1
            theheader["pixdim"][4] = 1.0

    # Prepare extra maps
    savelist = [
        ("lagtimes", "maxtime"),
        ("lagstrengths", "maxcorr"),
        ("lagsigma", "maxwidth"),
        ("R2", "maxcorrsq"),
        ("fitmask", "fitmask"),
        ("failreason", "corrfitfailreason"),
    ]
    MTT = np.square(lagsigma) - (optiondict["acwidth"] * optiondict["acwidth"])
    MTT = np.where(MTT > 0.0, MTT, 0.0)
    MTT = np.sqrt(MTT)
    savelist += [("MTT", "MTT")]
    if optiondict["calccoherence"]:
        savelist += [
            ("coherencepeakval", "coherencepeakval"),
            ("coherencepeakfreq", "coherencepeakfreq"),
        ]
    # if optiondict["similaritymetric"] == "mutualinfo":
    #    savelist += [("baseline", "baseline"), ("baselinedev", "baselinedev")]
    for mapname, mapsuffix in savelist:
        if optiondict["memprofile"]:
            memcheckpoint(f"about to write {mapname}" + "to" + mapsuffix)
        else:
            tide_util.logmem(f"about to write {mapname}" + "to" + mapsuffix)
        outmaparray[:] = 0.0
        outmaparray[validvoxels] = eval(mapname)[:]
        if optiondict["textio"]:
            tide_io.writenpvecs(
                outmaparray.reshape(nativespaceshape, 1),
                f"{outputname}_" + mapsuffix + ".txt",
            )
        else:
            if mapname == "fitmask":
                savename = f"{outputname}_desc-corrfit_mask"
            elif mapname == "failreason":
                savename = f"{outputname}_desc-corrfitfailreason_info"
            else:
                savename = f"{outputname}_desc-" + mapsuffix + "_map"
            bidsdict = bidsbasedict.copy()
            if mapname == "lagtimes" or mapname == "lagsigma" or mapname == "MTT":
                bidsdict["Units"] = "second"
            tide_io.writedicttojson(bidsdict, savename + ".json")
            if not fileiscifti:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, savename)
            else:
                tide_io.savetocifti(
                    outmaparray,
                    cifti_hdr,
                    theheader,
                    savename,
                    isseries=False,
                    names=[mapsuffix],
                )

    if optiondict["doglmfilt"] or optiondict["docvrmap"]:
        if optiondict["doglmfilt"]:
            maplist = [
                ("rvalue", "lfofilterR"),
                ("r2value", "lfofilterR2"),
                ("glmmean", "lfofilterMean"),
                ("fitcoeff", "lfofilterCoeff"),
                ("fitNorm", "lfofilterNorm"),
                ("initialvariance", "lfofilterInbandVarianceBefore"),
                ("finalvariance", "lfofilterInbandVarianceAfter"),
                ("varchange", "lfofilterInbandVarianceChange"),
            ]
        else:
            maplist = [
                ("rvalue", "CVRR"),
                ("r2value", "CVRR2"),
                ("fitcoeff", "CVR"),
                ("initialvariance", "lfofilterInbandVarianceBefore"),
                ("finalvariance", "lfofilterInbandVarianceAfter"),
                ("varchange", "CVRVariance"),
            ]
        for mapname, mapsuffix in maplist:
            if optiondict["memprofile"]:
                memcheckpoint(f"about to write {mapname}")
            else:
                tide_util.logmem(f"about to write {mapname}")
            outmaparray[:] = 0.0
            outmaparray[validvoxels] = eval(mapname)[:]
            if optiondict["textio"]:
                tide_io.writenpvecs(
                    outmaparray.reshape(nativespaceshape),
                    f"{outputname}_" + mapsuffix + ".txt",
                )
            else:
                savename = f"{outputname}_desc-" + mapsuffix + "_map"
                bidsdict = bidsbasedict.copy()
                tide_io.writedicttojson(bidsdict, savename + ".json")
                if not fileiscifti:
                    tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, savename)
                else:
                    tide_io.savetocifti(
                        outmaparray,
                        cifti_hdr,
                        theheader,
                        savename,
                        isseries=False,
                        names=[mapsuffix],
                    )
        del rvalue
        del r2value
        del fitcoeff
        del fitNorm
        del initialvariance
        del finalvariance
        del varchange

    for mapname, mapsuffix in [("meanvalue", "mean")]:
        if optiondict["memprofile"]:
            memcheckpoint(f"about to write {mapname}")
        else:
            tide_util.logmem(f"about to write {mapname}")
        outmaparray[:] = 0.0
        outmaparray[:] = eval(mapname)[:]

        if optiondict["textio"]:
            tide_io.writenpvecs(
                outmaparray.reshape(nativespaceshape),
                f"{outputname}_" + mapsuffix + ".txt",
            )
        else:
            savename = f"{outputname}_desc-" + mapsuffix + "_map"
            bidsdict = bidsbasedict.copy()
            tide_io.writedicttojson(bidsdict, savename + ".json")
            if not fileiscifti:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, savename)
            else:
                tide_io.savetocifti(
                    outmaparray,
                    cifti_hdr,
                    theheader,
                    savename,
                    isseries=False,
                    names=[mapsuffix],
                )
    del meanvalue

    if optiondict["numestreps"] > 0:
        for i in range(0, len(thepercentiles)):
            pmask = np.where(np.abs(lagstrengths) > pcts[i], fitmask, 0 * fitmask)
            outmaparray[:] = 0.0
            outmaparray[validvoxels] = pmask[:]
            if optiondict["textio"]:
                tide_io.writenpvecs(
                    outmaparray.reshape(nativespaceshape),
                    f"{outputname}_p_lt_" + thepvalnames[i] + "_mask.txt",
                )
            else:
                savename = f"{outputname}_desc-plt" + thepvalnames[i] + "_mask"
                if not fileiscifti:
                    tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, savename)
                else:
                    tide_io.savetocifti(
                        outmaparray,
                        cifti_hdr,
                        theheader,
                        savename,
                        isseries=False,
                        names=["p_lt_" + thepvalnames[i] + "_mask"],
                    )

    if (optiondict["passes"] > 1 or optiondict["globalpreselect"]) and optiondict[
        "refinestopreason"
    ] != "emptymask":
        outmaparray[:] = 0.0
        outmaparray[validvoxels] = refinemask[:]
        if optiondict["textio"]:
            tide_io.writenpvecs(
                outfmriarray.reshape(nativefmrishape), f"{outputname}_lagregressor.txt"
            )
        else:
            savename = f"{outputname}_desc-refine_mask"
            if optiondict["globalpreselect"]:
                savename = savename.replace("refine", "globalmeanpreselect")
            if not fileiscifti:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader, savename)
            else:
                tide_io.savetocifti(
                    outmaparray,
                    cifti_hdr,
                    theheader,
                    savename,
                    isseries=False,
                    names=["refinemask"],
                )
        del refinemask

    # clean up arrays that will no longer be needed
    del lagtimes
    del lagstrengths
    del lagsigma
    del R2
    del fitmask

    # now do the ones with other numbers of time points
    if not optiondict["textio"]:
        theheader = copy.deepcopy(nim_hdr)
        theheader["toffset"] = corrscale[corrorigin - lagmininpts]
        if fileiscifti:
            timeindex = theheader["dim"][0] - 1
            spaceindex = theheader["dim"][0]
            theheader["dim"][timeindex] = np.shape(outcorrarray)[1]
            theheader["dim"][spaceindex] = numspatiallocs
        else:
            theheader["dim"][4] = np.shape(outcorrarray)[1]
            theheader["pixdim"][4] = corrtr
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = gaussout[:, :]
    if optiondict["textio"]:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape), f"{outputname}_gaussout.txt")
    else:
        savename = f"{outputname}_desc-gaussout_info"
        if not fileiscifti:
            tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader, savename)
        else:
            tide_io.savetocifti(
                outcorrarray,
                cifti_hdr,
                theheader,
                savename,
                isseries=True,
                start=theheader["toffset"],
                step=corrtr,
            )

    del gaussout
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = windowout[:, :]
    if optiondict["textio"]:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape), f"{outputname}_windowout.txt")
    else:
        savename = f"{outputname}_desc-corrfitwindow_info"
        if not fileiscifti:
            tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader, savename)
        else:
            tide_io.savetocifti(
                outcorrarray,
                cifti_hdr,
                theheader,
                savename,
                isseries=True,
                start=theheader["toffset"],
                step=corrtr,
            )

    del windowout
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = corrout[:, :]
    if optiondict["textio"]:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape), f"{outputname}_corrout.txt")
    else:
        savename = f"{outputname}_desc-corrout_info"
        if not fileiscifti:
            tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader, savename)
        else:
            tide_io.savetocifti(
                outcorrarray,
                cifti_hdr,
                theheader,
                savename,
                isseries=True,
                start=theheader["toffset"],
                step=corrtr,
            )
    del corrout

    if not optiondict["textio"]:
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            timeindex = theheader["dim"][0] - 1
            spaceindex = theheader["dim"][0]
            theheader["dim"][timeindex] = np.shape(outfmriarray)[1]
            theheader["dim"][spaceindex] = numspatiallocs
        else:
            theheader["dim"][4] = np.shape(outfmriarray)[1]
            theheader["pixdim"][4] = fmritr

    if optiondict["savelagregressors"]:
        outfmriarray[validvoxels, :] = lagtc[:, :]
        if optiondict["textio"]:
            tide_io.writenpvecs(
                outfmriarray.reshape(nativefmrishape), f"{outputname}_lagregressor.txt"
            )
        else:
            savename = f"{outputname}_desc-lagregressor_bold"
            tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, savename)
            if not fileiscifti:
                tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, savename)
            else:
                tide_io.savetocifti(
                    outfmriarray,
                    cifti_hdr,
                    theheader,
                    savename,
                    isseries=True,
                    start=0.0,
                    step=fmritr,
                )
        del lagtc

    if optiondict["passes"] > 1:
        if optiondict["savelagregressors"]:
            outfmriarray[validvoxels, :] = shiftedtcs[:, :]
            if optiondict["textio"]:
                tide_io.writenpvecs(
                    outfmriarray.reshape(nativefmrishape),
                    f"{outputname}_shiftedtcs.txt",
                )
            else:
                savename = f"{outputname}_desc-shiftedtcs_bold"
                if not fileiscifti:
                    tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, savename)
                else:
                    tide_io.savetocifti(
                        outfmriarray,
                        cifti_hdr,
                        theheader,
                        savename,
                        isseries=True,
                        start=0.0,
                        step=fmritr,
                    )
        del shiftedtcs

    if optiondict["doglmfilt"] and optiondict["saveglmfiltered"]:
        if optiondict["savemovingsignal"]:
            outfmriarray[validvoxels, :] = movingsignal[:, :]
            if optiondict["textio"]:
                tide_io.writenpvecs(
                    outfmriarray.reshape(nativefmrishape),
                    f"{outputname}_movingsignal.txt",
                )
            else:
                savename = f"{outputname}_desc-lfofilterRemoved_bold"
                if not fileiscifti:
                    tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, savename)
                else:
                    tide_io.savetocifti(
                        outfmriarray,
                        cifti_hdr,
                        theheader,
                        savename,
                        isseries=True,
                        start=0.0,
                        step=fmritr,
                    )
        del movingsignal
        outfmriarray[validvoxels, :] = filtereddata[:, :]
        if optiondict["textio"]:
            tide_io.writenpvecs(
                outfmriarray.reshape(nativefmrishape), f"{outputname}_filtereddata.txt"
            )
        else:
            savename = f"{outputname}_desc-lfofilterCleaned_bold"
            if not fileiscifti:
                tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader, savename)
            else:
                tide_io.savetocifti(
                    outfmriarray,
                    cifti_hdr,
                    theheader,
                    savename,
                    isseries=True,
                    start=0.0,
                    step=fmritr,
                )
        del filtereddata

    TimingLGR.info("Finished saving maps")
    LGR.info("done")

    TimingLGR.info("Done")

    # Post refinement step 5 - process and save timing information
    nodeline = " ".join(
        [
            "Processed on",
            platform.node(),
            "(",
            optiondict["release_version"] + ",",
            optiondict["git_date"],
            ")",
        ]
    )

    optiondict["platform_information"] = nodeline
    tide_util.logmem("status")

    # shut down logging
    logging.shutdown()

    # reformat timing information
    timingdata, optiondict["totalruntime"] = tide_util.proctiminglogfile(
        f"{outputname}_runtimings.tsv"
    )
    tide_io.writevec(
        timingdata,
        f"{outputname}_desc-formattedruntimings_info.tsv",
    )

    # do a final save of the options file
    tide_io.writedicttojson(optiondict, f"{outputname}_desc-runoptions_info.json")

    # delete the canary file
    Path(f"{outputname}_ISRUNNING.txt").unlink()

    # created the finished file
    Path(f"{outputname}_DONE.txt").touch()
