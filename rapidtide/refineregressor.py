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
import gc
import logging
import sys

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, FastICA

import rapidtide.fit as tide_fit
import rapidtide.genericmultiproc as tide_genericmultiproc
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats

LGR = logging.getLogger("GENERAL")


def _procOneVoxelTimeShift(
    vox,
    voxelargs,
    **kwargs,
):
    options = {
        "detrendorder": 1,
        "offsettime": 0.0,
        "debug": False,
    }
    options.update(kwargs)
    detrendorder = options["detrendorder"]
    offsettime = options["offsettime"]
    debug = options["debug"]
    if debug:
        print(f"{detrendorder=} {offsettime=}")
    (
        fmritc,
        lagtime,
        padtrs,
        fmritr,
    ) = voxelargs
    if detrendorder > 0:
        normtc = tide_fit.detrend(fmritc, order=detrendorder, demean=True)
    else:
        normtc = fmritc + 0.0
    shifttr = -(-offsettime + lagtime) / fmritr  # lagtime is in seconds
    [shiftedtc, weights, paddedshiftedtc, paddedweights] = tide_resample.timeshift(
        normtc, shifttr, padtrs
    )
    return vox, shiftedtc, weights, paddedshiftedtc, paddedweights


def _packvoxeldata(voxnum, voxelargs):
    return [(voxelargs[0])[voxnum, :], (voxelargs[1])[voxnum], voxelargs[2], voxelargs[3]]


def _unpackvoxeldata(retvals, voxelproducts):
    (voxelproducts[0])[retvals[0], :] = retvals[1]
    (voxelproducts[1])[retvals[0], :] = retvals[2]
    (voxelproducts[2])[retvals[0], :] = retvals[3]
    (voxelproducts[3])[retvals[0], :] = retvals[4]


def alignvoxels(
    fmridata,
    fmritr,
    shiftedtcs,
    weights,
    paddedshiftedtcs,
    paddedweights,
    lagtimes,
    lagmask,
    detrendorder=1,
    offsettime=0.0,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    padtrs=60,
    debug=False,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    """
    This routine applies a timeshift to every voxel in the image.
    Inputs are:
        fmridata - the fmri data, filtered to the passband
        fmritr - the timestep of the data
        shiftedtcs,
        weights,
        paddedshiftedtcs,
        paddedweights,
        lagtimes, lagmask - the results of the correlation fit.
        detrendorder - the order of the polynomial to use to detrend the data
        offsettime - the global timeshift to apply to all timecourses
        nprocs - the number of processes to use if multiprocessing is enabled

    Explicit outputs are:
        volumetotal - the number of voxels processed

    Implicit outputs:
        shiftedtcs - voxelwise fmri data timeshifted to zero lag
        weights - the weights of every timepoint in the final regressor
        paddedshiftedtcs - voxelwise fmri data timeshifted to zero lag, with a buffer of padtrs on each end
        paddedweights - the weights of every timepoint in the final regressor, with a buffer of padtrs on each end


    Parameters
    ----------
    fmridata : 4D numpy float array
       fMRI data
    fmritr : float
        Data repetition rate, in seconds
    shiftedtcs : 4D numpy float array
        Destination array for time aligned voxel timecourses
    weights :  unknown
        unknown
    passnum : int
        Number of the pass (for labelling output)
    lagstrengths : 3D numpy float array
        Maximum correlation coefficient in every voxel
    lagtimes : 3D numpy float array
        Time delay of maximum crosscorrelation in seconds
    lagsigma : 3D numpy float array
        Gaussian width of the crosscorrelation peak, in seconds.
    lagmask : 3D numpy float array
        Mask of voxels with successful correlation fits.
    R2 : 3D numpy float array
        Square of the maximum correlation coefficient in every voxel
    theprefilter : function
        The filter function to use
    optiondict : dict
        Dictionary of all internal rapidtide configuration variables.
    padtrs : int, optional
        Number of timepoints to pad onto each end
    includemask : 3D array
        Mask of voxels to include in refinement.  Default is None (all voxels).
    excludemask : 3D array
        Mask of voxels to exclude from refinement.  Default is None (no voxels).
    debug : bool
        Enable additional debugging output.  Default is False
    rt_floatset : function
        Function to coerce variable types
    rt_floattype : {'float32', 'float64'}
        Data type for internal variables

    Returns
    -------
    volumetotal : int
        Number of voxels processed
    outputdata : float array
        New regressor
    maskarray : 3D array
        Mask of voxels used for refinement
    """
    inputshape = np.shape(fmridata)
    voxelargs = [fmridata, lagtimes, padtrs, fmritr]
    voxelfunc = _procOneVoxelTimeShift
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [
        shiftedtcs,
        weights,
        paddedshiftedtcs,
        paddedweights,
    ]
    if debug:
        print("alignvoxels: {inputshape}")
        print("volumetotal: {volumetotal}")

    # timeshift the valid voxels
    # NOTE need to figure out how to use kwargs to pass extra arguments
    volumetotal = tide_genericmultiproc.run_multiproc(
        voxelfunc,
        packfunc,
        unpackfunc,
        voxelargs,
        voxeltargets,
        inputshape,
        lagmask,
        LGR,
        nprocs,
        alwaysmultiproc,
        showprogressbar,
        chunksize,
        detrendorder=detrendorder,
        offsettime=offsettime,
        debug=debug,
    )

    LGR.info(
        "Timeshift applied to " + str(int(volumetotal)) + " voxels",
    )

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal


def makerefinemask(
    lagstrengths,
    lagtimes,
    lagsigma,
    lagmask,
    offsettime=0.0,
    ampthresh=0.3,
    lagmaskside="both",
    lagminthresh=0.5,
    lagmaxthresh=5.0,
    sigmathresh=100,
    cleanrefined=False,
    bipolar=False,
    includemask=None,
    excludemask=None,
    fixdelay=False,
    debug=False,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    """
    This routine determines which voxels should be used for regressor refinement.

    Parameters
    ----------
    lagstrengths : 3D numpy float array
        Maximum correlation coefficient in every voxel
    lagtimes : 3D numpy float array
        Time delay of maximum crosscorrelation in seconds
    lagsigma : 3D numpy float array
        Gaussian width of the crosscorrelation peak, in seconds.
    lagmask : 3D numpy float array
        Mask of voxels with successful correlation fits.
    offsettime: float
        The offset time in seconds to apply to all regressors
    ampthresh: float
        The lower limit of correlation values to consider for refine mask inclusion
    lagmaskside: str
        Which side of the lag values to consider - upper, lower, or both
    lagminthresh: float
        The lower limit of absolute lag values to consider for refine mask inclusion
    lagmaxthresh: float
        The upper limit of absolute lag values to consider for refine mask inclusion
    sigmathresh: float
        The upper limit of lag peak width for refine mask inclusion
    cleanrefined: bool
        If True,
    bipolar : bool
        If True, consider positive and negative correlation peaks
    includemask : 3D array
        Mask of voxels to include in refinement.  Default is None (all voxels).
    excludemask : 3D array
        Mask of voxels to exclude from refinement.  Default is None (no voxels).
    debug : bool
        Enable additional debugging output.  Default is False
    rt_floatset : function
        Function to coerce variable types
    rt_floattype : {'float32', 'float64'}
        Data type for internal variables

    Returns
    -------
    volumetotal : int
        Number of voxels processed
    maskarray : 3D array
        Mask of voxels used for refinement
    locationfails: int
        Number of locations eliminated due to the include and exclude masks
    ampfails: int
        Number of locations eliminated because the correlation value was too low
    lagfails: int
        Number of locations eliminated because the lag values were out of range
    sigmafails: int
        Number of locations eliminated because the correlation peak was too wide
    """

    if ampthresh < 0.0:
        if bipolar:
            theampthresh = tide_stats.getfracval(np.fabs(lagstrengths), -ampthresh, nozero=True)
        else:
            theampthresh = tide_stats.getfracval(lagstrengths, -ampthresh, nozero=True)
        LGR.info(f"setting ampthresh to the {-100.0 * ampthresh}th percentile ({theampthresh})")
    else:
        theampthresh = ampthresh
    if debug:
        print(f"makerefinemask: {theampthresh=}")
    if bipolar:
        ampmask = np.where(np.fabs(lagstrengths) >= theampthresh, np.int16(1), np.int16(0))
    else:
        ampmask = np.where(lagstrengths >= theampthresh, np.int16(1), np.int16(0))
    if fixdelay:
        delaymask = lagmask + 0
    else:
        if lagmaskside == "upper":
            delaymask = np.where(
                (lagtimes - offsettime) > lagminthresh,
                np.int16(1),
                np.int16(0),
            ) * np.where(
                (lagtimes - offsettime) < lagmaxthresh,
                np.int16(1),
                np.int16(0),
            )
        elif lagmaskside == "lower":
            delaymask = np.where(
                (lagtimes - offsettime) < -lagminthresh,
                np.int16(1),
                np.int16(0),
            ) * np.where(
                (lagtimes - offsettime) > -lagmaxthresh,
                np.int16(1),
                np.int16(0),
            )
        else:
            abslag = abs(lagtimes - offsettime)
            delaymask = np.where(abslag > lagminthresh, np.int16(1), np.int16(0)) * np.where(
                abslag < lagmaxthresh, np.int16(1), np.int16(0)
            )
        if debug:
            print(f"makerefinemask: {lagmaskside=}")
            print(f"makerefinemask: {lagminthresh=}")
            print(f"makerefinemask: {lagmaxthresh=}")
            print(f"makerefinemask: {offsettime=}")
    sigmamask = np.where(lagsigma < sigmathresh, np.int16(1), np.int16(0))
    locationmask = lagmask + 0
    if includemask is not None:
        locationmask = locationmask * includemask
    if excludemask is not None:
        locationmask = locationmask * (1 - excludemask)
    locationmask = locationmask.astype(np.int16)
    LGR.info("location mask created")

    # first generate the refine mask
    locationfails = np.sum(1 - locationmask)
    ampfails = np.sum(1 - ampmask * locationmask)
    lagfails = np.sum(1 - delaymask * locationmask)
    sigmafails = np.sum(1 - sigmamask * locationmask)
    refinemask = locationmask * ampmask * delaymask * sigmamask
    if tide_stats.getmasksize(refinemask) == 0:
        print("ERROR: no voxels in the refine mask:")
        print(
            "\n	",
            locationfails,
            " locationfails",
            "\n	",
            ampfails,
            " ampfails",
            "\n	",
            lagfails,
            " lagfails",
            "\n	",
            sigmafails,
            " sigmafails",
        )
        if (includemask is None) and (excludemask is None):
            print("\nRelax ampthresh, delaythresh, or sigmathresh - exiting")
        else:
            print(
                "\nChange include/exclude masks or relax ampthresh, delaythresh, or sigmathresh - exiting"
            )
        return 0, None, locationfails, ampfails, lagfails, sigmafails, 0

    if cleanrefined:
        shiftmask = locationmask
    else:
        shiftmask = refinemask
    volumetotal = np.sum(shiftmask)
    LGR.info(
        f"{int(volumetotal)} voxels will be used for refinement:"
        + f"\n	{locationfails} locationfails"
        + f"\n	{ampfails} ampfails"
        + f"\n	{lagfails} lagfails"
        + f"\n	{sigmafails} sigmafails"
    )
    numinmask = np.sum(lagmask)
    if numinmask is None:
        numinmask = 0

    return volumetotal, shiftmask, locationfails, ampfails, lagfails, sigmafails, numinmask


def prenorm(
    shiftedtcs,
    refinemask,
    lagtimes,
    lagmaxthresh,
    lagstrengths,
    R2vals,
    refineprenorm,
    refineweighting,
    debug=False,
):
    if debug:
        print(f"{shiftedtcs.shape=}"),
        print(f"{refinemask.shape=}"),
        print(f"{lagtimes.shape=}"),
        print(f"{lagmaxthresh=}"),
        print(f"{lagstrengths.shape=}"),
        print(f"{R2vals.shape=}"),
        print(f"{refineprenorm=}"),
        print(f"{refineweighting=}"),
    if refineprenorm == "mean":
        thedivisor = np.mean(shiftedtcs, axis=1)
    elif refineprenorm == "var":
        thedivisor = np.var(shiftedtcs, axis=1)
    elif refineprenorm == "std":
        thedivisor = np.std(shiftedtcs, axis=1)
    elif refineprenorm == "invlag":
        thedivisor = np.where(np.fabs(lagtimes) < lagmaxthresh, lagmaxthresh - lagtimes, 0.0)
    else:
        thedivisor = shiftedtcs[:, 0] * 0.0 + 1.0

    normfac = np.where(thedivisor != 0.0, 1.0 / thedivisor, 0.0)

    if refineweighting == "R":
        thisweight = lagstrengths
    elif refineweighting == "R2":
        thisweight = R2vals
    else:
        thisweight = np.where(lagstrengths > 0.0, 1.0, -1.0)
    thisweight *= refinemask

    if debug:
        print(f"{thedivisor.shape=}")
        print(f"{normfac.shape=}")
        print(f"{thisweight.shape=}")

    shiftedtcs *= (normfac * thisweight)[:, None]


def dorefine(
    shiftedtcs,
    refinemask,
    weights,
    theprefilter,
    fmritr,
    passnum,
    lagstrengths,
    lagtimes,
    refinetype,
    fmrifreq,
    outputname,
    detrendorder=1,
    pcacomponents=0.8,
    dodispersioncalc=False,
    dispersioncalc_lower=0.0,
    dispersioncalc_upper=0.0,
    dispersioncalc_step=0.0,
    windowfunc="hamming",
    cleanrefined=False,
    bipolar=False,
    debug=False,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    # now generate the refined timecourse(s)
    inputshape = np.shape(shiftedtcs)
    validlist = np.where(refinemask > 0)[0]
    volumetotal = len(validlist)
    refinevoxels = shiftedtcs[validlist, :]
    if bipolar:
        for thevoxel in range(len(validlist)):
            if lagstrengths[validlist][thevoxel] < 0.0:
                refinevoxels[thevoxel, :] *= -1.0
    refineweights = weights[validlist]
    weightsum = np.sum(refineweights, axis=0) / volumetotal
    averagedata = np.sum(refinevoxels, axis=0) / volumetotal
    if cleanrefined:
        invalidlist = np.where((1 - refinemask) > 0)[0]
        discardvoxels = shiftedtcs[invalidlist]
        discardweights = weights[invalidlist]
        discardweightsum = np.sum(discardweights, axis=0) / volumetotal
        averagediscard = np.sum(discardvoxels, axis=0) / volumetotal
    if dodispersioncalc:
        LGR.info("splitting regressors by time lag for phase delay estimation")
        laglist = np.arange(
            dispersioncalc_lower,
            dispersioncalc_upper,
            dispersioncalc_step,
        )
        dispersioncalcout = np.zeros((np.shape(laglist)[0], inputshape[1]), dtype=rt_floattype)
        fftlen = int(inputshape[1] // 2)
        fftlen -= fftlen % 2
        dispersioncalcspecmag = np.zeros((np.shape(laglist)[0], fftlen), dtype=rt_floattype)
        dispersioncalcspecphase = np.zeros((np.shape(laglist)[0], fftlen), dtype=rt_floattype)
        ###### BBF dispersioncalc fails when the number of timepoints is odd (or even - not sure).  Works the other way.
        for lagnum in range(0, np.shape(laglist)[0]):
            lower = laglist[lagnum] - dispersioncalc_step / 2.0
            upper = laglist[lagnum] + dispersioncalc_step / 2.0
            inlagrange = np.where(
                refinemask
                * np.where(lower < lagtimes, np.int16(1), np.int16(0))
                * np.where(lagtimes < upper, np.int16(1), np.int16(0))
            )[0]
            LGR.info(
                f"\tsumming {np.shape(inlagrange)[0]} regressors with lags from {lower} to {upper}"
            )
            if np.shape(inlagrange)[0] > 0:
                dispersioncalcout[lagnum, :] = tide_math.corrnormalize(
                    np.mean(shiftedtcs[inlagrange], axis=0),
                    detrendorder=detrendorder,
                    windowfunc=windowfunc,
                )
                (
                    freqs,
                    dispersioncalcspecmag[lagnum, :],
                    dispersioncalcspecphase[lagnum, :],
                ) = tide_math.polarfft(dispersioncalcout[lagnum, :], 1.0 / fmritr)
            inlagrange = None
        tide_io.writenpvecs(
            dispersioncalcout,
            outputname + "_dispersioncalcvecs_pass" + str(passnum) + ".txt",
        )
        tide_io.writenpvecs(
            dispersioncalcspecmag,
            outputname + "_dispersioncalcspecmag_pass" + str(passnum) + ".txt",
        )
        tide_io.writenpvecs(
            dispersioncalcspecphase,
            outputname + "_dispersioncalcspecphase_pass" + str(passnum) + ".txt",
        )
        tide_io.writenpvecs(
            freqs,
            outputname + "_dispersioncalcfreqs_pass" + str(passnum) + ".txt",
        )

    if pcacomponents < 0.0:
        pcacomponents = "mle"
    elif pcacomponents >= 1.0:
        pcacomponents = int(np.round(pcacomponents))
    elif pcacomponents == 0.0:
        print("0.0 is not an allowed value for pcacomponents")
        sys.exit()
    else:
        pcacomponents = pcacomponents
    icacomponents = 1

    if refinetype == "ica":
        LGR.info("performing ica refinement")
        thefit = FastICA(n_components=icacomponents).fit(refinevoxels)  # Reconstruct signals
        LGR.info(f"Using first of {len(thefit.components_)} components")
        icadata = thefit.components_[0]
        filteredavg = tide_math.corrnormalize(
            theprefilter.apply(fmrifreq, averagedata),
            detrendorder=detrendorder,
        )
        filteredica = tide_math.corrnormalize(
            theprefilter.apply(fmrifreq, icadata),
            detrendorder=detrendorder,
        )
        thepxcorr = pearsonr(filteredavg, filteredica)[0]
        LGR.info(f"ica/avg correlation = {thepxcorr}")
        if thepxcorr > 0.0:
            outputdata = 1.0 * icadata
        else:
            outputdata = -1.0 * icadata
    elif refinetype == "pca":
        # use the method of "A novel perspective to calibrate temporal delays in cerebrovascular reactivity
        # using hypercapnic and hyperoxic respiratory challenges". NeuroImage 187, 154?165 (2019).
        LGR.info(f"performing pca refinement with pcacomponents set to {pcacomponents}")
        try:
            thefit = PCA(n_components=pcacomponents).fit(refinevoxels)
        except ValueError:
            if pcacomponents == "mle":
                LGR.info("mle estimation failed - falling back to pcacomponents=0.8")
                thefit = PCA(n_components=0.8).fit(refinevoxels)
            else:
                print("unhandled math exception in PCA refinement - exiting")
                sys.exit()
        LGR.info(
            f"Using {len(thefit.components_)} component(s), accounting for "
            + f"{100.0 * np.cumsum(thefit.explained_variance_ratio_)[len(thefit.components_) - 1]}% of the variance"
        )
        reduceddata = thefit.inverse_transform(thefit.transform(refinevoxels))
        if debug:
            print("complex processing: reduceddata.shape =", reduceddata.shape)
        pcadata = np.mean(reduceddata, axis=0)
        filteredavg = tide_math.corrnormalize(
            theprefilter.apply(fmrifreq, averagedata),
            detrendorder=detrendorder,
        )
        filteredpca = tide_math.corrnormalize(
            theprefilter.apply(fmrifreq, pcadata),
            detrendorder=detrendorder,
        )
        thepxcorr = pearsonr(filteredavg, filteredpca)[0]
        LGR.info(f"pca/avg correlation = {thepxcorr}")
        if thepxcorr > 0.0:
            outputdata = 1.0 * pcadata
        else:
            outputdata = -1.0 * pcadata
    elif refinetype == "weighted_average":
        LGR.info("performing weighted averaging refinement")
        outputdata = np.nan_to_num(averagedata / weightsum)
    else:
        LGR.info("performing unweighted averaging refinement")
        outputdata = averagedata

    if cleanrefined:
        thefit, R2 = tide_fit.mlregress(averagediscard, averagedata)

        fitcoff = rt_floatset(thefit[0, 1])
        datatoremove = rt_floatset(fitcoff * averagediscard)
        outputdata -= datatoremove

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal, outputdata
