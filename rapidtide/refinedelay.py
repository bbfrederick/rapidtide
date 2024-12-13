#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
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
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.ndimage import median_filter
from statsmodels.robust import mad

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.workflows.glmfrommaps as tide_glmfrommaps


global ratiotooffsetfunc, maplimits


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def trainratiotooffset(
    lagtcgenerator,
    timeaxis,
    outputname,
    outputlevel,
    mindelay=-3.0,
    maxdelay=3.0,
    numpoints=501,
    smoothpts=3,
    edgepad=5,
    debug=False,
):
    global ratiotooffsetfunc, maplimits

    if debug:
        print("ratiotooffsetfunc:")
        lagtcgenerator.info(prefix="\t")
        print("\ttimeaxis:", timeaxis)
        print("\toutputname:", outputname)
        print("\tmindelay:", mindelay)
        print("\tmaxdelay:", maxdelay)
        print("\tsmoothpts:", smoothpts)
        print("\tedgepad:", edgepad)
        print("\tlagtcgenerator:", lagtcgenerator)
    # make a delay map
    delaystep = (maxdelay - mindelay) / (numpoints - 1)
    if debug:
        print(f"{delaystep=}")
        print(f"{mindelay=}")
        print(f"{maxdelay=}")
    lagtimes = np.linspace(
        mindelay - edgepad * delaystep,
        maxdelay + edgepad * delaystep,
        numpoints + 2 * edgepad,
        endpoint=True,
    )
    if debug:
        print(f"{mindelay=}")
        print(f"{maxdelay=}")
        print("lagtimes=", lagtimes)

    # now make synthetic fMRI data
    internalvalidfmrishape = (numpoints + 2 * edgepad, timeaxis.shape[0])
    fmridata = np.zeros(internalvalidfmrishape, dtype=float)
    fmrimask = np.ones(numpoints + 2 * edgepad, dtype=float)
    validvoxels = np.where(fmrimask > 0)[0]
    for i in range(numpoints + 2 * edgepad):
        fmridata[i, :] = lagtcgenerator.yfromx(timeaxis - lagtimes[i])

    rt_floattype = "float64"
    glmmean = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    rvalue = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    r2value = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    fitNorm = np.zeros((numpoints + 2 * edgepad, 2), dtype=rt_floattype)
    fitcoeff = np.zeros((numpoints + 2 * edgepad, 2), dtype=rt_floattype)
    movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    sampletime = timeaxis[1] - timeaxis[0]
    optiondict = {
        "glmthreshval": 0.0,
        "saveminimumglmfiles": False,
        "nprocs_makelaggedtcs": 1,
        "nprocs_glm": 1,
        "mp_chunksize": 1000,
        "showprogressbar": False,
        "alwaysmultiproc": False,
        "memprofile": False,
        "focaldebug": debug,
        "fmrifreq": 1.0 / sampletime,
        "textio": False,
    }

    glmderivratios = getderivratios(
        fmridata,
        validvoxels,
        timeaxis,
        0.0 * lagtimes,
        fmrimask,
        lagtcgenerator,
        "glm",
        "refinedelaytest",
        sampletime,
        glmmean,
        rvalue,
        r2value,
        fitNorm[:, :2],
        fitcoeff[:, :2],
        movingsignal,
        lagtc,
        filtereddata,
        None,
        None,
        optiondict,
        debug=debug,
    )
    if debug:
        print("before trimming")
        print(f"{glmderivratios.shape=}")
        print(f"{lagtimes.shape=}")
    smoothglmderivratios = tide_filt.unpadvec(
        smooth(tide_filt.padvec(glmderivratios, padlen=20, padtype="constant"), smoothpts),
        padlen=20,
    )
    glmderivratios = glmderivratios[edgepad:-edgepad]
    smoothglmderivratios = smoothglmderivratios[edgepad:-edgepad]
    lagtimes = lagtimes[edgepad:-edgepad]
    if debug:
        print("after trimming")
        print(f"{glmderivratios.shape=}")
        print(f"{smoothglmderivratios.shape=}")
        print(f"{lagtimes.shape=}")

    # make sure the mapping function is legal
    xaxis = smoothglmderivratios[::-1]
    yaxis = lagtimes[::-1]
    midpoint = int(len(xaxis) // 2)
    lowerlim = midpoint + 0
    while (lowerlim > 1) and xaxis[lowerlim] > xaxis[lowerlim - 1]:
        lowerlim -= 1
    upperlim = midpoint + 0
    while (upperlim < len(xaxis) - 2) and xaxis[upperlim] < xaxis[upperlim + 1]:
        upperlim += 1
    xaxis = xaxis[lowerlim : upperlim + 1]
    yaxis = yaxis[lowerlim : upperlim + 1]
    ratiotooffsetfunc = CubicSpline(xaxis, yaxis)
    maplimits = (xaxis[0], xaxis[-1])

    if outputlevel != "min":
        """tide_io.writebidstsv(
            f"{outputname}_desc-ratiotodelaymapping_timeseries",
            np.stack((xaxis, yaxis)),
            1.0,
            columns=["smoothglmderivratio", "delay"],
            extraheaderinfo={
                "Description": "The ratio of sLFO derivative to the sLFO, and the corresponding delay offset"
            },
            append=False,
        )"""
        resampaxis = np.linspace(xaxis[0], xaxis[-1], num=len(xaxis), endpoint=True)
        tide_io.writebidstsv(
            f"{outputname}_desc-ratiotodelayfunc_timeseries",
            ratiotooffsetfunc(resampaxis),
            1.0 / (resampaxis[1] - resampaxis[0]),
            starttime=resampaxis[0],
            columns=["delay"],
            extraheaderinfo={
                "Description": "The function mapping derivative ratio to delay"
            },
            append=False,
        )


def ratiotodelay(theratio):
    global ratiotooffsetfunc, maplimits
    if theratio < maplimits[0]:
        return ratiotooffsetfunc(maplimits[0])
    elif theratio > maplimits[1]:
        return ratiotooffsetfunc(maplimits[1])
    else:
        return ratiotooffsetfunc(theratio)


def getderivratios(
    fmri_data_valid,
    validvoxels,
    initial_fmri_x,
    lagtimes,
    fitmask,
    genlagtc,
    mode,
    outputname,
    oversamptr,
    glmmean,
    rvalue,
    r2value,
    fitNorm,
    fitcoeff,
    movingsignal,
    lagtc,
    filtereddata,
    LGR,
    TimingLGR,
    optiondict,
    debug=False,
):
    if debug:
        print("getderivratios")
        print(f"{fitNorm.shape=}")
        print(f"{fitcoeff.shape=}")
    voxelsprocessed_glm, regressorset, evset = tide_glmfrommaps.glmfrommaps(
        fmri_data_valid,
        validvoxels,
        initial_fmri_x,
        lagtimes,
        fitmask,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        glmmean,
        rvalue,
        r2value,
        fitNorm,
        fitcoeff,
        movingsignal,
        lagtc,
        filtereddata,
        LGR,
        TimingLGR,
        optiondict["glmthreshval"],
        optiondict["saveminimumglmfiles"],
        nprocs_makelaggedtcs=optiondict["nprocs_makelaggedtcs"],
        nprocs_glm=optiondict["nprocs_glm"],
        glmderivs=1,
        mp_chunksize=optiondict["mp_chunksize"],
        showprogressbar=optiondict["showprogressbar"],
        alwaysmultiproc=optiondict["alwaysmultiproc"],
        memprofile=optiondict["memprofile"],
        debug=debug,
    )

    # calculate the ratio of the first derivative to the main regressor
    glmderivratio = np.nan_to_num(fitcoeff[:, 1] / fitcoeff[:, 0])

    return glmderivratio


def filterderivratios(
    glmderivratio,
    nativespaceshape,
    validvoxels,
    thedims,
    patchthresh=3.0,
    gausssigma=0,
    fileiscifti=False,
    textio=False,
    rt_floattype="float64",
    debug=False,
):

    if debug:
        print("filterderivratios:")
        print(f"\t{patchthresh=}")
        print(f"\t{validvoxels.shape=}")
        print(f"\t{nativespaceshape=}")

    # filter the ratio to find weird values
    themad = mad(glmderivratio).astype(np.float64)
    print(f"MAD of GLM derivative ratios = {themad}")
    outmaparray, internalspaceshape = tide_io.makedestarray(
        nativespaceshape,
        textio=textio,
        fileiscifti=fileiscifti,
        rt_floattype=rt_floattype,
    )
    mappedglmderivratio = tide_io.populatemap(
        glmderivratio,
        internalspaceshape,
        validvoxels,
        outmaparray,
        debug=debug,
    )
    if textio or fileiscifti:
        medfilt = glmderivratio
        filteredarray = glmderivratio
    else:
        if debug:
            print(f"{glmderivratio.shape=}, {mappedglmderivratio.shape=}")
        medfilt = median_filter(
            mappedglmderivratio.reshape(nativespaceshape), size=(3, 3, 3)
        ).reshape(internalspaceshape)[validvoxels]
        filteredarray = np.where(
            np.fabs(glmderivratio - medfilt) > patchthresh * themad, medfilt, glmderivratio
        )
        if gausssigma > 0:
            mappedfilteredarray = tide_io.populatemap(
                filteredarray,
                internalspaceshape,
                validvoxels,
                outmaparray,
                debug=debug,
            )
            filteredarray = tide_filt.ssmooth(
                thedims[0],
                thedims[1],
                thedims[2],
                gausssigma,
                mappedfilteredarray.reshape(nativespaceshape),
            ).reshape(internalspaceshape)[validvoxels]

    return medfilt, filteredarray, themad
