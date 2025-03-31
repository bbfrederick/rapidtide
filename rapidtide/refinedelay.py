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
import numpy.polynomial.polynomial as poly
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.ndimage import median_filter
from statsmodels.robust import mad

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.workflows.regressfrommaps as tide_regressfrommaps

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
    trainwidth=0.0,
    trainstep=0.5,
    mindelay=-3.0,
    maxdelay=3.0,
    numpoints=501,
    smoothpts=3,
    edgepad=5,
    regressderivs=1,
    debug=False,
):
    global ratiotooffsetfunc, maplimits

    if debug:
        print("ratiotooffsetfunc:")
        lagtcgenerator.info(prefix="\t")
        print("\ttimeaxis:", timeaxis)
        print("\toutputname:", outputname)
        print("\ttrainwidth:", trainwidth)
        print("\ttrainstep:", trainstep)
        print("\tmindelay:", mindelay)
        print("\tmaxdelay:", maxdelay)
        print("\tsmoothpts:", smoothpts)
        print("\tedgepad:", edgepad)
        print("\tregressderivs:", regressderivs)
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

    # set up for getratioderivs call
    rt_floattype = "float64"
    internalvalidfmrishape = (numpoints + 2 * edgepad, timeaxis.shape[0])
    fmridata = np.zeros(internalvalidfmrishape, dtype=float)
    fmrimask = np.ones(numpoints + 2 * edgepad, dtype=float)
    validvoxels = np.where(fmrimask > 0)[0]
    sLFOfitmean = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    rvalue = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    r2value = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    fitNorm = np.zeros((numpoints + 2 * edgepad, 2), dtype=rt_floattype)
    fitcoeff = np.zeros((numpoints + 2 * edgepad, 2), dtype=rt_floattype)
    movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    sampletime = timeaxis[1] - timeaxis[0]
    optiondict = {
        "regressfiltthreshval": 0.0,
        "saveminimumsLFOfiltfiles": False,
        "nprocs_makelaggedtcs": 1,
        "nprocs_regressionfilt": 1,
        "mp_chunksize": 1000,
        "showprogressbar": False,
        "alwaysmultiproc": False,
        "memprofile": False,
        "focaldebug": debug,
        "fmrifreq": 1.0 / sampletime,
        "textio": False,
    }

    if trainwidth > 0.0:
        numsteps = int(trainwidth / trainstep)
        numsteps += 1 - numsteps % 2  # force numsteps to be odd
        numsteps = np.max((numsteps, 3))  # ensure at least 1 positive and 1 negative step
        trainoffsets = (
            np.linspace(0, numsteps * trainstep, numsteps, endpoint=True)
            - (numsteps // 2) * trainstep
        )
    else:
        trainoffsets = np.array([0.0], dtype=float)
    if debug:
        print("trainoffsets:", trainoffsets)

    for i in range(len(trainoffsets)):
        pass

    # now make synthetic fMRI data
    for i in range(numpoints + 2 * edgepad):
        fmridata[i, :] = lagtcgenerator.yfromx(timeaxis - lagtimes[i])

    regressderivratios, regressrvalues = getderivratios(
        fmridata,
        validvoxels,
        timeaxis,
        0.0 * lagtimes,
        fmrimask,
        lagtcgenerator,
        "glm",
        "refinedelaytest",
        sampletime,
        sLFOfitmean,
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
        regressderivs=regressderivs,
        debug=debug,
    )
    if debug:
        print("before trimming")
        print(f"{regressderivratios.shape=}")
        print(f"{lagtimes.shape=}")
    if regressderivs == 1:
        smoothregressderivratios = tide_filt.unpadvec(
            smooth(tide_filt.padvec(regressderivratios, padlen=20, padtype="constant"), smoothpts),
            padlen=20,
        )
        regressderivratios = regressderivratios[edgepad:-edgepad]
        smoothregressderivratios = smoothregressderivratios[edgepad:-edgepad]
    else:
        smoothregressderivratios = np.zeros_like(regressderivratios)
        for i in range(regressderivs):
            smoothregressderivratios[i, :] = tide_filt.unpadvec(
                smooth(
                    tide_filt.padvec(regressderivratios[i, :], padlen=20, padtype="constant"),
                    smoothpts,
                ),
                padlen=20,
            )
        regressderivratios = regressderivratios[:, edgepad:-edgepad]
        smoothregressderivratios = smoothregressderivratios[:, edgepad:-edgepad]
    lagtimes = lagtimes[edgepad:-edgepad]
    if debug:
        print("after trimming")
        print(f"{regressderivratios.shape=}")
        print(f"{smoothregressderivratios.shape=}")
        print(f"{lagtimes.shape=}")

    # make sure the mapping function is legal
    xaxis = smoothregressderivratios[::-1]
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
        resampaxis = np.linspace(xaxis[0], xaxis[-1], num=len(xaxis), endpoint=True)
        tide_io.writebidstsv(
            f"{outputname}_desc-ratiotodelayfunc_timeseries",
            ratiotooffsetfunc(resampaxis),
            1.0 / (resampaxis[1] - resampaxis[0]),
            starttime=resampaxis[0],
            columns=["delay"],
            extraheaderinfo={
                "Description": "The function mapping derivative ratio to delay",
                "minratio": f"{resampaxis[0]}",
                "maxratio": f"{resampaxis[-1]}",
            },
            xaxislabel="coefficientratio",
            yaxislabel="time",
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


def coffstodelay(thecoffs, mindelay=-3.0, maxdelay=3.0, debug=False):
    justaone = np.array([1.0], dtype=thecoffs.dtype)
    allcoffs = np.concatenate((justaone, thecoffs))
    theroots = (poly.Polynomial(allcoffs, domain=(mindelay, maxdelay))).roots()
    if theroots is None:
        return 0.0
    elif len(theroots) == 1:
        return theroots[0].real
    else:
        candidates = []
        for i in range(len(theroots)):
            if np.isreal(theroots[i]) and (mindelay <= theroots[i] <= maxdelay):
                if debug:
                    print(f"keeping root {i} ({theroots[i]})")
                candidates.append(theroots[i].real)
            else:
                if debug:
                    print(f"discarding root {i} ({theroots[i]})")
                else:
                    pass
        if len(candidates) > 0:
            chosen = candidates[np.argmin(np.fabs(np.array(candidates)))].real
            if debug:
                print(f"{theroots=}, {candidates=}, {chosen=}")
            return chosen
        return 0.0


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
    sLFOfitmean,
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
    regressderivs=1,
    starttr=None,
    endtr=None,
    debug=False,
):
    if starttr is None:
        starttr = 0
    if endtr is None:
        endtr = fmri_data_valid.shape[1]
    if debug:
        print("getderivratios")
        print(f"{fitNorm.shape=}")
        print(f"{fitcoeff.shape=}")
        print(f"{regressderivs=}")
        print(f"{starttr=}")
        print(f"{endtr=}")

    voxelsprocessed_regressionfilt, regressorset, evset = tide_regressfrommaps.regressfrommaps(
        fmri_data_valid[:, starttr:endtr],
        validvoxels,
        initial_fmri_x[starttr:endtr],
        lagtimes,
        fitmask,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm,
        fitcoeff,
        movingsignal,
        lagtc,
        filtereddata,
        LGR,
        TimingLGR,
        optiondict["regressfiltthreshval"],
        optiondict["saveminimumsLFOfiltfiles"],
        nprocs_makelaggedtcs=optiondict["nprocs_makelaggedtcs"],
        nprocs_regressionfilt=optiondict["nprocs_regressionfilt"],
        regressderivs=regressderivs,
        mp_chunksize=optiondict["mp_chunksize"],
        showprogressbar=optiondict["showprogressbar"],
        alwaysmultiproc=optiondict["alwaysmultiproc"],
        memprofile=optiondict["memprofile"],
        debug=debug,
    )

    # calculate the ratio of the first derivative to the main regressor
    if regressderivs == 1:
        regressderivratios = np.nan_to_num(fitcoeff[:, 1] / fitcoeff[:, 0])
    else:
        numvoxels = fitcoeff.shape[0]
        regressderivratios = np.zeros((regressderivs, numvoxels), dtype=np.float64)
        for i in range(regressderivs):
            regressderivratios[i, :] = np.nan_to_num(fitcoeff[:, i + 1] / fitcoeff[:, 0])

    return regressderivratios, rvalue


def filterderivratios(
    regressderivratios,
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
    themad = mad(regressderivratios).astype(np.float64)
    print(f"MAD of regression fit derivative ratios = {themad}")
    outmaparray, internalspaceshape = tide_io.makedestarray(
        nativespaceshape,
        textio=textio,
        fileiscifti=fileiscifti,
        rt_floattype=rt_floattype,
    )
    mappedregressderivratios = tide_io.populatemap(
        regressderivratios,
        internalspaceshape,
        validvoxels,
        outmaparray,
        debug=debug,
    )
    if textio or fileiscifti:
        medfilt = regressderivratios
        filteredarray = regressderivratios
    else:
        if debug:
            print(f"{regressderivratios.shape=}, {mappedregressderivratios.shape=}")
        medfilt = median_filter(
            mappedregressderivratios.reshape(nativespaceshape), size=(3, 3, 3)
        ).reshape(internalspaceshape)[validvoxels]
        filteredarray = np.where(
            np.fabs(regressderivratios - medfilt) > patchthresh * themad,
            medfilt,
            regressderivratios,
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
