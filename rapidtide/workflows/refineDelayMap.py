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
import numpy as np

import rapidtide.refinedelay as tide_refinedelay
import rapidtide.stats as tide_stats


def refineDelay(
    fmri_data_valid,
    initial_fmri_x,
    xdim,
    ydim,
    slicethickness,
    xsize,
    ysize,
    numslices,
    sLFOfiltmask,
    genlagtc,
    mode,
    oversamptr,
    sLFOfitmean,
    rvalue,
    r2value,
    fitNorm,
    fitcoeff,
    movingsignal,
    lagtc,
    filtereddata,
    outputname,
    validvoxels,
    nativespaceshape,
    theinputdata,
    lagtimes,
    optiondict,
    LGR,
    TimingLGR,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    # refine the delay value prior to calculating the sLFO filter
    regressderivratios, regressrvalues = tide_refinedelay.getderivratios(
        fmri_data_valid,
        validvoxels,
        initial_fmri_x,
        lagtimes,
        sLFOfiltmask,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm[:, : (optiondict["refineregressderivs"] + 1)],
        fitcoeff[:, : (optiondict["refineregressderivs"] + 1)],
        movingsignal,
        lagtc,
        filtereddata,
        LGR,
        TimingLGR,
        optiondict,
        regressderivs=optiondict["refineregressderivs"],
        debug=optiondict["debug"],
    )

    if optiondict["refineregressderivs"] == 1:
        medfiltregressderivratios, filteredregressderivratios, delayoffsetMAD = (
            tide_refinedelay.filterderivratios(
                regressderivratios,
                nativespaceshape,
                validvoxels,
                (xdim, ydim, slicethickness),
                gausssigma=optiondict["delayoffsetgausssigma"],
                patchthresh=optiondict["delaypatchthresh"],
                filetype=theinputdata.filetype,
                rt_floattype="float64",
                debug=optiondict["debug"],
            )
        )
        optiondict["delayoffsetMAD"] = delayoffsetMAD

        # find the mapping of derivative ratios to delays
        tide_refinedelay.trainratiotooffset(
            genlagtc,
            initial_fmri_x,
            outputname,
            optiondict["outputlevel"],
            mindelay=optiondict["mindelay"],
            maxdelay=optiondict["maxdelay"],
            numpoints=optiondict["numpoints"],
            debug=optiondict["debug"],
        )

        # now calculate the delay offsets
        delayoffset = np.zeros_like(filteredregressderivratios)
        if optiondict["debug"]:
            print(f"calculating delayoffsets for {filteredregressderivratios.shape[0]} voxels")
        for i in range(filteredregressderivratios.shape[0]):
            delayoffset[i], closestoffset = tide_refinedelay.ratiotodelay(
                filteredregressderivratios[i]
            )
    else:
        medfiltregressderivratios = np.zeros_like(regressderivratios)
        filteredregressderivratios = np.zeros_like(regressderivratios)
        delayoffsetMAD = np.zeros(optiondict["refineregressderivs"], dtype=float)
        for i in range(optiondict["refineregressderivs"]):
            (
                medfiltregressderivratios[i, :],
                filteredregressderivratios[i, :],
                delayoffsetMAD[i],
            ) = tide_refinedelay.filterderivratios(
                regressderivratios[i, :],
                (xsize, ysize, numslices),
                validvoxels,
                (xdim, ydim, slicethickness),
                gausssigma=optiondict["delayoffsetgausssigma"],
                patchthresh=optiondict["delaypatchthresh"],
                filetype=theinputdata.filetype,
                rt_floattype=rt_floattype,
                debug=optiondict["debug"],
            )
            optiondict[f"delayoffsetMAD_{i + 1}"] = delayoffsetMAD[i]

        # now calculate the delay offsets
        delayoffset = np.zeros_like(filteredregressderivratios[0, :])
        if optiondict["debug"]:
            print(f"calculating delayoffsets for {filteredregressderivratios.shape[1]} voxels")
        for i in range(filteredregressderivratios.shape[1]):
            delayoffset[i] = tide_refinedelay.coffstodelay(
                filteredregressderivratios[:, i],
                mindelay=optiondict["mindelay"],
                maxdelay=optiondict["maxdelay"],
            )

    namesuffix = "_desc-delayoffset_hist"
    if optiondict["dolinfitfilt"]:
        tide_stats.makeandsavehistogram(
            delayoffset[np.where(sLFOfiltmask > 0)],
            optiondict["histlen"],
            1,
            outputname + namesuffix,
            displaytitle="Histogram of delay offsets calculated from regression coefficients",
            dictvarname="delayoffsethist",
            thedict=optiondict,
        )

    return delayoffset, regressderivratios, medfiltregressderivratios, filteredregressderivratios
