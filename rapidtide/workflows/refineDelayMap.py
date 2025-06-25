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
    sLFOfiltmask,
    genlagtc,
    oversamptr,
    sLFOfitmean,
    rvalue,
    r2value,
    fitNorm,
    fitcoeff,
    lagtc,
    outputname,
    validvoxels,
    nativespaceshape,
    theinputdata,
    lagtimes,
    optiondict,
    LGR,
    TimingLGR,
    outputlevel="normal",
    gausssigma=-1,
    patchthresh=3.0,
    mindelay=-5.0,
    maxdelay=5.0,
    numpoints=501,
    histlen=101,
    rt_floatset=np.float64,
    rt_floattype="float64",
    debug=False,
):
    # do the calibration
    TimingLGR.info("Refinement calibration start")
    regressderivratios, regressrvalues = tide_refinedelay.getderivratios(
        fmri_data_valid,
        validvoxels,
        initial_fmri_x,
        lagtimes,
        sLFOfiltmask,
        genlagtc,
        "glm",
        outputname,
        oversamptr,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm[:, :2],
        fitcoeff[:, :2],
        None,
        lagtc,
        None,
        LGR,
        TimingLGR,
        optiondict,
        regressderivs=1,
        debug=debug,
    )

    medfiltregressderivratios, filteredregressderivratios, delayoffsetMAD = (
        tide_refinedelay.filterderivratios(
            regressderivratios,
            nativespaceshape,
            validvoxels,
            (xdim, ydim, slicethickness),
            gausssigma=gausssigma,
            patchthresh=patchthresh,
            filetype=theinputdata.filetype,
            rt_floattype=rt_floattype,
            debug=debug,
        )
    )

    # find the mapping of derivative ratios to delays
    tide_refinedelay.trainratiotooffset(
        genlagtc,
        initial_fmri_x,
        outputname,
        outputlevel,
        mindelay=mindelay,
        maxdelay=maxdelay,
        numpoints=numpoints,
        debug=debug,
    )

    # now calculate the delay offsets
    delayoffset = np.zeros_like(filteredregressderivratios)
    if debug:
        print(f"calculating delayoffsets for {filteredregressderivratios.shape[0]} voxels")
    for i in range(filteredregressderivratios.shape[0]):
        delayoffset[i], closestoffset = tide_refinedelay.ratiotodelay(
            filteredregressderivratios[i]
        )

    namesuffix = "_desc-delayoffset_hist"
    tide_stats.makeandsavehistogram(
        delayoffset[np.where(sLFOfiltmask > 0)],
        histlen,
        1,
        outputname + namesuffix,
        displaytitle="Histogram of delay offsets calculated from regression coefficients",
        dictvarname="delayoffsethist",
        thedict=optiondict,
    )

    return (
        delayoffset,
        regressderivratios,
        medfiltregressderivratios,
        filteredregressderivratios,
        delayoffsetMAD,
    )
