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
from scipy.ndimage import median_filter
from statsmodels.robust import mad

import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.workflows.glmfrommaps as tide_glmfrommaps


def refinedelay(
    fmri_data_valid,
    nativespaceshape,
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
    theheader,
    LGR,
    TimingLGR,
    optiondict,
    bidsbasedict,
    cifti_hdr,
    patchthresh=3.0,
    fileiscifti=False,
    textio=False,
    rt_floattype="float64",
    rt_floatset=np.float64,
):
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
        debug=optiondict["focaldebug"],
    )

    # calculate the ratio of the first derivative to the main regressor
    glmderivratio = np.nan_to_num(fitcoeff[:, 1] / fitcoeff[:, 0])
    delayoffset = -glmderivratio * optiondict["fmrifreq"]

    # filter the ratio to find weird values
    themad = mad(delayoffset).astype(np.float64)
    print(f"MAD of delay offsets = {themad}")
    outmaparray, internalspaceshape = tide_io.makedestarray(
        nativespaceshape,
        textio=textio,
        fileiscifti=fileiscifti,
        rt_floattype=rt_floattype,
    )
    mappeddelayoffset = tide_io.populatemap(
        delayoffset,
        internalspaceshape,
        validvoxels,
        outmaparray,
        debug=False,
    )
    if textio or fileiscifti:
        medfilt = delayoffset
        filteredarray = delayoffset
    else:
        print(f"{delayoffset.shape=}, {mappeddelayoffset.shape=}")
        medfilt = median_filter(
            mappeddelayoffset.reshape(nativespaceshape), size=(3, 3, 3)
        ).reshape(internalspaceshape)[validvoxels]
        filteredarray = np.where(
            np.fabs(delayoffset - medfilt) > patchthresh * themad, medfilt, delayoffset
        )
    savelist = [
        (glmderivratio, "glmderivratio", "map", None, "GLM derivative ratio"),
        (
            delayoffset,
            "rawdelayoffset",
            "map",
            "sec",
            "Delay offset calculated from GLM derivative ratio",
        ),
        (medfilt, "medfiltdelayoffset", "map", "sec", "Delay offset, median filtered"),
        (
            filteredarray,
            "delayoffset",
            "map",
            "sec",
            "Delay offset, selectively median filtered",
        ),
    ]
    if not optiondict["textio"]:
        if fileiscifti:
            timeindex = theheader["dim"][0] - 1
            spaceindex = theheader["dim"][0]
            theheader["dim"][timeindex] = 1
            theheader["dim"][spaceindex] = filteredarray.shape[0]
        else:
            theheader["dim"][0] = 3
            theheader["dim"][4] = 1
            theheader["pixdim"][4] = 1.0
    else:
        theheader = None
        cifti_hdr = None
    tide_io.savemaplist(
        outputname,
        savelist,
        validvoxels,
        nativespaceshape,
        theheader,
        bidsbasedict,
        textio=optiondict["textio"],
        fileiscifti=fileiscifti,
        rt_floattype=rt_floattype,
        cifti_hdr=cifti_hdr,
    )

    ratiolist = np.linspace(-patchthresh, patchthresh, 21, endpoint=True)
    outtcs = np.zeros((len(ratiolist), evset.shape[0]), dtype=rt_floattype)
    print(f"{glmderivratio.shape=}, {ratiolist.shape=}, {evset.shape=}, {outtcs.shape=}")
    colnames = []
    for ratioidx, theratio in enumerate(ratiolist):
        # print(f"{ratioidx=}, {theratio=}")
        outtcs[ratioidx, :] = tide_math.stdnormalize(evset[:, 0] + theratio * evset[:, 1])
        colnames.append(f"ratio_{str(theratio)}")
    tide_io.writebidstsv(
        f"{outputname}_desc-glmratio_timeseries",
        outtcs,
        optiondict["fmrifreq"],
        columns=colnames,
        extraheaderinfo={"Description": "GLM regressor for various derivative ratios"},
        append=False,
    )
    return delayoffset
