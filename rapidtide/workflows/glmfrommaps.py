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

import rapidtide.glmpass as tide_glmpass
import rapidtide.io as tide_io
import rapidtide.makelaggedtcs as tide_makelagged
import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util

try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False


def conditionalprofile():
    def resdec(f):
        if memprofilerexists:
            return profile(f)
        return f

    return resdec


@conditionalprofile()
def memcheckpoint(message):
    pass


def addmemprofiling(thefunc, memprofile, themessage):
    if memprofile:
        return profile(thefunc, precision=2)
    else:
        tide_util.logmem(themessage)
        return thefunc


def glmfrommaps(
    fmri_data_valid,
    glmmean,
    rvalue,
    r2value,
    fitNorm,
    fitcoeff,
    movingsignal,
    lagtc,
    filtereddata,
    lagtimes,
    fitmask,
    genlagtc,
    mode,
    outputname,
    oversamptr,
    LGR,
    TimingLGR,
    validvoxels,
    initial_fmri_x,
    threshval,
    nprocs_makelaggedtcs=1,
    nprocs_glm=1,
    glmderivs=0,
    mp_chunksize=50000,
    showprogressbar=True,
    alwaysmultiproc=False,
    memprofile=False,
    debug=False,
):
    if debug:
        print("GLMFROMMAPS: Starting")
    rt_floatset = np.float64
    rt_floattype = "float64"
    numvalidspatiallocs = np.shape(validvoxels)[0]

    # generate the voxel specific regressors
    LGR.info("Start lagged timecourse creation")
    TimingLGR.info("Start lagged timecourse creation")
    makelagged_func = addmemprofiling(
        tide_makelagged.makelaggedtcs,
        memprofile,
        "before making lagged timecourses",
    )
    voxelsprocessed_makelagged = makelagged_func(
        genlagtc,
        initial_fmri_x,
        fitmask,
        lagtimes,
        lagtc,
        nprocs=nprocs_makelaggedtcs,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        chunksize=mp_chunksize,
        rt_floatset=rt_floatset,
        rt_floattype=rt_floattype,
    )
    LGR.info("End lagged timecourse creation")
    TimingLGR.info(
        "Lagged timecourse creation end",
        {
            "message2": voxelsprocessed_makelagged,
            "message3": "voxels",
        },
    )

    # and do the filtering
    LGR.info("Start filtering operation")
    TimingLGR.info("Start filtering operation")
    glmpass_func = addmemprofiling(tide_glmpass.glmpass, memprofile, "before glmpass")
    if mode == "cvrmap":
        # set the threshval to zero
        glmthreshval = 0.0
    else:
        glmthreshval = threshval

    if glmderivs > 0:
        print(f"adding derivatives up to order {glmderivs} prior to regression")
        regressorset = tide_glmpass.makevoxelspecificderivs(lagtc, glmderivs)
    else:
        regressorset = lagtc
    voxelsprocessed_glm = glmpass_func(
        numvalidspatiallocs,
        fmri_data_valid,
        glmthreshval,
        regressorset,
        glmmean,
        rvalue,
        r2value,
        fitcoeff,
        fitNorm,
        movingsignal,
        filtereddata,
        nprocs=nprocs_glm,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        mp_chunksize=mp_chunksize,
        rt_floatset=rt_floatset,
        rt_floattype=rt_floattype,
    )
    if fitcoeff.ndim > 1:
        fitcoeff = fitcoeff[:, 0]
        fitNorm = fitNorm[:, 0]

    if mode == "cvrmap":
        # if we are doing a cvr map, multiply the fitcoeff by 100, so we are in percent
        fitcoeff *= 100.0

    # determine what was removed
    removeddata = fmri_data_valid - filtereddata
    noiseremoved = np.var(removeddata, axis=0)
    tide_io.writebidstsv(
        f"{outputname}_desc-lfoNoiseRemoved_timeseries",
        noiseremoved,
        1.0 / oversamptr,
        starttime=0.0,
        columns=[f"removedbyglm"],
        append=False,
    )

    return voxelsprocessed_glm
