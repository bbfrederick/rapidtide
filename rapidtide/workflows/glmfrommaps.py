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
    fmrifilename,
    theprefilter,
    lagtimes,
    fitmask,
    genlagtc,
    mode,
    outputname,
    fmritr,
    oversamptr,
    LGR,
    TimingLGR,
    fileiscifti,
    textio,
    numspatiallocs,
    timepoints,
    validstart,
    validend,
    validvoxels,
    internalvalidspaceshape,
    internalvalidfmrishape,
    initial_fmri_x,
    threshval,
    gausssigma=0.0,
    nprocs=1,
    glmderivs=0,
    mp_chunksize=50000,
    showprogressbar=True,
    alwaysmultiproc=False,
    memprofile=False,
    usesharedmem=False,
    glmsourcefile=None,
):
    rt_floatset = np.float64
    rt_outfloatset = np.float64
    rt_floattype = "float64"
    rt_outfloattype = np.float64
    numvalidspatiallocs = np.shape(validvoxels)[0]

    # GLM fitting, either to remove moving signal, or to calculate delayed CVR
    # write out the current version of the run options
    if mode == "glm":
        TimingLGR.info("GLM filtering start")
        LGR.info("\n\nGLM filtering")
    else:
        TimingLGR.info("CVR map generation start")
        LGR.info("\n\nCVR mapping")
    if (gausssigma > 0.0) or (glmsourcefile is not None) or mode == "cvrmap":
        if glmsourcefile is not None:
            LGR.info(f"reading in {glmsourcefile} for GLM filter, please wait")
            sourcename = glmsourcefile
        else:
            LGR.info(f"rereading {fmrifilename} for GLM filter, please wait")
            sourcename = fmrifilename
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
            ) = tide_io.readfromcifti(sourcename)
        else:
            if textio:
                nim_data = tide_io.readvecs(sourcename)
            else:
                nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(sourcename)

        fmri_data_valid = (
            nim_data.reshape((numspatiallocs, timepoints))[:, validstart : validend + 1]
        )[validvoxels, :] + 0.0

        if mode == "cvrmap":
            # percent normalize the fmri data
            LGR.info("normalzing data for CVR map")
            themean = np.mean(fmri_data_valid, axis=1)
            fmri_data_valid /= themean[:, None]

        # move fmri_data_valid into shared memory
        if usesharedmem:
            LGR.info("moving fmri data to shared memory")
            TimingLGR.info("Start moving fmri_data to shared memory")
            numpy2shared_func = addmemprofiling(
                tide_util.numpy2shared,
                memprofile,
                "before movetoshared (glm)",
            )
            fmri_data_valid = numpy2shared_func(fmri_data_valid, rt_floatset)
            TimingLGR.info("End moving fmri_data to shared memory")
        del nim_data

    # now allocate the arrays needed for GLM filtering
    internalvalidspaceshapederivs = (
        internalvalidspaceshape,
        glmderivs + 1,
    )
    if usesharedmem:
        glmmean, dummy, dummy = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        rvalue, dummy, dummy = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        r2value, dummy, dummy = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        fitNorm, dummy, dummy = tide_util.allocshared(
            internalvalidspaceshapederivs, rt_outfloatset
        )
        fitcoeff, dummy, dummy = tide_util.allocshared(
            internalvalidspaceshapederivs, rt_outfloatset
        )
        movingsignal, dummy, dummy = tide_util.allocshared(internalvalidfmrishape, rt_outfloatset)
        lagtc, dummy, dummy = tide_util.allocshared(internalvalidfmrishape, rt_floatset)
        filtereddata, dummy, dummy = tide_util.allocshared(internalvalidfmrishape, rt_outfloatset)
    else:
        glmmean = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        rvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        r2value = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitNorm = np.zeros(internalvalidspaceshapederivs, dtype=rt_outfloattype)
        fitcoeff = np.zeros(internalvalidspaceshapederivs, dtype=rt_outfloattype)
        movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
        lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
        filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)

    if memprofile:
        if mode == "glm":
            memcheckpoint("about to start glm noise removal...")
        else:
            memcheckpoint("about to start CVR magnitude estimation...")
    else:
        tide_util.logmem("before glm")

    # generate the voxel specific regressors
    LGR.info("Start lagged timecourse creation")
    TimingLGR.info("Start lagged timecourse creation")
    makelagged_func = addmemprofiling(
        tide_makelagged.makelaggedtcs,
        memprofile,
        "before making lagged timecourses",
    )
    voxelsprocessed_makelagged = tide_makelagged.makelaggedtcs(
        genlagtc,
        initial_fmri_x,
        fitmask,
        lagtimes,
        lagtc,
        nprocs=nprocs,
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
        nprocs=nprocs,
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

    # calculate the final bandlimited mean normalized variance
    finalvariance = tide_math.imagevariance(filtereddata, theprefilter, 1.0 / fmritr)

    del fmri_data_valid

    LGR.info("End filtering operation")
    TimingLGR.info(
        "GLM filtering end",
        {
            "message2": voxelsprocessed_glm,
            "message3": "voxels",
        },
    )
    if memprofile:
        memcheckpoint("...done")
    else:
        tide_util.logmem("after glm filter")
    LGR.info("")

    return (
        glmmean,
        rvalue,
        r2value,
        fitNorm,
        fitcoeff,
        movingsignal,
        lagtc,
        filtereddata,
        finalvariance,
    )
