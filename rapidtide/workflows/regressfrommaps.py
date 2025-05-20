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

import rapidtide.io as tide_io
import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.makelaggedtcs as tide_makelagged


def regressfrommaps(
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
    regressfiltthreshval,
    saveminimumsLFOfiltfiles,
    nprocs_makelaggedtcs=1,
    nprocs_regressionfilt=1,
    regressderivs=0,
    mp_chunksize=50000,
    showprogressbar=True,
    alwaysmultiproc=False,
    saveEVsandquit=False,
    debug=False,
):
    if debug:
        print("regressfrommaps: Starting")
        print(f"{nprocs_makelaggedtcs=}")
        print(f"{nprocs_regressionfilt=}")
        print(f"{regressderivs=}")
        print(f"{mp_chunksize=}")
        print(f"{showprogressbar=}")
        print(f"{alwaysmultiproc=}")
        print(f"{mode=}")
        print(f"{outputname=}")
        print(f"{oversamptr=}")
        print(f"{regressfiltthreshval=}")
    rt_floatset = np.float64
    rt_floattype = "float64"
    numvalidspatiallocs = np.shape(validvoxels)[0]

    # generate the voxel specific regressors
    if LGR is not None:
        LGR.info("Start lagged timecourse creation")
    if TimingLGR is not None:
        TimingLGR.info("Start lagged timecourse creation")

    voxelsprocessed_makelagged = tide_makelagged.makelaggedtcs(
        genlagtc,
        initial_fmri_x,
        fitmask,
        lagtimes,
        lagtc,
        LGR=LGR,
        nprocs=nprocs_makelaggedtcs,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        chunksize=mp_chunksize,
        rt_floatset=rt_floatset,
        rt_floattype=rt_floattype,
    )
    if debug:
        print(f"{lagtimes.shape=}")
        threshmask = np.where(fitmask > 0, 1, 0)
        print(f"{np.sum(threshmask)} nonzero mask voxels")
        print(f"after makelaggedtcs: shifted {voxelsprocessed_makelagged} timecourses")
    if LGR is not None:
        LGR.info("End lagged timecourse creation")
    if TimingLGR is not None:
        TimingLGR.info(
            "Lagged timecourse creation end",
            {
                "message2": voxelsprocessed_makelagged,
                "message3": "voxels",
            },
        )

    if regressderivs > 0:
        if debug:
            print(f"adding derivatives up to order {regressderivs} prior to regression")
        regressorset = tide_linfitfiltpass.makevoxelspecificderivs(lagtc, regressderivs)
        baseev = rt_floatset(genlagtc.yfromx(initial_fmri_x))
        evset = tide_linfitfiltpass.makevoxelspecificderivs(
            baseev.reshape((1, -1)), regressderivs
        ).reshape((-1, 2))
    else:
        if debug:
            print(f"using raw lagged regressors for regression")
        regressorset = lagtc
        evset = rt_floatset(genlagtc.yfromx(initial_fmri_x))
    if debug:
        print(f"{regressorset.shape=}")

    if saveEVsandquit:
        return 0, regressorset, evset

    # now do the filtering
    if LGR is not None:
        LGR.info("Start filtering operation")
    if TimingLGR is not None:
        TimingLGR.info("Start filtering operation")

    voxelsprocessed_regressionfilt = tide_linfitfiltpass.linfitfiltpass(
        numvalidspatiallocs,
        fmri_data_valid,
        regressfiltthreshval,
        regressorset,
        sLFOfitmean,
        rvalue,
        r2value,
        fitcoeff,
        fitNorm,
        movingsignal,
        filtereddata,
        nprocs=nprocs_regressionfilt,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        verbose=(LGR is not None),
        mp_chunksize=mp_chunksize,
        rt_floatset=rt_floatset,
        rt_floattype=rt_floattype,
    )

    if mode == "cvrmap":
        # if we are doing a cvr map, multiply the fitcoeff by 100, so we are in percent
        fitcoeff *= 100.0

    # determine what was removed
    removeddata = fmri_data_valid - filtereddata
    noiseremoved = np.var(removeddata, axis=0)
    if saveminimumsLFOfiltfiles:
        tide_io.writebidstsv(
            f"{outputname}_desc-lfofilterNoiseRemoved_timeseries",
            noiseremoved,
            1.0 / oversamptr,
            starttime=0.0,
            columns=[f"removedbyglm"],
            extraheaderinfo={
                "Description": "Variance over space of data removed by the sLFO filter at each timepoint"
            },
            append=False,
        )

    return voxelsprocessed_regressionfilt, regressorset, evset
