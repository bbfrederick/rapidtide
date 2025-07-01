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
from scipy import ndimage

import rapidtide.calcsimfunc as tide_calcsimfunc
import rapidtide.io as tide_io
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util


def simfuncDelay(
    fmri_data_valid,
    validsimcalcstart,
    validsimcalcend,
    osvalidsimcalcstart,
    osvalidsimcalcend,
    initial_fmri_x,
    os_fmri_x,
    theCorrelator,
    theMutualInformationator,
    cleaned_referencetc,
    corrout,
    meanval,
    corrscale,
    outputname,
    outcorrarray,
    validvoxels,
    nativecorrshape,
    theinputdata,
    theheader,
    lagmininpts,
    lagmaxinpts,
    thepass,
    optiondict,
    LGR,
    TimingLGR,
    similaritymetric="correlation",
    simcalcoffset=0,
    echocancel=False,
    checkpoint=False,
    nprocs=1,
    alwaysmultiproc=False,
    oversampfactor=2,
    interptype="univariate",
    showprogressbar=True,
    chunksize=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
    mklthreads=1,
    threaddebug=False,
    debug=False,
):
    # Step 1 - Correlation step
    if similaritymetric == "mutualinfo":
        similaritytype = "Mutual information"
    elif similaritymetric == "correlation":
        similaritytype = "Correlation"
    else:
        similaritytype = "MI enhanced correlation"
    LGR.info(f"\n\n{similaritytype} calculation, pass {thepass}")
    TimingLGR.info(f"{similaritytype} calculation start, pass {thepass}")

    tide_util.disablemkl(nprocs, debug=threaddebug)
    if similaritymetric == "mutualinfo":
        theMutualInformationator.setlimits(lagmininpts, lagmaxinpts)
        (
            voxelsprocessed_cp,
            theglobalmaxlist,
            trimmedcorrscale,
        ) = tide_calcsimfunc.correlationpass(
            fmri_data_valid[:, validsimcalcstart : validsimcalcend + 1],
            cleaned_referencetc,
            theMutualInformationator,
            initial_fmri_x[validsimcalcstart : validsimcalcend + 1],
            os_fmri_x[osvalidsimcalcstart : osvalidsimcalcend + 1],
            lagmininpts,
            lagmaxinpts,
            corrout,
            meanval,
            nprocs=nprocs,
            alwaysmultiproc=alwaysmultiproc,
            oversampfactor=oversampfactor,
            interptype=interptype,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
            debug=debug,
        )
    else:
        (
            voxelsprocessed_cp,
            theglobalmaxlist,
            trimmedcorrscale,
        ) = tide_calcsimfunc.correlationpass(
            fmri_data_valid[:, validsimcalcstart : validsimcalcend + 1],
            cleaned_referencetc,
            theCorrelator,
            initial_fmri_x[validsimcalcstart : validsimcalcend + 1],
            os_fmri_x[osvalidsimcalcstart : osvalidsimcalcend + 1],
            lagmininpts,
            lagmaxinpts,
            corrout,
            meanval,
            nprocs=nprocs,
            alwaysmultiproc=alwaysmultiproc,
            oversampfactor=oversampfactor,
            interptype=interptype,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
            debug=debug,
        )
    tide_util.enablemkl(mklthreads, debug=threaddebug)

    for i in range(len(theglobalmaxlist)):
        theglobalmaxlist[i] = corrscale[theglobalmaxlist[i]] - simcalcoffset
    namesuffix = "_desc-globallag_hist"
    tide_stats.makeandsavehistogram(
        np.asarray(theglobalmaxlist),
        len(corrscale),
        0,
        outputname + namesuffix,
        displaytitle="Histogram of lag times from global lag calculation",
        therange=(corrscale[0], corrscale[-1]),
        refine=False,
        dictvarname="globallaghist_pass" + str(thepass),
        append=(echocancel or (thepass > 1)),
        thedict=optiondict,
    )

    if checkpoint:
        outcorrarray[:, :] = 0.0
        outcorrarray[validvoxels, :] = corrout[:, :]
        if theinputdata.filetype == "text":
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

    return similaritytype
