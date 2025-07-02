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

import rapidtide.calcsimfunc as tide_calcsimfunc
import rapidtide.io as tide_io
import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.makelaggedtcs as tide_makelagged
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util


def makeRIPTiDeRegressors(
    initial_fmri_x,
    lagmin,
    lagmax,
    lagtcgenerator,
    LGR,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    targetstep=2.5,
    edgepad=0,
    rt_floatset=np.float64,
    rt_floattype="float64",
    debug=False,
):
    # make the RIPTiDe evs
    numdelays = int(np.round((lagmax - lagmin) / targetstep, 0))
    numregressors = numdelays + 2 * edgepad
    delaystep = (lagmax - lagmin) / numdelays
    delaystouse = np.linspace(
        lagmin - edgepad * delaystep,
        lagmax + edgepad * delaystep,
        numdelays + 2 * edgepad,
        endpoint=True,
    )
    if debug:
        print(f"{lagmin=}")
        print(f"{lagmax=}")
        print(f"{numdelays=}")
        print(f"{edgepad=}")
        print(f"{numregressors=}")
        print(f"{delaystep=}")
        print(f"{delaystouse=}, {len(delaystouse)}")
        print(f"{len(initial_fmri_x)}")

    regressorset = np.zeros((len(delaystouse), len(initial_fmri_x)), dtype=rt_floatset)

    dummy = tide_makelagged.makelaggedtcs(
        lagtcgenerator,
        initial_fmri_x,
        np.ones_like(delaystouse, dtype=np.float64),
        delaystouse,
        regressorset,
        LGR=LGR,
        nprocs=nprocs,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        chunksize=chunksize,
        rt_floatset=rt_floatset,
        rt_floattype=rt_floattype,
        debug=debug,
    )

    if debug:
        print(regressorset)

    return regressorset, delaystouse


def calcSimFunc(
    numvalidspatiallocs,
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
    regressorset,
    delayvals,
    sLFOfitmean,
    r2value,
    fitcoeff,
    fitNorm,
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
    elif similaritymetric == "riptide":
        similaritytype = "RIPTiDe"
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
    elif (similaritymetric == "correlation") or (similaritymetric == "hybrid"):
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
    elif similaritymetric == "riptide":
        # do the linear fit to the comb of delayed regressors
        for thedelay in range(len(delayvals)):
            print(f"Fitting delay {delayvals[thedelay]:.2f}")
            voxelsprocessed_cp = tide_linfitfiltpass.linfitfiltpass(
                numvalidspatiallocs,
                fmri_data_valid[:, validsimcalcstart : validsimcalcend + 1],
                0.0,
                regressorset[thedelay, validsimcalcstart : validsimcalcend + 1],
                sLFOfitmean,
                corrout[:, thedelay],
                r2value,
                fitcoeff,
                fitNorm,
                None,
                None,
                coefficientsonly=True,
                voxelspecific=False,
                nprocs=nprocs,
                alwaysmultiproc=alwaysmultiproc,
                showprogressbar=showprogressbar,
                verbose=(LGR is not None),
                chunksize=chunksize,
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype,
                debug=debug,
            )
    else:
        print("illegal similarity metric")

    tide_util.enablemkl(mklthreads, debug=threaddebug)

    if similaritymetric != "riptide":
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
