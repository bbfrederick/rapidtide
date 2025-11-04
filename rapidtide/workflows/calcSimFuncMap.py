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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.calcsimfunc as tide_calcsimfunc
import rapidtide.io as tide_io
import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.makelaggedtcs as tide_makelagged
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util


def makeRIPTiDeRegressors(
    initial_fmri_x: Any,
    lagmin: Any,
    lagmax: Any,
    lagtcgenerator: Any,
    LGR: Any,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    targetstep: float = 2.5,
    edgepad: int = 0,
    rt_floatset: Any = np.float64,
    rt_floattype: str = "float64",
    debug: bool = False,
) -> None:
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
    numvalidspatiallocs: Any,
    fmri_data_valid: Any,
    validsimcalcstart: Any,
    validsimcalcend: Any,
    osvalidsimcalcstart: Any,
    osvalidsimcalcend: Any,
    initial_fmri_x: Any,
    os_fmri_x: Any,
    theCorrelator: Any,
    theMutualInformationator: Any,
    cleaned_referencetc: Any,
    corrout: Any,
    regressorset: Any,
    delayvals: Any,
    sLFOfitmean: Any,
    r2value: Any,
    fitcoeff: Any,
    fitNorm: Any,
    meanval: Any,
    corrscale: Any,
    outputname: Any,
    outcorrarray: Any,
    validvoxels: Any,
    nativecorrshape: Any,
    theinputdata: Any,
    theheader: Any,
    lagmininpts: Any,
    lagmaxinpts: Any,
    thepass: Any,
    optiondict: Any,
    LGR: Any,
    TimingLGR: Any,
    similaritymetric: str = "correlation",
    simcalcoffset: int = 0,
    echocancel: bool = False,
    checkpoint: bool = False,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    oversampfactor: int = 2,
    interptype: str = "univariate",
    showprogressbar: bool = True,
    chunksize: int = 1000,
    rt_floatset: Any = np.float64,
    rt_floattype: str = "float64",
    mklthreads: int = 1,
    threaddebug: bool = False,
    debug: bool = False,
) -> None:
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
                constantevs=True,
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
