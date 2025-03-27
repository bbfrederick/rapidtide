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
"""
A class to impmement regressor refinement
"""
import copy

import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.refinedelay as tide_refinedelay
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
from rapidtide.tests.utils import mse


class DerivativeDelay:
    def __init__(
        self,
        internalvalidfmrishape,
        internalvalidpaddedfmrishape,
        pid,
        outputname,
        initial_fmri_x,
        paddedinitial_fmri_x,
        os_fmri_x,
        genlagtc,
        sharedmem=False,
        gausssigma=-1,
        numderivs=1,
        lagminthresh=0.25,
        lagmaxthresh=3.0,
        sigmathresh=1000.0,
        cleanrefined=False,
        bipolar=False,
        fixdelay=False,
        includemask=None,
        excludemask=None,
        LGR=None,
        nprocs=1,
        detrendorder=1,
        alwaysmultiproc=False,
        showprogressbar=True,
        chunksize=50000,
        padtrs=10,
        refineprenorm="var",
        refineweighting=None,
        refinetype="pca",
        pcacomponents=0.8,
        dodispersioncalc=False,
        dispersioncalc_lower=-5.0,
        dispersioncalc_upper=5.0,
        dispersioncalc_step=0.5,
        windowfunc="hamming",
        passes=3,
        maxpasses=15,
        convergencethresh=None,
        interptype="univariate",
        usetmask=False,
        tmask_y=None,
        tmaskos_y=None,
        fastresamplerpadtime=45.0,
        debug=False,
        rt_floattype="float64",
        rt_floatset=np.float64,
    ):
        self.outputname = outputname
        self.numderivs = numderivs
        if gausssigma < 0.0:
            # set gausssigma automatically
            self.gausssigma = np.mean([xdim, ydim, slicedim]) / 2.0
        else:
            self.gausssigma = gausssigma
        self.debug = debug
        self.setgenlagtc(genlagtc)

    def setgenlagtc(self, genlagtc):
        self.genlagtc = genlagtc

    def getderivratios(
        self, fmri_data_valid, validvoxels, initial_fmri_x, lagtimes_valid, corrmask_valid
    ):
        print("Refinement calibration start")
        regressderivratios = tide_refinedelay.getderivratios(
            fmri_data_valid,
            validvoxels,
            initial_fmri_x,
            lagtimes_valid,
            corrmask_valid,
            self.genlagtc,
            "glm",
            self.outputname,
            oversamptr,
            sLFOfitmean,
            rvalue,
            r2value,
            fitNorm[:, : (self.numderivs + 1)],
            fitcoeff[:, : (self.numderivs + 1)],
            movingsignal,
            lagtc,
            filtereddata,
            LGR,
            TimingLGR,
            therunoptions,
            regressderivs=self.numderivs,
            debug=self.debug,
        )

    def calibrate(self):
        if self.numderivs == 1:
            medfiltregressderivratios, filteredregressderivratios, delayoffsetMAD = (
                tide_refinedelay.filterderivratios(
                    regressderivratios,
                    (xsize, ysize, numslices),
                    validvoxels,
                    (xdim, ydim, slicedim),
                    gausssigma=args.delayoffsetgausssigma,
                    patchthresh=args.delaypatchthresh,
                    fileiscifti=False,
                    textio=False,
                    rt_floattype=rt_floattype,
                    debug=args.debug,
                )
            )

            # find the mapping of derivative ratios to delays
            tide_refinedelay.trainratiotooffset(
                self.genlagtc,
                initial_fmri_x,
                self.outputname,
                args.outputlevel,
                mindelay=args.mindelay,
                maxdelay=args.maxdelay,
                numpoints=args.numpoints,
                debug=args.debug,
            )
            TimingLGR.info("Refinement calibration end")

            # now calculate the delay offsets
            TimingLGR.info("Calculating delay offsets")
            delayoffset = np.zeros_like(filteredregressderivratios)
            if args.focaldebug:
                print(f"calculating delayoffsets for {filteredregressderivratios.shape[0]} voxels")
            for i in range(filteredregressderivratios.shape[0]):
                delayoffset[i] = tide_refinedelay.ratiotodelay(filteredregressderivratios[i])
            refinedvoxelstoreport = filteredregressderivratios.shape[0]
        else:
            medfiltregressderivratios = np.zeros_like(regressderivratios)
            filteredregressderivratios = np.zeros_like(regressderivratios)
            delayoffsetMAD = np.zeros(args.refineregressderivs, dtype=float)
            for i in range(args.refineregressderivs):
                (
                    medfiltregressderivratios[i, :],
                    filteredregressderivratios[i, :],
                    delayoffsetMAD[i],
                ) = tide_refinedelay.filterderivratios(
                    regressderivratios[i, :],
                    (xsize, ysize, numslices),
                    validvoxels,
                    (xdim, ydim, slicedim),
                    gausssigma=args.delayoffsetgausssigma,
                    patchthresh=args.delaypatchthresh,
                    fileiscifti=False,
                    textio=False,
                    rt_floattype=rt_floattype,
                    debug=args.debug,
                )

    def getdelays(self):
        # now calculate the delay offsets
        delayoffset = np.zeros_like(filteredregressderivratios[0, :])
        if self.debug:
            print(f"calculating delayoffsets for {filteredregressderivratios.shape[1]} voxels")
        for i in range(filteredregressderivratios.shape[1]):
            delayoffset[i] = tide_refinedelay.coffstodelay(
                filteredregressderivratios[:, i],
                mindelay=self.mindelay,
                maxdelay=self.maxdelay,
            )
        refinedvoxelstoreport = filteredregressderivratios.shape[1]

    def savestats(self):
        namesuffix = "_desc-delayoffset_hist"
        tide_stats.makeandsavehistogram(
            delayoffset,
            therunoptions["histlen"],
            1,
            self.outputname + namesuffix,
            displaytitle="Histogram of delay offsets calculated from coefficient ratios",
            dictvarname="delayoffsethist",
            thedict=None,
        )
