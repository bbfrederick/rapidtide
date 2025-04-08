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
"""
A class to impmement regressor refinement
"""
import copy

import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.refineregressor as tide_refineregressor
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
from rapidtide.tests.utils import mse


class RegressorRefiner:

    refinemaskvoxels = None

    def __init__(
        self,
        internalvalidfmrishape,
        internalvalidpaddedfmrishape,
        pid,
        outputname,
        initial_fmri_x,
        paddedinitial_fmri_x,
        os_fmri_x,
        sharedmem=False,
        offsettime=0.0,
        ampthresh=0.3,
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
        self.internalvalidfmrishape = internalvalidfmrishape
        self.internalvalidpaddedfmrishape = internalvalidpaddedfmrishape
        self.sharedmem = sharedmem
        self.outputname = outputname
        self.initial_fmri_x = initial_fmri_x
        self.paddedinitial_fmri_x = paddedinitial_fmri_x
        self.os_fmri_x = os_fmri_x

        self.offsettime = offsettime
        self.ampthresh = ampthresh
        self.lagminthresh = lagminthresh
        self.lagmaxthresh = lagmaxthresh
        self.sigmathresh = sigmathresh
        self.cleanrefined = cleanrefined
        self.bipolar = bipolar
        self.fixdelay = fixdelay
        self.LGR = LGR
        self.nprocs = nprocs
        self.detrendorder = detrendorder
        self.alwaysmultiproc = alwaysmultiproc
        self.showprogressbar = showprogressbar
        self.chunksize = chunksize
        self.padtrs = padtrs
        self.refineprenorm = refineprenorm
        self.refineweighting = refineweighting
        self.refinetype = refinetype
        self.pcacomponents = pcacomponents
        self.dodispersioncalc = dodispersioncalc
        self.dispersioncalc_lower = dispersioncalc_lower
        self.dispersioncalc_upper = dispersioncalc_upper
        self.dispersioncalc_step = dispersioncalc_step
        self.windowfunc = windowfunc
        self.passes = passes
        self.maxpasses = maxpasses
        self.convergencethresh = convergencethresh
        self.interptype = interptype
        self.usetmask = usetmask
        self.tmask_y = tmask_y
        self.tmaskos_y = tmaskos_y
        self.fastresamplerpadtime = fastresamplerpadtime
        self.debug = debug
        self.rt_floattype = rt_floattype
        self.rt_floatset = rt_floatset

        self.setmasks(includemask, excludemask)
        self.totalrefinementbytes = self._allocatemem(pid)

    def setmasks(self, includemask, excludemask):
        self.includemask = includemask
        self.excludemask = excludemask

    def _allocatemem(self, pid):
        if self.sharedmem:
            self.shiftedtcs, self.shiftedtcs_shm = tide_util.allocshared(
                self.internalvalidfmrishape, self.rt_floatset, name=f"shiftedtcs_{pid}"
            )
            self.weights, self.weights_shm = tide_util.allocshared(
                self.internalvalidfmrishape, self.rt_floatset, name=f"weights_{pid}"
            )
            self.paddedshiftedtcs, self.paddedshiftedtcs_shm = tide_util.allocshared(
                self.internalvalidpaddedfmrishape,
                self.rt_floatset,
                name=f"paddedshiftedtcs_{pid}",
            )
            self.paddedweights, self.paddedweights_shm = tide_util.allocshared(
                self.internalvalidpaddedfmrishape,
                self.rt_floatset,
                name=f"paddedweights_{pid}",
            )
            ramlocation = "in shared memory"
        else:
            self.shiftedtcs = np.zeros(self.internalvalidfmrishape, dtype=self.rt_floattype)
            self.weights = np.zeros(self.internalvalidfmrishape, dtype=self.rt_floattype)
            self.paddedshiftedtcs = np.zeros(
                self.internalvalidpaddedfmrishape, dtype=self.rt_floattype
            )
            self.paddedweights = np.zeros(
                self.internalvalidpaddedfmrishape, dtype=self.rt_floattype
            )
            ramlocation = "locally"
        totalrefinementbytes = (
            self.shiftedtcs.nbytes
            + self.weights.nbytes
            + self.paddedshiftedtcs.nbytes
            + self.paddedweights.nbytes
        )
        thesize, theunit = tide_util.format_bytes(totalrefinementbytes)
        print(f"allocated {thesize:.3f} {theunit} {ramlocation} for refinement")
        tide_util.logmem("after refinement array allocation")
        return totalrefinementbytes

    def cleanup(self):
        del self.paddedshiftedtcs
        del self.paddedweights
        del self.shiftedtcs
        del self.weights
        if self.sharedmem:
            tide_util.cleanup_shm(self.paddedshiftedtcs_shm)
            tide_util.cleanup_shm(self.paddedweights_shm)
            tide_util.cleanup_shm(self.shiftedtcs_shm)
            tide_util.cleanup_shm(self.weights_shm)

    def makemask(self, lagstrengths, lagtimes, lagsigma, fitmask):
        # create the refinement mask
        (
            self.refinemaskvoxels,
            self.refinemask,
            self.locationfails,
            self.ampfails,
            self.lagfails,
            self.sigmafails,
            self.numinmask,
        ) = tide_refineregressor.makerefinemask(
            lagstrengths,
            lagtimes,
            lagsigma,
            fitmask,
            offsettime=self.offsettime,
            ampthresh=self.ampthresh,
            lagminthresh=self.lagminthresh,
            lagmaxthresh=self.lagmaxthresh,
            sigmathresh=self.sigmathresh,
            cleanrefined=self.cleanrefined,
            bipolar=self.bipolar,
            includemask=self.includemask,
            excludemask=self.excludemask,
            fixdelay=self.fixdelay,
            debug=self.debug,
        )

        if self.numinmask == 0:
            self.LGR.critical("No voxels in refine mask - adjust thresholds or external masks")
            return False
        else:
            return True

    def getrefinemask(self):
        return self.refinemask

    def getpaddedshiftedtcs(self):
        return self.paddedshiftedtcs

    def alignvoxels(self, fmri_data_valid, fmritr, lagtimes):
        # align timecourses to prepare for refinement
        self.LGR.info("aligning timecourses")
        voxelsprocessed_rra = tide_refineregressor.alignvoxels(
            fmri_data_valid,
            fmritr,
            self.shiftedtcs,
            self.weights,
            self.paddedshiftedtcs,
            self.paddedweights,
            lagtimes,
            self.refinemask,
            nprocs=self.nprocs,
            detrendorder=self.detrendorder,
            offsettime=self.offsettime,
            alwaysmultiproc=self.alwaysmultiproc,
            showprogressbar=self.showprogressbar,
            chunksize=self.chunksize,
            padtrs=self.padtrs,
            debug=self.debug,
            rt_floatset=self.rt_floatset,
            rt_floattype=self.rt_floattype,
        )
        return voxelsprocessed_rra
        # self.LGR.info(f"align complete: {voxelsprocessed_rra=}")

    def prenormalize(self, lagtimes, lagstrengths, R2):
        tide_refineregressor.prenorm(
            self.paddedshiftedtcs,
            self.refinemask,
            lagtimes,
            self.lagmaxthresh,
            lagstrengths,
            R2,
            self.refineprenorm,
            self.refineweighting,
        )

    def refine(
        self,
        theprefilter,
        fmritr,
        thepass,
        lagstrengths,
        lagtimes,
        previousnormoutputdata,
        corrmasksize,
    ):
        (
            voxelsprocessed_rr,
            self.paddedoutputdata,
        ) = tide_refineregressor.dorefine(
            self.paddedshiftedtcs,
            self.refinemask,
            self.weights,
            theprefilter,
            fmritr,
            thepass,
            lagstrengths,
            lagtimes,
            self.refinetype,
            1.0 / fmritr,
            self.outputname,
            detrendorder=self.detrendorder,
            pcacomponents=self.pcacomponents,
            dodispersioncalc=self.dodispersioncalc,
            dispersioncalc_lower=self.dispersioncalc_lower,
            dispersioncalc_upper=self.dispersioncalc_upper,
            dispersioncalc_step=self.dispersioncalc_step,
            windowfunc=self.windowfunc,
            cleanrefined=self.cleanrefined,
            bipolar=self.bipolar,
            debug=self.debug,
            rt_floatset=self.rt_floatset,
            rt_floattype=self.rt_floattype,
        )
        outputdict = {}
        outputdict["refinemasksize_pass" + str(thepass)] = voxelsprocessed_rr
        outputdict["refinemaskpct_pass" + str(thepass)] = 100.0 * voxelsprocessed_rr / corrmasksize
        outputdict["refinelocationfails_pass" + str(thepass)] = self.locationfails
        outputdict["refineampfails_pass" + str(thepass)] = self.ampfails
        outputdict["refinelagfails_pass" + str(thepass)] = self.lagfails
        outputdict["refinesigmafails_pass" + str(thepass)] = self.sigmafails

        fmrifreq = 1.0 / fmritr
        if voxelsprocessed_rr > 0:
            paddednormoutputdata = tide_math.stdnormalize(
                theprefilter.apply(fmrifreq, self.paddedoutputdata)
            )
            outputdata = self.paddedoutputdata[self.padtrs : -self.padtrs]
            normoutputdata = tide_math.stdnormalize(theprefilter.apply(fmrifreq, outputdata))
            normunfilteredoutputdata = tide_math.stdnormalize(outputdata)
            tide_io.writebidstsv(
                f"{self.outputname}_desc-refinedmovingregressor_timeseries",
                normunfilteredoutputdata,
                fmrifreq,
                columns=["unfiltered_pass" + str(thepass)],
                extraheaderinfo={
                    "Description": "The raw and filtered probe regressor produced by the refinement procedure, at the time resolution of the data"
                },
                append=(thepass > 1),
            )
            tide_io.writebidstsv(
                f"{self.outputname}_desc-refinedmovingregressor_timeseries",
                normoutputdata,
                fmrifreq,
                columns=["filtered_pass" + str(thepass)],
                extraheaderinfo={
                    "Description": "The raw and filtered probe regressor produced by the refinement procedure, at the time resolution of the data"
                },
                append=True,
            )

        # check for convergence
        regressormse = mse(normoutputdata, previousnormoutputdata)
        outputdict["regressormse_pass" + str(thepass).zfill(2)] = regressormse
        self.LGR.info(f"regressor difference at end of pass {thepass:d} is {regressormse:.6f}")
        if self.convergencethresh is not None:
            if thepass >= self.maxpasses:
                self.LGR.info("refinement ended (maxpasses reached)")
                stoprefining = True
                refinestopreason = "maxpassesreached"
            elif regressormse < self.convergencethresh:
                self.LGR.info("refinement ended (refinement has converged")
                stoprefining = True
                refinestopreason = "convergence"
            else:
                stoprefining = False
        elif thepass >= self.passes:
            stoprefining = True
            refinestopreason = "passesreached"
        else:
            stoprefining = False
            refinestopreason = None
        outputdict["refinestopreason"] = refinestopreason

        if self.detrendorder > 0:
            resampnonosref_y = tide_fit.detrend(
                tide_resample.doresample(
                    self.paddedinitial_fmri_x,
                    paddednormoutputdata,
                    self.initial_fmri_x,
                    method=self.interptype,
                ),
                order=self.detrendorder,
                demean=True,
            )
            resampref_y = tide_fit.detrend(
                tide_resample.doresample(
                    self.paddedinitial_fmri_x,
                    paddednormoutputdata,
                    self.os_fmri_x,
                    method=self.interptype,
                ),
                order=self.detrendorder,
                demean=True,
            )
        else:
            resampnonosref_y = tide_resample.doresample(
                self.paddedinitial_fmri_x,
                paddednormoutputdata,
                self.initial_fmri_x,
                method=self.interptype,
            )
            resampref_y = tide_resample.doresample(
                self.paddedinitial_fmri_x,
                paddednormoutputdata,
                self.os_fmri_x,
                method=self.interptype,
            )
        if self.usetmask:
            resampnonosref_y *= self.tmask_y
            thefit, R2val = tide_fit.mlregress(self.tmask_y, resampnonosref_y)
            resampnonosref_y -= thefit[0, 1] * self.tmask_y
            resampref_y *= self.tmaskos_y
            thefit, R2val = tide_fit.mlregress(self.tmaskos_y, resampref_y)
            resampref_y -= thefit[0, 1] * self.tmaskos_y

        # reinitialize genlagtc for resampling
        previousnormoutputdata = np.zeros_like(normoutputdata)
        genlagtc = tide_resample.FastResampler(
            self.paddedinitial_fmri_x,
            paddednormoutputdata,
            padtime=self.fastresamplerpadtime,
        )
        genlagtc.save(f"{self.outputname}_desc-lagtcgenerator_timeseries")
        if self.debug:
            genlagtc.info()
        (
            outputdict[f"kurtosis_reference_pass{thepass + 1}"],
            outputdict[f"kurtosisz_reference_pass{thepass + 1}"],
            outputdict[f"kurtosisp_reference_pass{thepass + 1}"],
        ) = tide_stats.kurtosisstats(resampref_y)
        (
            outputdict[f"skewness_reference_pass{thepass + 1}"],
            outputdict[f"skewnessz_reference_pass{thepass + 1}"],
            outputdict[f"skewnessp_reference_pass{thepass + 1}"],
        ) = tide_stats.skewnessstats(resampref_y)
        if not stoprefining:
            tide_io.writebidstsv(
                f"{self.outputname}_desc-movingregressor_timeseries",
                tide_math.stdnormalize(resampnonosref_y),
                1.0 / fmritr,
                columns=["pass" + str(thepass + 1)],
                extraheaderinfo={
                    "Description": "The probe regressor used in each pass, at the time resolution of the data"
                },
                append=True,
            )
            oversampfreq = 1.0 / (self.os_fmri_x[1] - self.os_fmri_x[0])
            tide_io.writebidstsv(
                f"{self.outputname}_desc-oversampledmovingregressor_timeseries",
                tide_math.stdnormalize(resampref_y),
                oversampfreq,
                columns=["pass" + str(thepass + 1)],
                extraheaderinfo={
                    "Description": "The probe regressor used in each pass, at the time resolution used for calculating the similarity function"
                },
                append=True,
            )
        else:
            self.LGR.warning(f"refinement failed - terminating at end of pass {thepass}")
            stoprefining = True
            refinestopreason = "emptymask"

        return (
            voxelsprocessed_rr,
            outputdict,
            previousnormoutputdata,
            resampref_y,
            resampnonosref_y,
            stoprefining,
            refinestopreason,
            genlagtc,
        )
