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
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
from statsmodels.robust import mad

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util


class SimilarityFunctionator:
    reftc = None
    prepreftc = None
    testtc = None
    preptesttc = None
    timeaxis = None
    similarityfunclen = 0
    datavalid = False
    timeaxisvalid = False
    similarityfuncorigin = 0

    def __init__(
        self,
        Fs=0.0,
        similarityfuncorigin=0,
        lagmininpts=0,
        lagmaxinpts=0,
        ncprefilter=None,
        negativegradient=False,
        reftc=None,
        reftcstart=0.0,
        detrendorder=1,
        filterinputdata=True,
        debug=False,
    ):
        self.setFs(Fs)
        self.similarityfuncorigin = similarityfuncorigin
        self.lagmininpts = lagmininpts + 0
        self.lagmaxinpts = lagmaxinpts + 0
        self.ncprefilter = ncprefilter
        self.negativegradient = negativegradient
        self.reftc = reftc
        self.detrendorder = detrendorder
        self.filterinputdata = filterinputdata
        self.debug = debug
        if self.reftc is not None:
            self.setreftc(self.reftc)
            self.reftcstart = reftcstart + 0.0

    def setFs(self, Fs):
        self.Fs = Fs

    def preptc(self, thetc, isreftc=False):
        # prepare timecourse by filtering, normalizing, detrending, and applying a window function
        if isreftc:
            thenormtc = tide_math.corrnormalize(
                self.ncprefilter.apply(self.Fs, thetc),
                detrendorder=self.detrendorder,
                windowfunc=self.windowfunc,
            )
        else:
            if self.negativegradient:
                thenormtc = tide_math.corrnormalize(
                    -np.gradient(self.ncprefilter.apply(self.Fs, thetc)),
                    detrendorder=self.detrendorder,
                    windowfunc=self.windowfunc,
                )
            else:
                if self.filterinputdata:
                    thenormtc = tide_math.corrnormalize(
                        self.ncprefilter.apply(self.Fs, thetc),
                        detrendorder=self.detrendorder,
                        windowfunc=self.windowfunc,
                    )
                else:
                    thenormtc = tide_math.corrnormalize(
                        thetc,
                        detrendorder=self.detrendorder,
                        windowfunc=self.windowfunc,
                    )

        return thenormtc

    def trim(self, vector):
        return vector[
            self.similarityfuncorigin
            - self.lagmininpts : self.similarityfuncorigin
            + self.lagmaxinpts
        ]

    def getfunction(self, trim=True):
        if self.datavalid:
            if trim:
                return (
                    self.trim(self.thesimfunc),
                    self.trim(self.timeaxis),
                    self.theglobalmax,
                )
            else:
                return self.thesimfunc, self.timeaxis, self.theglobalmax
        else:
            if self.timeaxisvalid:
                if trim:
                    return None, self.trim(self.timeaxis), None
                else:
                    return None, self.timeaxis, None
            else:
                print("must calculate similarity function before fetching data")
                return None, None, None


class MutualInformationator(SimilarityFunctionator):
    def __init__(
        self,
        windowfunc="hamming",
        norm=True,
        madnorm=False,
        smoothingtime=-1.0,
        bins=20,
        sigma=0.25,
        *args,
        **kwargs,
    ):
        self.windowfunc = windowfunc
        self.norm = norm
        self.madnorm = madnorm
        self.bins = bins
        self.sigma = sigma
        self.smoothingtime = smoothingtime
        self.smoothingfilter = tide_filt.NoncausalFilter(filtertype="arb")
        self.mi_norm = 1.0
        if self.smoothingtime > 0.0:
            self.smoothingfilter.setfreqs(
                0.0, 0.0, 1.0 / self.smoothingtime, 1.0 / self.smoothingtime
            )
        super(MutualInformationator, self).__init__(*args, **kwargs)

    def setlimits(self, lagmininpts, lagmaxinpts):
        self.lagmininpts = lagmininpts
        self.lagmaxinpts = lagmaxinpts
        origpadtime = self.smoothingfilter.getpadtime()
        timespan = self.timeaxis[-1] - self.timeaxis[0]
        newpadtime = np.min([origpadtime, timespan])
        if newpadtime < origpadtime:
            print("lowering smoothing filter pad time to", newpadtime)
            self.smoothingfilter.setpadtime(newpadtime)

    def setbins(self, bins):
        self.bins = bins

    def setreftc(self, reftc, offset=0.0):
        self.reftc = reftc + 0.0
        self.prepreftc = self.preptc(self.reftc, isreftc=True)

        self.timeaxis, self.automi, self.similarityfuncorigin = tide_corr.cross_mutual_info(
            self.prepreftc,
            self.prepreftc,
            Fs=self.Fs,
            fast=True,
            negsteps=self.lagmininpts,
            possteps=self.lagmaxinpts,
            returnaxis=True,
        )

        self.timeaxis -= offset
        self.similarityfunclen = len(self.timeaxis)
        self.timeaxisvalid = True
        self.datavalid = False
        self.mi_norm = np.nan_to_num(1.0 / np.max(self.automi))
        if self.debug:
            print(f"MutualInformationator setreftc: {len(self.timeaxis)=}")
            print(f"MutualInformationator setreftc: {self.timeaxis}")
            print(f"MutualInformationator setreftc: {self.mi_norm=}")

    def getnormfac(self):
        return self.mi_norm

    def run(self, thetc, locs=None, trim=True, gettimeaxis=True):
        if len(thetc) != len(self.reftc):
            print(
                "MutualInformationator: timecourses are of different sizes:",
                len(thetc),
                "!=",
                len(self.reftc),
                "- exiting",
            )
            sys.exit()

        self.testtc = thetc
        self.preptesttc = self.preptc(self.testtc)

        if locs is not None:
            gettimeaxis = True

        # now calculate the similarity function
        if trim:
            retvals = tide_corr.cross_mutual_info(
                self.preptesttc,
                self.prepreftc,
                norm=self.norm,
                negsteps=self.lagmininpts,
                possteps=self.lagmaxinpts,
                locs=locs,
                madnorm=self.madnorm,
                returnaxis=gettimeaxis,
                fast=True,
                Fs=self.Fs,
                sigma=self.sigma,
                bins=self.bins,
            )
        else:
            retvals = tide_corr.cross_mutual_info(
                self.preptesttc,
                self.prepreftc,
                norm=self.norm,
                negsteps=-1,
                possteps=-1,
                locs=locs,
                madnorm=self.madnorm,
                returnaxis=gettimeaxis,
                fast=True,
                Fs=self.Fs,
                sigma=self.sigma,
                bins=self.bins,
            )
        if gettimeaxis:
            self.timeaxis, self.thesimfunc, self.similarityfuncorigin = (
                retvals[0],
                retvals[1],
                retvals[2],
            )
            self.timeaxisvalid = True
        else:
            self.thesimfunc = retvals[0]

        # normalize
        self.thesimfunc *= self.mi_norm

        if locs is not None:
            return self.thesimfunc

        if self.smoothingtime > 0.0:
            self.thesimfunc = self.smoothingfilter.apply(self.Fs, self.thesimfunc)

        self.similarityfunclen = len(self.thesimfunc)
        if trim:
            self.similarityfuncorigin = self.lagmininpts + 1
        else:
            self.similarityfuncorigin = self.similarityfunclen // 2 + 1

        # find the global maximum value
        self.theglobalmax = np.argmax(self.thesimfunc)
        self.datavalid = True

        # make a dummy filtered baseline
        self.filteredbaseline = self.thesimfunc * 0.0

        if trim:
            return (
                self.trim(self.thesimfunc),
                self.trim(self.timeaxis),
                self.theglobalmax,
            )
        else:
            return self.thesimfunc, self.timeaxis, self.theglobalmax


class Correlator(SimilarityFunctionator):
    def __init__(
        self,
        windowfunc="hamming",
        corrweighting="None",
        corrpadding=0,
        baselinefilter=None,
        *args,
        **kwargs,
    ):
        self.windowfunc = windowfunc
        self.corrweighting = corrweighting
        self.corrpadding = corrpadding
        self.baselinefilter = baselinefilter
        super(Correlator, self).__init__(*args, **kwargs)

    def setlimits(self, lagmininpts, lagmaxinpts):
        self.lagmininpts = lagmininpts
        self.lagmaxinpts = lagmaxinpts

    def setreftc(self, reftc, offset=0.0):
        self.reftc = reftc + 0.0
        self.prepreftc = self.preptc(self.reftc, isreftc=True)
        self.similarityfunclen = len(self.reftc) * 2 - 1
        self.similarityfuncorigin = self.similarityfunclen // 2 + 1

        # make the reference time axis
        self.timeaxis = (
            np.arange(0.0, self.similarityfunclen) * (1.0 / self.Fs)
            - ((self.similarityfunclen - 1) * (1.0 / self.Fs)) / 2.0
        ) - offset
        self.timeaxisvalid = True
        self.datavalid = False

    def run(self, thetc, trim=True):
        if len(thetc) != len(self.reftc):
            print(
                "Correlator: timecourses are of different sizes:",
                len(thetc),
                "!=",
                len(self.reftc),
                "- exiting",
            )
            sys.exit()

        self.testtc = thetc
        self.preptesttc = self.preptc(self.testtc)

        # now actually do the correlation
        self.thesimfunc = tide_corr.fastcorrelate(
            self.preptesttc,
            self.prepreftc,
            usefft=True,
            weighting=self.corrweighting,
            zeropadding=self.corrpadding,
            debug=self.debug,
        )
        self.similarityfunclen = len(self.thesimfunc)
        self.similarityfuncorigin = self.similarityfunclen // 2 + 1

        if self.baselinefilter is not None:
            self.filteredbaseline = self.baselinefilter.apply(self.Fs, self.thesimfunc)
        else:
            self.filteredbaseline = self.thesimfunc * 0.0

        # find the global maximum value
        self.theglobalmax = np.argmax(self.thesimfunc)
        self.datavalid = True

        if trim:
            return (
                self.trim(self.thesimfunc),
                self.trim(self.timeaxis),
                self.theglobalmax,
            )
        else:
            return self.thesimfunc, self.timeaxis, self.theglobalmax


class SimilarityFunctionFitter:
    corrtimeaxis = None
    FML_NOERROR = np.uint32(0x0000)

    FML_INITAMPLOW = np.uint32(0x0001)
    FML_INITAMPHIGH = np.uint32(0x0002)
    FML_INITWIDTHLOW = np.uint32(0x0004)
    FML_INITWIDTHHIGH = np.uint32(0x0008)
    FML_INITLAGLOW = np.uint32(0x0010)
    FML_INITLAGHIGH = np.uint32(0x0020)
    FML_INITFAIL = (
        FML_INITAMPLOW
        | FML_INITAMPHIGH
        | FML_INITWIDTHLOW
        | FML_INITWIDTHHIGH
        | FML_INITLAGLOW
        | FML_INITLAGHIGH
    )

    FML_FITAMPLOW = np.uint32(0x0100)
    FML_FITAMPHIGH = np.uint32(0x0200)
    FML_FITWIDTHLOW = np.uint32(0x0400)
    FML_FITWIDTHHIGH = np.uint32(0x0800)
    FML_FITLAGLOW = np.uint32(0x1000)
    FML_FITLAGHIGH = np.uint32(0x2000)
    FML_FITALGOFAIL = np.uint32(0x0400)
    FML_FITFAIL = (
        FML_FITAMPLOW
        | FML_FITAMPHIGH
        | FML_FITWIDTHLOW
        | FML_FITWIDTHHIGH
        | FML_FITLAGLOW
        | FML_FITLAGHIGH
        | FML_FITALGOFAIL
    )

    def __init__(
        self,
        corrtimeaxis=None,
        lagmin=-30.0,
        lagmax=30.0,
        absmaxsigma=1000.0,
        absminsigma=0.25,
        hardlimit=True,
        bipolar=False,
        lthreshval=0.0,
        uthreshval=1.0,
        debug=False,
        zerooutbadfit=True,
        maxguess=0.0,
        useguess=False,
        searchfrac=0.5,
        lagmod=1000.0,
        enforcethresh=True,
        allowhighfitamps=False,
        displayplots=False,
        functype="correlation",
        peakfittype="gauss",
    ):
        r"""

        Parameters
        ----------
        corrtimeaxis:  1D float array
            The time axis of the correlation function
        lagmin: float
            The minimum allowed lag time in seconds
        lagmax: float
            The maximum allowed lag time in seconds
        absmaxsigma: float
            The maximum allowed peak halfwidth in seconds
        hardlimit
        bipolar: boolean
            If true find the correlation peak with the maximum absolute value, regardless of sign
        lthreshval
        uthreshval
        debug
        zerooutbadfit
        maxguess
        useguess
        searchfrac
        lagmod
        enforcethresh
        displayplots

        Returns
        -------


        Methods
        -------
        fit(corrfunc):
            Fit the correlation function given in corrfunc and return the location of the peak in seconds, the maximum
            correlation value, the peak width
        setrange(lagmin, lagmax):
            Specify the search range for lag peaks, in seconds
        """
        self.setcorrtimeaxis(corrtimeaxis)
        self.lagmin = lagmin + 0.0
        self.lagmax = lagmax + 0.0
        self.absmaxsigma = absmaxsigma + 0.0
        self.absminsigma = absminsigma + 0.0
        self.hardlimit = hardlimit
        self.bipolar = bipolar
        self.lthreshval = lthreshval + 0.0
        self.uthreshval = uthreshval + 0.0
        self.debug = debug
        if functype == "correlation" or functype == "mutualinfo":
            self.functype = functype
        else:
            print("illegal functype")
            sys.exit()
        self.peakfittype = peakfittype
        self.zerooutbadfit = zerooutbadfit
        self.maxguess = maxguess + 0.0
        self.useguess = useguess
        self.searchfrac = searchfrac + 0.0
        self.lagmod = lagmod + 0.0
        self.enforcethresh = enforcethresh
        self.allowhighfitamps = allowhighfitamps
        self.displayplots = displayplots

    def _maxindex_noedge(self, corrfunc):
        """

        Parameters
        ----------
        corrfunc

        Returns
        -------

        """
        lowerlim = 0
        upperlim = len(self.corrtimeaxis) - 1
        done = False
        while not done:
            flipfac = 1.0
            done = True
            maxindex = (np.argmax(corrfunc[lowerlim:upperlim]) + lowerlim).astype("int32")
            if self.bipolar:
                minindex = (np.argmax(-corrfunc[lowerlim:upperlim]) + lowerlim).astype("int32")
                if np.fabs(corrfunc[minindex]) > np.fabs(corrfunc[maxindex]):
                    maxindex = minindex
                    flipfac = -1.0
            if upperlim == lowerlim:
                done = True
            if maxindex == 0:
                lowerlim += 1
                done = False
            if maxindex == upperlim:
                upperlim -= 1
                done = False
        return maxindex, flipfac

    def setfunctype(self, functype):
        self.functype = functype

    def setpeakfittype(self, peakfittype):
        self.peakfittype = peakfittype

    def setrange(self, lagmin, lagmax):
        self.lagmin = lagmin
        self.lagmax = lagmax

    def setcorrtimeaxis(self, corrtimeaxis):
        if corrtimeaxis is not None:
            self.corrtimeaxis = corrtimeaxis + 0.0
        else:
            self.corrtimeaxis = corrtimeaxis

    def setguess(self, useguess, maxguess=0.0):
        self.useguess = useguess
        self.maxguess = maxguess

    def setlthresh(self, lthreshval):
        self.lthreshval = lthreshval

    def setuthresh(self, uthreshval):
        self.uthreshval = uthreshval

    def diagnosefail(self, failreason):
        # define error values
        reasons = []
        if failreason.astype(np.uint32) & self.FML_INITAMPLOW:
            reasons.append("Initial amplitude too low")
        if failreason.astype(np.uint32) & self.FML_INITAMPHIGH:
            reasons.append("Initial amplitude too high")
        if failreason.astype(np.uint32) & self.FML_INITWIDTHLOW:
            reasons.append("Initial width too low")
        if failreason.astype(np.uint32) & self.FML_INITWIDTHHIGH:
            reasons.append("Initial width too high")
        if failreason.astype(np.uint32) & self.FML_INITLAGLOW:
            reasons.append("Initial Lag too low")
        if failreason.astype(np.uint32) & self.FML_INITLAGHIGH:
            reasons.append("Initial Lag too high")

        if failreason.astype(np.uint32) & self.FML_FITAMPLOW:
            reasons.append("Fit amplitude too low")
        if failreason.astype(np.uint32) & self.FML_FITAMPHIGH:
            reasons.append("Fit amplitude too high")
        if failreason.astype(np.uint32) & self.FML_FITWIDTHLOW:
            reasons.append("Fit width too low")
        if failreason.astype(np.uint32) & self.FML_FITWIDTHHIGH:
            reasons.append("Fit width too high")
        if failreason.astype(np.uint32) & self.FML_FITLAGLOW:
            reasons.append("Fit Lag too low")
        if failreason.astype(np.uint32) & self.FML_FITLAGHIGH:
            reasons.append("Fit Lag too high")
        if failreason.astype(np.uint32) & self.FML_FITALGOFAIL:
            reasons.append("Nonlinear fit failed")

        if len(reasons) > 0:
            return ", ".join(reasons)
        else:
            return "No error"

    def fit(self, incorrfunc):
        # check to make sure xcorr_x and xcorr_y match
        if self.corrtimeaxis is None:
            print("Correlation time axis is not defined - exiting")
            sys.exit()
        if len(self.corrtimeaxis) != len(incorrfunc):
            print(
                "Correlation time axis and values do not match in length (",
                len(self.corrtimeaxis),
                "!=",
                len(incorrfunc),
                "- exiting",
            )
            sys.exit()
        # set initial parameters
        # absmaxsigma is in seconds
        # maxsigma is in Hz
        # maxlag is in seconds
        warnings.filterwarnings("ignore", "Number*")
        failreason = self.FML_NOERROR
        maskval = np.uint16(1)  # start out assuming the fit will succeed
        binwidth = self.corrtimeaxis[1] - self.corrtimeaxis[0]

        # set the search range
        lowerlim = 0
        upperlim = len(self.corrtimeaxis) - 1
        if self.debug:
            print(
                "initial search indices are",
                lowerlim,
                "to",
                upperlim,
                "(",
                self.corrtimeaxis[lowerlim],
                self.corrtimeaxis[upperlim],
                ")",
            )

        # make an initial guess at the fit parameters for the gaussian
        # start with finding the maximum value and its location
        flipfac = 1.0
        corrfunc = incorrfunc + 0.0
        if self.useguess:
            maxindex = tide_util.valtoindex(self.corrtimeaxis, self.maxguess)
            if (corrfunc[maxindex] < 0.0) and self.bipolar:
                flipfac = -1.0
        else:
            maxindex, flipfac = self._maxindex_noedge(corrfunc)
        corrfunc *= flipfac
        maxlag_init = (1.0 * self.corrtimeaxis[maxindex]).astype("float64")
        maxval_init = corrfunc[maxindex].astype("float64")
        if self.debug:
            print(
                "maxindex, maxlag_init, maxval_init:",
                maxindex,
                maxlag_init,
                maxval_init,
            )

        # set the baseline and baselinedev levels
        if (self.functype == "correlation") or (self.functype == "hybrid"):
            baseline = 0.0
            baselinedev = 0.0
        else:
            # for mutual information, there is a nonzero baseline, so we want the difference from that.
            baseline = np.median(corrfunc)
            baselinedev = mad(corrfunc)
        if self.debug:
            print("baseline, baselinedev:", baseline, baselinedev)

        # then calculate the width of the peak
        if self.peakfittype == "fastquad" or self.peakfittype == "COM":
            peakstart = np.max([1, maxindex - 2])
            peakend = np.min([len(self.corrtimeaxis) - 2, maxindex + 2])
        else:
            # come here for peakfittype of None, quad, gauss, fastgauss
            thegrad = np.gradient(corrfunc).astype(
                "float64"
            )  # the gradient of the correlation function
            if (self.functype == "correlation") or (self.functype == "hybrid"):
                if self.peakfittype == "quad":
                    peakpoints = np.where(
                        corrfunc > maxval_init - 0.05, 1, 0
                    )  # mask for places where correlation exceeds searchfrac*maxval_init
                else:
                    peakpoints = np.where(
                        corrfunc > (baseline + self.searchfrac * (maxval_init - baseline)), 1, 0
                    )  # mask for places where correlation exceeds searchfrac*maxval_init
            else:
                # for mutual information, there is a flattish, nonzero baseline, so we want the difference from that.
                peakpoints = np.where(
                    corrfunc > (baseline + self.searchfrac * (maxval_init - baseline)),
                    1,
                    0,
                )

            peakpoints[0] = 0
            peakpoints[-1] = 0
            peakstart = np.max([1, maxindex - 1])
            peakend = np.min([len(self.corrtimeaxis) - 2, maxindex + 1])
            if self.debug:
                print("initial peakstart, peakend:", peakstart, peakend)
            if self.functype == "mutualinfo":
                while peakpoints[peakend + 1] == 1:
                    peakend += 1
                while peakpoints[peakstart - 1] == 1:
                    peakstart -= 1
            else:
                while thegrad[peakend + 1] <= 0.0 and peakpoints[peakend + 1] == 1:
                    peakend += 1
                while thegrad[peakstart - 1] >= 0.0 and peakpoints[peakstart - 1] == 1:
                    peakstart -= 1
            if self.debug:
                print("final peakstart, peakend:", peakstart, peakend)

            # deal with flat peak top
            while (
                peakend < (len(self.corrtimeaxis) - 3)
                and corrfunc[peakend] == corrfunc[peakend - 1]
            ):
                peakend += 1
            while peakstart > 2 and corrfunc[peakstart] == corrfunc[peakstart + 1]:
                peakstart -= 1
            if self.debug:
                print("peakstart, peakend after flattop correction:", peakstart, peakend)
                print("\n")
                for i in range(peakstart, peakend + 1):
                    print(self.corrtimeaxis[i], corrfunc[i])
                print("\n")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title("Peak sent to fitting routine")
                plt.plot(
                    self.corrtimeaxis[peakstart : peakend + 1],
                    corrfunc[peakstart : peakend + 1],
                    "r",
                )
                plt.show()

            # This is calculated from first principles, but it's always big by a factor or ~1.4.
            #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division
            maxsigma_init = np.float64(
                ((peakend - peakstart + 1) * binwidth / (2.0 * np.sqrt(-np.log(self.searchfrac))))
                / np.sqrt(2.0)
            )
            if self.debug:
                print("maxsigma_init:", maxsigma_init)

            # now check the values for errors
            if self.hardlimit:
                rangeextension = 0.0
            else:
                rangeextension = (self.lagmax - self.lagmin) * 0.75
            if not (
                (self.lagmin - rangeextension - binwidth)
                <= maxlag_init
                <= (self.lagmax + rangeextension + binwidth)
            ):
                if maxlag_init <= (self.lagmin - rangeextension - binwidth):
                    failreason |= self.FML_INITLAGLOW
                    maxlag_init = self.lagmin - rangeextension - binwidth
                else:
                    failreason |= self.FML_INITLAGHIGH
                    maxlag_init = self.lagmax + rangeextension + binwidth
                if self.debug:
                    print("bad initial")
            if maxsigma_init > self.absmaxsigma:
                failreason |= self.FML_INITWIDTHHIGH
                maxsigma_init = self.absmaxsigma
                if self.debug:
                    print("bad initial width - too high")
            if peakend - peakstart < 2:
                failreason |= self.FML_INITWIDTHLOW
                maxsigma_init = np.float64(
                    ((2 + 1) * binwidth / (2.0 * np.sqrt(-np.log(self.searchfrac)))) / np.sqrt(2.0)
                )
                if self.debug:
                    print("bad initial width - too low")
            if (self.functype == "correlation") or (self.functype == "hybrid"):
                if not (self.lthreshval <= maxval_init <= self.uthreshval) and self.enforcethresh:
                    failreason |= self.FML_INITAMPLOW
                    if self.debug:
                        print(
                            "bad initial amp:",
                            maxval_init,
                            "is less than",
                            self.lthreshval,
                        )
                if maxval_init < 0.0:
                    failreason |= self.FML_INITAMPLOW
                    maxval_init = 0.0
                    if self.debug:
                        print("bad initial amp:", maxval_init, "is less than 0.0")
                if (maxval_init > 1.0) and self.enforcethresh:
                    failreason |= self.FML_INITAMPHIGH
                    maxval_init = 1.0
                    if self.debug:
                        print("bad initial amp:", maxval_init, "is greater than 1.0")
            else:
                # somewhat different rules for mutual information peaks
                if ((maxval_init - baseline) < self.lthreshval * baselinedev) or (
                    maxval_init < baseline
                ):
                    failreason |= self.FML_INITAMPLOW
                    maxval_init = 0.0
                    if self.debug:
                        print("bad initial amp:", maxval_init, "is less than 0.0")
            if (failreason != self.FML_NOERROR) and self.zerooutbadfit:
                maxval = np.float64(0.0)
                maxlag = np.float64(0.0)
                maxsigma = np.float64(0.0)
            else:
                maxval = np.float64(maxval_init)
                maxlag = np.float64(maxlag_init)
                maxsigma = np.float64(maxsigma_init)

        # refine if necessary
        if self.peakfittype != "None":
            if self.peakfittype == "COM":
                X = self.corrtimeaxis[peakstart : peakend + 1] - baseline
                data = corrfunc[peakstart : peakend + 1]
                maxval = maxval_init
                maxlag = np.sum(X * data) / np.sum(data)
                maxsigma = 10.0
            elif self.peakfittype == "gauss":
                X = self.corrtimeaxis[peakstart : peakend + 1] - baseline
                data = corrfunc[peakstart : peakend + 1]
                # do a least squares fit over the top of the peak
                # p0 = np.array([maxval_init, np.fmod(maxlag_init, lagmod), maxsigma_init], dtype='float64')
                p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype="float64")
                if self.debug:
                    print("fit input array:", p0)
                try:
                    plsq, dummy = sp.optimize.leastsq(
                        tide_fit.gaussresiduals, p0, args=(data, X), maxfev=5000
                    )
                    maxval = plsq[0] + baseline
                    maxlag = np.fmod((1.0 * plsq[1]), self.lagmod)
                    maxsigma = plsq[2]
                except:
                    failreason |= self.FML_FITALGOFAIL
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                if self.debug:
                    print("fit output array:", [maxval, maxlag, maxsigma])
            elif self.peakfittype == "gausscf":
                X = self.corrtimeaxis[peakstart : peakend + 1] - baseline
                data = corrfunc[peakstart : peakend + 1]
                # do a least squares fit over the top of the peak
                try:
                    plsq, pcov = curve_fit(
                        tide_fit.gaussfunc,
                        X,
                        data,
                        p0=[maxval_init, maxlag_init, maxsigma_init],
                    )
                    maxval = plsq[0] + baseline
                    maxlag = np.fmod((1.0 * plsq[1]), self.lagmod)
                    maxsigma = plsq[2]
                except:
                    failreason |= self.FML_FITALGOFAIL
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                if self.debug:
                    print("fit output array:", [maxval, maxlag, maxsigma])
            elif self.peakfittype == "fastgauss":
                X = self.corrtimeaxis[peakstart : peakend + 1] - baseline
                data = corrfunc[peakstart : peakend + 1]
                # do a non-iterative fit over the top of the peak
                # 6/12/2015  This is just broken.  Gives quantized maxima
                maxlag = np.float64(1.0 * np.sum(X * data) / np.sum(data))
                maxsigma = np.float64(
                    np.sqrt(np.abs(np.sum((X - maxlag) ** 2 * data) / np.sum(data)))
                )
                maxval = np.float64(data.max()) + baseline
            elif self.peakfittype == "fastquad":
                maxlag, maxval, maxsigma, ismax, badfit = tide_fit.refinepeak_quad(
                    self.corrtimeaxis, corrfunc, maxindex
                )
            elif self.peakfittype == "quad":
                X = self.corrtimeaxis[peakstart : peakend + 1]
                data = corrfunc[peakstart : peakend + 1]
                try:
                    thecoffs = Polynomial.fit(X, data, 2).convert().coef[::-1]
                    a = thecoffs[0]
                    b = thecoffs[1]
                    c = thecoffs[2]
                    maxlag = -b / (2.0 * a)
                    maxval = a * maxlag * maxlag + b * maxlag + c
                    maxsigma = 1.0 / np.fabs(a)
                    if self.debug:
                        print("poly coffs:", a, b, c)
                        print("maxlag, maxval, maxsigma:", maxlag, maxval, maxsigma)
                except np.lib.polynomial.RankWarning:
                    failreason |= self.FML_FITALGOFAIL
                    maxlag = 0.0
                    maxval = 0.0
                    maxsigma = 0.0
                if self.debug:
                    print("\n")
                    for i in range(len(X)):
                        print(X[i], data[i])
                    print("\n")
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.set_title("Peak and fit")
                    plt.plot(X, data, "r")
                    plt.plot(X, c + b * X + a * X * X, "b")
                    plt.show()

            else:
                print("illegal peak refinement type")

            # check for errors in fit
            fitfail = False
            if self.bipolar:
                lowestcorrcoeff = -1.0
            else:
                lowestcorrcoeff = 0.0
            if (self.functype == "correlation") or (self.functype == "hybrid"):
                if maxval < lowestcorrcoeff:
                    failreason |= self.FML_FITAMPLOW
                    maxval = lowestcorrcoeff
                    if self.debug:
                        print("bad fit amp: maxval is lower than lower limit")
                    fitfail = True
                if np.abs(maxval) > 1.0:
                    if not self.allowhighfitamps:
                        failreason |= self.FML_FITAMPHIGH
                        if self.debug:
                            print(
                                "bad fit amp: magnitude of",
                                maxval,
                                "is greater than 1.0",
                            )
                        fitfail = True
                    maxval = 1.0 * np.sign(maxval)
            else:
                # different rules for mutual information peaks
                if ((maxval - baseline) < self.lthreshval * baselinedev) or (maxval < baseline):
                    failreason |= self.FML_FITAMPLOW
                    maxval_init = 0.0
                    if self.debug:
                        if (maxval - baseline) < self.lthreshval * baselinedev:
                            print(
                                "FITAMPLOW: maxval - baseline:",
                                maxval - baseline,
                                " < lthreshval * baselinedev:",
                                self.lthreshval * baselinedev,
                            )
                        if maxval < baseline:
                            print("FITAMPLOW: maxval < baseline:", maxval, baseline)
                    if self.debug:
                        print("bad fit amp: maxval is lower than lower limit")
            if (self.lagmin > maxlag) or (maxlag > self.lagmax):
                if self.debug:
                    print("bad lag after refinement")
                if self.lagmin > maxlag:
                    failreason |= self.FML_FITLAGLOW
                    maxlag = self.lagmin
                else:
                    failreason |= self.FML_FITLAGHIGH
                    maxlag = self.lagmax
                fitfail = True
            if maxsigma > self.absmaxsigma:
                failreason |= self.FML_FITWIDTHHIGH
                if self.debug:
                    print("bad width after refinement:", maxsigma, ">", self.absmaxsigma)
                maxsigma = self.absmaxsigma
                fitfail = True
            if maxsigma < self.absminsigma:
                failreason |= self.FML_FITWIDTHLOW
                if self.debug:
                    print("bad width after refinement:", maxsigma, "<", self.absminsigma)
                maxsigma = self.absminsigma
                fitfail = True
            if fitfail:
                if self.debug:
                    print("fit fail")
                if self.zerooutbadfit:
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                maskval = np.uint16(0)
            # print(maxlag_init, maxlag, maxval_init, maxval, maxsigma_init, maxsigma, maskval, failreason, fitfail)
        else:
            maxval = np.float64(maxval_init)
            maxlag = np.float64(np.fmod(maxlag_init, self.lagmod))
            maxsigma = np.float64(maxsigma_init)
            if failreason != self.FML_NOERROR:
                maskval = np.uint16(0)
            else:
                maskval = np.uint16(1)

        if self.debug or self.displayplots:
            print(
                "init to final: maxval",
                maxval_init,
                maxval,
                ", maxlag:",
                maxlag_init,
                maxlag,
                ", width:",
                maxsigma_init,
                maxsigma,
            )
        if self.displayplots and (self.peakfittype != "None") and (maskval != 0.0):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("Data and fit")
            hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
            plt.plot(
                X,
                data,
                "ro",
                hiresx,
                tide_fit.gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])),
                "b-",
            )
            plt.show()
        return (
            maxindex,
            maxlag,
            flipfac * maxval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        )


class FrequencyTracker:
    freqs = None
    times = None

    def __init__(self, lowerlim=0.1, upperlim=0.6, nperseg=32, Q=10.0, debug=False):
        self.lowerlim = lowerlim
        self.upperlim = upperlim
        self.nperseg = nperseg
        self.Q = Q
        self.debug = debug
        self.nfft = self.nperseg

    def track(self, x, fs):
        self.freqs, self.times, thespectrogram = sp.signal.spectrogram(
            np.concatenate(
                [np.zeros(int(self.nperseg // 2)), x, np.zeros(int(self.nperseg // 2))],
                axis=0,
            ),
            fs=fs,
            detrend="constant",
            scaling="spectrum",
            nfft=None,
            window=np.hamming(self.nfft),
            noverlap=(self.nperseg - 1),
        )
        lowerliminpts = tide_util.valtoindex(self.freqs, self.lowerlim)
        upperliminpts = tide_util.valtoindex(self.freqs, self.upperlim)

        if self.debug:
            print(self.times.shape, self.freqs.shape, thespectrogram.shape)
            print(self.times)

        # initialize the peak fitter
        thefitter = SimilarityFunctionFitter(
            corrtimeaxis=self.freqs,
            lagmin=self.lowerlim,
            lagmax=self.upperlim,
            absmaxsigma=10.0,
            absminsigma=0.1,
            debug=self.debug,
            peakfittype="fastquad",
            zerooutbadfit=False,
            useguess=False,
        )

        peakfreqs = np.zeros((thespectrogram.shape[1] - 1), dtype=float)
        for i in range(0, thespectrogram.shape[1] - 1):
            (
                maxindex,
                peakfreqs[i],
                maxval,
                maxsigma,
                maskval,
                failreason,
                peakstart,
                peakend,
            ) = thefitter.fit(thespectrogram[:, i])
            if not (lowerliminpts <= maxindex <= upperliminpts):
                peakfreqs[i] = -1.0

        return self.times[:-1], peakfreqs

    def clean(self, x, fs, times, peakfreqs, numharmonics=2):
        nyquistfreq = 0.5 * fs
        y = x * 0.0
        halfwidth = int(self.nperseg // 2)
        padx = np.concatenate([np.zeros(halfwidth), x, np.zeros(halfwidth)], axis=0)
        pady = np.concatenate([np.zeros(halfwidth), y, np.zeros(halfwidth)], axis=0)
        padweight = padx * 0.0
        if self.debug:
            print(fs, len(times), len(peakfreqs))
        for i in range(0, len(times)):
            centerindex = int(times[i] * fs)
            xstart = centerindex - halfwidth
            xend = centerindex + halfwidth
            if peakfreqs[i] > 0.0:
                filtsignal = padx[xstart:xend]
                numharmonics = np.min([numharmonics, int((nyquistfreq // peakfreqs[i]) - 1)])
                if self.debug:
                    print("numharmonics:", numharmonics, nyquistfreq // peakfreqs[i])
                for j in range(numharmonics + 1):
                    workingfreq = (j + 1) * peakfreqs[i]
                    if self.debug:
                        print("workingfreq:", workingfreq)
                    ws = [workingfreq * 0.95, workingfreq * 1.05]
                    wp = [workingfreq * 0.9, workingfreq * 1.1]
                    gpass = 1.0
                    gstop = 40.0
                    b, a = sp.signal.iirdesign(wp, ws, gpass, gstop, ftype="cheby2", fs=fs)
                    if self.debug:
                        print(
                            i,
                            j,
                            times[i],
                            centerindex,
                            halfwidth,
                            xstart,
                            xend,
                            xend - xstart,
                            wp,
                            ws,
                            len(a),
                            len(b),
                        )
                    filtsignal = sp.signal.filtfilt(b, a, sp.signal.filtfilt(b, a, filtsignal))
                pady[xstart:xend] += filtsignal
            else:
                pady[xstart:xend] += padx[xstart:xend]
            padweight[xstart:xend] += 1.0
        return (pady / padweight)[halfwidth:-halfwidth]
