#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016 Blaise Frederick
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
# $Author: frederic $
# $Date: 2016/07/12 13:50:29 $
# $Id: tide_funcs.py,v 1.4 2016/07/12 13:50:29 frederic Exp $
#
from __future__ import print_function, division

import matplotlib.pyplot as pl

import numpy as np
import scipy as sp
import warnings
import sys

import rapidtide.util as tide_util
import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math
import rapidtide.correlate as tide_corr


class correlator:
    oversampfreq = 0.0
    corrorigin = 0
    lagmininpts = 0
    lagmaxinpts = 0
    ncprefilter = None
    referencetc = None
    usewindowfunc = True
    detrendorder = 1
    windowfunc = 'hamming'
    corrweighting = 'none'
    reftc = None

    def __init__(self,
                 Fs=0.0,
                 corrorigin=0,
                 lagmininpts=0,
                 lagmaxinpts=0,
                 ncprefilter=None,
                 referencetc=None,
                 usewindowfunc=True,
                 detrendorder=1,
                 windowfunc='hamming',
                 corrweighting='none'):
        self.Fs = Fs
        self.corrorigin = corrorigin
        self.lagmininpts = lagmininpts
        self.lagmaxinpts = lagmaxinpts
        self.ncprefilter = ncprefilter
        self.referencetc = referencetc
        self.usewindowfunc = usewindowfunc
        self.detrendorder = detrendorder
        self.windowfunc = windowfunc
        self.corrweighting = corrweighting
        if self.referencetc is not None:
            self.setreftc(self.referencetc)

    def preptc(self, thetc):
        # prepare timecourse by filtering, normalizing, detrending, and applying a window function
        return tide_math.corrnormalize(self.ncprefilter.apply(self.Fs, thetc),
                                       prewindow=self.usewindowfunc,
                                       detrendorder=self.detrendorder,
                                       windowfunc=self.windowfunc)


    def setreftc(self, reftc):
        self.reftc = self.preptc(reftc)


    def setlimits(self, corrorigin, lagmininpts, lagmaxinpts):
        self.corrorigin = corrorigin
        self.lagmininpts = lagmininpts
        self.lagmaxinpts = lagmaxinpts


    def run(self, thetc):
        if len(thetc) != len(self.reftc):
            print('timecourses are of different sizes - exiting')
            sys.exit()

        preppedtc = self.preptc(thetc)

        # now actually do the correlation
        thexcorr = tide_corr.fastcorrelate(preppedtc, self.reftc, usefft=True, weighting=self.corrweighting)

        # find the global maximum value
        theglobalmax = np.argmax(thexcorr)

        return thexcorr[self.corrorigin - self.lagmininpts:self.corrorigin + self.lagmaxinpts], theglobalmax


class correlation_fitter:
    lagmin = -30.0
    lagmax = 30.0
    absmaxsigma = 1000.0
    hardlimit = True
    bipolar = False
    lthreshval = 0.0
    uthreshval = 1.0
    debug = False
    zerooutbadfit = True
    findmaxtype = 'gauss'
    lagmod = 1000.0
    corrtimeaxis = None

    def __init__(self,
                 corrtimeaxis=None,
                 lagmin=-30.0,
                 lagmax=30.0,
                 absmaxsigma=1000.0,
                 hardlimit=True,
                 bipolar=False,
                 lthreshval=0.0,
                 uthreshval=1.0,
                 debug=False,
                 findmaxtype='gauss',
                 zerooutbadfit=True,
                 refine=False,
                 maxguess=0.0,
                 useguess=False,
                 searchfrac=0.5,
                 fastgauss=False,
                 lagmod=1000.0,
                 enforcethresh=True,
                 displayplots=False):

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
        threshval
        uthreshval
        debug
        zerooutbadfit
        refine
        maxguess
        useguess
        searchfrac
        fastgauss
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
        self.corrtimeaxis = corrtimeaxis
        self.lagmin = lagmin
        self.lagmax = lagmax
        self.absmaxsigma = absmaxsigma
        self.hardlimit = hardlimit
        self.bipolar = bipolar
        self.lthreshval = lthreshval
        self.uthreshval = uthreshval
        self.debug=debug
        self.findmaxtype=findmaxtype
        self.zerooutbadfit = zerooutbadfit
        self.refine = refine
        self.maxguess = maxguess
        self.useguess = useguess
        self.searchfrac = searchfrac
        self.fastgauss = fastgauss
        self.lagmod = lagmod
        self.enforcethresh = enforcethresh
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
            maxindex = (np.argmax(corrfunc[lowerlim:upperlim]) + lowerlim).astype('int32')
            if self.bipolar:
                minindex = (np.argmax(np.fabs(corrfunc[lowerlim:upperlim])) + lowerlim).astype('int32')
                if np.fabs(corrfunc[minindex]) > np.fabs(corrfunc[maxindex]):
                    maxindex = minindex
                    flipfac = -1.0
            else:
                maxindex = (np.argmax(corrfunc[lowerlim:upperlim]) + lowerlim).astype('int32')
            if upperlim == lowerlim:
                done = True
            if maxindex == 0:
                lowerlim += 1
                done = False
            if maxindex == upperlim:
                upperlim -= 1
                done = False
        return maxindex, flipfac


    def setrange(self, lagmin, lagmax):
        self.lagmin = lagmin
        self.lagmax = lagmax


    def setcorrtimeaxis(self, corrtimeaxis):
        self.corrtimeaxis = corrtimeaxis


    def setguess(self, useguess, maxguess=0.0):
        self.useguess = useguess
        self.maxguess = maxguess


    def setlthresh(self, lthreshval):
        self.lthreshval = lthreshval


    def setuthresh(self, uthreshval):
        self.uthreshval = uthreshval


    def fit(self, corrfunc):
        # check to make sure xcorr_x and xcorr_y match
        if self.corrtimeaxis is None:
            print("Correlation time axis is not defined - exiting")
            sys.exit()
        if len(self.corrtimeaxis) != len(corrfunc):
            print('Correlation time axis and values do not match in length (',
                   len(self.corrtimeaxis),
                  '!=',
                  len(corrfunc),
                  '- exiting')
            sys.exit()
        # set initial parameters
        # absmaxsigma is in seconds
        # maxsigma is in Hz
        # maxlag is in seconds
        warnings.filterwarnings("ignore", "Number*")
        maxlag = np.float64(0.0)
        maxval = np.float64(0.0)
        maxsigma = np.float64(0.0)
        maskval = np.uint16(1)  # start out assuming the fit will succeed
        numlagbins = len(corrfunc)
        binwidth = self.corrtimeaxis[1] - self.corrtimeaxis[0]

        # define error values
        failreason = np.uint16(0)
        FML_BADAMPLOW = np.uint16(0x01)
        FML_BADAMPHIGH = np.uint16(0x02)
        FML_BADSEARCHWINDOW = np.uint16(0x04)
        FML_BADWIDTH = np.uint16(0x08)
        FML_BADLAG = np.uint16(0x10)
        FML_HITEDGE = np.uint16(0x20)
        FML_FITFAIL = np.uint16(0x40)
        FML_INITFAIL = np.uint16(0x80)

        # set the search range
        lowerlim = 0
        upperlim = len(self.corrtimeaxis) - 1
        if self.debug:
            print('initial search indices are', lowerlim, 'to', upperlim,
                  '(', self.corrtimeaxis[lowerlim], self.corrtimeaxis[upperlim], ')')

        # make an initial guess at the fit parameters for the gaussian
        # start with finding the maximum value and its location
        flipfac = 1.0
        if self.useguess:
            maxindex = tide_util.valtoindex(self.corrtimeaxis, self.maxguess)
        else:
            maxindex, flipfac = self._maxindex_noedge(corrfunc)
            corrfunc *= flipfac
        maxlag_init = (1.0 * self.corrtimeaxis[maxindex]).astype('float64')
        maxval_init = corrfunc[maxindex].astype('float64')
        if self.debug:
            print('maxindex, maxlag_init, maxval_init:', maxindex, maxlag_init, maxval_init)

        # then calculate the width of the peak
        thegrad = np.gradient(corrfunc).astype('float64')  # the gradient of the correlation function
        peakpoints = np.where(corrfunc > self.searchfrac * maxval_init, 1,
                              0)  # mask for places where correlaion exceeds serchfrac*maxval_init
        peakpoints[0] = 0
        peakpoints[-1] = 0
        peakstart = maxindex + 0
        peakend = maxindex + 0
        while peakend < (len(self.corrtimeaxis) - 2) and thegrad[peakend + 1] < 0.0 and peakpoints[peakend + 1] == 1:
            peakend += 1
        while peakstart > 1 and thegrad[peakstart - 1] > 0.0 and peakpoints[peakstart - 1] == 1:
            peakstart -= 1
        # This is calculated from first principles, but it's always big by a factor or ~1.4.
        #     Which makes me think I dropped a factor if sqrt(2).  So fix that with a final division
        maxsigma_init = np.float64(
            ((peakend - peakstart + 1) * binwidth / (2.0 * np.sqrt(-np.log(self.searchfrac)))) / np.sqrt(2.0))
        if self.debug:
            print('maxsigma_init:', maxsigma_init)

        # now check the values for errors
        if self.hardlimit:
            rangeextension = 0.0
        else:
            rangeextension = (self.lagmax - self.lagmin) * 0.75
        if not ((self.lagmin - rangeextension - binwidth) <= maxlag_init <= (self.lagmax + rangeextension + binwidth)):
            failreason |= (FML_INITFAIL | FML_BADLAG)
            if maxlag_init <= (self.lagmin - rangeextension - binwidth):
                maxlag_init = self.lagmin - rangeextension - binwidth
            else:
                maxlag_init = self.lagmax + rangeextension + binwidth
            if self.debug:
                print('bad initial')
        if maxsigma_init > self.absmaxsigma:
            failreason |= (FML_INITFAIL | FML_BADWIDTH)
            maxsigma_init = self.absmaxsigma
            if self.debug:
                print('bad initial width - too high')
        if peakend - peakstart < 2:
            failreason |= (FML_INITFAIL | FML_BADSEARCHWINDOW)
            maxsigma_init = np.float64(
                ((2 + 1) * binwidth / (2.0 * np.sqrt(-np.log(self.searchfrac)))) / np.sqrt(2.0))
            if self.debug:
                print('bad initial width - too low')
        if not (self.lthreshval <= maxval_init <= self.uthreshval) and self.enforcethresh:
            failreason |= (FML_INITFAIL | FML_BADAMPLOW)
            if self.debug:
                print('bad initial amp:', maxval_init, 'is less than', self.lthreshval)
        if (maxval_init < 0.0):
            failreason |= (FML_INITFAIL | FML_BADAMPLOW)
            maxval_init = 0.0
            if self.debug:
                print('bad initial amp:', maxval_init, 'is less than 0.0')
        if (maxval_init > 1.0):
            failreason |= (FML_INITFAIL | FML_BADAMPHIGH)
            maxval_init = 1.0
            if self.debug:
                print('bad initial amp:', maxval_init, 'is greater than 1.0')
        if failreason > 0 and self.zerooutbadfit:
            maxval = np.float64(0.0)
            maxlag = np.float64(0.0)
            maxsigma = np.float64(0.0)
        else:
            maxval = np.float64(maxval_init)
            maxlag = np.float64(maxlag_init)
            maxsigma = np.float64(maxsigma_init)

        # refine if necessary
        if self.refine:
            data = corrfunc[peakstart:peakend]
            X = self.corrtimeaxis[peakstart:peakend]
            if self.fastgauss:
                # do a non-iterative fit over the top of the peak
                # 6/12/2015  This is just broken.  Gives quantized maxima
                maxlag = np.float64(1.0 * sum(X * data) / sum(data))
                maxsigma = np.float64(np.sqrt(np.abs(np.sum((X - maxlag) ** 2 * data) / np.sum(data))))
                maxval = np.float64(data.max())
            else:
                # do a least squares fit over the top of the peak
                # p0 = np.array([maxval_init, np.fmod(maxlag_init, lagmod), maxsigma_init], dtype='float64')
                p0 = np.array([maxval_init, maxlag_init, maxsigma_init], dtype='float64')
                if self.debug:
                    print('fit input array:', p0)
                try:
                    plsq, dummy = sp.optimize.leastsq(tide_fit.gaussresiduals, p0, args=(data, X), maxfev=5000)
                    maxval = plsq[0]
                    maxlag = np.fmod((1.0 * plsq[1]), self.lagmod)
                    maxsigma = plsq[2]
                except:
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                if self.debug:
                    print('fit output array:', [maxval, maxlag, maxsigma])

            # check for errors in fit
            fitfail = False
            failreason = np.uint16(0)
            if not (0.0 <= np.fabs(maxval) <= 1.0):
                failreason |= (FML_FITFAIL + FML_BADAMPLOW)
                if self.debug:
                    print('bad amp after refinement')
                fitfail = True
            if (self.lagmin > maxlag) or (maxlag > self.lagmax):
                failreason |= (FML_FITFAIL + FML_BADLAG)
                if self.debug:
                    print('bad lag after refinement')
                if self.lagmin > maxlag:
                    maxlag = self.lagmin
                else:
                    maxlag = self.lagmax
                fitfail = True
            if maxsigma > self.absmaxsigma:
                failreason |= (FML_FITFAIL + FML_BADWIDTH)
                if self.debug:
                    print('bad width after refinement')
                maxsigma = self.absmaxsigma
                fitfail = True
            if not (0.0 < maxsigma):
                failreason |= (FML_FITFAIL + FML_BADSEARCHWINDOW)
                if self.debug:
                    print('bad width after refinement')
                maxsigma = 0.0
                fitfail = True
            if fitfail:
                if self.debug:
                    print('fit fail')
                if self.zerooutbadfit:
                    maxval = np.float64(0.0)
                    maxlag = np.float64(0.0)
                    maxsigma = np.float64(0.0)
                maskval = np.int16(0)
            # print(maxlag_init, maxlag, maxval_init, maxval, maxsigma_init, maxsigma, maskval, failreason, fitfail)
        else:
            maxval = np.float64(maxval_init)
            maxlag = np.float64(np.fmod(maxlag_init, self.lagmod))
            maxsigma = np.float64(maxsigma_init)
            if failreason > 0:
                maskval = np.uint16(0)

        if self.debug or self.displayplots:
            print("init to final: maxval", maxval_init, maxval, ", maxlag:", maxlag_init, maxlag, ", width:", maxsigma_init,
                  maxsigma)
        if self.displayplots and self.refine and (maskval != 0.0):
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title('Data and fit')
            hiresx = np.arange(X[0], X[-1], (X[1] - X[0]) / 10.0)
            pl.plot(X, data, 'ro', hiresx, gauss_eval(hiresx, np.array([maxval, maxlag, maxsigma])), 'b-')
            pl.show()
        return maxindex, maxlag, flipfac * maxval, maxsigma, maskval, failreason, peakstart, peakend