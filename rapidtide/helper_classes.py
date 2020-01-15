#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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


class fmridata:
    thedata = None
    theshape = None
    xsize = None
    ysize = None
    numslices = None
    realtimepoints = None
    timepoints = None
    slicesize = None
    numvox = None
    numskip = 0

    def __init__(self,
                 thedata,
                 zerodata=False,
                 copydata=False,
                 numskip=0):
        if zerodata:
            self.thedata = thedata * 0.0
        else:
            if copydata:
                self.thedata = thedata + 0.0
            else:
                self.thedata = thedata
        self.getsizes()
        self.setnumskip(numskip)


    def getsizes(self):
            self.theshape = self.thedata.shape
            self.xsize = self.theshape[0]
            self.ysize = self.theshape[1]
            self.numslices = self.theshape[2]
            try:
                self.realtimepoints = self.theshape[3]
            except KeyError:
                self.realtimepoints = 1
            self.slicesize = self.xsize * self.ysize
            self.numvox = self.slicesize * self.numslices

    def setnumskip(self, numskip):
        self.numskip = numskip
        self.timepoints = self.realtimepoints - self.numskip


    def byslice(self):
        return self.thedata[:, :, :, self.numskip:].reshape((self.slicesize, self.numslices, self.timepoints))


    def byvol(self):
        return self.thedata[:, :, :, self.numskip:].reshape((self.numvox, self.timepoints))


    def byvox(self):
        return self.thedata[:, :, :, self.numskip:]



class proberegressor:
    inputtimeaxis = None
    inputvec = None
    inputfreq = None
    inputstart = 0.0
    inputoffset = 0.0
    targettimeaxis = None
    targetvec = None
    targetfreq = None
    targetstart = 0.0
    targetoffset = 0.0

    def __init__(self,
                 inputvec,
                 inputfreq,
                 targetperiod,
                 targetpoints,
                 targetstartpoint,
                 targetoversample=1,
                 inputstart=0.0,
                 inputoffset=0.0,
                 targetstart=0.0,
                 targetoffset=0.0,

                 ):
        self.inputoffset = inputoffset
        self.setinputvec(inputvec, inputfreq, inputstart=inputstart)
        self.targetperiod = targetperiod
        self.makeinputtimeaxis(self)
        self.targetoversample = targetoversample
        self.targetpoints = targetpoints
        self.targetstartpoint = targetstartpoint

    def setinputvec(self, inputvec, inputfreq, inputstart=0.0):
        self.inputvec = inputvec
        self.inputfreq = inputfreq
        self.inputstart = inputstart

    def makeinputtimeaxis(self):
        self.inputtimeaxis = np.linspace(0.0, len(self.inputvec)) / self.inputfreq - (self.inputstarttime + self.inputoffset)

    def maketargettimeaxis(self):
        self.targettimeaxis = np.linspace(self.targetperiod * self.targetstartpoint,
                                     self.targetperiod * self.targetstartpoint + self.targetperiod * self.targetpoints,
                                     num=self.targetpoints,
                                     endpoint=True)
        os_fmri_x = np.arange(0.0, (validtimepoints - optiondict['addedskip']) * self.targetoversample - (
                self.targetoversample - 1)) * self.targetoversample * self.targetperiod + skiptime


class correlator:
    reftc = None
    prepreftc = None
    testtc = None
    preptesttc = None
    timeaxis = None
    corrlen = 0
    datavalid = False
    timeaxisvalid = False
    corrorigin = 0

    def __init__(self,
                 Fs=0.0,
                 corrorigin=0,
                 lagmininpts=0,
                 lagmaxinpts=0,
                 ncprefilter=None,
                 reftc=None,
                 detrendorder=1,
                 windowfunc='hamming',
                 corrweighting='none'):
        self.Fs = Fs
        self.corrorigin = corrorigin
        self.lagmininpts = lagmininpts
        self.lagmaxinpts = lagmaxinpts
        self.ncprefilter = ncprefilter
        self.reftc = reftc
        self.detrendorder = detrendorder
        self.windowfunc = windowfunc
        if self.windowfunc is not None:
            self.usewindowfunc = True
        else:
            self.usewindowfunc = False
        self.corrweighting = corrweighting
        if self.reftc is not None:
            self.setreftc(self.reftc)


    def preptc(self, thetc):
        # prepare timecourse by filtering, normalizing, detrending, and applying a window function
        return tide_math.corrnormalize(self.ncprefilter.apply(self.Fs, thetc),
                                       prewindow=self.usewindowfunc,
                                       detrendorder=self.detrendorder,
                                       windowfunc=self.windowfunc)


    def setreftc(self, reftc):
        self.reftc = reftc + 0.0
        self.prepreftc = self.preptc(self.reftc)
        self.corrlen = len(self.reftc) * 2 - 1
        self.corrorigin = self.corrlen // 2 + 1

        # make the time axis
        self.timeaxis = np.arange(0.0, self.corrlen) * (1.0 / self.Fs) \
                        - ((self.corrlen - 1) * (1.0 / self.Fs)) / 2.0
        self.timeaxisvalid = True
        self.datavalid = False


    def setlimits(self, lagmininpts, lagmaxinpts):
        self.lagmininpts = lagmininpts
        self.lagmaxinpts = lagmaxinpts


    def trim(self, vector):
        return vector[self.corrorigin - self.lagmininpts:self.corrorigin + self.lagmaxinpts]


    def getcorrelation(self, trim=True):
        if self.datavalid:
            if trim:
                return self.trim(self.thexcorr), self.trim(self.timeaxis), self.theglobalmax
            else:
                return self.thexcorr, self.timeaxis, self.theglobalmax
        else:
            if self.timeaxisvalid:
                if trim:
                    return None, self.trim(self.timeaxis), None
                else:
                    return None, self.timeaxis, None
            else:
                print('must run correlation before fetching data')
                return None, None, None


    def run(self, thetc, trim=True):
        if len(thetc) != len(self.reftc):
            print('timecourses are of different sizes - exiting')
            sys.exit()

        self.testtc = thetc
        self.preptesttc = self.preptc(self.testtc)

        # now actually do the correlation
        self.thexcorr = tide_corr.fastcorrelate(self.preptesttc, self.prepreftc, usefft=True, weighting=self.corrweighting)
        self.corrlen = len(self.thexcorr)
        self.corrorigin = self.corrlen // 2 + 1

        # find the global maximum value
        self.theglobalmax = np.argmax(self.thexcorr)
        self.datavalid = True

        if trim:
            return self.trim(self.thexcorr), self.trim(self.timeaxis), self.theglobalmax
        else:
            return self.thexcorr, self.timeaxis, self.theglobalmax


class correlation_fitter:
    corrtimeaxis = None
    FML_BADAMPLOW = np.uint16(0x01)
    FML_BADAMPHIGH = np.uint16(0x02)
    FML_BADSEARCHWINDOW = np.uint16(0x04)
    FML_BADWIDTHLOW = np.uint16(0x08)
    FML_BADWIDTHHIGH = np.uint16(0x10)
    FML_BADLAG = np.uint16(0x20)
    FML_FITFAIL = np.uint16(0x40)
    FML_INITFAIL = np.uint16(0x80)

    def __init__(self,
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
        self.setcorrtimeaxis(corrtimeaxis)
        self.lagmin = lagmin
        self.lagmax = lagmax
        self.absmaxsigma = absmaxsigma
        self.absminsigma = absminsigma
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
        if failreason.astype(np.uint16) & self.FML_BADAMPLOW:
            reasons.append('Fit amplitude too low')
        if failreason.astype(np.uint16) & self.FML_BADAMPHIGH:
            reasons.append('Fit amplitude too high')
        if failreason.astype(np.uint16) & self.FML_BADSEARCHWINDOW:
            reasons.append('Bad search window')
        if failreason.astype(np.uint16) & self.FML_BADWIDTHLOW:
            reasons.append('Bad fit width - value too low')
        if failreason.astype(np.uint16) & self.FML_BADWIDTHHIGH:
            reasons.append('Bad fit width - value too high')
        if failreason.astype(np.uint16) & self.FML_BADLAG:
            reasons.append('Lag out of range')
        if failreason.astype(np.uint16) & self.FML_FITFAIL:
            reasons.append('Refinement failed')
        if failreason.astype(np.uint16) & self.FML_INITFAIL:
            reasons.append('Initialization failed')
        if len(reasons) > 0:
            return ', '.join(reasons)
        else:
            return 'No error'

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
        failreason = np.uint(0)
        maskval = np.uint16(1)  # start out assuming the fit will succeed
        binwidth = self.corrtimeaxis[1] - self.corrtimeaxis[0]

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
        peakstart = np.max([1, maxindex - 1])
        peakend = np.min([len(self.corrtimeaxis) - 2, maxindex + 1])
        while thegrad[peakend + 1] <= 0.0 and peakpoints[peakend + 1] == 1:
            peakend += 1
        while thegrad[peakstart - 1] >= 0.0 and peakpoints[peakstart - 1] == 1:
            peakstart -= 1

        # deal with flat peak top
        while peakend < (len(self.corrtimeaxis) - 3) and corrfunc[peakend] == corrfunc[peakend - 1]:
            peakend += 1
        while peakstart > 2 and corrfunc[peakstart] == corrfunc[peakstart + 1]:
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
            failreason |= (self.FML_INITFAIL | self.FML_BADLAG)
            if maxlag_init <= (self.lagmin - rangeextension - binwidth):
                maxlag_init = self.lagmin - rangeextension - binwidth
            else:
                maxlag_init = self.lagmax + rangeextension + binwidth
            if self.debug:
                print('bad initial')
        if maxsigma_init > self.absmaxsigma:
            failreason |= (self.FML_INITFAIL | self.FML_BADWIDTHHIGH)
            maxsigma_init = self.absmaxsigma
            if self.debug:
                print('bad initial width - too high')
        if peakend - peakstart < 2:
            failreason |= (self.FML_INITFAIL | self.FML_BADSEARCHWINDOW)
            maxsigma_init = np.float64(
                ((2 + 1) * binwidth / (2.0 * np.sqrt(-np.log(self.searchfrac)))) / np.sqrt(2.0))
            if self.debug:
                print('bad initial width - too low')
        if not (self.lthreshval <= maxval_init <= self.uthreshval) and self.enforcethresh:
            failreason |= (self.FML_INITFAIL | self.FML_BADAMPLOW)
            if self.debug:
                print('bad initial amp:', maxval_init, 'is less than', self.lthreshval)
        if (maxval_init < 0.0):
            failreason |= (self.FML_INITFAIL | self.FML_BADAMPLOW)
            maxval_init = 0.0
            if self.debug:
                print('bad initial amp:', maxval_init, 'is less than 0.0')
        if (maxval_init > 1.0):
            failreason |= (self.FML_INITFAIL | self.FML_BADAMPHIGH)
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
            X = self.corrtimeaxis[peakstart:peakend + 1]
            data = corrfunc[peakstart:peakend + 1]
            '''if self.debug:
                print('peakstart, peakend', peakstart, peakend)
                #for i in range(len(data)):
                #    print(X[i], data[i], thegrad[i], )
                pl.figure()
                pl.plot(X, data, 'b')
                pl.plot(X,peakpoints[peakstart:peakend + 1], 'r')
                pl.plot(X, thegrad[peakstart:peakend + 1], 'g')'''
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
            if self.bipolar:
                lowestcorrcoeff = -1.0
            else:
                lowestcorrcoeff = 0.0
            if maxval < lowestcorrcoeff:
                failreason |= (self.FML_FITFAIL + self.FML_BADAMPLOW)
                maxval = lowestcorrcoeff
                if self.debug:
                    print('bad fit amp: maxval is lower than lower limit')
                fitfail = True
            if (np.abs(maxval) > 1.0):
                failreason |= (self.FML_FITFAIL | self.FML_BADAMPHIGH)
                maxval = 1.0 * np.sign(maxval)
                if self.debug:
                    print('bad fit amp: magnitude of', maxval, 'is greater than 1.0')
                fitfail = True
            if (self.lagmin > maxlag) or (maxlag > self.lagmax):
                failreason |= (self.FML_FITFAIL + self.FML_BADLAG)
                if self.debug:
                    print('bad lag after refinement')
                if self.lagmin > maxlag:
                    maxlag = self.lagmin
                else:
                    maxlag = self.lagmax
                fitfail = True
            if maxsigma > self.absmaxsigma:
                failreason |= (self.FML_FITFAIL + self.FML_BADWIDTHHIGH)
                if self.debug:
                    print('bad width after refinement:', maxsigma, '>', self.absmaxsigma)
                maxsigma = self.absmaxsigma
                fitfail = True
            if maxsigma < self.absminsigma:
                failreason |= (self.FML_FITFAIL + self.FML_BADWIDTHLOW)
                if self.debug:
                    print('bad width after refinement:', maxsigma, '<', self.absminsigma)
                maxsigma = self.absminsigma
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


class freqtrack:
    freqs = None
    times = None

    def __init__(self,
                 lowerlim=0.1,
                 upperlim=0.6,
                 nperseg=32,
                 Q=10.0,
                 debug=False):
        self.lowerlim = lowerlim
        self.upperlim = upperlim
        self.nperseg = nperseg
        self.Q = Q
        self.debug = debug
        self.nfft = self.nperseg


    def track(self, x, fs):

        self.freqs, self.times, thespectrogram = sp.signal.spectrogram(np.concatenate([np.zeros(int(self.nperseg // 2)), x, np.zeros(int(self.nperseg // 2))], axis=0),
                                                             fs=fs,
                                                             detrend='constant',
                                                             scaling='spectrum',
                                                             nfft=None,
                                                             window=np.hamming(self.nfft),
                                                             noverlap=(self.nperseg - 1))
        lowerliminpts = tide_util.valtoindex(self.freqs, self.lowerlim)
        upperliminpts = tide_util.valtoindex(self.freqs, self.upperlim)

        if self.debug:
            print(self.times.shape, self.freqs.shape, thespectrogram.shape)
            print(self.times)

        # intitialize the peak fitter
        thefitter = correlation_fitter(corrtimeaxis=self.freqs,
                                       lagmin=self.lowerlim,
                                       lagmax=self.upperlim,
                                       absmaxsigma=10.0,
                                       absminsigma=0.1,
                                       debug=self.debug,
                                       findmaxtype='gauss',
                                       zerooutbadfit=False,
                                       refine=True,
                                       useguess=False,
                                       fastgauss=False
                                       )

        peakfreqs = np.zeros((thespectrogram.shape[1] - 1), dtype=float)
        for i in range(0, thespectrogram.shape[1] - 1):
            maxindex, peakfreqs[i], maxval, maxsigma, maskval, failreason, peakstart, peakend  = thefitter.fit(thespectrogram[:, i])
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
                    print('numharmonics:', numharmonics, nyquistfreq // peakfreqs[i])
                for j in range(numharmonics + 1):
                    workingfreq = (j + 1) * peakfreqs[i]
                    if self.debug:
                        print('workingfreq:', workingfreq)
                    ws = [workingfreq * 0.95,  workingfreq * 1.05]
                    wp = [workingfreq * 0.9, workingfreq * 1.1]
                    gpass = 1.0
                    gstop = 40.0
                    b, a = sp.signal.iirdesign(wp, ws, gpass, gstop, ftype='cheby2', fs=fs)
                    if self.debug:
                        print(i, j, times[i], centerindex, halfwidth, xstart, xend, xend - xstart, wp, ws, len(a), len(b))
                    filtsignal = sp.signal.filtfilt(b, a, sp.signal.filtfilt(b, a, filtsignal))
                pady[xstart:xend] += filtsignal
            else:
                pady[xstart:xend] += padx[xstart:xend]
            padweight[xstart:xend] += 1.0
        return (pady / padweight)[halfwidth:-halfwidth]



