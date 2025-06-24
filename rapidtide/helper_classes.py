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


class fMRIDataset:
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
    validvoxels = None

    def __init__(self, thedata, zerodata=False, copydata=False, numskip=0):
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

    def setvalid(self, validvoxels):
        self.validvoxels = validvoxels

    def byslice(self):
        return self.thedata[:, :, :, self.numskip :].reshape(
            (self.slicesize, self.numslices, self.timepoints)
        )

    def byvol(self):
        return self.thedata[:, :, :, self.numskip :].reshape((self.numvox, self.timepoints))

    def byvox(self):
        return self.thedata[:, :, :, self.numskip :]


class ProbeRegressor:
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

    def __init__(
        self,
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
        self.makeinputtimeaxis()
        self.targetoversample = targetoversample
        self.targetpoints = targetpoints
        self.targetstartpoint = targetstartpoint

    def setinputvec(self, inputvec, inputfreq, inputstart=0.0):
        self.inputvec = inputvec
        self.inputfreq = inputfreq
        self.inputstart = inputstart

    def makeinputtimeaxis(self):
        self.inputtimeaxis = np.linspace(0.0, len(self.inputvec)) / self.inputfreq - (
            self.inputstarttime + self.inputoffset
        )

    def maketargettimeaxis(self):
        self.targettimeaxis = np.linspace(
            self.targetperiod * self.targetstartpoint,
            self.targetperiod * self.targetstartpoint + self.targetperiod * self.targetpoints,
            num=self.targetpoints,
            endpoint=True,
        )


class Coherer:
    reftc = None
    prepreftc = None
    testtc = None
    preptesttc = None
    freqaxis = None
    similarityfunclen = 0
    datavalid = False
    freqaxisvalid = False
    similarityfuncorigin = 0
    freqmin = None
    freqmax = None

    def __init__(
        self,
        Fs=0.0,
        freqmin=None,
        freqmax=None,
        ncprefilter=None,
        reftc=None,
        detrendorder=1,
        windowfunc="hamming",
        debug=False,
    ):
        self.Fs = Fs
        self.ncprefilter = ncprefilter
        self.reftc = reftc
        self.windowfunc = windowfunc
        self.detrendorder = detrendorder
        self.debug = debug
        if freqmin is not None:
            self.freqmin = freqmin
        if freqmax is not None:
            self.freqmax = freqmax
        if self.reftc is not None:
            self.setreftc(self.reftc)
        if self.debug:
            print("Coherer init:")
            print("\tFs:", self.Fs)
            print("\twindowfunc:", self.windowfunc)
            print("\tdetrendorder:", self.detrendorder)
            print("\tfreqmin:", self.freqmin)
            print("\tfreqmax:", self.freqmax)

    def preptc(self, thetc):
        # prepare timecourse by filtering, normalizing, detrending, and applying a window function
        return tide_math.corrnormalize(
            self.ncprefilter.apply(self.Fs, thetc),
            detrendorder=self.detrendorder,
            windowfunc="None",
        )

    def setlimits(self, freqmin, freqmax):
        self.freqmin = freqmin
        self.freqmax = freqmax
        if self.freqaxisvalid:
            self.freqmininpts = np.max([0, tide_util.valtoindex(self.freqaxis, self.freqmin)])
            self.freqmaxinpts = np.min(
                [
                    tide_util.valtoindex(self.freqaxis, self.freqmax),
                    len(self.freqaxis) - 1,
                ]
            )
        if self.debug:
            print("setlimits:")
            print("\tfreqmin,freqmax:", self.freqmin, self.freqmax)
            print("\tfreqmininpts,freqmaxinpts:", self.freqmininpts, self.freqmaxinpts)

    def getaxisinfo(self):
        return (
            self.freqaxis[self.freqmininpts],
            self.freqaxis[self.freqmaxinpts],
            self.freqaxis[1] - self.freqaxis[0],
            self.freqmaxinpts - self.freqmininpts,
        )

    def setreftc(self, reftc):
        self.reftc = reftc + 0.0
        self.prepreftc = self.preptc(self.reftc)

        # get frequency axis, etc
        self.freqaxis, self.thecoherence = sp.signal.coherence(
            self.prepreftc, self.prepreftc, fs=self.Fs
        )
        #                                                       window=self.windowfunc)'''
        self.similarityfunclen = len(self.thecoherence)
        self.similarityfuncorigin = 0
        self.freqaxisvalid = True
        self.datavalid = False
        if self.freqmin is None or self.freqmax is None:
            self.setlimits(self.freqaxis[0], self.freqaxis[-1])
        self.freqmininpts = tide_util.valtoindex(
            self.freqaxis, self.freqmin, discretization="floor", debug=self.debug
        )
        self.freqmaxinpts = tide_util.valtoindex(
            self.freqaxis, self.freqmax, discretization="ceiling", debug=self.debug
        )

    def trim(self, vector):
        return vector[self.freqmininpts : self.freqmaxinpts]

    def run(self, thetc, trim=True, alt=False):
        if len(thetc) != len(self.reftc):
            print(
                "Coherer: timecourses are of different sizes:",
                len(thetc),
                "!=",
                len(self.reftc),
                "- exiting",
            )
            sys.exit()

        self.testtc = thetc + 0.0
        self.preptesttc = self.preptc(self.testtc)

        # now actually do the coherence
        if self.debug:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(self.prepreftc, "r")
            plt.plot(self.preptesttc, "b")
            plt.legend(["reference", "test timecourse"])
            plt.show()

        if not alt:
            self.freqaxis, self.thecoherence = sp.signal.coherence(
                self.prepreftc, self.preptesttc, fs=self.Fs
            )
        else:
            self.freqaxis, self.thecsdxy = sp.signal.csd(
                10000.0 * self.prepreftc,
                10000.0 * self.preptesttc,
                fs=self.Fs,
                scaling="spectrum",
            )
            self.freqaxis, self.thecsdxx = sp.signal.csd(
                10000.0 * self.prepreftc,
                10000.0 * self.prepreftc,
                fs=self.Fs,
                scaling="spectrum",
            )
            self.freqaxis, self.thecsdyy = sp.signal.csd(
                10000.0 * self.preptesttc,
                10000.0 * self.preptesttc,
                fs=self.Fs,
                scaling="spectrum",
            )
            self.thecoherence = np.nan_to_num(
                abs(self.thecsdxy) ** 2 / (abs(self.thecsdxx) * abs(self.thecsdyy))
            )

        self.similarityfunclen = len(self.thecoherence)
        self.similarityfuncorigin = 0
        self.datavalid = True

        if trim:
            if alt:
                self.themax = np.argmax(self.thecoherence[self.freqmininpts : self.freqmaxinpts])
                return (
                    self.trim(self.thecoherence),
                    self.trim(self.freqaxis),
                    self.themax,
                    self.trim(self.thecsdxx),
                    self.trim(self.thecsdyy),
                    self.trim(self.thecsdxy),
                )
            else:
                self.themax = np.argmax(self.thecoherence[self.freqmininpts : self.freqmaxinpts])
                return (
                    self.trim(self.thecoherence),
                    self.trim(self.freqaxis),
                    self.themax,
                )
        else:
            if alt:
                self.themax = np.argmax(self.thecoherence)
                return (
                    self.thecoherence,
                    self.freqaxis,
                    self.themax,
                    self.thecsdxx,
                    self.thecsdyy,
                    self.thecsdxy,
                )

            else:
                self.themax = np.argmax(self.thecoherence)
                return self.thecoherence, self.freqaxis, self.themax
