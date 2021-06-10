#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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
"""Functions for calculating correlations and similar metrics between arrays."""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import pyfftw.interfaces.scipy_fftpack as fftpack
import scipy as sp
from numba import jit
from numpy.fft import irfftn, rfftn
from scipy import signal
from sklearn.metrics import mutual_info_score

import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.util as tide_util

pyfftw.interfaces.cache.enable()
LGR = logging.getLogger("GENERAL")

# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotbeaggressive = True
donotusenumba = True


# ----------------------------------------- Conditional imports ---------------------------------------
def conditionaljit():
    """Wrap functions in jit if numba is enabled."""

    def resdec(f):
        if donotusenumba:
            return f
        return jit(f, nopython=False)

    return resdec


def disablenumba():
    """Set a global variable to disable numba."""
    global donotusenumba
    donotusenumba = True


# --------------------------- Correlation functions -------------------------------------------------
def check_autocorrelation(
    corrscale,
    thexcorr,
    delta=0.1,
    acampthresh=0.1,
    aclagthresh=10.0,
    displayplots=False,
    detrendorder=1,
):
    """Check for autocorrelation in an array.

    Parameters
    ----------
    corrscale
    thexcorr
    delta
    acampthresh
    aclagthresh
    displayplots
    windowfunc
    detrendorder

    Returns
    -------
    sidelobetime
    sidelobeamp
    """
    lookahead = 2
    peaks = tide_fit.peakdetect(thexcorr, x_axis=corrscale, delta=delta, lookahead=lookahead)
    maxpeaks = np.asarray(peaks[0], dtype="float64")
    if len(peaks[0]) > 0:
        LGR.debug(peaks)
        zeropkindex = np.argmin(abs(maxpeaks[:, 0]))
        for i in range(zeropkindex + 1, maxpeaks.shape[0]):
            if maxpeaks[i, 0] > aclagthresh:
                return None, None
            if maxpeaks[i, 1] > acampthresh:
                sidelobetime = maxpeaks[i, 0]
                sidelobeindex = tide_util.valtoindex(corrscale, sidelobetime)
                sidelobeamp = thexcorr[sidelobeindex]
                numbins = 1
                while (sidelobeindex + numbins < np.shape(corrscale)[0] - 1) and (
                    thexcorr[sidelobeindex + numbins] > sidelobeamp / 2.0
                ):
                    numbins += 1
                sidelobewidth = (
                    corrscale[sidelobeindex + numbins] - corrscale[sidelobeindex]
                ) * 2.0
                fitstart = sidelobeindex - numbins
                fitend = sidelobeindex + numbins
                sidelobeamp, sidelobetime, sidelobewidth = tide_fit.gaussfit(
                    sidelobeamp,
                    sidelobetime,
                    sidelobewidth,
                    corrscale[fitstart : fitend + 1],
                    thexcorr[fitstart : fitend + 1],
                )

                if displayplots:
                    plt.plot(
                        corrscale[fitstart : fitend + 1],
                        thexcorr[fitstart : fitend + 1],
                        "k",
                        corrscale[fitstart : fitend + 1],
                        tide_fit.gauss_eval(
                            corrscale[fitstart : fitend + 1],
                            [sidelobeamp, sidelobetime, sidelobewidth],
                        ),
                        "r",
                    )
                    plt.show()
                return sidelobetime, sidelobeamp
    return None, None


def shorttermcorr_1D(
    data1, data2, sampletime, windowtime, samplestep=1, detrendorder=0, windowfunc="hamming",
):
    """Calculate short-term sliding-window correlation between two 1D arrays.

    Parameters
    ----------
    data1
    data2
    sampletime
    windowtime
    samplestep
    detrendorder
    windowfunc

    Returns
    -------
    times
    corrpertime
    ppertime
    """
    windowsize = int(windowtime // sampletime)
    halfwindow = int((windowsize + 1) // 2)
    times = []
    corrpertime = []
    ppertime = []
    for i in range(halfwindow, np.shape(data1)[0] - halfwindow, samplestep):
        dataseg1 = tide_math.corrnormalize(
            data1[i - halfwindow : i + halfwindow],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )
        dataseg2 = tide_math.corrnormalize(
            data2[i - halfwindow : i + halfwindow],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )
        thepcorr = sp.stats.stats.pearsonr(dataseg1, dataseg2)
        times.append(i * sampletime)
        corrpertime.append(thepcorr[0])
        ppertime.append(thepcorr[1])
    return (
        np.asarray(times, dtype="float64"),
        np.asarray(corrpertime, dtype="float64"),
        np.asarray(ppertime, dtype="float64"),
    )


def shorttermcorr_2D(
    data1,
    data2,
    sampletime,
    windowtime,
    samplestep=1,
    laglimits=None,
    weighting="None",
    zeropadding=0,
    windowfunc="None",
    detrendorder=0,
    display=False,
):
    """Calculate short-term sliding-window correlation between two 2D arrays.

    Parameters
    ----------
    data1
    data2
    sampletime
    windowtime
    samplestep
    laglimits
    weighting
    zeropadding
    windowfunc
    detrendorder
    display

    Returns
    -------
    times
    xcorrpertime
    Rvals
    delayvals
    valid
    """
    windowsize = int(windowtime // sampletime)
    halfwindow = int((windowsize + 1) // 2)

    if laglimits is not None:
        lagmin = laglimits[0]
        lagmax = laglimits[1]
    else:
        lagmin = -windowtime / 2.0
        lagmax = windowtime / 2.0

    LGR.debug(f"lag limits: {lagmin} {lagmax}")

    """dt = np.diff(time)[0]  # In days...
    fs = 1.0 / dt
    nfft = nperseg
    noverlap = (nperseg - 1)"""

    dataseg1 = tide_math.corrnormalize(
        data1[0 : 2 * halfwindow], detrendorder=detrendorder, windowfunc=windowfunc
    )
    dataseg2 = tide_math.corrnormalize(
        data2[0 : 2 * halfwindow], detrendorder=detrendorder, windowfunc=windowfunc
    )
    thexcorr = fastcorrelate(dataseg1, dataseg2, weighting=weighting, zeropadding=zeropadding)
    xcorrlen = np.shape(thexcorr)[0]
    xcorr_x = (
        np.arange(0.0, xcorrlen) * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    )
    xcorrpertime = []
    times = []
    Rvals = []
    delayvals = []
    valid = []
    for i in range(halfwindow, np.shape(data1)[0] - halfwindow, samplestep):
        dataseg1 = tide_math.corrnormalize(
            data1[i - halfwindow : i + halfwindow],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )
        dataseg2 = tide_math.corrnormalize(
            data2[i - halfwindow : i + halfwindow],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )
        times.append(i * sampletime)
        xcorrpertime.append(
            fastcorrelate(dataseg1, dataseg2, weighting=weighting, zeropadding=zeropadding)
        )
        (
            maxindex,
            thedelayval,
            theRval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        ) = tide_fit.findmaxlag_gauss(
            xcorr_x,
            xcorrpertime[-1],
            lagmin,
            lagmax,
            1000.0,
            refine=True,
            useguess=False,
            fastgauss=False,
            displayplots=False,
        )
        delayvals.append(thedelayval)
        Rvals.append(theRval)
        if failreason == 0:
            valid.append(1)
        else:
            valid.append(0)
    if display:
        plt.imshow(xcorrpertime)
    return (
        np.asarray(times, dtype="float64"),
        np.asarray(xcorrpertime, dtype="float64"),
        np.asarray(Rvals, dtype="float64"),
        np.asarray(delayvals, dtype="float64"),
        np.asarray(valid, dtype="float64"),
    )


def calc_MI(x, y, bins=50):
    """Calculate mutual information between two arrays.

    Notes
    -----
    From https://stackoverflow.com/questions/20491028/
    optimal-way-to-compute-pairwise-mutual-information-using-numpy/
    20505476#20505476
    """
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


@conditionaljit()
def mutual_info_2d(x, y, sigma=1, bins=(256, 256), fast=False, normalized=True, EPS=1.0e-6):
    """Compute (normalized) mutual information between two 1D variate from a joint histogram.

    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma : float, optional
        Sigma for Gaussian smoothing of the joint histogram.
        Default = 1.
    bins : tuple, optional
    fast : bool, optional
    normalized : bool
        If True, this will calculate the normalized mutual information from [1]_.
        Default = False.
    EPS : float, optional
        Default = 1.0e-6.

    Returns
    -------
    nmi: float
        the computed similariy measure

    Notes
    -----
    From Ionnis Pappas

    References
    ----------
    .. [1] Studholme,  jhill & jhawkes (1998).
           "A normalized entropy measure of 3-D medical image alignment".
           in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    """
    if fast:
        xstart = bins[0][0]
        xend = bins[0][-1]
        ystart = bins[1][0]
        yend = bins[1][-1]
        numxbins = len(bins[0]) - 1
        numybins = len(bins[1]) - 1
        cuts = (x >= xstart) & (x < xend) & (y >= ystart) & (y < yend)
        c = ((x[cuts] - xstart) / (xend - xstart) * numxbins).astype(np.int_)
        c += ((y[cuts] - ystart) / (yend - ystart) * numybins).astype(np.int_) * numxbins
        jh = np.bincount(c, minlength=numxbins * numybins).reshape(numxbins, numybins)
    else:
        jh, xbins, ybins = np.histogram2d(x, y, bins=bins)
        LGR.debug(f"{xbins} {ybins}")

    # smooth the jh with a gaussian filter of given sigma
    sp.ndimage.gaussian_filter(jh, sigma=sigma, mode="constant", output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))
    HX = -np.sum(s1 * np.log(s1))
    HY = -np.sum(s2 * np.log(s2))
    HXcommaY = -np.sum(jh * np.log(jh))
    # normfac = np.min([HX, HY])

    if normalized:
        mi = (HX + HY) / (HXcommaY) - 1.0
    else:
        mi = -(HXcommaY - HX - HY)

    LGR.debug(f"{HX} {HY} {HXcommaY} {mi}")

    return mi


@conditionaljit()
def cross_mutual_info(
    x,
    y,
    returnaxis=False,
    negsteps=-1,
    possteps=-1,
    locs=None,
    Fs=1.0,
    norm=True,
    madnorm=False,
    windowfunc="None",
    bins=-1,
    prebin=True,
    sigma=0.25,
    fast=True,
):
    """Calculate cross-mutual information between two 1D arrays.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    returnaxis : bool
        set to True to return the time axis
    negstaps: int
    possteps: int
    locs : list
        a set of offsets at which to calculate the cross mutual information
    Fs=1.0,
    norm : bool
        calculate normalized MI at each offset
    madnorm : bool
        set to True to normalize cross MI waveform by it's median average deviate
    windowfunc : str
        name of the window function to apply to input vectors prior to MI calculation
    bins : int
        number of bins in each dimension of the 2D histogram.  Set to -1 to set automatically
    prebin : bool
        set to true to cache 2D histogram for all offsets
    sigma : float
        histogram smoothing kernel
    fast: bool
        apply speed optimizations

    Returns
    -------
    if returnaxis is True:
        thexmi_x : 1D array
            the set of offsets at which cross mutual information is calcuated
        thexmi_y : 1D array
            the set of cross mutual information values
        len(thexmi_x): int
            the number of cross mutual information values returned
    else:
        thexmi_y : 1D array
            the set of cross mutual information values

    """

    normx = tide_math.corrnormalize(x, detrendorder=1, windowfunc=windowfunc)
    normy = tide_math.corrnormalize(y, detrendorder=1, windowfunc=windowfunc)

    # see if we are using the default number of bins
    if bins < 1:
        bins = int(np.sqrt(len(x) / 5))
        LGR.debug(f"cross_mutual_info: bins set to {bins}")

    # find the bin locations
    if prebin:
        jh, bins0, bins1 = np.histogram2d(normx, normy, bins=(bins, bins))
        bins2d = (bins0, bins1)
    else:
        bins2d = (bins, bins)
        fast = False

    if (negsteps == -1) or (negsteps > len(normy) - 1):
        negsteps = -len(normy) + 1
    else:
        negsteps = -negsteps
    if (possteps == -1) or (possteps > len(normx) - 1):
        possteps = len(normx) - 1
    else:
        possteps = possteps
    if locs is None:
        thexmi_y = np.zeros((-negsteps + possteps + 1))
        LGR.debug(f"negsteps, possteps, len(thexmi_y): {negsteps} {possteps} {len(thexmi_y)}")
        irange = range(negsteps, possteps + 1)
    else:
        thexmi_y = np.zeros((len(locs)), dtype=np.float64)
        irange = np.asarray(locs)
    destloc = -1
    for i in irange:
        if locs is None:
            destloc = i - negsteps
        else:
            destloc += 1
        if i < 0:
            thexmi_y[destloc] = mutual_info_2d(
                normx[: i + len(normy)],
                normy[-i:],
                bins=bins2d,
                normalized=norm,
                fast=fast,
                sigma=sigma,
            )
        elif i == 0:
            thexmi_y[destloc] = mutual_info_2d(
                normx, normy, bins=bins2d, normalized=norm, fast=fast, sigma=sigma,
            )
        else:
            thexmi_y[destloc] = mutual_info_2d(
                normx[i:],
                normy[: len(normy) - i],
                bins=bins2d,
                normalized=norm,
                fast=fast,
                sigma=sigma,
            )

    if madnorm:
        thexmi_y = tide_math.madnormalize(thexmi_y)

    if returnaxis:
        if locs is None:
            thexmi_x = (
                np.linspace(0.0, len(thexmi_y) / Fs, num=len(thexmi_y), endpoint=False)
                + negsteps / Fs
            )
            return thexmi_x, thexmi_y, negsteps + 1
        else:
            thexmi_x = irange
            return thexmi_x, thexmi_y, len(thexmi_x)
    else:
        return thexmi_y


def mutual_info_to_r(themi, d=1):
    """Convert mutual information to Pearson product-moment correlation."""
    return np.power(1.0 - np.exp(-2.0 * themi / d), -0.5)


def delayedcorr(data1, data2, delayval, timestep):
    """Calculate correlation between two 1D arrays, at specific delay.

    Parameters
    ----------
    data1
    data2
    delayval
    timestep

    Returns
    -------
    corr
    """
    return sp.stats.stats.pearsonr(
        data1, tide_resample.timeshift(data2, delayval / timestep, 30)[0]
    )


def cepstraldelay(data1, data2, timestep, displayplots=True):
    """
    Estimate delay between two signals using Choudhary's cepstral analysis method.

    Parameters
    ----------
    data1
    data2
    timestep
    displayplots

    Returns
    -------
    arr

    References
    ----------
    * Choudhary, H., Bahl, R. & Kumar, A.
      Inter-sensor Time Delay Estimation using cepstrum of sum and difference signals in
      underwater multipath environment. in 1-7 (IEEE, 2015). doi:10.1109/UT.2015.7108308
    """
    ceps1, _ = tide_math.complex_cepstrum(data1)
    ceps2, _ = tide_math.complex_cepstrum(data2)
    additive_cepstrum, _ = tide_math.complex_cepstrum(data1 + data2)
    difference_cepstrum, _ = tide_math.complex_cepstrum(data1 - data2)
    residual_cepstrum = additive_cepstrum - difference_cepstrum
    if displayplots:
        tvec = timestep * np.arange(0.0, len(data1))
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_title("cepstrum 1")
        ax1.set_xlabel("quefrency in seconds")
        plt.plot(tvec, ceps1.real, tvec, ceps1.imag)
        ax2 = fig.add_subplot(212)
        ax2.set_title("cepstrum 2")
        ax2.set_xlabel("quefrency in seconds")
        plt.plot(tvec, ceps2.real, tvec, ceps2.imag)
        plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title("additive_cepstrum")
        ax1.set_xlabel("quefrency in seconds")
        plt.plot(tvec, additive_cepstrum.real)
        ax2 = fig.add_subplot(312)
        ax2.set_title("difference_cepstrum")
        ax2.set_xlabel("quefrency in seconds")
        plt.plot(tvec, difference_cepstrum)
        ax3 = fig.add_subplot(313)
        ax3.set_title("residual_cepstrum")
        ax3.set_xlabel("quefrency in seconds")
        plt.plot(tvec, residual_cepstrum.real)
        plt.show()
    return timestep * np.argmax(residual_cepstrum.real[0 : len(residual_cepstrum) // 2])


class AliasedCorrelator:
    """An aliased correlator.

    Parameters
    ----------
    hiressignal : 1D array
        The unaliased waveform to match
    hires_Fs : float
        The sample rate of the unaliased waveform
    lores_Fs : float
        The sample rate of the aliased waveform
    timerange : 1D array
        The delays for which to calculate the correlation function
    hiresstarttime : float, optional
    loresstarttime : float, optional
    padtime : float, optional
    """

    def __init__(
        self,
        hiressignal,
        hires_Fs,
        lores_Fs,
        timerange,
        hiresstarttime=0.0,
        loresstarttime=0.0,
        padtime=30.0,
    ):
        self.hiressignal = hiressignal
        self.hires_Fs = hires_Fs
        self.hiresstarttime = hiresstarttime
        self.lores_Fs = lores_Fs
        self.timerange = timerange
        self.loresstarttime = loresstarttime
        self.highresaxis = (
            np.arange(0.0, len(self.hiressignal)) * (1.0 / self.hires_Fs) - self.hiresstarttime
        )
        self.padtime = padtime
        self.tcgenerator = tide_resample.FastResampler(
            self.highresaxis, self.hiressignal, padtime=self.padtime
        )
        self.aliasedsignals = {}

    def apply(self, loressignal, extraoffset):
        """Apply correlator to aliased waveform.

        Parameters
        ----------
        loressignal: 1D array
            The aliased waveform to match
        extraoffset: float
            Additional offset to apply to hiressignal (e.g. for slice offset)

        Returns
        -------
        corrfunc: 1D array
            The correlation function evaluated at timepoints of timerange
        """
        loresaxis = np.arange(0.0, len(loressignal)) * (1.0 / self.lores_Fs) - self.loresstarttime
        targetsignal = tide_math.corrnormalize(loressignal)
        corrfunc = self.timerange * 0.0
        for i in range(len(self.timerange)):
            theoffset = self.timerange[i] + extraoffset
            offsetkey = "{:.3f}".format(theoffset)
            try:
                aliasedhiressignal = self.aliasedsignals[offsetkey]
                # LGR.info(f"{offsetkey} - cache hit")
            except KeyError:
                # LGR.info(f"{offsetkey} - cache miss")
                self.aliasedsignals[offsetkey] = tide_math.corrnormalize(
                    self.tcgenerator.yfromx(loresaxis + theoffset)
                )
                aliasedhiressignal = self.aliasedsignals[offsetkey]
            corrfunc[i] = np.dot(aliasedhiressignal, targetsignal)
        return corrfunc


def arbcorr(
    input1,
    Fs1,
    input2,
    Fs2,
    start1=0.0,
    start2=0.0,
    windowfunc="hamming",
    method="univariate",
    debug=False,
):
    """Calculate something."""
    if Fs1 > Fs2:
        corrFs = Fs1
        matchedinput1 = input1
        matchedinput2 = tide_resample.upsample(input2, Fs2, corrFs, method=method, debug=debug)
    elif Fs2 > Fs1:
        corrFs = Fs2
        matchedinput1 = tide_resample.upsample(input1, Fs1, corrFs, method=method, debug=debug)
        matchedinput2 = input2
    else:
        corrFs = Fs1
        matchedinput1 = input1
        matchedinput2 = input2
    norm1 = tide_math.corrnormalize(matchedinput1, detrendorder=1, windowfunc=windowfunc)
    norm2 = tide_math.corrnormalize(matchedinput2, detrendorder=1, windowfunc=windowfunc)
    thexcorr_y = signal.fftconvolve(norm1, norm2[::-1], mode="full")
    thexcorr_x = (
        np.linspace(0.0, len(thexcorr_y) / corrFs, num=len(thexcorr_y), endpoint=False)
        - (len(norm1) // 2 + len(norm2) // 2) / corrFs
        + start1
        - start2
    )
    zeroloc = int(np.argmin(np.fabs(thexcorr_x)))
    LGR.debug(f"len(norm1) = {len(norm1)}")
    LGR.debug(f"len(norm2) = {len(norm2)}")
    LGR.debug(f"len(thexcorr_y) = {len(thexcorr_y)}")
    LGR.debug(f"zeroloc = {zeroloc}")
    return thexcorr_x, thexcorr_y, corrFs, zeroloc


def faststcorrelate(
    input1, input2, windowtype="hann", nperseg=32, weighting="None", displayplots=False
):
    """Perform correlation between short-time Fourier transformed arrays."""
    nfft = nperseg
    noverlap = nperseg - 1
    onesided = False
    boundary = "even"
    freqs, times, thestft1 = signal.stft(
        input1,
        fs=1.0,
        window=windowtype,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="linear",
        return_onesided=onesided,
        boundary=boundary,
        padded=True,
        axis=-1,
    )

    freqs, times, thestft2 = signal.stft(
        input2,
        fs=1.0,
        window=windowtype,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="linear",
        return_onesided=onesided,
        boundary=boundary,
        padded=True,
        axis=-1,
    )

    acorrfft1 = thestft1 * np.conj(thestft1)
    acorrfft2 = thestft2 * np.conj(thestft2)
    acorr1 = np.roll(fftpack.ifft(acorrfft1, axis=0).real, nperseg // 2, axis=0)[nperseg // 2, :]
    acorr2 = np.roll(fftpack.ifft(acorrfft2, axis=0).real, nperseg // 2, axis=0)[nperseg // 2, :]
    normfacs = np.sqrt(acorr1 * acorr2)
    product = thestft1 * np.conj(thestft2)
    stcorr = np.roll(fftpack.ifft(product, axis=0).real, nperseg // 2, axis=0)
    for i in range(len(normfacs)):
        stcorr[:, i] /= normfacs[i]

    timestep = times[1] - times[0]
    corrtimes = np.linspace(
        -timestep * (nperseg // 2), timestep * (nperseg // 2), num=nperseg, endpoint=False,
    )

    return corrtimes, times, stcorr


def fastcorrelate(
    input1, input2, usefft=True, zeropadding=0, weighting="None", displayplots=False, debug=False,
):
    """Perform a fast correlation between two arrays.

    Parameters
    ----------
    input1
    input2
    usefft
    zeropadding
    weighting
    displayplots
    debug

    Returns
    -------
    corr

    Notes
    -----
    From http://stackoverflow.com/questions/12323959/fast-cross-correlation-method-in-python.
    """
    len1 = len(input1)
    len2 = len(input2)
    outlen = len1 + len2 - 1
    if zeropadding < 0:
        # autopad
        newlen1 = len1 * 2
        newlen2 = len2 * 2
        paddedinput1 = np.zeros((newlen1), dtype=float)
        paddedinput2 = np.zeros((newlen2), dtype=float)
        paddedinput1[0:len1] = input1
        paddedinput2[0:len2] = input2
        startpt = (len1 + len2) // 2
    elif zeropadding > 0:
        # explicit pad
        newlen1 = len1 + zeropadding
        newlen2 = len2 + zeropadding
        paddedinput1 = np.zeros((newlen1), dtype=float)
        paddedinput2 = np.zeros((newlen2), dtype=float)
        paddedinput1[0:len1] = input1
        paddedinput2[0:len2] = input2
        startpt = zeropadding
    else:
        # no pad
        paddedinput1 = input1
        paddedinput2 = input2
        startpt = 0
    if debug:
        print(f"FASTCORRELATE - padding: {zeropadding}, startpt: {startpt}, outlen: {outlen}")

    if usefft:
        # Do an array flipped convolution, which is a correlation.
        if weighting == "None":
            return signal.fftconvolve(paddedinput1, paddedinput2[::-1], mode="full")[
                startpt : startpt + outlen
            ]
        else:
            return convolve_weighted_fft(
                paddedinput1,
                paddedinput2[::-1],
                mode="full",
                weighting=weighting,
                displayplots=displayplots,
            )[startpt : startpt + outlen]
    else:
        return np.correlate(paddedinput1, paddedinput2, mode="full")


def _centered(arr, newsize):
    """Return the center newsize portion of the array.

    Parameters
    ----------
    arr
    newsize

    Returns
    -------
    arr
    """
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _check_valid_mode_shapes(shape1, shape2):
    """Check that two shapes are 'valid' with respect to one another.

    Specifically, this checks that each item in one tuple is larger than or
    equal to corresponding item in another tuple.

    Parameters
    ----------
    shape1
    shape2

    Raises
    ------
    ValueError
        If at least one item in the first shape is not larger than or equal to
        the corresponding item in the second one.
    """
    for d1, d2 in zip(shape1, shape2):
        if not d1 >= d2:
            raise ValueError(
                "in1 should have at least as many items as in2 in "
                "every dimension for 'valid' mode."
            )


def convolve_weighted_fft(in1, in2, mode="full", weighting="None", displayplots=False):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.
    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`;
        if sizes of `in1` and `in2` are not equal then `in1` has to be the
        larger array.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if np.isscalar(in1) and np.isscalar(in2):  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same rank")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = np.issubdtype(in1.dtype, np.complex) or np.issubdtype(in2.dtype, np.complex)
    size = s1 + s2 - 1

    if mode == "valid":
        _check_valid_mode_shapes(s1, s2)

    # Always use 2**n-sized FFT
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    if not complex_result:
        fft1 = rfftn(in1, fsize)
        fft2 = rfftn(in2, fsize)
        theorigmax = np.max(np.absolute(irfftn(gccproduct(fft1, fft2, "None"), fsize)[fslice]))
        ret = irfftn(gccproduct(fft1, fft2, weighting, displayplots=displayplots), fsize)[
            fslice
        ].copy()
        ret = irfftn(gccproduct(fft1, fft2, weighting, displayplots=displayplots), fsize)[
            fslice
        ].copy()
        ret = ret.real
        ret *= theorigmax / np.max(np.absolute(ret))
    else:
        fft1 = fftpack.fftn(in1, fsize)
        fft2 = fftpack.fftn(in2, fsize)
        theorigmax = np.max(np.absolute(fftpack.ifftn(gccproduct(fft1, fft2, "None"))[fslice]))
        ret = fftpack.ifftn(gccproduct(fft1, fft2, weighting, displayplots=displayplots))[
            fslice
        ].copy()
        ret *= theorigmax / np.max(np.absolute(ret))

    # scale to preserve the maximum

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)


def gccproduct(fft1, fft2, weighting, threshfrac=0.1, displayplots=False):
    """Calculate product for generalized crosscorrelation.

    Parameters
    ----------
    fft1
    fft2
    weighting
    threshfrac
    displayplots

    Returns
    -------
    product
    """
    product = fft1 * fft2
    if weighting == "None":
        return product

    # calculate the weighting function
    if weighting == "liang":
        denom = np.square(
            np.sqrt(np.absolute(fft1 * np.conjugate(fft1)))
            + np.sqrt(np.absolute(fft2 * np.conjugate(fft2)))
        )
    elif weighting == "eckart":
        denom = np.sqrt(np.absolute(fft1 * np.conjugate(fft1))) * np.sqrt(
            np.absolute(fft2 * np.conjugate(fft2))
        )
    elif weighting == "phat":
        denom = np.absolute(product)
    else:
        raise ValueError("illegal weighting function specified in gccproduct")

    if displayplots:
        xvec = range(0, len(denom))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("reciprocal weighting function")
        plt.plot(xvec, abs(denom))
        plt.show()

    # now apply it while preserving the max
    theorigmax = np.max(np.absolute(denom))
    thresh = theorigmax * threshfrac
    if thresh > 0.0:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(
                np.where(np.absolute(denom) > thresh, product / denom, np.float64(0.0))
            )
    else:
        return 0.0 * product
