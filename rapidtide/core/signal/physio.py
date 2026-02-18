#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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

import time

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew
from statsmodels.robust import mad

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats


def cleanphysio(
    Fs: float,
    physiowaveform: NDArray,
    cutoff: float = 0.4,
    thresh: float = 0.2,
    nyquist: float | None = None,
    iscardiac: bool = True,
    debug: bool = False,
) -> tuple[NDArray, NDArray, NDArray, float]:
    if debug:
        print("Entering cleanphysio")
    print("Filtering")
    filtertype = "cardiac" if iscardiac else "resp"
    physiofilter = tide_filt.NoncausalFilter(filtertype, debug=debug)
    print("Envelope detection")
    envelope = tide_math.envdetect(
        Fs,
        tide_math.madnormalize(physiofilter.apply(Fs, tide_math.madnormalize(physiowaveform)[0]))[
            0
        ],
        cutoff=cutoff,
    )
    envmean = np.mean(envelope)
    envlowerlim = thresh * np.max(envelope)
    envelope = np.where(envelope >= envlowerlim, envelope, envlowerlim)

    arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop = physiofilter.getfreqs()
    physiofilter.settype("arb")
    arb_upper = 10.0
    arb_upperstop = arb_upper * 1.1
    if nyquist is not None and nyquist < arb_upper:
        arb_upper = nyquist
        arb_upperstop = nyquist
    physiofilter.setfreqs(arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop)
    filtphysiowaveform = tide_math.madnormalize(
        physiofilter.apply(Fs, tide_math.madnormalize(physiowaveform)[0])
    )[0]
    print("Normalizing")
    normphysio = tide_math.madnormalize(envmean * filtphysiowaveform / envelope)[0]
    if debug:
        print("Leaving cleanphysio")
    return filtphysiowaveform, normphysio, envelope, envmean


def findbadpts(
    thewaveform: NDArray,
    nameroot: str,
    outputroot: str,
    samplerate: float,
    infodict: dict,
    thetype: str = "mad",
    retainthresh: float = 0.89,
    mingap: float = 2.0,
    outputlevel: int = 0,
    debug: bool = True,
) -> tuple[NDArray, float | tuple[float, float]]:
    if thetype == "mad":
        absdev = np.fabs(thewaveform - np.median(thewaveform))
        medianval = np.median(thewaveform)
        sigma = mad(thewaveform, center=medianval)
        numsigma = np.sqrt(1.0 / (1.0 - retainthresh))
        thresh = numsigma * sigma
        thebadpts = np.where(absdev >= thresh, 1.0, 0.0)
        print("Bad point threshold set to", "{:.3f}".format(thresh), "using the", thetype)
    elif thetype == "fracval":
        lower, upper = tide_stats.getfracvals(
            thewaveform,
            [(1.0 - retainthresh) / 2.0, (1.0 + retainthresh) / 2.0],
        )
        therange = upper - lower
        lowerthresh = lower - therange
        upperthresh = upper + therange
        thebadpts = np.where((lowerthresh <= thewaveform) & (thewaveform <= upperthresh), 0.0, 1.0)
        thresh = (lowerthresh, upperthresh)
    else:
        raise ValueError("findbadpts error: Bad thresholding type")

    streakthresh = int(np.round(mingap * samplerate))
    lastbad = 0
    isbad = thebadpts[0] == 1.0
    for i in range(1, len(thebadpts)):
        if thebadpts[i] == 1.0:
            if not isbad:
                isbad = True
                if i - lastbad < streakthresh:
                    thebadpts[lastbad:i] = 1.0
            lastbad = i
        else:
            isbad = False
    if len(thebadpts) - lastbad - 1 < streakthresh:
        thebadpts[lastbad:] = 1.0
    if outputlevel > 0:
        tide_io.writevec(thebadpts, outputroot + "_" + nameroot + "_badpts.txt")
    infodict[nameroot + "_threshvalue"] = thresh
    infodict[nameroot + "_threshmethod"] = thetype
    return thebadpts, thresh


def approximateentropy(waveform: NDArray, m: int, r: float) -> float:
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        N = len(waveform)
        x = [[waveform[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(waveform)
    return abs(_phi(m + 1) - _phi(m))


def entropy(waveform: NDArray) -> float:
    return -np.sum(np.square(waveform) * np.nan_to_num(np.log2(np.square(waveform))))


def calcplethquality(
    waveform: NDArray,
    Fs: float,
    infodict: dict,
    suffix: str,
    outputroot: str,
    S_windowsecs: float = 5.0,
    K_windowsecs: float = 60.0,
    E_windowsecs: float = 1.0,
    detrendorder: int = 8,
    outputlevel: int = 0,
    initfile: bool = True,
    debug: bool = False,
) -> None:
    dt_waveform = tide_fit.detrend(waveform, order=detrendorder, demean=True)
    S_windowpts = int(np.round(S_windowsecs * Fs, 0))
    S_windowpts += 1 - S_windowpts % 2
    S_waveform = np.zeros_like(dt_waveform)
    K_windowpts = int(np.round(K_windowsecs * Fs, 0))
    K_windowpts += 1 - K_windowpts % 2
    K_waveform = np.zeros_like(dt_waveform)
    E_windowpts = int(np.round(E_windowsecs * Fs, 0))
    E_windowpts += 1 - E_windowpts % 2
    E_waveform = np.zeros_like(dt_waveform)
    for i in range(0, len(dt_waveform)):
        startpt = np.max([0, i - S_windowpts // 2])
        endpt = np.min([i + S_windowpts // 2, len(dt_waveform)])
        S_waveform[i] = skew(dt_waveform[startpt : endpt + 1], nan_policy="omit")
        startpt = np.max([0, i - K_windowpts // 2])
        endpt = np.min([i + K_windowpts // 2, len(dt_waveform)])
        K_waveform[i] = kurtosis(dt_waveform[startpt : endpt + 1], fisher=False)
        startpt = np.max([0, i - E_windowpts // 2])
        endpt = np.min([i + E_windowpts // 2, len(dt_waveform)])
        r = 0.2 * np.std(dt_waveform[startpt : endpt + 1])
        E_waveform[i] = approximateentropy(dt_waveform[startpt : endpt + 1], 2, r)

    infodict["S_sqi_mean" + suffix] = np.mean(S_waveform)
    infodict["S_sqi_median" + suffix] = np.median(S_waveform)
    infodict["S_sqi_std" + suffix] = np.std(S_waveform)
    infodict["K_sqi_mean" + suffix] = np.mean(K_waveform)
    infodict["K_sqi_median" + suffix] = np.median(K_waveform)
    infodict["K_sqi_std" + suffix] = np.std(K_waveform)
    infodict["E_sqi_mean" + suffix] = np.mean(E_waveform)
    infodict["E_sqi_median" + suffix] = np.median(E_waveform)
    infodict["E_sqi_std" + suffix] = np.std(E_waveform)

    if outputlevel > 1:
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            S_waveform,
            Fs,
            columns=["S_sqi" + suffix],
            append=(not initfile),
            debug=debug,
        )
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            K_waveform,
            Fs,
            columns=["K_sqi" + suffix],
            append=True,
            debug=debug,
        )
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            E_waveform,
            Fs,
            columns=["E_sqi" + suffix],
            append=True,
            debug=debug,
        )


def getphysiofile(
    waveformfile: str,
    inputfreq: float,
    inputstart: float | None,
    slicetimeaxis: NDArray,
    stdfreq: float,
    stdpoints: int,
    envcutoff: float,
    envthresh: float,
    timings: list,
    outputroot: str,
    slop: float = 0.25,
    outputlevel: int = 0,
    iscardiac: bool = True,
    debug: bool = False,
) -> tuple[NDArray, NDArray, float, int]:
    if debug:
        print("Entering getphysiofile")
    print("Reading physiological signal from file")

    filefreq, filestart, dummy, waveform_fullres, dummy, dummy = tide_io.readvectorsfromtextfile(
        waveformfile, onecol=True, debug=debug
    )
    if inputfreq < 0.0:
        if filefreq is not None:
            inputfreq = filefreq
        else:
            inputfreq = -inputfreq

    if inputstart is None:
        if filestart is not None:
            inputstart = filestart
        else:
            inputstart = 0.0

    inputtimeaxis = (
        np.linspace(
            0.0,
            (1.0 / inputfreq) * len(waveform_fullres),
            num=len(waveform_fullres),
            endpoint=False,
        )
        + inputstart
    )
    stdtimeaxis = (
        np.linspace(0.0, (1.0 / stdfreq) * stdpoints, num=stdpoints, endpoint=False) + inputstart
    )

    if (inputtimeaxis[0] > slop) or (inputtimeaxis[-1] < slicetimeaxis[-1] - slop):
        print("\tinputtimeaxis[0]:", inputtimeaxis[0])
        print("\tinputtimeaxis[-1]:", inputtimeaxis[-1])
        print("\tslicetimeaxis[0]:", slicetimeaxis[0])
        print("\tslicetimeaxis[-1]:", slicetimeaxis[-1])
        if inputtimeaxis[0] > slop:
            print("\tfailed condition 1:", inputtimeaxis[0], ">", slop)
        if inputtimeaxis[-1] < slicetimeaxis[-1] - slop:
            print("\tfailed condition 2:", inputtimeaxis[-1], "<", slicetimeaxis[-1] - slop)
        raise ValueError("getphysiofile: error - waveform file does not cover the fmri time range")
    timings.append(["Cardiac signal from physiology data read in", time.time(), None, None])

    cleanwaveform_fullres, normwaveform_fullres, waveformenv_fullres, envmean = cleanphysio(
        inputfreq,
        waveform_fullres,
        iscardiac=iscardiac,
        cutoff=envcutoff,
        thresh=envthresh,
        nyquist=inputfreq / 2.0,
        debug=debug,
    )

    if iscardiac:
        if outputlevel > 1:
            tide_io.writevec(waveform_fullres, outputroot + "_rawpleth_native.txt")
            tide_io.writevec(cleanwaveform_fullres, outputroot + "_pleth_native.txt")
            tide_io.writevec(waveformenv_fullres, outputroot + "_cardenvelopefromfile_native.txt")
        timings.append(["Cardiac signal from physiology data cleaned", time.time(), None, None])

    waveform_sliceres = tide_resample.doresample(
        inputtimeaxis, cleanwaveform_fullres, slicetimeaxis, method="univariate", padlen=0
    )
    waveform_stdres = tide_math.madnormalize(
        tide_resample.doresample(
            inputtimeaxis,
            cleanwaveform_fullres,
            stdtimeaxis,
            method="univariate",
            padlen=0,
        )
    )[0]

    timings.append(
        [
            "Cardiac signal from physiology data resampled to slice resolution and saved",
            time.time(),
            None,
            None,
        ]
    )
    if debug:
        print("Leaving getphysiofile")
    return waveform_sliceres, waveform_stdres, inputfreq, len(waveform_fullres)


__all__ = [
    "cleanphysio",
    "findbadpts",
    "approximateentropy",
    "entropy",
    "calcplethquality",
    "getphysiofile",
]
