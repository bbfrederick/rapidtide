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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

import rapidtide.core.signal.correlate as tide_corr
import rapidtide.core.signal.miscmath as tide_math
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.resample as tide_resample
import rapidtide.util as tide_util

SIGN_NORMAL = 1.0
SIGN_INVERTED = -1.0
SIGNAL_INVERSION_FACTOR = -1.0


def rrifromphase(timeaxis: NDArray, thephase: NDArray) -> None:
    return None


def cardiacsig(
    thisphase: float | NDArray,
    amps: tuple | NDArray = (1.0, 0.0, 0.0),
    phases: NDArray | None = None,
    overallphase: float = 0.0,
) -> float | NDArray:
    total = 0.0
    if phases is None:
        phases = np.zeros_like(amps)
    for i in range(len(amps)):
        total += amps[i] * np.cos((i + 1) * thisphase + phases[i] + overallphase)
    return total


@dataclass
class CardiacExtractionConfig:
    notchpct: float = 1.5
    notchrolloff: float = 0.5
    invertphysiosign: bool = False
    madnorm: bool = True
    nprocs: int = 1
    arteriesonly: bool = False
    fliparteries: bool = False
    debug: bool = False
    verbose: bool = False
    usemask: bool = True
    multiplicative: bool = True


@dataclass
class CardiacExtractionResult:
    hirescardtc: NDArray
    cardnormfac: float
    hiresresptc: NDArray
    respnormfac: float
    slicesamplerate: float
    numsteps: int
    sliceoffsets: NDArray
    cycleaverage: NDArray
    slicenorms: NDArray

    def __iter__(self):
        return iter(
            (
                self.hirescardtc,
                self.cardnormfac,
                self.hiresresptc,
                self.respnormfac,
                self.slicesamplerate,
                self.numsteps,
                self.sliceoffsets,
                self.cycleaverage,
                self.slicenorms,
            )
        )


def _validate_cardiacfromimage_inputs(
    normdata_byslice: NDArray,
    estweights_byslice: NDArray,
    numslices: int,
    timepoints: int,
    tr: float,
) -> None:
    if timepoints <= 0:
        raise ValueError(f"timepoints must be positive, got {timepoints}")
    if numslices <= 0:
        raise ValueError(f"numslices must be positive, got {numslices}")
    if tr <= 0:
        raise ValueError(f"tr must be positive, got {tr}")
    if normdata_byslice.shape[1] != numslices:
        raise ValueError(
            f"normdata_byslice slice dimension {normdata_byslice.shape[1]} does not match numslices {numslices}"
        )
    if normdata_byslice.shape[2] != timepoints:
        raise ValueError(
            f"normdata_byslice timepoint dimension {normdata_byslice.shape[2]} does not match timepoints {timepoints}"
        )
    if estweights_byslice.shape[1] != numslices:
        raise ValueError(
            f"estweights_byslice slice dimension {estweights_byslice.shape[1]} does not match numslices {numslices}"
        )


def _prepare_weights(
    estweights_byslice: NDArray,
    appflips_byslice: NDArray | None,
    arteriesonly: bool,
    fliparteries: bool,
) -> tuple[NDArray, NDArray]:
    if appflips_byslice is None:
        appflips_byslice = np.ones_like(estweights_byslice)
    else:
        if arteriesonly:
            appflips_byslice[np.where(appflips_byslice > 0.0)] = 0.0

    if fliparteries:
        theseweights_byslice = appflips_byslice.astype(np.float64) * estweights_byslice
    else:
        theseweights_byslice = estweights_byslice
    return appflips_byslice, theseweights_byslice


def _compute_slice_averages(
    normdata_byslice: NDArray,
    theseweights_byslice: NDArray,
    numslices: int,
    timepoints: int,
    numsteps: int,
    sliceoffsets: NDArray,
    signal_sign: float,
    madnorm: bool,
    usemask: bool,
    multiplicative: bool,
    verbose: bool,
) -> tuple[NDArray, NDArray, NDArray]:
    high_res_timecourse = np.zeros((timepoints * numsteps), dtype=np.float64)
    cycleaverage = np.zeros((numsteps), dtype=np.float64)
    slice_averages = np.zeros((numslices, timepoints), dtype=np.float64)
    slicenorms = np.zeros((numslices), dtype=np.float64)

    if not verbose:
        print("Averaging slices...")
    for slice_idx in range(numslices):
        if usemask:
            valid_voxel_indices = np.where(np.abs(theseweights_byslice[:, slice_idx]) > 0)[0]
        else:
            valid_voxel_indices = np.where(np.abs(theseweights_byslice[:, slice_idx] >= 0))[0]
        if len(valid_voxel_indices) > 0:
            weighted_slice_data = np.mean(
                normdata_byslice[valid_voxel_indices, slice_idx, :]
                * theseweights_byslice[valid_voxel_indices, slice_idx, np.newaxis],
                axis=0,
            )
            if madnorm:
                slice_averages[slice_idx, :], slicenorms[slice_idx] = tide_math.madnormalize(
                    weighted_slice_data
                )
            else:
                slice_averages[slice_idx, :] = weighted_slice_data
                slicenorms[slice_idx] = 1.0
            for t in range(timepoints):
                high_res_timecourse[numsteps * t + sliceoffsets[slice_idx]] += (
                    signal_sign * slice_averages[slice_idx, t]
                )
        elif verbose:
            print(f"CARDIACFROMIMAGE: slice {slice_idx} contains no non-zero voxels")

    for i in range(numsteps):
        cycleaverage[i] = np.mean(high_res_timecourse[i:-1:numsteps])
    for t in range(len(high_res_timecourse)):
        if multiplicative:
            high_res_timecourse[t] /= cycleaverage[t % numsteps] + 1.0
        else:
            high_res_timecourse[t] -= cycleaverage[t % numsteps]
    if not verbose:
        print("done")
    return high_res_timecourse, cycleaverage, slicenorms


def _normalize_and_filter_signal(
    prefilter: tide_filt.NoncausalFilter,
    slicesamplerate: float,
    filtered_timecourse: NDArray,
    slicenorms: NDArray,
) -> tuple[NDArray, float]:
    signal, normfac = tide_math.madnormalize(prefilter.apply(slicesamplerate, filtered_timecourse))
    signal *= SIGNAL_INVERSION_FACTOR
    normfac *= np.mean(slicenorms)
    return signal, normfac


def _extract_physiological_signals(
    filtered_timecourse: NDArray,
    slicesamplerate: float,
    cardprefilter: tide_filt.NoncausalFilter,
    respprefilter: tide_filt.NoncausalFilter,
    slicenorms: NDArray,
) -> tuple[NDArray, float, NDArray, float]:
    hirescardtc, cardnormfac = _normalize_and_filter_signal(
        cardprefilter, slicesamplerate, filtered_timecourse, slicenorms
    )
    hiresresptc, respnormfac = _normalize_and_filter_signal(
        respprefilter, slicesamplerate, filtered_timecourse, slicenorms
    )
    return hirescardtc, cardnormfac, hiresresptc, respnormfac


def cardiacfromimage(
    normdata_byslice: NDArray,
    estweights_byslice: NDArray,
    numslices: int,
    timepoints: int,
    tr: float,
    slicetimes: NDArray,
    cardprefilter: tide_filt.NoncausalFilter,
    respprefilter: tide_filt.NoncausalFilter,
    config: CardiacExtractionConfig,
    appflips_byslice: NDArray | None = None,
) -> CardiacExtractionResult:
    _validate_cardiacfromimage_inputs(
        normdata_byslice, estweights_byslice, numslices, timepoints, tr
    )
    numsteps, minstep, sliceoffsets = tide_io.sliceinfo(slicetimes, tr)
    print(
        len(slicetimes),
        "slice times with",
        numsteps,
        "unique values - diff is",
        f"{minstep:.3f}",
    )
    signal_sign = SIGN_INVERTED if config.invertphysiosign else SIGN_NORMAL
    appflips_byslice, theseweights_byslice = _prepare_weights(
        estweights_byslice, appflips_byslice, config.arteriesonly, config.fliparteries
    )
    print("Making slice means...")
    high_res_timecourse, cycleaverage, slicenorms = _compute_slice_averages(
        normdata_byslice,
        theseweights_byslice,
        numslices,
        timepoints,
        numsteps,
        sliceoffsets,
        signal_sign,
        config.madnorm,
        config.usemask,
        config.multiplicative,
        config.verbose,
    )
    if (np.max(high_res_timecourse) - np.min(high_res_timecourse)) == 0.0:
        raise ValueError("CARDIACFROMIMAGE: high_res_timecourse has no variation prior to filtering!")
    if (np.max(cycleaverage) - np.min(cycleaverage)) == 0.0:
        raise ValueError("CARDIACFROMIMAGE: cycleaverage has no variation prior to filtering!")
    if (np.max(slicenorms) - np.min(slicenorms)) == 0.0:
        raise ValueError("CARDIACFROMIMAGE: slicenorms has no variation prior to filtering!")

    slicesamplerate = 1.0 * numsteps / tr
    print(f"Slice sample rate is {slicesamplerate:.3f}")
    print("Notch filtering...")
    filtered_timecourse = tide_filt.harmonicnotchfilter(
        high_res_timecourse,
        slicesamplerate,
        1.0 / tr,
        notchpct=config.notchpct,
        debug=config.debug,
    )
    if (np.max(filtered_timecourse) - np.min(filtered_timecourse)) == 0.0:
        raise ValueError(
            "CARDIACFROMIMAGE: high_res_timecourse has no variation after notch filtering!"
        )
    hirescardtc, cardnormfac, hiresresptc, respnormfac = _extract_physiological_signals(
        filtered_timecourse, slicesamplerate, cardprefilter, respprefilter, slicenorms
    )
    if (np.max(hirescardtc) - np.min(hirescardtc)) == 0.0:
        raise ValueError("CARDIACFROMIMAGE: hirescardtc has no variation after extraction!")
    if (np.max(hiresresptc) - np.min(hiresresptc)) == 0.0:
        raise ValueError("CARDIACFROMIMAGE: hiresresptc has no variation after extraction!")
    return CardiacExtractionResult(
        hirescardtc=hirescardtc,
        cardnormfac=cardnormfac,
        hiresresptc=hiresresptc,
        respnormfac=respnormfac,
        slicesamplerate=slicesamplerate,
        numsteps=numsteps,
        sliceoffsets=sliceoffsets,
        cycleaverage=cycleaverage,
        slicenorms=slicenorms,
    )


def getperiodic(
    inputdata: NDArray,
    Fs: float,
    fundfreq: float,
    ncomps: int = 1,
    width: float = 0.4,
    debug: bool = False,
) -> NDArray:
    outputdata = np.zeros_like(inputdata)
    lowerdist = fundfreq - fundfreq / (1.0 + width)
    upperdist = fundfreq * width
    while ncomps * fundfreq >= Fs / 2.0:
        ncomps -= 1
        print(f"\tncomps reduced to {ncomps}")
    thefundfilter = tide_filt.NoncausalFilter(filtertype="arb")
    for component in range(ncomps):
        arb_lower = (component + 1) * fundfreq - lowerdist
        arb_upper = (component + 1) * fundfreq + upperdist
        arb_lowerstop = 0.9 * arb_lower
        arb_upperstop = 1.1 * arb_upper
        if debug:
            print(
                f"GETPERIODIC: component {component} - arb parameters:{arb_lowerstop}, {arb_lower}, {arb_upper}, {arb_upperstop}"
            )
        thefundfilter.setfreqs(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)
        outputdata += 1.0 * thefundfilter.apply(Fs, inputdata)
    return outputdata


def getcardcoeffs(
    cardiacwaveform: NDArray,
    slicesamplerate: float,
    minhr: float = 40.0,
    maxhr: float = 140.0,
    smoothlen: int = 101,
    debug: bool = False,
) -> float:
    if len(cardiacwaveform) > 1024:
        thex, they = welch(cardiacwaveform, slicesamplerate, nperseg=1024)
    else:
        thex, they = welch(cardiacwaveform, slicesamplerate)
    initpeakfreq = np.round(thex[np.argmax(they)] * 60.0, 2)
    if initpeakfreq > maxhr:
        initpeakfreq = maxhr
    if initpeakfreq < minhr:
        initpeakfreq = minhr
    if debug:
        print("initpeakfreq:", initpeakfreq, "BPM")
    freqaxis, spectrum = tide_filt.spectrum(
        tide_filt.hamming(len(cardiacwaveform)) * cardiacwaveform,
        Fs=slicesamplerate,
        mode="complex",
    )
    minbin = int(minhr // (60.0 * (freqaxis[1] - freqaxis[0])))
    maxbin = int(maxhr // (60.0 * (freqaxis[1] - freqaxis[0])))
    spectrum[:minbin] = 0.0
    spectrum[maxbin:] = 0.0
    from scipy.signal import savgol_filter

    ampspec = savgol_filter(np.abs(spectrum), smoothlen, 3)
    peakfreq = freqaxis[np.argmax(ampspec)]
    if debug:
        print("Cardiac fundamental frequency is", np.round(peakfreq * 60.0, 2), "BPM")
        print("normfac:", np.sqrt(2.0) * tide_math.rms(cardiacwaveform))
    return peakfreq


def checkcardmatch(
    reference: NDArray,
    candidate: NDArray,
    samplerate: float,
    refine: bool = True,
    zeropadding: int = 0,
    debug: bool = False,
) -> tuple[float, float, np.uint16]:
    thecardfilt = tide_filt.NoncausalFilter(filtertype="cardiac")
    trimlength = np.min([len(reference), len(candidate)])
    thexcorr = tide_corr.fastcorrelate(
        tide_math.corrnormalize(
            thecardfilt.apply(samplerate, reference), detrendorder=3, windowfunc="hamming"
        )[:trimlength],
        tide_math.corrnormalize(
            thecardfilt.apply(samplerate, candidate), detrendorder=3, windowfunc="hamming"
        )[:trimlength],
        usefft=True,
        zeropadding=zeropadding,
    )
    xcorrlen = len(thexcorr)
    sampletime = 1.0 / samplerate
    xcorr_x = np.r_[0.0:xcorrlen] * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    searchrange = 5.0
    trimstart = tide_util.valtoindex(xcorr_x, -2.0 * searchrange)
    trimend = tide_util.valtoindex(xcorr_x, 2.0 * searchrange)
    (
        maxindex,
        maxdelay,
        maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    ) = tide_fit.findmaxlag_gauss(
        xcorr_x[trimstart:trimend],
        thexcorr[trimstart:trimend],
        -searchrange,
        searchrange,
        3.0,
        refine=refine,
        zerooutbadfit=False,
        useguess=False,
        fastgauss=False,
        displayplots=False,
    )
    if debug:
        print(
            "CORRELATION:",
            maxindex,
            maxdelay,
            maxval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        )
    return maxval, maxdelay, failreason


def cardiaccycleaverage(
    sourcephases: NDArray,
    destinationphases: NDArray,
    waveform: NDArray,
    procpoints: int,
    congridbins: int,
    gridkernel: str,
    centric: bool,
    cache: bool = True,
    cyclic: bool = True,
) -> NDArray:
    rawapp_bypoint = np.zeros(len(destinationphases), dtype=np.float64)
    weight_bypoint = np.zeros(len(destinationphases), dtype=np.float64)
    for t in procpoints:
        thevals, theweights, theindices = tide_resample.congrid(
            destinationphases,
            tide_math.phasemod(sourcephases[t], centric=centric),
            1.0,
            congridbins,
            kernel=gridkernel,
            cache=cache,
            cyclic=cyclic,
        )
        for i in range(len(theindices)):
            weight_bypoint[theindices[i]] += theweights[i]
            rawapp_bypoint[theindices[i]] += theweights[i] * waveform[t]
    rawapp_bypoint = np.where(
        weight_bypoint > (np.max(weight_bypoint) / 50.0),
        np.nan_to_num(rawapp_bypoint / weight_bypoint),
        0.0,
    )
    minval = np.min(rawapp_bypoint[np.where(weight_bypoint > np.max(weight_bypoint) / 50.0)])
    rawapp_bypoint = np.where(
        weight_bypoint > np.max(weight_bypoint) / 50.0, rawapp_bypoint - minval, 0.0
    )
    return rawapp_bypoint, weight_bypoint


__all__ = [
    "rrifromphase",
    "cardiacsig",
    "CardiacExtractionConfig",
    "CardiacExtractionResult",
    "cardiacfromimage",
    "getperiodic",
    "getcardcoeffs",
    "checkcardmatch",
    "cardiaccycleaverage",
]
