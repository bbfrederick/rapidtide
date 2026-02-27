#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy as sp

from rapidtide.ffttools import showfftcache
from rapidtide.filter import (
    NoncausalFilter,
    Plethfilter,
    arb_pass,
    csdfilter,
    dobpfiltfilt,
    dobptransfuncfilt,
    dobptrapfftfilt,
    dohpfiltfilt,
    dohptransfuncfilt,
    dohptrapfftfilt,
    dolpfiltfilt,
    dolptransfuncfilt,
    dolptrapfftfilt,
    getfilterbandfreqs,
    gethptransfunc,
    getlpfftfunc,
    getlptransfunc,
    getlptrapfftfunc,
    harmonicnotchfilter,
    ifftfrompolar,
    mRect,
    padvec,
    polarfft,
    pspec,
    rect,
    savgolsmooth,
    setnotchfilter,
    spectralflatness,
    spectrum,
    ssmooth,
    transferfuncfilt,
    unpadvec,
    wiener_deconvolution,
    windowfunction,
)


def maketestwaves(timeaxis):
    tclen = len(timeaxis)
    testwaves = []
    testwaves.append(
        {
            "name": "constant high",
            "timeaxis": 1.0 * timeaxis,
            "waveform": np.ones((tclen), dtype="float"),
        }
    )
    testwaves.append(
        {
            "name": "white noise",
            "timeaxis": 1.0 * timeaxis,
            "waveform": 0.3 * np.random.normal(size=tclen),
        }
    )

    scratch = np.zeros_like(timeaxis)
    scratch[int(tclen / 5) : int(2 * tclen / 5)] = 1.0
    scratch[int(3 * tclen / 5) : int(4 * tclen / 5)] = 1.0
    testwaves.append(
        {
            "name": "block regressor",
            "timeaxis": 1.0 * timeaxis,
            "waveform": 1.0 * scratch,
        }
    )

    scratch = np.zeros_like(timeaxis)
    scratch[int(tclen / 2) :] = 1.0
    testwaves.append(
        {
            "name": "step regressor",
            "timeaxis": 1.0 * timeaxis,
            "waveform": 1.0 * scratch,
        }
    )
    return testwaves


def spectralfilterprops(thefilter, thefiltername, debug=False):
    lowerstop, lowerpass, upperpass, upperstop = thefilter["filter"].getfreqs()
    freqspace = thefilter["frequencies"][1] - thefilter["frequencies"][0]
    lowerstopindex = int(np.floor(lowerstop / freqspace))
    lowerpassindex = int(np.ceil(lowerpass / freqspace))
    upperpassindex = int(np.floor(upperpass / freqspace))
    upperstopindex = int(
        np.min([np.ceil(upperstop / freqspace), len(thefilter["frequencies"]) - 1])
    )
    print(f"max allowable index: {len(thefilter['frequencies']) - 1}")
    lowerstopindex = np.max([0, lowerstopindex])
    lowerpassindex = np.max([0, lowerpassindex])
    upperstopindex = np.min([len(thefilter["frequencies"]) - 1, upperstopindex])
    upperpassindex = np.min([len(thefilter["frequencies"]) - 1, upperpassindex])
    if debug:
        print("filter name:", thefiltername)
        print("freqspace:", freqspace)
        print("target freqs:", lowerstop, lowerpass, upperpass, upperstop)
        print("target indices:", lowerstopindex, lowerpassindex, upperpassindex, upperstopindex)
        print(
            "actual freqs:",
            thefilter["frequencies"][lowerstopindex],
            thefilter["frequencies"][lowerpassindex],
            thefilter["frequencies"][upperpassindex],
            thefilter["frequencies"][upperstopindex],
        )
    response = {}

    passbandmean = np.mean(thefilter["transferfunc"][lowerpassindex:upperpassindex])
    passbandmax = np.max(thefilter["transferfunc"][lowerpassindex:upperpassindex])
    passbandmin = np.min(thefilter["transferfunc"][lowerpassindex:upperpassindex])

    response["passbandripple"] = (passbandmax - passbandmin) / passbandmean

    if lowerstopindex > 2:
        response["lowerstopmean"] = (
            np.mean(thefilter["transferfunc"][0:lowerstopindex]) / passbandmean
        )
        response["lowerstopmax"] = (
            np.max(np.abs(thefilter["transferfunc"][0:lowerstopindex])) / passbandmean
        )
    else:
        response["lowerstopmean"] = 0.0
        response["lowerstopmax"] = 0.0

    if len(thefilter["transferfunc"]) - upperstopindex > 2:
        response["upperstopmean"] = (
            np.mean(thefilter["transferfunc"][upperstopindex:-1]) / passbandmean
        )
        response["upperstopmax"] = (
            np.max(np.abs(thefilter["transferfunc"][upperstopindex:-1])) / passbandmean
        )
    else:
        response["upperstopmean"] = 0.0
        response["upperstopmax"] = 0.0
    return response


def eval_filterprops(
    sampletime=0.72, tclengthinsecs=300.0, numruns=100, displayplots=False, debug=False
):
    np.random.seed(12345)
    tclen = int(tclengthinsecs // sampletime)
    print("Testing transfer function:")
    lowestfreq = 1.0 / (sampletime * tclen)
    nyquist = 0.5 / sampletime
    print(
        "    sampletime=",
        sampletime,
        ", timecourse length=",
        tclengthinsecs,
        "s,  possible frequency range:",
        lowestfreq,
        nyquist,
    )
    timeaxis = np.linspace(0.0, sampletime * tclen, num=tclen, endpoint=False)

    overall = np.random.normal(size=tclen)
    nperseg = np.min([tclen, 2048])
    f, dummy = sp.signal.welch(overall, fs=1.0 / sampletime, nperseg=nperseg)

    transferfunclist = ["brickwall", "trapezoidal", "butterworth"]

    allfilters = []

    # construct all the physiological filters
    for filtertype in ["None", "lfo", "resp", "cardiac", "hrv_lf", "hrv_hf", "hrv_vhf"]:
        testfilter = NoncausalFilter(filtertype=filtertype)
        lstest, lptest, uptest, ustest = testfilter.getfreqs()
        if lptest < nyquist:
            for transferfunc in transferfunclist:
                allfilters.append(
                    {
                        "name": filtertype + " " + transferfunc,
                        "filter": NoncausalFilter(
                            filtertype=filtertype,
                            transferfunc=transferfunc,
                            debug=False,
                        ),
                    }
                )

    # make the lowpass filters
    for transferfunc in transferfunclist:
        testfilter = NoncausalFilter(
            filtertype="arb",
            transferfunc=transferfunc,
            initlowerstop=0.0,
            initlowerpass=0.0,
            initupperpass=0.1,
            initupperstop=0.11,
        )
        lstest, lptest, uptest, ustest = testfilter.getfreqs()
        if lptest < nyquist:
            allfilters.append(
                {
                    "name": "0.1Hz LP " + transferfunc,
                    "filter": NoncausalFilter(
                        filtertype="arb",
                        transferfunc=transferfunc,
                        initlowerstop=0.0,
                        initlowerpass=0.0,
                        initupperpass=0.1,
                        initupperstop=0.11,
                        debug=False,
                    ),
                }
            )

    # make the highpass filters
    for transferfunc in transferfunclist:
        testfilter = NoncausalFilter(
            filtertype="arb",
            transferfunc=transferfunc,
            initlowerstop=0.09,
            initlowerpass=0.1,
            initupperpass=1.0e20,
            initupperstop=1.0e20,
        )
        lstest, lptest, uptest, ustest = testfilter.getfreqs()
        if lptest < nyquist:
            allfilters.append(
                {
                    "name": "0.1Hz HP " + transferfunc,
                    "filter": NoncausalFilter(
                        filtertype="arb",
                        transferfunc=transferfunc,
                        initlowerstop=0.09,
                        initlowerpass=0.1,
                        initupperpass=1.0e20,
                        initupperstop=1.0e20,
                        debug=False,
                    ),
                }
            )

    # calculate the transfer functions for the filters
    for index in range(0, len(allfilters)):
        psd_raw = 0.0 * dummy
        psd_filt = 0.0 * dummy
        for i in range(0, numruns):
            inputsig = np.random.normal(size=tclen)
            outputsig = allfilters[index]["filter"].apply(1.0 / sampletime, inputsig)
            f, raw = sp.signal.welch(inputsig, fs=1.0 / sampletime, nperseg=nperseg)
            f, filt = sp.signal.welch(outputsig, fs=1.0 / sampletime, nperseg=nperseg)
            psd_raw += raw
            psd_filt += filt
        if debug:
            print(f"freqspace for index {index} is {f[1] - f[0]}")
        allfilters[index]["frequencies"] = 1.0 * f
        allfilters[index]["transferfunc"] = psd_filt / psd_raw

    # show transfer functions
    if displayplots:
        legend = []
        plt.figure()
        plt.ylim([-1.0, 1.0 * len(allfilters)])
        offset = 0.0
        for thefilter in allfilters:
            plt.plot(thefilter["frequencies"], thefilter["transferfunc"] + offset)
            legend.append(thefilter["name"])
            offset += 1.0
        plt.legend(legend)
        plt.show()

    # test transfer function responses
    for thefilter in allfilters:
        response = spectralfilterprops(thefilter, thefilter["name"], debug=debug)
        print("    Evaluating", thefilter["name"], "transfer function")
        print("\tpassbandripple:", response["passbandripple"])
        print("\tlowerstopmax:", response["lowerstopmax"])
        print("\tlowerstopmean:", response["lowerstopmean"])
        print("\tupperstopmax:", response["upperstopmax"])
        print("\tupperstopmean:", response["upperstopmean"])
        # assert response['passbandripple'] < 0.45
        assert response["lowerstopmax"] < 1e4
        assert response["lowerstopmean"] < 1e4
        assert response["upperstopmax"] < 1e4
        assert response["upperstopmean"] < 1e4

    # construct some test waveforms for end effects
    testwaves = maketestwaves(timeaxis)

    # show the end effects waveforms
    if displayplots:
        plt.figure()
        plt.ylim([-2.2, 2.2 * len(testwaves)])
        for thewave in testwaves:
            legend = []
            offset = 0.0
            for thefilter in allfilters:
                plt.plot(
                    thewave["timeaxis"],
                    offset + thefilter["filter"].apply(1.0 / sampletime, thewave["waveform"]),
                )
                legend.append(thewave["name"] + ": " + thefilter["name"])
                offset += 1.25
            plt.legend(legend)
            plt.show()


def test_filterprops(displayplots=False, debug=False):
    eval_filterprops(
        sampletime=0.72, tclengthinsecs=300.0, numruns=100, displayplots=displayplots, debug=debug
    )
    eval_filterprops(
        sampletime=2.0, tclengthinsecs=300.0, numruns=100, displayplots=displayplots, debug=debug
    )
    eval_filterprops(
        sampletime=0.1, tclengthinsecs=30000.0, numruns=10, displayplots=displayplots, debug=debug
    )
    eval_filterprops(
        sampletime=0.1, tclengthinsecs=30000.1, numruns=10, displayplots=displayplots, debug=debug
    )
    showfftcache()


def test_plethfilter_initializes_coefficients():
    filt = Plethfilter(Fs=100.0, Fl=1.0, Fh=8.0)

    assert filt.Fn == 50.0
    assert filt.b.ndim == 1
    assert filt.a.ndim == 1
    assert len(filt.b) > 0
    assert len(filt.a) > 0


def test_plethfilter_apply_returns_finite_output():
    filt = Plethfilter(Fs=100.0, Fl=1.0, Fh=8.0)
    t = np.arange(1000) / 100.0
    signal = np.sin(2.0 * np.pi * 2.0 * t) + 0.5 * np.sin(2.0 * np.pi * 20.0 * t)

    filtered = filt.apply(signal)

    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_padvec_and_unpadvec_round_trip():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    padded = padvec(data, padlen=2, avlen=2, padtype="reflect")
    assert padded.shape[0] == data.shape[0] + 4
    assert np.allclose(unpadvec(padded, padlen=2), data)
    assert np.allclose(unpadvec(data, padlen=0), data)


def test_padvec_padding_modes_and_errors():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert np.allclose(padvec(data, padlen=2, padtype="zero"), np.array([0, 0, 1, 2, 3, 4, 0, 0]))
    assert np.allclose(padvec(data, padlen=2, padtype="cyclic"), np.array([3, 4, 1, 2, 3, 4, 1, 2]))
    assert np.allclose(
        padvec(data, padlen=2, padtype="constant"), np.array([1, 1, 1, 2, 3, 4, 4, 4])
    )
    assert np.allclose(
        padvec(data, padlen=2, avlen=2, padtype="constant+"),
        np.array([1.5, 1.5, 1.0, 2.0, 3.0, 4.0, 3.5, 3.5]),
    )
    with pytest.raises(ValueError):
        padvec(data, padlen=2, padtype="not-a-mode")
    with pytest.raises(RuntimeError):
        padvec(data, padlen=8, padtype="reflect")


def test_getfilterbandfreqs_valid_and_errors():
    lowerpass, upperpass, lowerstop, upperstop = getfilterbandfreqs("lfo")
    assert lowerpass == pytest.approx(0.01)
    assert upperpass == pytest.approx(0.15)
    assert lowerstop < lowerpass
    assert upperstop > upperpass
    assert getfilterbandfreqs("lfo", asrange=True) == "0.01-0.15Hz"
    with pytest.raises(SystemExit):
        getfilterbandfreqs("notaband")
    with pytest.raises(SystemExit):
        getfilterbandfreqs("lfo", species="mouse")


@pytest.mark.parametrize(
    "band",
    ["vlf", "lfo_legacy", "lfo_tight", "hrv_ulf", "hrv_vlf", "hrv_lf", "hrv_hf", "hrv_vhf"],
)
def test_getfilterbandfreqs_supported_bands(band):
    lowerpass, upperpass, lowerstop, upperstop = getfilterbandfreqs(band)
    assert lowerpass <= upperpass
    assert lowerstop <= upperstop


def test_noncausalfilter_accessors_and_setfreq_validation():
    filt = NoncausalFilter(filtertype="None")
    filt.settype("lfo")
    assert filt.gettype() == "lfo"
    filt.setbutterorder(5)
    assert filt.butterworthorder == 5
    filt.setdebug(True)
    assert filt.debug is True
    filt.setpadtime(2.5)
    assert filt.getpadtime() == pytest.approx(2.5)
    filt.setpadtype("cyclic")
    assert filt.getpadtype() == "cyclic"
    filt.settransferfunc("brickwall")
    assert filt.transferfunc == "brickwall"

    filt.setfreqs(0.1, 0.2, 0.4, 0.5)
    assert filt.getfreqs() == pytest.approx((0.1, 0.2, 0.4, 0.5))

    with pytest.raises(SystemExit):
        filt.setfreqs(0.2, 0.1, 0.4, 0.5)
    with pytest.raises(SystemExit):
        filt.setfreqs(0.1, 0.2, 0.5, 0.4)
    with pytest.raises(SystemExit):
        filt.setfreqs(0.1, 0.5, 0.4, 0.6)


def test_fft_polar_round_trip():
    x = np.array([0.0, 1.0, 0.5, -0.5, -1.0, 0.25], dtype=np.float64)
    magnitude, phase = polarfft(x)
    reconstructed = ifftfrompolar(magnitude, phase)
    assert reconstructed.shape == x.shape
    assert np.allclose(reconstructed, x, atol=1e-10)


def test_ssmooth_and_spectrum_modes():
    vol = np.zeros((9, 9, 9), dtype=np.float64)
    vol[4, 4, 4] = 1.0
    smoothed = ssmooth(1.0, 1.0, 1.0, 1.0, vol)
    assert smoothed.shape == vol.shape
    assert smoothed[4, 4, 4] < 1.0

    trace = np.sin(2.0 * np.pi * np.arange(128) / 32.0)
    for mode in ["real", "imag", "complex", "mag", "phase", "power"]:
        freqs, vals = spectrum(trace, Fs=1.0, mode=mode, trim=True)
        assert freqs.shape == vals.shape
    with pytest.raises(RuntimeError):
        spectrum(trace, mode="badmode")


def test_window_functions_and_dispatch():
    length = 16
    for wtype in ["hamming", "hann", "blackmanharris", "None"]:
        w = windowfunction(length, type=wtype)
        assert w.shape == (length,)
        assert np.isfinite(w).all()
    assert np.allclose(windowfunction(length, type="None"), np.ones(length))
    assert rect(10, 4).shape == (10,)
    mrect = mRect(32)
    assert mrect.shape == (32,)
    assert np.max(mrect) == pytest.approx(1.0)
    with pytest.raises(SystemExit):
        windowfunction(length, type="not-a-window")


def test_filtering_helpers_return_valid_outputs():
    fs = 50.0
    t = np.arange(400) / fs
    x = np.sin(2.0 * np.pi * 1.5 * t) + 0.4 * np.sin(2.0 * np.pi * 9.0 * t)

    y_lp_bw = dolpfiltfilt(fs, 4.0, x, order=3, padlen=20)
    y_hp_bw = dohpfiltfilt(fs, 0.8, x, order=3, padlen=20)
    y_bp_bw = dobpfiltfilt(fs, 0.8, 4.0, x, order=3, padlen=20)
    y_lp_fft = dolptransfuncfilt(fs, x, upperpass=4.0, upperstop=5.0, type="brickwall", padlen=20)
    y_hp_fft = dohptransfuncfilt(fs, x, lowerpass=0.8, lowerstop=0.6, type="brickwall", padlen=20)
    y_bp_fft = dobptransfuncfilt(
        fs, x, lowerpass=0.8, upperpass=4.0, lowerstop=0.6, upperstop=5.0, type="brickwall", padlen=20
    )
    y_lp_trap = dolptransfuncfilt(fs, x, upperpass=4.0, upperstop=5.0, type="trapezoidal", padlen=20)
    y_hp_trap = dohptransfuncfilt(fs, x, lowerpass=0.8, lowerstop=0.6, type="trapezoidal", padlen=20)
    y_bp_trap = dobptransfuncfilt(
        fs, x, lowerpass=0.8, upperpass=4.0, lowerstop=0.6, upperstop=5.0, type="trapezoidal", padlen=20
    )
    y_arb = arb_pass(
        fs,
        x,
        lowerstop=0.6,
        lowerpass=0.8,
        upperpass=4.0,
        upperstop=5.0,
        transferfunc="butterworth",
        padlen=20,
    )

    for y in [
        y_lp_bw,
        y_hp_bw,
        y_bp_bw,
        y_lp_fft,
        y_hp_fft,
        y_bp_fft,
        y_lp_trap,
        y_hp_trap,
        y_bp_trap,
        y_arb,
    ]:
        assert y.shape == x.shape
        assert np.isfinite(y).all()


def test_transfer_functions_and_signal_domain_helpers():
    fs = 50.0
    x = np.linspace(0.0, 1.0, 256, endpoint=False)
    lp_fft = getlpfftfunc(fs, 4.0, x)
    lp_trap = getlptrapfftfunc(fs, 4.0, 5.0, x)
    lp_bw = getlptransfunc(fs, x, upperpass=4.0, type="brickwall")
    lp_gauss = getlptransfunc(fs, x, upperpass=4.0, type="gaussian")
    hp_tf = gethptransfunc(fs, x, lowerpass=0.8, lowerstop=0.6, type="trapezoidal")

    for tf in [lp_fft, lp_trap, lp_bw, lp_gauss, hp_tf]:
        assert tf.shape == x.shape
        assert np.isfinite(tf).all()

    with pytest.raises(SystemExit):
        getlptransfunc(fs, x, upperpass=None)
    with pytest.raises(SystemExit):
        gethptransfunc(fs, x, lowerpass=None)

    y = transferfuncfilt(x, lp_fft)
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_notch_savgol_csd_wiener_and_spectrum_helpers():
    fs = 100.0
    t = np.arange(6000) / fs
    x = np.sin(2.0 * np.pi * 5.0 * t) + 0.2 * np.sin(2.0 * np.pi * 20.0 * t)

    notch = NoncausalFilter(filtertype="arb_stop", padtime=2.0)
    setnotchfilter(notch, thefreq=20.0, notchwidth=2.0)
    filtered_notch = notch.apply(fs, x)
    assert filtered_notch.shape == x.shape

    harmonic = harmonicnotchfilter(x, fs, Ffundamental=5.0, notchpct=1.0)
    assert harmonic.shape == x.shape

    smooth = savgolsmooth(x, smoothlen=31, polyorder=3)
    assert smooth.shape == x.shape

    csd = csdfilter(x, x + 0.05 * np.random.RandomState(0).normal(size=x.shape[0]), padlen=20)
    assert csd.shape == x.shape

    kernel = np.array([1.0, 0.5, 0.25], dtype=np.float64)
    deconv = wiener_deconvolution(x, kernel, lambd=0.1)
    assert deconv.shape == x.shape

    p = pspec(x)
    assert p.shape == x.shape
    assert np.isfinite(p).all()

    flat = spectralflatness(np.abs(p) + 1e-12)
    assert np.isfinite(flat)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_filterprops(displayplots=True, debug=True)
