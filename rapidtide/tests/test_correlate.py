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
#
from unittest.mock import patch

import numpy as np
import pytest

import rapidtide.core.signal.correlate as core_corr
import rapidtide.core.signal.correlate as tide_corr


def _make_pair(length=256, shift=5):
    rng = np.random.RandomState(42)
    x = rng.randn(length).astype(np.float64)
    y = np.roll(x, shift)
    return x, y


def check_autocorrelation_no_peaks(debug=False):
    if debug:
        print("check_autocorrelation_no_peaks")
    corrscale = np.linspace(-2.0, 2.0, 101)
    thexcorr = np.zeros_like(corrscale)
    with patch("rapidtide.core.signal.correlate.tide_fit.peakdetect", return_value=([], [])):
        t, a = tide_corr.check_autocorrelation(corrscale, thexcorr, displayplots=False)
    assert t is None
    assert a is None


def check_autocorrelation_detects_sidelobe(debug=False):
    if debug:
        print("check_autocorrelation_detects_sidelobe")
    corrscale = np.linspace(-3.0, 3.0, 121)
    thexcorr = np.exp(-((corrscale - 0.0) ** 2) / 0.04) + 0.35 * np.exp(-((corrscale - 1.2) ** 2) / 0.04)
    peaks = ([(0.0, 1.0), (1.2, 0.35)], [])
    with (
        patch("rapidtide.core.signal.correlate.tide_fit.peakdetect", return_value=peaks),
        patch("rapidtide.core.signal.correlate.tide_fit.gaussfit", return_value=(0.35, 1.2, 0.2)),
    ):
        t, a = tide_corr.check_autocorrelation(
            corrscale,
            thexcorr,
            acampthresh=0.1,
            aclagthresh=2.0,
            displayplots=False,
        )
    assert np.isclose(t, 1.2)
    assert np.isclose(a, 0.35)


def shorttermcorr_1d_basic(debug=False):
    if debug:
        print("shorttermcorr_1d_basic")
    x, y = _make_pair(length=300, shift=0)
    times, corrvals, pvals = tide_corr.shorttermcorr_1D(
        x, y, sampletime=0.5, windowtime=20.0, samplestep=3
    )
    assert len(times) == len(corrvals) == len(pvals)
    assert np.mean(corrvals) > 0.95


def shorttermcorr_2d_basic(debug=False):
    if debug:
        print("shorttermcorr_2d_basic")
    x, y = _make_pair(length=320, shift=3)
    times, xcorr, rvals, delays, valid = tide_corr.shorttermcorr_2D(
        x, y, sampletime=0.2, windowtime=12.0, samplestep=4, displayplots=False
    )
    assert len(times) == len(rvals) == len(delays) == len(valid)
    assert xcorr.shape[0] == len(times)
    assert set(np.unique(valid)).issubset({0.0, 1.0})


def calc_mi_basic(debug=False):
    if debug:
        print("calc_mi_basic")
    rng = np.random.RandomState(1)
    x = rng.randn(1000)
    y = x + 0.1 * rng.randn(1000)
    mi = tide_corr.calc_MI(x, y, bins=32)
    assert mi > 0.0


def mutual_info_functions(debug=False):
    if debug:
        print("mutual_info_functions")
    rng = np.random.RandomState(2)
    x = rng.randn(1500)
    y = 0.5 * x + 0.3 * rng.randn(1500)
    bins_edges = (np.linspace(-4, 4, 41), np.linspace(-4, 4, 41))
    mifast = tide_corr.mutual_info_2d_fast(x, y, bins_edges, sigma=0.5, normalized=True)
    mi = tide_corr.mutual_info_2d(x, y, bins=(40, 40), sigma=0.5, normalized=True)
    assert np.isfinite(mifast)
    assert np.isfinite(mi)
    assert mifast > 0.0
    assert mi > 0.0


def proc_mi_histogram(debug=False):
    if debug:
        print("proc_mi_histogram")
    jh = np.ones((20, 20), dtype=np.float64)
    mi_norm = tide_corr.proc_MI_histogram(jh.copy(), sigma=0.5, normalized=True)
    mi_raw = tide_corr.proc_MI_histogram(jh.copy(), sigma=0.5, normalized=False)
    assert np.isfinite(mi_norm)
    assert np.isfinite(mi_raw)


def cross_mutual_info_paths(debug=False):
    if debug:
        print("cross_mutual_info_paths")
    x, y = _make_pair(length=256, shift=2)
    # prebin/fast path with axis output
    xaxis, xmi, zeroloc = tide_corr.cross_mutual_info(
        x, y, returnaxis=True, negsteps=10, possteps=10, Fs=2.0, prebin=True, fast=True
    )
    assert len(xaxis) == len(xmi)
    assert isinstance(zeroloc, int)
    # locs/non-fast path with madnorm
    locs = np.array([-5, 0, 5], dtype=int)
    xaxis2, xmi2, n2 = tide_corr.cross_mutual_info(
        x, y, returnaxis=True, locs=locs, prebin=False, fast=False, madnorm=True
    )
    assert n2 == len(locs)
    assert len(xaxis2) == len(xmi2) == len(locs)


def mutual_info_to_r_and_delays(debug=False):
    if debug:
        print("mutual_info_to_r_and_delays")
    r = tide_corr.mutual_info_to_r(0.5, d=1)
    assert np.isfinite(r)
    assert r >= 1.0

    t = np.linspace(0, 20, 400, endpoint=False)
    x = np.sin(2.0 * np.pi * 0.3 * t)
    y = np.roll(x, 5)
    pear = tide_corr.delayedcorr(x, y, delayval=0.0, timestep=t[1] - t[0])
    pear_r = pear.statistic if hasattr(pear, "statistic") else pear[0]
    assert np.isfinite(pear_r)

    d = tide_corr.cepstraldelay(x, y, timestep=t[1] - t[0], displayplots=False)
    assert np.isfinite(d)


def aliased_and_samplerate_routines(debug=False):
    if debug:
        print("aliased_and_samplerate_routines")
    # Use longer signals to satisfy internal resampling filter padding requirements.
    hi = np.sin(np.linspace(0, 80 * np.pi, 4000, endpoint=False))
    low = np.sin(np.linspace(0, 80 * np.pi, 1000, endpoint=False))
    ac = tide_corr.AliasedCorrelator(hi, hires_Fs=20.0, numsteps=4)
    xaxis = ac.getxaxis()
    out = ac.apply(low, offset=0)
    assert len(xaxis) == 2 * len(hi) + 1
    assert len(out) == 2 * len(hi) + 1
    assert len(out) == len(xaxis)

    a1, a2, fs = tide_corr.matchsamplerates(low, 5.0, hi, 20.0)
    assert fs == 20.0
    assert len(a1) == len(a2)

    b1, b2, fs2 = tide_corr.matchsamplerates(hi, 20.0, low, 5.0)
    assert fs2 == 20.0
    assert len(b1) == len(b2)

    c1, c2, fs3 = tide_corr.matchsamplerates(low, 5.0, low.copy(), 5.0)
    assert fs3 == 5.0
    assert len(c1) == len(c2)

    xcorr_x, xcorr_y, corrfs, zeroloc = tide_corr.arbcorr(low, 5.0, low.copy(), 5.0)
    assert len(xcorr_x) == len(xcorr_y)
    assert corrfs == 5.0
    assert 0 <= zeroloc < len(xcorr_x)


def stfft_prime_and_fastcorr(debug=False):
    if debug:
        print("stfft_prime_and_fastcorr")
    x, y = _make_pair(length=128, shift=2)
    corrtimes, times, stcorr = tide_corr.faststcorrelate(x, y, nperseg=16)
    assert stcorr.shape == (16, len(times))
    assert len(corrtimes) == 16

    assert tide_corr.primefacs(12) == [2, 2, 3]
    assert tide_corr.primefacs(17) == [17]

    sig1 = np.zeros(200, dtype=float)
    sig2 = np.zeros(200, dtype=float)
    sig1[80] = 1.0
    sig2[90] = 1.0
    fc_fft = tide_corr.fastcorrelate(sig2, sig1, zeropadding=0)
    fc_nofft = tide_corr.fastcorrelate(sig2, sig1, usefft=False, zeropadding=0)
    std = np.correlate(sig2, sig1, mode="full")
    np.testing.assert_allclose(fc_fft, std, atol=1e-8)
    np.testing.assert_allclose(fc_nofft, std, atol=1e-8)

    for weighting in ["None", "liang", "eckart", "phat", "regressor"]:
        out = tide_corr.fastcorrelate(sig2, sig1, weighting=weighting)
        assert out.shape == std.shape


def centered_convolve_and_gcc(debug=False):
    if debug:
        print("centered_convolve_and_gcc")
    arr = np.arange(24).reshape(4, 6)
    centered = tide_corr._centered(arr, (2, 3))
    assert centered.shape == (2, 3)

    tide_corr._check_valid_mode_shapes((5, 5), (3, 2))
    with pytest.raises(ValueError):
        tide_corr._check_valid_mode_shapes((2, 2), (3, 2))

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0])
    conv_full = tide_corr.convolve_weighted_fft(a, b, mode="full", weighting="None")
    conv_same = tide_corr.convolve_weighted_fft(a, b, mode="same", weighting="phat")
    conv_valid = tide_corr.convolve_weighted_fft(a, b, mode="valid", weighting="eckart")
    assert conv_full.shape[0] == len(a) + len(b) - 1
    assert conv_same.shape[0] == len(a)
    assert conv_valid.shape[0] == len(a) - len(b) + 1

    with pytest.raises(ValueError):
        tide_corr.convolve_weighted_fft(np.array([[1.0]]), np.array([1.0]), mode="full")
    empty = tide_corr.convolve_weighted_fft(np.array([]), np.array([1.0]), mode="full")
    assert empty.size == 0

    fft1 = np.fft.fft(np.array([1.0, 0.0, 0.0, 0.0]))
    fft2 = np.fft.fft(np.array([0.0, 1.0, 0.0, 0.0]))
    for weighting in ["None", "liang", "eckart", "phat", "regressor"]:
        prod = tide_corr.gccproduct(fft1, fft2, weighting=weighting, compress=True)
        assert prod.shape == fft1.shape
    with pytest.raises(ValueError):
        tide_corr.gccproduct(fft1, fft2, weighting="badweight")


def aligntcwithref_basic(debug=False):
    if debug:
        print("aligntcwithref_basic")
    fs = 10.0
    t = np.linspace(0.0, 40.0, int(40 * fs), endpoint=False)
    fixed = np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.sin(2 * np.pi * 0.05 * t)
    moving = np.roll(fixed, 4)
    aligned, delay, maxval, failreason = tide_corr.aligntcwithref(
        fixed, moving, Fs=fs, lagmin=-2.0, lagmax=2.0, refine=True, display=False
    )
    assert aligned.shape == fixed.shape
    assert np.isfinite(delay)
    assert np.isfinite(maxval)
    assert failreason in [0, 1, 2, 3, 4, 5]


def test_correlate(debug=False):
    check_autocorrelation_no_peaks(debug=debug)
    check_autocorrelation_detects_sidelobe(debug=debug)
    shorttermcorr_1d_basic(debug=debug)
    shorttermcorr_2d_basic(debug=debug)
    calc_mi_basic(debug=debug)
    mutual_info_functions(debug=debug)
    proc_mi_histogram(debug=debug)
    cross_mutual_info_paths(debug=debug)
    mutual_info_to_r_and_delays(debug=debug)
    aliased_and_samplerate_routines(debug=debug)
    stfft_prime_and_fastcorr(debug=debug)
    centered_convolve_and_gcc(debug=debug)
    aligntcwithref_basic(debug=debug)
    assert callable(core_corr.fastcorrelate)


if __name__ == "__main__":
    test_correlate(debug=True)
