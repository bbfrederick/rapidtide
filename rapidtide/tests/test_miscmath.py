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
import numpy as np
import pytest

import rapidtide.miscmath as tide_math
from rapidtide.tests.utils import mse


def phase_test(debug=False):
    if debug:
        print("phase_test")
    z = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128)
    ph = tide_math.phase(z)
    assert ph.shape == z.shape
    np.testing.assert_allclose(
        ph, np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
    )


def polarfft_test(debug=False):
    if debug:
        print("polarfft_test")
    vec_even = np.sin(np.linspace(0, 6 * np.pi, 100, endpoint=False))
    freqs, magspec, phspec = tide_math.polarfft(vec_even, samplerate=10.0)
    assert len(freqs) == len(magspec) == len(phspec)

    vec_odd = np.sin(np.linspace(0, 6 * np.pi, 101, endpoint=False))
    freqs2, magspec2, phspec2 = tide_math.polarfft(vec_odd, samplerate=10.0)
    assert len(freqs2) == len(magspec2) == len(phspec2)


def cepstrum_tests(debug=False):
    if debug:
        print("cepstrum_tests")
    x = 1.0 + np.random.RandomState(123).rand(256)
    cceps, ndelay = tide_math.complex_cepstrum(x)
    rceps = tide_math.real_cepstrum(x)
    assert cceps.shape == x.shape
    assert rceps.shape == x.shape
    assert np.isfinite(ndelay)


def deriv_and_factor_tests(debug=False):
    if debug:
        print("deriv_and_factor_tests")
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    d = tide_math.thederiv(y)
    np.testing.assert_allclose(d, np.array([-0.5, 1.5, 2.5, 3.5, 2.0]))

    assert tide_math.primes(24) == [2, 2, 2, 3]
    assert tide_math.primes(17) == [17]
    assert tide_math.largestfac(24) == 3


def normalize_family_tests(debug=False):
    if debug:
        print("normalize_family_tests")
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for method in ["None", "percent", "variance", "stddev", "z", "p2p", "mad"]:
        out = tide_math.normalize(v, method=method)
        assert out.shape == v.shape

    with pytest.raises(ValueError):
        tide_math.normalize(v, method="not_a_method")

    z = tide_math.znormalize(v)
    np.testing.assert_allclose(np.mean(z), 0.0, atol=1e-10)

    cleaned, med, sigmad = tide_math.removeoutliers(
        np.array([0.0, 0.0, 100.0]), zerobad=True, outlierfac=1.0
    )
    assert med == 0.0
    assert sigmad >= 0.0
    assert cleaned[-1] == 0.0

    cleaned2, med2, sigmad2 = tide_math.removeoutliers(
        np.array([0.0, 0.0, 100.0]), zerobad=False, outlierfac=1.0
    )
    assert cleaned2[-1] == med2
    assert sigmad2 >= 0.0

    madnorm, m = tide_math.madnormalize(v)
    assert madnorm.shape == v.shape
    assert m > 0.0

    madnorm2, m2 = tide_math.madnormalize(np.ones(5))
    assert np.all(madnorm2 == 0.0)
    assert m2 == 0.0

    stdn = tide_math.stdnormalize(v)
    varn = tide_math.varnormalize(v)
    pcn = tide_math.pcnormalize(v)
    ppn = tide_math.ppnormalize(v)
    assert stdn.shape == varn.shape == pcn.shape == ppn.shape == v.shape
    np.testing.assert_allclose(np.mean(stdn), 0.0, atol=1e-10)

    # pcnormalize path with non-positive mean should return original vector
    vneg = -np.ones(5)
    np.testing.assert_allclose(tide_math.pcnormalize(vneg), vneg)


def imagevariance_test(debug=False):
    if debug:
        print("imagevariance_test")
    data = np.vstack([np.linspace(1.0, 2.0, 100), np.linspace(2.0, 3.0, 100)])

    out_no_filter = tide_math.imagevariance(data, None, samplefreq=1.0, meannorm=False)
    assert out_no_filter.shape == (2,)

    class DummyFilter:
        def apply(self, Fs, x):
            return 2.0 * x

    out_with_filter = tide_math.imagevariance(data, DummyFilter(), samplefreq=1.0, meannorm=False)
    # Variance scales by factor^2
    np.testing.assert_allclose(out_with_filter, 4.0 * out_no_filter)


def corrnormalize_test(debug=False):
    if debug:
        print("corrnormalize_test")
    x = np.random.RandomState(4).randn(256)
    for window in ["None", "hamming"]:
        for detrendorder in [0, 1]:
            out = tide_math.corrnormalize(x, detrendorder=detrendorder, windowfunc=window)
            assert out.shape == x.shape
            assert np.isfinite(np.sum(out))


def noiseamp_test(debug=False):
    if debug:
        print("noiseamp_test")
    x = np.random.RandomState(5).randn(500)
    filtrms, fit, startamp, endamp, changepct, changerate = tide_math.noiseamp(
        x, Fs=10.0, windowsize=20.0
    )
    assert filtrms.shape == x.shape
    assert fit.shape == x.shape
    assert np.isfinite(startamp)
    assert np.isfinite(endamp)
    assert np.isfinite(changepct)
    assert np.isfinite(changerate)


def rms_and_envelope_tests(debug=False):
    if debug:
        print("rms_and_envelope_tests")
    xaxis = np.linspace(0.0, 1.0, num=500, endpoint=False)
    therms = tide_math.rms(np.sin(2.0 * np.pi * xaxis))
    assert np.fabs(therms - np.sqrt(2.0) / 2.0) < 1e-5

    hifreq = 100.0
    lowfreq = 3.0
    basefunc = np.sin(hifreq * 2.0 * np.pi * xaxis)
    modfunc = 0.5 + 0.1 * np.sin(lowfreq * 2.0 * np.pi * xaxis)
    theenvelope = tide_math.envdetect(1.0, basefunc * modfunc, cutoff=0.1, padlen=100)
    assert theenvelope.shape == modfunc.shape
    assert mse(theenvelope, modfunc) < 0.04


def phasemod_and_trendfilt_tests(debug=False):
    if debug:
        print("phasemod_and_trendfilt_tests")
    phase = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    pm_centric = tide_math.phasemod(phase, centric=True)
    pm_noncentric = tide_math.phasemod(phase, centric=False)
    assert pm_centric.shape == phase.shape
    assert pm_noncentric.shape == phase.shape
    assert np.all(pm_noncentric >= 0.0)

    t = np.linspace(0.0, 1.0, 200, endpoint=False)
    trend = 5.0 + 2.0 * t
    signal = trend + 0.1 * np.sin(2.0 * np.pi * 10.0 * t)
    signal[50] = 20.0  # outlier
    filtered = tide_math.trendfilt(signal, order=1, ndevs=3.0, debug=False)
    assert filtered.shape == signal.shape
    assert np.isfinite(np.sum(filtered))
    # Outlier handling should move the sample closer to the baseline trend.
    assert np.fabs(filtered[50] - trend[50]) < np.fabs(signal[50] - trend[50])


def complexpca_tests(debug=False):
    if debug:
        print("complexpca_tests")
    rng = np.random.RandomState(6)
    X = rng.randn(50, 6) + 1j * rng.randn(50, 6)
    pca = tide_math.ComplexPCA(n_components=3)
    pca.fit(X, use_gpu=False)
    assert pca.components_ is not None
    assert pca.mean_ is not None
    assert pca.explained_variance_ratio_ is not None

    Z = pca.transform(X)
    Xrec = pca.inverse_transform(Z)
    assert Z.shape == (50, 6)
    assert Xrec.shape == X.shape
    assert np.isfinite(np.abs(Xrec).sum())


def test_miscmath(debug=False, displayplots=False):
    np.random.seed(12345)
    phase_test(debug=debug)
    polarfft_test(debug=debug)
    cepstrum_tests(debug=debug)
    deriv_and_factor_tests(debug=debug)
    normalize_family_tests(debug=debug)
    imagevariance_test(debug=debug)
    corrnormalize_test(debug=debug)
    noiseamp_test(debug=debug)
    rms_and_envelope_tests(debug=debug)
    phasemod_and_trendfilt_tests(debug=debug)
    complexpca_tests(debug=debug)


if __name__ == "__main__":
    test_miscmath(debug=True, displayplots=False)
