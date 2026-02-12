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
import numpy as np
import pytest

import rapidtide.fit as tide_fit


def _unwrap_jit(func):
    """Get the original Python function from a numba JIT-compiled function."""
    return getattr(func, "py_func", func)


# ========================= Evaluation functions =========================


class TestGaussEval:
    def test_peak_at_center(self):
        x = np.linspace(-5, 5, 101)
        p = np.array([2.0, 0.0, 1.0])
        y = tide_fit.gauss_eval(x, p)
        assert np.argmax(y) == 50
        assert np.isclose(y[50], 2.0, atol=1e-10)

    def test_peak_at_offset(self):
        x = np.linspace(-5, 5, 1001)
        p = np.array([3.0, 1.5, 0.5])
        y = tide_fit.gauss_eval(x, p)
        peak_idx = np.argmax(y)
        assert np.isclose(x[peak_idx], 1.5, atol=0.01)
        assert np.isclose(y[peak_idx], 3.0, atol=0.01)

    def test_symmetry(self):
        x = np.linspace(-5, 5, 101)
        p = np.array([1.0, 0.0, 1.0])
        y = tide_fit.gauss_eval(x, p)
        np.testing.assert_allclose(y[:50], y[50:][::-1][:-1], atol=1e-10)

    def test_width(self):
        x = np.linspace(-10, 10, 10001)
        sigma = 2.0
        p = np.array([1.0, 0.0, sigma])
        y = tide_fit.gauss_eval(x, p)
        half_max = 0.5
        above_half = np.where(y >= half_max)[0]
        fwhm_measured = x[above_half[-1]] - x[above_half[0]]
        fwhm_expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        assert np.isclose(fwhm_measured, fwhm_expected, atol=0.01)

    def test_zero_amplitude(self):
        x = np.linspace(-5, 5, 101)
        p = np.array([0.0, 0.0, 1.0])
        y = tide_fit.gauss_eval(x, p)
        np.testing.assert_allclose(y, 0.0, atol=1e-10)


class TestGaussSkEval:
    def test_zero_skewness_symmetric(self):
        x = np.linspace(-5, 5, 101)
        p = np.array([1.0, 0.0, 1.0, 0.0])
        y = tide_fit.gausssk_eval(x, p)
        assert y[50] > 0
        assert len(y) == 101

    def test_positive_skewness(self):
        x = np.linspace(-5, 5, 1001)
        p = np.array([1.0, 0.0, 1.0, 5.0])
        y = tide_fit.gausssk_eval(x, p)
        peak_idx = np.argmax(y)
        assert x[peak_idx] > 0.0  # positive skew shifts peak right

    def test_negative_skewness(self):
        x = np.linspace(-5, 5, 1001)
        p = np.array([1.0, 0.0, 1.0, -5.0])
        y = tide_fit.gausssk_eval(x, p)
        peak_idx = np.argmax(y)
        assert x[peak_idx] < 0.0  # negative skew shifts peak left


class TestKaiserBesselEval:
    def test_peak_at_center(self):
        x = np.linspace(-2, 2, 201)
        p = np.array([4.0, 1.0])
        y = tide_fit.kaiserbessel_eval(x, p)
        assert np.argmax(y) == 100  # center

    def test_zero_outside_support(self):
        x = np.array([2.0, 3.0, -2.0, -3.0])
        p = np.array([4.0, 1.0])
        y = tide_fit.kaiserbessel_eval(x, p)
        np.testing.assert_allclose(y[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(y[3], 0.0, atol=1e-10)

    def test_symmetry(self):
        x = np.linspace(-1, 1, 101)
        p = np.array([4.0, 1.0])
        y = tide_fit.kaiserbessel_eval(x, p)
        np.testing.assert_allclose(y[:50], y[50:][::-1][:-1], atol=1e-10)


class TestTrapezoidEval:
    def test_zero_before_start(self):
        p = np.array([5.0, 2.0, 1.0, 1.0])  # start, amplitude, risetime, falltime
        result = tide_fit.trapezoid_eval(3.0, 2.0, p)
        assert result == 0.0

    def test_rising_phase(self):
        p = np.array([0.0, 2.0, 1.0, 1.0])
        result = tide_fit.trapezoid_eval(0.5, 2.0, p)
        assert result > 0.0
        assert result < 2.0

    def test_falling_phase(self):
        p = np.array([0.0, 2.0, 1.0, 1.0])
        result = tide_fit.trapezoid_eval(3.0, 1.0, p)
        assert result > 0.0
        assert result < 2.0


class TestTrapezoidEvalLoop:
    def test_matches_scalar(self):
        x = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        p = np.array([0.0, 2.0, 1.0, 1.0])
        toplength = 2.0
        y = tide_fit.trapezoid_eval_loop(x, toplength, p)
        for i, xi in enumerate(x):
            assert np.isclose(y[i], tide_fit.trapezoid_eval(xi, toplength, p))


class TestRisetimeEval:
    def test_zero_before_start(self):
        p = np.array([2.0, 1.0, 0.5])  # x0, amplitude, tau
        result = tide_fit.risetime_eval(1.0, p)
        assert result == 0.0

    def test_rises_after_start(self):
        p = np.array([0.0, 1.0, 0.5])
        result = tide_fit.risetime_eval(1.0, p)
        assert result > 0.0
        assert result < 1.0

    def test_approaches_amplitude(self):
        p = np.array([0.0, 1.0, 0.5])
        result = tide_fit.risetime_eval(100.0, p)
        assert np.isclose(result, 1.0, atol=1e-5)


class TestRisetimeEvalLoop:
    def test_matches_scalar(self):
        x = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        p = np.array([0.0, 1.0, 0.5])
        y = tide_fit.risetime_eval_loop(x, p)
        for i, xi in enumerate(x):
            assert np.isclose(y[i], tide_fit.risetime_eval(xi, p))


class TestGaussfunc:
    def test_peak_height(self):
        x = np.array([0.0])
        result = tide_fit.gaussfunc(x, height=3.0, loc=0.0, FWHM=2.0)
        assert np.isclose(result[0], 3.0)

    def test_fwhm(self):
        FWHM = 2.0
        x = np.linspace(-10, 10, 10001)
        y = tide_fit.gaussfunc(x, height=1.0, loc=0.0, FWHM=FWHM)
        above_half = np.where(y >= 0.5)[0]
        fwhm_measured = x[above_half[-1]] - x[above_half[0]]
        assert np.isclose(fwhm_measured, FWHM, atol=0.01)


class TestSincfunc:
    def test_peak_at_center(self):
        x = np.array([2.0])
        result = tide_fit.sincfunc(x, height=1.0, loc=2.0, FWHM=1.0, baseline=0.0)
        assert np.isclose(result[0], 1.0, atol=1e-5)

    def test_baseline(self):
        x = np.array([100.0])  # far from peak
        result = tide_fit.sincfunc(x, height=1.0, loc=0.0, FWHM=1.0, baseline=5.0)
        assert np.isclose(result[0], 5.0, atol=0.1)


# ========================= Residual functions =========================


class TestGaussResiduals:
    def test_zero_residuals_on_exact_data(self):
        x = np.linspace(-5, 5, 101)
        p = np.array([1.0, 0.0, 1.0])
        y = tide_fit.gauss_eval(x, p)
        residuals = tide_fit.gaussresiduals(p, y, x)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_nonzero_residuals_on_wrong_params(self):
        x = np.linspace(-5, 5, 101)
        p_true = np.array([1.0, 0.0, 1.0])
        p_wrong = np.array([2.0, 0.0, 1.0])
        y = tide_fit.gauss_eval(x, p_true)
        residuals = tide_fit.gaussresiduals(p_wrong, y, x)
        assert np.max(np.abs(residuals)) > 0.0


class TestGaussSkResiduals:
    def test_zero_residuals_on_exact_data(self):
        x = np.linspace(-5, 5, 101)
        p = np.array([1.0, 0.0, 1.0, 0.5])
        y = tide_fit.gausssk_eval(x, p)
        residuals = tide_fit.gaussskresiduals(p, y, x)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


class TestTrapezoidResiduals:
    def test_zero_residuals_on_exact_data(self):
        x = np.linspace(0, 10, 100)
        p = np.array([1.0, 2.0, 1.0, 1.0])
        toplength = 3.0
        y = tide_fit.trapezoid_eval_loop(x, toplength, p)
        residuals = tide_fit.trapezoidresiduals(p, y, x, toplength)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


class TestRisetimeResiduals:
    def test_zero_residuals_on_exact_data(self):
        x = np.linspace(0, 10, 100)
        p = np.array([1.0, 2.0, 0.5])
        y = tide_fit.risetime_eval_loop(x, p)
        residuals = tide_fit.risetimeresiduals(p, y, x)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# ========================= Fitting functions =========================


class TestGaussfit:
    def test_recovers_parameters(self):
        x = np.linspace(-5, 5, 201)
        p_true = np.array([2.0, 0.5, 1.0])
        y = tide_fit.gauss_eval(x, p_true)
        h, l, w = tide_fit.gaussfit(1.5, 0.0, 1.5, x, y)
        assert np.isclose(h, 2.0, atol=0.1)
        assert np.isclose(l, 0.5, atol=0.1)
        assert np.isclose(np.abs(w), 1.0, atol=0.1)


class TestGaussfit2:
    def test_recovers_parameters(self):
        FWHM = 2.0
        x = np.linspace(-10, 10, 501)
        y = tide_fit.gaussfunc(x, height=3.0, loc=1.0, FWHM=FWHM)
        h, l, w = tide_fit.gaussfit2(2.5, 0.5, 1.5, x, y)
        assert np.isclose(h, 3.0, atol=0.1)
        assert np.isclose(l, 1.0, atol=0.1)
        assert np.isclose(w, FWHM, atol=0.1)


class TestGaussfitsk:
    def test_recovers_parameters(self):
        x = np.linspace(-5, 5, 201)
        p_true = np.array([1.0, 0.0, 1.0, 0.5])
        y = tide_fit.gausssk_eval(x, p_true)
        result = tide_fit.gaussfitsk(0.8, 0.1, 1.2, 0.3, x, y)
        assert len(result) == 4
        # the fit should produce values close to the true parameters
        assert np.isclose(result[0], 1.0, atol=0.3)
        assert np.isclose(result[1], 0.0, atol=0.3)
        assert np.isclose(result[2], 1.0, atol=0.3)


class TestSincfit:
    def test_recovers_parameters(self):
        x = np.linspace(-10, 10, 501)
        y = tide_fit.sincfunc(x, height=2.0, loc=0.5, FWHM=1.5, baseline=0.1)
        popt, pcov = tide_fit.sincfit(1.5, 0.0, 1.0, 0.0, x, y)
        assert np.isclose(popt[0], 2.0, atol=0.2)
        assert np.isclose(popt[1], 0.5, atol=0.2)
        assert np.isclose(popt[2], 1.5, atol=0.2)
        assert np.isclose(popt[3], 0.1, atol=0.2)


class TestFindrisetimefunc:
    def test_recovers_parameters(self):
        x = np.linspace(0, 20, 200)
        p_true = np.array([2.0, 3.0, 1.5])
        y = tide_fit.risetime_eval_loop(x, p_true)
        start, amp, rise, success = tide_fit.findrisetimefunc(
            x, y, initguess=np.array([1.0, 2.5, 1.0])
        )
        assert success == 1
        assert np.isclose(start, 2.0, atol=0.5)
        assert np.isclose(amp, 3.0, atol=0.5)
        assert np.isclose(rise, 1.5, atol=0.5)

    def test_out_of_bounds_returns_zero(self):
        x = np.linspace(0, 20, 200)
        p_true = np.array([2.0, 3.0, 1.5])
        y = tide_fit.risetime_eval_loop(x, p_true)
        start, amp, rise, success = tide_fit.findrisetimefunc(
            x, y, initguess=np.array([1.0, 2.5, 1.0]), maxrise=0.01
        )
        assert success == 0


class TestFindtrapezoidfunc:
    def test_recovers_parameters(self):
        x = np.linspace(0, 20, 200)
        toplength = 3.0
        p_true = np.array([2.0, 3.0, 1.5, 2.0])
        y = tide_fit.trapezoid_eval_loop(x, toplength, p_true)
        s, a, r, f, success = tide_fit.findtrapezoidfunc(
            x, y, toplength, initguess=np.array([1.0, 2.5, 1.0, 1.5])
        )
        assert success == 1
        assert np.isclose(s, 2.0, atol=0.5)
        assert np.isclose(a, 3.0, atol=0.5)


# ========================= Trend / Detrend =========================


class TestTrendgen:
    def test_constant(self):
        x = np.linspace(-2, 2, 101)
        coffs = np.array([5.0])  # constant
        y = tide_fit.trendgen(x, coffs, demean=True)
        np.testing.assert_allclose(y, 5.0, atol=1e-10)

    def test_linear(self):
        x = np.linspace(-2, 2, 101)
        coffs = np.array([2.0, 1.0])  # 2*x + 1
        y = tide_fit.trendgen(x, coffs, demean=True)
        expected = 2.0 * x + 1.0
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_demean_false(self):
        x = np.linspace(-2, 2, 101)
        coffs = np.array([2.0, 1.0])
        y = tide_fit.trendgen(x, coffs, demean=False)
        expected = 2.0 * x  # no constant term
        np.testing.assert_allclose(y, expected, atol=1e-10)


class TestDetrend:
    def test_removes_linear_trend_with_mean(self):
        """With demean=True, both slope and intercept are removed."""
        n = 100
        thetimepoints = np.arange(0.0, n, 1.0) - n / 2.0
        data = 3.0 * thetimepoints + 5.0
        detrended = tide_fit.detrend(data, order=1, demean=True)
        np.testing.assert_allclose(detrended, 0.0, atol=1e-5)

    def test_removes_slope_only(self):
        """With demean=False, only the slope is removed; constant remains."""
        n = 100
        thetimepoints = np.arange(0.0, n, 1.0) - n / 2.0
        data = 3.0 * thetimepoints + 5.0
        detrended = tide_fit.detrend(data, order=1, demean=False)
        # Only the slope should be removed, constant 5.0 remains
        np.testing.assert_allclose(detrended, 5.0, atol=1e-3)

    def test_removes_constant(self):
        data = np.ones(100) * 7.0
        detrended = tide_fit.detrend(data, order=0, demean=True)
        np.testing.assert_allclose(detrended, 0.0, atol=1e-10)

    def test_quadratic(self):
        n = 200
        thetimepoints = np.arange(0.0, n, 1.0) - n / 2.0
        data = 0.01 * thetimepoints**2 + 2.0 * thetimepoints
        detrended = tide_fit.detrend(data, order=2, demean=True)
        np.testing.assert_allclose(detrended, 0.0, atol=1e-3)

    def test_preserves_length(self):
        data = np.random.randn(100)
        detrended = tide_fit.detrend(data, order=1)
        assert len(detrended) == len(data)


# ========================= Prewhitening =========================


class TestPrewhiten:
    def test_output_same_length(self):
        np.random.seed(42)
        series = np.random.randn(200)
        whitened = tide_fit.prewhiten(series, nlags=2)
        assert len(whitened) == len(series)

    def test_output_same_length_ARIMA(self):
        np.random.seed(42)
        series = np.random.randn(200)
        whitened = tide_fit.prewhiten(series, nlags=None)
        assert len(whitened) == len(series)

    def test_produces_finite_output(self):
        np.random.seed(42)
        # create strongly autocorrelated series
        n = 1000
        series = np.zeros(n)
        series[0] = np.random.randn()
        for i in range(1, n):
            series[i] = 0.9 * series[i - 1] + 0.1 * np.random.randn()
        whitened = tide_fit.prewhiten(series, nlags=1)
        assert len(whitened) == len(series)
        assert np.all(np.isfinite(whitened))


class TestPrewhiten2:
    def test_output_same_length(self):
        np.random.seed(42)
        series = np.random.randn(200)
        whitened = tide_fit.prewhiten2(series, nlags=3)
        assert len(whitened) == len(series)


# ========================= Gasboxcar =========================


class TestGasboxcar:
    def test_returns_none(self):
        data = np.random.rand(100)
        result = tide_fit.gasboxcar(data, 10.0, 1.0, 2.0, 3.0, 4.0)
        assert result is None


# ========================= Peak refinement =========================


class TestRefinePeakQuad:
    def test_gaussian_peak(self):
        x = np.linspace(-5, 5, 101)
        y = np.exp(-0.5 * x**2)
        peak_idx = np.argmax(y)
        peakloc, peakval, peakwidth, ismax, badfit = tide_fit.refinepeak_quad(x, y, peak_idx)
        assert np.isclose(peakloc, 0.0, atol=0.1)
        assert np.isclose(peakval, 1.0, atol=0.1)
        assert ismax is True
        assert badfit is False

    def test_edge_returns_badfit(self):
        x = np.linspace(-5, 5, 101)
        y = np.exp(-0.5 * x**2)
        # Use stride=2 so that peakindex=0 triggers boundary check (0 < 2-1 = 1)
        peakloc, peakval, peakwidth, ismax, badfit = tide_fit.refinepeak_quad(x, y, 0, stride=2)
        assert badfit is True

    def test_minimum(self):
        x = np.linspace(-5, 5, 101)
        y = -np.exp(-0.5 * x**2)
        min_idx = np.argmin(y)
        peakloc, peakval, peakwidth, ismax, badfit = tide_fit.refinepeak_quad(x, y, min_idx)
        assert ismax is False

    def test_flat_region_badfit(self):
        refinepeak_quad = _unwrap_jit(tide_fit.refinepeak_quad)
        x = np.linspace(-5, 5, 101)
        y = np.ones_like(x)
        peakloc, peakval, peakwidth, ismax, badfit = refinepeak_quad(x, y, 50)
        assert badfit is True


class TestMaxindexNoedge:
    def test_finds_interior_max(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.1, 0.5, 0.8, 0.3, 0.2])
        idx, flip = tide_fit.maxindex_noedge(x, y)
        assert idx == 2
        assert flip == 1.0

    def test_bipolar_negative(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.1, 0.2, -0.9, 0.3, 0.2])
        idx, flip = tide_fit.maxindex_noedge(x, y, bipolar=True)
        assert idx == 2
        assert flip == -1.0

    def test_avoids_edge(self):
        x = np.arange(5, dtype=float)
        y = np.array([0.9, 0.5, 0.3, 0.2, 0.1])
        idx, flip = tide_fit.maxindex_noedge(x, y)
        # max is at edge (index 0), should be avoided
        assert idx >= 1


# ========================= findmaxlag_gauss =========================


class TestFindmaxlagGauss:
    def _make_gauss_correlation(self, center=2.0, sigma=1.0, npts=201):
        x = np.linspace(-10, 10, npts)
        y = 0.8 * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        return x, y

    def test_basic_no_refine(self):
        x, y = self._make_gauss_correlation(center=0.0, sigma=1.0)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = (
            tide_fit.findmaxlag_gauss(x, y, -5.0, 5.0, 3.0, refine=False)
        )
        assert maskval == 1
        assert np.isclose(maxlag, 0.0, atol=0.2)
        assert np.isclose(maxval, 0.8, atol=0.1)

    def test_with_refine(self):
        x, y = self._make_gauss_correlation(center=1.0, sigma=1.0)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = (
            tide_fit.findmaxlag_gauss(x, y, -5.0, 5.0, 3.0, refine=True)
        )
        assert maskval == 1
        assert np.isclose(maxlag, 1.0, atol=0.2)

    def test_below_threshold(self):
        x, y = self._make_gauss_correlation(center=0.0, sigma=1.0)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = (
            tide_fit.findmaxlag_gauss(x, y, -5.0, 5.0, 3.0, threshval=0.9, enforcethresh=True)
        )
        assert maskval == 0

    def test_out_of_range_lag(self):
        x, y = self._make_gauss_correlation(center=8.0, sigma=1.0)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = (
            tide_fit.findmaxlag_gauss(x, y, -5.0, 5.0, 3.0, refine=True, zerooutbadfit=True)
        )
        assert maskval == 0

    def test_fastgauss(self):
        x, y = self._make_gauss_correlation(center=0.0, sigma=1.0)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, fitstart, fitend = (
            tide_fit.findmaxlag_gauss(x, y, -5.0, 5.0, 3.0, refine=True, fastgauss=True)
        )
        assert maskval == 1
        assert np.isclose(maxlag, 0.0, atol=0.5)


# ========================= Gram-Schmidt =========================


class TestGramSchmidt:
    def test_orthonormal_output(self):
        vectors = np.array([[2.0, 1.0], [3.0, 4.0]])
        basis = tide_fit.gram_schmidt(vectors)
        assert basis.shape == (2, 2)
        # check orthogonality
        dot = np.dot(basis[0], basis[1])
        assert np.isclose(dot, 0.0, atol=1e-10)
        # check normality
        for i in range(basis.shape[0]):
            assert np.isclose(np.linalg.norm(basis[i]), 1.0, atol=1e-10)

    def test_linearly_dependent(self):
        vectors = np.array([[1.0, 0.0], [2.0, 0.0]])
        basis = tide_fit.gram_schmidt(vectors)
        assert basis.shape[0] == 1  # one dropped

    def test_three_vectors(self):
        vectors = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        basis = tide_fit.gram_schmidt(vectors)
        assert basis.shape == (3, 3)
        # pairwise orthogonality
        for i in range(3):
            for j in range(i + 1, 3):
                assert np.isclose(np.dot(basis[i], basis[j]), 0.0, atol=1e-10)


# ========================= Regression =========================


class TestMlregress:
    def test_simple_linear(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        coffs, R2 = tide_fit.mlregress(X, y)
        assert np.isclose(R2, 1.0, atol=1e-5)
        # slope should be 2.0
        assert np.isclose(coffs[0, 1], 2.0, atol=0.1)

    def test_with_intercept(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # y = 2*x + 1
        coffs, R2 = tide_fit.mlregress(X, y, intercept=True)
        assert np.isclose(R2, 1.0, atol=1e-5)
        assert np.isclose(coffs[0, 0], 1.0, atol=0.1)  # intercept
        assert np.isclose(coffs[0, 1], 2.0, atol=0.1)  # slope

    def test_multiple_features(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] - 1.0 * X[:, 2] + 5.0
        coffs, R2 = tide_fit.mlregress(X, y)
        assert np.isclose(R2, 1.0, atol=1e-5)

    def test_shape_mismatch_transposed(self):
        X = np.array([[1.0, 2.0, 3.0]])  # 1x3, needs transpose
        y = np.array([2.0, 4.0, 6.0])
        coffs, R2 = tide_fit.mlregress(X, y)
        assert R2 > 0.9

    def test_incompatible_shapes_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(AttributeError):
            tide_fit.mlregress(X, y)


class TestOlsregress:
    def test_simple_linear(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        params, r = tide_fit.olsregress(X, y)
        assert r > 0.99


class TestMlproject:
    def test_with_intercept(self):
        thefit = np.array([1.0, 2.0, 3.0])
        theevs = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        result = tide_fit.mlproject(thefit, theevs, intercept=True)
        expected = 1.0 + 2.0 * np.array([1, 2, 3]) + 3.0 * np.array([4, 5, 6])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_without_intercept(self):
        thefit = np.array([2.0, 3.0])
        theevs = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        result = tide_fit.mlproject(thefit, theevs, intercept=False)
        expected = 2.0 * np.array([1, 2, 3])
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ========================= Linear fit filtering =========================
# These functions are decorated with @conditionaljit() but use Python objects
# (sklearn, tqdm) that are incompatible with numba nopython mode. When numba
# is installed, we call the underlying Python function via .py_func.


class TestLinfitfilt:
    def test_removes_linear_component(self):
        linfitfilt = _unwrap_jit(tide_fit.linfitfilt)
        n = 100
        ev = np.linspace(0, 1, n).reshape(-1, 1)
        data = 3.0 * ev.ravel() + 1.0
        filtered, datatoremove, R2, coffs, intercept = linfitfilt(data, ev)
        # linfitfilt removes regressor components but keeps intercept
        assert np.std(filtered) < 1e-5
        assert R2 > 0.99

    def test_2d_evs(self):
        linfitfilt = _unwrap_jit(tide_fit.linfitfilt)
        n = 100
        ev1 = np.linspace(0, 1, n)
        ev2 = np.sin(np.linspace(0, 2 * np.pi, n))
        evs = np.column_stack([ev1, ev2])
        data = 2.0 * ev1 + 3.0 * ev2 + 1.0
        filtered, datatoremove, R2, coffs, intercept = linfitfilt(data, evs)
        # linfitfilt removes regressor components but keeps intercept
        assert np.std(filtered) < 1e-3
        assert R2 > 0.99


class TestExpandedlinfitfilt:
    def test_single_ev_ncomps1(self, monkeypatch):
        monkeypatch.setattr(tide_fit, "linfitfilt", _unwrap_jit(tide_fit.linfitfilt))
        expandedlinfitfilt = _unwrap_jit(tide_fit.expandedlinfitfilt)
        n = 100
        ev = np.linspace(0, 1, n).reshape(-1, 1)
        data = 3.0 * ev.ravel() + 1.0
        filtered, newevs, removed, R, coffs = expandedlinfitfilt(data, ev, ncomps=1)
        assert np.std(filtered) < 1e-5

    def test_polynomial_expansion(self, monkeypatch):
        monkeypatch.setattr(tide_fit, "linfitfilt", _unwrap_jit(tide_fit.linfitfilt))
        expandedlinfitfilt = _unwrap_jit(tide_fit.expandedlinfitfilt)
        n = 100
        ev = np.linspace(-1, 1, n)
        data = 2.0 * ev + 0.5 * ev**2 + 1.0
        filtered, newevs, removed, R, coffs = expandedlinfitfilt(data, ev, ncomps=2)
        assert np.std(filtered) < 1e-3


class TestDerivativelinfitfilt:
    def test_single_ev(self, monkeypatch):
        monkeypatch.setattr(tide_fit, "linfitfilt", _unwrap_jit(tide_fit.linfitfilt))
        derivativelinfitfilt = _unwrap_jit(tide_fit.derivativelinfitfilt)
        n = 100
        ev = np.linspace(0, 1, n)
        data = 3.0 * ev + 1.0
        filtered, newevs, removed, R, coffs = derivativelinfitfilt(data, ev, nderivs=1)
        assert len(filtered) == n

    def test_nderivs_zero(self, monkeypatch):
        monkeypatch.setattr(tide_fit, "linfitfilt", _unwrap_jit(tide_fit.linfitfilt))
        derivativelinfitfilt = _unwrap_jit(tide_fit.derivativelinfitfilt)
        n = 100
        ev = np.linspace(0, 1, n).reshape(-1, 1)
        data = 3.0 * ev.ravel() + 1.0
        filtered, newevs, removed, R, coffs = derivativelinfitfilt(data, ev, nderivs=0)
        assert np.std(filtered) < 1e-5


# ========================= Confound regression =========================


class TestConfoundregress:
    def test_removes_regressors(self):
        confoundregress = _unwrap_jit(tide_fit.confoundregress)
        np.random.seed(42)
        n_voxels = 10
        n_timepoints = 100
        regressors = np.random.randn(2, n_timepoints)
        # data is a linear combination of regressors
        weights = np.random.randn(n_voxels, 2)
        data = weights @ regressors
        filtered, r2 = confoundregress(data, regressors, showprogressbar=False)
        np.testing.assert_allclose(filtered, 0.0, atol=1e-5)
        for i in range(n_voxels):
            assert r2[i] > 0.99

    def test_output_shape(self):
        confoundregress = _unwrap_jit(tide_fit.confoundregress)
        np.random.seed(42)
        data = np.random.randn(5, 50)
        regressors = np.random.randn(2, 50)
        filtered, r2 = confoundregress(data, regressors, showprogressbar=False)
        assert filtered.shape == data.shape
        assert r2.shape == (5,)


# ========================= Expanded regressors =========================


class TestCalcexpandedregressors:
    def test_basic(self):
        confounddict = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "b": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        }
        regressors, labels = tide_fit.calcexpandedregressors(confounddict, deriv=False, order=1)
        assert regressors.shape == (2, 5)
        assert labels == ["a", "b"]

    def test_with_derivatives(self):
        confounddict = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
        regressors, labels = tide_fit.calcexpandedregressors(confounddict, deriv=True, order=1)
        assert regressors.shape == (2, 5)
        assert "a" in labels
        assert "a_deriv" in labels

    def test_with_higher_order(self):
        confounddict = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
        regressors, labels = tide_fit.calcexpandedregressors(confounddict, deriv=False, order=2)
        assert "a" in labels
        assert "a^2" in labels

    def test_with_labels_filter(self):
        confounddict = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "b": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        }
        regressors, labels = tide_fit.calcexpandedregressors(
            confounddict, labels=["a"], deriv=False, order=1
        )
        assert regressors.shape == (1, 5)
        assert labels == ["a"]

    def test_start_end_slicing(self):
        confounddict = {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
        regressors, labels = tide_fit.calcexpandedregressors(
            confounddict, start=1, end=3, deriv=False, order=1
        )
        assert regressors.shape == (1, 3)
        np.testing.assert_allclose(regressors[0], [2.0, 3.0, 4.0])


# ========================= Peak detection =========================


class TestGetpeaks:
    def test_finds_peaks(self):
        x = np.linspace(-10, 10, 1000)
        y = np.sin(x)
        peaks = tide_fit.getpeaks(x, y)
        assert len(peaks) > 0

    def test_bipolar(self):
        x = np.linspace(-10, 10, 1000)
        y = np.sin(x)
        peaks = tide_fit.getpeaks(x, y, bipolar=True)
        # should find both positive and negative peaks
        pos = [p for p in peaks if p[1] > 0]
        neg = [p for p in peaks if p[1] < 0]
        assert len(pos) > 0
        assert len(neg) > 0

    def test_xrange_filter(self):
        x = np.linspace(-10, 10, 1000)
        y = np.sin(x)
        peaks = tide_fit.getpeaks(x, y, xrange=(-2, 2))
        for p in peaks:
            assert -2 <= p[0] <= 2


class TestDatacheckPeakdetect:
    def test_with_x_axis(self):
        x, y = tide_fit._datacheck_peakdetect(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert len(x) == 3
        assert len(y) == 3

    def test_without_x_axis(self):
        x, y = tide_fit._datacheck_peakdetect(None, np.array([4, 5, 6]))
        np.testing.assert_array_equal(x, np.array([0, 1, 2]))

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            tide_fit._datacheck_peakdetect(np.array([1, 2]), np.array([4, 5, 6]))


class TestPeakdetect:
    def test_finds_maxima_and_minima(self):
        x = np.linspace(0, 4 * np.pi, 1000)
        y = np.sin(x)
        max_peaks, min_peaks = tide_fit.peakdetect(y, x, lookahead=50, delta=0.5)
        assert len(max_peaks) > 0
        assert len(min_peaks) > 0

    def test_invalid_lookahead_raises(self):
        with pytest.raises(ValueError):
            tide_fit.peakdetect(np.array([1, 2, 3]), lookahead=0)

    def test_negative_delta_raises(self):
        with pytest.raises(ValueError):
            tide_fit.peakdetect(np.array([1, 2, 3]), delta=-1.0)


# ========================= Scree tests =========================


class TestAfscreetest:
    def test_returns_int(self):
        # Eigenvalues with a clear elbow at index 2
        eigenvals = np.array([10.0, 8.0, 2.0, 0.5, 0.2, 0.1, 0.05])
        result = tide_fit.afscreetest(eigenvals)
        assert isinstance(result, int)

    def test_monotonic(self):
        eigenvals = np.array([100.0, 50.0, 10.0, 1.0, 0.1])
        result = tide_fit.afscreetest(eigenvals)
        assert isinstance(result, int)


class TestOcscreetest:
    def test_basic(self):
        eigenvals = np.array([10.0, 5.0, 0.5, 0.2, 0.1, 0.05])
        result = tide_fit.ocscreetest(eigenvals)
        assert isinstance(result, (int, np.integer))
        assert result >= 1

    def test_with_many_eigenvalues(self):
        eigenvals = np.array([10.0, 8.0, 5.0, 3.0, 2.0, 1.5, 1.3, 1.1, 0.8, 0.5, 0.2, 0.1])
        result = tide_fit.ocscreetest(eigenvals)
        assert isinstance(result, (int, np.integer))
        assert result >= 1


# ========================= Phase analysis =========================


class TestPhaseanalysis:
    def test_basic(self, capsys):
        t = np.linspace(0, 1, 500)
        signal = np.sin(2 * np.pi * 5 * t)
        phase, amp, analytic = tide_fit.phaseanalysis(signal)
        assert len(phase) == len(signal)
        assert len(amp) == len(signal)
        assert len(analytic) == len(signal)
        # amplitude envelope should be approximately 1
        assert np.mean(amp[50:-50]) > 0.8

    def test_phase_is_monotonic(self, capsys):
        t = np.linspace(0, 1, 500)
        signal = np.sin(2 * np.pi * 5 * t)
        phase, amp, analytic = tide_fit.phaseanalysis(signal)
        # unwrapped phase should be monotonically decreasing or increasing
        diffs = np.diff(phase[50:-50])
        # most diffs should have same sign
        signs = np.sign(diffs)
        dominant = np.sign(np.sum(signs))
        assert np.mean(signs == dominant) > 0.9


# ========================= simfuncpeakfit =========================


class TestSimfuncpeakfit:
    def _make_corrfunc(self, center=0.0, sigma=2.0, amp=0.8, n=1001):
        t = np.linspace(-30, 30, n)
        corr = amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)
        return corr, t

    def test_gauss_fit(self):
        corr, t = self._make_corrfunc(center=1.0, sigma=2.0, amp=0.7)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="gauss"
        )
        assert maskval == 1
        assert np.isclose(maxlag, 1.0, atol=0.5)
        assert np.isclose(maxval, 0.7, atol=0.2)

    def test_quad_fit(self):
        corr, t = self._make_corrfunc(center=0.0, sigma=2.0, amp=0.8)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="quad"
        )
        assert maskval == 1
        assert np.isclose(maxlag, 0.0, atol=0.5)

    def test_fastgauss_fit(self):
        corr, t = self._make_corrfunc(center=0.0, sigma=2.0, amp=0.8)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="fastgauss"
        )
        assert maskval == 1

    def test_fastquad_fit(self):
        corr, t = self._make_corrfunc(center=0.0, sigma=2.0, amp=0.8)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="fastquad"
        )
        assert maskval == 1

    def test_com_fit(self):
        corr, t = self._make_corrfunc(center=0.0, sigma=2.0, amp=0.8)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="COM"
        )
        assert maskval == 1

    def test_none_fit(self):
        corr, t = self._make_corrfunc(center=0.0, sigma=2.0, amp=0.8)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="None"
        )
        assert maskval == 1

    def test_below_threshold(self):
        """With peakfittype='None', init failure is not overridden by fit refinement."""
        corr, t = self._make_corrfunc(center=0.0, sigma=2.0, amp=0.3)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="None", lthreshval=0.5
        )
        assert maskval == 0

    def test_bipolar(self):
        t = np.linspace(-30, 30, 1001)
        corr = -0.8 * np.exp(-0.5 * ((t - 1.0) / 2.0) ** 2)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="gauss", bipolar=True
        )
        assert maskval == 1
        assert maxval < 0.0

    def test_useguess(self):
        corr, t = self._make_corrfunc(center=2.0, sigma=2.0, amp=0.7)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="gauss", useguess=True, maxguess=2.0
        )
        assert maskval == 1
        assert np.isclose(maxlag, 2.0, atol=0.5)

    def test_mutualinfo_functype(self):
        t = np.linspace(-30, 30, 1001)
        # mutual info has a nonzero baseline
        corr = 0.5 + 0.5 * np.exp(-0.5 * (t / 2.0) ** 2)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, ps, pe = tide_fit.simfuncpeakfit(
            corr, t, peakfittype="gauss", functype="mutualinfo"
        )
        # just check it runs without error
        assert isinstance(maskval, np.uint16)


# ========================= _maxindex_noedge (internal) =========================


class TestMaxindexNoedgeInternal:
    def test_basic(self):
        corrfunc = np.array([0.1, 0.5, 0.8, 0.3, 0.2])
        corrtimeaxis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        idx, flip = tide_fit._maxindex_noedge(corrfunc, corrtimeaxis)
        assert idx == 2
        assert flip == 1.0

    def test_bipolar(self):
        corrfunc = np.array([0.1, 0.2, -0.9, 0.3, 0.2])
        corrtimeaxis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        idx, flip = tide_fit._maxindex_noedge(corrfunc, corrtimeaxis, bipolar=True)
        assert idx == 2
        assert flip == -1.0


# ========================= Territory decomp & stats =========================


class TestTerritorydecomp:
    def test_basic_3d(self):
        np.random.seed(42)
        inputmap = np.random.rand(5, 5, 5)
        template = np.random.rand(5, 5, 5)
        atlas = np.ones((5, 5, 5), dtype=int)
        fitmap, coffs, r2s = tide_fit.territorydecomp(inputmap, template, atlas)
        assert fitmap.shape == inputmap.shape
        assert coffs.shape[0] == 1  # nummaps
        assert r2s.shape[0] == 1

    def test_multiple_territories(self):
        np.random.seed(42)
        inputmap = np.random.rand(6, 6, 6)
        template = np.random.rand(6, 6, 6)
        atlas = np.zeros((6, 6, 6), dtype=int)
        atlas[:3, :, :] = 1
        atlas[3:, :, :] = 2
        fitmap, coffs, r2s = tide_fit.territorydecomp(inputmap, template, atlas)
        assert coffs.shape[1] == 2  # 2 territories

    def test_fitorder_zero(self):
        np.random.seed(42)
        inputmap = np.random.rand(5, 5, 5)
        template = np.random.rand(5, 5, 5)
        atlas = np.ones((5, 5, 5), dtype=int)
        fitmap, coffs, r2s = tide_fit.territorydecomp(inputmap, template, atlas, fitorder=0)
        assert fitmap.shape == inputmap.shape


class TestTerritorystats:
    def test_basic_3d(self):
        np.random.seed(42)
        inputmap = np.random.rand(5, 5, 5)
        atlas = np.ones((5, 5, 5), dtype=int)
        result = tide_fit.territorystats(inputmap, atlas)
        assert len(result) == 9
        statsmap, means, stds, medians, mads, variances, skewnesses, kurtoses, entropies = result
        assert statsmap.shape == inputmap.shape
        assert means.shape == (1, 1)

    def test_multiple_territories(self):
        np.random.seed(42)
        inputmap = np.random.rand(6, 6, 6)
        atlas = np.zeros((6, 6, 6), dtype=int)
        atlas[:3, :, :] = 1
        atlas[3:, :, :] = 2
        result = tide_fit.territorystats(inputmap, atlas)
        means = result[1]
        assert means.shape == (1, 2)

    def test_with_mask(self):
        np.random.seed(42)
        inputmap = np.random.rand(5, 5, 5)
        atlas = np.ones((5, 5, 5), dtype=int)
        mask = np.ones((5, 5, 5))
        mask[0, :, :] = 0
        result = tide_fit.territorystats(inputmap, atlas, inputmask=mask)
        assert len(result) == 9


# ========================= Run the tests =========================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
