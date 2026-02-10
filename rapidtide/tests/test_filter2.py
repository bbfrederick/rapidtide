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

from rapidtide.filter import (
    NoncausalFilter,
    arb_pass,
    blackmanharris,
    csdfilter,
    dobpfftfilt,
    dobpfiltfilt,
    dobptransfuncfilt,
    dobptrapfftfilt,
    dohpfftfilt,
    dohpfiltfilt,
    dohptransfuncfilt,
    dohptrapfftfilt,
    dolpfftfilt,
    dolpfiltfilt,
    dolptransfuncfilt,
    dolptrapfftfilt,
    getfilterbandfreqs,
    gethptransfunc,
    getlpfftfunc,
    getlptransfunc,
    getlptrapfftfunc,
    hamming,
    hann,
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

# ==================== Helpers ====================

FS = 100.0  # sample rate in Hz
NPOINTS = 1000
PADLEN = 20


def _make_test_signal():
    """Generate a signal with known frequency content: 5 Hz + 25 Hz."""
    t = np.arange(NPOINTS) / FS
    return np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * 25.0 * t)


def _make_broadband_signal(seed=42):
    """Generate broadband noise for filter testing."""
    rng = np.random.RandomState(seed)
    return rng.randn(NPOINTS)


# ==================== padvec tests ====================


def padvec_reflect(debug=False):
    """Test padvec with reflect padding."""
    if debug:
        print("padvec_reflect")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = padvec(data, padlen=3, padtype="reflect")
    assert len(result) == len(data) + 2 * 3
    # Middle portion should be the original data
    np.testing.assert_array_equal(result[3:-3], data)


def padvec_zero(debug=False):
    """Test padvec with zero padding."""
    if debug:
        print("padvec_zero")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = padvec(data, padlen=2, padtype="zero")
    assert len(result) == len(data) + 4
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[-1] == 0.0
    assert result[-2] == 0.0
    np.testing.assert_array_equal(result[2:-2], data)


def padvec_cyclic(debug=False):
    """Test padvec with cyclic padding."""
    if debug:
        print("padvec_cyclic")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = padvec(data, padlen=2, padtype="cyclic")
    assert len(result) == len(data) + 4
    # Start should be last 2 elements of data
    assert result[0] == data[-2]
    assert result[1] == data[-1]
    # End should be first 2 elements of data
    assert result[-2] == data[0]
    assert result[-1] == data[1]


def padvec_constant(debug=False):
    """Test padvec with constant padding."""
    if debug:
        print("padvec_constant")
    data = np.array([3.0, 2.0, 1.0, 4.0, 5.0])
    result = padvec(data, padlen=2, padtype="constant")
    assert result[0] == data[0]
    assert result[1] == data[0]
    assert result[-1] == data[-1]
    assert result[-2] == data[-1]


def padvec_constant_plus(debug=False):
    """Test padvec with constant+ padding uses average."""
    if debug:
        print("padvec_constant_plus")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = padvec(data, padlen=3, avlen=3, padtype="constant+")
    # Start should be mean of first 3 elements = 2.0
    assert result[0] == pytest.approx(2.0)
    # End should be mean of last 3 elements = 9.0
    assert result[-1] == pytest.approx(9.0)


def padvec_zero_padlen(debug=False):
    """Test padvec with padlen=0 returns original data."""
    if debug:
        print("padvec_zero_padlen")
    data = np.array([1.0, 2.0, 3.0])
    result = padvec(data, padlen=0)
    np.testing.assert_array_equal(result, data)


def padvec_padlen_too_large(debug=False):
    """Test padvec raises RuntimeError when padlen > data length."""
    if debug:
        print("padvec_padlen_too_large")
    data = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError):
        padvec(data, padlen=10)


def padvec_invalid_padtype(debug=False):
    """Test padvec raises ValueError for invalid padtype."""
    if debug:
        print("padvec_invalid_padtype")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        padvec(data, padlen=2, padtype="invalid")


# ==================== unpadvec tests ====================


def unpadvec_basic(debug=False):
    """Test unpadvec removes padding from both ends."""
    if debug:
        print("unpadvec_basic")
    data = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
    result = unpadvec(data, padlen=2)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def unpadvec_zero_padlen(debug=False):
    """Test unpadvec with padlen=0 returns original."""
    if debug:
        print("unpadvec_zero_padlen")
    data = np.array([1.0, 2.0, 3.0])
    result = unpadvec(data, padlen=0)
    np.testing.assert_array_equal(result, data)


def unpadvec_roundtrip(debug=False):
    """Test padvec/unpadvec roundtrip preserves data."""
    if debug:
        print("unpadvec_roundtrip")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    for padtype in ["reflect", "zero", "cyclic", "constant"]:
        padded = padvec(data, padlen=3, padtype=padtype)
        recovered = unpadvec(padded, padlen=3)
        np.testing.assert_array_equal(recovered, data)


# ==================== ssmooth tests ====================


def ssmooth_basic(debug=False):
    """Test spatial smoothing on 3D data."""
    if debug:
        print("ssmooth_basic")
    data = np.zeros((10, 10, 10))
    data[5, 5, 5] = 1.0
    result = ssmooth(1.0, 1.0, 1.0, 2.0, data)
    assert result.shape == (10, 10, 10)
    # Peak should be reduced by smoothing
    assert result[5, 5, 5] < 1.0
    # Neighbors should be non-zero
    assert result[5, 5, 4] > 0.0


def ssmooth_anisotropic(debug=False):
    """Test spatial smoothing with different voxel sizes."""
    if debug:
        print("ssmooth_anisotropic")
    data = np.zeros((10, 10, 10))
    data[5, 5, 5] = 1.0
    result = ssmooth(1.0, 2.0, 3.0, 2.0, data)
    assert result.shape == (10, 10, 10)
    # Smoothing should be different in different directions
    assert result[5, 5, 5] < 1.0


# ==================== Butterworth filter tests ====================


def dolpfiltfilt_basic(debug=False):
    """Test Butterworth lowpass filter attenuates high frequencies."""
    if debug:
        print("dolpfiltfilt_basic")
    sig = _make_test_signal()
    # LP at 10 Hz should keep 5 Hz, remove 25 Hz
    filtered = dolpfiltfilt(FS, 10.0, sig, order=4, padlen=PADLEN)
    assert len(filtered) == len(sig)
    # Check power at 5 Hz is preserved (mostly)
    fft_orig = np.abs(np.fft.rfft(sig))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freq_bins = np.fft.rfftfreq(len(sig), 1.0 / FS)
    idx_5 = np.argmin(np.abs(freq_bins - 5.0))
    idx_25 = np.argmin(np.abs(freq_bins - 25.0))
    # 25 Hz should be attenuated
    assert fft_filt[idx_25] < fft_orig[idx_25] * 0.1


def dohpfiltfilt_basic(debug=False):
    """Test Butterworth highpass filter attenuates low frequencies."""
    if debug:
        print("dohpfiltfilt_basic")
    sig = _make_test_signal()
    # HP at 15 Hz should keep 25 Hz, remove 5 Hz
    filtered = dohpfiltfilt(FS, 15.0, sig, order=4, padlen=PADLEN)
    assert len(filtered) == len(sig)
    fft_orig = np.abs(np.fft.rfft(sig))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freq_bins = np.fft.rfftfreq(len(sig), 1.0 / FS)
    idx_5 = np.argmin(np.abs(freq_bins - 5.0))
    # 5 Hz should be attenuated
    assert fft_filt[idx_5] < fft_orig[idx_5] * 0.1


def dobpfiltfilt_basic(debug=False):
    """Test Butterworth bandpass filter."""
    if debug:
        print("dobpfiltfilt_basic")
    sig = _make_test_signal()
    # BP 3-8 Hz should keep 5 Hz, remove 25 Hz
    filtered = dobpfiltfilt(FS, 3.0, 8.0, sig, order=4, padlen=PADLEN)
    assert len(filtered) == len(sig)
    fft_orig = np.abs(np.fft.rfft(sig))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freq_bins = np.fft.rfftfreq(len(sig), 1.0 / FS)
    idx_25 = np.argmin(np.abs(freq_bins - 25.0))
    assert fft_filt[idx_25] < fft_orig[idx_25] * 0.1


# ==================== FFT brickwall filter tests ====================


def getlpfftfunc_basic(debug=False):
    """Test brickwall LP transfer function shape."""
    if debug:
        print("getlpfftfunc_basic")
    data = np.zeros(100)
    tf = getlpfftfunc(100.0, 20.0, data)
    assert len(tf) == 100
    # Passband should be 1.0
    assert tf[0] == 1.0
    # Far stopband should be 0.0
    assert tf[len(tf) // 2] == 0.0


def dolpfftfilt_basic(debug=False):
    """Test FFT brickwall lowpass filter."""
    if debug:
        print("dolpfftfilt_basic")
    sig = _make_test_signal()
    filtered = dolpfftfilt(FS, 10.0, sig, padlen=PADLEN)
    assert len(filtered) == len(sig)
    fft_filt = np.abs(np.fft.rfft(filtered))
    freq_bins = np.fft.rfftfreq(len(sig), 1.0 / FS)
    idx_25 = np.argmin(np.abs(freq_bins - 25.0))
    assert fft_filt[idx_25] < 1.0  # 25 Hz should be attenuated


def dohpfftfilt_basic(debug=False):
    """Test FFT brickwall highpass filter."""
    if debug:
        print("dohpfftfilt_basic")
    sig = _make_test_signal()
    filtered = dohpfftfilt(FS, 15.0, sig, padlen=PADLEN)
    assert len(filtered) == len(sig)
    fft_filt = np.abs(np.fft.rfft(filtered))
    freq_bins = np.fft.rfftfreq(len(sig), 1.0 / FS)
    idx_5 = np.argmin(np.abs(freq_bins - 5.0))
    assert fft_filt[idx_5] < 1.0  # 5 Hz should be attenuated


def dobpfftfilt_basic(debug=False):
    """Test FFT brickwall bandpass filter."""
    if debug:
        print("dobpfftfilt_basic")
    sig = _make_test_signal()
    filtered = dobpfftfilt(FS, 3.0, 8.0, sig, padlen=PADLEN)
    assert len(filtered) == len(sig)


# ==================== Trapezoidal FFT filter tests ====================


def getlptrapfftfunc_basic(debug=False):
    """Test trapezoidal LP transfer function has smooth transition."""
    if debug:
        print("getlptrapfftfunc_basic")
    data = np.zeros(200)
    tf = getlptrapfftfunc(100.0, 20.0, 25.0, data)
    assert len(tf) == 200
    assert tf[0] == 1.0
    # In the transition band, values should be between 0 and 1
    passbin = int(20.0 / 100.0 * 200)
    cutoffbin = int(25.0 / 100.0 * 200)
    midbin = (passbin + cutoffbin) // 2
    assert 0.0 < tf[midbin] < 1.0


def dolptrapfftfilt_basic(debug=False):
    """Test trapezoidal lowpass FFT filter."""
    if debug:
        print("dolptrapfftfilt_basic")
    sig = _make_test_signal()
    filtered = dolptrapfftfilt(FS, 10.0, 12.0, sig, padlen=PADLEN)
    assert len(filtered) == len(sig)


def dohptrapfftfilt_basic(debug=False):
    """Test trapezoidal highpass FFT filter."""
    if debug:
        print("dohptrapfftfilt_basic")
    sig = _make_test_signal()
    filtered = dohptrapfftfilt(FS, 13.0, 15.0, sig, padlen=PADLEN)
    assert len(filtered) == len(sig)


def dobptrapfftfilt_basic(debug=False):
    """Test trapezoidal bandpass FFT filter."""
    if debug:
        print("dobptrapfftfilt_basic")
    sig = _make_test_signal()
    filtered = dobptrapfftfilt(FS, 2.0, 3.0, 8.0, 10.0, sig, padlen=PADLEN)
    assert len(filtered) == len(sig)


# ==================== Transfer function filter tests ====================


def transferfuncfilt_basic(debug=False):
    """Test direct transfer function filtering."""
    if debug:
        print("transferfuncfilt_basic")
    sig = _make_test_signal()
    # Make a simple lowpass transfer function (all pass)
    tf = np.ones(len(sig))
    result = transferfuncfilt(sig, tf)
    assert len(result) == len(sig)
    np.testing.assert_allclose(result, sig, atol=1e-10)


def transferfuncfilt_zeros(debug=False):
    """Test transfer function with zeros kills signal."""
    if debug:
        print("transferfuncfilt_zeros")
    sig = _make_test_signal()
    tf = np.zeros(len(sig))
    result = transferfuncfilt(sig, tf)
    np.testing.assert_allclose(result, np.zeros(len(sig)), atol=1e-10)


# ==================== getlptransfunc / gethptransfunc tests ====================


def getlptransfunc_brickwall(debug=False):
    """Test LP transfer function brickwall type."""
    if debug:
        print("getlptransfunc_brickwall")
    data = np.zeros(200)
    tf = getlptransfunc(100.0, data, upperpass=20.0, type="brickwall")
    assert len(tf) == 200
    assert tf[0] == 1.0
    # Well past cutoff should be zero
    assert tf[100] == 0.0


def getlptransfunc_gaussian(debug=False):
    """Test LP transfer function Gaussian type."""
    if debug:
        print("getlptransfunc_gaussian")
    data = np.zeros(200)
    tf = getlptransfunc(100.0, data, upperpass=20.0, type="gaussian")
    assert len(tf) == 200
    # Should be ~1 at DC
    assert tf[0] > 0.9
    # Should decay smoothly
    assert tf[10] > tf[20]


def getlptransfunc_trapezoidal(debug=False):
    """Test LP transfer function trapezoidal type."""
    if debug:
        print("getlptransfunc_trapezoidal")
    data = np.zeros(200)
    tf = getlptransfunc(100.0, data, upperpass=20.0, upperstop=25.0, type="trapezoidal")
    assert len(tf) == 200
    assert tf[0] == 1.0


def gethptransfunc_brickwall(debug=False):
    """Test HP transfer function brickwall type."""
    if debug:
        print("gethptransfunc_brickwall")
    data = np.zeros(200)
    tf = gethptransfunc(100.0, data, lowerpass=20.0, type="brickwall")
    assert len(tf) == 200
    # DC should be 0 (high-pass removes DC)
    assert tf[0] == 0.0


def gethptransfunc_trapezoidal(debug=False):
    """Test HP transfer function trapezoidal type."""
    if debug:
        print("gethptransfunc_trapezoidal")
    data = np.zeros(200)
    tf = gethptransfunc(100.0, data, lowerstop=15.0, lowerpass=20.0, type="trapezoidal")
    assert len(tf) == 200
    assert tf[0] == 0.0


# ==================== dolptransfuncfilt / dohptransfuncfilt / dobptransfuncfilt ====================


def dolptransfuncfilt_basic(debug=False):
    """Test LP transfer function filter."""
    if debug:
        print("dolptransfuncfilt_basic")
    sig = _make_test_signal()
    filtered = dolptransfuncfilt(FS, sig, upperpass=10.0, type="brickwall", padlen=PADLEN)
    assert len(filtered) == len(sig)


def dolptransfuncfilt_gaussian(debug=False):
    """Test LP transfer function filter with Gaussian."""
    if debug:
        print("dolptransfuncfilt_gaussian")
    sig = _make_test_signal()
    filtered = dolptransfuncfilt(FS, sig, upperpass=10.0, type="gaussian", padlen=PADLEN)
    assert len(filtered) == len(sig)


def dohptransfuncfilt_basic(debug=False):
    """Test HP transfer function filter."""
    if debug:
        print("dohptransfuncfilt_basic")
    sig = _make_test_signal()
    filtered = dohptransfuncfilt(FS, sig, lowerpass=15.0, type="brickwall", padlen=PADLEN)
    assert len(filtered) == len(sig)


def dobptransfuncfilt_basic(debug=False):
    """Test BP transfer function filter."""
    if debug:
        print("dobptransfuncfilt_basic")
    sig = _make_test_signal()
    filtered = dobptransfuncfilt(
        FS, sig, lowerpass=3.0, upperpass=8.0, type="brickwall", padlen=PADLEN
    )
    assert len(filtered) == len(sig)


def dobptransfuncfilt_trapezoidal(debug=False):
    """Test BP transfer function filter with trapezoidal."""
    if debug:
        print("dobptransfuncfilt_trapezoidal")
    sig = _make_test_signal()
    filtered = dobptransfuncfilt(
        FS,
        sig,
        lowerpass=3.0,
        upperpass=8.0,
        lowerstop=2.0,
        upperstop=10.0,
        type="trapezoidal",
        padlen=PADLEN,
    )
    assert len(filtered) == len(sig)


# ==================== wiener_deconvolution tests ====================


def wiener_deconvolution_basic(debug=False):
    """Test Wiener deconvolution returns correct length."""
    if debug:
        print("wiener_deconvolution_basic")
    sig = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    kernel = np.array([1.0, 0.5, 0.25])
    result = wiener_deconvolution(sig, kernel, lambd=0.1)
    assert len(result) == len(sig)


def wiener_deconvolution_identity(debug=False):
    """Test deconvolution with identity kernel recovers signal approximately."""
    if debug:
        print("wiener_deconvolution_identity")
    sig = np.zeros(64)
    sig[32] = 1.0  # impulse
    kernel = np.array([1.0])
    result = wiener_deconvolution(sig, kernel, lambd=1e-10)
    assert len(result) == len(sig)


# ==================== pspec / spectralflatness / spectrum tests ====================


def pspec_basic(debug=False):
    """Test power spectrum computation."""
    if debug:
        print("pspec_basic")
    sig = _make_test_signal()
    ps = pspec(sig)
    assert len(ps) == len(sig)
    assert np.all(np.real(ps) >= 0)


def spectralflatness_white_noise(debug=False):
    """Test spectral flatness of white noise is close to 1."""
    if debug:
        print("spectralflatness_white_noise")
    rng = np.random.RandomState(42)
    noise = np.abs(rng.randn(1000)) + 0.01  # Must be positive
    sf = spectralflatness(noise)
    # White noise should have high flatness
    assert 0.0 < sf <= 1.0


def spectralflatness_tonal(debug=False):
    """Test spectral flatness of tonal signal is low."""
    if debug:
        print("spectralflatness_tonal")
    spectrum_arr = np.zeros(1000)
    spectrum_arr[50] = 100.0
    spectrum_arr += 0.001  # Avoid log(0)
    sf = spectralflatness(spectrum_arr)
    assert sf < 0.5


def spectrum_power_mode(debug=False):
    """Test spectrum function in power mode."""
    if debug:
        print("spectrum_power_mode")
    sig = _make_test_signal()
    freqs, vals = spectrum(sig, Fs=FS, mode="power", trim=True)
    assert len(freqs) == len(vals)
    assert freqs[0] == 0.0
    assert freqs[-1] < FS / 2.0


def spectrum_mag_mode(debug=False):
    """Test spectrum in magnitude mode."""
    if debug:
        print("spectrum_mag_mode")
    sig = _make_test_signal()
    freqs, vals = spectrum(sig, Fs=FS, mode="mag", trim=True)
    assert np.all(np.real(vals) >= 0)


def spectrum_complex_mode(debug=False):
    """Test spectrum in complex mode."""
    if debug:
        print("spectrum_complex_mode")
    sig = _make_test_signal()
    freqs, vals = spectrum(sig, Fs=FS, mode="complex", trim=True)
    assert np.iscomplexobj(vals)


def spectrum_phase_mode(debug=False):
    """Test spectrum in phase mode."""
    if debug:
        print("spectrum_phase_mode")
    sig = _make_test_signal()
    freqs, vals = spectrum(sig, Fs=FS, mode="phase", trim=True)
    assert len(freqs) == len(vals)


def spectrum_real_mode(debug=False):
    """Test spectrum in real mode."""
    if debug:
        print("spectrum_real_mode")
    sig = _make_test_signal()
    freqs, vals = spectrum(sig, Fs=FS, mode="real", trim=True)
    assert len(freqs) == len(vals)


def spectrum_imag_mode(debug=False):
    """Test spectrum in imag mode."""
    if debug:
        print("spectrum_imag_mode")
    sig = _make_test_signal()
    freqs, vals = spectrum(sig, Fs=FS, mode="imag", trim=True)
    assert len(freqs) == len(vals)


def spectrum_untrimmed(debug=False):
    """Test spectrum with trim=False returns full FFT."""
    if debug:
        print("spectrum_untrimmed")
    sig = _make_test_signal()
    freqs, vals = spectrum(sig, Fs=FS, mode="power", trim=False)
    assert len(freqs) == len(sig)


def spectrum_illegal_mode(debug=False):
    """Test spectrum with illegal mode raises RuntimeError."""
    if debug:
        print("spectrum_illegal_mode")
    sig = _make_test_signal()
    with pytest.raises(RuntimeError):
        spectrum(sig, Fs=FS, mode="invalid_mode")


# ==================== NoncausalFilter tests ====================


def ncfilter_init_none(debug=False):
    """Test NoncausalFilter with type None."""
    if debug:
        print("ncfilter_init_none")
    filt = NoncausalFilter(filtertype="None")
    assert filt.gettype() == "None"
    data = _make_test_signal()
    result = filt.apply(FS, data)
    np.testing.assert_array_equal(result, data)


def ncfilter_init_lfo(debug=False):
    """Test NoncausalFilter with LFO type has correct frequency range."""
    if debug:
        print("ncfilter_init_lfo")
    filt = NoncausalFilter(filtertype="lfo")
    assert filt.gettype() == "lfo"
    ls, lp, up, us = filt.getfreqs()
    assert lp == pytest.approx(0.01)
    assert up == pytest.approx(0.15)
    assert ls < lp
    assert us > up


def ncfilter_settype(debug=False):
    """Test settype changes the filter type."""
    if debug:
        print("ncfilter_settype")
    filt = NoncausalFilter(filtertype="lfo")
    filt.settype("cardiac")
    assert filt.gettype() == "cardiac"
    ls, lp, up, us = filt.getfreqs()
    assert lp == pytest.approx(0.66)


def ncfilter_setfreqs(debug=False):
    """Test setfreqs with arbitrary frequencies."""
    if debug:
        print("ncfilter_setfreqs")
    filt = NoncausalFilter(filtertype="arb")
    filt.setfreqs(0.01, 0.02, 0.1, 0.12)
    ls, lp, up, us = filt.getfreqs()
    assert ls == pytest.approx(0.01)
    assert lp == pytest.approx(0.02)
    assert up == pytest.approx(0.1)
    assert us == pytest.approx(0.12)


def ncfilter_gettype(debug=False):
    """Test gettype returns correct type."""
    if debug:
        print("ncfilter_gettype")
    filt = NoncausalFilter(filtertype="resp")
    assert filt.gettype() == "resp"


def ncfilter_setbutterorder(debug=False):
    """Test setbutterorder changes order."""
    if debug:
        print("ncfilter_setbutterorder")
    filt = NoncausalFilter()
    filt.setbutterorder(8)
    assert filt.butterworthorder == 8
    filt.setbutterorder()
    assert filt.butterworthorder == 3


def ncfilter_setdebug(debug=False):
    """Test setdebug changes debug flag."""
    if debug:
        print("ncfilter_setdebug")
    filt = NoncausalFilter()
    filt.setdebug(True)
    assert filt.debug is True
    filt.setdebug(False)
    assert filt.debug is False


def ncfilter_padtime(debug=False):
    """Test set/get padtime."""
    if debug:
        print("ncfilter_padtime")
    filt = NoncausalFilter()
    filt.setpadtime(60.0)
    assert filt.getpadtime() == pytest.approx(60.0)


def ncfilter_padtype(debug=False):
    """Test set/get padtype."""
    if debug:
        print("ncfilter_padtype")
    filt = NoncausalFilter()
    filt.setpadtype("zero")
    assert filt.getpadtype() == "zero"


def ncfilter_settransferfunc(debug=False):
    """Test settransferfunc."""
    if debug:
        print("ncfilter_settransferfunc")
    filt = NoncausalFilter()
    filt.settransferfunc("butterworth")
    assert filt.transferfunc == "butterworth"


def ncfilter_apply_lfo(debug=False):
    """Test applying LFO filter to a long broadband signal."""
    if debug:
        print("ncfilter_apply_lfo")
    # Need a long signal for LFO band (0.01-0.15 Hz)
    fs = 10.0
    npoints = 5000  # 500 seconds
    sig = np.random.RandomState(42).randn(npoints)
    filt = NoncausalFilter(filtertype="lfo", padtime=30.0)
    result = filt.apply(fs, sig)
    assert len(result) == npoints


def ncfilter_apply_arb_bandpass(debug=False):
    """Test applying arbitrary bandpass filter."""
    if debug:
        print("ncfilter_apply_arb_bandpass")
    filt = NoncausalFilter(
        filtertype="arb",
        initlowerstop=2.0,
        initlowerpass=3.0,
        initupperpass=8.0,
        initupperstop=10.0,
        padtime=5.0,
    )
    sig = _make_test_signal()
    result = filt.apply(FS, sig)
    assert len(result) == len(sig)


def ncfilter_apply_arb_stop(debug=False):
    """Test applying arbitrary stopband filter."""
    if debug:
        print("ncfilter_apply_arb_stop")
    filt = NoncausalFilter(
        filtertype="arb_stop",
        initlowerstop=4.0,
        initlowerpass=4.5,
        initupperpass=5.5,
        initupperstop=6.0,
        padtime=5.0,
    )
    sig = _make_test_signal()
    result = filt.apply(FS, sig)
    assert len(result) == len(sig)
    # 5 Hz component should be attenuated
    fft_orig = np.abs(np.fft.rfft(sig))
    fft_filt = np.abs(np.fft.rfft(result))
    freq_bins = np.fft.rfftfreq(len(sig), 1.0 / FS)
    idx_5 = np.argmin(np.abs(freq_bins - 5.0))
    assert fft_filt[idx_5] < fft_orig[idx_5] * 0.5


def ncfilter_apply_butterworth(debug=False):
    """Test applying filter with butterworth transfer function."""
    if debug:
        print("ncfilter_apply_butterworth")
    filt = NoncausalFilter(
        filtertype="arb",
        transferfunc="butterworth",
        initlowerstop=2.0,
        initlowerpass=3.0,
        initupperpass=8.0,
        initupperstop=10.0,
        padtime=5.0,
    )
    sig = _make_test_signal()
    result = filt.apply(FS, sig)
    assert len(result) == len(sig)


def ncfilter_apply_ringstop(debug=False):
    """Test ringstop filter type."""
    if debug:
        print("ncfilter_apply_ringstop")
    filt = NoncausalFilter(filtertype="ringstop", padtime=5.0)
    sig = _make_test_signal()
    result = filt.apply(FS, sig)
    assert len(result) == len(sig)


def ncfilter_allphysio_types(debug=False):
    """Test that all physiological filter types can be initialized."""
    if debug:
        print("ncfilter_allphysio_types")
    types = [
        "vlf", "lfo", "lfo_legacy", "lfo_tight", "resp", "cardiac",
        "hrv_ulf", "hrv_vlf", "hrv_lf", "hrv_hf", "hrv_vhf",
        "vlf_stop", "lfo_stop", "lfo_legacy_stop", "lfo_tight_stop",
        "resp_stop", "cardiac_stop",
        "hrv_ulf_stop", "hrv_vlf_stop", "hrv_lf_stop", "hrv_hf_stop", "hrv_vhf_stop",
    ]
    for ftype in types:
        filt = NoncausalFilter(filtertype=ftype)
        assert filt.gettype() == ftype
        ls, lp, up, us = filt.getfreqs()
        assert ls <= lp
        if up > 0:
            assert up <= us


def ncfilter_correctfreq(debug=False):
    """Test correctfreq adjusts out-of-range frequencies."""
    if debug:
        print("ncfilter_correctfreq")
    # Create a filter with frequencies that would exceed Nyquist for a low Fs
    filt = NoncausalFilter(
        filtertype="arb",
        initlowerstop=0.0,
        initlowerpass=0.0,
        initupperpass=0.4,
        initupperstop=0.5,
        correctfreq=True,
    )
    # Apply at Fs=1.0 Hz → Nyquist = 0.5 Hz, upperpass=0.4 < 0.5 so OK
    sig = np.random.RandomState(42).randn(200)
    result = filt.apply(1.0, sig)
    assert len(result) == len(sig)


# ==================== setnotchfilter / harmonicnotchfilter tests ====================


def setnotchfilter_basic(debug=False):
    """Test setnotchfilter configures filter correctly."""
    if debug:
        print("setnotchfilter_basic")
    filt = NoncausalFilter()
    setnotchfilter(filt, 50.0, notchwidth=2.0)
    assert filt.gettype() == "arb_stop"
    ls, lp, up, us = filt.getfreqs()
    assert lp == pytest.approx(49.0)
    assert up == pytest.approx(51.0)


def harmonicnotchfilter_basic(debug=False):
    """Test harmonic notch filter removes fundamental and harmonics."""
    if debug:
        print("harmonicnotchfilter_basic")
    npts = 10000  # Long signal to support default padtime=30s
    t = np.arange(npts) / FS
    # Signal with 10 Hz fundamental and 20 Hz harmonic + broadband noise
    sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    sig += np.random.RandomState(42).randn(npts) * 0.3
    result = harmonicnotchfilter(sig, FS, 10.0, notchpct=5.0)
    assert len(result) == len(sig)
    # 10 Hz should be attenuated
    fft_orig = np.abs(np.fft.rfft(sig))
    fft_filt = np.abs(np.fft.rfft(result))
    freq_bins = np.fft.rfftfreq(len(sig), 1.0 / FS)
    idx_10 = np.argmin(np.abs(freq_bins - 10.0))
    assert fft_filt[idx_10] < fft_orig[idx_10] * 0.5


def harmonicnotchfilter_none_pct(debug=False):
    """Test harmonicnotchfilter with None notchpct returns unchanged data."""
    if debug:
        print("harmonicnotchfilter_none_pct")
    sig = _make_test_signal()
    result = harmonicnotchfilter(sig, FS, 10.0, notchpct=None)
    # With notchpct=None, no filtering is done
    np.testing.assert_array_equal(result, sig)


# ==================== savgolsmooth tests ====================


def savgolsmooth_basic(debug=False):
    """Test Savitzky-Golay smoothing."""
    if debug:
        print("savgolsmooth_basic")
    sig = _make_test_signal()
    noise = np.random.RandomState(42).randn(len(sig)) * 0.5
    noisy = sig + noise
    smoothed = savgolsmooth(noisy, smoothlen=21, polyorder=3)
    assert len(smoothed) == len(noisy)
    # Smoothed should be closer to original than noisy
    assert np.std(smoothed - sig) < np.std(noisy - sig)


# ==================== csdfilter tests ====================


def csdfilter_basic(debug=False):
    """Test cross spectral density filter."""
    if debug:
        print("csdfilter_basic")
    rng = np.random.RandomState(42)
    common = np.sin(2 * np.pi * 5 * np.arange(NPOINTS) / FS)
    obs = common + rng.randn(NPOINTS) * 0.3
    result = csdfilter(obs, common, padlen=PADLEN)
    assert len(result) == len(obs)


# ==================== arb_pass tests ====================


def arb_pass_lowpass_trapezoidal(debug=False):
    """Test arb_pass as lowpass with trapezoidal filter."""
    if debug:
        print("arb_pass_lowpass_trapezoidal")
    sig = _make_test_signal()
    result = arb_pass(
        FS, sig, 0.0, 0.0, 10.0, 12.0,
        transferfunc="trapezoidal", padlen=PADLEN,
    )
    assert len(result) == len(sig)


def arb_pass_highpass_trapezoidal(debug=False):
    """Test arb_pass as highpass with trapezoidal filter."""
    if debug:
        print("arb_pass_highpass_trapezoidal")
    sig = _make_test_signal()
    result = arb_pass(
        FS, sig, 13.0, 15.0, FS / 2.0, FS / 2.0,
        transferfunc="trapezoidal", padlen=PADLEN,
    )
    assert len(result) == len(sig)


def arb_pass_bandpass_trapezoidal(debug=False):
    """Test arb_pass as bandpass with trapezoidal filter."""
    if debug:
        print("arb_pass_bandpass_trapezoidal")
    sig = _make_test_signal()
    result = arb_pass(
        FS, sig, 2.0, 3.0, 8.0, 10.0,
        transferfunc="trapezoidal", padlen=PADLEN,
    )
    assert len(result) == len(sig)


def arb_pass_bandpass_butterworth(debug=False):
    """Test arb_pass as bandpass with butterworth filter."""
    if debug:
        print("arb_pass_bandpass_butterworth")
    sig = _make_test_signal()
    result = arb_pass(
        FS, sig, 2.0, 3.0, 8.0, 10.0,
        transferfunc="butterworth", padlen=PADLEN,
    )
    assert len(result) == len(sig)


def arb_pass_bandpass_brickwall(debug=False):
    """Test arb_pass as bandpass with brickwall filter."""
    if debug:
        print("arb_pass_bandpass_brickwall")
    sig = _make_test_signal()
    result = arb_pass(
        FS, sig, 2.0, 3.0, 8.0, 10.0,
        transferfunc="brickwall", padlen=PADLEN,
    )
    assert len(result) == len(sig)


def arb_pass_lowpass_butterworth(debug=False):
    """Test arb_pass as lowpass with butterworth."""
    if debug:
        print("arb_pass_lowpass_butterworth")
    sig = _make_test_signal()
    result = arb_pass(
        FS, sig, 0.0, 0.0, 10.0, 12.0,
        transferfunc="butterworth", padlen=PADLEN,
    )
    assert len(result) == len(sig)


def arb_pass_highpass_butterworth(debug=False):
    """Test arb_pass as highpass with butterworth."""
    if debug:
        print("arb_pass_highpass_butterworth")
    sig = _make_test_signal()
    result = arb_pass(
        FS, sig, 13.0, 15.0, FS / 2.0, FS / 2.0,
        transferfunc="butterworth", padlen=PADLEN,
    )
    assert len(result) == len(sig)


# ==================== getfilterbandfreqs tests ====================


def getfilterbandfreqs_all_bands(debug=False):
    """Test getfilterbandfreqs for all known bands."""
    if debug:
        print("getfilterbandfreqs_all_bands")
    bands = [
        "vlf", "lfo", "lfo_legacy", "lfo_tight", "resp", "cardiac",
        "hrv_ulf", "hrv_vlf", "hrv_lf", "hrv_hf", "hrv_vhf",
    ]
    for band in bands:
        lp, up, ls, us = getfilterbandfreqs(band)
        assert ls <= lp
        if up > 0:
            assert up <= us
        if debug:
            print(f"  {band}: ls={ls}, lp={lp}, up={up}, us={us}")


def getfilterbandfreqs_asrange(debug=False):
    """Test getfilterbandfreqs with asrange=True returns string."""
    if debug:
        print("getfilterbandfreqs_asrange")
    result = getfilterbandfreqs("lfo", asrange=True)
    assert isinstance(result, str)
    assert "Hz" in result
    assert "0.01" in result
    assert "0.15" in result


def getfilterbandfreqs_transitionfrac(debug=False):
    """Test getfilterbandfreqs with custom transition fraction."""
    if debug:
        print("getfilterbandfreqs_transitionfrac")
    lp1, up1, ls1, us1 = getfilterbandfreqs("lfo", transitionfrac=0.05)
    lp2, up2, ls2, us2 = getfilterbandfreqs("lfo", transitionfrac=0.20)
    # Wider transition should have more extreme stop frequencies
    assert ls2 < ls1
    assert us2 > us1


# ==================== FFT helper function tests ====================


def polarfft_roundtrip(debug=False):
    """Test polarfft -> ifftfrompolar roundtrip recovers original signal."""
    if debug:
        print("polarfft_roundtrip")
    sig = _make_test_signal()
    mag, phase = polarfft(sig)
    assert len(mag) == len(sig)
    assert len(phase) == len(sig)
    recovered = ifftfrompolar(mag, phase)
    np.testing.assert_allclose(recovered, sig, atol=1e-10)


def polarfft_magnitude(debug=False):
    """Test polarfft magnitude is non-negative."""
    if debug:
        print("polarfft_magnitude")
    sig = _make_test_signal()
    mag, phase = polarfft(sig)
    assert np.all(mag >= 0)


def ifftfrompolar_basic(debug=False):
    """Test ifftfrompolar with known values."""
    if debug:
        print("ifftfrompolar_basic")
    r = np.array([4.0, 0.0, 0.0, 0.0])
    theta = np.zeros(4)
    result = ifftfrompolar(r, theta)
    # IFFT of [4, 0, 0, 0] = [1, 1, 1, 1]
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0, 1.0], atol=1e-10)


# ==================== Window function tests ====================


def blackmanharris_basic(debug=False):
    """Test Blackman-Harris window properties."""
    if debug:
        print("blackmanharris_basic")
    w = blackmanharris(64)
    assert len(w) == 64
    # First element should be near zero
    assert w[0] < 0.01
    # Max should be near the middle
    assert np.argmax(w) == len(w) // 2 or np.argmax(w) == len(w) // 2 - 1


def blackmanharris_cache(debug=False):
    """Test that blackmanharris caches results."""
    if debug:
        print("blackmanharris_cache")
    w1 = blackmanharris(32)
    w2 = blackmanharris(32)
    assert w1 is w2  # Same object from cache


def hann_basic(debug=False):
    """Test Hann window properties."""
    if debug:
        print("hann_basic")
    w = hann(64)
    assert len(w) == 64
    assert w[0] == pytest.approx(0.0)
    # Max should be near the middle
    mid = len(w) // 2
    assert w[mid] > 0.9


def hann_cache(debug=False):
    """Test that hann caches results."""
    if debug:
        print("hann_cache")
    w1 = hann(48)
    w2 = hann(48)
    assert w1 is w2


def hamming_basic(debug=False):
    """Test Hamming window properties."""
    if debug:
        print("hamming_basic")
    w = hamming(64)
    assert len(w) == 64
    # First element should be ~0.08
    assert 0.05 < w[0] < 0.15


def hamming_cache(debug=False):
    """Test that hamming caches results."""
    if debug:
        print("hamming_cache")
    w1 = hamming(56)
    w2 = hamming(56)
    assert w1 is w2


def rect_basic(debug=False):
    """Test rectangular window."""
    if debug:
        print("rect_basic")
    w = rect(10, 4)
    assert len(w) == 10
    # Center 4 elements should be 1.0
    assert w[4] == 1.0
    assert w[5] == 1.0
    # Edges should be 0.0
    assert w[0] == 0.0
    assert w[-1] == 0.0


def mRect_basic(debug=False):
    """Test modified rectangular window."""
    if debug:
        print("mRect_basic")
    w = mRect(100)
    assert len(w) == 100
    # Max should be 1.0 after normalization
    assert np.max(w) == pytest.approx(1.0)


def mRect_custom_params(debug=False):
    """Test mRect with custom parameters."""
    if debug:
        print("mRect_custom_params")
    w = mRect(100, alpha=0.3, omegac=0.05, phi=np.pi / 4)
    assert len(w) == 100
    assert np.max(w) == pytest.approx(1.0)


def windowfunction_hamming(debug=False):
    """Test windowfunction with hamming type."""
    if debug:
        print("windowfunction_hamming")
    w = windowfunction(64, type="hamming")
    w_ref = hamming(64)
    np.testing.assert_array_equal(w, w_ref)


def windowfunction_hann(debug=False):
    """Test windowfunction with hann type."""
    if debug:
        print("windowfunction_hann")
    w = windowfunction(64, type="hann")
    w_ref = hann(64)
    np.testing.assert_array_equal(w, w_ref)


def windowfunction_blackmanharris(debug=False):
    """Test windowfunction with blackmanharris type."""
    if debug:
        print("windowfunction_blackmanharris")
    w = windowfunction(64, type="blackmanharris")
    w_ref = blackmanharris(64)
    np.testing.assert_array_equal(w, w_ref)


def windowfunction_none(debug=False):
    """Test windowfunction with None type returns ones."""
    if debug:
        print("windowfunction_none")
    w = windowfunction(64, type="None")
    np.testing.assert_array_equal(w, np.ones(64))


# ==================== Main test function ====================


def test_filter2(debug=False):
    # padvec tests
    if debug:
        print("Running padvec tests")
    padvec_reflect(debug=debug)
    padvec_zero(debug=debug)
    padvec_cyclic(debug=debug)
    padvec_constant(debug=debug)
    padvec_constant_plus(debug=debug)
    padvec_zero_padlen(debug=debug)
    padvec_padlen_too_large(debug=debug)
    padvec_invalid_padtype(debug=debug)

    # unpadvec tests
    if debug:
        print("Running unpadvec tests")
    unpadvec_basic(debug=debug)
    unpadvec_zero_padlen(debug=debug)
    unpadvec_roundtrip(debug=debug)

    # ssmooth tests
    if debug:
        print("Running ssmooth tests")
    ssmooth_basic(debug=debug)
    ssmooth_anisotropic(debug=debug)

    # Butterworth filter tests
    if debug:
        print("Running Butterworth filter tests")
    dolpfiltfilt_basic(debug=debug)
    dohpfiltfilt_basic(debug=debug)
    dobpfiltfilt_basic(debug=debug)

    # FFT brickwall filter tests
    if debug:
        print("Running FFT brickwall filter tests")
    getlpfftfunc_basic(debug=debug)
    dolpfftfilt_basic(debug=debug)
    dohpfftfilt_basic(debug=debug)
    dobpfftfilt_basic(debug=debug)

    # Trapezoidal FFT filter tests
    if debug:
        print("Running trapezoidal FFT filter tests")
    getlptrapfftfunc_basic(debug=debug)
    dolptrapfftfilt_basic(debug=debug)
    dohptrapfftfilt_basic(debug=debug)
    dobptrapfftfilt_basic(debug=debug)

    # Transfer function filter tests
    if debug:
        print("Running transfer function filter tests")
    transferfuncfilt_basic(debug=debug)
    transferfuncfilt_zeros(debug=debug)

    # getlptransfunc / gethptransfunc tests
    if debug:
        print("Running LP/HP transfer function tests")
    getlptransfunc_brickwall(debug=debug)
    getlptransfunc_gaussian(debug=debug)
    getlptransfunc_trapezoidal(debug=debug)
    gethptransfunc_brickwall(debug=debug)
    gethptransfunc_trapezoidal(debug=debug)

    # dolptransfuncfilt / dohptransfuncfilt / dobptransfuncfilt tests
    if debug:
        print("Running LP/HP/BP transfer function filter tests")
    dolptransfuncfilt_basic(debug=debug)
    dolptransfuncfilt_gaussian(debug=debug)
    dohptransfuncfilt_basic(debug=debug)
    dobptransfuncfilt_basic(debug=debug)
    dobptransfuncfilt_trapezoidal(debug=debug)

    # wiener deconvolution tests
    if debug:
        print("Running Wiener deconvolution tests")
    wiener_deconvolution_basic(debug=debug)
    wiener_deconvolution_identity(debug=debug)

    # pspec / spectralflatness / spectrum tests
    if debug:
        print("Running spectral analysis tests")
    pspec_basic(debug=debug)
    spectralflatness_white_noise(debug=debug)
    spectralflatness_tonal(debug=debug)
    spectrum_power_mode(debug=debug)
    spectrum_mag_mode(debug=debug)
    spectrum_complex_mode(debug=debug)
    spectrum_phase_mode(debug=debug)
    spectrum_real_mode(debug=debug)
    spectrum_imag_mode(debug=debug)
    spectrum_untrimmed(debug=debug)
    spectrum_illegal_mode(debug=debug)

    # NoncausalFilter tests
    if debug:
        print("Running NoncausalFilter tests")
    ncfilter_init_none(debug=debug)
    ncfilter_init_lfo(debug=debug)
    ncfilter_settype(debug=debug)
    ncfilter_setfreqs(debug=debug)
    ncfilter_gettype(debug=debug)
    ncfilter_setbutterorder(debug=debug)
    ncfilter_setdebug(debug=debug)
    ncfilter_padtime(debug=debug)
    ncfilter_padtype(debug=debug)
    ncfilter_settransferfunc(debug=debug)
    ncfilter_apply_lfo(debug=debug)
    ncfilter_apply_arb_bandpass(debug=debug)
    ncfilter_apply_arb_stop(debug=debug)
    ncfilter_apply_butterworth(debug=debug)
    ncfilter_apply_ringstop(debug=debug)
    ncfilter_allphysio_types(debug=debug)
    ncfilter_correctfreq(debug=debug)

    # setnotchfilter / harmonicnotchfilter tests
    if debug:
        print("Running notch filter tests")
    setnotchfilter_basic(debug=debug)
    harmonicnotchfilter_basic(debug=debug)
    harmonicnotchfilter_none_pct(debug=debug)

    # savgolsmooth tests
    if debug:
        print("Running savgolsmooth tests")
    savgolsmooth_basic(debug=debug)

    # csdfilter tests
    if debug:
        print("Running csdfilter tests")
    csdfilter_basic(debug=debug)

    # arb_pass tests
    if debug:
        print("Running arb_pass tests")
    arb_pass_lowpass_trapezoidal(debug=debug)
    arb_pass_highpass_trapezoidal(debug=debug)
    arb_pass_bandpass_trapezoidal(debug=debug)
    arb_pass_bandpass_butterworth(debug=debug)
    arb_pass_bandpass_brickwall(debug=debug)
    arb_pass_lowpass_butterworth(debug=debug)
    arb_pass_highpass_butterworth(debug=debug)

    # getfilterbandfreqs tests
    if debug:
        print("Running getfilterbandfreqs tests")
    getfilterbandfreqs_all_bands(debug=debug)
    getfilterbandfreqs_asrange(debug=debug)
    getfilterbandfreqs_transitionfrac(debug=debug)

    # FFT helper tests
    if debug:
        print("Running FFT helper tests")
    polarfft_roundtrip(debug=debug)
    polarfft_magnitude(debug=debug)
    ifftfrompolar_basic(debug=debug)

    # Window function tests
    if debug:
        print("Running window function tests")
    blackmanharris_basic(debug=debug)
    blackmanharris_cache(debug=debug)
    hann_basic(debug=debug)
    hann_cache(debug=debug)
    hamming_basic(debug=debug)
    hamming_cache(debug=debug)
    rect_basic(debug=debug)
    mRect_basic(debug=debug)
    mRect_custom_params(debug=debug)
    windowfunction_hamming(debug=debug)
    windowfunction_hann(debug=debug)
    windowfunction_blackmanharris(debug=debug)
    windowfunction_none(debug=debug)


if __name__ == "__main__":
    test_filter2(debug=True)
