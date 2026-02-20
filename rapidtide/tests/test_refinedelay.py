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

import rapidtide.core.delay.refinedelay as core_refinedelay
import rapidtide.core.delay.refinedelay as tide_refinedelay


class DummyLagTCGenerator:
    def info(self, prefix=""):
        return None

    def yfromx(self, x):
        return np.sin(x)


def smooth_tests(debug=False):
    if debug:
        print("smooth_tests")
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = tide_refinedelay.smooth(x, 3)
    assert y.shape == x.shape
    np.testing.assert_allclose(y, np.convolve(x, np.ones(3) / 3.0, mode="same"))


def trainratiotooffset_and_ratiotodelay_tests(debug=False):
    if debug:
        print("trainratiotooffset_and_ratiotodelay_tests")

    def fake_getderivratios(
        fmri_data_valid,
        validvoxels,
        initial_fmri_x,
        lagtimes,
        fitmask,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm,
        fitcoeff,
        movingsignal,
        lagtc,
        filtereddata,
        LGR,
        TimingLGR,
        optiondict,
        regressderivs=1,
        timemask=None,
        starttr=None,
        endtr=None,
        debug=False,
    ):
        # Monotonic decreasing response in lag index. trainratiotooffset reverses
        # this axis internally, so it becomes monotonic increasing for spline fit.
        npts = fmri_data_valid.shape[0]
        out = np.linspace(1.0, -1.0, npts, endpoint=True)
        # derive the training offset encoded in initial_fmri_x from DummyLagTCGenerator
        if initial_fmri_x.shape[0] > 1:
            thisoffset = np.arcsin(np.clip(initial_fmri_x[0], -1.0, 1.0))
        else:
            thisoffset = 0.0
        return out + thisoffset, rvalue

    def fake_writebidstsv(*args, **kwargs):
        return None

    saved_getderivratios = tide_refinedelay.getderivratios
    saved_writebidstsv = tide_refinedelay.tide_io.writebidstsv
    try:
        tide_refinedelay.getderivratios = fake_getderivratios
        tide_refinedelay.tide_io.writebidstsv = fake_writebidstsv
        tide_refinedelay.trainratiotooffset(
            lagtcgenerator=DummyLagTCGenerator(),
            timeaxis=np.linspace(0.0, 20.0, 100, endpoint=False),
            outputname="dummy",
            outputlevel="full",
            trainlagmin=-0.5,
            trainlagmax=0.5,
            trainlagstep=0.5,
            mindelay=-2.0,
            maxdelay=2.0,
            numpoints=81,
            smoothpts=3,
            edgepad=3,
            regressderivs=1,
            verbose=False,
            debug=False,
        )
    finally:
        tide_refinedelay.getderivratios = saved_getderivratios
        tide_refinedelay.tide_io.writebidstsv = saved_writebidstsv

    assert len(tide_refinedelay.ratiotooffsetfunc) == 3
    assert len(tide_refinedelay.funcoffsets) == 3
    assert tide_refinedelay.maplimits[0] < tide_refinedelay.maplimits[1]

    center_ratio = 0.5 * (tide_refinedelay.maplimits[0] + tide_refinedelay.maplimits[1])
    delay, closest = tide_refinedelay.ratiotodelay(center_ratio, offset=0.49, debug=False)
    assert np.isfinite(delay)
    assert np.isfinite(closest)
    assert np.fabs(closest - 0.5) < 1e-6

    below, c0 = tide_refinedelay.ratiotodelay(tide_refinedelay.maplimits[0] - 10.0, offset=0.0)
    above, c1 = tide_refinedelay.ratiotodelay(tide_refinedelay.maplimits[1] + 10.0, offset=0.0)
    assert np.isfinite(below)
    assert np.isfinite(above)
    assert np.isfinite(c0)
    assert np.isfinite(c1)


def coffstodelay_tests(debug=False):
    if debug:
        print("coffstodelay_tests")

    saved_poly = tide_refinedelay.poly.Polynomial
    try:

        class DummyPolyNone:
            def __init__(self, coeffs, domain=None):
                pass

            def roots(self):
                return None

        tide_refinedelay.poly.Polynomial = DummyPolyNone
        d0 = tide_refinedelay.coffstodelay(
            np.array([-1.0]), mindelay=-3.0, maxdelay=3.0, debug=False
        )
        assert np.fabs(d0) < 1e-10

        class DummyPolySingle:
            def __init__(self, coeffs, domain=None):
                pass

            def roots(self):
                return np.array([1.25 + 0.0j])

        tide_refinedelay.poly.Polynomial = DummyPolySingle
        d1 = tide_refinedelay.coffstodelay(
            np.array([-1.0]), mindelay=-3.0, maxdelay=3.0, debug=False
        )
        assert np.fabs(d1 - 1.25) < 1e-10

        class DummyPolyMany:
            def __init__(self, coeffs, domain=None):
                pass

            def roots(self):
                return np.array([2.5 + 0.0j, -0.5 + 0.0j, 10.0 + 0.0j, 0.0 + 1.0j])

        tide_refinedelay.poly.Polynomial = DummyPolyMany
        d2 = tide_refinedelay.coffstodelay(
            np.array([0.0, 1.0]), mindelay=-3.0, maxdelay=3.0, debug=False
        )
        assert np.fabs(d2 + 0.5) < 1e-10
    finally:
        tide_refinedelay.poly.Polynomial = saved_poly


def getderivratios_tests(debug=False):
    if debug:
        print("getderivratios_tests")

    captured = {"timemask": None, "xshape": None}

    def fake_regressfrommaps(
        fmri_data_valid,
        validvoxels,
        initial_fmri_x,
        lagtimes,
        fitmask,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm,
        fitcoeff,
        movingsignal,
        lagtc,
        filtereddata,
        LGR,
        TimingLGR,
        regressfiltthreshval,
        saveminimumsLFOfiltfiles,
        nprocs_makelaggedtcs=1,
        nprocs_regressionfilt=1,
        regressderivs=1,
        chunksize=1000,
        showprogressbar=False,
        alwaysmultiproc=False,
        coefficientsonly=True,
        timemask=None,
        debug=False,
    ):
        captured["timemask"] = timemask
        captured["xshape"] = initial_fmri_x.shape
        fitcoeff[:, 0] = 2.0
        for i in range(regressderivs):
            fitcoeff[:, i + 1] = 2.0 * (i + 1)
        return fitcoeff.shape[0], None, None

    saved_regress = tide_refinedelay.tide_regressfrommaps.regressfrommaps
    tide_refinedelay.tide_regressfrommaps.regressfrommaps = fake_regressfrommaps
    try:
        nvox = 5
        nt = 20
        fmri = np.random.RandomState(1).randn(nvox, nt)
        valid = np.arange(nvox, dtype=int)
        xaxis = np.linspace(0.0, 10.0, nt, endpoint=False)
        lagtimes = np.zeros(nvox)
        fitmask = np.ones(nvox)
        sLFOfitmean = np.zeros(nvox)
        rvalue = np.zeros(nvox)
        r2value = np.zeros(nvox)
        fitNorm = np.zeros((nvox, 3))
        fitcoeff = np.zeros((nvox, 3))
        movingsignal = np.zeros_like(fmri)
        lagtc = np.zeros_like(fmri)
        filtereddata = np.zeros_like(fmri)
        optiondict = {
            "regressfiltthreshval": 0.0,
            "nprocs_makelaggedtcs": 1,
            "nprocs_regressionfilt": 1,
            "mp_chunksize": 1000,
            "showprogressbar": False,
            "alwaysmultiproc": False,
        }

        ratios1, rvals1 = tide_refinedelay.getderivratios(
            fmri,
            valid,
            xaxis,
            lagtimes,
            fitmask,
            DummyLagTCGenerator(),
            "glm",
            "dummy",
            1.0,
            sLFOfitmean,
            rvalue,
            r2value,
            fitNorm[:, :2],
            fitcoeff[:, :2],
            movingsignal,
            lagtc,
            filtereddata,
            None,
            None,
            optiondict,
            regressderivs=1,
            timemask=np.ones(nt),
            starttr=2,
            endtr=12,
            debug=False,
        )
        assert ratios1.shape == (nvox,)
        np.testing.assert_allclose(ratios1, np.ones(nvox))
        assert rvals1.shape == (nvox,)
        assert captured["xshape"] == (10,)
        assert captured["timemask"].shape == (10,)

        fitcoeff2 = np.zeros((nvox, 3))
        ratios2, _ = tide_refinedelay.getderivratios(
            fmri,
            valid,
            xaxis,
            lagtimes,
            fitmask,
            DummyLagTCGenerator(),
            "glm",
            "dummy",
            1.0,
            sLFOfitmean,
            rvalue.copy(),
            r2value.copy(),
            fitNorm,
            fitcoeff2,
            movingsignal,
            lagtc,
            filtereddata,
            None,
            None,
            optiondict,
            regressderivs=2,
            timemask=None,
            starttr=None,
            endtr=None,
            debug=False,
        )
        assert ratios2.shape == (2, nvox)
        np.testing.assert_allclose(ratios2[0], np.ones(nvox))
        np.testing.assert_allclose(ratios2[1], 2.0 * np.ones(nvox))
    finally:
        tide_refinedelay.tide_regressfrommaps.regressfrommaps = saved_regress


def filterderivratios_tests(debug=False):
    if debug:
        print("filterderivratios_tests")

    ratios_1d = np.array([1.0, 1.0, 1.0, 10.0], dtype=float)
    validvox = np.array([0, 1, 2, 3], dtype=int)
    nativespaceshape_nifti = (2, 2, 1)

    # non-nifti branch
    ratios_text = ratios_1d.reshape((-1, 1))
    med_non, filt_non, mad_non = tide_refinedelay.filterderivratios(
        ratios_text,
        (4, 1),
        validvox,
        thedims=(1.0, 1.0, 1.0),
        patchthresh=3.0,
        filetype="text",
        verbose=False,
        debug=False,
    )
    np.testing.assert_allclose(med_non, ratios_text)
    np.testing.assert_allclose(filt_non, ratios_text)
    assert np.isfinite(mad_non)

    saved_ssmooth = tide_refinedelay.tide_filt.ssmooth

    def fake_ssmooth(dx, dy, dz, sigma, arr):
        return arr

    tide_refinedelay.tide_filt.ssmooth = fake_ssmooth
    try:
        med_nii, filt_nii, mad_nii = tide_refinedelay.filterderivratios(
            ratios_1d,
            nativespaceshape_nifti,
            validvox,
            thedims=(1.0, 1.0, 1.0),
            patchthresh=1.0,
            gausssigma=1.0,
            filetype="nifti",
            verbose=False,
            debug=False,
        )
    finally:
        tide_refinedelay.tide_filt.ssmooth = saved_ssmooth

    assert med_nii.shape == ratios_1d.shape
    assert filt_nii.shape == ratios_1d.shape
    assert np.isfinite(mad_nii)
    # the outlier should be pulled toward neighborhood median
    assert filt_nii[-1] < ratios_1d[-1]


def test_refinedelay(debug=False, local=False):
    np.random.seed(12345)
    smooth_tests(debug=debug)
    trainratiotooffset_and_ratiotodelay_tests(debug=debug)
    coffstodelay_tests(debug=debug)
    getderivratios_tests(debug=debug)
    filterderivratios_tests(debug=debug)
    assert callable(core_refinedelay.filterderivratios)


if __name__ == "__main__":
    test_refinedelay(debug=True, local=True)
