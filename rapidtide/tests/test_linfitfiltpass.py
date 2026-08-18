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
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.linfitfiltpass as tide_linfitfiltpass
from rapidtide.tests.utils import mse


def gen2d(xsize=150, xcycles=11, tsize=200, tcycles=13, mean=10.0):
    thearray = np.zeros((xsize, tsize), dtype=np.float64)
    xwaves = np.zeros((xsize, tsize), dtype=np.float64)
    twaves = np.zeros((xsize, tsize), dtype=np.float64)
    xmax = 2.0 * np.pi * xcycles
    tmax = 2.0 * np.pi * tcycles
    xfreq = xmax / xsize
    tfreq = tmax / tsize
    for i in range(tsize):
        thearray[:, i] = np.sin(np.linspace(0.0, xmax, xsize, endpoint=False))
        xwaves[:, i] = np.sin(np.linspace(0.0, xmax, xsize, endpoint=False))
    for i in range(xsize):
        thearray[i, :] *= np.sin(np.linspace(0.0, tmax, tsize, endpoint=False))
        twaves[i, :] = np.sin(np.linspace(0.0, tmax, tsize, endpoint=False))
    return thearray, xwaves, twaves


def test_linfitfiltpass(debug=True, displayplots=False):
    np.random.seed(12345)
    xsize = 150
    xcycles = 7
    tsize = 200
    tcycles = 23
    mean = 100.0
    noiselevel = 5.0

    targetarray, xwaveforms, twaveforms = gen2d(
        xsize=xsize, xcycles=xcycles, tsize=tsize, tcycles=tcycles
    )
    if debug:
        print(f"{twaveforms.shape=}")
        print(f"{xwaveforms.shape=}")
    testarray = targetarray + np.random.random((xsize, tsize)) + mean
    if displayplots:
        plt.figure()
        plt.imshow(targetarray)
        plt.show()

    filtereddata = 0.0 * testarray
    datatoremove = 0.0 * testarray
    threshval = 0.01
    meanvals_t = np.zeros(tsize, dtype=np.float64)
    rvals_t = np.zeros(tsize, dtype=np.float64)
    r2vals_t = np.zeros(tsize, dtype=np.float64)
    fitcoffs_t = np.zeros((xsize, tsize), dtype=np.float64)
    fitNorm_t = np.zeros((xsize, tsize), dtype=np.float64)

    meanvals_x = np.zeros(xsize, dtype=np.float64)
    rvals_x = np.zeros(xsize, dtype=np.float64)
    r2vals_x = np.zeros(xsize, dtype=np.float64)
    fitcoffs_x = np.zeros((xsize, tsize), dtype=np.float64)
    fitNorm_x = np.zeros((xsize, tsize), dtype=np.float64)

    for confoundregress in [True, False]:
        if confoundregress:
            twaveformrange = np.transpose(twaveforms[:6, :])
            xwaveformrange = xwaveforms[:, :6]
            print(f"{twaveformrange.shape=} - {xwaveformrange.shape=}")
        else:
            twaveformrange = twaveforms
            xwaveformrange = xwaveforms
        for procbyvoxel in [True, False]:
            if procbyvoxel:
                waveforms = twaveformrange
                meanvals = meanvals_x
                rvals = rvals_x
                r2vals = r2vals_x
                fitcoffs = fitcoffs_x
                fitNorm = fitNorm_x
                direction = "space"
            else:
                waveforms = xwaveformrange
                meanvals = meanvals_t
                rvals = rvals_t
                r2vals = r2vals_t
                fitcoffs = fitcoffs_t
                fitNorm = fitNorm_t
                direction = "time"
            for nprocs in [1, 2]:
                if nprocs == 1:
                    procstring = "single"
                else:
                    procstring = "multi"
                for thisthreshval in [threshval, None]:
                    if thisthreshval is None:
                        maskstatus = "no mask"
                    else:
                        maskstatus = f"threshold={threshval}"

                    if debug:
                        print(
                            f"confoundregress={confoundregress}, proc by {direction}, {procstring} proc, {maskstatus}"
                        )
                    tide_linfitfiltpass.linfitfiltpass(
                        xsize,
                        testarray,
                        thisthreshval,
                        waveforms,
                        meanvals,
                        rvals,
                        r2vals,
                        fitcoffs,
                        fitNorm,
                        datatoremove,
                        filtereddata,
                        showprogressbar=False,
                        procbyvoxel=procbyvoxel,
                        nprocs=nprocs,
                        confoundregress=confoundregress,
                    )
                    if displayplots:
                        plt.figure()
                        plt.imshow(datatoremove)
                        plt.show()
                        plt.imshow(filtereddata)
                        plt.show()
                    if debug:
                        print(f"\tMSE: {mse(datatoremove, targetarray)}\n")
                    if not confoundregress:
                        assert mse(datatoremove, targetarray) < 1e-3


def test_linfitfiltpass_coefficientsonly_constantevs_paths(debug=False):
    """Exercise coefficientsonly=True with constantevs both False and True."""
    if debug:
        print("linfitfiltpass_coefficientsonly_constantevs_paths")

    numvox = 3
    tpts = 5
    fmri_data = np.arange(numvox * tpts, dtype=np.float64).reshape((numvox, tpts))
    theevs = np.vstack(
        [
            np.linspace(1.0, 2.0, tpts),
            np.linspace(11.0, 12.0, tpts),
            np.linspace(21.0, 22.0, tpts),
        ]
    )
    threshval = None
    sentinel = -999.0

    for constantevs in [False, True]:
        meanvalue = np.zeros(numvox, dtype=np.float64)
        rvalue = np.zeros(numvox, dtype=np.float64)
        r2value = np.zeros(numvox, dtype=np.float64)
        fitcoeff = np.zeros(numvox, dtype=np.float64)
        fitNorm = np.zeros(numvox, dtype=np.float64)
        datatoremove = np.full((numvox, tpts), sentinel, dtype=np.float64)
        filtereddata = np.full((numvox, tpts), sentinel, dtype=np.float64)
        received_evs = []

        def _mock_proc(item, evs, data, rt_floattype=np.dtype(np.float64)):
            received_evs.append(np.array(evs, copy=True))
            return (
                item,  # index
                1.0 + item,  # mean
                0.5,  # r
                0.25,  # r2
                7.0 + item,  # coeff
                0.7,  # fitNorm
                np.ones_like(data),  # datatoremove (ignored in coefficientsonly path)
                np.ones_like(data) * 2.0,  # filtereddata (ignored in coefficientsonly path)
            )

        with patch("rapidtide.linfitfiltpass._procOneRegressionFitItem", side_effect=_mock_proc):
            items = tide_linfitfiltpass.linfitfiltpass(
                numvox,
                fmri_data,
                threshval,
                theevs,
                meanvalue,
                rvalue,
                r2value,
                fitcoeff,
                fitNorm,
                datatoremove,
                filtereddata,
                nprocs=1,
                procbyvoxel=True,
                coefficientsonly=True,
                constantevs=constantevs,
                showprogressbar=False,
            )

        assert items == numvox
        np.testing.assert_allclose(meanvalue, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(rvalue, [0.5, 0.5, 0.5])
        np.testing.assert_allclose(r2value, [0.25, 0.25, 0.25])
        np.testing.assert_allclose(fitcoeff, [7.0, 8.0, 9.0])
        np.testing.assert_allclose(fitNorm, [0.7, 0.7, 0.7])

        # In coefficientsonly mode, these arrays should not be modified.
        assert np.all(datatoremove == sentinel)
        assert np.all(filtereddata == sentinel)

        if not constantevs:
            for vox in range(numvox):
                np.testing.assert_allclose(received_evs[vox], theevs[vox, :])
        else:
            for vox in range(numvox):
                np.testing.assert_allclose(received_evs[vox], theevs)


# ==================== _procOneRegressionFitItem tests ====================


def _fitdata(nvox=4, tpts=24, seed=0):
    """Build a shared regressor and per-voxel data that is a clean multiple of it.

    Parameters
    ----------
    nvox : int, optional
        Number of voxels.  Default is 4.
    tpts : int, optional
        Number of timepoints.  Default is 24.
    seed : int, optional
        Seed for the random regressor.  Default is 0.

    Returns
    -------
    tuple of (ndarray, ndarray)
        The (tpts,) regressor and the (nvox, tpts) data, where voxel i is (i + 1) times
        the regressor plus a small amount of noise.
    """
    rng = np.random.RandomState(seed)
    theev = rng.randn(tpts)
    thedata = np.outer(np.arange(1.0, nvox + 1), theev) + 0.001 * rng.randn(nvox, tpts)
    return theev, thedata


def test_procOneRegressionFitItem_univariate(debug=False):
    """Test the 1D regressor path recovers a known slope and intercept."""
    if debug:
        print("procOneRegressionFitItem_univariate")
    theev = np.linspace(0.0, 1.0, 20)
    thedata = 3.0 * theev + 7.0
    vox, intercept, rval, r2, fitcoeff, fitNorm, datatoremove, residual = (
        tide_linfitfiltpass._procOneRegressionFitItem(5, theev, thedata)
    )
    assert vox == 5
    assert abs(intercept - 7.0) < 1e-10
    assert abs(fitcoeff - 3.0) < 1e-10
    assert abs(r2 - 1.0) < 1e-10
    assert abs(rval - 1.0) < 1e-10
    # fitNorm is the coefficient expressed relative to the intercept
    assert abs(fitNorm - 3.0 / 7.0) < 1e-10
    np.testing.assert_allclose(datatoremove, 3.0 * theev, atol=1e-10)
    np.testing.assert_allclose(residual, thedata - 3.0 * theev, atol=1e-10)


def test_procOneRegressionFitItem_negative_coefficient(debug=False):
    """Test a negative fit coefficient makes the returned r value negative."""
    if debug:
        print("procOneRegressionFitItem_negative_coefficient")
    theev = np.linspace(0.0, 1.0, 20)
    result = tide_linfitfiltpass._procOneRegressionFitItem(0, theev, -3.0 * theev + 5.0)
    assert result[4] < 0.0
    # r is the signed square root of R2, so a negative slope gives a negative r
    assert abs(result[2] + 1.0) < 1e-10
    assert abs(result[3] - 1.0) < 1e-10


def test_procOneRegressionFitItem_multivariate(debug=False):
    """Test the 2D regressor path recovers both coefficients."""
    if debug:
        print("procOneRegressionFitItem_multivariate")
    npts = 40
    rng = np.random.RandomState(3)
    ev1, ev2 = rng.randn(npts), rng.randn(npts)
    theevs = np.vstack([ev1, ev2]).T
    thedata = 2.0 * ev1 - 5.0 * ev2 + 11.0
    result = tide_linfitfiltpass._procOneRegressionFitItem(2, theevs, thedata)
    assert result[0] == 2
    assert abs(result[1] - 11.0) < 1e-8
    fitcoeffs = np.atleast_2d(result[4])
    assert fitcoeffs.shape == (1, 2)
    np.testing.assert_allclose(fitcoeffs[0], [2.0, -5.0], atol=1e-8)
    assert abs(result[3] - 1.0) < 1e-10
    np.testing.assert_allclose(result[6], 2.0 * ev1 - 5.0 * ev2, atol=1e-8)


def test_procOneRegressionFitItem_multivariate_negative_first_coeff(debug=False):
    """Test the coefficient sign is taken from the first coefficient in the 2D path."""
    if debug:
        print("procOneRegressionFitItem_multivariate_negative_first_coeff")
    npts = 40
    rng = np.random.RandomState(4)
    ev1, ev2 = rng.randn(npts), rng.randn(npts)
    theevs = np.vstack([ev1, ev2]).T
    result = tide_linfitfiltpass._procOneRegressionFitItem(0, theevs, -4.0 * ev1 + ev2)
    assert result[2] < 0.0


def test_procOneRegressionFitItem_zero_coefficients(debug=False):
    """Test constant data drives the coefficients and R2 to zero in both paths.

    Constant data is the reachable route into the "all coefficients are zero" guards; the
    regression returns the constant as the intercept and nothing for the slopes.
    """
    if debug:
        print("procOneRegressionFitItem_zero_coefficients")
    npts = 12
    ev1 = np.linspace(0.0, 1.0, npts)
    theevs2d = np.vstack([ev1, ev1**2]).T
    constdata = np.full(npts, 7.0)

    for theevs, label in [(ev1, "1D"), (theevs2d, "2D")]:
        result = tide_linfitfiltpass._procOneRegressionFitItem(0, theevs, constdata)
        assert abs(result[1] - 7.0) < 1e-10, f"{label} intercept"
        assert result[3] == 0.0, f"{label} R2 should be forced to zero"
        assert result[2] == 0.0, f"{label} r should be zero"
        np.testing.assert_allclose(np.atleast_1d(result[4]).ravel(), 0.0, atol=1e-12)
        # nothing is removed, so the residual is the original data
        np.testing.assert_allclose(result[7], constdata, atol=1e-10)


# ==================== linfitfiltpass masking tests ====================


def test_linfitfiltpass_threshval_mean_mask(debug=False):
    """Test a threshold masks out voxels by mean when the data mean exceeds its stdev."""
    if debug:
        print("linfitfiltpass_threshval_mean_mask")
    tpts = 20
    theev, _ = _fitdata(tpts=tpts)
    # two bright voxels and two dim ones; means are large compared to the stdevs
    thedata = np.vstack(
        [
            100.0 + 0.1 * theev,
            100.0 + 0.2 * theev,
            1.0 + 0.1 * theev,
            1.0 + 0.2 * theev,
        ]
    )
    nvox = thedata.shape[0]
    r2value = np.zeros(nvox)
    filtereddata = np.zeros((nvox, tpts))
    sentinel = -42.0
    filtereddata[:] = sentinel
    items = tide_linfitfiltpass.linfitfiltpass(
        nvox,
        thedata,
        50.0,
        theev,
        None,
        None,
        r2value,
        None,
        None,
        None,
        filtereddata,
        nprocs=1,
        confoundregress=True,
        showprogressbar=False,
        verbose=False,
    )
    assert items == 2, f"expected the two bright voxels only, got {items}"
    assert np.all(filtereddata[2:, :] == sentinel), "dim voxels should be untouched"
    assert np.all(filtereddata[:2, :] != sentinel), "bright voxels should be filtered"


def test_linfitfiltpass_threshval_std_mask(debug=False):
    """Test the threshold falls back to the standard deviation when it exceeds the mean.

    ``linfitfiltpass`` picks between a mean-based and a stdev-based mask by comparing the
    two averaged over the image, so zero-mean data with large swings takes the stdev branch.
    """
    if debug:
        print("linfitfiltpass_threshval_std_mask")
    tpts = 20
    theev, _ = _fitdata(tpts=tpts)
    thedata = np.vstack([50.0 * theev, 40.0 * theev, 0.01 * theev, 0.02 * theev])
    thedata -= np.mean(thedata, axis=1)[:, None]
    nvox = thedata.shape[0]
    assert np.mean(np.std(thedata, axis=1)) > np.mean(np.mean(thedata, axis=1))
    r2value = np.zeros(nvox)
    filtereddata = np.zeros((nvox, tpts))
    items = tide_linfitfiltpass.linfitfiltpass(
        nvox,
        thedata,
        1.0,
        theev,
        None,
        None,
        r2value,
        None,
        None,
        None,
        filtereddata,
        nprocs=1,
        confoundregress=True,
        showprogressbar=False,
        verbose=False,
    )
    assert items == 2, f"expected the two high variance voxels only, got {items}"


def test_linfitfiltpass_explicit_validmask(debug=False):
    """Test an explicit validmask overrides the threshold based mask entirely."""
    if debug:
        print("linfitfiltpass_explicit_validmask")
    tpts = 20
    theev, thedata = _fitdata(nvox=4, tpts=tpts)
    nvox = thedata.shape[0]
    r2value = np.zeros(nvox)
    filtereddata = np.zeros((nvox, tpts))
    validmask = np.array([0, 1, 1, 0])
    items = tide_linfitfiltpass.linfitfiltpass(
        nvox,
        thedata,
        None,
        theev,
        None,
        None,
        r2value,
        None,
        None,
        None,
        filtereddata,
        nprocs=1,
        validmask=validmask,
        confoundregress=True,
        showprogressbar=False,
        verbose=False,
    )
    assert items == 2
    assert r2value[0] == 0.0 and r2value[3] == 0.0
    assert r2value[1] != 0.0 and r2value[2] != 0.0


# ==================== constantevs regression tests ====================


def _run_constantevs(nprocs, procbyvoxel, nvox=4, tpts=24):
    """Run linfitfiltpass with a single shared regressor and return the output arrays.

    Parameters
    ----------
    nprocs : int
        Number of processes; >1 takes the multiprocessing path.
    procbyvoxel : bool
        Whether to process by voxel or by timepoint.
    nvox : int, optional
        Number of voxels.  Default is 4.
    tpts : int, optional
        Number of timepoints.  Default is 24.

    Returns
    -------
    dict
        The populated output arrays, keyed by name.
    """
    theev, voxdata = _fitdata(nvox=nvox, tpts=tpts)
    thedata = voxdata if procbyvoxel else np.ascontiguousarray(voxdata.T)
    arrays = {
        "meanvalue": np.zeros(nvox),
        "rvalue": np.zeros(nvox),
        "r2value": np.zeros(nvox),
        "fitcoeff": np.zeros(nvox),
        "fitNorm": np.zeros(nvox),
        "datatoremove": np.zeros_like(thedata),
        "filtereddata": np.zeros_like(thedata),
    }
    tide_linfitfiltpass.linfitfiltpass(
        nvox,
        thedata,
        None,
        theev,
        arrays["meanvalue"],
        arrays["rvalue"],
        arrays["r2value"],
        arrays["fitcoeff"],
        arrays["fitNorm"],
        arrays["datatoremove"],
        arrays["filtereddata"],
        nprocs=nprocs,
        constantevs=True,
        procbyvoxel=procbyvoxel,
        showprogressbar=False,
        verbose=False,
    )
    return arrays


def test_linfitfiltpass_constantevs_single_matches_multiproc(debug=False):
    """Test constantevs gives identical results single threaded and multiprocessed.

    The single process path used to ignore ``constantevs`` outside the ``coefficientsonly``
    branch and index the shared regressor per voxel, raising IndexError, while the
    multiprocessing path handled it correctly - so the same call behaved differently
    depending only on ``nprocs``.
    """
    if debug:
        print("linfitfiltpass_constantevs_single_matches_multiproc")
    for procbyvoxel in [True, False]:
        single = _run_constantevs(1, procbyvoxel)
        multi = _run_constantevs(2, procbyvoxel)
        for key in single:
            np.testing.assert_allclose(
                single[key],
                multi[key],
                atol=1e-12,
                err_msg=f"{key} differs between nprocs=1 and nprocs=2 (procbyvoxel={procbyvoxel})",
            )
        # the data is a clean multiple of the regressor, so the coefficients are 1, 2, 3, 4
        np.testing.assert_allclose(single["fitcoeff"], [1.0, 2.0, 3.0, 4.0], atol=1e-2)


def test_linfitfiltpass_constantevs_passes_whole_regressor(debug=False):
    """Test constantevs hands the whole regressor to the fitter rather than a slice."""
    if debug:
        print("linfitfiltpass_constantevs_passes_whole_regressor")
    nvox, tpts = 3, 10
    theev = np.linspace(1.0, 2.0, tpts)
    thedata = np.ones((nvox, tpts))
    received = []

    def _mock_proc(item, evs, data, rt_floattype=np.dtype(np.float64)):
        received.append(np.array(evs, copy=True))
        return (item, 1.0, 0.5, 0.25, 2.0, 0.7, np.zeros_like(data), np.zeros_like(data))

    with patch("rapidtide.linfitfiltpass._procOneRegressionFitItem", side_effect=_mock_proc):
        tide_linfitfiltpass.linfitfiltpass(
            nvox,
            thedata,
            None,
            theev,
            np.zeros(nvox),
            np.zeros(nvox),
            np.zeros(nvox),
            np.zeros(nvox),
            np.zeros(nvox),
            np.zeros((nvox, tpts)),
            np.zeros((nvox, tpts)),
            nprocs=1,
            constantevs=True,
            showprogressbar=False,
            verbose=False,
        )
    assert len(received) == nvox
    for vox in range(nvox):
        np.testing.assert_allclose(received[vox], theev)


# ==================== multiprocessing path tests ====================


def _run_both_procs(procbyvoxel, mode, twodcoeffs=False, nvox=5, tpts=30):
    """Run one output mode through both the single and multiprocessing paths.

    Parameters
    ----------
    procbyvoxel : bool
        Whether to process by voxel or by timepoint.
    mode : str
        One of "default", "coefficientsonly", or "confoundregress".
    twodcoeffs : bool, optional
        If True, use a two column design matrix so the coefficient arrays are 2D.
        Default is False.
    nvox : int, optional
        Number of voxels.  Default is 5.
    tpts : int, optional
        Number of timepoints.  Default is 30.

    Returns
    -------
    tuple of (dict, dict)
        The output arrays from the single process and multiprocessing runs.
    """
    rng = np.random.RandomState(11)
    if twodcoeffs:
        theevs = rng.randn(tpts, 2)
        base = theevs[:, 0] + 0.5 * theevs[:, 1]
    else:
        theevs = rng.randn(tpts)
        base = theevs
    voxdata = np.outer(np.arange(1.0, nvox + 1), base) + 0.001 * rng.randn(nvox, tpts)

    def _once(nprocs):
        thedata = voxdata if procbyvoxel else np.ascontiguousarray(voxdata.T)
        coeffshape = (nvox, 2) if twodcoeffs and procbyvoxel else (nvox,)
        if twodcoeffs and not procbyvoxel:
            coeffshape = (2, nvox)
        arrays = {
            "meanvalue": np.zeros(nvox),
            "rvalue": np.zeros(nvox),
            "r2value": np.zeros(nvox),
            "fitcoeff": np.zeros(coeffshape),
            "fitNorm": np.zeros(coeffshape),
            "datatoremove": np.zeros_like(thedata),
            "filtereddata": np.zeros_like(thedata),
        }
        tide_linfitfiltpass.linfitfiltpass(
            nvox,
            thedata,
            None,
            theevs,
            arrays["meanvalue"],
            arrays["rvalue"],
            arrays["r2value"],
            arrays["fitcoeff"],
            arrays["fitNorm"],
            arrays["datatoremove"],
            arrays["filtereddata"],
            nprocs=nprocs,
            constantevs=True,
            coefficientsonly=(mode == "coefficientsonly"),
            confoundregress=(mode == "confoundregress"),
            procbyvoxel=procbyvoxel,
            showprogressbar=False,
            verbose=False,
        )
        return arrays

    return _once(1), _once(2)


def test_linfitfiltpass_multiproc_matches_singleproc(debug=False):
    """Test every output mode agrees between the single and multiprocessing paths."""
    if debug:
        print("linfitfiltpass_multiproc_matches_singleproc")
    for procbyvoxel in [True, False]:
        for mode in ["default", "coefficientsonly", "confoundregress"]:
            single, multi = _run_both_procs(procbyvoxel, mode)
            for key in single:
                np.testing.assert_allclose(
                    single[key],
                    multi[key],
                    atol=1e-12,
                    err_msg=f"{key} differs for {mode}, procbyvoxel={procbyvoxel}",
                )
            if debug:
                print(f"  {mode}, procbyvoxel={procbyvoxel}: agree")


def test_linfitfiltpass_multiproc_2d_coefficients(debug=False):
    """Test the 2D coefficient array branches agree between both processing paths."""
    if debug:
        print("linfitfiltpass_multiproc_2d_coefficients")
    for procbyvoxel in [True, False]:
        for mode in ["default", "coefficientsonly"]:
            single, multi = _run_both_procs(procbyvoxel, mode, twodcoeffs=True)
            assert single["fitcoeff"].ndim == 2
            for key in single:
                np.testing.assert_allclose(
                    single[key],
                    multi[key],
                    atol=1e-12,
                    err_msg=f"{key} differs for 2D {mode}, procbyvoxel={procbyvoxel}",
                )


def test_linfitfiltpass_alwaysmultiproc(debug=False):
    """Test alwaysmultiproc forces the multiprocessing path even with nprocs of 1."""
    if debug:
        print("linfitfiltpass_alwaysmultiproc")
    nvox, tpts = 4, 24
    theev, thedata = _fitdata(nvox=nvox, tpts=tpts)
    r2value = np.zeros(nvox)
    filtereddata = np.zeros((nvox, tpts))
    items = tide_linfitfiltpass.linfitfiltpass(
        nvox,
        thedata,
        None,
        theev,
        None,
        None,
        r2value,
        None,
        None,
        None,
        filtereddata,
        nprocs=1,
        alwaysmultiproc=True,
        constantevs=True,
        confoundregress=True,
        showprogressbar=False,
        verbose=False,
    )
    assert items == nvox
    assert np.all(r2value > 0.99)


def test_linfitfiltpass_debug_with_optional_arrays_absent(debug=False):
    """Test the debug report copes with the output arrays that are legitimately None.

    ``confoundregress`` and ``coefficientsonly`` modes leave several of the output arrays
    unallocated, but the end of run debug report used to dereference all of them
    unconditionally - so passing debug=True crashed, which is how ``retroregress --debug``
    came to be unusable.
    """
    if debug:
        print("linfitfiltpass_debug_with_optional_arrays_absent")
    nvox, tpts = 4, 20
    theev, thedata = _fitdata(nvox=nvox, tpts=tpts)

    # confoundregress: only r2value and filtereddata are populated
    items = tide_linfitfiltpass.linfitfiltpass(
        nvox,
        thedata,
        None,
        theev,
        None,
        None,
        np.zeros(nvox),
        None,
        None,
        None,
        np.zeros((nvox, tpts)),
        nprocs=1,
        constantevs=True,
        confoundregress=True,
        showprogressbar=False,
        verbose=False,
        debug=True,
    )
    assert items == nvox

    # coefficientsonly: datatoremove stays None
    items = tide_linfitfiltpass.linfitfiltpass(
        nvox,
        thedata,
        None,
        theev,
        np.zeros(nvox),
        np.zeros(nvox),
        np.zeros(nvox),
        np.zeros(nvox),
        np.zeros(nvox),
        None,
        np.zeros((nvox, tpts)),
        nprocs=1,
        constantevs=True,
        coefficientsonly=True,
        showprogressbar=False,
        verbose=False,
        debug=True,
    )
    assert items == nvox


def test_linfitfiltpass_debug_and_progressbar(debug=False):
    """Test the debug reporting and progress bar paths run without error."""
    if debug:
        print("linfitfiltpass_debug_and_progressbar")
    nvox, tpts = 3, 16
    theev, thedata = _fitdata(nvox=nvox, tpts=tpts)
    arrays = {
        "meanvalue": np.zeros(nvox),
        "rvalue": np.zeros(nvox),
        "r2value": np.zeros(nvox),
        "fitcoeff": np.zeros(nvox),
        "fitNorm": np.zeros(nvox),
        "datatoremove": np.zeros((nvox, tpts)),
        "filtereddata": np.zeros((nvox, tpts)),
    }
    items = tide_linfitfiltpass.linfitfiltpass(
        nvox,
        thedata,
        None,
        theev,
        arrays["meanvalue"],
        arrays["rvalue"],
        arrays["r2value"],
        arrays["fitcoeff"],
        arrays["fitNorm"],
        arrays["datatoremove"],
        arrays["filtereddata"],
        nprocs=1,
        constantevs=True,
        showprogressbar=True,
        debug=True,
        verbose=True,
    )
    assert items == nvox


# ==================== makevoxelspecificderivs tests ====================


def test_makevoxelspecificderivs_zero_derivatives(debug=False):
    """Test requesting zero derivatives returns the input unchanged."""
    if debug:
        print("makevoxelspecificderivs_zero_derivatives")
    theevs = np.arange(12, dtype=np.float64).reshape((3, 4))
    result = tide_linfitfiltpass.makevoxelspecificderivs(theevs, nderivs=0)
    np.testing.assert_allclose(result, theevs)


def test_makevoxelspecificderivs_first_derivative(debug=False):
    """Test the first derivative column is the gradient of the input timecourse."""
    if debug:
        print("makevoxelspecificderivs_first_derivative")
    tpts = 20
    theevs = np.vstack([np.linspace(0.0, 4.0, tpts), np.linspace(0.0, -2.0, tpts)])
    # run with debug on so the shape reporting path is exercised too
    result = tide_linfitfiltpass.makevoxelspecificderivs(theevs, nderivs=1, debug=True)
    assert result.shape == (2, tpts, 2)
    for vox in range(2):
        np.testing.assert_allclose(result[vox, :, 0], theevs[vox, :])
        # the first Taylor coefficient is 1/1! = 1, so this is just the gradient
        np.testing.assert_allclose(result[vox, :, 1], np.gradient(theevs[vox, :]), atol=1e-12)


def test_makevoxelspecificderivs_second_derivative(debug=False):
    """Test higher derivatives are scaled by the reciprocal factorial of their order."""
    if debug:
        print("makevoxelspecificderivs_second_derivative")
    tpts = 25
    theevs = np.atleast_2d(np.linspace(0.0, 1.0, tpts) ** 2)
    result = tide_linfitfiltpass.makevoxelspecificderivs(theevs, nderivs=2)
    assert result.shape == (1, tpts, 3)
    firstderiv = np.gradient(theevs[0, :])
    np.testing.assert_allclose(result[0, :, 1], firstderiv, atol=1e-12)
    # each successive column differentiates the previous one and scales by 1/n!
    np.testing.assert_allclose(result[0, :, 2], np.gradient(firstderiv) / 2.0, atol=1e-12)


# ==================== confoundregress tests ====================


def _confounddata(nvox=6, tpts=60, nregressors=3, seed=5):
    """Build confound regressors and data containing a scaled copy of each.

    Parameters
    ----------
    nvox : int, optional
        Number of voxels.  Default is 6.
    tpts : int, optional
        Number of timepoints.  Default is 60.
    nregressors : int, optional
        Number of confound regressors.  Default is 3.
    seed : int, optional
        Random seed.  Default is 5.

    Returns
    -------
    tuple of (ndarray, list of str, ndarray)
        The regressors, their labels, and the data array.
    """
    rng = np.random.RandomState(seed)
    theregressors = rng.randn(nregressors, tpts)
    labels = [f"confound_{i:02d}" for i in range(nregressors)]
    thedata = np.zeros((nvox, tpts))
    for vox in range(nvox):
        thedata[vox, :] = 100.0 + (vox + 1) * theregressors[vox % nregressors, :]
    return theregressors, labels, thedata


def test_confoundregress_removes_confounds(debug=False):
    """Test confound regression reduces the variance explained by the regressors."""
    if debug:
        print("confoundregress_removes_confounds")
    theregressors, labels, thedata = _confounddata()
    outregressors, outlabels, filtereddata, r2value = tide_linfitfiltpass.confoundregress(
        theregressors, labels, thedata, 2.0, showprogressbar=False, debug=debug
    )
    assert filtereddata.shape == thedata.shape
    assert r2value.shape == (thedata.shape[0],)
    # every voxel is a pure linear combination of the confounds, so the fit is near perfect
    assert np.all(r2value > 0.99), f"r2 values too low: {r2value}"
    # the confound signal is gone from each timecourse; the per-voxel mean is kept, so
    # compare temporal variance rather than the variance of the array as a whole
    residualvar = np.var(filtereddata, axis=1)
    originalvar = np.var(thedata, axis=1)
    assert np.all(residualvar < 1e-6 * originalvar), f"residual variance too high: {residualvar}"


def test_confoundregress_does_not_mutate_input(debug=False):
    """Test the caller's regressor array survives the call unmodified.

    The regressors are normalized, optionally filtered and orthogonalized in place; because
    the leading slice produced a view rather than a copy, all of that used to be written
    straight back into the array the caller passed in.
    """
    if debug:
        print("confoundregress_does_not_mutate_input")
    theregressors, labels, thedata = _confounddata()
    original = theregressors.copy()
    outregressors, _, _, _ = tide_linfitfiltpass.confoundregress(
        theregressors, labels, thedata, 2.0, showprogressbar=False
    )
    np.testing.assert_allclose(
        theregressors, original, err_msg="confoundregress modified its input array"
    )
    # the processed regressors still come back through the return value
    assert not np.allclose(outregressors, original[:, : outregressors.shape[1]])


def test_confoundregress_no_orthogonalization(debug=False):
    """Test skipping orthogonalization keeps the original regressor labels and count."""
    if debug:
        print("confoundregress_no_orthogonalization")
    theregressors, labels, thedata = _confounddata()
    outregressors, outlabels, _, r2value = tide_linfitfiltpass.confoundregress(
        theregressors, labels, thedata, 2.0, orthogonalize=False, showprogressbar=False
    )
    assert outlabels == labels
    assert outregressors.shape[0] == theregressors.shape[0]
    assert np.all(r2value > 0.99)


def test_confoundregress_orthogonalization_relabels(debug=False):
    """Test orthogonalization replaces the labels with generated orthogconfound names."""
    if debug:
        print("confoundregress_orthogonalization_relabels")
    theregressors, labels, thedata = _confounddata()
    outregressors, outlabels, _, _ = tide_linfitfiltpass.confoundregress(
        theregressors, labels, thedata, 2.0, orthogonalize=True, showprogressbar=False
    )
    assert len(outlabels) == outregressors.shape[0]
    for i, thelabel in enumerate(outlabels):
        assert thelabel == "orthogconfound_{:02d}".format(i)


def test_confoundregress_timecourse_trimming(debug=False):
    """Test tcstart and tcend trim the regressors before fitting."""
    if debug:
        print("confoundregress_timecourse_trimming")
    tpts = 60
    theregressors, labels, thedata = _confounddata(tpts=tpts)
    trimmeddata = thedata[:, 5:45]
    outregressors, _, filtereddata, _ = tide_linfitfiltpass.confoundregress(
        theregressors,
        labels,
        trimmeddata,
        2.0,
        tcstart=5,
        tcend=45,
        showprogressbar=False,
    )
    assert outregressors.shape[1] == 40
    assert filtereddata.shape == trimmeddata.shape


def test_confoundregress_with_filtering(debug=False):
    """Test the optional high and low pass filtering of the regressors runs."""
    if debug:
        print("confoundregress_with_filtering")
    theregressors, labels, thedata = _confounddata(tpts=100)
    for tchp, tclp in [(0.01, None), (None, 0.15), (0.01, 0.15)]:
        outregressors, _, filtereddata, r2value = tide_linfitfiltpass.confoundregress(
            theregressors.copy(),
            list(labels),
            thedata,
            2.0,
            tchp=tchp,
            tclp=tclp,
            showprogressbar=False,
        )
        assert filtereddata.shape == thedata.shape
        assert np.all(np.isfinite(outregressors))


def test_confoundregress_no_surviving_regressors(debug=False):
    """Test the early return when orthogonalization removes every regressor.

    Gram-Schmidt drops regressors that are linear combinations of earlier ones, so a set of
    all-identical regressors collapses; the function returns the data untouched and a None
    R2 rather than attempting a fit.
    """
    if debug:
        print("confoundregress_no_surviving_regressors")
    tpts = 40
    # a set of constant regressors has nothing to orthogonalize against
    theregressors = np.zeros((3, tpts))
    labels = ["a", "b", "c"]
    thedata = np.random.RandomState(9).randn(4, tpts) + 100.0
    outregressors, outlabels, filtereddata, r2value = tide_linfitfiltpass.confoundregress(
        theregressors, labels, thedata, 2.0, orthogonalize=True, showprogressbar=False
    )
    if len(outlabels) == 0:
        assert r2value is None
        assert filtereddata is thedata
    else:
        # if anything survived, the call must still have produced a usable result
        assert r2value is not None


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_linfitfiltpass(debug=True, displayplots=True)
    test_linfitfiltpass_coefficientsonly_constantevs_paths(debug=True)
    test_procOneRegressionFitItem_univariate(debug=True)
    test_procOneRegressionFitItem_negative_coefficient(debug=True)
    test_procOneRegressionFitItem_multivariate(debug=True)
    test_procOneRegressionFitItem_multivariate_negative_first_coeff(debug=True)
    test_procOneRegressionFitItem_zero_coefficients(debug=True)
    test_linfitfiltpass_threshval_mean_mask(debug=True)
    test_linfitfiltpass_threshval_std_mask(debug=True)
    test_linfitfiltpass_explicit_validmask(debug=True)
    test_linfitfiltpass_constantevs_single_matches_multiproc(debug=True)
    test_linfitfiltpass_constantevs_passes_whole_regressor(debug=True)
    test_linfitfiltpass_multiproc_matches_singleproc(debug=True)
    test_linfitfiltpass_multiproc_2d_coefficients(debug=True)
    test_linfitfiltpass_alwaysmultiproc(debug=True)
    test_linfitfiltpass_debug_with_optional_arrays_absent(debug=True)
    test_linfitfiltpass_debug_and_progressbar(debug=True)
    test_makevoxelspecificderivs_zero_derivatives(debug=True)
    test_makevoxelspecificderivs_first_derivative(debug=True)
    test_makevoxelspecificderivs_second_derivative(debug=True)
    test_confoundregress_removes_confounds(debug=True)
    test_confoundregress_does_not_mutate_input(debug=True)
    test_confoundregress_no_orthogonalization(debug=True)
    test_confoundregress_orthogonalization_relabels(debug=True)
    test_confoundregress_timecourse_trimming(debug=True)
    test_confoundregress_with_filtering(debug=True)
    test_confoundregress_no_surviving_regressors(debug=True)
