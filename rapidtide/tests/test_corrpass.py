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
import builtins

import matplotlib as mpl
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

import rapidtide.calcsimfunc as tide_calcsimfunc
import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.resample as tide_resample
import rapidtide.simFuncClasses as tide_simFuncClasses
import rapidtide.simfuncfit as tide_simfuncfit
from rapidtide.tests.utils import mse


def test_calcsimfunc(debug=False, displayplots=False):
    # make the lfo filter
    lfofilter = tide_filt.NoncausalFilter(filtertype="lfo")

    # make some data
    oversampfactor = 2
    numvoxels = 100
    numtimepoints = 500
    tr = 0.72
    Fs = 1.0 / tr
    init_fmri_x = np.linspace(0.0, numtimepoints, numtimepoints, endpoint=False) * tr
    oversampfreq = oversampfactor * Fs
    os_fmri_x = np.linspace(
        0.0, numtimepoints * oversampfactor, numtimepoints * oversampfactor
    ) * (1.0 / oversampfreq)

    theinputdata = np.zeros((numvoxels, numtimepoints), dtype=np.float64)
    meanval = np.zeros((numvoxels), dtype=np.float64)

    testfreq = 0.075
    msethresh = 1e-3

    # make the starting regressor
    sourcedata = np.sin(2.0 * np.pi * testfreq * os_fmri_x)
    numpasses = 1

    # make the timeshifted data
    shiftstart = -5.0
    shiftend = 5.0
    voxelshifts = np.linspace(shiftstart, shiftend, numvoxels, endpoint=False)
    for i in range(numvoxels):
        theinputdata[i, :] = np.sin(2.0 * np.pi * testfreq * (init_fmri_x - voxelshifts[i]))

    if displayplots:
        plt.figure()
        plt.plot(sourcedata)
        plt.show()
    genlagtc = tide_resample.FastResampler(os_fmri_x, sourcedata)

    thexcorr = tide_corr.fastcorrelate(sourcedata, sourcedata)
    xcorrlen = len(thexcorr)
    xcorr_x = (
        np.linspace(0.0, xcorrlen, xcorrlen, endpoint=False) * tr
        - (xcorrlen * tr) / 2.0
        + tr / 2.0
    )

    if displayplots:
        plt.figure()
        plt.plot(xcorr_x, thexcorr)
        plt.show()

    corrzero = xcorrlen // 2
    lagmin = -10.0
    lagmax = 10.0
    lagmininpts = int((-lagmin * oversampfreq) - 0.5)
    lagmaxinpts = int((lagmax * oversampfreq) + 0.5)

    searchstart = int(np.round(corrzero + lagmin / tr))
    searchend = int(np.round(corrzero + lagmax / tr))
    numcorrpoints = lagmaxinpts + lagmininpts
    corrout = np.zeros((numvoxels, numcorrpoints), dtype=np.float64)
    lagmask = np.zeros((numvoxels), dtype=np.float64)
    failimage = np.zeros((numvoxels), dtype=np.float64)
    lagtimes = np.zeros((numvoxels), dtype=np.float64)
    lagstrengths = np.zeros((numvoxels), dtype=np.float64)
    lagsigma = np.zeros((numvoxels), dtype=np.float64)
    gaussout = np.zeros((numvoxels, numcorrpoints), dtype=np.float64)
    windowout = np.zeros((numvoxels, numcorrpoints), dtype=np.float64)
    R2 = np.zeros((numvoxels), dtype=np.float64)
    lagtc = np.zeros((numvoxels, numtimepoints), dtype=np.float64)

    optiondict = {
        "numestreps": 10000,
        "interptype": "univariate",
        "showprogressbar": debug,
        "detrendorder": 3,
        "windowfunc": "hamming",
        "corrweighting": "None",
        "nprocs": 1,
        "widthlimit": 1000.0,
        "bipolar": False,
        "fixdelay": False,
        "peakfittype": "gauss",
        "lagmin": lagmin,
        "lagmax": lagmax,
        "absminsigma": 0.25,
        "absmaxsigma": 25.0,
        "edgebufferfrac": 0.0,
        "lthreshval": 0.0,
        "uthreshval": 1.1,
        "debug": False,
        "enforcethresh": True,
        "lagmod": 1000.0,
        "searchfrac": 0.5,
        "mp_chunksize": 1000,
        "oversampfactor": oversampfactor,
        "despeckle_thresh": 5.0,
        "zerooutbadfit": False,
        "permutationmethod": "shuffle",
        "hardlimit": True,
    }

    theprefilter = tide_filt.NoncausalFilter("lfo")
    theCorrelator = tide_simFuncClasses.Correlator(
        Fs=oversampfreq,
        ncprefilter=theprefilter,
        detrendorder=optiondict["detrendorder"],
        windowfunc=optiondict["windowfunc"],
        corrweighting=optiondict["corrweighting"],
    )

    thefitter = tide_simFuncClasses.SimilarityFunctionFitter(
        lagmod=optiondict["lagmod"],
        lthreshval=optiondict["lthreshval"],
        uthreshval=optiondict["uthreshval"],
        bipolar=optiondict["bipolar"],
        lagmin=optiondict["lagmin"],
        lagmax=optiondict["lagmax"],
        absmaxsigma=optiondict["absmaxsigma"],
        absminsigma=optiondict["absminsigma"],
        debug=optiondict["debug"],
        peakfittype=optiondict["peakfittype"],
        zerooutbadfit=optiondict["zerooutbadfit"],
        searchfrac=optiondict["searchfrac"],
        enforcethresh=optiondict["enforcethresh"],
        hardlimit=optiondict["hardlimit"],
    )

    if debug:
        print(optiondict)

    theCorrelator.setlimits(lagmininpts, lagmaxinpts)
    theCorrelator.setreftc(sourcedata)
    dummy, trimmedcorrscale, dummy = theCorrelator.getfunction()
    thefitter.setcorrtimeaxis(trimmedcorrscale)

    for thenprocs in [1, -1]:
        for i in range(numpasses):
            (
                voxelsprocessed_cp,
                theglobalmaxlist,
                trimmedcorrscale,
            ) = tide_calcsimfunc.correlationpass(
                theinputdata,
                sourcedata,
                theCorrelator,
                init_fmri_x,
                os_fmri_x,
                lagmininpts,
                lagmaxinpts,
                corrout,
                meanval,
                nprocs=thenprocs,
                oversampfactor=optiondict["oversampfactor"],
                interptype=optiondict["interptype"],
                showprogressbar=optiondict["showprogressbar"],
                chunksize=optiondict["mp_chunksize"],
            )

            if displayplots:
                plt.figure()
                plt.plot(trimmedcorrscale, corrout[numvoxels // 2, :], "k")
                plt.show()

            voxelsprocessed_fc = tide_simfuncfit.fitcorr(
                trimmedcorrscale,
                thefitter,
                corrout,
                lagmask,
                failimage,
                lagtimes,
                lagstrengths,
                lagsigma,
                gaussout,
                windowout,
                R2,
                nprocs=optiondict["nprocs"],
                fixdelay=optiondict["fixdelay"],
                showprogressbar=optiondict["showprogressbar"],
                chunksize=optiondict["mp_chunksize"],
                despeckle_thresh=optiondict["despeckle_thresh"],
            )
            if displayplots:
                plt.figure()
                plt.plot(voxelshifts, "k")
                plt.plot(lagtimes, "r")
                plt.show()

            if debug:
                for i in range(numvoxels):
                    print(
                        voxelshifts[i],
                        lagtimes[i],
                        lagstrengths[i],
                        lagsigma[i],
                        failimage[i],
                    )

            assert mse(voxelshifts, lagtimes) < msethresh


def test_correlationpass_gpu_matches_cpu(debug=False):
    # Small deterministic synthetic dataset for CPU/GPU parity checks.
    oversampfactor = 1
    numvoxels = 12
    numtimepoints = 160
    tr = 0.8
    Fs = 1.0 / tr
    init_fmri_x = np.linspace(0.0, numtimepoints, numtimepoints, endpoint=False) * tr
    os_fmri_x = init_fmri_x.copy()

    testfreq = 0.06
    referencetc = np.sin(2.0 * np.pi * testfreq * init_fmri_x)
    fmridata = np.zeros((numvoxels, numtimepoints), dtype=np.float64)
    for i in range(numvoxels):
        shift = 0.15 * i
        fmridata[i, :] = np.sin(2.0 * np.pi * testfreq * (init_fmri_x - shift))

    lagmin = -10.0
    lagmax = 10.0
    lagmininpts = int((-lagmin * Fs) - 0.5)
    lagmaxinpts = int((lagmax * Fs) + 0.5)
    numcorrpoints = lagmaxinpts + lagmininpts

    prefilt = tide_filt.NoncausalFilter("lfo")
    corr_cpu = tide_simFuncClasses.Correlator(
        Fs=Fs,
        ncprefilter=prefilt,
        detrendorder=3,
        windowfunc="hamming",
        corrweighting="None",
    )
    corr_gpu = tide_simFuncClasses.Correlator(
        Fs=Fs,
        ncprefilter=prefilt,
        detrendorder=3,
        windowfunc="hamming",
        corrweighting="None",
    )

    corrout_cpu = np.zeros((numvoxels, numcorrpoints), dtype=np.float64)
    meanval_cpu = np.zeros((numvoxels), dtype=np.float64)
    vox_cpu, gmax_cpu, corrscale_cpu = tide_calcsimfunc.correlationpass(
        fmridata,
        referencetc,
        corr_cpu,
        init_fmri_x,
        os_fmri_x,
        lagmininpts,
        lagmaxinpts,
        corrout_cpu,
        meanval_cpu,
        nprocs=1,
        oversampfactor=oversampfactor,
        interptype="univariate",
        showprogressbar=False,
    )

    corrout_gpu = np.zeros((numvoxels, numcorrpoints), dtype=np.float64)
    meanval_gpu = np.zeros((numvoxels), dtype=np.float64)
    vox_gpu, gmax_gpu, corrscale_gpu = tide_calcsimfunc.correlationpass_gpu(
        fmridata,
        referencetc,
        corr_gpu,
        init_fmri_x,
        os_fmri_x,
        lagmininpts,
        lagmaxinpts,
        corrout_gpu,
        meanval_gpu,
        oversampfactor=oversampfactor,
        interptype="univariate",
        showprogressbar=False,
        device="auto",
        batchsize=4,
        fallback_to_cpu=True,
    )

    assert vox_gpu == vox_cpu
    assert np.array_equal(
        np.asarray(gmax_gpu, dtype=np.int64), np.asarray(gmax_cpu, dtype=np.int64)
    )
    assert np.array_equal(corrscale_gpu, corrscale_cpu)
    assert np.allclose(meanval_gpu, meanval_cpu, atol=1e-10, rtol=1e-7)
    # Allow small floating-point differences when GPU path executes.
    assert np.allclose(corrout_gpu, corrout_cpu, atol=5e-4, rtol=2e-3)

    # If torch is present and a supported GPU backend exists, also exercise the strict
    # non-fallback code path so this test validates a true GPU execution path.
    torch = pytest.importorskip("torch")
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_cuda or has_mps:
        corrout_gpu2 = np.zeros((numvoxels, numcorrpoints), dtype=np.float64)
        meanval_gpu2 = np.zeros((numvoxels), dtype=np.float64)
        vox_gpu2, gmax_gpu2, corrscale_gpu2 = tide_calcsimfunc.correlationpass_gpu(
            fmridata,
            referencetc,
            tide_simFuncClasses.Correlator(
                Fs=Fs,
                ncprefilter=prefilt,
                detrendorder=3,
                windowfunc="hamming",
                corrweighting="None",
            ),
            init_fmri_x,
            os_fmri_x,
            lagmininpts,
            lagmaxinpts,
            corrout_gpu2,
            meanval_gpu2,
            oversampfactor=oversampfactor,
            interptype="univariate",
            showprogressbar=False,
            device="auto",
            batchsize=4,
            fallback_to_cpu=False,
        )
        assert vox_gpu2 == vox_cpu
        assert np.array_equal(
            np.asarray(gmax_gpu2, dtype=np.int64), np.asarray(gmax_cpu, dtype=np.int64)
        )
        assert np.array_equal(corrscale_gpu2, corrscale_cpu)
        assert np.allclose(meanval_gpu2, meanval_cpu, atol=1e-10, rtol=1e-7)
        assert np.allclose(corrout_gpu2, corrout_cpu, atol=5e-4, rtol=2e-3)
        if debug:
            print("cpu and gpu outputs match!")


def test_correlationpass_gpu_phat_on_real_hardware(debug=False):
    """Repeat the CPU/GPU phat comparison on whatever GPU this machine has.

    The other phat parity test pins device resolution to torch's CPU device so it
    runs everywhere, which means it never touches a real backend.  phat is the
    default weighting for rapidtide, and it is the weighting whose arithmetic - a
    per-bin division and two rescalings - is most exposed to a backend's fp32
    quirks, so it is worth checking against actual hardware where there is some.
    """
    torch = pytest.importorskip("torch")
    if not (
        torch.cuda.is_available()
        or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    ):
        pytest.skip("no GPU backend available")

    theinputs = _makegpuinputs()
    (
        thecpuvoxels,
        thecpuglobalmax,
        thecpuscale,
        thecpucorrout,
        thecpumeanval,
    ) = _cpucorrelationpass(theinputs, corrweighting="phat")

    thegpuvoxels, thegpuglobalmax, thegpuscale, thegpucorrout, thegpumeanval = _rungpupass(
        theinputs,
        _makecorrelator(theinputs["Fs"], corrweighting="phat"),
        device="auto",
        batchsize=4,
        fallback_to_cpu=False,
        debug=debug,
    )

    assert thegpuvoxels == thecpuvoxels
    assert np.array_equal(thegpuscale, thecpuscale)
    assert np.array_equal(
        np.asarray(thegpuglobalmax, dtype=np.int64),
        np.asarray(thecpuglobalmax, dtype=np.int64),
    )
    np.testing.assert_allclose(thegpumeanval, thecpumeanval, atol=1e-10)
    # looser than the CPU-device comparison: a real backend's fp32 FFT need not agree
    # with the CPU's to the last bit
    np.testing.assert_allclose(thegpucorrout, thecpucorrout, atol=5e-4, rtol=2e-3)
    # the rescaling is the point, so check the amplitude explicitly rather than
    # letting a relative tolerance hide a scale factor
    assert np.max(np.abs(thegpucorrout)) == pytest.approx(np.max(np.abs(thecpucorrout)), rel=1e-3)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_calcsimfunc(debug=True, displayplots=True)
    test_correlationpass_gpu_matches_cpu(debug=True)

# ==================== _resolve_torch_device ====================


class _FakeBackends:
    """Stands in for torch.backends, with a controllable mps availability."""

    class _Mps:
        def __init__(self, available):
            self._available = available

        def is_available(self):
            return self._available

    def __init__(self, hasmps=True, mpsavailable=False):
        if hasmps:
            self.mps = self._Mps(mpsavailable)


class _FakeTorch:
    """A minimal stand in for the torch module, for device resolution only.

    Device resolution is pure dispatch over what a torch build reports, so a fake
    lets every branch be exercised on any machine - including the CUDA and ROCm
    paths, which no test could otherwise reach on a CPU only or Apple runner.

    Parameters
    ----------
    cudaavailable : bool
        What torch.cuda.is_available() reports.
    hasmps : bool
        Whether torch.backends has an mps attribute at all.
    mpsavailable : bool
        What torch.backends.mps.is_available() reports.
    hip : str or None
        torch.version.hip, which is set only on ROCm builds.
    """

    def __init__(self, cudaavailable=False, hasmps=True, mpsavailable=False, hip=None):
        class _Cuda:
            @staticmethod
            def is_available():
                return cudaavailable

        class _Version:
            pass

        self.cuda = _Cuda()
        self.backends = _FakeBackends(hasmps=hasmps, mpsavailable=mpsavailable)
        self.version = _Version()
        self.version.hip = hip

    @staticmethod
    def device(thename):
        return f"device:{thename}"


def resolve_torch_device_auto_prefers_cuda(debug=False):
    """auto takes CUDA when it is there, MPS otherwise, and refuses when neither is."""
    if debug:
        print("resolve_torch_device_auto_prefers_cuda")

    # CUDA present, and preferred even when MPS is also available
    thefake = _FakeTorch(cudaavailable=True, mpsavailable=True)
    assert tide_calcsimfunc._resolve_torch_device(thefake, "auto") == "device:cuda"

    # no CUDA, fall through to MPS
    thefake = _FakeTorch(cudaavailable=False, mpsavailable=True)
    assert tide_calcsimfunc._resolve_torch_device(thefake, "auto") == "device:mps"

    # neither: auto has nothing to offer and says so
    thefake = _FakeTorch(cudaavailable=False, mpsavailable=False)
    with pytest.raises(RuntimeError, match="No supported GPU backend"):
        tide_calcsimfunc._resolve_torch_device(thefake, "auto")

    # a torch build with no mps attribute at all must not raise AttributeError
    thefake = _FakeTorch(cudaavailable=False, hasmps=False)
    with pytest.raises(RuntimeError, match="No supported GPU backend"):
        tide_calcsimfunc._resolve_torch_device(thefake, "auto")


def resolve_torch_device_explicit_backends(debug=False):
    """Naming a backend explicitly must fail loudly when it is not available, rather
    than silently substituting a different one."""
    if debug:
        print("resolve_torch_device_explicit_backends")

    # cuda
    assert (
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(cudaavailable=True), "cuda")
        == "device:cuda"
    )
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(cudaavailable=False), "cuda")

    # mps
    assert (
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(mpsavailable=True), "mps")
        == "device:mps"
    )
    with pytest.raises(RuntimeError, match="MPS is not available"):
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(mpsavailable=False), "mps")
    with pytest.raises(RuntimeError, match="MPS is not available"):
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(hasmps=False), "mps")

    # rocm reports itself as a cuda device, but only on a hip enabled build
    assert (
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(cudaavailable=True, hip="6.0"), "rocm")
        == "device:cuda"
    )
    with pytest.raises(RuntimeError, match="GPU backend is not available"):
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(cudaavailable=False), "rocm")
    with pytest.raises(RuntimeError, match="not ROCm-enabled"):
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(cudaavailable=True, hip=None), "rocm")

    # case is not significant
    assert (
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(cudaavailable=True), "CUDA")
        == "device:cuda"
    )


def resolve_torch_device_rejects_an_unknown_name(debug=False):
    """An unrecognised device name is a caller error, and the message has to say what
    the legal choices are."""
    if debug:
        print("resolve_torch_device_rejects_an_unknown_name")

    with pytest.raises(ValueError, match="auto, cuda, rocm, mps"):
        tide_calcsimfunc._resolve_torch_device(_FakeTorch(), "tpu")


# ==================== correlationpass_gpu guards and fallbacks ====================


def _makegpuinputs(numvoxels=8, numtimepoints=128, tr=0.8):
    """Build a small deterministic dataset for the GPU entry point.

    Returns
    -------
    dict
        Everything correlationpass_gpu needs, keyed by argument name.
    """
    Fs = 1.0 / tr
    init_fmri_x = np.linspace(0.0, numtimepoints, numtimepoints, endpoint=False) * tr
    testfreq = 0.06
    referencetc = np.sin(2.0 * np.pi * testfreq * init_fmri_x)
    fmridata = np.zeros((numvoxels, numtimepoints), dtype=np.float64)
    for i in range(numvoxels):
        fmridata[i, :] = np.sin(2.0 * np.pi * testfreq * (init_fmri_x - 0.15 * i))

    lagmininpts = int((10.0 * Fs) - 0.5)
    lagmaxinpts = int((10.0 * Fs) + 0.5)
    return {
        "fmridata": fmridata,
        "referencetc": referencetc,
        "fmri_x": init_fmri_x,
        "os_fmri_x": init_fmri_x.copy(),
        "lagmininpts": lagmininpts,
        "lagmaxinpts": lagmaxinpts,
        "numcorrpoints": lagmaxinpts + lagmininpts,
        "numvoxels": numvoxels,
        "Fs": Fs,
    }


def _makecorrelator(Fs, corrweighting="None"):
    """Build a Correlator configured the way the GPU path expects."""
    return tide_simFuncClasses.Correlator(
        Fs=Fs,
        ncprefilter=tide_filt.NoncausalFilter("lfo"),
        detrendorder=3,
        windowfunc="hamming",
        corrweighting=corrweighting,
    )


def correlationpass_gpu_rejects_a_bad_batchsize(debug=False):
    """A batch has to contain at least one voxel."""
    if debug:
        print("correlationpass_gpu_rejects_a_bad_batchsize")

    theinputs = _makegpuinputs()
    with pytest.raises(ValueError, match="batchsize must be >= 1"):
        tide_calcsimfunc.correlationpass_gpu(
            theinputs["fmridata"],
            theinputs["referencetc"],
            _makecorrelator(theinputs["Fs"]),
            theinputs["fmri_x"],
            theinputs["os_fmri_x"],
            theinputs["lagmininpts"],
            theinputs["lagmaxinpts"],
            np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"])),
            np.zeros(theinputs["numvoxels"]),
            batchsize=0,
        )


def _rungpupass(theinputs, thecorrelator, **thekwargs):
    """Call correlationpass_gpu with fresh output arrays.

    Returns
    -------
    tuple
        (voxelcount, globalmaxlist, corrscale, corrout, meanval)
    """
    thecorrout = np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64)
    themeanval = np.zeros(theinputs["numvoxels"], dtype=np.float64)
    thevoxels, theglobalmax, thecorrscale = tide_calcsimfunc.correlationpass_gpu(
        theinputs["fmridata"],
        theinputs["referencetc"],
        thecorrelator,
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        thecorrout,
        themeanval,
        showprogressbar=False,
        **thekwargs,
    )
    return thevoxels, theglobalmax, thecorrscale, thecorrout, themeanval


def correlationpass_gpu_falls_back_for_unsupported_weighting(debug=False):
    """Only 'None' and 'phat' weighting are implemented on the GPU.  Anything else
    has to hand off to the CPU rather than silently computing the wrong thing."""
    if debug:
        print("correlationpass_gpu_falls_back_for_unsupported_weighting")

    theinputs = _makegpuinputs()
    thevoxels, dummy, dummy2, thecorrout, themeanval = _rungpupass(
        theinputs, _makecorrelator(theinputs["Fs"], corrweighting="liang")
    )

    assert thevoxels == theinputs["numvoxels"]
    assert np.any(thecorrout != 0.0), "the fallback produced nothing"

    # and with the fallback refused, the unsupported weighting is an error
    with pytest.raises(Exception):
        _rungpupass(
            theinputs,
            _makecorrelator(theinputs["Fs"], corrweighting="liang"),
            fallback_to_cpu=False,
        )


def correlationpass_gpu_falls_back_without_a_gpu(debug=False):
    """On a machine with no GPU the entry point still has to produce results.  This
    is the path every CPU only runner takes, so it is forced here rather than left to
    depend on the hardware the tests happen to run on."""
    if debug:
        print("correlationpass_gpu_falls_back_without_a_gpu")

    # this exercises the "torch is here but has no GPU" path, so it needs torch to be
    # importable; the no-torch-at-all path is covered separately below
    pytest.importorskip("torch")

    theinputs = _makegpuinputs()

    # what the CPU implementation gives, for comparison
    thecpucorrout = np.zeros(
        (theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64
    )
    thecpumeanval = np.zeros(theinputs["numvoxels"], dtype=np.float64)
    thecpuvoxels, dummy, dummy2 = tide_calcsimfunc.correlationpass_cpu(
        theinputs["fmridata"],
        theinputs["referencetc"],
        _makecorrelator(theinputs["Fs"]),
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        thecpucorrout,
        thecpumeanval,
        showprogressbar=False,
    )

    # no GPU backend available
    with patch.object(
        tide_calcsimfunc,
        "_resolve_torch_device",
        side_effect=RuntimeError("No supported GPU backend found (CUDA/ROCm/MPS)."),
    ):
        thevoxels, dummy3, dummy4, thecorrout, themeanval = _rungpupass(
            theinputs, _makecorrelator(theinputs["Fs"])
        )

    assert thevoxels == thecpuvoxels
    np.testing.assert_allclose(thecorrout, thecpucorrout, atol=1e-10)
    np.testing.assert_allclose(themeanval, thecpumeanval, atol=1e-10)

    # and refusing the fallback surfaces the error instead of returning nothing
    with patch.object(
        tide_calcsimfunc,
        "_resolve_torch_device",
        side_effect=RuntimeError("No supported GPU backend found (CUDA/ROCm/MPS)."),
    ):
        with pytest.raises(RuntimeError, match="No supported GPU backend"):
            _rungpupass(theinputs, _makecorrelator(theinputs["Fs"]), fallback_to_cpu=False)


def correlationpass_gpu_falls_back_without_torch(debug=False):
    """torch is an optional dependency, so its absence has to be handled the same
    way as a missing GPU."""
    if debug:
        print("correlationpass_gpu_falls_back_without_torch")

    theinputs = _makegpuinputs()

    thereal_import = builtins.__import__

    def theblockedimport(thename, *theargs, **thekwargs):
        if thename == "torch":
            raise ImportError("No module named 'torch'")
        return thereal_import(thename, *theargs, **thekwargs)

    with patch.object(builtins, "__import__", theblockedimport):
        thevoxels, dummy, dummy2, thecorrout, dummy3 = _rungpupass(
            theinputs, _makecorrelator(theinputs["Fs"])
        )
    assert thevoxels == theinputs["numvoxels"]
    assert np.any(thecorrout != 0.0), "the no-torch fallback produced nothing"

    with patch.object(builtins, "__import__", theblockedimport):
        with pytest.raises(ImportError):
            _rungpupass(theinputs, _makecorrelator(theinputs["Fs"]), fallback_to_cpu=False)


def correlationpass_gpu_checks_the_reference_length(debug=False):
    """The reference timecourse has to match the oversampled time axis, or the
    correlation would be computed against a misaligned regressor.

    The check sits downstream of device resolution, so on a machine with no GPU the
    fallback to the CPU implementation intercepts the bad input before the check is
    ever reached.  Resolution is therefore pinned to the CPU torch device, which runs
    the GPU code path everywhere without needing GPU hardware.
    """
    if debug:
        print("correlationpass_gpu_checks_the_reference_length")

    torch = pytest.importorskip("torch")

    theinputs = _makegpuinputs()
    theinputs["referencetc"] = theinputs["referencetc"][:-5]
    with patch.object(tide_calcsimfunc, "_resolve_torch_device", return_value=torch.device("cpu")):
        with pytest.raises(ValueError, match="does not match"):
            _rungpupass(theinputs, _makecorrelator(theinputs["Fs"]))


def correlationpass_gpu_reference_mismatch_reaches_the_cpu_path(debug=False):
    """Without a GPU the same bad input takes the fallback instead, and the CPU
    implementation responds by calling sys.exit rather than raising.

    Pinned, not fixed: the exit lives in Correlator.run in simFuncClasses and is
    long-standing behaviour shared by every caller, so changing it is a separate
    decision.  The consequence worth recording is that the two implementations of the
    same entry point reject the same input in incompatible ways - one catchable, one
    not.
    """
    if debug:
        print("correlationpass_gpu_reference_mismatch_reaches_the_cpu_path")

    theinputs = _makegpuinputs()
    theinputs["referencetc"] = theinputs["referencetc"][:-5]
    with patch.object(
        tide_calcsimfunc,
        "_resolve_torch_device",
        side_effect=RuntimeError("No supported GPU backend found (CUDA/ROCm/MPS)."),
    ):
        with pytest.raises(SystemExit):
            _rungpupass(theinputs, _makecorrelator(theinputs["Fs"]))


def test_gpu_entrypoint_on_a_machine_with_no_gpu(nogpu):
    """The CPU-only case, driven through the real device resolution code.

    The sub-tests above reach this path by mocking _resolve_torch_device, which means
    they would keep passing if that routine stopped raising RuntimeError and the
    fallback in correlationpass_gpu stopped catching it.  Here nothing in rapidtide is
    mocked - the nogpu fixture patches torch itself, so resolution genuinely fails and
    the fallback has to handle it.
    """
    torch = pytest.importorskip("torch")

    # the fixture has to actually take the hardware away, or this proves nothing
    assert not torch.cuda.is_available()
    assert not torch.backends.mps.is_available()

    theinputs = _makegpuinputs()

    thecpucorrout = np.zeros(
        (theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64
    )
    thecpumeanval = np.zeros(theinputs["numvoxels"], dtype=np.float64)
    thecpuvoxels, dummy, dummy2 = tide_calcsimfunc.correlationpass_cpu(
        theinputs["fmridata"],
        theinputs["referencetc"],
        _makecorrelator(theinputs["Fs"]),
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        thecpucorrout,
        thecpumeanval,
        showprogressbar=False,
    )

    thevoxels, dummy3, dummy4, thecorrout, themeanval = _rungpupass(
        theinputs, _makecorrelator(theinputs["Fs"])
    )

    assert thevoxels == thecpuvoxels
    np.testing.assert_allclose(thecorrout, thecpucorrout, atol=1e-10)
    np.testing.assert_allclose(themeanval, thecpumeanval, atol=1e-10)

    # and with the fallback refused, the real resolution failure surfaces
    with pytest.raises(RuntimeError, match="No supported GPU backend"):
        _rungpupass(theinputs, _makecorrelator(theinputs["Fs"]), fallback_to_cpu=False)


def test_calcsimfuncdevices(debug=False):
    """Entry point for the device resolution and GPU fallback tests."""
    resolve_torch_device_auto_prefers_cuda(debug=debug)
    resolve_torch_device_explicit_backends(debug=debug)
    resolve_torch_device_rejects_an_unknown_name(debug=debug)
    correlationpass_gpu_rejects_a_bad_batchsize(debug=debug)
    correlationpass_gpu_falls_back_for_unsupported_weighting(debug=debug)
    correlationpass_gpu_falls_back_without_a_gpu(debug=debug)
    correlationpass_gpu_falls_back_without_torch(debug=debug)
    correlationpass_gpu_checks_the_reference_length(debug=debug)
    correlationpass_gpu_reference_mismatch_reaches_the_cpu_path(debug=debug)


# ==================== single voxel worker ====================


def _preparedcorrelator(theinputs, corrweighting="None"):
    """Build a Correlator that has already been given its reference and limits.

    Parameters
    ----------
    theinputs : dict
        As returned by _makegpuinputs.
    corrweighting : str, optional
        Weighting to configure the Correlator with.

    Returns
    -------
    Correlator
        Ready to have run() called on it.
    """
    thecorrelator = _makecorrelator(theinputs["Fs"], corrweighting=corrweighting)
    thecorrelator.setreftc(theinputs["referencetc"])
    thecorrelator.setlimits(theinputs["lagmininpts"], theinputs["lagmaxinpts"])
    return thecorrelator


def procOneVoxelCorrelation_can_skip_the_resampling(capsys, debug=False):
    """An oversampling factor below one means the timecourse is used as it stands.

    Every other caller resamples onto the oversampled time axis first, so this branch
    is the one taken when the data is already at the correlation sample rate and a
    resample would only add interpolation error.
    """
    if debug:
        print("procOneVoxelCorrelation_can_skip_the_resampling")

    theinputs = _makegpuinputs()
    thecorrelator = _preparedcorrelator(theinputs)
    thetc = np.zeros(len(theinputs["os_fmri_x"]), dtype=np.float64)
    thevoxelargs = tide_calcsimfunc._packvoxeldata(
        3,
        [
            thetc,
            thecorrelator,
            theinputs["fmri_x"],
            theinputs["fmridata"],
            theinputs["os_fmri_x"],
            [],
            np.zeros(100, dtype=np.float64),
        ],
    )

    thevox, themean, dummy, dummy2, dummy3, dummy4 = tide_calcsimfunc._procOneVoxelCorrelation(
        3, thevoxelargs, oversampfactor=0, debug=True
    )

    assert thevox == 3
    # the timecourse was copied straight across, with no interpolation applied
    np.testing.assert_allclose(thetc, theinputs["fmridata"][3, :], atol=1e-12)
    assert themean == pytest.approx(np.mean(theinputs["fmridata"][3, :]))
    # debug names the two settings that decide which branch was just taken
    theoutput = capsys.readouterr().out
    assert "oversampfactor=0" in theoutput
    assert "interptype=" in theoutput


# ==================== correlationpass_cpu ====================


def correlationpass_cpu_can_be_verbose(capsys, debug=False):
    """The debug report names the reference length, which is the value that has to
    agree with the data for the correlation to mean anything."""
    if debug:
        print("correlationpass_cpu_can_be_verbose")

    theinputs = _makegpuinputs()
    tide_calcsimfunc.correlationpass_cpu(
        theinputs["fmridata"],
        theinputs["referencetc"],
        _makecorrelator(theinputs["Fs"]),
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64),
        np.zeros(theinputs["numvoxels"], dtype=np.float64),
        showprogressbar=False,
        debug=True,
    )

    assert f"length {len(theinputs['referencetc'])}" in capsys.readouterr().out


def correlationpass_cpu_reports_uncollectable_garbage(debug=False):
    """A voxelwise pass allocates heavily, so the collector result is logged.

    The interesting message is the one that only appears when something could not be
    freed, which a healthy run never produces.
    """
    if debug:
        print("correlationpass_cpu_reports_uncollectable_garbage")

    theinputs = _makegpuinputs()
    thelogger = MagicMock()
    with (
        patch.object(tide_calcsimfunc.gc, "collect", return_value=7),
        patch.object(tide_calcsimfunc, "LGR", thelogger),
    ):
        tide_calcsimfunc.correlationpass_cpu(
            theinputs["fmridata"],
            theinputs["referencetc"],
            _makecorrelator(theinputs["Fs"]),
            theinputs["fmri_x"],
            theinputs["os_fmri_x"],
            theinputs["lagmininpts"],
            theinputs["lagmaxinpts"],
            np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64),
            np.zeros(theinputs["numvoxels"], dtype=np.float64),
            showprogressbar=False,
        )

    themessages = [thecall.args[0] for thecall in thelogger.info.call_args_list]
    assert any("unable to collect 7 objects" in m for m in themessages)


def correlationpass_cpu_can_run_multiprocessed(debug=False):
    """More than one process must give the same answer as one.

    nprocs greater than one is what a production run uses, and it takes an entirely
    separate path through genericmultiproc, so agreement is not automatic.
    """
    if debug:
        print("correlationpass_cpu_can_run_multiprocessed")

    theinputs = _makegpuinputs()

    theserial = np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64)
    theserialmean = np.zeros(theinputs["numvoxels"], dtype=np.float64)
    theserialcount, dummy, theserialscale = tide_calcsimfunc.correlationpass_cpu(
        theinputs["fmridata"],
        theinputs["referencetc"],
        _makecorrelator(theinputs["Fs"]),
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        theserial,
        theserialmean,
        nprocs=1,
        showprogressbar=False,
    )

    theparallel = np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64)
    theparallelmean = np.zeros(theinputs["numvoxels"], dtype=np.float64)
    theparallelcount, dummy2, theparallelscale = tide_calcsimfunc.correlationpass_cpu(
        theinputs["fmridata"],
        theinputs["referencetc"],
        _makecorrelator(theinputs["Fs"]),
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        theparallel,
        theparallelmean,
        nprocs=2,
        showprogressbar=False,
    )

    assert theparallelcount == theserialcount == theinputs["numvoxels"]
    np.testing.assert_allclose(theparallelscale, theserialscale, atol=1e-12)
    np.testing.assert_allclose(theparallelmean, theserialmean, atol=1e-12)
    np.testing.assert_allclose(theparallel, theserial, atol=1e-10)


# ==================== correlationpass dispatch ====================


def correlationpass_dispatches_on_usegpu(debug=False):
    """The dispatcher is the only place batchsize picks up its default.

    Left unset it inherits chunksize, so a caller that tunes chunksize for the CPU
    silently changes the GPU batching too - worth pinning, since the two numbers mean
    different things.
    """
    if debug:
        print("correlationpass_dispatches_on_usegpu")

    theinputs = _makegpuinputs()
    thecorrout = np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64)
    themeanval = np.zeros(theinputs["numvoxels"], dtype=np.float64)
    theargs = (
        theinputs["fmridata"],
        theinputs["referencetc"],
        _makecorrelator(theinputs["Fs"]),
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        thecorrout,
        themeanval,
    )

    with (
        patch.object(
            tide_calcsimfunc, "correlationpass_gpu", return_value=(0, [], None)
        ) as thegpu,
        patch.object(
            tide_calcsimfunc, "correlationpass_cpu", return_value=(0, [], None)
        ) as thecpu,
    ):
        tide_calcsimfunc.correlationpass(*theargs, usegpu=True, chunksize=64)
        assert thegpu.call_count == 1
        assert thecpu.call_count == 0
        # unset batchsize takes its value from chunksize
        assert thegpu.call_args.kwargs["batchsize"] == 64

        tide_calcsimfunc.correlationpass(*theargs, usegpu=True, chunksize=64, batchsize=8)
        assert thegpu.call_args.kwargs["batchsize"] == 8

        # and by default nothing goes near the GPU at all
        tide_calcsimfunc.correlationpass(*theargs)
        assert thegpu.call_count == 2
        assert thecpu.call_count == 1
        assert "batchsize" not in thecpu.call_args.kwargs


# ==================== correlationpass_gpu, real device path ====================


def _oncpudevice():
    """Force device resolution to the torch CPU device.

    The GPU implementation is otherwise unreachable on a machine without GPU
    hardware, because resolution fails and the fallback hands the work to the CPU
    implementation instead.  Pinning resolution to torch's own CPU device runs the
    GPU code - the batching, the trimming, the weighting - everywhere.

    Returns
    -------
    unittest.mock._patch
        A context manager patching _resolve_torch_device.
    """
    import torch

    return patch.object(
        tide_calcsimfunc, "_resolve_torch_device", return_value=torch.device("cpu")
    )


def correlationpass_gpu_can_be_verbose(capsys, debug=False):
    """Debug names both the reference length and the device actually chosen."""
    if debug:
        print("correlationpass_gpu_can_be_verbose")

    pytest.importorskip("torch")
    theinputs = _makegpuinputs()
    with _oncpudevice():
        _rungpupass(theinputs, _makecorrelator(theinputs["Fs"]), debug=True, fallback_to_cpu=False)

    theoutput = capsys.readouterr().out
    assert f"length {len(theinputs['referencetc'])}" in theoutput
    assert "using device: cpu" in theoutput


def correlationpass_gpu_checks_the_fft_really_ran_on_the_device(debug=False):
    """A backend can advertise itself and then quietly execute FFTs somewhere else.

    That would produce correct numbers at CPU speed while the run reports a GPU, so
    the probe checks where the result actually landed rather than trusting the
    availability flag.
    """
    if debug:
        print("correlationpass_gpu_checks_the_fft_really_ran_on_the_device")

    torch = pytest.importorskip("torch")
    theinputs = _makegpuinputs()

    def thedisplacedrfft(theinput, *theargs, **thekwargs):
        """Answer a probe with a result on a different device than it was asked for."""
        return torch.zeros(theinput.shape[-1] // 2 + 1, dtype=torch.complex64, device="meta")

    # what the CPU implementation produces, for comparison
    thecpucorrout = np.zeros(
        (theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64
    )
    tide_calcsimfunc.correlationpass_cpu(
        theinputs["fmridata"],
        theinputs["referencetc"],
        _makecorrelator(theinputs["Fs"]),
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        thecpucorrout,
        np.zeros(theinputs["numvoxels"], dtype=np.float64),
        showprogressbar=False,
    )

    with _oncpudevice(), patch.object(torch.fft, "rfft", thedisplacedrfft):
        dummy, dummy2, dummy3, thecorrout, dummy4 = _rungpupass(
            theinputs, _makecorrelator(theinputs["Fs"])
        )
    np.testing.assert_allclose(thecorrout, thecpucorrout, atol=1e-10)

    # and with the fallback refused the misplaced execution is an error, not a shrug
    with _oncpudevice(), patch.object(torch.fft, "rfft", thedisplacedrfft):
        with pytest.raises(RuntimeError, match="not executing on requested device"):
            _rungpupass(theinputs, _makecorrelator(theinputs["Fs"]), fallback_to_cpu=False)


def correlationpass_gpu_checks_the_output_width(debug=False):
    """corrout is written a whole batch at a time, so a width that disagrees with the
    lag limits has to be caught before anything is written into it."""
    if debug:
        print("correlationpass_gpu_checks_the_output_width")

    pytest.importorskip("torch")
    theinputs = _makegpuinputs()
    thewrongwidth = np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"] + 3))
    with _oncpudevice():
        with pytest.raises(ValueError, match="does not match corrout width"):
            tide_calcsimfunc.correlationpass_gpu(
                theinputs["fmridata"],
                theinputs["referencetc"],
                _makecorrelator(theinputs["Fs"]),
                theinputs["fmri_x"],
                theinputs["os_fmri_x"],
                theinputs["lagmininpts"],
                theinputs["lagmaxinpts"],
                thewrongwidth,
                np.zeros(theinputs["numvoxels"]),
                showprogressbar=False,
                fallback_to_cpu=False,
            )


def correlationpass_gpu_resamples_onto_the_oversampled_axis(debug=False):
    """When the two time axes differ the data has to be interpolated first.

    The equal-axis shortcut is what the other GPU tests take, so oversampling is the
    only way to reach the resampling branch - and it is the configuration a real
    rapidtide run uses, since oversampling is on by default.
    """
    if debug:
        print("correlationpass_gpu_resamples_onto_the_oversampled_axis")

    pytest.importorskip("torch")

    oversampfactor = 2
    numvoxels = 6
    numtimepoints = 64
    tr = 0.8
    theinit_fmri_x = np.linspace(0.0, numtimepoints, numtimepoints, endpoint=False) * tr
    theos_fmri_x = np.linspace(
        0.0, numtimepoints * oversampfactor, numtimepoints * oversampfactor, endpoint=False
    ) * (tr / oversampfactor)
    theosFs = oversampfactor / tr

    testfreq = 0.06
    thereferencetc = np.sin(2.0 * np.pi * testfreq * theos_fmri_x)
    thefmridata = np.zeros((numvoxels, numtimepoints), dtype=np.float64)
    for i in range(numvoxels):
        thefmridata[i, :] = np.sin(2.0 * np.pi * testfreq * (theinit_fmri_x - 0.2 * i))

    lagmininpts = int((10.0 * theosFs) - 0.5)
    lagmaxinpts = int((10.0 * theosFs) + 0.5)
    numcorrpoints = lagmaxinpts + lagmininpts

    thecpucorrout = np.zeros((numvoxels, numcorrpoints), dtype=np.float64)
    thecpumeanval = np.zeros(numvoxels, dtype=np.float64)
    tide_calcsimfunc.correlationpass_cpu(
        thefmridata,
        thereferencetc,
        _makecorrelator(theosFs),
        theinit_fmri_x,
        theos_fmri_x,
        lagmininpts,
        lagmaxinpts,
        thecpucorrout,
        thecpumeanval,
        oversampfactor=oversampfactor,
        showprogressbar=False,
    )

    thegpucorrout = np.zeros((numvoxels, numcorrpoints), dtype=np.float64)
    thegpumeanval = np.zeros(numvoxels, dtype=np.float64)
    with _oncpudevice():
        thecount, dummy, dummy2 = tide_calcsimfunc.correlationpass_gpu(
            thefmridata,
            thereferencetc,
            _makecorrelator(theosFs),
            theinit_fmri_x,
            theos_fmri_x,
            lagmininpts,
            lagmaxinpts,
            thegpucorrout,
            thegpumeanval,
            oversampfactor=oversampfactor,
            showprogressbar=False,
            batchsize=4,
            fallback_to_cpu=False,
        )

    assert thecount == numvoxels
    # the resampled means are what prove the interpolation ran on both sides
    np.testing.assert_allclose(thegpumeanval, thecpumeanval, atol=1e-10)
    # resolution is pinned to the torch CPU device, so this is deterministic and only
    # has to absorb the single precision the GPU implementation works in
    np.testing.assert_allclose(thegpucorrout, thecpucorrout, atol=1e-6, rtol=1e-5)


def _cpucorrelationpass(theinputs, corrweighting="None"):
    """Run the CPU implementation and hand back its outputs.

    Parameters
    ----------
    theinputs : dict
        As returned by _makegpuinputs.
    corrweighting : str, optional
        Weighting to configure the Correlator with.

    Returns
    -------
    tuple
        (voxelcount, globalmaxlist, corrscale, corrout, meanval)
    """
    thecorrout = np.zeros((theinputs["numvoxels"], theinputs["numcorrpoints"]), dtype=np.float64)
    themeanval = np.zeros(theinputs["numvoxels"], dtype=np.float64)
    thevoxels, theglobalmax, thecorrscale = tide_calcsimfunc.correlationpass_cpu(
        theinputs["fmridata"],
        theinputs["referencetc"],
        _makecorrelator(theinputs["Fs"], corrweighting=corrweighting),
        theinputs["fmri_x"],
        theinputs["os_fmri_x"],
        theinputs["lagmininpts"],
        theinputs["lagmaxinpts"],
        thecorrout,
        themeanval,
        showprogressbar=False,
    )
    return thevoxels, theglobalmax, thecorrscale, thecorrout, themeanval


def correlationpass_gpu_matches_the_cpu_for_every_weighting(debug=False):
    """The two implementations have to agree to single precision, phat included.

    phat needs three things to line up with the CPU, and each of them was wrong at
    some point: the transform runs at the next power of two above the correlation
    length (phat rescales each frequency bin, so a different transform length puts
    the weighting on a different set of bins); bins below a tenth of the largest
    magnitude are zeroed rather than clamped; and the weighted correlation is then
    rescaled so its peak matches the peak of the *unweighted* correlation.

    That last step is the one that matters downstream.  Without it the peaks still
    land in the right places, so a lag map looks fine, but the values are smaller by
    an arbitrary data-dependent factor - and lthreshval, uthreshval, the
    null-distribution significance thresholds and the refinement weighting all read
    absolute correlation strength.  So this checks the actual amplitudes, not just
    the shapes.

    Several sizes are used because the transform length is rounded up to a power of
    two: a correlation length that is already a power of two exercises different
    padding from one that is not.
    """
    if debug:
        print("correlationpass_gpu_matches_the_cpu_for_every_weighting")

    pytest.importorskip("torch")

    for theweighting in ["None", "phat"]:
        for thevoxelcount, thetimepoints in [(8, 128), (12, 160), (5, 100)]:
            theinputs = _makegpuinputs(numvoxels=thevoxelcount, numtimepoints=thetimepoints)
            (
                thecpuvoxels,
                thecpuglobalmax,
                thecpuscale,
                thecpucorrout,
                thecpumeanval,
            ) = _cpucorrelationpass(theinputs, corrweighting=theweighting)

            with _oncpudevice():
                (
                    thegpuvoxels,
                    thegpuglobalmax,
                    thegpuscale,
                    thegpucorrout,
                    thegpumeanval,
                ) = _rungpupass(
                    theinputs,
                    _makecorrelator(theinputs["Fs"], corrweighting=theweighting),
                    batchsize=4,
                    fallback_to_cpu=False,
                )

            thecontext = f"{theweighting=} {thevoxelcount=} {thetimepoints=}"
            assert thegpuvoxels == thecpuvoxels, thecontext
            assert np.array_equal(thegpuscale, thecpuscale), thecontext
            np.testing.assert_allclose(thegpumeanval, thecpumeanval, atol=1e-10)
            assert np.array_equal(
                np.asarray(thegpuglobalmax, dtype=np.int64),
                np.asarray(thecpuglobalmax, dtype=np.int64),
            ), thecontext

            # the amplitudes, not just the shapes: the GPU work is done in single
            # precision, so a few parts in ten million is as close as it can get
            np.testing.assert_allclose(
                thegpucorrout, thecpucorrout, atol=1e-6, rtol=1e-5, err_msg=thecontext
            )


def correlationpass_gpu_phat_is_not_the_same_as_unweighted(debug=False):
    """phat has to actually do something.

    Since the weighted result is rescaled back to the unweighted peak height, a phat
    correlation now peaks near the same value an unweighted one does - which would
    also be true if the weighting were quietly being skipped.  This separates those
    two cases by checking the shape away from the peak.
    """
    if debug:
        print("correlationpass_gpu_phat_is_not_the_same_as_unweighted")

    pytest.importorskip("torch")
    theinputs = _makegpuinputs()

    with _oncpudevice():
        dummy, dummy2, dummy3, theunweighted, dummy4 = _rungpupass(
            theinputs,
            _makecorrelator(theinputs["Fs"], corrweighting="None"),
            fallback_to_cpu=False,
        )
        dummy5, dummy6, dummy7, thephat, dummy8 = _rungpupass(
            theinputs,
            _makecorrelator(theinputs["Fs"], corrweighting="phat"),
            fallback_to_cpu=False,
        )

    # both are back on the same scale
    assert np.max(np.abs(theunweighted)) == pytest.approx(np.max(np.abs(thephat)), rel=1e-3)
    # but they are not the same function - phat flattens the spectrum, which sharpens
    # the peak and changes the sidelobes
    assert np.max(np.abs(thephat - theunweighted)) > 0.01


def test_calcsimfuncinternals(debug=False):
    """Entry point for the sub-tests that need no fixtures."""
    correlationpass_cpu_reports_uncollectable_garbage(debug=debug)
    correlationpass_cpu_can_run_multiprocessed(debug=debug)
    correlationpass_dispatches_on_usegpu(debug=debug)
    correlationpass_gpu_checks_the_fft_really_ran_on_the_device(debug=debug)
    correlationpass_gpu_checks_the_output_width(debug=debug)
    correlationpass_gpu_resamples_onto_the_oversampled_axis(debug=debug)
    correlationpass_gpu_matches_the_cpu_for_every_weighting(debug=debug)
    correlationpass_gpu_phat_is_not_the_same_as_unweighted(debug=debug)


def test_calcsimfuncmessages(capsys):
    """Entry point for the sub-tests that capture printed output."""
    procOneVoxelCorrelation_can_skip_the_resampling(capsys)
    correlationpass_cpu_can_be_verbose(capsys)
    correlationpass_gpu_can_be_verbose(capsys)
