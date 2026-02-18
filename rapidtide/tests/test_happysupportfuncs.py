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
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

_THISFILE = os.path.abspath(__file__)
_REPOROOT = os.path.abspath(os.path.join(os.path.dirname(_THISFILE), "..", ".."))
if _REPOROOT not in sys.path:
    sys.path.insert(0, _REPOROOT)

import rapidtide.core.signal as core_signal
import rapidtide.core.signal.happy_supportfuncs as hs
import rapidtide.happy_supportfuncs as legacy_hs


class DummyFilter:
    def __init__(self, scale=1.0):
        self.scale = scale
        self._freqs = (0.1, 0.2, 0.3, 0.4)

    def apply(self, Fs, x):
        return self.scale * np.asarray(x)

    def setfreqs(self, a, b, c, d):
        self._freqs = (a, b, c, d)

    def getfreqs(self):
        return self._freqs

    def settype(self, filtertype):
        return None


class DummyInputData:
    def __init__(self, xsize=1, ysize=1, numslices=2, timepoints=4):
        self.xsize = xsize
        self.ysize = ysize
        self.numslices = numslices
        self.timepoints = timepoints
        self._byvol = (
            np.arange(xsize * ysize * numslices * timepoints, dtype=float).reshape(
                xsize * ysize * numslices, timepoints
            )
            + 1.0
        )
        self._byslice = self._byvol.reshape(xsize * ysize, numslices, timepoints)

    def byvol(self):
        return self._byvol.copy()

    def byslice(self):
        return self._byslice.copy()

    def getdims(self):
        return self.xsize, self.ysize, self.numslices, self.timepoints

    def copyheader(self, numtimepoints=None, tr=None):
        return {"dim": [0, self.xsize, self.ysize, self.numslices, numtimepoints], "tr": tr}


def basic_routines(debug=False):
    if debug:
        print("basic_routines")

    t = np.linspace(0.0, 1.0, 8, endpoint=False)
    phase = 2.0 * np.pi * t
    assert hs.rrifromphase(t, phase) is None

    phaseimage = np.random.RandomState(1).randn(5, 5, 5)
    jump, jolt, lap = hs.phasejolt(phaseimage)
    assert jump.shape == phaseimage.shape
    assert jolt.shape == phaseimage.shape
    assert lap.shape == phaseimage.shape
    assert np.isfinite(np.sum(jump))

    scalar = hs.cardiacsig(0.0, amps=(1.0, 0.5), phases=np.array([0.0, 0.0]))
    assert np.fabs(scalar - 1.5) < 1e-10
    arr = hs.cardiacsig(np.array([0.0, np.pi]), amps=(1.0, 0.0, 0.0))
    assert arr.shape == (2,)

    x = np.array([0.0, 1.0, 2.0])
    m = np.array([1.0, 2.0, 3.0])
    assert np.fabs(hs.theCOM(x, m) - (8.0 / 6.0)) < 1e-12

    noisy = np.sin(np.linspace(0.0, 4.0 * np.pi, 101)) + 0.1 * np.random.RandomState(2).randn(101)
    smoothed = hs.savgolsmooth(noisy, smoothlen=21, polyorder=3)
    assert smoothed.shape == noisy.shape

    apen = hs.approximateentropy(np.sin(np.linspace(0.0, 2.0 * np.pi, 80)), 2, 0.2)
    assert np.isfinite(apen)
    e = hs.entropy(np.array([1.0, 0.5, 0.25]))
    assert np.isfinite(e)

    summary_keys = hs.summarizerun({}, getkeys=True)
    summary_vals = hs.summarizerun({"corrcoeff_raw2pleth": 0.9}, getkeys=False)
    assert "corrcoeff_raw2pleth" in summary_keys
    assert summary_vals.startswith("0.9")

    cmax, cmaxi, cmin, cmini = hs.circularderivs(np.array([1.0, 2.0, 1.0, 0.0]))
    assert cmaxi >= 0
    assert cmini >= 0
    assert cmax >= cmin


def cardiac_pipeline_routines(debug=False):
    if debug:
        print("cardiac_pipeline_routines")

    normdata = np.ones((4, 2, 6), dtype=float)
    estweights = np.ones((4, 2), dtype=float)
    hs._validate_cardiacfromimage_inputs(normdata, estweights, 2, 6, 2.0)
    with pytest.raises(ValueError):
        hs._validate_cardiacfromimage_inputs(normdata, estweights, 2, 0, 2.0)

    appflips, theseweights = hs._prepare_weights(estweights, None, arteriesonly=False, fliparteries=False)
    assert appflips.shape == estweights.shape
    assert np.allclose(theseweights, estweights)

    af2 = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    appflips2, theseweights2 = hs._prepare_weights(
        estweights,
        af2.copy(),
        arteriesonly=True,
        fliparteries=True,
    )
    assert appflips2.shape == estweights.shape
    assert theseweights2.shape == estweights.shape

    normdata2 = np.random.RandomState(3).randn(6, 2, 8)
    w2 = np.ones((6, 2))
    hrtc, cyc, slicenorms = hs._compute_slice_averages(
        normdata2,
        w2,
        numslices=2,
        timepoints=8,
        numsteps=2,
        sliceoffsets=np.array([0, 1], dtype=int),
        signal_sign=1.0,
        madnorm=True,
        usemask=True,
        multiplicative=False,
        verbose=False,
    )
    assert hrtc.shape == (16,)
    assert cyc.shape == (2,)
    assert slicenorms.shape == (2,)

    signal, normfac = hs._normalize_and_filter_signal(DummyFilter(scale=2.0), 2.0, hrtc, slicenorms)
    assert signal.shape == hrtc.shape
    assert np.isfinite(normfac)

    hirescard, cardnorm, hiresresp, respnorm = hs._extract_physiological_signals(
        hrtc,
        2.0,
        DummyFilter(scale=1.0),
        DummyFilter(scale=0.5),
        slicenorms,
    )
    assert hirescard.shape == hrtc.shape
    assert hiresresp.shape == hrtc.shape
    assert np.isfinite(cardnorm)
    assert np.isfinite(respnorm)

    rng = np.random.RandomState(4)
    normdata3 = rng.randn(10, 2, 8)
    normdata3[:, 1, :] *= 3.0
    estweights3 = np.ones((10, 2))
    config = hs.CardiacExtractionConfig(verbose=False, madnorm=True, usemask=True, multiplicative=True)
    # Keep this as a unit test of happy_supportfuncs control flow: short synthetic
    # vectors can violate padding requirements in the underlying filter implementation.
    with patch("rapidtide.core.signal.happy_supportfuncs.tide_filt.harmonicnotchfilter", side_effect=lambda x, *args, **kwargs: x):
        result = hs.cardiacfromimage(
            normdata3,
            estweights3,
            numslices=2,
            timepoints=8,
            tr=2.0,
            slicetimes=np.array([0.0, 1.0]),
            cardprefilter=DummyFilter(scale=1.0),
            respprefilter=DummyFilter(scale=0.8),
            config=config,
            appflips_byslice=None,
        )
    assert isinstance(result, hs.CardiacExtractionResult)
    assert result.hirescardtc.shape == (16,)
    unpacked = tuple(result)
    assert len(unpacked) == 9


def frequency_routines(debug=False):
    if debug:
        print("frequency_routines")

    class DummyArbFilter:
        def __init__(self, filtertype="arb"):
            self.count = 0

        def setfreqs(self, a, b, c, d):
            return None

        def apply(self, Fs, x):
            self.count += 1
            return np.asarray(x)

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_filt.NoncausalFilter", DummyArbFilter):
        x = np.ones(64)
        y = hs.getperiodic(x, Fs=10.0, fundfreq=3.0, ncomps=5, width=0.4, debug=False)
        assert y.shape == x.shape
        assert np.isfinite(np.sum(y))

    fs = 50.0
    t = np.linspace(0.0, 20.0, int(20.0 * fs), endpoint=False)
    waveform = np.sin(2.0 * np.pi * 1.2 * t)
    peakfreq = hs.getcardcoeffs(waveform, slicesamplerate=fs, minhr=50.0, maxhr=100.0, smoothlen=51)
    assert np.fabs(peakfreq - 1.2) < 0.25


def detrend_normalize_routines(debug=False):
    if debug:
        print("detrend_normalize_routines")

    vox = np.linspace(0.0, 1.0, 40) + 0.05 * np.random.RandomState(5).randn(40)
    idx, dvox = hs._procOneVoxelDetrend(3, (vox,), detrendorder=1, demean=True)
    assert idx == 3
    assert dvox.shape == vox.shape

    data = np.vstack([np.arange(10, dtype=float), np.arange(10, dtype=float) + 1.0])
    packed = hs._packDetrendvoxeldata(1, [data])
    assert len(packed) == 1
    assert packed[0].shape == (10,)

    voxelproducts = [np.zeros((2, 10), dtype=float)]
    hs._unpackDetrendvoxeldata((1, np.arange(10, dtype=float)), voxelproducts)
    assert np.allclose(voxelproducts[0][1, :], np.arange(10, dtype=float))

    fmri = np.random.RandomState(6).randn(5, 30) + 10.0
    valid = np.array([0, 1, 2, 4], dtype=int)
    timings = []
    norm, demean, means, meds, mads = hs.normalizevoxels(
        fmri.copy(),
        detrendorder=1,
        validvoxels=valid,
        time=__import__("time"),
        timings=timings,
        mpcode=False,
        showprogressbar=False,
        debug=False,
    )
    assert norm.shape == fmri.shape
    assert demean.shape == fmri.shape
    assert means.shape == (5,)
    assert meds.shape == (5,)
    assert mads.shape == (5,)
    assert len(timings) >= 1

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_genericmultiproc.run_multiproc", return_value=fmri.shape[0]):
        timings2 = []
        hs.normalizevoxels(
            fmri.copy(),
            detrendorder=1,
            validvoxels=valid,
            time=__import__("time"),
            timings=timings2,
            mpcode=True,
            nprocs=1,
            showprogressbar=False,
            debug=False,
        )
    assert len(timings2) >= 1


def physio_quality_routines(debug=False):
    if debug:
        print("physio_quality_routines")

    fs = 25.0
    t = np.linspace(0.0, 20.0, int(20.0 * fs), endpoint=False)
    waveform = np.sin(2.0 * np.pi * 1.2 * t) + 0.05 * np.random.RandomState(7).randn(len(t))
    with patch("rapidtide.core.signal.happy_supportfuncs.tide_filt.NoncausalFilter", return_value=DummyFilter(scale=1.0)):
        filt, norm, env, envmean = hs.cleanphysio(
            fs,
            waveform,
            cutoff=0.3,
            thresh=0.2,
            iscardiac=True,
            debug=False,
        )
    assert filt.shape == waveform.shape
    assert norm.shape == waveform.shape
    assert env.shape == waveform.shape
    assert np.isfinite(envmean)

    infodict = {}
    bad_mad, thresh_mad = hs.findbadpts(
        waveform,
        "card",
        "/tmp/unused",
        samplerate=fs,
        infodict=infodict,
        thetype="mad",
        outputlevel=0,
        debug=False,
    )
    assert bad_mad.shape == waveform.shape
    assert np.isscalar(thresh_mad)
    bad_frac, thresh_frac = hs.findbadpts(
        waveform,
        "resp",
        "/tmp/unused",
        samplerate=fs,
        infodict=infodict,
        thetype="fracval",
        outputlevel=0,
        debug=False,
    )
    assert bad_frac.shape == waveform.shape
    assert len(thresh_frac) == 2
    with pytest.raises(ValueError):
        hs.findbadpts(
            waveform,
            "bad",
            "/tmp/unused",
            samplerate=fs,
            infodict={},
            thetype="not_real",
            outputlevel=0,
            debug=False,
        )

    qdict = {}
    hs.calcplethquality(
        waveform,
        Fs=fs,
        infodict=qdict,
        suffix="_x",
        outputroot="/tmp/unused",
        outputlevel=0,
        debug=False,
    )
    assert "S_sqi_mean_x" in qdict
    assert "K_sqi_mean_x" in qdict
    assert "E_sqi_mean_x" in qdict

    with patch(
        "rapidtide.core.signal.happy_supportfuncs.tide_io.readvectorsfromtextfile",
        return_value=(100.0, 0.0, None, np.sin(2.0 * np.pi * 1.5 * np.linspace(0.0, 2.0, 200)), None, None),
    ), patch(
        "rapidtide.core.signal.happy_supportfuncs.cleanphysio",
        return_value=(
            np.sin(2.0 * np.pi * 1.5 * np.linspace(0.0, 2.0, 200)),
            np.sin(2.0 * np.pi * 1.5 * np.linspace(0.0, 2.0, 200)),
            np.ones(200),
            1.0,
        ),
    ):
        timings = []
        slicetimeaxis = np.linspace(0.0, 1.5, 80, endpoint=False)
        wave_slice, wave_std, inpfreq, npts = hs.getphysiofile(
            waveformfile="ignored.txt",
            inputfreq=-100.0,
            inputstart=None,
            slicetimeaxis=slicetimeaxis,
            stdfreq=25.0,
            stdpoints=60,
            envcutoff=0.3,
            envthresh=0.2,
            timings=timings,
            outputroot="/tmp/unused",
            outputlevel=0,
            iscardiac=True,
            debug=False,
        )
    assert wave_slice.shape == slicetimeaxis.shape
    assert wave_std.shape == (60,)
    assert np.fabs(inpfreq - 100.0) < 1e-12
    assert npts == 200
    assert len(timings) >= 2

    with patch(
        "rapidtide.core.signal.happy_supportfuncs.tide_io.readfromnifti",
        return_value=(None, np.ones((2, 2, 2)), {"dim": [0, 2, 2, 2, 1]}, [0, 2, 2, 2, 1], None),
    ), patch("rapidtide.core.signal.happy_supportfuncs.tide_io.parseniftidims", return_value=(2, 2, 2, 1)), patch(
        "rapidtide.core.signal.happy_supportfuncs.tide_io.checkspacematch", return_value=True
    ):
        mask = hs.readextmask("mask.nii.gz", {"dim": [0, 2, 2, 2, 1]}, 2, 2, 2, debug=False)
    assert mask.shape == (2, 2, 2)

    with patch(
        "rapidtide.core.signal.happy_supportfuncs.tide_io.readfromnifti",
        return_value=(None, np.ones((2, 2, 2, 2)), {"dim": [0, 2, 2, 2, 2]}, [0, 2, 2, 2, 2], None),
    ), patch("rapidtide.core.signal.happy_supportfuncs.tide_io.parseniftidims", return_value=(2, 2, 2, 2)), patch(
        "rapidtide.core.signal.happy_supportfuncs.tide_io.checkspacematch", return_value=True
    ):
        with pytest.raises(ValueError):
            hs.readextmask("mask4d.nii.gz", {"dim": [0, 2, 2, 2, 1]}, 2, 2, 2, debug=False)

    with patch(
        "rapidtide.core.signal.happy_supportfuncs.tide_filt.NoncausalFilter",
        return_value=DummyFilter(scale=1.0),
    ), patch(
        "rapidtide.core.signal.happy_supportfuncs.tide_corr.fastcorrelate",
        return_value=np.array([0.0, 0.3, 1.0, 0.2, 0.0]),
    ), patch(
        "rapidtide.core.signal.happy_supportfuncs.tide_fit.findmaxlag_gauss",
        return_value=(2, 0.1, 0.95, 0.2, 1, "ok", 1, 3),
    ):
        maxval, maxdelay, failreason = hs.checkcardmatch(
            reference=np.sin(2.0 * np.pi * 1.0 * np.linspace(0.0, 2.0, 100)),
            candidate=np.sin(2.0 * np.pi * 1.0 * np.linspace(0.0, 2.0, 100)),
            samplerate=20.0,
            refine=True,
            zeropadding=0,
            debug=False,
        )
    assert np.fabs(maxval - 0.95) < 1e-12
    assert np.fabs(maxdelay - 0.1) < 1e-12
    assert failreason == "ok"


def projection_helpers_routines(debug=False):
    if debug:
        print("projection_helpers_routines")

    destphases = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    srcphases = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    waveform = np.sin(srcphases)
    procpoints = np.arange(len(srcphases), dtype=int)

    def fake_congrid(dest, src, val, bins, kernel=None, cache=True, cyclic=True, debug=False):
        idx = int(np.argmin(np.abs(dest - src)))
        return np.array([1.0]), np.array([1.0]), np.array([idx], dtype=int)

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_resample.congrid", side_effect=fake_congrid):
        cycavg, weights = hs.cardiaccycleaverage(
            srcphases,
            destphases,
            waveform,
            procpoints,
            congridbins=4,
            gridkernel="kaiser",
            centric=True,
            cache=True,
            cyclic=True,
        )
    assert cycavg.shape == destphases.shape
    assert weights.shape == destphases.shape
    assert np.max(weights) > 0.0

    nvox, nslices, ntr, destpoints = 3, 1, 4, 4
    validlocslist = [np.array([0, 1], dtype=int)]
    proctrs = np.array([0, 1, 2], dtype=int)
    demean = np.random.RandomState(8).randn(nvox, nslices, ntr)
    fmri = np.random.RandomState(9).randn(nvox, nslices, ntr)
    outphases = np.linspace(0.0, 2.0 * np.pi, destpoints, endpoint=False)
    cardphasevals = np.zeros((nslices, ntr))
    weights_byslice = np.zeros((nvox, nslices, destpoints), dtype=float)
    cine_byslice = np.zeros((nvox, nslices, destpoints), dtype=float)
    rawapp_byslice = np.zeros((nvox, nslices, destpoints), dtype=float)

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_resample.congrid", side_effect=fake_congrid):
        ret = hs._procOnePhaseProject(
            0,
            (
                validlocslist,
                proctrs,
                demean,
                fmri,
                outphases,
                cardphasevals,
                4,
                "kaiser",
                weights_byslice.copy(),
                cine_byslice.copy(),
                destpoints,
                rawapp_byslice.copy(),
            ),
            cache=True,
            debug=False,
        )
    assert ret[0] == 0
    assert ret[1].shape == (nvox, destpoints)

    packed_phase = hs._packslicedataPhaseProject(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert len(packed_phase) == 12

    vp = [np.zeros((3, 1, 4)), np.zeros((3, 1, 4)), np.zeros((3, 1, 4))]
    hs._unpackslicedataPhaseProject((0, np.ones((3, 4)), 2.0 * np.ones((3, 4)), 3.0 * np.ones((3, 4)), np.array([0, 2], dtype=int)), vp)
    assert np.all(vp[0][[0, 2], 0, :] == 1.0)
    assert np.all(vp[1][[0, 2], 0, :] == 2.0)
    assert np.all(vp[2][[0, 2], 0, :] == 3.0)

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_resample.congrid", side_effect=fake_congrid) as cg:
        hs.preloadcongrid(np.linspace(0.0, 1.0, 8), congridbins=4, gridkernel="kaiser", cyclic=True, debug=False)
        assert cg.call_count > 1000

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_resample.congrid", side_effect=fake_congrid):
        hs.phaseprojectpass(
            numslices=1,
            demeandata_byslice=demean,
            fmri_data_byslice=fmri,
            validlocslist=validlocslist,
            proctrs=proctrs,
            weights_byslice=weights_byslice,
            cine_byslice=cine_byslice,
            rawapp_byslice=rawapp_byslice,
            outphases=outphases,
            cardphasevals=cardphasevals,
            congridbins=4,
            gridkernel="kaiser",
            destpoints=destpoints,
            mpcode=False,
            showprogressbar=False,
            cache=True,
            debug=False,
        )
    assert np.isfinite(np.sum(rawapp_byslice))

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_genericmultiproc.run_multiproc", return_value=1):
        hs.phaseprojectpass(
            numslices=1,
            demeandata_byslice=demean,
            fmri_data_byslice=fmri,
            validlocslist=validlocslist,
            proctrs=proctrs,
            weights_byslice=weights_byslice.copy(),
            cine_byslice=cine_byslice.copy(),
            rawapp_byslice=rawapp_byslice.copy(),
            outphases=outphases,
            cardphasevals=cardphasevals,
            congridbins=4,
            gridkernel="kaiser",
            destpoints=destpoints,
            mpcode=True,
            nprocs=1,
            showprogressbar=False,
            cache=True,
            debug=False,
        )

    smooth_filter = DummyFilter(scale=1.0)
    derivs = np.zeros((nvox, nslices, destpoints), dtype=float)
    ret2 = hs._procOneSliceSmoothing(
        0,
        (validlocslist, rawapp_byslice.copy(), smooth_filter, 1.0, derivs.copy()),
        debug=False,
    )
    assert ret2[0] == 0
    assert ret2[1].shape == (nvox, destpoints)

    packed_sm = hs._packslicedataSliceSmoothing(0, [1, 2, 3, 4, 5, 6])
    assert len(packed_sm) == 5

    vp2 = [np.zeros((3, 1, 4)), np.zeros((3, 1, 4))]
    hs._unpackslicedataSliceSmoothing((0, np.ones((3, 4)), 2.0 * np.ones((3, 4))), vp2)
    assert np.all(vp2[0][:, 0, :] == 1.0)
    assert np.all(vp2[1][:, 0, :] == 2.0)

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_genericmultiproc.run_multiproc", return_value=1):
        hs.tcsmoothingpass(
            numslices=1,
            validlocslist=validlocslist,
            rawapp_byslice=rawapp_byslice.copy(),
            appsmoothingfilter=smooth_filter,
            phaseFs=1.0,
            derivatives_byslice=derivs.copy(),
            nprocs=1,
            alwaysmultiproc=False,
            showprogressbar=False,
            debug=False,
        )


def high_level_routines(debug=False):
    if debug:
        print("high_level_routines")

    input_data = DummyInputData(xsize=1, ysize=2, numslices=2, timepoints=4)
    numspatiallocs = input_data.xsize * input_data.ysize
    destpoints = 4
    demean = np.random.RandomState(10).randn(numspatiallocs, 2, 4)
    means = np.ones((numspatiallocs, 2)) * 2.0
    rawapp = np.zeros((numspatiallocs, 2, destpoints))
    app = np.zeros_like(rawapp)
    normapp = np.zeros_like(rawapp)
    weights = np.zeros_like(rawapp)
    cine = np.zeros_like(rawapp)
    projmask = np.ones((numspatiallocs, 2))
    derivs = np.zeros((numspatiallocs, 2, destpoints))
    derivs[:, :, 0] = 1.0
    derivs[:, :, 2] = -1.0
    proctrs = np.array([0, 1], dtype=int)
    sliceoffsets = np.array([0, 1], dtype=int)
    cardphasevals = np.zeros((2, 4))
    outphases = np.linspace(0.0, 2.0 * np.pi, destpoints, endpoint=False)
    corrfunc = np.zeros((numspatiallocs, 2, 3))
    waveamp = np.zeros((numspatiallocs, 2))
    wavedelay = np.zeros((numspatiallocs, 2))
    wavedelaycom = np.zeros((numspatiallocs, 2))
    corrected = np.zeros_like(rawapp)
    args = SimpleNamespace(
        verbose=False,
        congridbins=4,
        gridkernel="kaiser",
        destpoints=destpoints,
        congridcache=True,
        mpphaseproject=False,
        nprocs=1,
        showprogressbar=False,
        smoothapp=False,
        fliparteries=True,
        doaliasedcorrelation=False,
    )
    appflips = hs.phaseproject(
        input_data=input_data,
        demeandata_byslice=demean,
        means_byslice=means,
        rawapp_byslice=rawapp,
        app_byslice=app,
        normapp_byslice=normapp,
        weights_byslice=weights,
        cine_byslice=cine,
        projmask_byslice=projmask,
        derivatives_byslice=derivs,
        proctrs=proctrs,
        thispass=0,
        args=args,
        sliceoffsets=sliceoffsets,
        cardphasevals=cardphasevals,
        outphases=outphases,
        appsmoothingfilter=DummyFilter(scale=1.0),
        phaseFs=1.0,
        thecorrfunc_byslice=corrfunc,
        waveamp_byslice=waveamp,
        wavedelay_byslice=wavedelay,
        wavedelayCOM_byslice=wavedelaycom,
        corrected_rawapp_byslice=corrected,
        corrstartloc=0,
        correndloc=2,
        thealiasedcorrx=np.array([-1.0, 0.0, 1.0]),
        theAliasedCorrelator=None,
    )
    assert appflips.shape == (numspatiallocs, 2)

    app2d = app.reshape((numspatiallocs * input_data.numslices, destpoints))
    normapp2d = normapp.reshape((numspatiallocs * input_data.numslices, destpoints))
    vessel_valid = np.array([0, 1], dtype=int)
    with patch("rapidtide.core.signal.happy_supportfuncs.tide_util.logmem", return_value=None):
        hardvesselthresh, softvesselthresh = hs.findvessels(
            app=app2d,
            normapp=normapp2d,
            validlocs=vessel_valid,
            numspatiallocs=numspatiallocs * input_data.numslices,
            outputroot="/tmp/unused",
            unnormvesselmap=True,
            destpoints=destpoints,
            softvesselfrac=0.5,
            histlen=10,
            outputlevel=0,
            debug=False,
        )
    assert np.isfinite(hardvesselthresh)
    assert np.isfinite(softvesselthresh)

    with patch("rapidtide.core.signal.happy_supportfuncs.tide_io.savetonifti") as save_patch:
        hs.upsampleimage(
            input_data=DummyInputData(xsize=1, ysize=1, numslices=2, timepoints=3),
            numsteps=2,
            sliceoffsets=np.array([0, 1], dtype=int),
            slicesamplerate=4.0,
            outputroot="/tmp/upsampletest",
        )
        assert save_patch.call_count == 1

    def fake_phaseprojectpass(
        numslices,
        demeandata_byslice,
        fmri_data_byslice,
        validlocslist,
        proctrs,
        weights_byslice,
        cine_byslice,
        rawapp_byslice,
        outphases,
        cardphasevals,
        congridbins,
        gridkernel,
        destpoints,
        **kwargs,
    ):
        base = 1.0 + float(np.sum(proctrs))
        for s in range(numslices):
            for v in validlocslist[s]:
                rawapp_byslice[v, s, :] = base + np.arange(destpoints, dtype=float)

    with patch("rapidtide.core.signal.happy_supportfuncs.phaseprojectpass", side_effect=fake_phaseprojectpass):
        wmap = hs.wrightmap(
            input_data=input_data,
            demeandata_byslice=demean,
            rawapp_byslice=np.zeros((numspatiallocs, 2, destpoints)),
            projmask_byslice=projmask,
            outphases=outphases,
            cardphasevals=cardphasevals,
            proctrs=np.array([0, 1, 2, 3], dtype=int),
            congridbins=4,
            gridkernel="kaiser",
            destpoints=destpoints,
            iterations=2,
            nprocs=1,
            verbose=False,
            debug=False,
        )
    assert wmap.shape == (input_data.xsize, input_data.ysize, input_data.numslices)

    video = np.random.RandomState(11).randn(3, 3, 3, 2)
    projmask3d = np.ones((3, 3, 3), dtype=int)
    flowhdr = {"dim": [0, 3, 3, 3, 2]}
    with patch("rapidtide.core.signal.happy_supportfuncs.tide_io.savetonifti") as savepatch:
        flow = hs.calc_3d_optical_flow(
            video=video,
            projmask=projmask3d,
            flowhdr=flowhdr,
            outputroot="/tmp/flowtest",
            window_size=3,
            debug=False,
        )
        assert savepatch.call_count == 4
    assert flow.shape == (3, 3, 3, 2, 3)


def test_happysupportfuncs(debug=False, local=False):
    np.random.seed(12345)
    basic_routines(debug=debug)
    cardiac_pipeline_routines(debug=debug)
    frequency_routines(debug=debug)
    detrend_normalize_routines(debug=debug)
    physio_quality_routines(debug=debug)
    projection_helpers_routines(debug=debug)
    high_level_routines(debug=debug)
    assert callable(legacy_hs.cardiacfromimage)
    assert callable(core_signal.cardiacfromimage)
    assert callable(core_signal.cleanphysio)
    assert callable(core_signal.findvessels)


if __name__ == "__main__":
    test_happysupportfuncs(debug=True, local=True)
