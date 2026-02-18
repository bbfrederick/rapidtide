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
from unittest.mock import patch

import numpy as np
import pytest

_THISFILE = os.path.abspath(__file__)
_REPOROOT = os.path.abspath(os.path.join(os.path.dirname(_THISFILE), "..", ".."))
if _REPOROOT not in sys.path:
    sys.path.insert(0, _REPOROOT)

from rapidtide.core.io.mask_io import (
    saveregionaltimeseries as core_saveregionaltimeseries,
)
from rapidtide.core.masks.mask_ops import getmaskset as core_getmaskset
from rapidtide.core.masks.mask_ops import makeepimask as core_makeepimask
from rapidtide.core.masks.mask_ops import maketmask as core_maketmask
from rapidtide.core.masks.mask_ops import readamask as core_readamask
from rapidtide.core.masks.mask_ops import resampmask as core_resampmask
from rapidtide.core.masks.region_signal import getregionsignal as core_getregionsignal


class DummyFilter:
    def apply(self, Fs, data):
        return np.asarray(data) + 1.0


class DummyPCAFit:
    def __init__(self, n_components=1):
        self.components_ = np.ones((n_components, 4))
        self.explained_variance_ratio_ = np.ones(n_components) / float(n_components)
        self.n_samples_ = 4
        self.n_features_in_ = 4

    def transform(self, x):
        return np.asarray(x)[:, : len(self.components_)]

    def inverse_transform(self, t):
        return np.tile(np.mean(np.asarray(t), axis=1, keepdims=True), (1, 4))


class DummyPCA:
    def __init__(self, n_components=0.8):
        self.n_components = n_components

    def fit(self, x):
        if self.n_components == "mle":
            raise ValueError("forced mle failure")
        ncomp = 1 if isinstance(self.n_components, float) else int(self.n_components)
        return DummyPCAFit(n_components=max(1, ncomp))


def resampmask_and_makeepimask_tests(debug=False):
    if debug:
        print("resampmask_and_makeepimask_tests")

    m = np.array([[0, 1], [1, 0]], dtype=np.uint16)
    out = core_resampmask(m, 2.0)
    assert np.array_equal(out, m)

    with patch("rapidtide.core.masks.mask_ops.masking.compute_epi_mask", return_value="epi_mask") as p_mask:
        ret = core_makeepimask("dummy_nim")
        assert ret == "epi_mask"
        p_mask.assert_called_once_with("dummy_nim")


def maketmask_tests(debug=False):
    if debug:
        print("maketmask_tests")

    timeaxis = np.arange(0.0, 10.0, 1.0)
    base = np.zeros_like(timeaxis)

    with patch("rapidtide.core.masks.mask_ops.tide_io.readvecs", return_value=np.array([[1, 0, 2, 0, 0, 1, 0, 0, 1, 1]])):
        out1 = core_maketmask("dummy.txt", timeaxis, base.copy())
    assert np.array_equal(out1, np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1], dtype=float))

    with patch("rapidtide.core.masks.mask_ops.tide_io.readvecs", return_value=np.array([[1.0, 5.0], [2.0, 2.0]])):
        out2 = core_maketmask("dummy.txt", timeaxis, base.copy())
    assert np.sum(out2) > 0

    with patch("rapidtide.core.masks.mask_ops.tide_io.readvecs", return_value=np.array([[1, 0, 2]])):
        with pytest.raises(ValueError):
            core_maketmask("dummy.txt", timeaxis, base.copy())


def readamask_tests(debug=False):
    if debug:
        print("readamask_tests")

    with patch("rapidtide.core.masks.mask_ops.tide_io.readvecs", return_value=np.array([1, 0, 2, 3])):
        out = core_readamask("dummy.txt", nim_hdr=None, xsize=4, istext=True)
        assert out.dtype == np.uint16
        assert np.array_equal(out, np.array([1, 0, 1, 1], dtype=np.uint16))

    with patch("rapidtide.core.masks.mask_ops.tide_io.readvecs", return_value=np.array([1, 0, 2])):
        with pytest.raises(ValueError):
            core_readamask("dummy.txt", nim_hdr=None, xsize=4, istext=True)

    with patch(
        "rapidtide.core.masks.mask_ops.tide_io.readfromnifti",
        return_value=(None, np.array([[0.2, 0.6], [0.7, 0.1]]), {"hdr": 1}, [0], [0]),
    ), patch("rapidtide.core.masks.mask_ops.tide_io.checkspacematch", return_value=True):
        out2 = core_readamask("mask.nii.gz", nim_hdr={"hdr": 2}, xsize=4, thresh=0.5)
        assert np.array_equal(out2, np.array([[0, 1], [1, 0]], dtype=np.uint16))

    with patch(
        "rapidtide.core.masks.mask_ops.tide_io.readfromnifti",
        return_value=(None, np.array([[1, 2], [3, 4]]), {"hdr": 1}, [0], [0]),
    ), patch("rapidtide.core.masks.mask_ops.tide_io.checkspacematch", return_value=True):
        out3 = core_readamask(
            "mask.nii.gz", nim_hdr={"hdr": 2}, xsize=4, thresh=None, valslist=[2, 4]
        )
        assert np.array_equal(out3, np.array([[0, 1], [0, 1]], dtype=np.uint16))

    with patch(
        "rapidtide.core.masks.mask_ops.tide_io.readfromnifti",
        return_value=(None, np.ones((2, 2)), {"hdr": 1}, [0], [0]),
    ), patch("rapidtide.core.masks.mask_ops.tide_io.checkspacematch", return_value=False):
        with pytest.raises(ValueError):
            core_readamask("mask.nii.gz", nim_hdr={"hdr": 2}, xsize=4, thresh=None)


def getmaskset_tests(debug=False):
    if debug:
        print("getmaskset_tests")

    include = np.array([1, 1, 0, 0], dtype=np.uint16)
    exclude = np.array([0, 1, 0, 0], dtype=np.uint16)
    extra = np.array([1, 0, 1, 0], dtype=np.uint16)
    masks = [include, exclude, extra]
    readamask_fn = lambda *args, **kwargs: masks.pop(0)
    inc, exc, ext = core_getmaskset(
        "test",
        "include.nii.gz",
        [1],
        "exclude.nii.gz",
        [1],
        datahdr={},
        numspatiallocs=4,
        extramask="extra.nii.gz",
        debug=False,
        readamask_fn=readamask_fn,
    )
    assert np.array_equal(inc, include)
    assert np.array_equal(exc, exclude)
    assert np.array_equal(ext, extra)

    with pytest.raises(ValueError):
        core_getmaskset(
            "test",
            "include.nii.gz",
            [1],
            None,
            None,
            datahdr={},
            numspatiallocs=4,
            readamask_fn=lambda *args, **kwargs: np.zeros(4, dtype=np.uint16),
        )

    with pytest.raises(ValueError):
        core_getmaskset(
            "test",
            None,
            None,
            "exclude.nii.gz",
            [1],
            datahdr={},
            numspatiallocs=4,
            readamask_fn=lambda *args, **kwargs: np.ones(4, dtype=np.uint16),
        )

    with pytest.raises(ValueError):
        masks = [
            np.array([1, 0, 0, 0], dtype=np.uint16),
            np.array([1, 1, 1, 1], dtype=np.uint16),
        ]
        core_getmaskset(
            "test",
            "include.nii.gz",
            [1],
            "exclude.nii.gz",
            [1],
            datahdr={},
            numspatiallocs=4,
            readamask_fn=lambda *args, **kwargs: masks.pop(0),
        )


def getregionsignal_tests(debug=False):
    if debug:
        print("getregionsignal_tests")

    indata = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [10.0, 10.0, 10.0, 10.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    includemask = np.array([1, 1, 0, 1], dtype=np.uint16)
    excludemask = np.array([0, 0, 0, 1], dtype=np.uint16)

    s_sum, m_sum = core_getregionsignal(
        indata, includemask=includemask, excludemask=excludemask, signalgenmethod="sum"
    )
    assert s_sum.shape == (4,)
    assert np.all(np.isfinite(s_sum))
    assert np.array_equal(m_sum, np.array([1, 1, 0, 0], dtype=np.uint16))

    s_ms, _ = core_getregionsignal(
        indata, includemask=includemask, excludemask=excludemask, signalgenmethod="meanscale"
    )
    assert s_ms.shape == (4,)

    s_pca, _ = core_getregionsignal(indata, signalgenmethod="pca", pca_class=DummyPCA)
    assert s_pca.shape == (4,)
    s_pca_mle, _ = core_getregionsignal(
        indata,
        signalgenmethod="pca",
        pcacomponents="mle",
        pca_class=DummyPCA,
    )
    assert s_pca_mle.shape == (4,)

    np.random.seed(123)
    s_rand, _ = core_getregionsignal(indata, signalgenmethod="random")
    assert s_rand.shape == (4,)

    s_filt, _ = core_getregionsignal(indata, signalgenmethod="sum", filter=DummyFilter())
    assert s_filt.shape == (4,)

    with pytest.raises(ValueError):
        core_getregionsignal(indata, signalgenmethod="nonesuch")


def saveregionaltimeseries_tests(debug=False):
    if debug:
        print("saveregionaltimeseries_tests")

    fake_tc = np.array([0.1, 0.2, 0.3, 0.4])
    fake_mask = np.array([1, 1, 0, 0], dtype=np.uint16)
    p_get = lambda *args, **kwargs: (fake_tc, fake_mask)
    with patch("rapidtide.core.io.mask_io.tide_io.writebidstsv") as p_write:
        out_tc, out_mask = core_saveregionaltimeseries(
            tcdesc="global",
            tcname="gms",
            fmridata=np.ones((4, 4)),
            includemask=np.array([1, 1, 1, 1], dtype=np.uint16),
            fmrifreq=0.5,
            outputname="sub-01_task-rest",
            filter=None,
            initfile=True,
            excludemask=None,
            filedesc="regional",
            suffix="_test",
            signalgenmethod="sum",
            pcacomponents=0.8,
            debug=False,
            getregionsignal_fn=p_get,
            writebidstsv_fn=p_write,
        )
        p_write.assert_called_once()
    assert np.array_equal(out_tc, fake_tc)
    assert np.array_equal(out_mask, fake_mask)


def test_maskutil(debug=False, displayplots=False):
    np.random.seed(12345)
    resampmask_and_makeepimask_tests(debug=debug)
    maketmask_tests(debug=debug)
    readamask_tests(debug=debug)
    getmaskset_tests(debug=debug)
    getregionsignal_tests(debug=debug)
    saveregionaltimeseries_tests(debug=debug)
    assert callable(core_resampmask)
    assert callable(core_makeepimask)
    assert callable(core_maketmask)
    assert callable(core_readamask)
    assert callable(core_getmaskset)
    assert callable(core_getregionsignal)
    assert callable(core_saveregionaltimeseries)


if __name__ == "__main__":
    test_maskutil(debug=True, displayplots=False)
