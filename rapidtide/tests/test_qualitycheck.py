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

import rapidtide.qualitycheck as tide_quality


class DummyOverlay:
    def __init__(self, data, xsize=1.0, ysize=1.0, zsize=1.0):
        self.data = data
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = zsize
        self.funcmask = None

    def setFuncMask(self, themask):
        self.funcmask = themask

    def updateStats(self):
        return None

    def summarize(self):
        return None


class DummyRegressor:
    def __init__(self, specaxis, specdata):
        self.specaxis = specaxis
        self.specdata = specdata
        self.kurtosis = 0.5
        self.kurtosis_z = 0.1
        self.kurtosis_p = 0.9
        self.skewness = 0.2
        self.skewness_z = 0.05
        self.skewness_p = 0.8


class DummyDataset:
    def __init__(
        self,
        name,
        rootname,
        anatname=None,
        geommaskname=None,
        graymaskspec=None,
        whitemaskspec=None,
        userise=False,
        usecorrout=False,
        useatlas=False,
        forcetr=False,
        forceoffset=False,
        offsettime=0.0,
        verbose=False,
        init_LUT=False,
    ):
        # Keep masks large enough that post-erosion voxel counts still satisfy
        # SciPy skew/kurtosis test sample-size requirements.
        shape = (8, 8, 8)
        lagmask = np.ones(shape, dtype=int)
        lagtimes = np.random.RandomState(1).randn(*shape)
        lagstrengths = np.clip(np.random.RandomState(2).rand(*shape), 0.0, 1.0)
        mtt = np.abs(np.random.RandomState(3).randn(*shape))
        graymask = np.zeros(shape, dtype=int)
        graymask[:4, :, :] = 1
        whitemask = np.zeros(shape, dtype=int)
        whitemask[4:, :, :] = 1

        self.numberofpasses = 2
        self.regressorfilterlimits = [0.2, 0.8]
        self.regressorsimcalclimits = [0.1, 0.9]
        self.regressors = {
            "pass1": DummyRegressor(
                np.linspace(0.0, 1.0, 200), np.random.RandomState(4).rand(200)
            ),
            "pass2": DummyRegressor(
                np.linspace(0.0, 1.0, 200), np.random.RandomState(5).rand(200)
            ),
        }
        self.overlays = {
            "lagmask": DummyOverlay(lagmask),
            "refinemask": DummyOverlay(lagmask),
            "meanmask": DummyOverlay(lagmask),
            "preselectmask": DummyOverlay(lagmask),
            "p_lt_0p050_mask": DummyOverlay(lagmask),
            "p_lt_0p010_mask": DummyOverlay(lagmask),
            "p_lt_0p005_mask": DummyOverlay(lagmask),
            "p_lt_0p001_mask": DummyOverlay(lagmask),
            "desc-plt0p001_mask": DummyOverlay(lagmask),
            "lagtimes": DummyOverlay(lagtimes, xsize=2.0, ysize=2.0, zsize=2.0),
            "lagstrengths": DummyOverlay(lagstrengths),
            "MTT": DummyOverlay(mtt),
            "graymask": DummyOverlay(graymask),
            "whitemask": DummyOverlay(whitemask),
        }


def prepmask_and_getmasksize_tests(debug=False):
    if debug:
        print("prepmask_and_getmasksize_tests")
    mask = np.ones((5, 5, 5), dtype=int)
    eroded = tide_quality.prepmask(mask)
    assert eroded.shape == mask.shape
    assert tide_quality.getmasksize(mask) == 125
    assert tide_quality.getmasksize(np.zeros_like(mask)) == 0


def checkregressors_tests(debug=False):
    if debug:
        print("checkregressors_tests")
    regressors = {
        "pass1": DummyRegressor(np.linspace(0.0, 1.0, 200), np.random.RandomState(10).rand(200)),
        "pass2": DummyRegressor(np.linspace(0.0, 1.0, 200), np.random.RandomState(11).rand(200)),
    }
    metrics = tide_quality.checkregressors(
        regressors, numpasses=2, filterlimits=[0.2, 0.8], debug=False
    )
    required = [
        "first_kurtosis",
        "first_kurtosis_z",
        "first_kurtosis_p",
        "first_skewness",
        "first_skewness_z",
        "first_skewness_p",
        "first_spectralflatness",
        "last_kurtosis",
        "last_kurtosis_z",
        "last_kurtosis_p",
        "last_skewness",
        "last_skewness_z",
        "last_skewness_p",
        "last_spectralflatness",
    ]
    for key in required:
        assert key in metrics
        assert np.isfinite(metrics[key])


def gethistmetrics_and_checkmap_tests(debug=False):
    if debug:
        print("gethistmetrics_and_checkmap_tests")
    rng = np.random.RandomState(12)
    themap = rng.randn(20, 20)
    themask = np.ones((20, 20), dtype=int)
    outdict = {}
    tide_quality.gethistmetrics(
        themap,
        themask,
        outdict,
        thehistlabel="testhist",
        histlen=51,
        rangemin=-2.0,
        rangemax=2.0,
        nozero=False,
        savehist=True,
        ignorefirstpoint=False,
        debug=False,
    )
    assert outdict["voxelsincluded"] > 0
    assert "histbincenters" in outdict
    assert "histvalues" in outdict
    assert np.isfinite(outdict["pct50"])

    # empty-after-nozero path
    outdict2 = {}
    tide_quality.gethistmetrics(
        np.zeros((10, 10)),
        np.ones((10, 10), dtype=int),
        outdict2,
        thehistlabel="emptyhist",
        nozero=True,
        savehist=True,
        debug=False,
    )
    assert outdict2["voxelsincluded"] == 0
    assert outdict2["pct50"] is None
    assert outdict2["histvalues"] is None

    cmap = tide_quality.checkmap(
        themap,
        themask,
        histlen=51,
        rangemin=-2.0,
        rangemax=2.0,
        histlabel="checkmaphist",
        ignorefirstpoint=False,
        savehist=True,
        debug=False,
    )
    assert "pct50" in cmap
    assert "peakloc" in cmap
    assert np.isfinite(cmap["pct50"])


def qualitycheck_tests(debug=False):
    if debug:
        print("qualitycheck_tests")
    with patch("rapidtide.qualitycheck.RapidtideDataset", DummyDataset):
        out1 = tide_quality.qualitycheck(
            "dummyroot",
            graymaskspec=None,
            whitemaskspec=None,
            anatname=None,
            geommaskname=None,
            userise=False,
            usecorrout=False,
            useatlas=False,
            forcetr=False,
            forceoffset=False,
            offsettime=0.0,
            verbose=False,
            debug=False,
        )
        assert out1["passes"] == 2
        assert "mask" in out1
        assert "regressor" in out1
        assert "lag" in out1
        assert "laggrad" in out1
        assert "strength" in out1
        assert "MTT" in out1

        out2 = tide_quality.qualitycheck(
            "dummyroot",
            graymaskspec="graymask.nii.gz",
            whitemaskspec="whitemask.nii.gz",
            debug=False,
        )
        assert "grayonly-lag" in out2
        assert "grayonly-laggrad" in out2
        assert "grayonly-strength" in out2
        assert "whiteonly-lag" in out2
        assert "whiteonly-laggrad" in out2
        assert "whiteonly-strength" in out2


def test_qualitycheck(debug=False, displayplots=False):
    np.random.seed(12345)
    prepmask_and_getmasksize_tests(debug=debug)
    checkregressors_tests(debug=debug)
    gethistmetrics_and_checkmap_tests(debug=debug)
    qualitycheck_tests(debug=debug)


if __name__ == "__main__":
    test_qualitycheck(debug=True, displayplots=False)
