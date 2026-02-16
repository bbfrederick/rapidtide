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

import rapidtide.stats as tide_stats


def distribution_and_significance_tests(debug=False):
    if debug:
        print("distribution_and_significance_tests")

    rng = np.random.RandomState(1)
    data = np.clip(rng.normal(loc=0.4, scale=0.15, size=2000), 0.0, 1.0)
    histlen = 101
    thehist = np.histogram(data, histlen, range=(0.0, 1.0))

    tide_stats.printthresholds([0.1, 0.2], [0.95, 0.99], "thresholds")

    gaussfit = tide_stats.fitgausspdf(thehist, histlen, data, displayplots=False, nozero=False, debug=False)
    jsbfit = tide_stats.fitjsbpdf(thehist, histlen, data, displayplots=False, nozero=False, debug=False)
    assert len(gaussfit) == 4
    assert len(jsbfit) == 5

    # Routine currently only initializes internals and returns None
    assert tide_stats.getjohnsonppf(0.95, jsbfit[:4], jsbfit[4]) is None

    pcts = np.array([0.95, 0.99])
    p_data, p_fit, histfit = tide_stats.sigFromDistributionData(
        data,
        histlen=histlen,
        thepercentiles=pcts,
        similaritymetric="correlation",
        displayplots=False,
        twotail=False,
        nozero=False,
        dosighistfit=True,
        debug=False,
    )
    assert len(p_data) == len(pcts)
    assert len(p_fit) == len(pcts)
    assert histfit is not None

    # twotail and mutualinfo path
    p_data2, p_fit2, histfit2 = tide_stats.sigFromDistributionData(
        data,
        histlen=histlen,
        thepercentiles=pcts,
        similaritymetric="mutualinfo",
        displayplots=False,
        twotail=True,
        nozero=False,
        dosighistfit=True,
        debug=False,
    )
    assert len(p_data2) == len(pcts)
    assert len(p_fit2) == len(pcts)
    assert histfit2 is not None

    # no nonzero path
    none_data, none_fit, none_histfit = tide_stats.sigFromDistributionData(
        np.zeros(100),
        histlen=51,
        thepercentiles=pcts,
        similaritymetric="correlation",
        displayplots=False,
        twotail=False,
        nozero=False,
        dosighistfit=True,
        debug=False,
    )
    assert none_data is None
    assert none_fit is None
    assert none_histfit is None

    # LUT init and clamp branches
    nlp = tide_stats.neglog10pfromr(0.5, histfit, initialize=True, neglogpmin=0.0, neglogpmax=4.0, debug=False)
    assert np.isfinite(nlp)
    nlp_low = tide_stats.neglog10pfromr(-1.0, histfit, initialize=False, neglogpmin=0.0, neglogpmax=4.0, debug=False)
    nlp_high = tide_stats.neglog10pfromr(2.0, histfit, initialize=False, neglogpmin=0.0, neglogpmax=4.0, debug=False)
    assert np.fabs(nlp_low - 0.0) < 1e-12
    assert np.fabs(nlp_high - 4.0) < 1e-12

    with patch("rapidtide.stats.tide_io.readvecs", return_value=[jsbfit]):
        rvals = tide_stats.rfromp("ignored_fitfile.txt", [0.90, 0.95, 0.99])
    assert len(rvals) == 3


def correlation_transform_tests(debug=False):
    if debug:
        print("correlation_transform_tests")

    tval = tide_stats.tfromr(0.5, nsamps=100, dfcorrfac=1.0, oversampfactor=1.0, returnp=False)
    assert np.isfinite(tval)
    tval2, tp = tide_stats.tfromr(0.3, nsamps=120, returnp=True)
    assert np.isfinite(tval2)
    assert 0.0 <= tp <= 1.0
    tinf, pzero = tide_stats.tfromr(1.0, nsamps=50, returnp=True)
    assert np.isinf(tinf)
    assert np.fabs(pzero) < 1e-12

    p2 = tide_stats.pfromz(2.0, twotailed=True)
    p1 = tide_stats.pfromz(2.0, twotailed=False)
    assert 0.0 <= p2 <= 1.0
    assert 0.0 <= p1 <= 1.0
    assert p2 >= p1

    zval = tide_stats.zfromr(0.5, nsamps=100, returnp=False)
    assert np.isfinite(zval)
    zval2, zp = tide_stats.zfromr(0.4, nsamps=120, returnp=True)
    assert np.isfinite(zval2)
    assert 0.0 <= zp <= 1.0
    zinf, zp0 = tide_stats.zfromr(1.0, nsamps=120, returnp=True)
    assert np.isinf(zinf)
    assert np.fabs(zp0) < 1e-12

    zdiff = tide_stats.zofcorrdiff(0.5, 0.2, 80, 90)
    sed = tide_stats.stderrofdiff(80, 90)
    fz = tide_stats.fisher(0.5)
    assert np.isfinite(zdiff)
    assert np.isfinite(sed)
    assert np.isfinite(fz)


def timeseries_stats_tests(debug=False):
    if debug:
        print("timeseries_stats_tests")

    rng = np.random.RandomState(2)
    x = rng.randn(512)
    xp = tide_stats.permute_phase(x)
    assert xp.shape == x.shape
    assert np.isfinite(np.sum(xp))

    s, sz, sp = tide_stats.skewnessstats(x)
    k, kz, kp = tide_stats.kurtosisstats(x)
    assert np.isfinite(s)
    assert np.isfinite(sz)
    assert 0.0 <= sp <= 1.0
    assert np.isfinite(k)
    assert np.isfinite(kz)
    assert 0.0 <= kp <= 1.0

    fmri = rng.randn(6, 100)
    mins, maxs, means, stds, meds, mads, skews, kurts = tide_stats.fmristats(fmri)
    assert mins.shape == (6,)
    assert maxs.shape == (6,)
    assert means.shape == (6,)
    assert stds.shape == (6,)
    assert meds.shape == (6,)
    assert mads.shape == (6,)
    assert skews.shape == (6,)
    assert kurts.shape == (6,)

    Y = rng.randn(12, 4)
    icc1 = tide_stats.fast_ICC_rep_anova(Y, nocache=False, debug=False)
    icc2 = tide_stats.fast_ICC_rep_anova(Y, nocache=False, debug=False)  # cached path
    icc3 = tide_stats.fast_ICC_rep_anova(Y, nocache=True, debug=False)   # explicit no-cache path
    for res in [icc1, icc2, icc3]:
        assert len(res) == 6
        assert np.isfinite(res[0])


def histogram_and_mask_tests(debug=False):
    if debug:
        print("histogram_and_mask_tests")

    rng = np.random.RandomState(3)
    x1 = rng.normal(loc=10.0, scale=1.2, size=3000)
    x2 = rng.normal(loc=18.0, scale=1.2, size=1200)
    data = np.concatenate([x1, x2])

    peakloc, peakheight, peakwidth = tide_stats.gethistprops(data, histlen=101, refine=False)
    assert np.isfinite(peakloc)
    assert np.isfinite(peakheight)
    assert np.isfinite(peakwidth)
    peakloc2, _, _ = tide_stats.gethistprops(data, histlen=101, refine=True, pickleft=True)
    assert np.isfinite(peakloc2)

    thehist = np.histogram(data, 101)
    ph, pl, pw, com = tide_stats.prochistogram(thehist, refine=False, pickleft=False, ignorefirstpoint=False, debug=False)
    assert np.isfinite(ph)
    assert np.isfinite(pl)
    assert np.isfinite(pw)
    assert np.isfinite(com)
    ph2, pl2, pw2, com2 = tide_stats.prochistogram(thehist, refine=True, pickleft=True, ignorefirstpoint=True, debug=False)
    assert np.isfinite(ph2)
    assert np.isfinite(pl2)
    assert np.isfinite(pw2)
    assert np.isfinite(com2)

    pct = tide_stats.percentilefromloc(data, pl, nozero=False)
    assert 0.0 <= pct <= 100.0

    mh = tide_stats.makehistogram(
        data,
        histlen=101,
        therange=(data.min(), data.max()),
        pickleft=False,
        refine=False,
        normalize=True,
        ignorefirstpoint=False,
        debug=False,
    )
    assert len(mh) == 6

    echolag, echoratio = tide_stats.echoloc(data, histlen=101, startoffset=3.0)
    assert np.isfinite(echolag)
    assert np.isfinite(echoratio)

    # file-writing path
    with patch("rapidtide.stats.tide_io.writenpvecs", return_value=None), patch(
        "rapidtide.stats.tide_io.writebidstsv", return_value=None
    ):
        tide_stats.makeandsavehistogram(
            data,
            histlen=101,
            endtrim=0,
            outname="dummy_hist",
            displayplots=False,
            refine=True,
            normalize=False,
            thedict=None,
            append=False,
            debug=False,
        )

    # dictionary path
    hdict = {}
    with patch("rapidtide.stats.tide_io.writebidstsv", return_value=None):
        tide_stats.makeandsavehistogram(
            data,
            histlen=101,
            endtrim=0,
            outname="dummy_hist2",
            displayplots=False,
            refine=False,
            normalize=False,
            dictvarname="myhist",
            thedict=hdict,
            append=False,
            debug=False,
        )
    assert "myhist_centerofmass.txt" in hdict
    assert "myhist_peak.txt" in hdict

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    sym = tide_stats.symmetrize(a, antisymmetric=False, zerodiagonal=False)
    asym = tide_stats.symmetrize(a, antisymmetric=True, zerodiagonal=True)
    assert sym.shape == a.shape
    assert asym.shape == a.shape
    assert np.fabs(np.trace(asym)) < 1e-12

    # fit-based routines
    fit_hist = np.histogram(np.clip(rng.normal(loc=0.4, scale=0.12, size=2500), 0.0, 1.0), 101, range=(0.0, 1.0))
    fit_params = tide_stats.fitjsbpdf(fit_hist, 101, np.clip(rng.normal(loc=0.4, scale=0.12, size=2500), 0.0, 1.0))
    pmask1 = tide_stats.makepmask(np.array([0.1, 0.3, 0.8]), pval=0.05, sighistfit=fit_params, onesided=True)
    pmask2 = tide_stats.makepmask(np.array([0.1, -0.3, 0.8]), pval=0.05, sighistfit=fit_params, onesided=False)
    assert pmask1.shape == (3,)
    assert pmask2.shape == (3,)

    frac_single = tide_stats.getfracval(data, 0.5, nozero=False)
    frac_multi = tide_stats.getfracvals(data, [0.02, 0.5, 0.98], nozero=False, debug=False)
    frac_multi_nz = tide_stats.getfracvals(np.array([0.0, 0.0, 1.0, 2.0]), [0.25, 0.75], nozero=True, debug=False)
    frac_fit = tide_stats.getfracvalsfromfit(fit_params, [0.90, 0.95])
    assert np.isfinite(frac_single)
    assert len(frac_multi) == 3
    assert len(frac_multi_nz) == 2
    assert len(frac_fit) == 2

    image = rng.randn(8, 8, 4) + 10.0
    mask1 = tide_stats.makemask(image, threshpct=25.0, verbose=False, nozero=False, noneg=False)
    mask2 = tide_stats.makemask(image - 20.0, threshpct=25.0, verbose=False, nozero=True, noneg=True)
    size1 = tide_stats.getmasksize(mask1)
    size2 = tide_stats.getmasksize(mask2)
    assert mask1.shape == image.shape
    assert mask2.shape == image.shape
    assert 0 <= size1 <= mask1.size
    assert 0 <= size2 <= mask2.size


def test_stats(debug=False, displayplots=False):
    np.random.seed(12345)
    distribution_and_significance_tests(debug=debug)
    correlation_transform_tests(debug=debug)
    timeseries_stats_tests(debug=debug)
    histogram_and_mask_tests(debug=debug)


if __name__ == "__main__":
    test_stats(debug=True, displayplots=False)
