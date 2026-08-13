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

    gaussfit = tide_stats.fitgausspdf(
        thehist, histlen, data, displayplots=False, nozero=False, debug=False
    )
    jsbfit = tide_stats.fitjsbpdf(
        thehist, histlen, data, displayplots=False, nozero=False, debug=False
    )
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
    nlp = tide_stats.neglog10pfromr(
        0.5, histfit, initialize=True, neglogpmin=0.0, neglogpmax=4.0, debug=False
    )
    assert np.isfinite(nlp)
    nlp_low = tide_stats.neglog10pfromr(
        -1.0, histfit, initialize=False, neglogpmin=0.0, neglogpmax=4.0, debug=False
    )
    nlp_high = tide_stats.neglog10pfromr(
        2.0, histfit, initialize=False, neglogpmin=0.0, neglogpmax=4.0, debug=False
    )
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
    icc3 = tide_stats.fast_ICC_rep_anova(Y, nocache=True, debug=False)  # explicit no-cache path
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
    ph, pl, pw, com = tide_stats.prochistogram(
        thehist, refine=False, pickleft=False, ignorefirstpoint=False, debug=False
    )
    assert np.isfinite(ph)
    assert np.isfinite(pl)
    assert np.isfinite(pw)
    assert np.isfinite(com)
    ph2, pl2, pw2, com2 = tide_stats.prochistogram(
        thehist, refine=True, pickleft=True, ignorefirstpoint=True, debug=False
    )
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
    with (
        patch("rapidtide.stats.tide_io.writenpvecs", return_value=None),
        patch("rapidtide.stats.tide_io.writebidstsv", return_value=None),
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
    fit_hist = np.histogram(
        np.clip(rng.normal(loc=0.4, scale=0.12, size=2500), 0.0, 1.0), 101, range=(0.0, 1.0)
    )
    fit_params = tide_stats.fitjsbpdf(
        fit_hist, 101, np.clip(rng.normal(loc=0.4, scale=0.12, size=2500), 0.0, 1.0)
    )
    pmask1 = tide_stats.makepmask(
        np.array([0.1, 0.3, 0.8]), pval=0.05, sighistfit=fit_params, onesided=True
    )
    pmask2 = tide_stats.makepmask(
        np.array([0.1, -0.3, 0.8]), pval=0.05, sighistfit=fit_params, onesided=False
    )
    assert pmask1.shape == (3,)
    assert pmask2.shape == (3,)

    frac_single = tide_stats.getfracval(data, 0.5, nozero=False)
    frac_multi = tide_stats.getfracvals(data, [0.02, 0.5, 0.98], nozero=False, debug=False)
    frac_multi_nz = tide_stats.getfracvals(
        np.array([0.0, 0.0, 1.0, 2.0]), [0.25, 0.75], nozero=True, debug=False
    )
    frac_fit = tide_stats.getfracvalsfromfit(fit_params, [0.90, 0.95])
    assert np.isfinite(frac_single)
    assert len(frac_multi) == 3
    assert len(frac_multi_nz) == 2
    assert len(frac_fit) == 2

    image = rng.randn(8, 8, 4) + 10.0
    mask1 = tide_stats.makemask(image, threshpct=25.0, verbose=False, nozero=False, noneg=False)
    mask2 = tide_stats.makemask(
        image - 20.0, threshpct=25.0, verbose=False, nozero=True, noneg=True
    )
    size1 = tide_stats.getmasksize(mask1)
    size2 = tide_stats.getmasksize(mask2)
    assert mask1.shape == image.shape
    assert mask2.shape == image.shape
    assert 0 <= size1 <= mask1.size
    assert 0 <= size2 <= mask2.size


def summarizevoxels_tests(debug=False):
    """summarizevoxels dispatches on a method name, and each branch has to mean what
    it says.  A dispatch table is exactly the kind of thing that silently degrades to
    "always the mean" if a branch is mistyped, so pin every method against an
    independently computed answer."""
    if debug:
        print("summarizevoxels_tests")

    # 2D input is (voxels, timepoints), so every statistic runs down axis 0 and comes
    # back per timepoint
    thevoxels = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])

    np.testing.assert_allclose(
        tide_stats.summarizevoxels(thevoxels, method="mean"), np.mean(thevoxels, axis=0)
    )
    np.testing.assert_allclose(
        tide_stats.summarizevoxels(thevoxels, method="sum"), np.sum(thevoxels, axis=0)
    )
    np.testing.assert_allclose(
        tide_stats.summarizevoxels(thevoxels, method="median"), np.median(thevoxels, axis=0)
    )
    np.testing.assert_allclose(
        tide_stats.summarizevoxels(thevoxels, method="std"), np.std(thevoxels, axis=0)
    )
    from statsmodels.robust import mad as thereferencemad

    themad = tide_stats.summarizevoxels(thevoxels, method="MAD")
    np.testing.assert_allclose(themad, thereferencemad(thevoxels, axis=0))
    # MAD and std are different estimators, so a collapse to std would be visible
    assert not np.allclose(themad, np.std(thevoxels, axis=0))

    # CoV is a percentage, and is the only method that is not a bare numpy reduction
    thecov = tide_stats.summarizevoxels(thevoxels, method="CoV")
    np.testing.assert_allclose(
        thecov, 100.0 * np.std(thevoxels, axis=0) / np.mean(thevoxels, axis=0)
    )
    # and the methods really are distinct, so a collapsed dispatch would show up
    assert not np.allclose(
        tide_stats.summarizevoxels(thevoxels, method="mean"),
        tide_stats.summarizevoxels(thevoxels, method="sum"),
    )

    # 1D input has no time axis, so every method collapses to a scalar
    theseries = np.array([2.0, 4.0, 6.0, 8.0])
    for themethod in ("mean", "sum", "median", "std", "MAD", "CoV"):
        theresult = tide_stats.summarizevoxels(theseries, method=themethod)
        assert np.isscalar(theresult) or theresult.ndim == 0, f"{themethod} did not reduce"
    np.testing.assert_allclose(tide_stats.summarizevoxels(theseries, method="mean"), 5.0)
    np.testing.assert_allclose(
        tide_stats.summarizevoxels(theseries, method="CoV"),
        100.0 * np.std(theseries) / np.mean(theseries),
    )

    # NaNs are scrubbed rather than propagated, so a single bad voxel cannot wipe out
    # a whole region summary
    thenanvoxels = np.array([[1.0, 2.0], [np.nan, 4.0]])
    assert np.all(np.isfinite(tide_stats.summarizevoxels(thenanvoxels, method="mean")))

    # A CoV whose mean is zero is undefined.  It has to be scrubbed to zero rather
    # than left as an infinity, because these values are written into the runoptions
    # JSON and Infinity is not legal JSON.  The scrubbing has to happen after the
    # percentage scaling, or the multiplier overflows the sanitised value back to inf.
    thezeromean = tide_stats.summarizevoxels(np.array([-1.0, 1.0]), method="CoV")
    assert np.isfinite(thezeromean), f"zero-mean CoV came back as {thezeromean}"
    assert thezeromean == 0.0
    # the same for the per-timepoint path, where only one column is degenerate
    thecolumns = tide_stats.summarizevoxels(np.array([[-1.0, 2.0], [1.0, 4.0]]), method="CoV")
    assert np.all(np.isfinite(thecolumns)), thecolumns
    assert thecolumns[0] == 0.0 and thecolumns[1] != 0.0

    # an unknown method is a programming error, not something to silently default
    try:
        tide_stats.summarizevoxels(thevoxels, method="nosuchmethod")
    except ValueError as theerror:
        assert "nosuchmethod" in str(theerror)
    else:
        raise AssertionError("an unknown summary method was accepted")


def regionstats_tests(debug=False):
    """regionstats crosses the fit mask with each tissue mask and summarizes four ways.
    The masks are optional, so the shape of the returned dict is part of its contract."""
    if debug:
        print("regionstats_tests")

    thenumvoxels = 40
    rng = np.random.RandomState(5)
    themap = rng.randn(thenumvoxels) + 5.0
    thefitmask = np.ones(thenumvoxels)
    thevalidvoxels = np.arange(thenumvoxels)

    thestats = ["mean", "median", "std", "MAD"]

    # with no tissue masks only the fit region is reported
    theresult = tide_stats.regionstats(themap, thevalidvoxels, thefitmask, None, None, None, None)
    assert set(theresult.keys()) == {f"fit_{s}" for s in thestats}
    np.testing.assert_allclose(theresult["fit_mean"], np.mean(themap))
    np.testing.assert_allclose(theresult["fit_median"], np.median(themap))

    # each supplied mask adds its own family of keys, and only its own
    thebrainmask = np.ones(thenumvoxels)
    thegraymask = np.zeros(thenumvoxels)
    thegraymask[:10] = 1
    theresult = tide_stats.regionstats(
        themap, thevalidvoxels, thefitmask, thebrainmask, thegraymask, None, None
    )
    assert set(theresult.keys()) == {
        f"{theregion}_{thestat}" for theregion in ("fit", "brain", "gray") for thestat in thestats
    }
    # the gray summary is over the gray voxels alone, not the whole map
    np.testing.assert_allclose(theresult["gray_mean"], np.mean(themap[:10]))
    np.testing.assert_allclose(theresult["brain_mean"], np.mean(themap))

    # the fit mask really does intersect: halving it halves what gray sees
    thehalffit = np.zeros(thenumvoxels)
    thehalffit[:5] = 1
    theresult = tide_stats.regionstats(
        themap, thevalidvoxels, thehalffit, None, thegraymask, None, None
    )
    np.testing.assert_allclose(theresult["gray_mean"], np.mean(themap[:5]))

    # all four tissue masks together
    thewhitemask = np.zeros(thenumvoxels)
    thewhitemask[10:20] = 1
    thecsfmask = np.zeros(thenumvoxels)
    thecsfmask[20:30] = 1
    theresult = tide_stats.regionstats(
        themap, thevalidvoxels, thefitmask, thebrainmask, thegraymask, thewhitemask, thecsfmask
    )
    assert len(theresult) == 5 * len(thestats)
    np.testing.assert_allclose(theresult["white_mean"], np.mean(themap[10:20]))
    np.testing.assert_allclose(theresult["csf_mean"], np.mean(themap[20:30]))
    if debug:
        print(f"  regionstats produced {len(theresult)} keys")


def pdf_fit_tests(debug=False):
    """fitgausspdf and fitjsbpdf both fit a distribution to a histogram and hand back
    their parameters with the zero bin tacked on the end.  The zero bin is special:
    rapidtide's null distributions have a spike at R=0 from failed fits, and --nozero
    is what says to ignore it."""
    if debug:
        print("pdf_fit_tests")

    rng = np.random.RandomState(11)
    thehistlen = 51

    # a clean gaussian, so the recovered centre and width are checkable
    thetruecentre, thetruewidth = 0.4, 0.08
    thedata = rng.normal(thetruecentre, thetruewidth, 20000)
    thedata = thedata[(thedata > 0.0) & (thedata < 1.0)]
    thehist = np.histogram(thedata, bins=thehistlen, range=(0.0, 1.0))

    thespiked = np.concatenate([thedata, np.zeros(500)])
    thespikedhist = np.histogram(thespiked, bins=thehistlen, range=(0.0, 1.0))
    # the fixture must actually have a spike, or the nozero assertion below is vacuous
    assert thespikedhist[0][0] > 0, "no zero-bin spike was planted"

    theparams = tide_stats.fitgausspdf(thehist, thehistlen, thedata, nozero=True)
    assert len(theparams) == 4, "expected height, centre, width and the zero term"
    theheight, thecentre, thewidth, thezeroterm = theparams
    if debug:
        print(f"  gauss fit: centre {thecentre:.3f} width {abs(thewidth):.3f}")
    assert abs(thecentre - thetruecentre) < 0.05, f"centre came back at {thecentre}"
    assert abs(abs(thewidth) - thetruewidth) < 0.05, f"width came back at {thewidth}"
    assert thezeroterm == 0.0

    # nozero discards the zero bin even when there is a real spike sitting in it
    thezeroed = tide_stats.fitgausspdf(thespikedhist, thehistlen, thespiked, nozero=True)
    assert thezeroed[3] == 0.0, f"nozero left a zero term of {thezeroed[3]}"

    # without nozero the same spike is preserved and handed back
    theparams = tide_stats.fitgausspdf(thespikedhist, thehistlen, thespiked, nozero=False)
    if debug:
        print(f"  zero term with a planted spike: {theparams[3]}")
    assert theparams[3] > 0.0, "the zero bin spike was not carried through"

    # the Johnson SB fit returns its own four parameters plus the zero term
    thejsbparams = tide_stats.fitjsbpdf(thespikedhist, thehistlen, thespiked, nozero=True)
    assert len(thejsbparams) == 5
    assert np.all(np.isfinite(thejsbparams))
    assert thejsbparams[4] == 0.0
    thejsbparams = tide_stats.fitjsbpdf(thespikedhist, thehistlen, thespiked, nozero=False)
    assert thejsbparams[4] > 0.0


def reporting_path_tests(debug=False):
    """The debug and displayplots branches are reporting, not arithmetic, but they run
    inside the fitting routines and a stale f-string in one of them takes a whole
    analysis down.  Exercise them with the plotting stubbed out."""
    if debug:
        print("reporting_path_tests")

    rng = np.random.RandomState(17)
    thehistlen = 51
    thedata = np.clip(rng.normal(0.4, 0.1, 5000), 0.0, 1.0)
    thehist = np.histogram(thedata, bins=thehistlen, range=(0.0, 1.0))

    with patch("rapidtide.stats.plt") as mock_plt:
        tide_stats.fitgausspdf(
            thehist, thehistlen, thedata, displayplots=True, nozero=False, debug=True
        )
        tide_stats.fitjsbpdf(
            thehist, thehistlen, thedata, displayplots=True, nozero=False, debug=True
        )
        assert mock_plt.show.call_count == 2, "the display branches did not run"

    # gethistprops with an explicit range, and its debug reporting
    thepeakloc, thepeakheight, thepeakwidth = tide_stats.gethistprops(
        thedata, histlen=thehistlen, therange=(0.0, 1.0), refine=True
    )
    assert np.isfinite(thepeakloc) and np.isfinite(thepeakwidth)

    # percentilefromloc with nozero drops the zero entries before ranking.  Padding the
    # data with an equal number of zeros puts half the samples below the distribution,
    # so 0.4 sits at roughly the 75th percentile with them and the 50th without.
    thepadded = np.concatenate([thedata, np.zeros(5000)])
    thewithzeros = tide_stats.percentilefromloc(thepadded, 0.4, nozero=False)
    thewithoutzeros = tide_stats.percentilefromloc(thepadded, 0.4, nozero=True)
    if debug:
        print(f"  percentile with zeros {thewithzeros}, without {thewithoutzeros}")
    assert thewithoutzeros < thewithzeros, "nozero did not drop the zero entries"
    assert 70.0 < thewithzeros < 80.0
    assert 45.0 < thewithoutzeros < 55.0

    # getfracvals debug reporting, and sigFromDistributionData with the fit disabled
    tide_stats.getfracvals(thedata, [0.25, 0.75], nozero=False, debug=True)
    thepcts, thefitpcts, thehistfit = tide_stats.sigFromDistributionData(
        thedata,
        histlen=thehistlen,
        thepercentiles=np.array([0.95]),
        dosighistfit=False,
        displayplots=False,
        debug=True,
    )
    assert len(thepcts) == 1
    # one placeholder per requested percentile, not an empty list: callers index this
    # alongside pcts_data, so the two have to stay the same length
    assert list(thefitpcts) == [None], f"got {thefitpcts}"
    assert thehistfit is None

    # neglog10pfromr debug reporting needs a fit to work from
    dummy, dummy2, thehistfit = tide_stats.sigFromDistributionData(
        thedata, histlen=thehistlen, thepercentiles=np.array([0.95]), dosighistfit=True
    )
    theneglogp = tide_stats.neglog10pfromr(0.5, thehistfit, initialize=True, debug=True)
    assert np.isfinite(theneglogp)

    # makemask verbose reporting
    theimage = rng.randn(6, 6, 3) + 10.0
    themask = tide_stats.makemask(theimage, threshpct=25.0, verbose=True)
    assert themask.shape == theimage.shape


def makehistogram_binsize_tests(debug=False):
    """makehistogram can be asked for a bin WIDTH instead of a bin count.

    np.linspace counts bin edges and demands an integer, so passing the float
    (range / binsize + 1) straight through used to raise TypeError for every caller.
    No internal caller passes binsize, which is why it went unnoticed.
    """
    if debug:
        print("makehistogram_binsize_tests")

    thedata = np.random.RandomState(29).uniform(0.0, 1.0, 2000)
    theresult = tide_stats.makehistogram(thedata, None, binsize=0.05, therange=[0.0, 1.0])
    thehist = theresult[0]
    if debug:
        print(f"  binsize 0.05 over [0, 1] gave {len(thehist[0])} bins")
    # 20 bins of width 0.05 spanning the range, so 21 edges
    assert len(thehist[1]) == 21, f"got {len(thehist[1])} edges"
    assert len(thehist[0]) == 20
    np.testing.assert_allclose(np.diff(thehist[1]), 0.05)
    # every sample landed somewhere
    assert thehist[0].sum() == len(thedata)

    # a coarser width gives correspondingly fewer bins
    thecoarse = tide_stats.makehistogram(thedata, None, binsize=0.25, therange=[0.0, 1.0])[0]
    assert len(thecoarse[0]) == 4


def makeandsavehistogram_tests(debug=False):
    """makeandsavehistogram writes the histogram to disk and optionally folds its peak
    properties into a dict for the runoptions.  Both outputs matter downstream."""
    if debug:
        print("makeandsavehistogram_tests")

    import tempfile

    rng = np.random.RandomState(23)
    thedata = rng.normal(5.0, 1.0, 4000)
    thedict = {}

    with tempfile.TemporaryDirectory() as thedir:
        theoutname = f"{thedir}/thehist"
        with patch("rapidtide.stats.plt") as mock_plt:
            tide_stats.makeandsavehistogram(
                thedata,
                101,
                0,
                theoutname,
                displaytitle="a test histogram",
                displayplots=True,
                refine=True,
                therange=(0.0, 10.0),
                normalize=True,
                dictvarname="thepeak",
                thedict=thedict,
                debug=True,
            )
            assert mock_plt.show.call_count == 1

        import os

        thewritten = [thename for thename in os.listdir(thedir) if thename.startswith("thehist")]
        if debug:
            print(f"  wrote {thewritten}, dict keys {sorted(thedict.keys())}")
        assert thewritten, "no histogram file was written"

    # the peak properties are folded into the supplied dict under the given name
    assert any(thekey.startswith("thepeak") for thekey in thedict), sorted(thedict.keys())
    # and the recorded peak is near the true centre of the distribution
    thepeakkeys = [thekey for thekey in thedict if "peak" in thekey and "loc" in thekey.lower()]
    if thepeakkeys:
        assert abs(thedict[thepeakkeys[0]] - 5.0) < 1.0


def test_stats(debug=False, displayplots=False):
    np.random.seed(12345)
    distribution_and_significance_tests(debug=debug)
    correlation_transform_tests(debug=debug)
    timeseries_stats_tests(debug=debug)
    histogram_and_mask_tests(debug=debug)
    summarizevoxels_tests(debug=debug)
    regionstats_tests(debug=debug)
    pdf_fit_tests(debug=debug)
    reporting_path_tests(debug=debug)
    makehistogram_binsize_tests(debug=debug)
    makeandsavehistogram_tests(debug=debug)


if __name__ == "__main__":
    test_stats(debug=True, displayplots=False)
