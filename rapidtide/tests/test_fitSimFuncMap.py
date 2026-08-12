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
"""Tests for rapidtide.workflows.fitSimFuncMap.

These exercise the two untested halves of the module: the masked median filter
that drives despeckle target selection, and the fitSimFunc orchestration routine
itself.  fitSimFunc is tested against the REAL fitter and the REAL fitcorr rather
than mocks, so the assertions are about delays actually being repaired, not about
which functions got called.
"""

import tempfile
import warnings
from typing import Any, Optional, Tuple
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

import rapidtide.io as tide_io
import rapidtide.peakeval as tide_peakeval
import rapidtide.simFuncClasses as tide_simFuncClasses
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.workflows.fitSimFuncMap as tide_fitSimFuncMap
from rapidtide.workflows.fitSimFuncMap import fitSimFunc, masked_median_filter

# ---------------------------------------------------------------------------
# masked_median_filter
# ---------------------------------------------------------------------------


def test_masked_median_filter_none_mask_matches_scipy(debug: bool = False) -> None:
    """With no mask the filter must be exactly scipy's, not a reimplementation of it."""
    therng = np.random.default_rng(1234)
    thedata = therng.normal(size=(7, 8, 9))
    for thesize in (3, 5):
        theresult = masked_median_filter(thedata, size=thesize)
        theexpected = ndimage.median_filter(thedata, size=thesize, mode="reflect")
        if debug:
            print(f"size {thesize}: max deviation {np.max(np.abs(theresult - theexpected))}")
        assert np.array_equal(theresult, theexpected)


def test_masked_median_filter_pad_mode_mapping(debug: bool = False) -> None:
    """The scipy->numpy padding mode table must be right.

    scipy and numpy use the same five words for different things: scipy's
    'reflect' is numpy's 'symmetric' and scipy's 'mirror' is numpy's 'reflect'.
    Getting the mapping backwards changes every boundary voxel and raises no
    error, so pin it down by requiring the masked path with an all-in-mask
    mask to reproduce scipy's unmasked answer exactly.
    """
    therng = np.random.default_rng(99)
    thedata = therng.normal(size=(6, 7, 8))
    thefullmask = np.ones_like(thedata, dtype=bool)

    # For these modes padding the all-True mask still yields all True, so every
    # window is fully in mask and the answers must agree everywhere.
    for themode in ("reflect", "nearest", "wrap", "mirror"):
        theresult = masked_median_filter(thedata, size=3, mask=thefullmask, mode=themode)
        theexpected = ndimage.median_filter(thedata, size=3, mode=themode)
        if debug:
            print(f"{themode}: max deviation {np.max(np.abs(theresult - theexpected))}")
        assert np.allclose(theresult, theexpected), f"mode {themode} does not match scipy"

    # 'constant' pads the mask with False, so the masked path deliberately drops
    # the padded region instead of voting with scipy's cval of 0.  They must still
    # agree in the interior, where no padding is involved.
    theresult = masked_median_filter(thedata, size=3, mask=thefullmask, mode="constant")
    theexpected = ndimage.median_filter(thedata, size=3, mode="constant")
    assert np.allclose(theresult[1:-1, 1:-1, 1:-1], theexpected[1:-1, 1:-1, 1:-1])


def test_masked_median_filter_ignores_out_of_mask_voxels(debug: bool = False) -> None:
    """Hand computed case: an out of mask spike must not reach any median."""
    thedata = np.array([1.0, 2.0, 100.0, 4.0, 5.0])
    themask = np.array([1, 1, 0, 1, 1])

    theresult = masked_median_filter(thedata, size=3, mask=themask, mode="nearest")
    # index 0: [1,1,2] all in mask                     -> 1.0
    # index 1: [1,2,100], the 100 is out of mask       -> median(1,2)   = 1.5
    # index 2: [2,100,4], the centre is out of mask    -> median(2,4)   = 3.0
    # index 3: [100,4,5], the 100 is out of mask       -> median(4,5)   = 4.5
    # index 4: [4,5,5] all in mask                     -> 5.0
    theexpected = np.array([1.0, 1.5, 3.0, 4.5, 5.0])
    if debug:
        print(f"got {theresult}, expected {theexpected}")
    assert np.allclose(theresult, theexpected)

    # and the unmasked filter must differ, or the case proves nothing
    theunmasked = masked_median_filter(thedata, size=3, mode="nearest")
    assert not np.allclose(theunmasked, theexpected)


def test_masked_median_filter_chunking_is_transparent(debug: bool = False) -> None:
    """The memory capped chunked path must give bit identical results to the
    single shot path.  The chunk loop indexes into a reshaped sliding window
    view, so an arithmetic slip there would silently permute voxels."""
    therng = np.random.default_rng(31415)
    thedata = therng.normal(size=(9, 10, 11))
    themask = therng.random(thedata.shape) > 0.3

    theoneshot = masked_median_filter(thedata, size=3, mask=themask)

    theoriginal = tide_fitSimFuncMap._MAX_CHUNK_BYTES
    try:
        # force a chunk size of a handful of voxels: 27 kernel entries * 8 bytes
        tide_fitSimFuncMap._MAX_CHUNK_BYTES = 27 * 8 * 7
        thechunked = masked_median_filter(thedata, size=3, mask=themask)
    finally:
        tide_fitSimFuncMap._MAX_CHUNK_BYTES = theoriginal

    if debug:
        print(f"max deviation {np.nanmax(np.abs(thechunked - theoneshot))}")
    assert np.array_equal(thechunked, theoneshot, equal_nan=True)


def test_masked_median_filter_empty_window_is_nan(debug: bool = False) -> None:
    """A voxel with no in mask neighbours has no median to report.  It must come
    back NaN rather than zero, since zero is a legal delay and would be taken as
    a despeckle target."""
    thedata = np.ones((5, 5, 5))
    themask = np.zeros((5, 5, 5), dtype=bool)
    themask[0, 0, 0] = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        theresult = masked_median_filter(thedata, size=3, mask=themask)

    if debug:
        print(f"corner {theresult[0, 0, 0]}, centre {theresult[2, 2, 2]}")
    # the lone in mask voxel and its neighbours see it, so they are finite
    assert np.isfinite(theresult[0, 0, 0])
    # the middle of the volume sees nothing in mask
    assert np.isnan(theresult[2, 2, 2])


def test_masked_median_filter_anisotropic_kernel(debug: bool = False) -> None:
    """A per axis kernel tuple must be honoured, not collapsed to a scalar."""
    therng = np.random.default_rng(2718)
    thedata = therng.normal(size=(6, 7, 8))
    thefullmask = np.ones_like(thedata, dtype=bool)
    thesize = (3, 5, 1)

    theresult = masked_median_filter(thedata, size=thesize, mask=thefullmask, mode="nearest")
    theexpected = ndimage.median_filter(thedata, size=thesize, mode="nearest")
    if debug:
        print(f"max deviation {np.max(np.abs(theresult - theexpected))}")
    assert np.allclose(theresult, theexpected)

    # a kernel of 1 along the last axis must not behave like a 3 there
    theisotropic = masked_median_filter(thedata, size=3, mask=thefullmask, mode="nearest")
    assert not np.allclose(theresult, theisotropic)


# ---------------------------------------------------------------------------
# fitSimFunc scaffolding
# ---------------------------------------------------------------------------


class _DummyLogger:
    """Stands in for LGR and TimingLGR, recording what the run said.

    TimingLGR is called with an extra dict of message fields, so every method
    tolerates trailing positional and keyword arguments.

    Attributes
    ----------
    messages : list of str
        Every message logged, in order.  Several tests read this to check the
        order operations ran in.
    debug : bool
        Echo each message to stdout as it arrives.
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialise an empty message log.

        Parameters
        ----------
        debug : bool, optional
            Echo each message to stdout.  Default is False.

        Returns
        -------
        None
        """
        self.messages: list = []
        self.debug = debug

    def info(self, themessage: Any, *theargs: Any, **thekwargs: Any) -> None:
        """Record a message.

        Parameters
        ----------
        themessage : Any
            The message, stringified before storing.
        *theargs : Any
            Extra positional fields, accepted and ignored.
        **thekwargs : Any
            Extra keyword fields, accepted and ignored.

        Returns
        -------
        None
        """
        self.messages.append(str(themessage))
        if self.debug:
            print(themessage)

    def warning(self, themessage: Any, *theargs: Any, **thekwargs: Any) -> None:
        """Record a warning, which goes to the same log as everything else.

        Parameters
        ----------
        themessage : Any
            The message, stringified before storing.
        *theargs : Any
            Extra positional fields, accepted and ignored.
        **thekwargs : Any
            Extra keyword fields, accepted and ignored.

        Returns
        -------
        None
        """
        self.info(themessage)

    verbose = info


class _DummyInputData:
    """Stands in for theinputdata, which is only consulted for output routing.

    Attributes
    ----------
    filetype : str
        One of 'nifti', 'cifti', or 'text'; selects the output branch taken.
    cifti_hdr : None
        Passed straight through to savemaplist, never inspected here.
    """

    def __init__(self, filetype: str = "nifti") -> None:
        """Initialise the stand in.

        Parameters
        ----------
        filetype : str, optional
            The file type to report.  Default is 'nifti'.

        Returns
        -------
        None
        """
        self.filetype = filetype
        self.cifti_hdr = None


def _gaussian(thex: NDArray, theamp: float, theloc: float, thesigma: float) -> NDArray:
    """A Gaussian bump, used to synthesise similarity functions.

    A Gaussian is the shape the default 'gauss' peak fitter expects, so a fit to
    one of these is exact and the recovered lag is the planted lag.  That is what
    lets the tests assert on delays to a tight tolerance.

    Parameters
    ----------
    thex : NDArray
        The lag axis to evaluate on, in seconds.
    theamp : float
        Peak amplitude.
    theloc : float
        Peak location, in seconds.
    thesigma : float
        Peak width, in seconds.

    Returns
    -------
    NDArray
        The evaluated bump, same shape as thex.
    """
    return theamp * np.exp(-((thex - theloc) ** 2) / (2.0 * thesigma**2))


def _makecorrout(
    theshape: Tuple[int, int, int] = (8, 8, 8),
    thebadvoxels: Optional[list] = None,
    thetruelag: float = 0.0,
    thesidelobelag: float = 10.0,
) -> Tuple[NDArray, NDArray, list]:
    """Build a synthetic similarity function volume with planted peak picking errors.

    Every voxel gets a Gaussian peak at ``thetruelag``.  The voxels listed in
    ``thebadvoxels`` additionally get a TALLER Gaussian at ``thesidelobelag``, so
    an unconstrained peak pick lands on the wrong lobe for exactly those voxels -
    the error that despeckling and delay resolution both exist to repair.

    Parameters
    ----------
    theshape : tuple of int
        Native space shape of the volume.
    thebadvoxels : list of int or None
        Flat indices to plant a dominant wrong lobe in.  Defaults to a few
        well separated interior voxels.
    thetruelag : float
        Lag of the peak every voxel gets, in seconds.
    thesidelobelag : float
        Lag of the taller decoy peak, in seconds.

    Returns
    -------
    corrout : NDArray
        Similarity functions, shape (numvoxels, numlags).
    corrscale : NDArray
        The lag axis, in seconds.
    thebadvoxels : list of int
        The flat indices that were given a decoy peak.
    """
    thecorrscale = np.linspace(-15.0, 15.0, 121)
    thenumvoxels = int(np.prod(theshape))
    if thebadvoxels is None:
        # interior voxels, far enough apart that none is in another's median kernel
        thebadvoxels = [
            int(np.ravel_multi_index(thepoint, theshape))
            for thepoint in ((2, 2, 2), (2, 5, 5), (5, 2, 5), (5, 5, 2))
        ]

    thecorrout = np.zeros((thenumvoxels, len(thecorrscale)), dtype=np.float64)
    thecorrout[:, :] = _gaussian(thecorrscale, 0.75, thetruelag, 2.0)
    for thevox in thebadvoxels:
        thecorrout[thevox, :] = _gaussian(thecorrscale, 0.55, thetruelag, 2.0) + _gaussian(
            thecorrscale, 0.85, thesidelobelag, 2.0
        )
    return thecorrout, thecorrscale, thebadvoxels


def _makewalledinvolume() -> Tuple[Tuple[int, int, int], NDArray, NDArray, NDArray, int]:
    """Build a good voxel walled in by a solid block of failed fits.

    Driving voxels out of the fit mask has to be done with a lever that does not
    depend on how a fit converges, or the fixture stops reproducing across scipy
    and numpy versions.  The lever used here is the fitter's lag range check,
    ``lagmin > maxlag``, which is a plain comparison against the fitted peak
    location: the wall voxels have a clean, fully sampled Gaussian at -15 s while
    the fitter's lagmin is -10 s, a 5 s margin on a quantity that is recovered to
    better than the 0.25 s sample spacing.  The failure sets FML_FITLAGLOW, which
    counts as a fit failure and so zeroes the mask, and clamps the reported lag to
    lagmin, which is the -10 s the wall then holds.

    (An earlier version drove the mask with the width ceiling instead.  That reads
    the fitted sigma of a heavily truncated Gaussian, which is poorly conditioned,
    and it did not reproduce on CI.)

    Returns
    -------
    theshape : tuple of int
        Native space shape of the volume.
    thecorrout : NDArray
        Similarity functions, shape (numvoxels, numlags).
    thecorrscale : NDArray
        The lag axis, in seconds.
    theoutofmask : NDArray
        Boolean volume, True where the fit is expected to fail.
    thecentreindex : int
        Flat index of the single in mask voxel inside the wall.
    """
    theshape = (10, 10, 10)
    thenumvoxels = int(np.prod(theshape))
    thecorrscale = np.linspace(-20.0, 20.0, 161)

    theoutofmask = np.zeros(theshape, dtype=bool)
    theoutofmask[3:8, 3:8, 3:8] = True
    thecentre = (5, 5, 5)
    theoutofmask[thecentre] = False
    thecentreindex = int(np.ravel_multi_index(thecentre, theshape))

    thecorrout = np.zeros((thenumvoxels, len(thecorrscale)))
    thecorrout[:, :] = _gaussian(thecorrscale, 0.75, 0.0, 2.0)
    thecorrout[theoutofmask.reshape(-1), :] = _gaussian(thecorrscale, 0.75, -15.0, 2.0)
    # the walled in voxel has a subsidiary peak at -9, inside the range a refit
    # targeted on the wall's -10 would search, so a wrong despeckle decision here
    # produces a visibly moved delay rather than a silent no op
    thecorrout[thecentreindex, :] = _gaussian(thecorrscale, 0.75, 0.0, 2.0) + _gaussian(
        thecorrscale, 0.5, -9.0, 1.5
    )
    return theshape, thecorrout, thecorrscale, theoutofmask, thecentreindex


def _makeoptiondict(**theoverrides: Any) -> dict:
    """A minimal but complete optiondict for fitSimFunc.

    Parameters
    ----------
    **theoverrides : Any
        Keys to override in the defaults below.

    Returns
    -------
    dict
        An option dictionary holding every key fitSimFunc reads, configured for
        single process, no progress bar, no output saving.
    """
    theoptiondict = {
        "similaritymetric": "correlation",
        "nprocs_peakeval": 1,
        "nprocs_fitcorr": 1,
        "mklthreads": 1,
        "threaddebug": False,
        "alwaysmultiproc": False,
        "bipolar": False,
        "oversampfactor": 1,
        "interptype": "univariate",
        "showprogressbar": False,
        "mp_chunksize": 1000,
        "fixdelay": False,
        "despeckle_thresh": 5.0,
        "despeckle_passes": 4,
        "despeckle_kernel_size": 3,
        "savedespecklemasks": False,
        "resolvedelays": False,
        "resolvepasses": 3,
        "saveresolvemaps": False,
        "passes": 1,
        "corrmasksize": 512,
        "lagmin": -15.0,
        "lagmax": 15.0,
    }
    theoptiondict.update(theoverrides)
    return theoptiondict


def _makefitargs(
    theoptiondict: dict,
    thecorrout: NDArray,
    thecorrscale: NDArray,
    theshape: Tuple[int, int, int],
    theoutputdir: str,
    debug: bool = False,
) -> dict:
    """Assemble the full fitSimFunc argument set for a synthetic volume.

    Parameters
    ----------
    theoptiondict : dict
        The option dictionary the run is configured by.
    thecorrout : NDArray
        Synthetic similarity functions, shape (numvoxels, numlags).
    thecorrscale : NDArray
        The lag axis, in seconds.
    theshape : tuple of int
        Native space shape of the volume.
    theoutputdir : str
        Directory the run may write its runoptions json into.
    debug : bool, optional
        Echo logger traffic to stdout.

    Returns
    -------
    dict
        Keyword arguments ready to splat into fitSimFunc.
    """
    thenumvoxels = int(np.prod(theshape))
    thenumlags = len(thecorrscale)
    thevalidvoxels = np.arange(thenumvoxels)

    thefitter = tide_simFuncClasses.SimilarityFunctionFitter(
        corrtimeaxis=thecorrscale,
        lagmin=theoptiondict["lagmin"],
        lagmax=theoptiondict["lagmax"],
        absmaxsigma=1000.0,
        absminsigma=0.25,
        bipolar=theoptiondict["bipolar"],
        peakfittype="gauss",
        zerooutbadfit=False,
    )

    thetimeaxis = np.linspace(0.0, 100.0, 100)
    return {
        "fmri_data_valid": np.zeros((thenumvoxels, 100)),
        "validsimcalcstart": 0,
        "validsimcalcend": 99,
        "osvalidsimcalcstart": 0,
        "osvalidsimcalcend": 99,
        "initial_fmri_x": thetimeaxis,
        "os_fmri_x": thetimeaxis,
        "theMutualInformationator": None,
        "cleaned_referencetc": np.zeros(100),
        "corrout": thecorrout,
        "outputname": f"{theoutputdir}/testout",
        "validvoxels": thevalidvoxels,
        "nativespaceshape": theshape,
        "bidsbasedict": {},
        "numspatiallocs": thenumvoxels,
        "gaussout": np.zeros((thenumvoxels, thenumlags)),
        "theinitialdelay": 0.0,
        "windowout": np.zeros((thenumvoxels, thenumlags)),
        "R2": np.zeros(thenumvoxels),
        "thesizes": np.array([3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]),
        "internalspaceshape": theshape,
        "numvalidspatiallocs": thenumvoxels,
        "theinputdata": _DummyInputData(),
        "theheader": {
            "dim": np.array([3, 8, 8, 8, 1, 1, 1, 1]),
            "pixdim": np.array([1.0] * 8),
        },
        "theFitter": thefitter,
        "fitmask": np.zeros(thenumvoxels, dtype=np.uint16),
        "lagtimes": np.zeros(thenumvoxels, dtype=np.float64),
        "lagstrengths": np.zeros(thenumvoxels, dtype=np.float64),
        "lagsigma": np.zeros(thenumvoxels, dtype=np.float64),
        "failreason": np.zeros(thenumvoxels, dtype=np.uint32),
        "outmaparray": np.zeros(thenumvoxels, dtype=np.float64),
        "trimmedcorrscale": thecorrscale,
        "similaritytype": "correlation",
        "thepass": 1,
        "optiondict": theoptiondict,
        "LGR": _DummyLogger(debug=debug),
        "TimingLGR": _DummyLogger(debug=debug),
    }


# ---------------------------------------------------------------------------
# fitSimFunc, simple fit path
# ---------------------------------------------------------------------------


def test_fitsimfunc_simplefit_locates_peak(debug: bool = False) -> None:
    """The simple fit path must return the upsampled argmax as the lag.

    It bypasses the fitter entirely, so this is the only check that its
    hand rolled basedelay plus index times step arithmetic is right.
    """
    theshape = (4, 4, 4)
    thetruelag = 4.0
    thecorrout, thecorrscale, dummy = _makecorrout(
        theshape=theshape, thebadvoxels=[], thetruelag=thetruelag
    )
    theoptiondict = _makeoptiondict()
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        theresult = fitSimFunc(**theargs, simplefit=True, upsampfac=8)

    thelagtimes = theargs["lagtimes"]
    if debug:
        print(f"recovered lags: min {thelagtimes.min()}, max {thelagtimes.max()}")

    # the upsampled grid step is the corrscale step over upsampfac
    thestep = (thecorrscale[1] - thecorrscale[0]) / 8.0
    assert np.allclose(thelagtimes, thetruelag, atol=thestep)
    # every voxel is marked fit, sigma is the fixed placeholder, strength is the peak
    assert np.all(theargs["fitmask"] == 1)
    assert np.all(theargs["lagsigma"] == 1.0)
    assert np.allclose(theargs["lagstrengths"], 0.75, atol=0.01)
    # the simple path does no despeckling, so it has no despeckle mask to hand back
    assert theresult is None


def test_fitsimfunc_simplefit_bipolar_picks_negative_peak(debug: bool = False) -> None:
    """Under bipolar the larger magnitude peak wins even when it is negative;
    without it, the positive peak must win."""
    theshape = (3, 3, 3)
    thenumvoxels = int(np.prod(theshape))
    thecorrscale = np.linspace(-15.0, 15.0, 121)
    thecorrout = np.zeros((thenumvoxels, len(thecorrscale)))
    thecorrout[:, :] = _gaussian(thecorrscale, 0.4, 3.0, 2.0) - _gaussian(
        thecorrscale, 0.9, -6.0, 2.0
    )

    theresults = {}
    for thebipolar in (False, True):
        theoptiondict = _makeoptiondict(bipolar=thebipolar)
        with tempfile.TemporaryDirectory() as thedir:
            theargs = _makefitargs(
                theoptiondict, thecorrout.copy(), thecorrscale, theshape, thedir, debug=debug
            )
            fitSimFunc(**theargs, simplefit=True, upsampfac=8)
        theresults[thebipolar] = (theargs["lagtimes"].copy(), theargs["lagstrengths"].copy())

    if debug:
        print(f"unipolar lag {theresults[False][0][0]}, bipolar lag {theresults[True][0][0]}")

    assert np.allclose(theresults[False][0], 3.0, atol=0.3), "unipolar must take the positive peak"
    assert np.all(theresults[False][1] > 0.0)
    assert np.allclose(theresults[True][0], -6.0, atol=0.3), "bipolar must take the deeper trough"
    assert np.all(theresults[True][1] < 0.0)


# ---------------------------------------------------------------------------
# fitSimFunc, full fit path
# ---------------------------------------------------------------------------


def test_fitsimfunc_without_despeckling_keeps_the_wrong_lobe(debug: bool = False) -> None:
    """Control arm.  With despeckling off the planted voxels must KEEP their wrong
    delay, which is what makes the repair test below meaningful rather than a
    restatement of what the fitter already does."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, thebadvoxels = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(despeckle_passes=0)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        theresult = fitSimFunc(**theargs)

    thelagtimes = theargs["lagtimes"]
    if debug:
        print(f"planted voxel lags: {thelagtimes[thebadvoxels]}")

    assert np.allclose(thelagtimes[thebadvoxels], 10.0, atol=0.5)
    thegoodvoxels = np.setdiff1d(np.arange(len(thelagtimes)), thebadvoxels)
    assert np.allclose(thelagtimes[thegoodvoxels], 0.0, atol=0.1)
    # no despeckling means no despeckle include mask
    assert theresult is None


def test_fitsimfunc_despeckling_repairs_planted_outliers(debug: bool = False) -> None:
    """The headline behaviour: voxels that picked the wrong similarity lobe must
    come back at the right delay after despeckling.

    This runs the real fitter and the real fitcorr, so it covers the whole chain:
    median filter target selection, the -1000000.0 sentinel that marks voxels to
    leave alone, the restricted range refit, and the range restoration afterwards.
    """
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, thebadvoxels = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(despeckle_passes=4)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        theresult = fitSimFunc(**theargs)

    thelagtimes = theargs["lagtimes"]
    if debug:
        print(f"planted voxel lags after despeckling: {thelagtimes[thebadvoxels]}")
        print({k: v for k, v in theoptiondict.items() if "despecklemask" in k})

    assert np.allclose(thelagtimes[thebadvoxels], 0.0, atol=0.1), "outliers were not repaired"
    # and the voxels that were already right must not have been disturbed
    thegoodvoxels = np.setdiff1d(np.arange(len(thelagtimes)), thebadvoxels)
    assert np.allclose(thelagtimes[thegoodvoxels], 0.0, atol=0.1)

    # exactly the planted voxels should have been refit on the first subpass
    assert theoptiondict["despecklemasksize_pass1_d1"] == len(thebadvoxels)
    assert np.isclose(
        theoptiondict["despecklemaskpct_pass1_d1"],
        100.0 * len(thebadvoxels) / theoptiondict["corrmasksize"],
    )

    # the include mask is returned for the caller to reuse, one entry per spatial loc
    assert theresult is not None
    assert theresult.shape == (int(np.prod(theshape)),)
    # everything converged, so nothing is left flagged
    assert np.count_nonzero(theresult) == 0

    # the fitter's range must be handed back intact, or the next pass fits in a
    # window left over from the last despeckled voxel
    assert theargs["theFitter"].lagmin == theoptiondict["lagmin"]
    assert theargs["theFitter"].lagmax == theoptiondict["lagmax"]


def test_fitsimfunc_despeckling_terminates_early_when_converged(debug: bool = False) -> None:
    """The convergence guard must stop the subpass loop once the flagged count
    stops falling, rather than burning through every requested subpass."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, thebadvoxels = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(despeckle_passes=8)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)

    therecorded = sorted(k for k in theoptiondict if k.startswith("despecklemasksize_"))
    if debug:
        print(f"subpasses that actually refit: {therecorded}")
        print([m for m in theargs["LGR"].messages if "Terminating" in m])

    # the outliers are gone after the first subpass, so no later subpass refits
    assert therecorded == ["despecklemasksize_pass1_d1"]
    assert any("Terminating despeckling" in m for m in theargs["LGR"].messages)


def test_fitsimfunc_despeckling_is_a_noop_on_a_clean_map(debug: bool = False) -> None:
    """With nothing to repair, despeckling must leave every delay untouched and
    record no refits."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape, thebadvoxels=[])
    theoptiondict = _makeoptiondict(despeckle_passes=4)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        theresult = fitSimFunc(**theargs)

    if debug:
        print(f"lag range {theargs['lagtimes'].min()} to {theargs['lagtimes'].max()}")
    assert np.allclose(theargs["lagtimes"], 0.0, atol=0.1)
    assert not [k for k in theoptiondict if k.startswith("despecklemasksize_")]
    assert np.count_nonzero(theresult) == 0


def test_fitsimfunc_despeckling_refuses_to_move_a_voxel_too_far(debug: bool = False) -> None:
    """A despeckle refit is bounded: the fitter's range is the target lag plus or
    minus half the despeckle threshold, and a refit that lands outside is thrown
    away rather than clamped in.

    The voxel here has no peak within reach of its neighbourhood median, so the
    honest outcome is to leave it alone.  Accepting the fit anyway would park it
    at the edge of the search window, manufacturing a delay that no peak supports.
    """
    theshape = (8, 8, 8)
    thenumvoxels = int(np.prod(theshape))
    thecorrscale = np.linspace(-15.0, 15.0, 121)
    thestubborn = [
        int(np.ravel_multi_index(thepoint, theshape)) for thepoint in ((2, 2, 2), (5, 5, 5))
    ]

    thecorrout = np.zeros((thenumvoxels, len(thecorrscale)))
    thecorrout[:, :] = _gaussian(thecorrscale, 0.75, 0.0, 2.0)
    for thevox in thestubborn:
        # nearest peak to the neighbourhood median of 0 sits at 4 s, well beyond
        # the 2.5 s the despeckle threshold of 5 s allows
        thecorrout[thevox, :] = _gaussian(thecorrscale, 0.55, 4.0, 1.5) + _gaussian(
            thecorrscale, 0.85, 10.0, 1.5
        )

    theoptiondict = _makeoptiondict(despeckle_passes=8, despeckle_thresh=5.0)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)

    thelagtimes = theargs["lagtimes"]
    if debug:
        print(f"stubborn voxel lags: {thelagtimes[thestubborn]}")
        print({k: v for k, v in theoptiondict.items() if "despecklemask" in k})

    # left at the lobe it originally chose, not dragged to the 2.5 s range edge
    # and not moved to the out of reach 4 s peak
    assert np.allclose(thelagtimes[thestubborn], 10.0, atol=0.5)

    # nothing was successfully repaired, and the convergence guard must notice that
    # the flagged count has stopped falling and stop, rather than refitting the same
    # voxels for all eight requested subpasses
    assert theoptiondict["despecklemasksize_pass1_d1"] == 0
    therecorded = sorted(k for k in theoptiondict if k.startswith("despecklemasksize_"))
    assert therecorded == ["despecklemasksize_pass1_d1"], "the convergence guard did not fire"


def test_fitsimfunc_despeckling_ignores_out_of_mask_neighbours(debug: bool = False) -> None:
    """Voxels that failed to fit must not vote in the despeckle median.

    A good voxel surrounded by failed fits is the case that matters: if the median
    counted them, its target would be their nonsense delay and the refit would drag
    a perfectly good voxel onto a sidelobe.  The fit mask is what prevents that, so
    plant exactly that geometry and require the good voxel to survive untouched.
    """
    theshape, thecorrout, thecorrscale, theoutofmask, thecentreindex = _makewalledinvolume()

    # First establish that the fixture really does come out with the planted
    # geometry, with despeckling off so nothing can perturb the fit mask.  If this
    # arm ever fails the fixture is broken, not the code under test.
    theoptiondict = _makeoptiondict(
        despeckle_passes=0, corrmasksize=int(np.prod(theshape)), lagmin=-10.0, lagmax=10.0
    )
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout.copy(), thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)

    if debug:
        print(f"out of mask voxels: {int((theargs['fitmask'] == 0).sum())}")
    assert np.array_equal(theargs["fitmask"].reshape(theshape) == 0, theoutofmask)
    assert theargs["fitmask"][thecentreindex] == 1
    assert np.isclose(theargs["lagtimes"][thecentreindex], 0.0, atol=0.1)

    # Now the behaviour: with despeckling on, the walled in voxel must survive.
    theoptiondict = _makeoptiondict(
        despeckle_passes=4, corrmasksize=int(np.prod(theshape)), lagmin=-10.0, lagmax=10.0
    )
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout.copy(), thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)

    thelagtimes = theargs["lagtimes"]
    if debug:
        print(f"centre voxel lag after despeckling: {thelagtimes[thecentreindex]}")

    # Counting the out of mask wall would put the median at -10, which is 10 s from
    # this voxel and so over the 5 s threshold; the refit would then find the
    # subsidiary peak at -9 and the delay would move there.
    assert np.isclose(
        thelagtimes[thecentreindex], 0.0, atol=0.1
    ), "an out of mask neighbourhood dragged an in mask voxel off its peak"


def test_fitsimfunc_resolvedelays_reassigns_and_records(debug: bool = False) -> None:
    """Delay resolution must run when asked, repair the planted lobe errors, and
    leave both a reassignment count and a preresolve audit trail behind."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, thebadvoxels = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(resolvedelays=True, resolvepasses=3, despeckle_passes=0)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)

    thelagtimes = theargs["lagtimes"]
    if debug:
        print(f"planted voxel lags after resolution: {thelagtimes[thebadvoxels]}")
        print({k: v for k, v in theoptiondict.items() if k.startswith("resolve")})

    assert theoptiondict["resolved_pass1"] == len(thebadvoxels)
    assert np.allclose(thelagtimes[thebadvoxels], 0.0, atol=0.3)

    # the shift metrics only exist because preresolvelags is captured
    # unconditionally; if that copy is ever made conditional again these go missing
    assert "resolveshiftmedian_pass1" in theoptiondict
    assert theoptiondict["resolveshiftmedian_pass1"] > 5.0
    # every planted error moved in the same direction, toward zero
    assert theoptiondict["resolveshiftposfrac_pass1"] == 0.0


def test_fitsimfunc_resolvedelays_off_by_default(debug: bool = False) -> None:
    """Resolution must not fire unless requested, and must not leave keys behind."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, thebadvoxels = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(despeckle_passes=0)
    theoptiondict.pop("resolvedelays")
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)

    if debug:
        print([k for k in theoptiondict if "resolve" in k])
    assert "resolved_pass1" not in theoptiondict
    assert not [k for k in theoptiondict if k.startswith("resolveshift")]
    # untouched by resolution, the planted errors survive
    assert np.allclose(theargs["lagtimes"][thebadvoxels], 10.0, atol=0.5)


def test_fitsimfunc_resolution_that_changes_nothing_records_no_shift(
    debug: bool = False,
) -> None:
    """Resolution on a map with nothing to repair must reassign nobody, and the
    shift metrics must be absent rather than present and zero, so a tabulation can
    tell "resolution did nothing" apart from "resolution moved things by zero"."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape, thebadvoxels=[])
    theoptiondict = _makeoptiondict(resolvedelays=True, despeckle_passes=0)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)

    if debug:
        print({k: v for k, v in theoptiondict.items() if k.startswith("resolve")})
    assert theoptiondict["resolved_pass1"] == 0
    assert "resolveshiftmedian_pass1" not in theoptiondict
    assert "resolveshiftposfrac_pass1" not in theoptiondict
    # the plain delay metrics are still recorded, they do not depend on resolution
    assert "delayoutlierfrac_pass1" in theoptiondict


def test_fitsimfunc_records_delay_metrics(debug: bool = False) -> None:
    """Every full fit pass must land quality metrics in the optiondict; they are
    the only record of map quality a completed run leaves behind."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(despeckle_passes=0, thepass=1)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        theargs["thepass"] = 2
        fitSimFunc(**theargs)

    if debug:
        print({k: v for k, v in theoptiondict.items() if k.startswith("delay")})
    for thekey in (
        "delayoutliers_pass2",
        "delayoutlierfrac_pass2",
        "delayjumps_pass2",
        "delayp50_pass2",
        "delayiqr_pass2",
        "delayrailedfrac_pass2",
        "delaylongfrac_pass2",
    ):
        assert thekey in theoptiondict, f"{thekey} was not recorded"

    # the planted voxels are 10 s from a neighbourhood sitting at 0, so they are
    # outliers by the 5 s despeckle threshold and each starts a jump in 6 directions
    assert theoptiondict["delayoutliers_pass2"] >= 4
    assert theoptiondict["delayjumps_pass2"] >= 24


def test_fitsimfunc_writes_runoptions_before_fitting(debug: bool = False) -> None:
    """The stage marker is written to disk BEFORE the fit, so that a run killed
    during fitting can still be diagnosed."""
    theshape = (4, 4, 4)
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape, thebadvoxels=[])
    theoptiondict = _makeoptiondict(despeckle_passes=0, corrmasksize=64)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        theargs["thepass"] = 3
        with patch.object(tide_io, "writedicttojson") as thewriter:
            fitSimFunc(**theargs)

    assert thewriter.call_count == 1
    thewrittendict, thewrittenname = thewriter.call_args[0]
    if debug:
        print(f"wrote {thewrittenname} at stage {thewrittendict['currentstage']}")
    assert thewrittenname.endswith("_desc-runoptions_info.json")
    assert thewrittendict["currentstage"] == "presimfuncfit_pass3"


def test_fitsimfunc_hybrid_seeds_fitcorr_with_mi_peaks(debug: bool = False) -> None:
    """Under the hybrid metric the mutual information prefit must supply the
    initial lags handed to fitcorr.  Nothing downstream reports this, so a
    regression here would quietly turn hybrid into plain correlation."""
    theshape = (2, 2, 2)
    thenumvoxels = int(np.prod(theshape))
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape, thebadvoxels=[])
    theoptiondict = _makeoptiondict(similaritymetric="hybrid", despeckle_passes=0)

    # one MI peak per voxel, at a distinct lag so a mix up would be visible
    thepeakdict = {str(i): [[float(i) - 3.0, 0.5]] for i in range(thenumvoxels)}
    theexpectedpeaks = np.array([float(i) - 3.0 for i in range(thenumvoxels)])

    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        with patch.object(
            tide_peakeval, "peakevalpass", return_value=(thenumvoxels, thepeakdict)
        ) as thepeakeval:
            with patch.object(tide_simfuncfit, "fitcorr", return_value=thenumvoxels) as thefitcorr:
                fitSimFunc(**theargs)

    assert thepeakeval.call_count == 1
    assert thefitcorr.call_count == 1
    thepassedlags = thefitcorr.call_args.kwargs["initiallags"]
    if debug:
        print(f"initial lags passed to fitcorr: {thepassedlags}")
    assert thepassedlags is not None
    assert np.allclose(thepassedlags, theexpectedpeaks)


def test_fitsimfunc_nonhybrid_passes_no_initial_lags(debug: bool = False) -> None:
    """The complement of the hybrid test: a plain correlation fit must start
    unconstrained, not from a stale peak list."""
    theshape = (2, 2, 2)
    thenumvoxels = int(np.prod(theshape))
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape, thebadvoxels=[])
    theoptiondict = _makeoptiondict(despeckle_passes=0)

    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        with patch.object(tide_peakeval, "peakevalpass") as thepeakeval:
            with patch.object(tide_simfuncfit, "fitcorr", return_value=thenumvoxels) as thefitcorr:
                fitSimFunc(**theargs)

    if debug:
        print(f"peakeval calls: {thepeakeval.call_count}")
    assert thepeakeval.call_count == 0, "the MI prefit must not run outside hybrid"
    assert thefitcorr.call_args.kwargs["initiallags"] is None


def test_fitsimfunc_saves_despeckle_masks_on_the_final_pass(debug: bool = False) -> None:
    """--savedespecklemasks must emit the diagnostic maps, and only on the final
    pass.  The header is mutated to 3D in place first, so check that too."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, thebadvoxels = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(savedespecklemasks=True, passes=2)

    # not the final pass: nothing should be written
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout.copy(), thecorrscale, theshape, thedir, debug=debug
        )
        theargs["thepass"] = 1
        with patch.object(tide_io, "savemaplist") as thesaver:
            fitSimFunc(**theargs)
    assert thesaver.call_count == 0, "diagnostics must not be written on a non-final pass"

    # the final pass: the per subpass maps plus the summary mask
    theoptiondict = _makeoptiondict(savedespecklemasks=True, passes=2)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout.copy(), thecorrscale, theshape, thedir, debug=debug
        )
        theargs["thepass"] = 2
        with patch.object(tide_io, "savemaplist") as thesaver:
            fitSimFunc(**theargs)

    thenames = [
        thename for thecall in thesaver.call_args_list for dummy, thename, *rest in thecall[0][1]
    ]
    if debug:
        print(f"saved maps: {thenames}")
    assert "despeckle_p2_d1" in thenames
    assert "despeckleinitlags_p2_d1" in thenames
    assert "despecklemedianlags_p2_d1" in thenames
    assert "despeckle" in thenames

    # the header must have been reduced to a single 3D volume
    theheader = theargs["theheader"]
    assert theheader["dim"][0] == 3
    assert theheader["dim"][4] == 1
    assert theheader["pixdim"][4] == 1.0


def test_fitsimfunc_saves_resolve_maps_on_the_final_pass(debug: bool = False) -> None:
    """--saveresolvemaps must emit the before map, the shift, and the changed
    mask, so that what the smoothness prior did stays inspectable."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, thebadvoxels = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(
        resolvedelays=True, saveresolvemaps=True, despeckle_passes=0, passes=1
    )
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        with patch.object(tide_io, "savemaplist") as thesaver:
            fitSimFunc(**theargs)

    assert thesaver.call_count == 1
    themaplist = thesaver.call_args[0][1]
    thenames = [thename for dummy, thename, *rest in themaplist]
    if debug:
        print(f"saved maps: {thenames}")
    assert thenames == ["maxtimepreresolve", "resolveshift", "resolvechanged"]

    # the before map must actually hold the pre resolution values, and the shift
    # must be the difference, not a copy of the after map
    thepreresolve = dict((thename, thedata) for thedata, thename, *rest in themaplist)
    assert np.allclose(thepreresolve["maxtimepreresolve"][thebadvoxels], 10.0, atol=0.5)
    assert np.allclose(thepreresolve["resolveshift"][thebadvoxels], -10.0, atol=0.5)
    assert np.count_nonzero(thepreresolve["resolvechanged"]) == len(thebadvoxels)

    # these are single volume 3D maps, so the header must say so before it is used
    theheader = theargs["theheader"]
    assert theheader["dim"][0] == 3
    assert theheader["dim"][4] == 1
    assert theheader["pixdim"][4] == 1.0


def _runciftipass(theoptiondict: dict, debug: bool = False) -> Tuple[Any, dict]:
    """Run one fitSimFunc pass over CIFTI style data with savemaplist captured.

    Parameters
    ----------
    theoptiondict : dict
        The option dictionary to run with.
    debug : bool, optional
        Echo logger traffic to stdout.

    Returns
    -------
    thesaver : MagicMock
        The patched savemaplist, for call inspection.
    theheader : dict
        The header dict as fitSimFunc left it.
    """
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        theargs["theinputdata"] = _DummyInputData(filetype="cifti")
        # a CIFTI header declares its dimensionality in dim[0]; the space extent
        # lives at dim[dim[0]] and the time extent one slot below it
        theargs["theheader"] = {"dim": np.array([6, 1, 1, 1, 1, 128, 64, 1])}
        with patch.object(tide_io, "savemaplist") as thesaver:
            fitSimFunc(**theargs)
    return thesaver, theargs["theheader"]


def test_fitsimfunc_cifti_output_reshapes_the_header(debug: bool = False) -> None:
    """CIFTI headers carry space and time in different slots than NIfTI ones, so
    every save path has a separate branch for them.  Cover all three.

    Note the two configurations are run separately on purpose: with resolution on,
    it repairs the planted lobe errors before despeckling ever sees them, so the
    per subpass despeckle save never fires.  That is the documented ordering, but
    it means one run cannot exercise both.
    """
    thenumvoxels = 8 * 8 * 8

    # resolution save path
    thesaver, theheader = _runciftipass(
        _makeoptiondict(resolvedelays=True, saveresolvemaps=True, despeckle_passes=0, passes=1),
        debug=debug,
    )
    if debug:
        print(f"resolve arm: header dim {theheader['dim']}, calls {thesaver.call_count}")
    assert thesaver.call_count == 1
    # time slot collapsed to a single volume, space slot set to the voxel count
    assert theheader["dim"][5] == 1
    assert theheader["dim"][6] == thenumvoxels

    # despeckle save paths: the per subpass maps and the summary mask
    thesaver, theheader = _runciftipass(
        _makeoptiondict(savedespecklemasks=True, despeckle_passes=4, passes=1), debug=debug
    )
    thenames = [
        thename for thecall in thesaver.call_args_list for dummy, thename, *rest in thecall[0][1]
    ]
    if debug:
        print(f"despeckle arm: header dim {theheader['dim']}, saved {thenames}")
    assert thesaver.call_count == 2
    assert "despeckle_p1_d1" in thenames and "despeckle" in thenames
    assert theheader["dim"][5] == 1
    assert theheader["dim"][6] == thenumvoxels
    for thecall in thesaver.call_args_list:
        assert thecall.kwargs["filetype"] == "cifti"


def test_fitsimfunc_resolution_runs_before_despeckling(debug: bool = False) -> None:
    """Order matters and is not symmetric: resolution makes the discrete lobe
    choice, then despeckling refits the residual outliers.  Doing it the other
    way round discards despeckle's non peak values."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(resolvedelays=True, despeckle_passes=4)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)

    themessages = [m for m in theargs["TimingLGR"].messages if "start, pass" in m]
    if debug:
        print(themessages)
    theresolveindex = next(i for i, m in enumerate(themessages) if "resolution start" in m)
    thedespeckleindex = next(i for i, m in enumerate(themessages) if "despeckle start" in m)
    assert theresolveindex < thedespeckleindex


def test_fitsimfunc_tolerates_missing_sidelobe_measurement(debug: bool = False) -> None:
    """The sidelobe amplitude is frequently absent from the optiondict.  Delay
    resolution no longer gates on it, and must not assume it is a number."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape)

    # absent
    theoptiondict = _makeoptiondict(resolvedelays=True, despeckle_passes=0)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout.copy(), thecorrscale, theshape, thedir, debug=debug
        )
        fitSimFunc(**theargs)
    assert any("no sidelobe amplitude measured" in m for m in theargs["LGR"].messages)

    # present, with and without a lag to go with it
    for thelag in (None, 6.5):
        theoptiondict = _makeoptiondict(resolvedelays=True, despeckle_passes=0)
        theoptiondict["acsidelobeamp_pass1"] = 0.42
        theoptiondict["acsidelobelag_pass1"] = thelag
        with tempfile.TemporaryDirectory() as thedir:
            theargs = _makefitargs(
                theoptiondict, thecorrout.copy(), thecorrscale, theshape, thedir, debug=debug
            )
            fitSimFunc(**theargs)
        themessage = [m for m in theargs["LGR"].messages if "sidelobe amplitude" in m][0]
        if debug:
            print(themessage)
        assert "0.420" in themessage
        assert ("6.500s" in themessage) == (thelag is not None)


def test_fitsimfunc_text_input_skips_header_munging(debug: bool = False) -> None:
    """Text input has no NIfTI header to reshape; touching it would raise."""
    theshape = (8, 8, 8)
    thecorrout, thecorrscale, dummy = _makecorrout(theshape=theshape)
    theoptiondict = _makeoptiondict(savedespecklemasks=True, passes=1)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makefitargs(
            theoptiondict, thecorrout, thecorrscale, theshape, thedir, debug=debug
        )
        theargs["theinputdata"] = _DummyInputData(filetype="text")
        theargs["theheader"] = None
        with patch.object(tide_io, "savemaplist") as thesaver:
            fitSimFunc(**theargs)

    if debug:
        print(f"savemaplist calls: {thesaver.call_count}")
    assert thesaver.call_count == 2


def test_recorddelaymetrics_logs_the_reason_it_failed(debug: bool = False) -> None:
    """Metrics are instrumentation and swallow their own exceptions, so a silent
    swallow would hide a real bug forever.  When a logger is supplied the reason
    must reach it."""
    thelogger = _DummyLogger(debug=debug)
    theoptiondict: dict = {}
    tide_fitSimFuncMap.recorddelaymetrics(
        theoptiondict,
        1,
        np.zeros(10),
        np.ones(10),
        np.arange(10),
        (4, 4, 4),  # inconsistent with the numspatiallocs below
        1000,
        5.0,
        LGR=thelogger,
    )
    if debug:
        print(thelogger.messages)
    assert not theoptiondict
    assert any("delay metrics for pass 1 FAILED" in m for m in thelogger.messages)


if __name__ == "__main__":
    test_masked_median_filter_none_mask_matches_scipy(debug=True)
    test_masked_median_filter_pad_mode_mapping(debug=True)
    test_masked_median_filter_ignores_out_of_mask_voxels(debug=True)
    test_masked_median_filter_chunking_is_transparent(debug=True)
    test_masked_median_filter_empty_window_is_nan(debug=True)
    test_masked_median_filter_anisotropic_kernel(debug=True)
    test_fitsimfunc_simplefit_locates_peak(debug=True)
    test_fitsimfunc_simplefit_bipolar_picks_negative_peak(debug=True)
    test_fitsimfunc_without_despeckling_keeps_the_wrong_lobe(debug=True)
    test_fitsimfunc_despeckling_repairs_planted_outliers(debug=True)
    test_fitsimfunc_despeckling_terminates_early_when_converged(debug=True)
    test_fitsimfunc_despeckling_is_a_noop_on_a_clean_map(debug=True)
    test_fitsimfunc_despeckling_refuses_to_move_a_voxel_too_far(debug=True)
    test_fitsimfunc_despeckling_ignores_out_of_mask_neighbours(debug=True)
    test_fitsimfunc_resolvedelays_reassigns_and_records(debug=True)
    test_fitsimfunc_resolvedelays_off_by_default(debug=True)
    test_fitsimfunc_resolution_that_changes_nothing_records_no_shift(debug=True)
    test_fitsimfunc_records_delay_metrics(debug=True)
    test_fitsimfunc_writes_runoptions_before_fitting(debug=True)
    test_fitsimfunc_hybrid_seeds_fitcorr_with_mi_peaks(debug=True)
    test_fitsimfunc_nonhybrid_passes_no_initial_lags(debug=True)
    test_fitsimfunc_saves_despeckle_masks_on_the_final_pass(debug=True)
    test_fitsimfunc_saves_resolve_maps_on_the_final_pass(debug=True)
    test_fitsimfunc_cifti_output_reshapes_the_header(debug=True)
    test_fitsimfunc_resolution_runs_before_despeckling(debug=True)
    test_fitsimfunc_tolerates_missing_sidelobe_measurement(debug=True)
    test_fitsimfunc_text_input_skips_header_munging(debug=True)
    test_recorddelaymetrics_logs_the_reason_it_failed(debug=True)
