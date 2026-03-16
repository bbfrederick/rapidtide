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
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.simFuncClasses as tide_simFuncClasses
import rapidtide.simfuncfit as tide_simfuncfit


def _compute_acf(signal: NDArray, oversamptr: float, lagmax: float) -> Tuple[NDArray, NDArray]:
    """Compute a normalised autocorrelation function for *signal*.

    Parameters
    ----------
    signal : NDArray
        Input timecourse at oversampled resolution.
    oversamptr : float
        Oversampled TR (seconds per sample).
    lagmax : float
        Maximum lag (seconds) to retain in the output.

    Returns
    -------
    lags : NDArray
        Lag axis in seconds, shape (2*lag_max_pts + 1,).
    acf : NDArray
        Normalised ACF values, shape (2*lag_max_pts + 1,).
    """
    n = len(signal)
    lag_max_pts = min(int(lagmax / oversamptr), n - 1)
    sig = tide_math.stdnormalize(signal)
    full = np.correlate(sig, sig, mode="full") / n
    center = n - 1
    acf = full[center - lag_max_pts : center + lag_max_pts + 1]
    lags = np.arange(-lag_max_pts, lag_max_pts + 1) * oversamptr
    return lags, acf


def sharpen_regressor(
    resampref_y: NDArray,
    oversamptr: float,
    oversampfreq: float,
    lagmax: float,
    numpadtrs: int,
    outputname: str,
    noise_level: float = 0.01,
    max_iters: int = 5,
    ampthresh: float = 0.2,
    acwidth: float = 2.0,
    debug: bool = False,
) -> NDArray:
    """Remove ACF sidelobe structure from *resampref_y* using Wiener deconvolution.

    Attempts Wiener deconvolution of the regressor against its own ACF
    structure, then falls back to iterative multi-echo subtraction if the
    Wiener step does not reduce sidelobe amplitude by at least 20 %.

    Parameters
    ----------
    resampref_y : NDArray
        The regressor at oversampled resolution.
    oversamptr : float
        Oversampled TR in seconds.
    oversampfreq : float
        Oversampled sampling frequency (Hz).
    lagmax : float
        Maximum lag searched (seconds); used for ACF window.
    numpadtrs : int
        Padding timepoints for ``tide_resample.timeshift``.
    outputname : str
        Output file base name for saving the sharpened regressor.
    noise_level : float, optional
        Wiener regularisation parameter λ.  Default 0.01.
    max_iters : int, optional
        Maximum iterations for multi-echo fallback.  Default 5.
    ampthresh : float, optional
        Sidelobe amplitude threshold (fraction of central peak).  Default 0.2.
    acwidth : float, optional
        Half-width of the central peak region excluded from sidelobe detection
        (seconds).  Default 2.0.
    debug : bool, optional
        Print debug information.  Default False.

    Returns
    -------
    NDArray
        Sharpened regressor, same length as *resampref_y*.
    """
    LGR_local = logging.getLogger("rapidtide")

    # --- Step 1: compute regressor ACF ---
    acf_lags, acf = _compute_acf(resampref_y, oversamptr, lagmax)

    # check if there is anything to sharpen
    sidelobes_before = tide_corr.find_all_acf_sidelobes(
        acf_lags, acf, ampthresh=ampthresh, acwidth=acwidth, debug=debug
    )
    if not sidelobes_before:
        LGR_local.info("sharpen_regressor: no significant sidelobes found; skipping")
        return resampref_y

    LGR_local.info(f"sharpen_regressor: {len(sidelobes_before)} sidelobe(s) before sharpening")

    # --- Step 2: attempt Wiener deconvolution ---
    sharpened = resampref_y.copy()
    wiener_succeeded = False
    try:
        n = len(resampref_y)
        # Fit a Gaussian to the central peak of the ACF to get ACF_s
        zero_idx = np.argmin(np.abs(acf_lags))
        central_amp = acf[zero_idx]
        # estimate central-peak half-width
        hw = 1
        while zero_idx + hw < len(acf) - 1 and acf[zero_idx + hw] > central_amp * 0.5:
            hw += 1
        central_width = acf_lags[zero_idx + hw] * 2.0

        acf_gauss = tide_fit.gauss_eval(
            acf_lags, [central_amp, 0.0, max(central_width, oversamptr * 2)]
        )
        if debug:
            print(f"sharpen_regressor: central peak ACF: {acf_gauss}")

        # Wiener deconvolution in frequency domain
        R_f = np.fft.rfft(resampref_y, n=n)
        H_f = np.fft.rfft(acf, n=n) / (np.fft.rfft(acf_gauss, n=n) + 1e-30)
        lam = noise_level * np.max(np.abs(H_f))
        S_hat_f = R_f / (H_f + lam)
        sharpened_wiener = np.fft.irfft(S_hat_f, n=n)

        # normalise to same RMS as original
        rms_orig = np.std(resampref_y)
        rms_sharp = np.std(sharpened_wiener)
        if rms_sharp > 1e-10:
            sharpened_wiener *= rms_orig / rms_sharp

        # --- Step 3: validate ---
        acf_lags_w, acf_w = _compute_acf(sharpened_wiener, oversamptr, lagmax)
        sidelobes_after = tide_corr.find_all_acf_sidelobes(
            acf_lags_w, acf_w, ampthresh=ampthresh, acwidth=acwidth, debug=debug
        )
        before_max = max(abs(s[1]) for s in sidelobes_before)
        after_max = max((abs(s[1]) for s in sidelobes_after), default=0.0)
        if after_max < 0.8 * before_max:
            sharpened = sharpened_wiener
            wiener_succeeded = True
            LGR_local.info(
                f"sharpen_regressor: Wiener succeeded "
                f"(sidelobe max {before_max:.3f} -> {after_max:.3f})"
            )
    except Exception as exc:
        LGR_local.warning(
            f"sharpen_regressor: Wiener deconvolution failed ({exc}); using fallback"
        )

    # --- Step 4: multi-echo fallback ---
    if not wiener_succeeded:
        LGR_local.info("sharpen_regressor: using multi-echo iterative fallback")
        current = resampref_y.copy()
        for iteration in range(max_iters):
            acf_lags_c, acf_c = _compute_acf(current, oversamptr, lagmax)
            sidelobes = tide_corr.find_all_acf_sidelobes(
                acf_lags_c, acf_c, ampthresh=ampthresh, acwidth=acwidth, debug=debug
            )
            if not sidelobes:
                LGR_local.info(f"sharpen_regressor: fallback converged at iteration {iteration}")
                break
            for tau, amp in sidelobes:
                shifttr = tau / oversamptr
                echotc, _, _, _ = tide_resample.timeshift(current, shifttr, numpadtrs)
                # zero out the extrapolated region
                nshift = int(np.ceil(abs(shifttr)))
                if shifttr > 0:
                    echotc[:nshift] = 0.0
                elif shifttr < 0:
                    echotc[-nshift:] = 0.0
                echofit, _ = tide_fit.mlregress(echotc, current)
                current -= echofit[0, 1] * echotc
        sharpened = current

    # --- Step 5: save and return ---
    tide_io.writebidstsv(
        f"{outputname}_desc-sharpenedregressor_timeseries",
        np.vstack(
            (
                tide_math.stdnormalize(resampref_y),
                tide_math.stdnormalize(sharpened),
            )
        ),
        oversampfreq,
        columns=["original", "sharpened"],
        extraheaderinfo={
            "Description": "Original and sharpened probe regressor (ACF sidelobe removal)"
        },
        append=False,
    )
    return sharpened


def cleanregressor(
    outputname: Any,
    thepass: Any,
    referencetc: Any,
    resampref_y: Any,
    resampnonosref_y: Any,
    fmrifreq: Any,
    oversampfreq: Any,
    osvalidsimcalcstart: Any,
    osvalidsimcalcend: Any,
    lagmininpts: Any,
    lagmaxinpts: Any,
    theFitter: Any,
    theCorrelator: Any,
    lagmin: Any,
    lagmax: Any,
    LGR: Optional[Any] = None,
    check_autocorrelation: bool = True,
    fix_autocorrelation: bool = True,
    despeckle_thresh: float = 5.0,
    lthreshval: float = 0.0,
    fixdelay: bool = False,
    detrendorder: int = 3,
    windowfunc: str = "hamming",
    respdelete: bool = False,
    displayplots: bool = False,
    debug: bool = False,
    rt_floattype: np.dtype = np.float64,
) -> Tuple[
    NDArray,
    NDArray,
    NDArray,
    float,
    float | None,
    float | None,
    float,
    float | None,
    float | None,
]:
    """
    Clean and preprocess a regressor signal by checking and correcting autocorrelation properties.

    This function performs several operations on the input regressor signal, including:
    detrending, normalization, optional filtering to remove periodic components, and
    autocorrelation analysis to detect and correct sidelobes. It returns cleaned versions
    of the regressor and associated metadata for further use in time series analysis.

    Parameters
    ----------
    outputname : Any
        Base name for output files.
    thepass : Any
        Pass identifier, used for labeling output files.
    referencetc : Any
        Reference time course data (normalized).
    resampref_y : Any
        Resampled reference signal.
    resampnonosref_y : Any
        Non-oversampled reference signal.
    fmrifreq : Any
        fMRI sampling frequency.
    oversampfreq : Any
        Oversampled frequency.
    osvalidsimcalcstart : Any
        Start index for valid data in oversampled signal.
    osvalidsimcalcend : Any
        End index for valid data in oversampled signal.
    lagmininpts : Any
        Minimum lag in samples for autocorrelation calculation.
    lagmaxinpts : Any
        Maximum lag in samples for autocorrelation calculation.
    theFitter : Any
        Fitter object for fitting autocorrelation data.
    theCorrelator : Any
        Correlator object for computing cross-correlations.
    lagmin : Any
        Minimum lag in seconds.
    lagmax : Any
        Maximum lag in seconds.
    LGR : Optional[Any], optional
        Logger object for logging messages. Default is None.
    check_autocorrelation : bool, optional
        If True, perform autocorrelation checks. Default is True.
    fix_autocorrelation : bool, optional
        If True, attempt to fix detected autocorrelation issues. Default is True.
    despeckle_thresh : float, optional
        Threshold for despeckling autocorrelation data. Default is 5.0.
    lthreshval : float, optional
        Low threshold value for fitting. Default is 0.0.
    fixdelay : bool, optional
        If True, fix delay in fitting. Default is False.
    detrendorder : int, optional
        Order of detrending polynomial. Default is 3.
    windowfunc : str, optional
        Window function to use for normalization. Default is "hamming".
    respdelete : bool, optional
        If True, remove periodic components from the reference signal. Default is False.
    displayplots : bool, optional
        If True, display plots during processing. Default is False.
    debug : bool, optional
        If True, print debugging information. Default is False.
    rt_floattype : np.dtype, optional
        Float type setting for rapidtide processing. Default is np.float64.

    Returns
    -------
    tuple
        A tuple containing:
        - cleaned_resampref_y : NDArray
            Cleaned resampled reference signal.
        - cleaned_referencetc : NDArray
            Cleaned reference time course.
        - cleaned_nonosreferencetc : NDArray
            Cleaned non-oversampled reference signal.
        - despeckle_thresh : float
            Updated despeckle threshold.
        - sidelobeamp : float or None
            Amplitude of detected sidelobe, or None if not found.
        - sidelobetime : float or None
            Time of detected sidelobe in seconds, or None if not found.
        - lagmod : float
            Lag modulus value used for correction.
        - acwidth : float or None
            Width of autocorrelation function, or None if not computed.
        - absmaxsigma : float or None
            Absolute maximum sigma value, or None if not computed.

    Notes
    -----
    - If `respdelete` is True, the function applies frequency tracking and filtering to remove
      periodic components from the reference signal.
    - Autocorrelation analysis is performed using `tide_corr.check_autocorrelation` to detect
      sidelobes that may affect the regressor quality.
    - If `fix_autocorrelation` is True, detected sidelobes are corrected by applying a notch filter
      and adjusting the lag modulus.

    Examples
    --------
    >>> cleanregressor(
    ...     outputname="test_output",
    ...     thepass=1,
    ...     referencetc=ref_tc,
    ...     resampref_y=resamp_ref,
    ...     resampnonosref_y=resamp_nonos_ref,
    ...     fmrifreq=2.0,
    ...     oversampfreq=10.0,
    ...     osvalidsimcalcstart=0,
    ...     osvalidsimcalcend=100,
    ...     lagmininpts=5,
    ...     lagmaxinpts=20,
    ...     theFitter=fitter,
    ...     theCorrelator=correlator,
    ...     lagmin=-10,
    ...     lagmax=10,
    ...     check_autocorrelation=True,
    ...     fix_autocorrelation=True,
    ...     detrendorder=3,
    ...     windowfunc="hamming",
    ...     respdelete=False,
    ...     displayplots=False,
    ...     debug=False,
    ... )
    """
    # print debugging info
    if debug:
        print("cleanregressor:")
        print(f"\t{thepass=}")
        print(f"\t{lagmininpts=}")
        print(f"\t{lagmaxinpts=}")
        print(f"\t{lagmin=}")
        print(f"\t{lagmax=}")
        print(f"\t{detrendorder=}")
        print(f"\t{windowfunc=}")
        print(f"\t{respdelete=}")
        print(f"\t{check_autocorrelation=}")
        print(f"\t{fix_autocorrelation=}")
        print(f"\t{despeckle_thresh=}")
        print(f"\t{lthreshval=}")
        print(f"\t{fixdelay=}")
        print(f"\t{check_autocorrelation=}")
        print(f"\t{displayplots=}")
        print(f"\t{rt_floattype=}")

    # check the regressor for periodic components in the passband
    dolagmod = True
    doreferencenotch = True
    if respdelete:
        resptracker = tide_simFuncClasses.FrequencyTracker(nperseg=64)
        thetimes, thefreqs = resptracker.track(resampref_y, oversampfreq)
        tide_io.writevec(thefreqs, f"{outputname}_peakfreaks_pass{thepass}.txt")
        resampref_y = resptracker.clean(resampref_y, oversampfreq, thetimes, thefreqs)
        tide_io.writevec(resampref_y, f"{outputname}_respfilt_pass{thepass}.txt")
        referencetc = tide_math.corrnormalize(
            resampref_y[osvalidsimcalcstart : osvalidsimcalcend + 1],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )

    if check_autocorrelation:
        if LGR is not None:
            LGR.info("checking reference regressor autocorrelation properties")
        lagmod = 1000.0
        lagindpad = np.max((lagmininpts, lagmaxinpts))
        acmininpts = lagindpad
        acmaxinpts = lagindpad
        theCorrelator.setreftc(referencetc)
        theCorrelator.setlimits(acmininpts, acmaxinpts)
        # theCorrelator.setlimits(lagmininpts, lagmaxinpts)
        print("check_autocorrelation:", acmininpts, acmaxinpts, lagmininpts, lagmaxinpts)
        thexcorr, accheckcorrscale, theglobalmax = theCorrelator.run(
            resampref_y[osvalidsimcalcstart : osvalidsimcalcend + 1],
            trim=False,
        )
        theFitter.setcorrtimeaxis(accheckcorrscale)
        (
            dummy,
            dummy,
            dummy,
            acwidth,
            dummy,
            dummy,
            dummy,
            dummy,
        ) = tide_simfuncfit.onesimfuncfit(
            thexcorr,
            theFitter,
            despeckle_thresh=despeckle_thresh,
            lthreshval=lthreshval,
            fixdelay=fixdelay,
            rt_floattype=rt_floattype,
        )
        tide_io.writebidstsv(
            f"{outputname}_desc-autocorr_timeseries",
            thexcorr,
            1.0 / (accheckcorrscale[1] - accheckcorrscale[0]),
            starttime=accheckcorrscale[0],
            extraheaderinfo={
                "Description": "Autocorrelation of the probe regressor for each pass"
            },
            columns=[f"pass{thepass}"],
            append=(thepass > 1),
        )
        thelagthresh = np.max((abs(lagmin), abs(lagmax)))
        theampthresh = 0.1
        if LGR is not None:
            LGR.info(
                f"searching for sidelobes with amplitude > {theampthresh} "
                f"with abs(lag) < {thelagthresh} s"
            )
        if debug:
            print(
                (
                    f"searching for sidelobes with amplitude > {theampthresh} "
                    f"with abs(lag) < {thelagthresh} s"
                )
            )
        sidelobetime, sidelobeamp = tide_corr.check_autocorrelation(
            accheckcorrscale,
            thexcorr,
            acampthresh=theampthresh,
            aclagthresh=thelagthresh,
            detrendorder=detrendorder,
            displayplots=displayplots,
            debug=debug,
        )
        if debug:
            print(f"check_autocorrelation returned: {sidelobetime=}, {sidelobeamp=}")
        absmaxsigma = acwidth * 10.0
        passsuffix = "_pass" + str(thepass)
        if sidelobetime is not None:
            despeckle_thresh = np.max([despeckle_thresh, sidelobetime / 2.0])
            if LGR is not None:
                LGR.warning(
                    f"\n\nWARNING: check_autocorrelation found bad sidelobe at {sidelobetime} "
                    f"seconds ({1.0 / sidelobetime} Hz)..."
                )
            # bidsify
            """tide_io.writebidstsv(
                f"{outputname}_desc-movingregressor_timeseries",
                tide_math.stdnormalize(resampnonosref_y),
                1.0 / fmritr,
                columns=["pass1"],
                append=False,
            )"""
            tide_io.writenpvecs(
                np.array([sidelobetime]),
                f"{outputname}_autocorr_sidelobetime" + passsuffix + ".txt",
            )
            if fix_autocorrelation:
                if LGR is not None:
                    LGR.info("Removing sidelobe")
                if dolagmod:
                    if LGR is not None:
                        LGR.info("subjecting lag times to modulus")
                    lagmod = sidelobetime / 2.0
                if doreferencenotch:
                    if LGR is not None:
                        LGR.info("removing spectral component at sidelobe frequency")
                    acstopfreq = 1.0 / sidelobetime
                    acfixfilter = tide_filt.NoncausalFilter(
                        debug=False,
                    )
                    acfixfilter.settype("arb_stop")
                    acfixfilter.setfreqs(
                        acstopfreq * 0.9,
                        acstopfreq * 0.95,
                        acstopfreq * 1.05,
                        acstopfreq * 1.1,
                    )
                    cleaned_resampref_y = tide_math.corrnormalize(
                        acfixfilter.apply(oversampfreq, resampref_y),
                        windowfunc="None",
                        detrendorder=detrendorder,
                    )
                    cleaned_referencetc = tide_math.corrnormalize(
                        cleaned_resampref_y[osvalidsimcalcstart : osvalidsimcalcend + 1],
                        detrendorder=detrendorder,
                        windowfunc=windowfunc,
                    )
                    cleaned_nonosreferencetc = tide_math.stdnormalize(
                        acfixfilter.apply(fmrifreq, resampnonosref_y)
                    )
                    tide_io.writebidstsv(
                        f"{outputname}_desc-cleanedreferencefmrires_info",
                        cleaned_nonosreferencetc,
                        fmrifreq,
                        columns=[f"pass{thepass}"],
                        append=(thepass > 1),
                    )
                    tide_io.writebidstsv(
                        f"{outputname}_desc-cleanedreference_info",
                        cleaned_referencetc,
                        oversampfreq,
                        columns=[f"pass{thepass}"],
                        append=(thepass > 1),
                    )
                    tide_io.writebidstsv(
                        f"{outputname}_desc-cleanedresamprefy_info",
                        cleaned_resampref_y,
                        oversampfreq,
                        columns=[f"pass{thepass}"],
                        append=(thepass > 1),
                    )
            else:
                cleaned_resampref_y = 1.0 * tide_math.corrnormalize(
                    resampref_y,
                    windowfunc="None",
                    detrendorder=detrendorder,
                )
                cleaned_referencetc = 1.0 * referencetc
                cleaned_nonosreferencetc = 1.0 * resampnonosref_y
        else:
            if LGR is not None:
                LGR.info("no sidelobes found in range")
            cleaned_resampref_y = 1.0 * tide_math.corrnormalize(
                resampref_y,
                windowfunc="None",
                detrendorder=detrendorder,
            )
            cleaned_referencetc = 1.0 * referencetc
            cleaned_nonosreferencetc = 1.0 * resampnonosref_y
    else:
        sidelobetime = None
        sidelobeamp = None
        lagmod = 1000.0
        acwidth = None
        absmaxsigma = None
        cleaned_resampref_y = 1.0 * tide_math.corrnormalize(
            resampref_y, windowfunc="None", detrendorder=detrendorder
        )
        cleaned_referencetc = 1.0 * referencetc
        cleaned_nonosreferencetc = 1.0 * resampnonosref_y

    return (
        cleaned_resampref_y,
        cleaned_referencetc,
        cleaned_nonosreferencetc,
        despeckle_thresh,
        sidelobeamp,
        sidelobetime,
        lagmod,
        acwidth,
        absmaxsigma,
    )
