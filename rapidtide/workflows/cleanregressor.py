#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.helper_classes as tide_classes
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.simfuncfit as tide_simfuncfit


def cleanregressor(
    outputname,
    thepass,
    referencetc,
    resampref_y,
    resampnonosref_y,
    fmrifreq,
    oversampfreq,
    osvalidsimcalcstart,
    osvalidsimcalcend,
    lagmininpts,
    lagmaxinpts,
    theFitter,
    theCorrelator,
    lagmin,
    lagmax,
    LGR=None,
    check_autocorrelation=True,
    fix_autocorrelation=True,
    despeckle_thresh=5.0,
    lthreshval=0.0,
    fixdelay=False,
    detrendorder=3,
    windowfunc="hamming",
    respdelete=False,
    displayplots=False,
    debug=False,
    rt_floattype="float64",
    rt_floatset=np.float64,
):
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
        print(f"\t{rt_floatset=}")

    # check the regressor for periodic components in the passband
    dolagmod = True
    doreferencenotch = True
    if respdelete:
        resptracker = tide_classes.FrequencyTracker(nperseg=64)
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
        thexcorr, accheckcorrscale, dummy = theCorrelator.run(
            resampref_y[osvalidsimcalcstart : osvalidsimcalcend + 1]
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
            rt_floatset=rt_floatset,
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
