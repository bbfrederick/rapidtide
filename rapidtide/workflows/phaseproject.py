import time

import numpy as np

import rapidtide.stats as tide_stats
import rapidtide.util as tide_util


def phaseproject(
    datatoproject,
    means,
    destpoints,
    numsteps,
    centric=True,
    passstring="",
    badpointlist=None,
    congridbins=3.0,
    gridkernel="kaiser",
):

    xsize, ysize, numslices, timepoints = datatoproject.shape
    # construct the destination arrays
    tide_util.logmem("before making destination arrays")
    app = np.zeros((xsize, ysize, numslices, destpoints), dtype=np.float64)
    app_byslice = app.reshape((xsize * ysize, numslices, destpoints))
    cine = np.zeros((xsize, ysize, numslices, destpoints), dtype=np.float64)
    cine_byslice = cine.reshape((xsize * ysize, numslices, destpoints))
    rawapp = np.zeros((xsize, ysize, numslices, destpoints), dtype=np.float64)
    rawapp_byslice = rawapp.reshape((xsize * ysize, numslices, destpoints))
    corrected_rawapp = np.zeros((xsize, ysize, numslices, destpoints), dtype=np.float64)
    corrected_rawapp_byslice = rawapp.reshape((xsize * ysize, numslices, destpoints))
    normapp = np.zeros((xsize, ysize, numslices, destpoints), dtype=np.float64)
    normapp_byslice = normapp.reshape((xsize * ysize, numslices, destpoints))
    weights = np.zeros((xsize, ysize, numslices, destpoints), dtype=np.float64)
    weight_byslice = weights.reshape((xsize * ysize, numslices, destpoints))
    derivatives = np.zeros((xsize, ysize, numslices, 4), dtype=np.float64)
    derivatives_byslice = derivatives.reshape((xsize * ysize, numslices, 4))

    timings.append(["Output arrays allocated" + passstring, time.time(), None, None])

    if centric:
        outphases = np.linspace(-np.pi, np.pi, num=destpoints, endpoint=False)
    else:
        outphases = np.linspace(0.0, 2.0 * np.pi, num=destpoints, endpoint=False)
    phasestep = outphases[1] - outphases[0]

    #######################################################################################################
    #
    # now do the phase projection
    #
    #
    datatoproject_byslice = datatoproject.reshape((xsize * ysize, numslices, timepoints))
    means_byslice = means.reshape((xsize * ysize, numslices))

    timings.append(["Phase projection to image started" + passstring, time.time(), None, None])
    print("Starting phase projection")
    proctrs = range(timepoints)  # proctrs is the list of all fmri trs to be projected
    procpoints = range(
        timepoints * numsteps
    )  # procpoints is the list of all sliceres datapoints to be projected
    if badpointlist is not None:
        censortrs = np.zeros(timepoints, dtype="int")
        censorpoints = np.zeros(timepoints * numsteps, dtype="int")
        censortrs[np.where(badpointlist > 0.0)[0] // numsteps] = 1
        censorpoints[np.where(badpointlist > 0.0)[0]] = 1
        proctrs = np.where(censortrs < 1)[0]
        procpoints = np.where(censorpoints < 1)[0]

    # do phase averaging
    app_bypoint = cardiaccycleaverage(
        instantaneous_cardiacphase,
        outphases,
        cardfromfmri_sliceres,
        procpoints,
        congridbins,
        gridkernel,
        centric,
        cyclic=True,
    )
    if thispass == numpasses - 1:
        if args.bidsoutput:
            tide_io.writebidstsv(
                outputroot + "_desc-cardiaccyclefromfmri_timeseries",
                app_bypoint,
                1.0 / (outphases[1] - outphases[0]),
                starttime=outphases[0],
                columns=["cardiaccyclefromfmri"],
                append=False,
                debug=args.debug,
            )
        else:
            tide_io.writevec(app_bypoint, outputroot + "_cardcyclefromfmri.txt")

    # now do time averaging
    lookaheadval = int(slicesamplerate / 4.0)
    print("lookaheadval = ", lookaheadval)
    wrappedcardiacphase = tide_math.phasemod(instantaneous_cardiacphase, centric=centric)
    max_peaks, min_peaks = tide_fit.peakdetect(wrappedcardiacphase, lookahead=lookaheadval)
    # start on a maximum
    if max_peaks[0][0] > min_peaks[0][0]:
        min_peaks = min_peaks[1:]
    # work only with pairs
    if len(max_peaks) > len(min_peaks):
        max_peaks = max_peaks[:-1]

    zerophaselocs = []
    for idx, peak in enumerate(max_peaks):
        minloc = min_peaks[idx][0]
        maxloc = max_peaks[idx][0]
        minval = min_peaks[idx][1]
        maxval = max_peaks[idx][1]
        if minloc > 0:
            if wrappedcardiacphase[minloc - 1] < wrappedcardiacphase[minloc]:
                minloc -= 1
                minval = wrappedcardiacphase[minloc]
        phasediff = minval - (maxval - 2.0 * np.pi)
        timediff = minloc - maxloc
        zerophaselocs.append(1.0 * minloc - (minval - outphases[0]) * timediff / phasediff)
        # print(idx, [maxloc, maxval], [minloc, minval], phasediff, timediff, zerophaselocs[-1])
    instantaneous_cardiactime = instantaneous_cardiacphase * 0.0

    whichpeak = 0
    for t in procpoints:
        if whichpeak < len(zerophaselocs) - 1:
            if t > zerophaselocs[whichpeak + 1]:
                whichpeak += 1
        if t > zerophaselocs[whichpeak]:
            instantaneous_cardiactime[t] = (t - zerophaselocs[whichpeak]) / slicesamplerate
        # print(t, whichpeak, zerophaselocs[whichpeak], instantaneous_cardiactime[t])
    maxtime = (
        np.ceil(
            int(
                1.02
                * tide_stats.getfracval(instantaneous_cardiactime, 0.98)
                // args.pulsereconstepsize
            )
        )
        * args.pulsereconstepsize
    )
    outtimes = np.linspace(
        0.0, maxtime, num=int(maxtime / args.pulsereconstepsize), endpoint=False
    )
    atp_bypoint = cardiaccycleaverage(
        instantaneous_cardiactime,
        outtimes,
        cardfromfmri_sliceres,
        procpoints,
        congridbins,
        gridkernel,
        False,
        cyclic=True,
    )
    if thispass == numpasses - 1:
        if args.bidsoutput:
            tide_io.writebidstsv(
                outputroot + "_desc-cardpulsefromfmri_timeseries",
                atp_bypoint,
                1.0 / (outtimes[1] - outtimes[0]),
                starttime=outtimes[0],
                columns=["pulsefromfmri"],
                append=False,
                debug=args.debug,
            )
        else:
            tide_io.writevec(atp_bypoint, outputroot + "_cardpulsefromfmri.txt")

    if not args.verbose:
        print("Phase projecting...")

    # make a lowpass filter for the projected data. Limit frequency to 3 cycles per 2pi (1/6th Fs)
    phaseFs = 1.0 / phasestep
    phaseFc = phaseFs / 6.0
    appsmoothingfilter = tide_filt.NoncausalFilter("arb", cyclic=True, padtime=0.0)
    appsmoothingfilter.setfreqs(0.0, 0.0, phaseFc, phaseFc)

    # setup for aliased correlation if we're going to do it
    if args.doaliasedcorrelation and (thispass == numpasses - 1):
        if args.cardiacfilename:
            signal_sliceres = pleth_sliceres
            # signal_stdres = pleth_stdres
        else:
            signal_sliceres = cardfromfmri_sliceres
            # signal_stdres = dlfilteredcard_stdres
        corrsearchvals = (
            np.linspace(0.0, args.aliasedcorrelationwidth, num=args.aliasedcorrelationpts)
            - args.aliasedcorrelationwidth / 2.0
        )
        theAliasedCorrelator = tide_corr.AliasedCorrelator(
            signal_sliceres,
            slicesamplerate,
            mrsamplerate,
            corrsearchvals,
            padtime=args.aliasedcorrelationwidth,
        )
        thecorrfunc = np.zeros(
            (xsize, ysize, numslices, args.aliasedcorrelationpts), dtype=np.float64
        )
        thecorrfunc_byslice = thecorrfunc.reshape(
            (xsize * ysize, numslices, args.aliasedcorrelationpts)
        )
        wavedelay = np.zeros((xsize, ysize, numslices), dtype=np.float64)
        wavedelay_byslice = wavedelay.reshape((xsize * ysize, numslices))
        waveamp = np.zeros((xsize, ysize, numslices), dtype=np.float64)
        waveamp_byslice = waveamp.reshape((xsize * ysize, numslices))

    # now project the data
    for theslice in range(numslices):
        if args.showprogressbar:
            tide_util.progressbar(theslice + 1, numslices, label="Percent complete")
        if args.verbose:
            print("Phase projecting for slice", theslice)
        validlocs = np.where(projmask_byslice[:, theslice] > 0)[0]
        # indexlist = range(0, len(cardphasevals[theslice, :]))
        if len(validlocs) > 0:
            for t in proctrs:
                filteredmr = -datatoproject_byslice[validlocs, theslice, t]
                cinemr = (
                    datatoproject_byslice[validlocs, theslice, t]
                    + means_byslice[validlocs, theslice, t]
                )
                thevals, theweights, theindices = tide_resample.congrid(
                    outphases,
                    cardphasevals[theslice, t],
                    1.0,
                    congridbins,
                    kernel=gridkernel,
                    cyclic=True,
                )
                for i in range(len(theindices)):
                    weight_byslice[validlocs, theslice, theindices[i]] += theweights[i]
                    rawapp_byslice[validlocs, theslice, theindices[i]] += (
                        theweights[i] * filteredmr
                    )
                    cine_byslice[validlocs, theslice, theindices[i]] += theweights[i] * cinemr
            for d in range(destpoints):
                if weight_byslice[validlocs[0], theslice, d] == 0.0:
                    weight_byslice[validlocs, theslice, d] = 1.0
            rawapp_byslice[validlocs, theslice, :] = np.nan_to_num(
                rawapp_byslice[validlocs, theslice, :] / weight_byslice[validlocs, theslice, :]
            )
            cine_byslice[validlocs, theslice, :] = np.nan_to_num(
                cine_byslice[validlocs, theslice, :] / weight_byslice[validlocs, theslice, :]
            )
        else:
            rawapp_byslice[:, theslice, :] = 0.0
            cine_byslice[:, theslice, :] = 0.0

        # smooth the projected data along the time dimension
        if args.smoothapp:
            for loc in validlocs:
                rawapp_byslice[loc, theslice, :] = appsmoothingfilter.apply(
                    phaseFs, rawapp_byslice[loc, theslice, :]
                )
                derivatives_byslice[loc, theslice, :] = circularderivs(
                    rawapp_byslice[loc, theslice, :]
                )
        appflips_byslice = np.where(
            -derivatives_byslice[:, :, 2] > derivatives_byslice[:, :, 0], -1.0, 1.0
        )
        timecoursemean = np.mean(rawapp_byslice[validlocs, theslice, :], axis=1).reshape((-1, 1))
        if args.fliparteries:
            corrected_rawapp_byslice[validlocs, theslice, :] = (
                rawapp_byslice[validlocs, theslice, :] - timecoursemean
            ) * appflips_byslice[validlocs, theslice, None] + timecoursemean
            if args.doaliasedcorrelation and (thispass == numpasses - 1):
                for theloc in validlocs:
                    thecorrfunc_byslice[theloc, theslice, :] = theAliasedCorrelator.apply(
                        -appflips_byslice[theloc, theslice]
                        * datatoproject_byslice[theloc, theslice, :],
                        -thetimes[theslice][0],
                    )
                    maxloc = np.argmax(thecorrfunc_byslice[theloc, theslice, :])
                    wavedelay_byslice[theloc, theslice] = corrsearchvals[maxloc]
                    waveamp_byslice[theloc, theslice] = thecorrfunc_byslice[
                        theloc, theslice, maxloc
                    ]
        else:
            corrected_rawapp_byslice[validlocs, theslice, :] = rawapp_byslice[
                validlocs, theslice, :
            ]
            if args.doaliasedcorrelation and (thispass == numpasses - 1):
                for theloc in validlocs:
                    thecorrfunc_byslice[theloc, theslice, :] = theAliasedCorrelator.apply(
                        -datatoproject_byslice[theloc, theslice, :],
                        -thetimes[theslice][0],
                    )
                    maxloc = np.argmax(np.abs(thecorrfunc_byslice[theloc, theslice, :]))
                    wavedelay_byslice[theloc, theslice] = corrsearchvals[maxloc]
                    waveamp_byslice[theloc, theslice] = thecorrfunc_byslice[
                        theloc, theslice, maxloc
                    ]
        timecoursemin = np.min(corrected_rawapp_byslice[validlocs, theslice, :], axis=1).reshape(
            (-1, 1)
        )
        app_byslice[validlocs, theslice, :] = (
            corrected_rawapp_byslice[validlocs, theslice, :] - timecoursemin
        )
        normapp_byslice[validlocs, theslice, :] = np.nan_to_num(
            app_byslice[validlocs, theslice, :] / means_byslice[validlocs, theslice, None]
        )
