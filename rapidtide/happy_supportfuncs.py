#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2018-2021 Blaise Frederick
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
import copy
import time
import warnings

import numpy as np
from scipy.signal import savgol_filter, welch
from scipy.stats import kurtosis, skew
from statsmodels.robust import mad
from tqdm import tqdm

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False

try:
    import rapidtide.dlfilter as tide_dlfilt

    dlfilterexists = True
    print("dlfilter exists")
except ImportError:
    dlfilterexists = False
    print("dlfilter does not exist")


def rrifromphase(timeaxis, thephase):
    return None


def calc_3d_optical_flow(video, projmask, flowhdr, outputroot, window_size=3, debug=False):
    # window Define the window size for Lucas-Kanade method
    # Get the number of frames, height, and width of the video
    singlehdr = copy.deepcopy(flowhdr)
    singlehdr["dim"][4] = 1
    xsize, ysize, zsize, num_frames = video.shape

    # Create an empty array to store the optical flow vectors
    flow_vectors = np.zeros((xsize, ysize, zsize, num_frames, 3))

    if debug:
        print(
            f"calc_3d_optical_flow: calculating flow in {xsize}, {ysize}, {zsize}, {num_frames} array with window_size {window_size}"
        )

    # Loop over all pairs of consecutive frames
    for i in range(num_frames):
        if debug:
            print(f"calculating flow for time point {i}")
        prev_frame = video[:, :, :, i]
        next_frame = video[:, :, :, (i + 1) % num_frames]

        # Initialize the flow vectors to zero
        flow = np.zeros((xsize, ysize, zsize, 3))

        # Loop over each pixel in the image
        for z in range(window_size // 2, zsize - window_size // 2):
            if debug:
                print(f"\tz={z}")
            for y in range(window_size // 2, ysize - window_size // 2):
                for x in range(window_size // 2, zsize - window_size // 2):
                    if projmask[x, y, z] > 0:
                        # Define the window around the pixel
                        window_prev = prev_frame[
                            x - window_size // 2 : x + window_size // 2 + 1,
                            y - window_size // 2 : y + window_size // 2 + 1,
                            z - window_size // 2 : z + window_size // 2 + 1,
                        ]
                        window_next = next_frame[
                            x - window_size // 2 : x + window_size // 2 + 1,
                            y - window_size // 2 : y + window_size // 2 + 1,
                            z - window_size // 2 : z + window_size // 2 + 1,
                        ]

                        # Compute the gradient of the window in x, y, and z directions
                        grad_x = np.gradient(window_prev)[0]
                        grad_y = np.gradient(window_prev)[1]
                        grad_z = np.gradient(window_prev)[2]

                        # Compute the temporal gradient between two frames
                        grad_t = window_next - window_prev

                        # Compute the optical flow vector using Lucas-Kanade method
                        A = np.vstack((grad_x.ravel(), grad_y.ravel(), grad_z.ravel())).T
                        b = -grad_t.ravel()
                        flow_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                        # Store the optical flow vector in the result array
                        flow[x, y, z, 0] = flow_vec[0]
                        flow[x, y, z, 1] = flow_vec[1]
                        flow[x, y, z, 2] = flow_vec[2]

        # Store the optical flow vectors in the result array
        flow_vectors[:, :, :, i, 0] = flow[..., 0]
        flow_vectors[:, :, :, i, 1] = flow[..., 1]
        flow_vectors[:, :, :, i, 2] = flow[..., 2]
        thename = f"{outputroot}_desc-flow_phase-{str(i).zfill(2)}_map"
        tide_io.savetonifti(flow_vectors[:, :, :, i, :], flowhdr, thename)
        thename = f"{outputroot}_desc-flowmag_phase-{str(i).zfill(2)}_map"
        tide_io.savetonifti(
            np.sqrt(np.sum(np.square(flow_vectors[:, :, :, i, :]), axis=3)), singlehdr, thename
        )

    return flow_vectors


def cardiacsig(thisphase, amps=(1.0, 0.0, 0.0), phases=None, overallphase=0.0):
    total = 0.0
    if phases is None:
        phases = amps * 0.0
    for i in range(len(amps)):
        total += amps[i] * np.cos((i + 1) * thisphase + phases[i] + overallphase)
    return total


def cardiacfromimage(
    normdata_byslice,
    estmask_byslice,
    numslices,
    timepoints,
    tr,
    slicetimes,
    cardprefilter,
    respprefilter,
    notchpct=1.5,
    invertphysiosign=False,
    madnorm=True,
    nprocs=1,
    arteriesonly=False,
    fliparteries=False,
    debug=False,
    appflips_byslice=None,
    verbose=False,
    usemask=True,
    multiplicative=True,
):
    # find out what timepoints we have, and their spacing
    numsteps, minstep, sliceoffsets = tide_io.sliceinfo(slicetimes, tr)
    print(
        len(slicetimes),
        "slice times with",
        numsteps,
        "unique values - diff is",
        "{:.3f}".format(minstep),
    )

    # set inversion factor
    if invertphysiosign:
        thesign = -1.0
    else:
        thesign = 1.0

    # make sure there is an appflips array
    if appflips_byslice is None:
        appflips_byslice = estmask_byslice * 0.0 + 1.0
    else:
        if arteriesonly:
            appflips_byslice[np.where(appflips_byslice > 0.0)] = 0.0

    # make slice means
    print("Making slice means...")
    hirestc = np.zeros((timepoints * numsteps), dtype=np.float64)
    cycleaverage = np.zeros((numsteps), dtype=np.float64)
    sliceavs = np.zeros((numslices, timepoints), dtype=np.float64)
    slicenorms = np.zeros((numslices), dtype=np.float64)
    if not verbose:
        print("Averaging slices...")
    if fliparteries:
        thismask_byslice = appflips_byslice.astype(np.int64) * estmask_byslice
    else:
        thismask_byslice = estmask_byslice
    for theslice in range(numslices):
        if verbose:
            print("Averaging slice", theslice)
        if usemask:
            validestvoxels = np.where(np.abs(thismask_byslice[:, theslice]) > 0)[0]
        else:
            validestvoxels = np.where(np.abs(thismask_byslice[:, theslice] >= 0))[0]
        if len(validestvoxels) > 0:
            if madnorm:
                sliceavs[theslice, :], slicenorms[theslice] = tide_math.madnormalize(
                    np.mean(
                        normdata_byslice[validestvoxels, theslice, :]
                        * thismask_byslice[validestvoxels, theslice, np.newaxis],
                        axis=0,
                    ),
                    returnnormfac=True,
                )
            else:
                sliceavs[theslice, :] = np.mean(
                    normdata_byslice[validestvoxels, theslice, :]
                    * thismask_byslice[validestvoxels, theslice, np.newaxis],
                    axis=0,
                )
                slicenorms[theslice] = 1.0
            for t in range(timepoints):
                hirestc[numsteps * t + sliceoffsets[theslice]] += thesign * sliceavs[theslice, t]
    for i in range(numsteps):
        cycleaverage[i] = np.mean(hirestc[i:-1:numsteps])
    for t in range(len(hirestc)):
        if multiplicative:
            hirestc[t] /= cycleaverage[t % numsteps] + 1.0
        else:
            hirestc[t] -= cycleaverage[t % numsteps]
    if not verbose:
        print("done")
    slicesamplerate = 1.0 * numsteps / tr
    print("Slice sample rate is ", "{:.3f}".format(slicesamplerate))

    # delete the TR frequency and the first subharmonic
    print("Notch filtering...")
    filthirestc = tide_filt.harmonicnotchfilter(
        hirestc, slicesamplerate, 1.0 / tr, notchpct=notchpct, debug=debug
    )

    # now get the cardiac and respiratory waveforms
    hirescardtc, cardnormfac = tide_math.madnormalize(
        cardprefilter.apply(slicesamplerate, filthirestc), returnnormfac=True
    )
    hirescardtc *= -1.0
    cardnormfac *= np.mean(slicenorms)

    hiresresptc, respnormfac = tide_math.madnormalize(
        respprefilter.apply(slicesamplerate, filthirestc), returnnormfac=True
    )
    hiresresptc *= -1.0
    respnormfac *= np.mean(slicenorms)

    return (
        hirescardtc,
        cardnormfac,
        hiresresptc,
        respnormfac,
        slicesamplerate,
        numsteps,
        sliceoffsets,
        cycleaverage,
        slicenorms,
    )


def savgolsmooth(data, smoothlen=101, polyorder=3):
    return savgol_filter(data, smoothlen, polyorder)


def getperiodic(inputdata, Fs, fundfreq, ncomps=1, width=0.4, debug=False):
    outputdata = inputdata * 0.0
    lowerdist = fundfreq - fundfreq / (1.0 + width)
    upperdist = fundfreq * width
    if debug:
        print(f"GETPERIODIC: starting with fundfreq={fundfreq}, ncomps={ncomps}, Fs={Fs}")
    while ncomps * fundfreq >= Fs / 2.0:
        ncomps -= 1
        print(f"\tncomps reduced to {ncomps}")
    thefundfilter = tide_filt.NoncausalFilter(filtertype="arb")
    for component in range(ncomps):
        arb_lower = (component + 1) * fundfreq - lowerdist
        arb_upper = (component + 1) * fundfreq + upperdist
        arb_lowerstop = 0.9 * arb_lower
        arb_upperstop = 1.1 * arb_upper
        if debug:
            print(
                f"GETPERIODIC: component {component} - arb parameters:{arb_lowerstop}, {arb_lower}, {arb_upper}, {arb_upperstop}"
            )
        thefundfilter.setfreqs(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)
        outputdata += 1.0 * thefundfilter.apply(Fs, inputdata)
    return outputdata


def getcardcoeffs(
    cardiacwaveform,
    slicesamplerate,
    minhr=40.0,
    maxhr=140.0,
    smoothlen=101,
    debug=False,
):
    if len(cardiacwaveform) > 1024:
        thex, they = welch(cardiacwaveform, slicesamplerate, nperseg=1024)
    else:
        thex, they = welch(cardiacwaveform, slicesamplerate)
    initpeakfreq = np.round(thex[np.argmax(they)] * 60.0, 2)
    if initpeakfreq > maxhr:
        initpeakfreq = maxhr
    if initpeakfreq < minhr:
        initpeakfreq = minhr
    if debug:
        print("initpeakfreq:", initpeakfreq, "BPM")
    freqaxis, spectrum = tide_filt.spectrum(
        tide_filt.hamming(len(cardiacwaveform)) * cardiacwaveform,
        Fs=slicesamplerate,
        mode="complex",
    )
    # remove any spikes at zero frequency
    minbin = int(minhr // (60.0 * (freqaxis[1] - freqaxis[0])))
    maxbin = int(maxhr // (60.0 * (freqaxis[1] - freqaxis[0])))
    spectrum[:minbin] = 0.0
    spectrum[maxbin:] = 0.0

    # find the max
    ampspec = savgolsmooth(np.abs(spectrum), smoothlen=smoothlen)
    peakfreq = freqaxis[np.argmax(ampspec)]
    if debug:
        print("Cardiac fundamental frequency is", np.round(peakfreq * 60.0, 2), "BPM")
    normfac = np.sqrt(2.0) * tide_math.rms(cardiacwaveform)
    if debug:
        print("normfac:", normfac)
    return peakfreq


def normalizevoxels(fmri_data, detrendorder, validvoxels, time, timings, showprogressbar=False):
    print("Normalizing voxels...")
    normdata = fmri_data * 0.0
    demeandata = fmri_data * 0.0
    starttime = time.time()
    # detrend if we are going to
    numspatiallocs = fmri_data.shape[0]
    if detrendorder > 0:
        print("Detrending to order", detrendorder, "...")
        for idx, thevox in enumerate(
            tqdm(
                validvoxels,
                desc="Voxel",
                unit="voxels",
                disable=(not showprogressbar),
            )
        ):
            fmri_data[thevox, :] = tide_fit.detrend(
                fmri_data[thevox, :], order=detrendorder, demean=False
            )
        timings.append(["Detrending finished", time.time(), numspatiallocs, "voxels"])
        print(" done")

    means = np.mean(fmri_data[:, :], axis=1).flatten()
    demeandata[validvoxels, :] = fmri_data[validvoxels, :] - means[validvoxels, None]
    normdata[validvoxels, :] = np.nan_to_num(demeandata[validvoxels, :] / means[validvoxels, None])
    medians = np.median(normdata[:, :], axis=1).flatten()
    mads = mad(normdata[:, :], axis=1).flatten()
    timings.append(["Normalization finished", time.time(), numspatiallocs, "voxels"])
    print("Normalization took", "{:.3f}".format(time.time() - starttime), "seconds")
    return normdata, demeandata, means, medians, mads


def cleanphysio(
    Fs, physiowaveform, cutoff=0.4, thresh=0.2, nyquist=None, iscardiac=True, debug=False
):
    # first bandpass the cardiac signal to calculate the envelope
    if debug:
        print("Entering cleanphysio")

    print("Filtering")
    physiofilter = tide_filt.NoncausalFilter("cardiac", debug=debug)

    print("Envelope detection")
    envelope = tide_math.envdetect(
        Fs,
        tide_math.madnormalize(physiofilter.apply(Fs, tide_math.madnormalize(physiowaveform))),
        cutoff=cutoff,
    )
    envmean = np.mean(envelope)

    # now patch the envelope function to eliminate very low values
    envlowerlim = thresh * np.max(envelope)
    envelope = np.where(envelope >= envlowerlim, envelope, envlowerlim)

    # now high pass the waveform to eliminate baseline
    arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop = physiofilter.getfreqs()
    physiofilter.settype("arb")
    arb_upper = 10.0
    arb_upperstop = arb_upper * 1.1
    if nyquist is not None:
        if nyquist < arb_upper:
            arb_upper = nyquist
            arb_upperstop = nyquist
    physiofilter.setfreqs(arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop)
    filtphysiowaveform = tide_math.madnormalize(
        physiofilter.apply(Fs, tide_math.madnormalize(physiowaveform))
    )
    print("Normalizing")
    normphysio = tide_math.madnormalize(envmean * filtphysiowaveform / envelope)

    # return the filtered waveform, the normalized waveform, and the envelope
    if debug:
        print("Leaving cleanphysio")
    return filtphysiowaveform, normphysio, envelope, envmean


def findbadpts(
    thewaveform,
    nameroot,
    outputroot,
    samplerate,
    infodict,
    thetype="mad",
    retainthresh=0.89,
    mingap=2.0,
    outputlevel=0,
    debug=True,
):
    # if thetype == 'triangle' or thetype == 'mad':
    if thetype == "mad":
        absdev = np.fabs(thewaveform - np.median(thewaveform))
        # if thetype == 'triangle':
        #    thresh = threshold_triangle(np.reshape(absdev, (len(absdev), 1)))
        medianval = np.median(thewaveform)
        sigma = mad(thewaveform, center=medianval)
        numsigma = np.sqrt(1.0 / (1.0 - retainthresh))
        thresh = numsigma * sigma
        thebadpts = np.where(absdev >= thresh, 1.0, 0.0)
        print(
            "Bad point threshhold set to",
            "{:.3f}".format(thresh),
            "using the",
            thetype,
            "method for",
            nameroot,
        )
    elif thetype == "fracval":
        lower, upper = tide_stats.getfracvals(
            thewaveform,
            [(1.0 - retainthresh) / 2.0, (1.0 + retainthresh) / 2.0],
        )
        therange = upper - lower
        lowerthresh = lower - therange
        upperthresh = upper + therange
        thebadpts = np.where((lowerthresh <= thewaveform) & (thewaveform <= upperthresh), 0.0, 1.0)
        thresh = (lowerthresh, upperthresh)
        print(
            "Values outside of ",
            "{:.3f}".format(lowerthresh),
            "to",
            "{:.3f}".format(upperthresh),
            "marked as bad using the",
            thetype,
            "method for",
            nameroot,
        )
    else:
        raise ValueError("findbadpts error: Bad thresholding type")

    # now fill in gaps
    streakthresh = int(np.round(mingap * samplerate))
    lastbad = 0
    if thebadpts[0] == 1.0:
        isbad = True
    else:
        isbad = False
    for i in range(1, len(thebadpts)):
        if thebadpts[i] == 1.0:
            if not isbad:
                # streak begins
                isbad = True
                if i - lastbad < streakthresh:
                    thebadpts[lastbad:i] = 1.0
            lastbad = i
        else:
            isbad = False
    if len(thebadpts) - lastbad - 1 < streakthresh:
        thebadpts[lastbad:] = 1.0

    if outputlevel > 0:
        tide_io.writevec(thebadpts, outputroot + "_" + nameroot + "_badpts.txt")
    infodict[nameroot + "_threshvalue"] = thresh
    infodict[nameroot + "_threshmethod"] = thetype
    return thebadpts


def approximateentropy(waveform, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[waveform[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(waveform)

    return abs(_phi(m + 1) - _phi(m))


def entropy(waveform):
    return -np.sum(np.square(waveform) * np.nan_to_num(np.log2(np.square(waveform))))


def calcplethquality(
    waveform,
    Fs,
    infodict,
    suffix,
    outputroot,
    S_windowsecs=5.0,
    K_windowsecs=60.0,
    E_windowsecs=1.0,
    detrendorder=8,
    outputlevel=0,
    initfile=True,
    debug=False,
):
    """

    Parameters
    ----------
    waveform: array-like
        The cardiac waveform to be assessed
    Fs: float
        The sample rate of the data
    S_windowsecs: float
        Skewness window duration in seconds.  Defaults to 5.0 (optimal for discrimination of "good" from "acceptable"
        and "unfit" according to Elgendi)
    K_windowsecs: float
        Skewness window duration in seconds.  Defaults to 2.0 (after Selveraj)
    E_windowsecs: float
        Entropy window duration in seconds.  Defaults to 0.5 (after Selveraj)
    detrendorder: int
        Order of detrending polynomial to apply to plethysmogram.
    debug: boolean
        Turn on extended output

    Returns
    -------
    S_sqi_mean: float
        The mean value of the quality index over all time
    S_std_mean: float
        The standard deviation of the quality index over all time
    S_waveform: array
        The quality metric over all timepoints
    K_sqi_mean: float
        The mean value of the quality index over all time
    K_std_mean: float
        The standard deviation of the quality index over all time
    K_waveform: array
        The quality metric over all timepoints
    E_sqi_mean: float
        The mean value of the quality index over all time
    E_std_mean: float
        The standard deviation of the quality index over all time
    E_waveform: array
        The quality metric over all timepoints


    Calculates the windowed skewness, kurtosis, and entropy quality metrics described in Elgendi, M.
    "Optimal Signal Quality Index for Photoplethysmogram Signals". Bioengineering 2016, Vol. 3, Page 21 3, 21 (2016).
    """
    # detrend the waveform
    dt_waveform = tide_fit.detrend(waveform, order=detrendorder, demean=True)

    # calculate S_sqi and K_sqi over a sliding window.  Window size should be an odd number of points.
    S_windowpts = int(np.round(S_windowsecs * Fs, 0))
    S_windowpts += 1 - S_windowpts % 2
    S_waveform = dt_waveform * 0.0
    K_windowpts = int(np.round(K_windowsecs * Fs, 0))
    K_windowpts += 1 - K_windowpts % 2
    K_waveform = dt_waveform * 0.0
    E_windowpts = int(np.round(E_windowsecs * Fs, 0))
    E_windowpts += 1 - E_windowpts % 2
    E_waveform = dt_waveform * 0.0

    if debug:
        print("S_windowsecs, S_windowpts:", S_windowsecs, S_windowpts)
        print("K_windowsecs, K_windowpts:", K_windowsecs, K_windowpts)
        print("E_windowsecs, E_windowpts:", E_windowsecs, E_windowpts)
    for i in range(0, len(dt_waveform)):
        startpt = np.max([0, i - S_windowpts // 2])
        endpt = np.min([i + S_windowpts // 2, len(dt_waveform)])
        S_waveform[i] = skew(dt_waveform[startpt : endpt + 1], nan_policy="omit")

        startpt = np.max([0, i - K_windowpts // 2])
        endpt = np.min([i + K_windowpts // 2, len(dt_waveform)])
        K_waveform[i] = kurtosis(dt_waveform[startpt : endpt + 1], fisher=False)

        startpt = np.max([0, i - E_windowpts // 2])
        endpt = np.min([i + E_windowpts // 2, len(dt_waveform)])
        # E_waveform[i] = entropy(dt_waveform[startpt:endpt + 1])
        r = 0.2 * np.std(dt_waveform[startpt : endpt + 1])
        E_waveform[i] = approximateentropy(dt_waveform[startpt : endpt + 1], 2, r)

    S_sqi_mean = np.mean(S_waveform)
    S_sqi_std = np.std(S_waveform)
    K_sqi_mean = np.mean(K_waveform)
    K_sqi_std = np.std(K_waveform)
    E_sqi_mean = np.mean(E_waveform)
    E_sqi_std = np.std(E_waveform)

    infodict["S_sqi_mean" + suffix] = S_sqi_mean
    infodict["S_sqi_std" + suffix] = S_sqi_std
    infodict["K_sqi_mean" + suffix] = K_sqi_mean
    infodict["K_sqi_std" + suffix] = K_sqi_std
    infodict["E_sqi_mean" + suffix] = E_sqi_mean
    infodict["E_sqi_std" + suffix] = E_sqi_std

    if outputlevel > 1:
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            S_waveform,
            Fs,
            columns=["S_sqi" + suffix],
            append=(not initfile),
            debug=debug,
        )
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            K_waveform,
            Fs,
            columns=["K_sqi" + suffix],
            append=True,
            debug=debug,
        )
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            E_waveform,
            Fs,
            columns=["E_sqi" + suffix],
            append=True,
            debug=debug,
        )


def getphysiofile(
    waveformfile,
    inputfreq,
    inputstart,
    slicetimeaxis,
    stdfreq,
    stdpoints,
    envcutoff,
    envthresh,
    timings,
    outputroot,
    slop=0.25,
    outputlevel=0,
    iscardiac=True,
    debug=False,
):
    if debug:
        print("Entering getphysiofile")
    print("Reading physiological signal from file")

    # check file type
    filefreq, filestart, dummy, waveform_fullres, dummy, dummy = tide_io.readvectorsfromtextfile(
        waveformfile, onecol=True, debug=debug
    )
    if inputfreq < 0.0:
        if filefreq is not None:
            inputfreq = filefreq
        else:
            inputfreq = -inputfreq

    if inputstart is None:
        if filestart is not None:
            inputstart = filestart
        else:
            inputstart = 0.0

    if debug:
        print("inputfreq:", inputfreq)
        print("inputstart:", inputstart)
        print("waveform_fullres:", waveform_fullres)
        print("waveform_fullres.shape:", waveform_fullres.shape)
    inputtimeaxis = (
        np.linspace(
            0.0,
            (1.0 / inputfreq) * len(waveform_fullres),
            num=len(waveform_fullres),
            endpoint=False,
        )
        + inputstart
    )
    stdtimeaxis = (
        np.linspace(0.0, (1.0 / stdfreq) * stdpoints, num=stdpoints, endpoint=False) + inputstart
    )

    if debug:
        print("getphysiofile: input time axis start, stop, step, freq, length")
        print(
            inputtimeaxis[0],
            inputtimeaxis[-1],
            inputtimeaxis[1] - inputtimeaxis[0],
            1.0 / (inputtimeaxis[1] - inputtimeaxis[0]),
            len(inputtimeaxis),
        )
        print("getphysiofile: slice time axis start, stop, step, freq, length")
        print(
            slicetimeaxis[0],
            slicetimeaxis[-1],
            slicetimeaxis[1] - slicetimeaxis[0],
            1.0 / (slicetimeaxis[1] - slicetimeaxis[0]),
            len(slicetimeaxis),
        )
    if (inputtimeaxis[0] > slop) or (inputtimeaxis[-1] < slicetimeaxis[-1] - slop):
        print("\tinputtimeaxis[0]:", inputtimeaxis[0])
        print("\tinputtimeaxis[-1]:", inputtimeaxis[-1])
        print("\tslicetimeaxis[0]:", slicetimeaxis[0])
        print("\tslicetimeaxis[-1]:", slicetimeaxis[-1])
        if inputtimeaxis[0] > slop:
            print("\tfailed condition 1:", inputtimeaxis[0], ">", slop)
        if inputtimeaxis[-1] < slicetimeaxis[-1] - slop:
            print(
                "\tfailed condition 2:",
                inputtimeaxis[-1],
                "<",
                slicetimeaxis[-1] - slop,
            )
        raise ValueError("getphysiofile: error - waveform file does not cover the fmri time range")
    if debug:
        print("waveform_fullres: len=", len(waveform_fullres), "vals=", waveform_fullres)
        print("inputfreq =", inputfreq)
        print("inputstart =", inputstart)
        print("inputtimeaxis: len=", len(inputtimeaxis), "vals=", inputtimeaxis)
    timings.append(["Cardiac signal from physiology data read in", time.time(), None, None])

    # filter and amplitude correct the waveform to remove gain fluctuations
    cleanwaveform_fullres, normwaveform_fullres, waveformenv_fullres, envmean = cleanphysio(
        inputfreq,
        waveform_fullres,
        iscardiac=iscardiac,
        cutoff=envcutoff,
        thresh=envthresh,
        nyquist=inputfreq / 2.0,
        debug=debug,
    )

    if iscardiac:
        if outputlevel > 1:
            tide_io.writevec(waveform_fullres, outputroot + "_rawpleth_native.txt")
            tide_io.writevec(cleanwaveform_fullres, outputroot + "_pleth_native.txt")
            tide_io.writevec(waveformenv_fullres, outputroot + "_cardenvelopefromfile_native.txt")
        timings.append(["Cardiac signal from physiology data cleaned", time.time(), None, None])

    # resample to slice time resolution and save
    waveform_sliceres = tide_resample.doresample(
        inputtimeaxis, cleanwaveform_fullres, slicetimeaxis, method="univariate", padlen=0
    )

    # resample to standard resolution and save
    waveform_stdres = tide_math.madnormalize(
        tide_resample.doresample(
            inputtimeaxis,
            cleanwaveform_fullres,
            stdtimeaxis,
            method="univariate",
            padlen=0,
        )
    )

    timings.append(
        [
            "Cardiac signal from physiology data resampled to slice resolution and saved",
            time.time(),
            None,
            None,
        ]
    )

    if debug:
        print("Leaving getphysiofile")
    return waveform_sliceres, waveform_stdres, inputfreq, len(waveform_fullres)


def readextmask(thefilename, nim_hdr, xsize, ysize, numslices, debug=False):
    (
        extmask,
        extmask_data,
        extmask_hdr,
        theextmaskdims,
        theextmasksizes,
    ) = tide_io.readfromnifti(thefilename)
    (
        xsize_extmask,
        ysize_extmask,
        numslices_extmask,
        timepoints_extmask,
    ) = tide_io.parseniftidims(theextmaskdims)
    if debug:
        print(
            f"Mask dimensions: {xsize_extmask}, {ysize_extmask}, {numslices_extmask}, {timepoints_extmask}"
        )
    if not tide_io.checkspacematch(nim_hdr, extmask_hdr):
        raise ValueError("Dimensions of mask do not match the fmri data - exiting")
    if timepoints_extmask > 1:
        raise ValueError("Mask must have only 3 dimensions - exiting")
    return extmask_data


def checkcardmatch(reference, candidate, samplerate, refine=True, zeropadding=0, debug=False):
    """

    Parameters
    ----------
    reference: 1D numpy array
        The cardiac waveform to compare to
    candidate: 1D numpy array
        The cardiac waveform to be assessed
    samplerate: float
        The sample rate of the data in Hz
    refine: bool, optional
        Whether to refine the peak fit.  Default is True.
    zeropadding: int, optional
        Specify the length of correlation padding to use.
    debug: bool, optional
        Output additional information for debugging

    Returns
    -------
    maxval: float
        The maximum value of the crosscorrelation function
    maxdelay: float
        The time, in seconds, where the maximum crosscorrelation occurs.
    failreason: flag
        Reason why the fit failed (0 if no failure)
    """
    thecardfilt = tide_filt.NoncausalFilter(filtertype="cardiac")
    trimlength = np.min([len(reference), len(candidate)])
    thexcorr = tide_corr.fastcorrelate(
        tide_math.corrnormalize(
            thecardfilt.apply(samplerate, reference),
            detrendorder=3,
            windowfunc="hamming",
        )[:trimlength],
        tide_math.corrnormalize(
            thecardfilt.apply(samplerate, candidate),
            detrendorder=3,
            windowfunc="hamming",
        )[:trimlength],
        usefft=True,
        zeropadding=zeropadding,
    )
    xcorrlen = len(thexcorr)
    sampletime = 1.0 / samplerate
    xcorr_x = np.r_[0.0:xcorrlen] * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    searchrange = 5.0
    trimstart = tide_util.valtoindex(xcorr_x, -2.0 * searchrange)
    trimend = tide_util.valtoindex(xcorr_x, 2.0 * searchrange)
    (
        maxindex,
        maxdelay,
        maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    ) = tide_fit.findmaxlag_gauss(
        xcorr_x[trimstart:trimend],
        thexcorr[trimstart:trimend],
        -searchrange,
        searchrange,
        3.0,
        refine=refine,
        zerooutbadfit=False,
        useguess=False,
        fastgauss=False,
        displayplots=False,
    )
    if debug:
        print(
            "CORRELATION: maxindex, maxdelay, maxval, maxsigma, maskval, failreason, peakstart, peakend:",
            maxindex,
            maxdelay,
            maxval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        )
    return maxval, maxdelay, failreason


def cardiaccycleaverage(
    sourcephases,
    destinationphases,
    waveform,
    procpoints,
    congridbins,
    gridkernel,
    centric,
    cyclic=True,
):
    rawapp_bypoint = np.zeros(len(destinationphases), dtype=np.float64)
    weight_bypoint = np.zeros(len(destinationphases), dtype=np.float64)
    for t in procpoints:
        thevals, theweights, theindices = tide_resample.congrid(
            destinationphases,
            tide_math.phasemod(sourcephases[t], centric=centric),
            1.0,
            congridbins,
            kernel=gridkernel,
            cyclic=cyclic,
        )
        for i in range(len(theindices)):
            weight_bypoint[theindices[i]] += theweights[i]
            rawapp_bypoint[theindices[i]] += theweights[i] * waveform[t]
    rawapp_bypoint = np.where(
        weight_bypoint > np.max(weight_bypoint) / 50.0,
        np.nan_to_num(rawapp_bypoint / weight_bypoint),
        0.0,
    )
    minval = np.min(rawapp_bypoint[np.where(weight_bypoint > np.max(weight_bypoint) / 50.0)])
    rawapp_bypoint = np.where(
        weight_bypoint > np.max(weight_bypoint) / 50.0, rawapp_bypoint - minval, 0.0
    )
    return rawapp_bypoint, weight_bypoint


def circularderivs(timecourse):
    firstderiv = np.diff(timecourse, append=[timecourse[0]])
    return (
        np.max(firstderiv),
        np.argmax(firstderiv),
        np.min(firstderiv),
        np.argmin(firstderiv),
    )


def phaseproject(
    datatoproject,
    means,
    destpoints,
    numsteps,
    timings,
    cardfromfmri_sliceres,
    instantaneous_cardiacphase,
    thispass,
    numpasses,
    args,
    outputroot,
    slicesamplerate,
    pleth_sliceres,
    mrsamplerate,
    projmask_byslice,
    cardphasevals,
    thetimes,
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
        tide_io.writebidstsv(
            outputroot + "_desc-cardiaccyclefromfmri_timeseries",
            app_bypoint,
            1.0 / (outphases[1] - outphases[0]),
            starttime=outphases[0],
            columns=["cardiaccyclefromfmri"],
            append=False,
            debug=args.debug,
        )

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
        tide_io.writebidstsv(
            outputroot + "_desc-cardpulsefromfmri_timeseries",
            atp_bypoint,
            1.0 / (outtimes[1] - outtimes[0]),
            starttime=outtimes[0],
            columns=["pulsefromfmri"],
            append=False,
            debug=args.debug,
        )

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
            numsteps,
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
    for theslice in tqdm(
        range(numslices),
        desc="Slice",
        unit="slices",
    ):
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
