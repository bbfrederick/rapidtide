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
import getopt
import os
import platform
import sys
import time
import warnings

import numpy as np
import scipy as sp
from matplotlib.pyplot import figure, plot, show
from scipy.signal import savgol_filter, welch
from scipy.stats import kurtosis, skew
from statsmodels.robust import mad

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.glmpass as tide_glmpass
import rapidtide.helper_classes as tide_classes
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.multiproc as tide_multiproc
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


def usage():
    print(
        os.path.basename(sys.argv[0]), "- Hypersampling by Analytic Phase Projection - Yay!",
    )
    print("")
    print("usage: ", os.path.basename(sys.argv[0]), " fmrifile slicetimefile outputroot")
    print("")
    print("required arguments:")
    print("    fmrifile:                      - NIFTI file containing BOLD fMRI data")
    print(
        "    slicetimefile:                 - Text file containing the offset time in seconds of each slice relative"
    )
    print(
        "                                     to the start of the TR, one value per line, OR the BIDS sidecar JSON file"
    )
    print("                                     for the fmrifile (contains the SliceTiming field")
    print("    outputroot:                    - Base name for all output files")
    print("")
    print("optional arguments:")
    print("")
    print("Processing steps:")
    print(
        "    --cardcalconly                 - Stop after all cardiac regressor calculation steps (before phase projection)."
    )
    print(
        "    --dodlfilter                   - Refine cardiac waveform from the fMRI data using a deep learning filter."
    )
    print(
        "                                     NOTE: this will only work if you have a working Keras installation;"
    )
    print("                                     if not, this option is ignored.")
    print(
        "                                     OTHER NOTE: Some versions of tensorflow seem to have some weird conflict"
    )
    print(
        "                                     with MKL which I can't seem to be able to fix.  If the dl filter bombs"
    )
    print(
        "                                     complaining about multiple openmp libraries, try rerunning with the"
    )
    print(
        "                                     secret and inadvisable '--usesuperdangerousworkaround' flag.  Good luck!"
    )
    print(
        "    --model=MODELNAME              - Use model MODELNAME for dl filter (default is model_revised - from the revised NeuroImage paper.)"
    )
    print(
        "    --spatialglm                   - Generate voxelwise aliased synthetic cardiac regressors and filter"
    )
    print("                                     them out")
    print(
        "    --temporalglm                  - Generate synthetic cardiac signal maps for each timepoint and filter"
    )
    print("                                     them out")
    print("")
    print("Performance:")
    print(
        "    --mklthreads=NTHREADS          - Use NTHREADS MKL threads to accelerate processing (defaults to 1 - more"
    )
    print(
        "                                     threads up to the number of cores can accelerate processing a lot, but"
    )
    print(
        "                                     can really kill you on clusters unless you're very careful.  Use at your"
    )
    print("                                     own risk.)")
    print("")
    print("Preprocessing:")
    print(
        "    --numskip=SKIP                 - Skip SKIP tr's at the beginning of the fMRI file (default is 0)."
    )
    print(
        "    --motskip=SKIP                 - Skip SKIP tr's at the beginning of the motion regressor file (default is 0)."
    )
    print(
        "    --motionfile=MOTFILE[:COLSPEC] - Read 6 columns of motion regressors out of MOTFILE text file."
    )
    print(
        "                                     (with timepoints rows) and regress them, their derivatives, "
    )
    print(
        "                                     and delayed derivatives out of the data prior to analysis."
    )
    print(
        "                                     If COLSPEC is present, use the comma separated list of ranges to"
    )
    print(
        "                                     specify X, Y, Z, RotX, RotY, and RotZ, in that order.  For"
    )
    print(
        "                                     example, :3-5,7,0,9 would use columns 3, 4, 5, 7, 0 and 9"
    )
    print("                                     for X, Y, Z, RotX, RotY, RotZ, respectively")
    print(
        "    --motionhp=HPFREQ              - Highpass filter motion regressors to HPFREQ Hz prior to regression"
    )
    print(
        "    --motionlp=LPFREQ              - Lowpass filter motion regressors to HPFREQ Hz prior to regression"
    )
    print("")
    print("Cardiac estimation tuning:")
    print(
        "                                     the generation of the cardiac waveform (default is no variance masking.)"
    )
    print(
        "    --estmask=MASKNAME             - Generation of cardiac waveform from data will be restricted to"
    )
    print(
        "                                     voxels in MASKNAME and weighted by the mask intensity.  If this is "
    )
    print(
        "                                     selected, happy will only make a single pass through the data (the"
    )
    print(
        "                                     initial vessel mask generation pass will be skipped)."
    )
    print(
        "    --minhr=MINHR                  - Limit lower cardiac frequency search range to MINHR BPM (default is 40)"
    )
    print(
        "    --maxhr=MAXHR                  - Limit upper cardiac frequency search range to MAXHR BPM (default is 140)"
    )
    print(
        "    --minhrfilt=MINHR              - Highpass filter cardiac waveform estimate to MINHR BPM (default is 40)"
    )
    print(
        "    --maxhrfilt=MAXHR              - Lowpass filter cardiac waveform estimate to MAXHR BPM (default is 1000)"
    )
    print(
        "    --envcutoff=CUTOFF             - Lowpass filter cardiac normalization envelope to CUTOFF Hz (default is 0.4)"
    )
    print(
        "    --notchwidth=WIDTH             - Set the width of the notch filter, in percent of the notch frequency"
    )
    print("                                     (default is 1.5)")

    print("")
    print("External cardiac waveform options:")
    print(
        "    --cardiacfile=FILE[:COL]       - Read the cardiac waveform from file FILE.  If COL is an integer,"
    )
    print(
        "                                     and FILE is a text file, use the COL'th column.  If FILE is a BIDS "
    )
    print(
        "                                     format json file, use column named COL. If no file is specified, "
    )
    print("                                     estimate the cardiac signal from the fMRI data.")
    print(
        "    --cardiacfreq=FREQ             - Cardiac waveform in cardiacfile has sample frequency FREQ "
    )
    print(
        "                                     (default is 32Hz). NB: --cardiacfreq and --cardiactstep"
    )
    print("                                     are two ways to specify the same thing")
    print(
        "    --cardiactstep=TSTEP           - Cardiac waveform in file has sample time step TSTEP "
    )
    print(
        "                                     (default is 0.03125s) NB: --cardiacfreq and --cardiactstep"
    )
    print("                                     are two ways to specify the same thing")
    print(
        "    --cardiacstart=START           - The time delay in seconds into the cardiac file, corresponding"
    )
    print("                                     in the first TR of the fMRI file (default is 0.0)")
    print(
        "    --stdfreq=FREQ                 - Frequency to which the cardiac signals are resampled for output."
    )
    print("                                     Default is 25.")
    print(
        "    --forcehr=BPM                  - Force heart rate fundamental detector to be centered at BPM"
    )
    print(
        "                                     (overrides peak frequencies found from spectrum).  Useful"
    )
    print(
        "                                     if there is structured noise that confuses the peak finder."
    )
    print("")
    print("Phase projection tuning:")
    print("    --outputbins=BINS              - Number of output phase bins (default is 32)")
    print(
        "    --gridbins=BINS                - Width of the gridding kernel in output phase bins (default is 3.0)"
    )
    print(
        "    --gridkernel=KERNEL            - Convolution gridding kernel.  Options are 'old', 'gauss', and 'kaiser'"
    )
    print("                                     (default is 'kaiser')")
    print(
        "    --projmask=MASKNAME            - Phase projection will be restricted to voxels in MASKNAME"
    )
    print("                                     (overrides normal intensity mask.)")
    print(
        "    --projectwithraw               - Use fMRI derived cardiac waveform as phase source for projection, even"
    )
    print("                                     if a plethysmogram is supplied")
    print(
        "    --fliparteries                 - Attempt to detect arterial signals and flip over the timecourses after"
    )
    print(
        "                                     phase projection (since relative arterial blood susceptibility is"
    )
    print("                                     inverted relative to venous blood).")
    # print("    --arteriesonly                 - Restrict cardiac waveform estimation to putative arteries only.")
    print("")
    print("Debugging arguments (probably not of interest to users):")
    print(
        "    --aliasedcorrelation           - Attempt to calculate absolute delay using an aliased correlation (experimental)."
    )
    print(
        "    --noprogressbar                - Disable progress bars - useful if saving output to files"
    )
    print("    --debug                        - Turn on debugging information")
    print(
        "    --increaseoutputlevel          - Increase the output level to output more intermediate files (default=1)"
    )
    print(
        "    --decreaseoutputlevel          - Decrease the output level to output fewer intermediate files (default=1)"
    )
    print("    --nodetrend                    - Disable data detrending")
    print(
        "    --noorthog                     - Disable orthogonalization of motion confound regressors"
    )
    print("    --disablenotch                 - Disable subharmonic notch filter")
    print(
        "    --nomask                       - Disable data masking for calculating cardiac waveform"
    )
    print(
        "    --nocensor                     - Bad points will not be excluded from analytic phase projection"
    )
    print("    --noappsmooth                  - Disable smoothing app file in the phase direction")
    print(
        "    --nophasefilt                  - Disable the phase trend filter (probably not a good idea)"
    )
    print(
        "    --nocardiacalign               - Disable alignment of pleth signal to fMRI derived cardiac signal."
    )
    print("                                     to blood vessels")
    print(
        "    --saveinfoasjson               - Save the info file in json format rather than text.  Will eventually"
    )
    print(
        "    --trimcorrelations             - Some physiological timecourses don't cover the entire length of the"
    )
    print(
        "                                     fMRI experiment.  Use this option to trim other waveforms to match "
    )
    print("                                     when calculating correlations.")
    print(
        "    --saveintermediate             - Save some data from intermediate passes to help debugging."
    )

    return ()


def rrifromphase(timeaxis, thephase):
    return None


def cardiacsig(thisphase, amps=[1.0, 0.0, 0.0], phases=None, overallphase=0.0):
    total = 0.0
    if phases is None:
        phases = amps * 0.0
    for i in range(len(amps)):
        total += amps[i] * np.cos((i + 1) * thisphase + phases[i] + overallphase)
    return total


def physiofromimage(
    normdata_byslice,
    mask_byslice,
    numslices,
    timepoints,
    tr,
    slicetimes,
    cardprefilter,
    respprefilter,
    notchpct=1.5,
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
        len(slicetimes), "slice times with", numsteps, "unique values - diff is", minstep,
    )

    # make sure there is an appflips array
    if appflips_byslice is None:
        appflips_byslice = mask_byslice * 0.0 + 1.0
    else:
        if arteriesonly:
            appflips_byslice[np.where(appflips_byslice > 0.0)] = 0.0

    # make slice means
    print("making slice means...")
    hirestc = np.zeros((timepoints * numsteps), dtype=np.float64)
    cycleaverage = np.zeros((numsteps), dtype=np.float64)
    sliceavs = np.zeros((numslices, timepoints), dtype=np.float64)
    slicenorms = np.zeros((numslices), dtype=np.float64)
    if not verbose:
        print("averaging slices...")
    if fliparteries:
        thismask_byslice = appflips_byslice.astype(np.int64) * mask_byslice
    else:
        thismask_byslice = mask_byslice
    for theslice in range(numslices):
        if verbose:
            print("averaging slice", theslice)
        if usemask:
            validvoxels = np.where(np.abs(thismask_byslice[:, theslice]) > 0)[0]
        else:
            validvoxels = np.where(np.abs(thismask_byslice[:, theslice] >= 0))[0]
        if len(validvoxels) > 0:
            if madnorm:
                sliceavs[theslice, :], slicenorms[theslice] = tide_math.madnormalize(
                    np.mean(
                        normdata_byslice[validvoxels, theslice, :]
                        * thismask_byslice[validvoxels, theslice, np.newaxis],
                        axis=0,
                    ),
                    returnnormfac=True,
                )
            else:
                sliceavs[theslice, :] = np.mean(
                    normdata_byslice[validvoxels, theslice, :]
                    * thismask_byslice[validvoxels, theslice, np.newaxis],
                    axis=0,
                )
                slicenorms[theslice] = 1.0
            for t in range(timepoints):
                hirestc[numsteps * t + sliceoffsets[theslice]] += sliceavs[theslice, t]
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
    print("slice sample rate is ", slicesamplerate)

    # delete the TR frequency and the first subharmonic
    print("notch filtering...")
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
        cycleaverage,
        slicenorms,
    )


def savgolsmooth(data, smoothlen=101, polyorder=3):
    return savgol_filter(data, smoothlen, polyorder)


def getfundamental(inputdata, Fs, fundfreq):
    arb_lower = 0.71 * fundfreq
    arb_upper = 1.4 * fundfreq
    arb_lowerstop = 0.9 * arb_lower
    arb_upperstop = 1.1 * arb_upper
    thefundfilter = tide_filt.NoncausalFilter(filtertype="arb")
    thefundfilter.setfreqs(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)
    return thefundfilter.apply(Fs, inputdata)


def getcardcoeffs(
    cardiacwaveform,
    slicesamplerate,
    minhr=40.0,
    maxhr=140.0,
    smoothlen=101,
    debug=False,
    display=False,
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
    if display:
        figure()
        plot(freqaxis, ampspec, "r")
        show()
    peakfreq = freqaxis[np.argmax(ampspec)]
    if debug:
        print("cardiac fundamental frequency is", np.round(peakfreq * 60.0, 2), "BPM")
    normfac = np.sqrt(2.0) * tide_math.rms(cardiacwaveform)
    if debug:
        print("normfac:", normfac)
    return peakfreq


def normalizevoxels(fmri_data, detrendorder, validvoxels, time, timings, showprogressbar=False):
    print("normalizing voxels...")
    normdata = fmri_data * 0.0
    demeandata = fmri_data * 0.0
    starttime = time.time()
    # detrend if we are going to
    numspatiallocs = fmri_data.shape[0]
    reportstep = int(numspatiallocs // 100)
    if detrendorder > 0:
        print("detrending to order", detrendorder, "...")
        for idx, thevox in enumerate(validvoxels):
            if ((idx % reportstep == 0) or (idx == len(validvoxels) - 1)) and showprogressbar:
                tide_util.progressbar(idx + 1, len(validvoxels), label="Percent complete")
            fmri_data[thevox, :] = tide_fit.detrend(
                fmri_data[thevox, :], order=detrendorder, demean=False
            )
        timings.append(["Detrending finished", time.time(), numspatiallocs, "voxels"])
        print(" done")

    means = np.mean(fmri_data[:, :], axis=1).flatten()
    demeandata[validvoxels, :] = fmri_data[validvoxels, :] - means[validvoxels, None]
    normdata[validvoxels, :] = np.nan_to_num(demeandata[validvoxels, :] / means[validvoxels, None])
    timings.append(["Normalization finished", time.time(), numspatiallocs, "voxels"])
    print("normalization took", time.time() - starttime, "seconds")
    return normdata, demeandata, means


def cleancardiac(Fs, plethwaveform, cutoff=0.4, thresh=0.2, nyquist=None, debug=False):
    # first bandpass the cardiac signal to calculate the envelope
    if debug:
        print("entering cleancardiac")
    plethfilter = tide_filt.NoncausalFilter("cardiac")
    print("filtering")
    print("envelope detection")
    envelope = tide_math.envdetect(
        Fs,
        tide_math.madnormalize(plethfilter.apply(Fs, tide_math.madnormalize(plethwaveform))),
        cutoff=cutoff,
    )
    envmean = np.mean(envelope)

    # now patch the envelope function to eliminate very low values
    envlowerlim = thresh * np.max(envelope)
    envelope = np.where(envelope >= envlowerlim, envelope, envlowerlim)

    # now high pass the plethysmogram to eliminate baseline
    arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop = plethfilter.getfreqs()
    plethfilter.settype("arb")
    arb_upper = 10.0
    arb_upperstop = arb_upper * 1.1
    if nyquist is not None:
        if nyquist < arb_upper:
            arb_upper = nyquist
            arb_upperstop = nyquist
    plethfilter.setfreqs(arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop)
    filtplethwaveform = tide_math.madnormalize(
        plethfilter.apply(Fs, tide_math.madnormalize(plethwaveform))
    )
    print("normalizing")
    normpleth = tide_math.madnormalize(envmean * filtplethwaveform / envelope)

    # return the filtered waveform, the normalized waveform, and the envelope
    if debug:
        print("leaving cleancardiac")
    return filtplethwaveform, normpleth, envelope, envmean


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
            "bad point threshhold set to", thresh, "using the", thetype, "method for", nameroot,
        )
    elif thetype == "fracval":
        lower, upper = tide_stats.getfracvals(
            thewaveform, [(1.0 - retainthresh) / 2.0, (1.0 + retainthresh) / 2.0],
        )
        therange = upper - lower
        lowerthresh = lower - therange
        upperthresh = upper + therange
        thebadpts = np.where((lowerthresh <= thewaveform) & (thewaveform <= upperthresh), 0.0, 1.0)
        thresh = (lowerthresh, upperthresh)
        print(
            "values outside of ",
            lowerthresh,
            "to",
            upperthresh,
            "marked as bad using the",
            thetype,
            "method for",
            nameroot,
        )
    else:
        print("bad thresholding type")
        sys.exit()

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
        if debug:
            print(
                i,
                startpt,
                endpt,
                endpt - startpt + 1,
                S_waveform[i],
                K_waveform[i],
                E_waveform[i],
            )

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
        tide_io.writevec(S_waveform, outputroot + suffix + "_S_sqi_" + str(Fs) + "Hz.txt")
        tide_io.writevec(K_waveform, outputroot + suffix + "_K_sqi_" + str(Fs) + "Hz.txt")
        tide_io.writevec(E_waveform, outputroot + suffix + "_E_sqi_" + str(Fs) + "Hz.txt")


def getphysiofile(
    cardiacfile,
    colnum,
    colname,
    inputfreq,
    inputstart,
    slicetimeaxis,
    stdfreq,
    envcutoff,
    envthresh,
    timings,
    infodict,
    outputroot,
    outputlevel=0,
    debug=False,
):
    if debug:
        print("entering getphysiofile")
    print("reading cardiac signal from file")
    infodict["cardiacfromfmri"] = False

    # check file type
    filebase, extension = os.path.splitext(cardiacfile)
    if debug:
        print("filebase:", filebase)
        print("extension:", extension)
    if extension == ".json":
        inputfreq, inputstart, pleth_fullres = tide_io.readcolfrombidstsv(
            cardiacfile, columnname=colname, columnnum=colnum, debug=debug
        )
    else:
        pleth_fullres = np.transpose(tide_io.readvecs(cardiacfile))
        print(pleth_fullres.shape)
        if len(pleth_fullres.shape) != 1:
            pleth_fullres = pleth_fullres[:, colnum].flatten()
    if debug:
        print("inputfreq:", inputfreq)
        print("inputstart:", inputstart)
        print("pleth_fullres:", pleth_fullres)
    inputtimeaxis = (
        sp.arange(0.0, (1.0 / inputfreq) * len(pleth_fullres), 1.0 / inputfreq) + inputstart
    )
    if inputtimeaxis[0] > 0.0 or inputtimeaxis[-1] < slicetimeaxis[-1]:
        print("getphysiofile: error - plethysmogram waveform does not cover the fmri time range")
        sys.exit()
    if debug:
        print("pleth_fullres: len=", len(pleth_fullres), "vals=", pleth_fullres)
        print("inputfreq =", inputfreq)
        print("inputstart =", inputstart)
        print("inputtimeaxis: len=", len(inputtimeaxis), "vals=", inputtimeaxis)
    timings.append(["Cardiac signal from physiology data read in", time.time(), None, None])

    # filter and amplitude correct the waveform to remove gain fluctuations
    cleanpleth_fullres, normpleth_fullres, plethenv_fullres, envmean = cleancardiac(
        inputfreq,
        pleth_fullres,
        cutoff=envcutoff,
        thresh=envthresh,
        nyquist=inputfreq / 2.0,
        debug=debug,
    )
    infodict["plethsamplerate"] = inputfreq
    infodict["numplethpts_fullres"] = len(pleth_fullres)

    if outputlevel > 1:
        tide_io.writevec(pleth_fullres, outputroot + "_rawpleth_native.txt")
        tide_io.writevec(cleanpleth_fullres, outputroot + "_pleth_native.txt")
        tide_io.writevec(plethenv_fullres, outputroot + "_cardenvelopefromfile_native.txt")
    timings.append(["Cardiac signal from physiology data cleaned", time.time(), None, None])

    # resample to slice time resolution and save
    pleth_sliceres = tide_resample.doresample(
        inputtimeaxis, cleanpleth_fullres, slicetimeaxis, method="univariate", padlen=0
    )
    infodict["numplethpts_sliceres"] = len(pleth_sliceres)

    # resample to standard resolution and save
    pleth_stdres = tide_math.madnormalize(
        tide_resample.arbresample(
            cleanpleth_fullres, inputfreq, stdfreq, decimate=True, debug=False
        )
    )
    infodict["numplethpts_stdres"] = len(pleth_stdres)

    timings.append(
        [
            "Cardiac signal from physiology data resampled to slice resolution and saved",
            time.time(),
            None,
            None,
        ]
    )

    if debug:
        print("leaving getphysiofile")
    return pleth_sliceres, pleth_stdres


def readextmask(thefilename, nim_hdr, xsize, ysize, numslices):
    (extmask, extmask_data, extmask_hdr, theextmaskdims, theextmasksizes,) = tide_io.readfromnifti(
        thefilename
    )
    (
        xsize_extmask,
        ysize_extmask,
        numslices_extmask,
        timepoints_extmask,
    ) = tide_io.parseniftidims(theextmaskdims)
    if not tide_io.checkspacematch(nim_hdr, extmask_hdr):
        print("Dimensions of mask do not match the fmri data - exiting")
        sys.exit()
    if timepoints_extmask > 1:
        print("Mask must have only 3 dimensions - exiting")
        sys.exit()
    return extmask_data.reshape(xsize * ysize, numslices)


def checkcardmatch(reference, candidate, samplerate, refine=True, debug=False):
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
            thecardfilt.apply(samplerate, reference), detrendorder=3, windowfunc="hamming",
        )[:trimlength],
        tide_math.corrnormalize(
            thecardfilt.apply(samplerate, candidate), detrendorder=3, windowfunc="hamming",
        )[:trimlength],
        usefft=True,
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
    return rawapp_bypoint


def circularderivs(timecourse):
    firstderiv = np.diff(timecourse, append=[timecourse[0]])
    return (
        np.max(firstderiv),
        np.argmax(firstderiv),
        np.min(firstderiv),
        np.argmin(firstderiv),
    )


def findphasecuts(phases):
    max_peaks = []
    min_peaks = []
    thisval = phases[0]
    for i in range(1, len(phases)):
        if thisval - phases[i] > np.pi:
            max_peaks.append([i - 1, thisval])
            min_peaks.append([i, phases[i]])
        thisval = phases[i]
    return max_peaks, min_peaks


def happy_main(thearguments):
    # get the command line parameters
    debug = False
    centric = True
    savetcsastsv = False
    histlen = 100
    smoothlen = 101
    envcutoff = 0.4
    envthresh = 0.2
    maskthreshpct = 10.0
    upsamplefac = 100
    destpoints = 32
    congridbins = 3.0
    gridkernel = "kaiser"
    cardiacfilename = None
    colnum = None
    colname = None
    inputfreq = 32.0
    inputstart = 0.0
    dospatialglm = False
    dotemporalglm = False
    notchpct = 1.5
    minhr = 40.0
    maxhr = 140.0
    minhrfilt = 40.0
    maxhrfilt = 1000.0
    softvesselfrac = 0.4
    infodict = {}
    stdfreq = 25.0
    nprocs = 1
    mklthreads = 1
    savecardiacnoise = True
    forcedhr = None
    usemaskcardfromfmri = True
    censorbadpts = True
    estmaskname = None
    projmaskname = None
    detrendorder = 3
    filtphase = True
    savemotionglmfilt = False
    motionfilename = None
    cardcalconly = False
    domadnorm = True
    numskip = 0
    motskip = 0
    dodlfilter = False
    modelname = "model_revised"
    motionhp = None
    motionlp = None
    motfilt_pos = False
    motfilt_deriv = True
    motfilt_derivdelayed = True
    orthogonalize = True
    mpfix = False
    aligncardiac = True
    projectwithraw = False
    saveinfoasjson = False
    outputlevel = 1
    verbose = False
    smoothapp = True
    unnormvesselmap = True
    fliparteries = False
    pulsereconstepsize = 0.01
    arteriesonly = False
    saveintermediate = False
    showprogressbar = True
    doaliasedcorrelation = False
    aliasedcorrelationwidth = 1.25
    aliasedcorrelationpts = 101
    # start the clock!
    timings = [["Start", time.time(), None, None]]

    """print(
        "***********************************************************************************************************************************")
    print("NOTICE:  This program is NOT released yet - it's a work in progress and is nowhere near done.  That's why")
    print("there's no documentation or mention in the release notes.  If you want to play with it, be my guest, but be")
    print("aware of the following:")
    print("    1) Any given version of this program may or may not work, or may work in a way different than ")
    print("       a) previous versions, b) what I say it does, c) what I think it does, and d) what you want it to do.")
    print(
        "    2) I am intending to write a paper on this, and if you take this code and scoop me, I'll be peeved. That's just rude.")
    print("    3) For all I know this program might burn down your house, leave your milk out of the refrigerator, or ")
    print("       poison your dog.  USE AT YOUR OWN RISK.")
    print(
        "***********************************************************************************************************************************")
    print("")"""

    fmrifilename = thearguments[1]
    slicetimename = thearguments[2]
    outputroot = thearguments[3]

    infodict["fmrifilename"] = fmrifilename
    infodict["slicetimename"] = slicetimename
    infodict["outputroot"] = outputroot

    # save program version
    (
        infodict["release_version"],
        infodict["git_longtag"],
        infodict["git_date"],
        infodict["git_isdirty"],
    ) = tide_util.version()

    # record the machine we ran on
    infodict["hostname"] = platform.node()

    print("running version", infodict["release_version"], "on host", infodict["hostname"])

    optparsestart = 4

    # now scan for optional arguments
    try:
        opts, args = getopt.getopt(
            thearguments[optparsestart:],
            "x",
            [
                "cardiacfile=",
                "cardiacfreq=",
                "cardiactstep=",
                "cardiacstart=",
                "maxhr=",
                "minhr=",
                "maxhrfilt=",
                "minhrfilt=",
                "envcutoff=",
                "envthresh=",
                "notchwidth=",
                "disablenotch",
                "nodetrend",
                "motionfile=",
                "spatialglm",
                "temporalglm",
                "debug",
                "motionhp=",
                "motionlp=",
                "cardcalconly",
                "outputbins=",
                "gridbins=",
                "gridkernel=",
                "stdfreq=",
                "nprocs=",
                "mklthreads=",
                "arteriesonly",
                "estmask=",
                "projmask=",
                "smoothlen=",
                "forcehr=",
                "numskip=",
                "motskip=",
                "nocensor",
                "noappsmooth",
                "nomadnorm",
                "dodlfilter",
                "noncentric",
                "model=",
                "noprogressbar",
                "usesuperdangerousworkaround",
                "saveintermediate",
                "savemotionglmfilt",
                "saveinfoasjson",
                "savetcsastsv",
                "nophasefilt",
                "projectwithraw",
                "trimcorrelations",
                "fliparteries",
                "nomask",
                "noorthog",
                "nocardiacalign",
                "nomotderiv",
                "nomotderivdelayed",
                "increaseoutputlevel",
                "decreaseoutputlevel",
                "aliasedcorrelation",
                "help",
            ],
        )
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -x not recognized"
        usage()
        sys.exit(2)

    formattedcmdline = [thearguments[0] + " \\"]
    for thearg in range(1, optparsestart):
        formattedcmdline.append("\t" + thearguments[thearg] + " \\")

    for o, a in opts:
        linkchar = " "
        if o == "-x":
            print("Got an x")
        elif o == "--motionfile":
            motionfilename = a
            print("Will regress motion out of data prior to analysis")
        elif o == "--saveintermediate":
            saveintermediate = True
            print("Will save some data from intermediate passes")
        elif o == "--aliasedcorrelation":
            doaliasedcorrelation = True
            print("Will get absolute cardiac delay using aliased correlation function")
        elif o == "--arteriesonly":
            arteriesonly = True
            print("Will only use arterial blood for generating cardiac waveform")
        elif o == "--spatialglm":
            dospatialglm = True
            print("Will generate and remove cardiac signal maps for each timepoint")
        elif o == "--temporalglm":
            dotemporalglm = True
            print("Will generate and remove cardiac timecourses from each voxel")
        elif o == "--disablenotch":
            notchpct = None
            print("Disabling subharmonic notch filter")
        elif o == "--nodetrend":
            detrendorder = 0
            print("Will disable data detrending")
        elif o == "--debug":
            debug = True
            print("Extended debugging messages")
        elif o == "--increaseoutputlevel":
            outputlevel += 1
            print("Increased output level to", outputlevel)
        elif o == "--decreaseoutputlevel":
            outputlevel -= 1
            if outputlevel < 0:
                outputlevel = 0
            print("Decreased output level to", outputlevel)
        elif o == "--savemotionglmfilt":
            savemotionglmfilt = True
        elif o == "--fliparteries":
            fliparteries = True
            print("Will detect and invert arterial timecourses.")
        elif o == "--nophasefilt":
            filtphase = False
            print("Disabling phase trend filter")
        elif o == "--nocardiacalign":
            aligncardiac = False
            print("Disabling cardiac alignment")
        elif o == "--noncentric":
            centric = False
            print("Performing noncentric projection")
        elif o == "--dodlfilter":
            if dlfilterexists:
                dodlfilter = True
                print("Will apply deep learning filter to enhance the cardiac waveforms")
            else:
                print(
                    "dlfilter not found - check to make sure Keras is installed and working.  Disabling."
                )
        elif o == "--model":
            linkchar = "="
            modelname = a
            print("Will use", modelname, "for the deep learning filter;")
        elif o == "--noprogressbar":
            showprogressbar = False
            print("Will disable progress bars")
        elif o == "--cardcalconly":
            cardcalconly = True
            print("Will stop processing after calculating cardiac waveforms")
        elif o == "--noappsmooth":
            smoothapp = False
            print("Will not smooth projection along phase direction")
        elif o == "--nomask":
            usemaskcardfromfmri = False
        elif o == "--nocensor":
            censorbadpts = False
            print("Will not censor bad points")
        elif o == "--projectwithraw":
            projectwithraw = True
            print("Will use fmri derived cardiac waveform as phase source for projection")
        elif o == "--nomadnorm":
            domadnorm = False
            print("Disabling MAD normalization between slices")
        elif o == "--outputbins":
            linkchar = "="
            destpoints = int(a)
            print("Will use", destpoints, "output bins")
        elif o == "--numskip":
            linkchar = "="
            numskip = int(a)
            print("Skipping first", numskip, "fmri trs")
        elif o == "--motskip":
            linkchar = "="
            motskip = int(a)
            print("Skipping first", motskip, "motion trs")
        elif o == "--smoothlen":
            linkchar = "="
            smoothlen = int(a)
            smoothlen = smoothlen + (1 - smoothlen % 2)
            print("Will set savitsky-golay window to", smoothlen)
        elif o == "--gridbins":
            linkchar = "="
            congridbins = float(a)
            print("Will use a convolution gridding kernel of width", congridbins, "bins")
        elif o == "--gridkernel":
            linkchar = "="
            gridkernel = a
            if gridkernel == "kaiser":
                print("Will use a kaiser-bessel gridding kernel")
            elif gridkernel == "gauss":
                print("Will use a gaussian gridding kernel")
            elif gridkernel == "old":
                print("Falling back to old style gridding")
            else:
                print("Illegal gridding kernel specified - aborting")
                sys.exit()
        elif o == "--usesuperdangerousworkaround":
            mpfix = True
            print("Trying super dangerous workaround to make dlfilter work")
        elif o == "--notchwidth":
            linkchar = "="
            notchpct = float(a)
            print("Setting notchwidth to", notchpct, "%")
        elif o == "--nprocs":
            linkchar = "="
            nprocs = int(a)
            if nprocs < 1:
                nprocs = tide_multiproc.maxcpus()
            print("Will use", nprocs, "processors for long calculations")
        elif o == "--mklthreads":
            mklthreads = int(a)
            linkchar = "="
            if mklexists:
                mklmaxthreads = mkl.get_max_threads()
                if mklthreads > mklmaxthreads:
                    print("mkl max threads =", mklmaxthreads, " - using max")
                    mklthreads = mklmaxthreads

                print(
                    "Will use", mklthreads, "MKL threads for accelerated numpy processing.",
                )
            else:
                print("MKL not present - ignoring --mklthreads")
        elif o == "--stdfreq":
            linkchar = "="
            stdfreq = float(a)
            print("Setting common output frequency to", stdfreq)
        elif o == "--envcutoff":
            linkchar = "="
            envcutoff = float(a)
            print("Will set top of cardiac envelope band to", envcutoff)
        elif o == "--envthresh":
            linkchar = "="
            envthresh = float(a)
            print(
                "Will set lowest value of cardiac envelope band to",
                envthresh,
                "x the maximum value",
            )
        elif o == "--minhr":
            newval = float(a)
            print(
                "Will set bottom of cardiac search range to", newval, "BPM from", minhr, "BPM",
            )
            minhr = newval
        elif o == "--maxhr":
            linkchar = "="
            newval = float(a)
            print(
                "Will set top of cardiac search range to", newval, "BPM from", maxhr, "BPM",
            )
            maxhr = newval
        elif o == "--minhrfilt":
            linkchar = "="
            newval = float(a)
            print(
                "Will set bottom of cardiac band to",
                newval,
                "BPM from",
                minhrfilt,
                "BPM when estimating waveform",
            )
            minhrfilt = newval
        elif o == "--maxhrfilt":
            linkchar = "="
            newval = float(a)
            print(
                "Will set top of cardiac band to",
                newval,
                "BPM from",
                maxhrfilt,
                "BPM when estimating waveform",
            )
            maxhrfilt = newval
        elif o == "--forcehr":
            linkchar = "="
            forcedhr = float(a) / 60.0
            print("Force heart rate detector to", forcedhr * 60.0, "BPM")
        elif o == "--motionhp":
            linkchar = "="
            motionhp = float(a)
            print("Will highpass motion regressors at", motionhp, "Hz prior to regression")
        elif o == "--motionlp":
            linkchar = "="
            motionlp = float(a)
            print("Will lowpass motion regressors at", motionlp, "Hz prior to regression")
        elif o == "--savetcsastsv":
            savetcsastsv = True
            print("Will save timecourses in BIDS tsv format")
        elif o == "--saveinfoasjson":
            saveinfoasjson = True
            print("Will save info file in json format")
        elif o == "--trimcorrelations":
            trimcorrelations = True
            print("Will be tolerant of short physiological timecourses")
        elif o == "--noorthog":
            orthogonalize = False
            print("Will not orthogonalize motion regressors")
        elif o == "--nomotderivdelayed":
            motfilt_derivdelayed = False
            print("Will not use motion position regressors")
        elif o == "--nomotderiv":
            motfilt_deriv = False
            print("Will not use motion derivative regressors")
        elif o == "--estmask":
            estmaskname = a
            usemaskcardfromfmri = True
            print("Will restrict cardiac waveform fit to voxels in", estmaskname)
        elif o == "--projmask":
            projmaskname = a
            useintensitymask = False
            usemaskcardfromfmri = True
            print("Will restrict phase projection to voxels in", projmaskname)
        elif o == "--cardiacfile":
            linkchar = "="
            inputlist = a.split(":")
            cardiacfilename = inputlist[0]
            if len(inputlist) > 1:
                try:
                    colnum = int(inputlist[1])
                except ValueError:
                    colname = inputlist[1]
            print("Will use cardiac file", cardiacfilename)
        elif o == "--cardiacfreq":
            linkchar = "="
            inputfreq = float(a)
            print("Setting cardiac sample frequency to ", inputfreq)
        elif o == "--cardiactstep":
            linkchar = "="
            inputfreq = 1.0 / float(a)
            print("Setting cardiac sample time step to ", float(a))
        elif o == "--cardiacstart":
            linkchar = "="
            inputstart = float(a)
            print("Setting cardiac start time to ", inputstart)
        elif o == "--help":
            usage()
            sys.exit()
        else:
            assert False, "unhandled option: " + o
        formattedcmdline.append("\t" + o + linkchar + a + " \\")
    formattedcmdline[len(formattedcmdline) - 1] = formattedcmdline[len(formattedcmdline) - 1][:-2]

    # write out the command used
    tide_util.savecommandline(thearguments, outputroot)
    tide_io.writevec(formattedcmdline, outputroot + "_formattedcommandline.txt")

    memfile = open(outputroot + "_memusage.csv", "w")
    tide_util.logmem(None, file=memfile)

    # set the number of MKL threads to use
    if mklexists:
        mkl.set_num_threads(mklthreads)

    # if we are going to do a glm, make sure we are generating app matrix
    if (dotemporalglm or dospatialglm) and cardcalconly:
        print("doing glm fit requires phase projection - setting cardcalconly to False")
        cardcalconly = False

    # save important configuration options
    infodict["aliasedcorrelationpts"] = aliasedcorrelationpts
    infodict["aliasedcorrelationwidth"] = aliasedcorrelationwidth
    infodict["aligncardiac"] = aligncardiac
    infodict["arteriesonly"] = arteriesonly
    infodict["cardcalconly"] = cardcalconly
    infodict["cardiacfilename"] = cardiacfilename
    infodict["censorbadpts"] = censorbadpts
    infodict["centric"] = centric
    infodict["colname"] = colname
    infodict["colnum"] = colnum
    infodict["congridbins"] = congridbins
    infodict["debug"] = debug
    infodict["destpoints"] = destpoints
    infodict["detrendorder"] = detrendorder
    infodict["doaliasedcorrelation"] = doaliasedcorrelation
    infodict["dodlfilter"] = dodlfilter
    infodict["domadnorm"] = domadnorm
    infodict["dospatialglm"] = dospatialglm
    infodict["dotemporalglm"] = dotemporalglm
    infodict["envcutoff"] = envcutoff
    infodict["envthresh"] = envthresh
    infodict["filtphase"] = filtphase
    infodict["fliparteries"] = fliparteries
    infodict["forcedhr"] = forcedhr
    infodict["gridkernel"] = gridkernel
    infodict["histlen"] = histlen
    infodict["inputfreq"] = inputfreq
    infodict["inputstart"] = inputstart
    infodict["maskthreshpct"] = maskthreshpct
    infodict["maxhr"] = maxhr
    infodict["maxhrfilt"] = maxhrfilt
    infodict["minhr"] = minhr
    infodict["minhrfilt"] = minhrfilt
    infodict["mklthreads"] = mklthreads
    infodict["modelname"] = modelname
    infodict["motionfilename"] = motionfilename
    infodict["motionhp"] = motionhp
    infodict["motionlp"] = motionlp
    infodict["motskip"] = motskip
    infodict["mpfix"] = mpfix
    infodict["nprocs"] = nprocs
    infodict["numskip"] = numskip
    infodict["orthogonalize"] = orthogonalize
    infodict["outputlevel"] = outputlevel
    infodict["projectwithraw"] = projectwithraw
    infodict["projmaskname"] = projmaskname
    infodict["pulsereconstepsize"] = pulsereconstepsize
    infodict["savecardiacnoise"] = savecardiacnoise
    infodict["saveinfoasjson"] = saveinfoasjson
    infodict["saveintermediate"] = saveintermediate
    infodict["showprogressbar"] = showprogressbar
    infodict["smoothapp"] = smoothapp
    infodict["smoothlen"] = smoothlen
    infodict["softvesselfrac"] = softvesselfrac
    infodict["stdfreq"] = stdfreq
    infodict["unnormvesselmap"] = unnormvesselmap
    infodict["upsamplefac"] = upsamplefac
    infodict["usemaskcardfromfmri"] = usemaskcardfromfmri
    infodict["verbose"] = verbose

    # set up cardiac filter
    arb_lower = minhrfilt / 60.0
    arb_upper = maxhrfilt / 60.0
    thecardbandfilter = tide_filt.NoncausalFilter()
    thecardbandfilter.settype("arb")
    arb_lowerstop = arb_lower * 0.9
    arb_upperstop = arb_upper * 1.1
    thecardbandfilter.setfreqs(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)
    therespbandfilter = tide_filt.NoncausalFilter()
    therespbandfilter.settype("resp")
    infodict["filtermaxbpm"] = arb_upper * 60.0
    infodict["filterminbpm"] = arb_lower * 60.0
    infodict["notchpct"] = notchpct
    timings.append(["Argument parsing done", time.time(), None, None])

    # read in the image data
    tide_util.logmem("before reading in fmri data", file=memfile)
    nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
    input_data = tide_classes.fMRIDataset(nim_data, numskip=numskip)
    timepoints = input_data.timepoints
    xsize = input_data.xsize
    ysize = input_data.ysize
    numslices = input_data.numslices

    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
    spaceunit, timeunit = nim_hdr.get_xyzt_units()
    if timeunit == "msec":
        tr /= 1000.0
    mrsamplerate = 1.0 / tr
    print("tr is", tr, "seconds, mrsamplerate is", mrsamplerate)
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    infodict["tr"] = tr
    infodict["mrsamplerate"] = mrsamplerate
    timings.append(["Image data read in", time.time(), None, None])

    # remap to space by time
    fmri_data = input_data.byvol()
    del nim_data

    # make and save a mask of the voxels to process based on image intensity
    tide_util.logmem("before mask creation", file=memfile)
    mask = np.uint16(
        tide_stats.makemask(np.mean(fmri_data[:, :], axis=1), threshpct=maskthreshpct)
    )
    validvoxels = np.where(mask > 0)[0]
    theheader = copy.deepcopy(nim_hdr)
    theheader["dim"][4] = 1
    timings.append(["Mask created", time.time(), None, None])
    if outputlevel > 0:
        tide_io.savetonifti(
            mask.reshape((xsize, ysize, numslices)), theheader, outputroot + "_mask"
        )
    timings.append(["Mask saved", time.time(), None, None])
    mask_byslice = mask.reshape((xsize * ysize, numslices))

    # read in projection mask if present otherwise fall back to intensity mask
    if projmaskname is not None:
        tide_util.logmem("before reading in projmask", file=memfile)
        projmask_byslice = readextmask(
            projmaskname, nim_hdr, xsize, ysize, numslices
        ) * np.float64(mask_byslice)
    else:
        projmask_byslice = mask_byslice

    # filter out motion regressors here
    if motionfilename is not None:
        timings.append(["Motion filtering start", time.time(), None, None])
        (motionregressors, motionregressorlabels, filtereddata,) = tide_glmpass.motionregress(
            motionfilename,
            fmri_data[validvoxels, :],
            tr,
            orthogonalize=orthogonalize,
            motstart=motskip,
            motionhp=motionhp,
            motionlp=motionlp,
            position=motfilt_pos,
            deriv=motfilt_deriv,
            derivdelayed=motfilt_derivdelayed,
        )
        fmri_data[validvoxels, :] = filtereddata[:, :]
        infodict["numorthogmotregressors"] = motionregressors.shape[0]
        timings.append(["Motion filtering end", time.time(), numspatiallocs, "voxels"])
        tide_io.writenpvecs(motionregressors, outputroot + "_orthogonalizedmotion.txt")
        if savemotionglmfilt:
            tide_io.savetonifti(
                fmri_data.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                outputroot + "_motionfiltered",
            )
            timings.append(["Motion filtered data saved", time.time(), numspatiallocs, "voxels"])

    # get slice times
    slicetimes = tide_io.getslicetimesfromfile(slicetimename)
    timings.append(["Slice times determined", time.time(), None, None])

    # normalize the input data
    tide_util.logmem("before normalization", file=memfile)
    normdata, demeandata, means = normalizevoxels(
        fmri_data, detrendorder, validvoxels, time, timings, showprogressbar=showprogressbar,
    )
    normdata_byslice = normdata.reshape((xsize * ysize, numslices, timepoints))

    # read in estimation mask if present. Otherwise, otherwise use intensity mask.
    infodict["estmaskname"] = estmaskname
    if debug:
        print(estmaskname)
    if estmaskname is not None:
        tide_util.logmem("before reading in estmask", file=memfile)
        estmask_byslice = readextmask(estmaskname, nim_hdr, xsize, ysize, numslices) * np.float64(
            mask_byslice
        )
        print("using estmask from file", estmaskname)
        numpasses = 1
    else:
        # just fall back to the intensity mask
        estmask_byslice = mask_byslice.astype("float64")
        numpasses = 2
        print("not using separate estimation mask - doing initial estimate using intensity mask")
    if fliparteries:
        # add another pass to refine the waveform after getting the new appflips
        numpasses += 1
        print("adding a pass to regenerate cardiac waveform using bettter appflips")

    infodict["numpasses"] = numpasses

    # if we have an estimation mask, run procedure once.  If not, run once to get a vessel mask, then rerun.
    appflips_byslice = None
    for thispass in range(numpasses):
        if numpasses > 1:
            print()
            print()
            print("starting pass", thispass + 1, "of", numpasses)
            passstring = " - pass " + str(thispass + 1)
        else:
            passstring = ""
        # now get an estimate of the cardiac signal
        print("estimating cardiac signal from fmri data")
        tide_util.logmem("before cardiacfromimage", file=memfile)
        (
            cardfromfmri_sliceres,
            cardfromfmri_normfac,
            respfromfmri_sliceres,
            respfromfmri_normfac,
            slicesamplerate,
            numsteps,
            cycleaverage,
            slicenorms,
        ) = physiofromimage(
            normdata_byslice,
            estmask_byslice,
            numslices,
            timepoints,
            tr,
            slicetimes,
            thecardbandfilter,
            therespbandfilter,
            madnorm=domadnorm,
            nprocs=nprocs,
            notchpct=notchpct,
            fliparteries=fliparteries,
            arteriesonly=arteriesonly,
            usemask=usemaskcardfromfmri,
            appflips_byslice=appflips_byslice,
            debug=debug,
            verbose=verbose,
        )
        timings.append(
            ["Cardiac signal generated from image data" + passstring, time.time(), None, None,]
        )
        infodict["cardfromfmri_normfac"] = cardfromfmri_normfac
        slicetimeaxis = np.linspace(
            0.0, tr * timepoints, num=(timepoints * numsteps), endpoint=False
        )
        if thispass == numpasses - 1:
            tide_io.writevec(cycleaverage, outputroot + "_cycleaverage.txt")
            tide_io.writevec(cardfromfmri_sliceres, outputroot + "_cardfromfmri_sliceres.txt")
        else:
            if saveintermediate:
                tide_io.writevec(
                    cycleaverage, outputroot + "_cycleaverage_pass" + str(thispass + 1) + ".txt",
                )
                tide_io.writevec(
                    cardfromfmri_sliceres,
                    outputroot + "_cardfromfmri_sliceres_pass" + str(thispass + 1) + ".txt",
                )

        # stash away a copy of the waveform if we need it later
        raw_cardfromfmri_sliceres = np.array(cardfromfmri_sliceres)

        # find bad points in cardiac from fmri
        thebadcardpts = findbadpts(
            cardfromfmri_sliceres, "cardfromfmri_sliceres", outputroot, slicesamplerate, infodict,
        )

        cardiacwaveform = np.array(cardfromfmri_sliceres)
        badpointlist = np.array(thebadcardpts)

        infodict["slicesamplerate"] = slicesamplerate
        infodict["numcardpts_sliceres"] = timepoints * numsteps
        infodict["numsteps"] = numsteps
        infodict["slicenorms"] = slicenorms

        # find key components of cardiac waveform
        print("extracting harmonic components")
        if outputlevel > 1:
            if thispass == numpasses - 1:
                tide_io.writevec(
                    cardfromfmri_sliceres * (1.0 - thebadcardpts),
                    outputroot + "_cardfromfmri_sliceres_censored.txt",
                )
        peakfreq_bold = getcardcoeffs(
            (1.0 - thebadcardpts) * cardiacwaveform,
            slicesamplerate,
            minhr=minhr,
            maxhr=maxhr,
            smoothlen=smoothlen,
            debug=debug,
        )
        infodict["cardiacbpm_bold"] = np.round(peakfreq_bold * 60.0, 2)
        infodict["cardiacfreq_bold"] = peakfreq_bold
        timings.append(
            ["Cardiac signal from image data analyzed" + passstring, time.time(), None, None,]
        )

        # resample to standard frequency
        cardfromfmri_stdres = tide_math.madnormalize(
            tide_resample.arbresample(
                cardfromfmri_sliceres, slicesamplerate, stdfreq, decimate=True, debug=False,
            )
        )

        if thispass == numpasses - 1:
            tide_io.writevec(
                cardfromfmri_stdres, outputroot + "_cardfromfmri_" + str(stdfreq) + "Hz.txt",
            )
        else:
            if saveintermediate:
                tide_io.writevec(
                    cardfromfmri_stdres,
                    outputroot
                    + "_cardfromfmri_"
                    + str(stdfreq)
                    + "Hz_pass"
                    + str(thispass + 1)
                    + ".txt",
                )
        infodict["numcardpts_stdres"] = len(cardfromfmri_stdres)

        # normalize the signal to remove envelope effects
        (
            filtcardfromfmri_stdres,
            normcardfromfmri_stdres,
            cardfromfmrienv_stdres,
            envmean,
        ) = cleancardiac(
            stdfreq,
            cardfromfmri_stdres,
            cutoff=envcutoff,
            nyquist=slicesamplerate / 2.0,
            thresh=envthresh,
        )
        if thispass == numpasses - 1:
            tide_io.writevec(
                normcardfromfmri_stdres,
                outputroot + "_normcardfromfmri_" + str(stdfreq) + "Hz.txt",
            )
            tide_io.writevec(
                cardfromfmrienv_stdres, outputroot + "_cardfromfmrienv_" + str(stdfreq) + "Hz.txt",
            )
        else:
            if saveintermediate:
                tide_io.writevec(
                    normcardfromfmri_stdres,
                    outputroot
                    + "_normcardfromfmri_"
                    + str(stdfreq)
                    + "Hz_pass"
                    + str(thispass + 1)
                    + ".txt",
                )
                tide_io.writevec(
                    cardfromfmrienv_stdres,
                    outputroot
                    + "_cardfromfmrienv_"
                    + str(stdfreq)
                    + "Hz_pass"
                    + str(thispass + 1)
                    + ".txt",
                )

        # calculate quality metrics
        calcplethquality(
            normcardfromfmri_stdres,
            stdfreq,
            infodict,
            "_bold",
            outputroot,
            outputlevel=outputlevel,
        )

        thebadcardpts_stdres = findbadpts(
            cardfromfmri_stdres,
            "cardfromfmri_" + str(stdfreq) + "Hz",
            outputroot,
            stdfreq,
            infodict,
        )

        timings.append(
            [
                "Cardiac signal from image data resampled and saved" + passstring,
                time.time(),
                None,
                None,
            ]
        )

        # apply the deep learning filter if we're going to do that
        if dodlfilter:
            if dlfilterexists:
                if mpfix:
                    print("performing super dangerous openmp workaround")
                    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                modelpath = os.path.join(
                    os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0],
                    "rapidtide",
                    "data",
                    "models",
                )
                thedlfilter = tide_dlfilt.DeepLearningFilter(modelpath=modelpath)
                thedlfilter.loadmodel(modelname)
                infodict["dlfiltermodel"] = modelname
                normdlfilteredcard = thedlfilter.apply(normcardfromfmri_stdres)
                dlfilteredcard = thedlfilter.apply(cardfromfmri_stdres)
                if thispass == numpasses - 1:
                    tide_io.writevec(
                        normdlfilteredcard,
                        outputroot + "_normcardfromfmri_dlfiltered_" + str(stdfreq) + "Hz.txt",
                    )
                    tide_io.writevec(
                        dlfilteredcard,
                        outputroot + "_cardfromfmri_dlfiltered_" + str(stdfreq) + "Hz.txt",
                    )
                else:
                    if saveintermediate:
                        tide_io.writevec(
                            normdlfilteredcard,
                            outputroot
                            + "_normcardfromfmri_dlfiltered_"
                            + str(stdfreq)
                            + "Hz_pass"
                            + str(thispass + 1)
                            + ".txt",
                        )
                        tide_io.writevec(
                            dlfilteredcard,
                            outputroot
                            + "_cardfromfmri_dlfiltered_"
                            + str(stdfreq)
                            + "Hz_pass"
                            + str(thispass + 1)
                            + ".txt",
                        )

                # calculate quality metrics
                calcplethquality(
                    dlfilteredcard,
                    stdfreq,
                    infodict,
                    "_dlfiltered",
                    outputroot,
                    outputlevel=outputlevel,
                )

                # downsample to sliceres from stdres
                # cardfromfmri_sliceres = tide_math.madnormalize(
                #    tide_resample.arbresample(dlfilteredcard, stdfreq, slicesamplerate, decimate=True, debug=False))
                stdtimeaxis = (1.0 / stdfreq) * np.linspace(
                    0.0, len(dlfilteredcard), num=(len(dlfilteredcard)), endpoint=False
                )
                arb_lowerstop = 0.0
                arb_lowerpass = 0.0
                arb_upperpass = slicesamplerate / 2.0
                arb_upperstop = slicesamplerate / 2.0
                theaafilter = tide_filt.NoncausalFilter(filtertype="arb")
                theaafilter.setfreqs(arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop)

                cardfromfmri_sliceres = tide_math.madnormalize(
                    tide_resample.doresample(
                        stdtimeaxis,
                        theaafilter.apply(stdfreq, dlfilteredcard),
                        slicetimeaxis,
                        method="univariate",
                        padlen=0,
                    )
                )
                if thispass == numpasses - 1:
                    tide_io.writevec(
                        cardfromfmri_sliceres,
                        outputroot + "_cardfromfmri_dlfiltered_sliceres.txt",
                    )
                infodict["used_dlreconstruction_filter"] = True
                peakfreq_dlfiltered = getcardcoeffs(
                    cardfromfmri_sliceres,
                    slicesamplerate,
                    minhr=minhr,
                    maxhr=maxhr,
                    smoothlen=smoothlen,
                    debug=debug,
                )
                infodict["cardiacbpm_dlfiltered"] = np.round(peakfreq_dlfiltered * 60.0, 2)
                infodict["cardiacfreq_dlfiltered"] = peakfreq_dlfiltered

                # check the match between the raw and filtered cardiac signals
                maxval, maxdelay, failreason = checkcardmatch(
                    raw_cardfromfmri_sliceres, cardfromfmri_sliceres, slicesamplerate, debug=debug,
                )
                print(
                    "Filtered cardiac fmri waveform delay is",
                    maxdelay,
                    "relative to raw fMRI data",
                )
                print("Correlation coefficient between cardiac regressors:", maxval)
                infodict["corrcoeff_raw2filt"] = maxval + 0
                infodict["delay_raw2filt"] = maxdelay + 0
                infodict["failreason_raw2filt"] = failreason + 0

                timings.append(
                    ["Deep learning filter applied" + passstring, time.time(), None, None,]
                )
            else:
                print("dlfilter could not be loaded - skipping")

        # get the cardiac signal from a file, if specified
        if cardiacfilename is not None:
            tide_util.logmem("before cardiacfromfile", file=memfile)
            pleth_sliceres, pleth_stdres = getphysiofile(
                cardiacfilename,
                colnum,
                colname,
                inputfreq,
                inputstart,
                slicetimeaxis,
                stdfreq,
                envcutoff,
                envthresh,
                timings,
                infodict,
                outputroot,
                outputlevel=outputlevel,
                debug=False,
            )

            if dodlfilter and dlfilterexists:
                maxval, maxdelay, failreason = checkcardmatch(
                    pleth_sliceres, cardfromfmri_sliceres, slicesamplerate, debug=debug
                )
                print(
                    "Input cardiac waveform delay is", maxdelay, "relative to filtered fMRI data",
                )
                print("Correlation coefficient between cardiac regressors:", maxval)
                infodict["corrcoeff_filt2pleth"] = maxval + 0
                infodict["delay_filt2pleth"] = maxdelay + 0
                infodict["failreason_filt2pleth"] = failreason + 0

            # check the match between the bold and physio cardiac signals
            maxval, maxdelay, failreason = checkcardmatch(
                pleth_sliceres, raw_cardfromfmri_sliceres, slicesamplerate, debug=debug
            )
            print("Input cardiac waveform delay is", maxdelay, "relative to fMRI data")
            print("Correlation coefficient between cardiac regressors:", maxval)
            infodict["corrcoeff_raw2pleth"] = maxval + 0
            infodict["delay_raw2pleth"] = maxdelay + 0
            infodict["failreason_raw2pleth"] = failreason + 0

            # align the pleth signal with the cardiac signal derived from the data
            if aligncardiac:
                alignpts_sliceres = -maxdelay / slicesamplerate  # maxdelay is in seconds
                pleth_sliceres, dummy1, dummy2, dummy2 = tide_resample.timeshift(
                    pleth_sliceres, alignpts_sliceres, int(10.0 * slicesamplerate)
                )
                alignpts_stdres = -maxdelay * stdfreq  # maxdelay is in seconds
                pleth_stdres, dummy1, dummy2, dummy3 = tide_resample.timeshift(
                    pleth_stdres, alignpts_stdres, int(10.0 * stdfreq)
                )
            if thispass == numpasses - 1:
                tide_io.writevec(pleth_sliceres, outputroot + "_pleth_sliceres.txt")
                tide_io.writevec(pleth_stdres, outputroot + "_pleth_" + str(stdfreq) + "Hz.txt")

            # now clean up cardiac signal
            filtpleth_stdres, normpleth_stdres, plethenv_stdres, envmean = cleancardiac(
                stdfreq, pleth_stdres, cutoff=envcutoff, thresh=envthresh
            )
            if thispass == numpasses - 1:
                tide_io.writevec(
                    normpleth_stdres, outputroot + "_normpleth_" + str(stdfreq) + "Hz.txt",
                )
                tide_io.writevec(
                    plethenv_stdres, outputroot + "_plethenv_" + str(stdfreq) + "Hz.txt"
                )

            # calculate quality metrics
            calcplethquality(
                filtpleth_stdres, stdfreq, infodict, "_pleth", outputroot, outputlevel=outputlevel,
            )

            if dodlfilter and dlfilterexists:
                dlfilteredpleth = thedlfilter.apply(pleth_stdres)
                if thispass == numpasses - 1:
                    tide_io.writevec(
                        dlfilteredpleth,
                        outputroot + "_pleth_dlfiltered_" + str(stdfreq) + "Hz.txt",
                    )
                    maxval, maxdelay, failreason = checkcardmatch(
                        pleth_stdres, dlfilteredpleth, stdfreq, debug=debug
                    )
                    print(
                        "Filtered pleth cardiac waveform delay is",
                        maxdelay,
                        "relative to raw pleth data",
                    )
                    print("Correlation coefficient between pleth regressors:", maxval)
                    infodict["corrcoeff_pleth2filtpleth"] = maxval + 0
                    infodict["delay_pleth2filtpleth"] = maxdelay + 0
                    infodict["failreason_pleth2filtpleth"] = failreason + 0

            # find bad points in plethysmogram
            thebadplethpts_sliceres = findbadpts(
                pleth_sliceres,
                "pleth_sliceres",
                outputroot,
                slicesamplerate,
                infodict,
                thetype="fracval",
            )

            thebadplethpts_stdres = findbadpts(
                pleth_stdres,
                "pleth_" + str(stdfreq) + "Hz",
                outputroot,
                stdfreq,
                infodict,
                thetype="fracval",
            )
            timings.append(
                [
                    "Cardiac signal from physiology data resampled to standard and saved"
                    + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )

            # find key components of cardiac waveform
            filtpleth = tide_math.madnormalize(
                thecardbandfilter.apply(slicesamplerate, pleth_sliceres)
            )
            peakfreq_file = getcardcoeffs(
                (1.0 - thebadplethpts_sliceres) * filtpleth,
                slicesamplerate,
                minhr=minhr,
                maxhr=maxhr,
                smoothlen=smoothlen,
                debug=debug,
            )
            timings.append(
                [
                    "Cardiac coefficients calculated from pleth waveform" + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )
            infodict["cardiacbpm_pleth"] = np.round(peakfreq_file * 60.0, 2)
            infodict["cardiacfreq_pleth"] = peakfreq_file
            timings.append(
                [
                    "Cardiac signal from physiology data analyzed" + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )
            timings.append(
                [
                    "Cardiac parameters extracted from physiology data" + passstring,
                    time.time(),
                    None,
                    None,
                ]
            )

            if not projectwithraw:
                cardiacwaveform = np.array(pleth_sliceres)
                badpointlist = 1.0 - (1.0 - thebadplethpts_sliceres) * (1.0 - badpointlist)

            infodict["pleth"] = True
            peakfreq = peakfreq_file
        else:
            infodict["pleth"] = False
            peakfreq = peakfreq_bold
        if outputlevel > 0:
            if thispass == numpasses - 1:
                tide_io.writevec(badpointlist, outputroot + "_overall_sliceres_badpts.txt")

        #  extract the fundamental
        if forcedhr is not None:
            peakfreq = forcedhr
            infodict["forcedhr"] = peakfreq
        if cardiacfilename is None:
            filthiresfund = tide_math.madnormalize(
                getfundamental(cardiacwaveform * (1.0 - thebadcardpts), slicesamplerate, peakfreq)
            )
        else:
            filthiresfund = tide_math.madnormalize(
                getfundamental(cardiacwaveform, slicesamplerate, peakfreq)
            )
        if outputlevel > 1:
            if thispass == numpasses - 1:
                tide_io.writevec(filthiresfund, outputroot + "_cardiacfundamental_sliceres.txt")

        # now calculate the phase waveform
        tide_util.logmem("before analytic phase analysis", file=memfile)
        instantaneous_phase, amplitude_envelope = tide_fit.phaseanalysis(filthiresfund)
        if outputlevel > 0:
            if thispass == numpasses - 1:
                tide_io.writevec(amplitude_envelope, outputroot + "_ampenv_sliceres.txt")
                tide_io.writevec(
                    instantaneous_phase, outputroot + "_instphase_unwrapped_sliceres.txt",
                )

        if filtphase:
            print("filtering phase waveform")
            instantaneous_phase = tide_math.trendfilt(instantaneous_phase, debug=False)
            if outputlevel > 1:
                if thispass == numpasses - 1:
                    tide_io.writevec(
                        instantaneous_phase, outputroot + "_filtered_instphase_unwrapped.txt",
                    )
        initialphase = instantaneous_phase[0]
        infodict["phi0"] = initialphase
        timings.append(["Phase waveform generated" + passstring, time.time(), None, None])

        # account for slice time offests
        offsets_byslice = np.zeros((xsize * ysize, numslices), dtype=np.float64)
        for i in range(numslices):
            offsets_byslice[:, i] = slicetimes[i]

        # remap offsets to space by time
        fmri_offsets = offsets_byslice.reshape(numspatiallocs)

        # save the information file
        if saveinfoasjson:
            tide_io.writedicttojson(infodict, outputroot + "_info.json")
        else:
            tide_io.writedict(infodict, outputroot + "_info.txt")

        # interpolate the instantaneous phase
        upsampledslicetimeaxis = np.linspace(
            0.0, tr * timepoints, num=(timepoints * numsteps * upsamplefac), endpoint=False,
        )
        interpphase = tide_math.phasemod(
            tide_resample.doresample(
                slicetimeaxis,
                instantaneous_phase,
                upsampledslicetimeaxis,
                method="univariate",
                padlen=0,
            ),
            centric=centric,
        )
        if outputlevel > 1:
            if thispass == numpasses - 1:
                tide_io.writevec(interpphase, outputroot + "_interpinstphase.txt")

        if cardcalconly:
            print("cardiac waveform calculations done - exiting")
            # Process and save timing information
            nodeline = "Processed on " + platform.node()
            tide_util.proctiminginfo(
                timings, outputfile=outputroot + "_runtimings.txt", extraheader=nodeline
            )
            tide_util.logmem("final", file=memfile)
            sys.exit()

        # find the phase values for all timepoints in all slices
        phasevals = np.zeros((numslices, timepoints), dtype=np.float64)
        thetimes = []
        for theslice in range(numslices):
            thetimes.append(
                np.linspace(0.0, tr * timepoints, num=timepoints, endpoint=False)
                + slicetimes[theslice]
            )
            phasevals[theslice, :] = tide_math.phasemod(
                tide_resample.doresample(
                    slicetimeaxis,
                    instantaneous_phase,
                    thetimes[-1],
                    method="univariate",
                    padlen=0,
                ),
                centric=centric,
            )
            if debug:
                if thispass == numpasses - 1:
                    tide_io.writevec(
                        thetimes[-1], outputroot + "_times_" + str(theslice).zfill(2) + ".txt",
                    )
                    tide_io.writevec(
                        phasevals[theslice, :],
                        outputroot + "_phasevals_" + str(theslice).zfill(2) + ".txt",
                    )
        timings.append(
            ["Slice phases determined for all timepoints" + passstring, time.time(), None, None,]
        )

        # construct the destination arrays
        tide_util.logmem("before making destination arrays", file=memfile)
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
        demeandata_byslice = demeandata.reshape((xsize * ysize, numslices, timepoints))
        means_byslice = means.reshape((xsize * ysize, numslices))

        timings.append(["Phase projection to image started" + passstring, time.time(), None, None])
        print("starting phase projection")
        proctrs = range(timepoints)  # proctrs is the list of all fmri trs to be projected
        procpoints = range(
            timepoints * numsteps
        )  # procpoints is the list of all sliceres datapoints to be projected
        if censorbadpts:
            censortrs = np.zeros(timepoints, dtype="int")
            censorpoints = np.zeros(timepoints * numsteps, dtype="int")
            censortrs[np.where(badpointlist > 0.0)[0] // numsteps] = 1
            censorpoints[np.where(badpointlist > 0.0)[0]] = 1
            proctrs = np.where(censortrs < 1)[0]
            procpoints = np.where(censorpoints < 1)[0]

        # do phase averaging
        app_bypoint = cardiaccycleaverage(
            instantaneous_phase,
            outphases,
            cardfromfmri_sliceres,
            procpoints,
            congridbins,
            gridkernel,
            centric,
            cyclic=True,
        )
        if thispass == numpasses - 1:
            tide_io.writevec(app_bypoint, outputroot + "_cardcyclefromfmri.txt")

        # now do time averaging
        lookaheadval = int(slicesamplerate / 4.0)
        print("lookaheadval = ", lookaheadval)
        wrappedphase = tide_math.phasemod(instantaneous_phase, centric=centric)
        max_peaks, min_peaks = tide_fit.peakdetect(wrappedphase, lookahead=lookaheadval)
        # start on a maximum
        if max_peaks[0][0] > min_peaks[0][0]:
            min_peaks = min_peaks[1:]
        # work only with pairs
        if len(max_peaks) > len(min_peaks):
            max_peaks = max_peaks[:-1]

        # max_peaks, min_peaks = findphasecuts(tide_math.phasemod(instantaneous_phase, centric=centric))
        zerophaselocs = []
        for idx, peak in enumerate(max_peaks):
            minloc = min_peaks[idx][0]
            maxloc = max_peaks[idx][0]
            minval = min_peaks[idx][1]
            maxval = max_peaks[idx][1]
            if minloc > 0:
                if wrappedphase[minloc - 1] < wrappedphase[minloc]:
                    minloc -= 1
                    minval = wrappedphase[minloc]
            phasediff = minval - (maxval - 2.0 * np.pi)
            timediff = minloc - maxloc
            zerophaselocs.append(1.0 * minloc - (minval - outphases[0]) * timediff / phasediff)
            # print(idx, [maxloc, maxval], [minloc, minval], phasediff, timediff, zerophaselocs[-1])
        instantaneous_time = instantaneous_phase * 0.0

        whichpeak = 0
        for t in procpoints:
            if whichpeak < len(zerophaselocs) - 1:
                if t > zerophaselocs[whichpeak + 1]:
                    whichpeak += 1
            if t > zerophaselocs[whichpeak]:
                instantaneous_time[t] = (t - zerophaselocs[whichpeak]) / slicesamplerate
            # print(t, whichpeak, zerophaselocs[whichpeak], instantaneous_time[t])
        maxtime = (
            np.ceil(
                int(1.02 * tide_stats.getfracval(instantaneous_time, 0.98) // pulsereconstepsize)
            )
            * pulsereconstepsize
        )
        outtimes = np.linspace(0.0, maxtime, num=int(maxtime / pulsereconstepsize), endpoint=False)
        atp_bypoint = cardiaccycleaverage(
            instantaneous_time,
            outtimes,
            cardfromfmri_sliceres,
            procpoints,
            congridbins,
            gridkernel,
            False,
            cyclic=True,
        )
        if thispass == numpasses - 1:
            tide_io.writevec(atp_bypoint, outputroot + "_cardpulsefromfmri.txt")
        else:
            if saveintermediate:
                tide_io.writevec(
                    atp_bypoint,
                    outputroot + "_cardpulsefromfmri_pass" + str(thispass + 1) + ".txt",
                )

        if not verbose:
            print("phase projecting...")

        # make a lowpass filter for the projected data. Limit frequency to 3 cycles per 2pi (1/6th Fs)
        phaseFs = 1.0 / phasestep
        phaseFc = phaseFs / 6.0
        appsmoothingfilter = tide_filt.NoncausalFilter("arb", cyclic=True, padtime=0.0)
        appsmoothingfilter.setfreqs(0.0, 0.0, phaseFc, phaseFc)

        # setup for aliased correlation if we're going to do it
        if doaliasedcorrelation and (thispass == numpasses - 1):
            if cardiacfilename:
                signal_stdres = pleth_stdres
            else:
                signal_stdres = dlfilteredcard
            corrsearchvals = (
                np.linspace(0.0, aliasedcorrelationwidth, num=aliasedcorrelationpts)
                - aliasedcorrelationwidth / 2.0
            )
            theAliasedCorrelator = tide_corr.AliasedCorrelator(
                signal_stdres,
                stdfreq,
                mrsamplerate,
                corrsearchvals,
                padtime=aliasedcorrelationwidth,
            )
            thecorrfunc = np.zeros(
                (xsize, ysize, numslices, aliasedcorrelationpts), dtype=np.float64
            )
            thecorrfunc_byslice = thecorrfunc.reshape(
                (xsize * ysize, numslices, aliasedcorrelationpts)
            )
            wavedelay = np.zeros((xsize, ysize, numslices), dtype=np.float)
            wavedelay_byslice = wavedelay.reshape((xsize * ysize, numslices))
            waveamp = np.zeros((xsize, ysize, numslices), dtype=np.float)
            waveamp_byslice = waveamp.reshape((xsize * ysize, numslices))

        # now project the data
        fmri_data_byslice = input_data.byslice()
        for theslice in range(numslices):
            if showprogressbar:
                tide_util.progressbar(theslice + 1, numslices, label="Percent complete")
            if verbose:
                print("phase projecting for slice", theslice)
            validlocs = np.where(projmask_byslice[:, theslice] > 0)[0]
            # indexlist = range(0, len(phasevals[theslice, :]))
            if len(validlocs) > 0:
                for t in proctrs:
                    filteredmr = -demeandata_byslice[validlocs, theslice, t]
                    cinemr = fmri_data_byslice[validlocs, theslice, t]
                    thevals, theweights, theindices = tide_resample.congrid(
                        outphases,
                        phasevals[theslice, t],
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
            if smoothapp:
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
            timecoursemean = np.mean(rawapp_byslice[validlocs, theslice, :], axis=1).reshape(
                (-1, 1)
            )
            if fliparteries:
                corrected_rawapp_byslice[validlocs, theslice, :] = (
                    rawapp_byslice[validlocs, theslice, :] - timecoursemean
                ) * appflips_byslice[validlocs, theslice, None] + timecoursemean
                if doaliasedcorrelation and (thispass == numpasses - 1):
                    for theloc in validlocs:
                        thecorrfunc_byslice[theloc, theslice, :] = theAliasedCorrelator.apply(
                            -appflips_byslice[theloc, theslice]
                            * demeandata_byslice[theloc, theslice, :],
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
                if doaliasedcorrelation and (thispass == numpasses - 1):
                    for theloc in validlocs:
                        thecorrfunc_byslice[theloc, theslice, :] = theAliasedCorrelator.apply(
                            -demeandata_byslice[theloc, theslice, :], -thetimes[theslice][0],
                        )
                        maxloc = np.argmax(np.abs(thecorrfunc_byslice[theloc, theslice, :]))
                        wavedelay_byslice[theloc, theslice] = corrsearchvals[maxloc]
                        waveamp_byslice[theloc, theslice] = thecorrfunc_byslice[
                            theloc, theslice, maxloc
                        ]
            timecoursemin = np.min(
                corrected_rawapp_byslice[validlocs, theslice, :], axis=1
            ).reshape((-1, 1))
            app_byslice[validlocs, theslice, :] = (
                corrected_rawapp_byslice[validlocs, theslice, :] - timecoursemin
            )
            normapp_byslice[validlocs, theslice, :] = np.nan_to_num(
                app_byslice[validlocs, theslice, :] / means_byslice[validlocs, theslice, None]
            )
        if not verbose:
            print(" done")
        timings.append(
            ["Phase projection to image completed" + passstring, time.time(), None, None,]
        )
        print("phase projection done")

        # save the analytic phase projection image
        theheader = copy.deepcopy(nim_hdr)
        theheader["dim"][4] = destpoints
        theheader["toffset"] = -np.pi
        theheader["pixdim"][4] = 2.0 * np.pi / destpoints
        if thispass == numpasses - 1:
            tide_io.savetonifti(app, theheader, outputroot + "_app")
            tide_io.savetonifti(normapp, theheader, outputroot + "_normapp")
            tide_io.savetonifti(cine, theheader, outputroot + "_cine")
            if outputlevel > 0:
                tide_io.savetonifti(rawapp, theheader, outputroot + "_rawapp")
        timings.append(["Phase projected data saved" + passstring, time.time(), None, None])

        if doaliasedcorrelation and thispass == numpasses - 1:
            theheader = copy.deepcopy(nim_hdr)
            theheader["dim"][4] = aliasedcorrelationpts
            theheader["toffset"] = 0.0
            theheader["pixdim"][4] = corrsearchvals[1] - corrsearchvals[0]
            tide_io.savetonifti(thecorrfunc, theheader, outputroot + "_corrfunc")
            theheader["dim"][4] = 1
            tide_io.savetonifti(wavedelay, theheader, outputroot + "_wavedelay")
            tide_io.savetonifti(waveamp, theheader, outputroot + "_waveamp")

        # make and save a voxel intensity histogram
        if unnormvesselmap:
            app2d = app.reshape((numspatiallocs, destpoints))
        else:
            app2d = normapp.reshape((numspatiallocs, destpoints))
        validlocs = np.where(mask > 0)[0]
        histinput = app2d[validlocs, :].reshape((len(validlocs), destpoints))
        if outputlevel > 0:
            tide_stats.makeandsavehistogram(histinput, histlen, 0, outputroot + "_histogram")

        # find vessel threshholds
        tide_util.logmem("before making vessel masks", file=memfile)
        hardvesselthresh = tide_stats.getfracvals(np.max(histinput, axis=1), [0.98])[0] / 2.0
        softvesselthresh = softvesselfrac * hardvesselthresh
        print("hard, soft vessel threshholds set to", hardvesselthresh, softvesselthresh)

        # save a vessel masked version of app
        if unnormvesselmap:
            vesselmask = np.where(np.max(app, axis=3) > softvesselthresh, 1, 0)
        else:
            vesselmask = np.where(np.max(normapp, axis=3) > softvesselthresh, 1, 0)
        maskedapp2d = np.array(app2d)
        maskedapp2d[np.where(vesselmask.reshape(numspatiallocs) == 0)[0], :] = 0.0
        if outputlevel > 1:
            if thispass == numpasses - 1:
                tide_io.savetonifti(
                    maskedapp2d.reshape((xsize, ysize, numslices, destpoints)),
                    theheader,
                    outputroot + "_maskedapp",
                )
        del maskedapp2d
        timings.append(
            ["Vessel masked phase projected data saved" + passstring, time.time(), None, None,]
        )

        # save multiple versions of the hard vessel mask
        if unnormvesselmap:
            vesselmask = np.where(np.max(app, axis=3) > hardvesselthresh, 1, 0)
            minphase = np.argmin(app, axis=3) * 2.0 * np.pi / destpoints - np.pi
            maxphase = np.argmax(app, axis=3) * 2.0 * np.pi / destpoints - np.pi
        else:
            vesselmask = np.where(np.max(normapp, axis=3) > hardvesselthresh, 1, 0)
            minphase = np.argmin(normapp, axis=3) * 2.0 * np.pi / destpoints - np.pi
            maxphase = np.argmax(normapp, axis=3) * 2.0 * np.pi / destpoints - np.pi
        risediff = (maxphase - minphase) * vesselmask
        arteries = np.where(appflips_byslice.reshape((xsize, ysize, numslices)) < 0, vesselmask, 0)
        veins = np.where(appflips_byslice.reshape((xsize, ysize, numslices)) > 0, vesselmask, 0)
        theheader = copy.deepcopy(nim_hdr)
        theheader["dim"][4] = 1
        if thispass == numpasses - 1:
            tide_io.savetonifti(vesselmask, theheader, outputroot + "_vesselmask")
            if outputlevel > 0:
                tide_io.savetonifti(minphase, theheader, outputroot + "_minphase")
                tide_io.savetonifti(maxphase, theheader, outputroot + "_maxphase")
                tide_io.savetonifti(arteries, theheader, outputroot + "_arteries")
                tide_io.savetonifti(veins, theheader, outputroot + "_veins")
        timings.append(["Masks saved" + passstring, time.time(), None, None])

        # now get ready to start again with a new mask
        estmask_byslice = vesselmask.reshape((xsize * ysize, numslices)) + 0

    # save a vessel image
    if unnormvesselmap:
        vesselmap = np.max(app, axis=3)
    else:
        vesselmap = np.max(normapp, axis=3)
    tide_io.savetonifti(vesselmap, theheader, outputroot + "_vesselmap")
    tide_io.savetonifti(
        np.where(appflips_byslice.reshape((xsize, ysize, numslices)) < 0, vesselmap, 0.0),
        theheader,
        outputroot + "_arterymap",
    )
    tide_io.savetonifti(
        np.where(appflips_byslice.reshape((xsize, ysize, numslices)) > 0, vesselmap, 0.0),
        theheader,
        outputroot + "_veinmap",
    )

    # now generate aliased cardiac signals and regress them out of the data
    if dotemporalglm or dospatialglm:
        # generate the signals
        timings.append(["Cardiac signal regression started", time.time(), None, None])
        tide_util.logmem("before cardiac regression", file=memfile)
        print("generating cardiac regressors")
        cardiacnoise = fmri_data * 0.0
        cardiacnoise_byslice = cardiacnoise.reshape((xsize * ysize, numslices, timepoints))
        phaseindices = (cardiacnoise * 0.0).astype(np.int16)
        phaseindices_byslice = phaseindices.reshape((xsize * ysize, numslices, timepoints))
        for theslice in range(numslices):
            print("calculating cardiac noise for slice", theslice)
            validlocs = np.where(projmask_byslice[:, theslice] > 0)[0]
            for t in range(timepoints):
                phaseindices_byslice[validlocs, theslice, t] = tide_util.valtoindex(
                    outphases, phasevals[theslice, t]
                )
                cardiacnoise_byslice[validlocs, theslice, t] = rawapp_byslice[
                    validlocs, theslice, phaseindices_byslice[validlocs, theslice, t]
                ]
        theheader = copy.deepcopy(nim_hdr)
        timings.append(["Cardiac signal generated", time.time(), None, None])
        if savecardiacnoise:
            tide_io.savetonifti(
                cardiacnoise.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                outputroot + "_cardiacnoise",
            )
            tide_io.savetonifti(
                phaseindices.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                outputroot + "_phaseindices",
            )
            timings.append(["Cardiac signal saved", time.time(), None, None])

        # now remove them
        tide_util.logmem("before cardiac removal", file=memfile)
        print("removing cardiac signal with GLM")
        filtereddata = 0.0 * fmri_data
        validlocs = np.where(mask > 0)[0]
        numvalidspatiallocs = len(validlocs)
        threshval = 0.0
        if dospatialglm:
            meanvals = np.zeros(timepoints, dtype=np.float64)
            rvals = np.zeros(timepoints, dtype=np.float64)
            r2vals = np.zeros(timepoints, dtype=np.float64)
            fitcoffs = np.zeros(timepoints, dtype=np.float64)
            fitNorm = np.zeros(timepoints, dtype=np.float64)
            datatoremove = 0.0 * fmri_data
            print("running spatial glm on", timepoints, "timepoints")
            tide_glmpass.glmpass(
                timepoints,
                fmri_data[validlocs, :],
                threshval,
                cardiacnoise[validlocs, :],
                meanvals,
                rvals,
                r2vals,
                fitcoffs,
                fitNorm,
                datatoremove[validlocs, :],
                filtereddata[validlocs, :],
                reportstep=(timepoints // 100),
                mp_chunksize=10,
                procbyvoxel=False,
                nprocs=nprocs,
            )
            print(datatoremove.shape, cardiacnoise.shape, fitcoffs.shape)
            datatoremove[validlocs, :] = np.multiply(cardiacnoise[validlocs, :], fitcoffs[:, None])
            filtereddata = fmri_data - datatoremove
            timings.append(
                [
                    "Cardiac signal spatial regression finished",
                    time.time(),
                    timepoints,
                    "timepoints",
                ]
            )
            tide_io.writevec(fitcoffs, outputroot + "_fitcoff.txt")
            tide_io.writevec(meanvals, outputroot + "_fitmean.txt")
            tide_io.writevec(rvals, outputroot + "_fitR.txt")
            theheader = copy.deepcopy(nim_hdr)
            tide_io.savetonifti(
                filtereddata.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                outputroot + "_temporalfiltereddata",
            )
            tide_io.savetonifti(
                datatoremove.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                outputroot + "_temporaldatatoremove",
            )
            timings.append(
                ["Cardiac signal spatial regression files written", time.time(), None, None,]
            )

        if dotemporalglm:
            meanvals = np.zeros(numspatiallocs, dtype=np.float64)
            rvals = np.zeros(numspatiallocs, dtype=np.float64)
            r2vals = np.zeros(numspatiallocs, dtype=np.float64)
            fitcoffs = np.zeros(numspatiallocs, dtype=np.float64)
            fitNorm = np.zeros(numspatiallocs, dtype=np.float64)
            datatoremove = 0.0 * fmri_data
            print("running temporal glm on", numvalidspatiallocs, "voxels")
            tide_glmpass.glmpass(
                numvalidspatiallocs,
                fmri_data[validlocs, :],
                threshval,
                cardiacnoise[validlocs, :],
                meanvals[validlocs],
                rvals[validlocs],
                r2vals[validlocs],
                fitcoffs[validlocs],
                fitNorm[validlocs],
                datatoremove[validlocs, :],
                filtereddata[validlocs, :],
                procbyvoxel=True,
                nprocs=nprocs,
            )
            datatoremove[validlocs, :] = np.multiply(cardiacnoise[validlocs, :], fitcoffs[:, None])
            filtereddata[validlocs, :] = fmri_data[validlocs, :] - datatoremove
            timings.append(
                [
                    "Cardiac signal temporal regression finished",
                    time.time(),
                    numspatiallocs,
                    "voxels",
                ]
            )
            theheader = copy.deepcopy(nim_hdr)
            theheader["dim"][4] = 1
            tide_io.savetonifti(
                fitcoffs.reshape((xsize, ysize, numslices)), theheader, outputroot + "_fitamp",
            )
            tide_io.savetonifti(
                meanvals.reshape((xsize, ysize, numslices)), theheader, outputroot + "_fitamp",
            )
            tide_io.savetonifti(
                rvals.reshape((xsize, ysize, numslices)), theheader, outputroot + "_fitR",
            )

            theheader = copy.deepcopy(nim_hdr)
            tide_io.savetonifti(
                filtereddata.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                outputroot + "_temporalfiltereddata",
            )
            tide_io.savetonifti(
                datatoremove.reshape((xsize, ysize, numslices, timepoints)),
                theheader,
                outputroot + "_temporaldatatoremove",
            )
            timings.append(
                ["Cardiac signal temporal regression files written", time.time(), None, None,]
            )

    timings.append(["Done", time.time(), None, None])

    # Process and save timing information
    nodeline = "Processed on " + platform.node()
    tide_util.proctiminginfo(
        timings, outputfile=outputroot + "_runtimings.txt", extraheader=nodeline
    )

    tide_util.logmem("final", file=memfile)


if __name__ == "__main__":

    # grab the command line arguments then pass them off.
    nargs = len(sys.argv)
    if nargs < 4:
        usage()
        exit()

    happy_main(sys.argv)
