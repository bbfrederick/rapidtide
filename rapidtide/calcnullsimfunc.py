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
import sys


import numpy as np


import rapidtide.filter as tide_filt
import rapidtide.miscmath as tide_math
import rapidtide.genericmultiproc as tide_genericmultiproc


# note: rawtimecourse has been filtered, but NOT windowed
def _procOneNullCorrelationx(
    vox,
    voxelargs,
    **kwargs,
):

    options = {
        "permutationmethod": "shuffle",
        "debug": False,
    }
    options.update(kwargs)
    permutationmethod = options["permutationmethod"]
    debug = options["debug"]
    if debug:
        print(f"{permutationmethod=}")
    (
        normalizedreftc,
        rawtcfft_r,
        rawtcfft_ang,
        theCorrelator,
        thefitter,
    ) = voxelargs

    # make a shuffled copy of the regressors
    if permutationmethod == "shuffle":
        permutedtc = np.random.permutation(normalizedreftc)
        # apply the appropriate filter
        # permutedtc = theCorrelator.ncprefilter.apply(Fs, permutedtc)
    elif permutationmethod == "phaserandom":
        permutedtc = tide_filt.ifftfrompolar(rawtcfft_r, np.random.permutation(rawtcfft_ang))
    else:
        print("illegal shuffling method")
        sys.exit()

    # crosscorrelate with original
    thexcorr_y, thexcorr_x, dummy = theCorrelator.run(permutedtc)

    # fit the correlation
    thefitter.setcorrtimeaxis(thexcorr_x)
    (
        maxindex,
        maxlag,
        maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    ) = thefitter.fit(thexcorr_y)

    return vox, maxval


def _packvoxeldata(voxnum, voxelargs):
    return [voxelargs[0], voxelargs[1], voxelargs[2], voxelargs[3], voxelargs[4]]


def _unpackvoxeldata(retvals, voxelproducts):
    (voxelproducts[0])[retvals[0]] = retvals[1]


def getNullDistributionData(
    Fs,
    theCorrelator,
    thefitter,
    LGR,
    numestreps=0,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    permutationmethod="shuffle",
    rt_floatset=np.float64,
    rt_floattype="float64",
    debug=False,
):
    r"""Calculate a set of null correlations to determine the distribution of correlation values.  This can
    be used to find the spurious correlation threshold

    Parameters
    ----------
    Fs: float
        The sample frequency of rawtimecourse, in Hz

    rawtimecourse : 1D numpy array
        The test regressor.  This should be filtered to the desired bandwidth, but NOT windowed.
        :param rawtimecourse:

    corrscale: 1D numpy array
        The time axis of the cross correlation function.

    filterfunc: function
        This is a preconfigured NoncausalFilter function which is used to filter data to the desired bandwidth

    corrorigin: int
        The bin number in the correlation timescale corresponding to 0.0 seconds delay

    negbins: int
        The lower edge of the search range for correlation peaks, in number of bins below corrorigin

    posbins: int
        The upper edge of the search range for correlation peaks, in number of bins above corrorigin

    """
    inputshape = np.asarray([numestreps])
    normalizedreftc = theCorrelator.ncprefilter.apply(
        Fs,
        tide_math.corrnormalize(
            theCorrelator.reftc,
            windowfunc="None",
            detrendorder=theCorrelator.detrendorder,
        ),
    )
    rawtcfft_r, rawtcfft_ang = tide_filt.polarfft(normalizedreftc)
    corrlist = np.zeros((numestreps), dtype=rt_floattype)
    voxelmask = np.ones((numestreps), dtype=rt_floattype)
    voxelargs = [normalizedreftc, rawtcfft_r, rawtcfft_ang, theCorrelator, thefitter]
    voxelfunc = _procOneNullCorrelationx
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [
        corrlist,
    ]

    volumetotal = tide_genericmultiproc.run_multiproc(
        voxelfunc,
        packfunc,
        unpackfunc,
        voxelargs,
        voxeltargets,
        inputshape,
        voxelmask,
        LGR,
        nprocs,
        alwaysmultiproc,
        showprogressbar,
        chunksize,
        permutationmethod=permutationmethod,
        debug=debug,
    )

    # return the distribution data
    numnonzero = len(np.where(corrlist != 0.0)[0])
    print(
        "{:d} non-zero correlations out of {:d} ({:.2f}%)".format(
            numnonzero, len(corrlist), 100.0 * numnonzero / len(corrlist)
        )
    )
    return corrlist
