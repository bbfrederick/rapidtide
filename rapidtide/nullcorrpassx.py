#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
# $Author: frederic $
# $Date: 2016/07/11 14:50:43 $
# $Id: rapidtide,v 1.161 2016/07/11 14:50:43 frederic Exp $
#
#
#

from __future__ import print_function, division

import numpy as np
import rapidtide.multiproc as tide_multiproc
import rapidtide.util as tide_util
import rapidtide.miscmath as tide_math
import rapidtide.filter as tide_filt

import rapidtide.corrpassx as tide_corrpass
import rapidtide.corrfitx as tide_corrfit
import sys


# note: rawtimecourse has been filtered, but NOT windowed
def _procOneNullCorrelationx(normalizedreftc,
                             rawtcfft_r, rawtcfft_ang,
                             Fs,
                             thecorrelator,
                             thefitter,
                             despeckle_thresh=5.0,
                             fixdelay=False,
                             fixeddelayvalue=0.0,
                             permutationmethod='shuffle',
                             disablethresholds=False,
                             rt_floatset=np.float64,
                             rt_floattype='float64'):

    # make a shuffled copy of the regressors
    if permutationmethod == 'shuffle':
        permutedtc = np.random.permutation(normalizedreftc)
    elif permutationmethod == 'phaserandom':
        #permutedtc = tide_filt.ifftfrompolar(rawtcfft_r, np.random.uniform(low=-np.pi, high=np.pi, size=len(rawtcfft_r)))
        permutedtc = tide_filt.ifftfrompolar(rawtcfft_r, np.random.permutation(rawtcfft_ang))
    else:
        print('illegal shuffling method')
        sys.exit()

    # apply the appropriate filter
    #permutedtc = thecorrelator.ncprefilter.apply(Fs, permutedtc)

    # crosscorrelate with original
    thexcorr_y, thexcorr_x, dummy = thecorrelator.run(permutedtc)

    # fit the correlation
    thefitter.setcorrtimeaxis(thexcorr_x)
    maxindex, maxlag, maxval, maxsigma, maskval, peakstart, peakend, failreason = thefitter.fit(thexcorr_y)

    #    '''tide_corrfit.onecorrfitx(thexcorr_y,
    #            thefitter,
    #            disablethresholds=disablethresholds,
    #            despeckle_thresh=despeckle_thresh,
    #           fixdelay=fixdelay,
    #            fixeddelayvalue=fixeddelayvalue,
    #            rt_floatset=rt_floatset,
    #            rt_floattype=rt_floattype)'''

    return maxval


def getNullDistributionDatax(rawtimecourse,
                             Fs,
                             thecorrelator,
                             thefitter,
                             despeckle_thresh=5.0,
                             fixdelay=False,
                             fixeddelayvalue=0.0,
                             numestreps=0,
                             nprocs=1,
                             showprogressbar=True,
                             chunksize=1000,
                             permutationmethod='shuffle',
                             rt_floatset=np.float64,
                             rt_floattype='float64'):
    r"""Calculate a set of null correlations to determine the distribution of correlation values.  This can
    be used to find the spurious correlation threshold

    Parameters
    ----------
    rawtimecourse : 1D numpy array
        The test regressor.  This should be filtered to the desired bandwidth, but NOT windowed.
        :param rawtimecourse:

    corrscale: 1D numpy array
        The time axis of the cross correlation function.

    filterfunc: function
        This is a preconfigured noncausalfilter function which is used to filter data to the desired bandwidth

    Fs: float
        The sample frequency of rawtimecourse, in Hz

    corrorigin: int
        The bin number in the correlation timescale corresponding to 0.0 seconds delay

    negbins: int
        The lower edge of the search range for correlation peaks, in number of bins below corrorigin

    posbins: int
        The upper edge of the search range for correlation peaks, in number of bins above corrorigin

    """

    inputshape = np.asarray([numestreps])
    normalizedreftc = thecorrelator.ncprefilter.apply(Fs, tide_math.corrnormalize(thecorrelator.reftc,
                                                                                  prewindow=False,
                                                                                  detrendorder=thecorrelator.detrendorder)
                                                      )
    rawtcfft_r, rawtcfft_ang = tide_filt.polarfft(normalizedreftc)
    if nprocs > 1:
        # define the consumer function here so it inherits most of the arguments
        def nullCorrelation_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(_procOneNullCorrelationx(normalizedreftc,
                                                      rawtcfft_r, rawtcfft_ang,
                                                      Fs,
                                                      thecorrelator,
                                                      thefitter,
                                                      despeckle_thresh=despeckle_thresh,
                                                      fixdelay=fixdelay,
                                                      fixeddelayvalue=fixeddelayvalue,
                                                      permutationmethod=permutationmethod,
                                                      rt_floatset=rt_floatset,
                                                      rt_floattype=rt_floattype))

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(nullCorrelation_consumer,
                                                inputshape, None,
                                                nprocs=nprocs,
                                                showprogressbar=showprogressbar,
                                                chunksize=chunksize)

        # unpack the data
        corrlist = np.asarray(data_out, dtype=rt_floattype)
    else:
        corrlist = np.zeros((numestreps), dtype=rt_floattype)

        for i in range(0, numestreps):
            # make a shuffled copy of the regressors
            if permutationmethod == 'shuffle':
                permutedtc = np.random.permutation(normalizedreftc)
            elif permutationmethod == 'phaserandom':
                # permutedtc = tide_filt.ifftfrompolar(rawtcfft_r, np.random.uniform(low=-np.pi, high=np.pi, size=len(rawtcfft_r)))
                permutedtc = tide_filt.ifftfrompolar(rawtcfft_r, np.random.permutation(rawtcfft_ang))
            else:
                print('illegal shuffling method')
                sys.exit()

            # crosscorrelate with original, fit, and return the maximum value, and add it to the list
            thexcorr = _procOneNullCorrelationx(normalizedreftc,
                                                rawtcfft_r, rawtcfft_ang,
                                                Fs,
                                                thecorrelator,
                                                thefitter,
                                                despeckle_thresh=despeckle_thresh,
                                                fixdelay=fixdelay,
                                                fixeddelayvalue=fixeddelayvalue,
                                                permutationmethod=permutationmethod,
                                                rt_floatset=rt_floatset,
                                                rt_floattype=rt_floattype)
            corrlist[i] = thexcorr

            # progress
            if showprogressbar:
                tide_util.progressbar(i + 1, numestreps, label='Percent complete')

        # jump to line after progress bar
        print()

    # return the distribution data
    numnonzero = len(np.where(corrlist != 0.0)[0])
    print(numnonzero, 'non-zero correlations out of', len(corrlist), '(', 100.0 * numnonzero / len(corrlist), '%)')
    return corrlist
