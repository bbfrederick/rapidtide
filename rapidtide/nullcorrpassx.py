#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016 Blaise Frederick
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

import rapidtide.corrpass as tide_corrpass
import rapidtide.corrfitx as tide_corrfit


# note: sourcetimecourse has been filtered, but NOT windowed
def _procOneNullCorrelationx(iteration,
                             sourcetimecourse,
                             filterfunc,
                             Fs,
                             corrscale,
                             corrorigin,
                             negbins,
                             posbins,
                             optiondict,
                             rt_floatset=np.float64,
                             rt_floattype='float64'):
    # make a shuffled copy of the regressors
    if optiondict['permutationmethod'] == 'shuffle':
        permutedtc = np.random.permutation(sourcetimecourse)
    else:
        permutedtc = np.random.permutation(sourcetimecourse)

    # apply the appropriate filter
    permutedtc = filterfunc.apply(Fs, permutedtc)

    normalizedsourcetc = tide_math.corrnormalize(sourcetimecourse,
                                              prewindow=optiondict['usewindowfunc'],
                                              detrendorder=optiondict['detrendorder'],
                                              windowfunc=optiondict['windowfunc'])

    # crosscorrelate with original
    thexcorr, dummy = tide_corrpass.onecorrelation(permutedtc, Fs, corrorigin, negbins, posbins,
                                                   filterfunc,
                                                   normalizedsourcetc,
                                                   usewindowfunc=optiondict['usewindowfunc'],
                                                   detrendorder=optiondict['detrendorder'],
                                                   windowfunc=optiondict['windowfunc'],
                                                   corrweighting=optiondict['corrweighting'])

    # fit the correlation
    thefitter.setcorrtimeaxis(corrscale[corrorigin - negbins:corrorigin + posbins])
    maxindex, maxlag, maxval, maxsigma, maskval, peakstart, peakend, failreason = \
        tide_corrfit.onecorrfitx(thexcorr,
                                 thefitter,
                                 disablethresholds=True,
                                 despeckle_thresh=optiondict['despeckle_thresh'],
                                 fixdelay=optiondict['fixdelay'],
                                 fixeddelayvalue=optiondict['fixeddelayvalue'],
                                 rt_floatset=rt_floatset,
                                 rt_floattype=rt_floattype)

    return maxval


def getNullDistributionDatax(sourcetimecourse,
                             corrscale,
                             filterfunc,
                             Fs,
                             corrorigin,
                             negbins,
                             posbins,
                             thefitter,
                             rt_floatset=np.float64,
                             rt_floattype='float64'):
    r"""Calculate a set of null correlations to determine the distribution of correlation values.  This can
    be used to find the spurious correlation threshold

    Parameters
    ----------
    sourcetimecourse : 1D numpy array
        The test regressor.  This should be filtered to the desired bandwidth, but NOT windowed.
        :param sourcetimecourse:

    corrscale: 1D numpy array
        The time axis of the cross correlation function.

    filterfunc: function
        This is a preconfigured noncausalfilter function which is used to filter data to the desired bandwidth

    Fs: float
        The sample frequency of sourcetimecourse, in Hz

    corrorigin: int
        The bin number in the correlation timescale corresponding to 0.0 seconds delay

    negbins: int
        The lower edge of the search range for correlation peaks, in number of bins below corrorigin

    posbins: int
        The upper edge of the search range for correlation peaks, in number of bins above corrorigin

    optiondict: dict
        The rapidtide option dictionary containing a number of additional parameters

    """

    inputshape = np.asarray([optiondict['numestreps']])
    if optiondict['nprocs'] > 1:
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
                    outQ.put(_procOneNullCorrelationx(val, sourcetimecourse, filterfunc, Fs, corrscale, corrorigin,
                                                     negbins, posbins, thefitter,
                                                     rt_floatset=rt_floatset,
                                                     rt_floattype=rt_floattype))

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(nullCorrelation_consumer,
                                                inputshape, None,
                                                nprocs=optiondict['nprocs'],
                                                showprogressbar=True,
                                                chunksize=optiondict['mp_chunksize'])

        # unpack the data
        corrlist = np.asarray(data_out, dtype=rt_floattype)
    else:
        corrlist = np.zeros((optiondict['numestreps']), dtype=rt_floattype)

        for i in range(0, optiondict['numestreps']):
            # make a shuffled copy of the regressors
            permutedtc = np.random.permutation(sourcetimecourse)

            # crosscorrelate with original, fit, and return the maximum value, and add it to the list
            thexcorr = _procOneNullCorrelationx(i, sourcetimecourse, filterfunc, Fs, corrscale, corrorigin,
                                               negbins, posbins, thefitter,
                                               rt_floatset=rt_floatset,
                                               rt_floattype=rt_floattype)
            corrlist[i] = thexcorr

            # progress
            if optiondict['showprogressbar']:
                tide_util.progressbar(i + 1, optiondict['numestreps'], label='Percent complete')

        # jump to line after progress bar
        print()

    # return the distribution data
    numnonzero = len(np.where(corrlist != 0.0)[0])
    print(numnonzero, 'non-zero correlations out of', len(corrlist), '(', 100.0 * numnonzero / len(corrlist), '%)')
    return corrlist
