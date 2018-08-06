#!/usr/bin/env python
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

import rapidtide.corrpass as tide_corrpass
import rapidtide.corrfit as tide_corrfit


def _procOneNullCorrelationx(iteration, indata, ncprefilter, oversampfreq, corrscale, corrorigin, lagmininpts,
                            lagmaxinpts, optiondict,
                            rt_floatset=np.float64,
                            rt_floattype='float64'):
    # make a shuffled copy of the regressors
    shuffleddata = np.random.permutation(indata)

    # crosscorrelate with original
    thexcorr, dummy = tide_corrpass.onecorrelation(shuffleddata, oversampfreq, corrorigin, lagmininpts, lagmaxinpts,
                                                   ncprefilter,
                                                   indata,
                                                   optiondict)

    # fit the correlation
    maxindex, maxlag, maxval, maxsigma, maskval, peakstart, peakend, failreason = \
        tide_corrfit.onecorrfitx(thexcorr, corrscale[corrorigin - lagmininpts:corrorigin + lagmaxinpts],
                                optiondict, disablethresholds=True,
                                rt_floatset=rt_floatset,
                                rt_floattype=rt_floattype
                                )

    return maxval


def getNullDistributionDatax(indata,
                             corrscale,
                             ncprefilter,
                             oversampfreq,
                             corrorigin,
                             lagmininpts,
                             lagmaxinpts,
                             optiondict,
                             rt_floatset=np.float64,
                             rt_floattype='float64'):
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
                    outQ.put(_procOneNullCorrelationx(val, indata, ncprefilter, oversampfreq, corrscale, corrorigin,
                                                     lagmininpts, lagmaxinpts, optiondict,
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
            shuffleddata = np.random.permutation(indata)

            # crosscorrelate with original, fit, and return the maximum value, and add it to the list
            thexcorr = _procOneNullCorrelationx(i, indata, ncprefilter, oversampfreq, corrscale, corrorigin,
                                               lagmininpts, lagmaxinpts, optiondict,
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

# old style null correlation calculation below this point


def _procOneNullCorrelation(iteration, indata, ncprefilter, oversampfreq, corrscale, corrorigin, lagmininpts,
                           lagmaxinpts, optiondict,
                           rt_floatset=np.float64,
                           rt_floattype='float64'
                           ):
    # make a shuffled copy of the regressors
    shuffleddata = np.random.permutation(indata)

    # crosscorrelate with original
    thexcorr, dummy = tide_corrpass.onecorrelation(shuffleddata, oversampfreq, corrorigin, lagmininpts, lagmaxinpts,
                                                   ncprefilter,
                                                   indata,
                                                   optiondict)

    # fit the correlation
    maxindex, maxlag, maxval, maxsigma, maskval, failreason = \
        tide_corrfit.onecorrfit(thexcorr, corrscale[corrorigin - lagmininpts:corrorigin + lagmaxinpts],
                   optiondict,
                   rt_floatset=rt_floatset,
                   rt_floattype=rt_floattype)

    return maxval


def getNullDistributionData(indata,
                            corrscale,
                            ncprefilter,
                            oversampfreq,
                            corrorigin,
                            lagmininpts,
                            lagmaxinpts,
                            optiondict,
                            rt_floatset=np.float64,
                            rt_floattype='float64'
                            ):
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
                    outQ.put(_procOneNullCorrelation(val,
                                                     indata,
                                                     ncprefilter,
                                                     oversampfreq,
                                                     corrscale,
                                                     corrorigin,
                                                     lagmininpts,
                                                     lagmaxinpts,
                                                     optiondict,
                                                     rt_floatset=rt_floatset,
                                                     rt_floattype=rt_floattype
                                                     ))

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(nullCorrelation_consumer,
                                                inputshape, None,
                                                nprocs=optiondict['nprocs'],
                                                showprogressbar=True,
                                                chunksize=optiondict['mp_chunksize'])

        # unpack the data
        volumetotal = 0
        corrlist = np.asarray(data_out, dtype=rt_floattype)
    else:
        corrlist = np.zeros((optiondict['numestreps']), dtype=rt_floattype)

        for i in range(0, optiondict['numestreps']):
            # make a shuffled copy of the regressors
            shuffleddata = np.random.permutation(indata)

            # crosscorrelate with original
            thexcorr, dummy = tide_corrpass.onecorrelation(shuffleddata,
                                                           oversampfreq,
                                                           corrorigin,
                                                           lagmininpts,
                                                           lagmaxinpts,
                                                           ncprefilter,
                                                           indata,
                                                           optiondict
                                                           )

            # fit the correlation
            maxindex, maxlag, maxval, maxsigma, maskval, failreason = \
                tide_corrfit.onecorrfit(thexcorr,
                           corrscale[corrorigin - lagmininpts:corrorigin + lagmaxinpts],
                           optiondict,
                           rt_floatset=rt_floatset,
                           rt_floattype=rt_floattype
                           )

            # find and tabulate correlation coefficient at optimal lag
            corrlist[i] = maxval

            # progress
            if optiondict['showprogressbar']:
                tide_util.progressbar(i + 1, optiondict['numestreps'], label='Percent complete')

        # jump to line after progress bar
        print()

    # return the distribution data
    numnonzero = len(np.where(corrlist != 0.0)[0])
    print(numnonzero, 'non-zero correlations out of', len(corrlist), '(', 100.0 * numnonzero / len(corrlist), '%)')
    return corrlist

