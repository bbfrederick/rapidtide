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

import rapidtide.corrpass as tide_corrpass
import rapidtide.corrfit as tide_corrfit


def _procOneNullCorrelation(iteration, sourcetimecourse, filterfunc, Fs, corrscale, corrorigin, negbins,
                           posbins, optiondict,
                           rt_floatset=np.float64,
                           rt_floattype='float64'
                           ):
    # make a shuffled copy of the regressors
    permutedtc = np.random.permutation(sourcetimecourse)

    # crosscorrelate with original
    thexcorr, dummy = tide_corrpass.onecorrelation(permutedtc,
                                                   Fs,
                                                   corrorigin,
                                                   negbins,
                                                   posbins,
                                                   filterfunc,
                                                   sourcetimecourse,
                                                   usewindowfunc=optiondict['usewindowfunc'],
                                                   detrendorder=optiondict['detrendorder'],
                                                   windowfunc=optiondict['windowfunc'],
                                                   corrweighting=optiondict['corrweighting'])

    # fit the correlation
    maxindex, maxlag, maxval, maxsigma, maskval, failreason = \
        tide_corrfit.onecorrfit(thexcorr, corrscale[corrorigin - negbins:corrorigin + posbins],
                   optiondict,
                   zerooutbadfit=True,
                   rt_floatset=rt_floatset,
                   rt_floattype=rt_floattype)

    return maxval


def getNullDistributionData(sourcetimecourse,
                            corrscale,
                            filterfunc,
                            Fs,
                            corrorigin,
                            negbins,
                            posbins,
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
                                                     sourcetimecourse,
                                                     filterfunc,
                                                     Fs,
                                                     corrscale,
                                                     corrorigin,
                                                     negbins,
                                                     posbins,
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
            permutedtc = np.random.permutation(sourcetimecourse)

            # crosscorrelate with original
            thexcorr, dummy = tide_corrpass.onecorrelation(permutedtc,
                                                           Fs,
                                                           corrorigin,
                                                           negbins,
                                                           posbins,
                                                           filterfunc,
                                                           sourcetimecourse,
                                                           usewindowfunc=optiondict['usewindowfunc'],
                                                           detrendorder=optiondict['detrendorder'],
                                                           windowfunc=optiondict['windowfunc'],
                                                           corrweighting=optiondict['corrweighting'])

            # fit the correlation
            maxindex, maxlag, maxval, maxsigma, maskval, failreason = \
                tide_corrfit.onecorrfit(thexcorr,
                           corrscale[corrorigin - negbins:corrorigin + posbins],
                           optiondict,
                           zerooutbadfit=True,
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

