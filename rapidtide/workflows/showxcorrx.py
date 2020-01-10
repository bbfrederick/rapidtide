#!/usr/bin/env python
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
#       $Date: 2016/07/11 14:50:43 $
#       $Id: showxcorr,v 1.41 2016/07/11 14:50:43 frederic Exp $
#
from __future__ import print_function, division

import argparse
import scipy as sp
from scipy import pi
from scipy.stats.stats import pearsonr
import numpy as np
from numpy import r_, argmax, zeros
from numpy.random import permutation
import matplotlib.pyplot as plt

import rapidtide.miscmath as tide_math
import rapidtide.stats as tide_stats
import rapidtide.io as tide_io
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.correlate as tide_corr
from rapidtide.workflows.parser_funcs import is_valid_file, is_float


def _get_null_distribution(indata, xcorr_x, thefilter, prewindow, detrendorder,
                           searchstart, searchend, Fs, dofftcorr,
                           windowfunc='hamming', corrweighting='none',
                           numreps=1000):
    """
    Get an empirical null distribution from the data.
    """
    print('estimating significance distribution using {0} '
          'repetitions'.format(numreps))
    corrlist = zeros(numreps, dtype='float')
    corrlist_pear = zeros(numreps, dtype='float')
    xcorr_x_trim = xcorr_x[searchstart:searchend + 1]

    filteredindata = tide_math.corrnormalize(thefilter.apply(Fs, indata),
                                             prewindow=prewindow,
                                             detrendorder=detrendorder,
                                             windowfunc=windowfunc)
    for i in range(numreps):
        # make a shuffled copy of the regressors
        shuffleddata = permutation(indata)

        # filter it
        filteredshuffleddata = np.nan_to_num(
            tide_math.corrnormalize(thefilter.apply(Fs, shuffleddata),
                                    prewindow=prewindow,
                                    detrendorder=detrendorder,
                                    windowfunc=windowfunc))

        # crosscorrelate with original
        theshuffledxcorr = tide_corr.fastcorrelate(filteredindata,
                                                   filteredshuffleddata,
                                                   usefft=dofftcorr,
                                                   weighting=corrweighting)

        # find and tabulate correlation coefficient at optimal lag
        theshuffledxcorr_trim = theshuffledxcorr[searchstart:searchend + 1]
        maxdelay = xcorr_x_trim[argmax(theshuffledxcorr_trim)]
        corrlist[i] = theshuffledxcorr_trim[argmax(theshuffledxcorr_trim)]

        # find and tabulate correlation coefficient at 0 lag
        corrlist_pear[i] = pearsonr(filteredindata, filteredshuffleddata)[0]

    # return the distribution data
    return corrlist, corrlist_pear


def _get_parser():
    """
    Argument parser for showxcorrx
    """
    parser = argparse.ArgumentParser(description='Calculate and display '
                                                 'crosscorrelation between two'
                                                 ' timeseries.')
    # Required arguments
    parser.add_argument('infilename1',
                        type=lambda x: is_valid_file(parser, x),
                        help='Text file containing a timeseries')
    parser.add_argument('infilename2',
                        type=lambda x: is_valid_file(parser, x),
                        help='Text file containing a timeseries')
    parser.add_argument('Fs',
                        type=lambda x: is_float(parser, x),
                        help='The sample rate of the timecourses, in Hz')

    # Optional arguments
    parser.add_argument('-l',
                        dest='thelabel',
                        action='store',
                        metavar='LABEL',
                        type=str,
                        help=('Label for the delay value'),
                        default=None)
    parser.add_argument('-s',
                        dest='starttime',
                        action='store',
                        metavar='STARTTIME',
                        type=lambda x: is_float(parser, x),
                        help=('Time of first datapoint to use in seconds in '
                              'the first file'),
                        default=0.)
    parser.add_argument('-D',
                        dest='duration',
                        action='store',
                        metavar='DURATION',
                        type=lambda x: is_float(parser, x),
                        help=('Amount of data to use in seconds'),
                        default=1000000.)
    parser.add_argument('-r',
                        dest='searchrange',
                        action='store',
                        metavar='RANGE',
                        type=lambda x: is_float(parser, x),
                        help=('Restrict peak search range to +/- RANGE '
                              'seconds (default is +/-15)'),
                        default=15.)
    parser.add_argument('-d',
                        dest='display',
                        action='store_false',
                        help=('Turns off display of graph'),
                        default=True)
    parser.add_argument('-T',
                        dest='trimdata',
                        action='store_true',
                        help=('Trim data to match'),
                        default=False)
    parser.add_argument('-A',
                        dest='summarymode',
                        action='store_true',
                        help=('Print data on a single summary line'),
                        default=False)
    parser.add_argument('-a',
                        dest='labelline',
                        action='store_true',
                        help=('If summary mode is on, add a header line '
                              'showing what values mean'),
                        default=False)
    parser.add_argument('-f',
                        dest='flipregressor',
                        action='store_true',
                        help=('Negate (flip) second regressor'),
                        default=False)
    parser.add_argument('--windowfunc',
                        dest='windowfunc',
                        action='store',
                        choices=['hamming', 'blackmanharris', 'hann', 'none'],
                        help=('Window function to apply before correlation '
                              '(default is hamming)"'),
                        default='hamming')
    parser.add_argument('--cepstral',
                        dest='calccepstraldelay',
                        action='store_true',
                        help=("Check time delay using Choudhary's cepstral "
                              "technique"),
                        default=False)
    parser.add_argument('--savecorr',
                        dest='corroutputfile',
                        action='store',
                        metavar='FILE',
                        type=str,
                        help=('Save the correlation function to the file FILE '
                              'in xy format'),
                        default=False)
    parser.add_argument('-z',
                        dest='controlvariablefile',
                        action='store',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x),
                        help=('Use the columns of FILE as controlling '
                              'variables and return the partial correlation'),
                        default=None)
    parser.add_argument('-N',
                        dest='numreps',
                        action='store',
                        metavar='TRIALS',
                        type=int,
                        help=('Estimate significance thresholds by Monte '
                              'Carlo with TRIALS repetition'),
                        default=0)

    filttype = parser.add_argument_group('Filter types')
    ft_mutex = filttype.add_mutually_exclusive_group()
    ft_mutex.add_argument('-F', '--arb',
                          dest='arbvec',
                          action='store',
                          nargs='+',
                          type=lambda x: is_float(parser, x),
                          metavar=('LOWERFREQ UPPERFREQ',
                                   'LOWERSTOP UPPERSTOP'),
                          help=('Filter data and regressors from LOWERFREQ to '
                                'UPPERFREQ. LOWERSTOP and UPPERSTOP can also '
                                'be specified, or will be calculated '
                                'automatically'),
                          default=None)
    ft_mutex.add_argument('--filtertype',
                          dest='filtertype',
                          action='store',
                          type=str,
                          choices=['arb', 'vlf', 'lfo', 'resp', 'cardiac'],
                          help=('Filter data and regressors to specific band'),
                          default='arb')
    ft_mutex.add_argument('-V', '--vlf',
                          dest='filtertype',
                          action='store_const',
                          const='vlf',
                          help=('Filter data and regressors to VLF band'),
                          default='arb')
    ft_mutex.add_argument('-L', '--lfo',
                          dest='filtertype',
                          action='store_const',
                          const='lfo',
                          help=('Filter data and regressors to LFO band'),
                          default='arb')
    ft_mutex.add_argument('-R', '--resp',
                          dest='filtertype',
                          action='store_const',
                          const='resp',
                          help=('Filter data and regressors to respiratory '
                                'band'),
                          default='arb')
    ft_mutex.add_argument('-C', '--cardiac',
                          dest='filtertype',
                          action='store_const',
                          const='cardiac',
                          help=('Filter data and regressors to cardiac band'),
                          default='arb')

    cc_group = parser.add_argument_group('Correlation weighting options')
    cc_mutex = cc_group.add_mutually_exclusive_group()
    cc_mutex.add_argument('--corrweighting',
                          dest='corrweighting',
                          action='store',
                          type=str,
                          choices=['none', 'phat', 'liang', 'eckart'],
                          help=('Method to use for cross-correlation '
                                'weighting.'),
                          default='none')
    cc_mutex.add_argument('--detrendorder',
                          dest='detrendorder',
                          action='store',
                          type=int,
                          help='Disable linear trend removal',
                          default=1)
    cc_mutex.add_argument('--nowindow',
                          dest='prewindow',
                          action='store_false',
                          help='Do not prewindow data before correlation',
                          default=True)

    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help=('Print things'),
                        default=False)
    return parser


def showxcorrx_workflow(infilename1, infilename2, Fs,
                        thelabel='', starttime=0., duration=1000000.,
                        searchrange=15.,
                        display=True, trimdata=False,
                        summarymode=False, labelline=False,
                        flipregressor=False, windowfunc='hamming',
                        calccepstraldelay=False, corroutputfile=False,
                        controlvariablefile=None, numreps=0,
                        arbvec=None, filtertype='arb', corrweighting='none',
                        detrendorder=1, prewindow=True, verbose=False):
    r"""Calculate and display crosscorrelation between two timeseries.

    Parameters
    ----------
    infilename1 : str
        The name of a text file containing a timeseries, one timepoint per line.
    infilename2 : str
        The name of a text file containing a timeseries, one timepoint per line.
    Fs : float
        The sample rate of the time series, in Hz.
    thelabel : str, optional
        The label for the output graph.  Default is blank.
    starttime : float, optional
        Time offset into the timeseries, in seconds, to start using the time data.  Default is 0
    duration : float, optional
        Length of time from each time series, in seconds, to use for the cross-correlation.  Default is the entire time series.
    searchrange : float, optional
        Only search for cross-correlation peaks between -searchrange and +searchrange seconds (default is 15).
    display : bool, optional
        Plot cross-correlation function in a matplotlib window.  Default is True.
    trimdata : bool, optional
        Trim time series to the length of the shorter series.  Default is False.
    summarymode : bool, optional
        Output a table of interesting results for later processing.  Default is False.
    labelline : bool, optional
        Print an explanatory header line over the summary information.  Default is False.
    flipregressor : bool, optional
        Invert timeseries 2 prior to cross-correlation.
    windowfunc : {'hamming', 'hann', 'blackmanharris'}
        Window function to apply prior to cross-correlation.  Default is 'hamming'.
    calccepstraldelay : bool, optional
        Use cepstral estimation of delay.  Default is False.
    corroutputfile : bool, optional
        Save the correlation function to a file.  Default is False.
    controlvariablefile : bool, optional
        Save internal variables to a text file.  Default is False.
    numreps : int, optional
        Number of null correlations to perform to estimate significance.  Default is 10000
    arbvec : [float,float,float,float], optional
        Frequency limits of the arb_pass filter.
    filtertype : 'none', 'card', 'lfo', 'vlf', 'resp', 'arb'
        Type of filter to apply data prior to correlation.  Default is 'none'
    corrweighting : {'none', 'Liang', 'Eckart', 'PHAT'}, optional
         Weighting function to apply to the crosscorrelation in the Fourier domain.  Default is 'none'
    detrendorder : int, optional
       Order of polynomial used to detrend crosscorrelation inputs.  Default is 1 (0 disables)
    prewindow : bool, optional
        Apply window function prior to cross-correlation.  Default is True.
    verbose : bool, optional
        Print internal status information.  Default is False.

    Notes
    -----
    This workflow writes out several files:

    If corroutputfile is defined:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    corrlist.txt              A file
    corrlist_pear.txt         A file
    [corroutputfile]          Correlation function
    ======================    =================================================

    If debug is True:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    filtereddata1.txt         Something
    filtereddata2.txt         Something
    ======================    =================================================
    """
    # Constants that could be arguments
    dofftcorr = True
    writecorrlists = False
    debug = False
    showpearson = True

    # These are unnecessary and should be simplified
    dopartial = bool(controlvariablefile)
    uselabel = bool(thelabel)
    dumpfiltered = bool(debug)

    if labelline:
        # TS: should prob reflect this in the parser, but it's not a big deal
        summarymode = True

    if numreps == 0:
        estimate_significance = False
    else:
        estimate_significance = True

    savecorrelation = bool(corroutputfile)

    theprefilter = tide_filt.noncausalfilter()

    if arbvec is not None and filtertype != 'arb':
        raise ValueError('Argument arbvec must be None if filtertype is '
                         'not arb')

    if arbvec is not None:
        if len(arbvec) == 2:
            arb_lower = float(arbvec[0])
            arb_upper = float(arbvec[1])
            arb_lowerstop = 0.9 * float(arbvec[0])
            arb_upperstop = 1.1 * float(arbvec[1])
        elif len(arbvec) == 4:
            arb_lower = float(arbvec[0])
            arb_upper = float(arbvec[1])
            arb_lowerstop = float(arbvec[2])
            arb_upperstop = float(arbvec[3])
        theprefilter.settype('arb')
        theprefilter.setfreqs(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)
    else:
        theprefilter.settype(filtertype)

    inputdata1 = tide_io.readvec(infilename1)
    inputdata2 = tide_io.readvec(infilename2)
    numpoints = len(inputdata1)

    startpoint1 = max([int(starttime * Fs), 0])
    if debug:
        print('startpoint set to ', startpoint1)
    endpoint1 = min([startpoint1 + int(duration * Fs), int(len(inputdata1))])
    if debug:
        print('endpoint set to ', endpoint1)
    endpoint2 = min([int(duration * Fs), int(len(inputdata1)),
                     int(len(inputdata2))])
    trimdata1 = inputdata1[startpoint1:endpoint1]
    trimdata2 = inputdata2[0:endpoint2]

    if trimdata:
        minlen = np.min([len(trimdata1), len(trimdata2)])
        trimdata1 = trimdata1[0:minlen]
        trimdata2 = trimdata2[0:minlen]

    # band limit the regressor if that is needed
    if theprefilter.gettype() != 'none':
        if verbose:
            print("filtering to ", theprefilter.gettype(), " band")
    print(windowfunc)
    filtereddata1 = tide_math.corrnormalize(theprefilter.apply(Fs, trimdata1),
                                            prewindow=prewindow,
                                            detrendorder=detrendorder,
                                            windowfunc=windowfunc)
    filtereddata2 = tide_math.corrnormalize(theprefilter.apply(Fs, trimdata2),
                                            prewindow=prewindow,
                                            detrendorder=detrendorder,
                                            windowfunc=windowfunc)
    if flipregressor:
        filtereddata2 *= -1.0

    if dumpfiltered:
        tide_io.writenpvecs(filtereddata1, 'filtereddata1.txt')
        tide_io.writenpvecs(filtereddata2, 'filtereddata2.txt')

    if dopartial:
        controlvars = tide_io.readvecs(controlvariablefile)
        numregressors = len(controlvars)  # Added by TS. Not sure if works.
        regressorvec = []
        for j in range(0, numregressors):
            regressorvec.append(tide_math.corrnormalize(
                theprefilter.apply(Fs, controlvars[j, :]),
                prewindow=prewindow,
                detrendorder=detrendorder,
                windowfunc=windowfunc))

        if (np.max(filtereddata1) - np.min(filtereddata1)) > 0.0:
            thefit, filtereddata1 = tide_fit.mlregress(regressorvec,
                                                       filtereddata1)

        if (np.max(filtereddata2) - np.min(filtereddata2)) > 0.0:
            thefit, filtereddata2 = tide_fit.mlregress(regressorvec,
                                                       filtereddata2)

    thexcorr = tide_corr.fastcorrelate(filtereddata1, filtereddata2,
                                       usefft=dofftcorr,
                                       weighting=corrweighting,
                                       displayplots=debug)

    if calccepstraldelay:
        cepdelay = tide_corr.cepstraldelay(filtereddata1, filtereddata2,
                                           1.0 / Fs, displayplots=display)
        cepcoff = tide_corr.delayedcorr(filtereddata1, filtereddata2, cepdelay,
                                        1.0 / Fs)
        print('cepstral delay time is {0}, correlation is {1}'.format(cepdelay,
                                                                      cepcoff))
    thepxcorr = pearsonr(filtereddata1, filtereddata2)

    # calculate the coherence
    f, Cxy = sp.signal.coherence(
        tide_math.corrnormalize(theprefilter.apply(Fs, trimdata1), prewindow=prewindow,
                                detrendorder=detrendorder, windowfunc=windowfunc),
        tide_math.corrnormalize(theprefilter.apply(Fs, trimdata2), prewindow=prewindow,
                                detrendorder=detrendorder, windowfunc=windowfunc),
        Fs)

    # calculate the cross spectral density
    f, Pxy = sp.signal.csd(
        tide_math.corrnormalize(theprefilter.apply(Fs, trimdata1), prewindow=prewindow,
                                detrendorder=detrendorder, windowfunc=windowfunc),
        tide_math.corrnormalize(theprefilter.apply(Fs, trimdata2), prewindow=prewindow,
                                detrendorder=detrendorder, windowfunc=windowfunc),
        Fs)

    xcorrlen = len(thexcorr)
    sampletime = 1.0 / Fs
    xcorr_x = r_[0:xcorrlen] * sampletime - (xcorrlen * sampletime) / 2.0\
        + sampletime / 2.0
    halfwindow = int(searchrange * Fs)
    corrzero = xcorrlen // 2
    searchstart = corrzero - halfwindow
    searchend = corrzero + halfwindow
    xcorr_x_trim = xcorr_x[searchstart:searchend + 1]
    thexcorr_trim = thexcorr[searchstart:searchend + 1]
    if debug:
        print('searching for peak correlation over range ', searchstart,
              searchend)

    maxdelay = xcorr_x_trim[argmax(thexcorr_trim)]
    if debug:
        print('maxdelay before refinement', maxdelay)

    dofindmaxlag = True
    if dofindmaxlag:
        print('executing findmaxlag')
        (maxindex, maxdelay, maxval, maxsigma, maskval, failreason, peakstart,
         peakend) = tide_fit.findmaxlag_gauss(
             xcorr_x_trim, thexcorr_trim, -searchrange, searchrange, 1000.0,
             refine=True,
             useguess=False,
             fastgauss=False,
             displayplots=False)
        print(maxindex, maxdelay, maxval, maxsigma, maskval, failreason)
        R = maxval
    if debug:
        print('maxdelay after refinement', maxdelay)
        if failreason > 0:
            print('failreason =', failreason)
    else:
        R = thexcorr_trim[argmax(thexcorr_trim)]

    # set the significance threshold
    if estimate_significance:
        # generate a list of correlations from shuffled data
        (corrlist,
         corrlist_pear) = _get_null_distribution(trimdata1, xcorr_x,
                                                 theprefilter, prewindow,
                                                 detrendorder, searchstart,
                                                 searchend, Fs, dofftcorr,
                                                 corrweighting=corrweighting,
                                                 numreps=numreps,
                                                 windowfunc=windowfunc)

        # calculate percentiles for the crosscorrelation from the distribution
        histlen = 100
        thepercentiles = [0.95, 0.99, 0.995]

        (pcts, pcts_fit,
         histfit) = tide_stats.sigFromDistributionData(corrlist, histlen,
                                                       thepercentiles)
        if debug:
            tide_stats.printthresholds(pcts, thepercentiles,
                                       ('Crosscorrelation significance '
                                        'thresholds from data:'))
            tide_stats.printthresholds(pcts_fit, thepercentiles,
                                       ('Crosscorrelation significance '
                                        'thresholds from fit:'))

        # calculate significance for the pearson correlation
        (pearpcts, pearpcts_fit,
         histfit) = tide_stats.sigFromDistributionData(corrlist_pear, histlen,
                                                       thepercentiles)
        if debug:
            tide_stats.printthresholds(pearpcts, thepercentiles,
                                       ('Pearson correlation significance '
                                        'thresholds from data:'))
            tide_stats.printthresholds(pearpcts_fit, thepercentiles,
                                       ('Pearson correlation significance '
                                        'thresholds from fit:'))

        if writecorrlists:
            tide_io.writenpvecs(corrlist, 'corrlist.txt')
            tide_io.writenpvecs(corrlist_pear, 'corrlist_pear.txt')

    def printthresholds(pcts, thepercentiles, labeltext):
        print(labeltext)
        for i in range(0, len(pcts)):
            print('\tp <', "{:.3f}".format(1.0 - thepercentiles[i]), ': ',
                  pcts[i])

    # report the pearson correlation
    if showpearson and verbose:
        print('Pearson_R:\t', thepxcorr[0])
        if estimate_significance:
            for idx, percentile in enumerate(thepercentiles):
                print('    pear_p(', "{:.3f}".format(1.0 - percentile), '):\t',
                      pearpcts[idx])
        print("")

    if debug:
        print(thepxcorr)

    if verbose:
        if uselabel:
            print(thelabel, ":\t", maxdelay)
        else:
            print("Crosscorrelation_Rmax:\t", R)
            print("Crosscorrelation_maxdelay:\t", maxdelay)
            if estimate_significance:
                for idx, percentile in enumerate(thepercentiles):
                    print('    xc_p(', "{:.3f}".format(1.0 - percentile),
                          '):\t', pcts[idx])
            print(infilename1, "[0 seconds] == ", infilename2, "[",
                  -1 * maxdelay, " seconds]")

    if summarymode:
        if estimate_significance:
            if uselabel:
                if labelline:
                    print('thelabel', 'pearson_R', 'pearson_R(p=0.05)',
                          'xcorr_R', 'xcorr_R(P=0.05)', 'xcorr_maxdelay')
                print(thelabel, thepxcorr[0], pearpcts_fit[0], R, pcts_fit[0],
                      -1 * maxdelay)
            else:
                if labelline:
                    print('pearson_R', 'pearson_R(p=0.05)', 'xcorr_R',
                          'xcorr_R(P=0.05)', 'xcorr_maxdelay')
                print(thepxcorr[0], pearpcts_fit[0], R, pcts_fit[0],
                      -1 * maxdelay)
        else:
            if uselabel:
                if labelline:
                    print('thelabel', 'pearson_r', 'pearson_p', 'xcorr_R',
                          'xcorr_maxdelay')
                print(thelabel, thepxcorr[0], thepxcorr[1], R, -1 * maxdelay)
            else:
                if labelline:
                    print('pearson_r\tpearson_p\txcorr_R\txcorr_t\t'
                          'xcorr_maxdelay')
                print(thepxcorr[0], '\t', thepxcorr[1], '\t', R, '\t',
                      -1 * maxdelay)

    if savecorrelation:
        tide_io.writenpvecs(np.stack((xcorr_x, thexcorr), axis=0),
                            corroutputfile)

    if display:
        fig, ax = plt.subplots()
        # ax.set_title('GCC')
        ax.plot(xcorr_x, thexcorr, 'k')
        if debug:
            fig, ax = plt.subplots()
            ax.plot(f, Cxy)
            fig = plt.subplots()
            ax.plot(f, np.sqrt(np.abs(Pxy)) / np.max(np.sqrt(np.abs(Pxy))))
            ax.plot(f, np.angle(Pxy) / (2.0 * pi * f))
        fig.show()


def _main(argv=None):
    """
    Compile arguments for showxcorrx workflow.
    """
    args = vars(_get_parser().parse_args(argv))

    # Additional argument parsing not handled by argparse
    if args['arbvec'] is not None:
        if len(args['arbvec']) == 2:
            args['arbvec'].append(args['arbvec'][0] * 0.9)
            args['arbvec'].append(args['arbvec'][1] * 1.1)
        elif len(args['arbvec']) != 4:
            raise ValueError("Argument '-F' or '--arbvec' must be either two "
                             "or four floats.")

    showxcorrx_workflow(**args)


if __name__ == '__main__':
    _main()
