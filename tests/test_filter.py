#!/usr/bin/env python
from __future__ import print_function, division

from tide_funcs import noncausalfilter
import numpy as np
import pylab as plt

def testfilter(display=False):
    tr = 0.1
    testlen = 1000
    freqs = [0.009, 0.1, 0.3, 1.0, 4.0]
    timeaxis = np.arange(0.0, 1.0 * testlen) * tr
    overall = 0.0 * timeaxis
    signals = []
    for thefreq in freqs:
        signals.append(0.25 * np.sin(2.0 * np.pi * timeaxis * thefreq))
        overall += signals[-1]

    legend = ['Raw signal']
        
    lfoout = [signals[1]]
    legend.append('LFO source')
    lfofilter = noncausalfilter(filtertype='lfo',usebutterworth=False, usetrapfftfilt=False)
    lfoout.append(lfofilter.apply(1.0/tr, overall))
    legend.append('LFO brickwall')
    lfofilter = noncausalfilter(filtertype='lfo',usebutterworth=False, usetrapfftfilt=True)
    lfoout.append(lfofilter.apply(1.0/tr, overall))
    legend.append('LFO trap FFT')
    lfofilter = noncausalfilter(filtertype='lfo',usebutterworth=True, butterworthorder=6, usetrapfftfilt=False)
    lfoout.append(lfofilter.apply(1.0/tr, overall))
    legend.append('LFO butterworth')

    respout = [signals[2]]
    legend.append('Respiratory source')
    respfilter = noncausalfilter(filtertype='resp',usebutterworth=False, usetrapfftfilt=False)
    respout.append(respfilter.apply(1.0/tr, overall))
    legend.append('Respiratory brickwall')
    respfilter = noncausalfilter(filtertype='resp',usebutterworth=False, usetrapfftfilt=True)
    respout.append(respfilter.apply(1.0/tr, overall))
    legend.append('Respiratory trap FFT')
    respfilter = noncausalfilter(filtertype='resp',usebutterworth=True, butterworthorder=6, usetrapfftfilt=False)
    respout.append(respfilter.apply(1.0/tr, overall))
    legend.append('Respiratory butterworth')

    cardout = [signals[3]]
    legend.append('Cardiac source')
    cardfilter = noncausalfilter(filtertype='cardiac',usebutterworth=False, usetrapfftfilt=False)
    cardout.append(cardfilter.apply(1.0/tr, overall))
    legend.append('Cardiac brickwall')
    cardfilter = noncausalfilter(filtertype='cardiac',usebutterworth=False, usetrapfftfilt=True)
    cardout.append(cardfilter.apply(1.0/tr, overall))
    legend.append('Cardiac trap FFT')
    cardfilter = noncausalfilter(filtertype='cardiac',usebutterworth=True, butterworthorder=6, usetrapfftfilt=False)
    cardout.append(cardfilter.apply(1.0/tr, overall))
    legend.append('Cardiac butterworth')

    if display:
        plt.figure()
        plt.ylim([-1.0, 5.0])
        plt.plot(overall)
        plt.hold(True)
        for tc in lfoout:
            plt.plot(tc + 1.0)
        for tc in respout:
            plt.plot(tc + 2.0)
        for tc in cardout:
            plt.plot(tc + 3.0)
        plt.legend(legend)
        plt.show()

    aethresh = 1 
    for i in range(1,4):
        for j in range(0, len(cardout[0])):
            print(cardout[0][j], cardout[1][j], cardout[2][j], cardout[3][j])
        np.testing.assert_almost_equal(cardout[0], cardout[i], aethresh)
   
        for j in range(0, len(respout[0])):
            print(respout[0][j], respout[1][j], respout[2][j], respout[3][j])
        np.testing.assert_almost_equal(respout[0], respout[i], aethresh)

        for j in range(0, len(lfoout[0])):
            print(lfoout[0][j], lfoout[1][j], lfoout[2][j], lfoout[3][j])
        np.testing.assert_almost_equal(lfoout[0], lfoout[i], aethresh)

def main():
    testfilter(display=True)

if __name__ == '__main__':
    main()
