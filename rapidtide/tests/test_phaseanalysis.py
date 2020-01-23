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
#       $Date: 2016/07/11 14:50:43 $
#       $Id: showxcorr,v 1.41 2016/07/11 14:50:43 frederic Exp $
#
from __future__ import print_function, division

import os
import numpy as np
import scipy as sp

import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util
import rapidtide.io as tide_io
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.resample as tide_resample
import rapidtide.correlate as tide_corr
import rapidtide.multiproc as tide_multiproc
import rapidtide.glmpass as tide_glmpass
from rapidtide.tests.utils import mse

import matplotlib.pyplot as plt

def eval_phaseanalysis(phasestep=0.01, amplitude=1.0, numpoints=100, display=False):
    # read in some data
    phases = sp.linspace(0.0, numpoints * phasestep, num=numpoints, endpoint=False)
    testwaveform = amplitude * np.cos(phases)
    if display:
        plt.figure()
        plt.plot(phases)
        plt.show()

        plt.figure()
        plt.plot(testwaveform)
        plt.show()

    # now calculate the phase waveform
    instantaneous_phase, amplitude_envelope = tide_fit.phaseanalysis(testwaveform)
    filtered_phase = tide_math.trendfilt(instantaneous_phase, order=3, ndevs=2.0)
    initialphase = instantaneous_phase[0]

    if display:
        plt.figure()
        plt.plot(instantaneous_phase)
        plt.plot(filtered_phase)
        plt.plot(phases)
        plt.show()

    return mse(phases, instantaneous_phase), mse(phases, filtered_phase)


def test_phaseanalysis(debug=False, display=False):
    msethresh = 1e-3
    instantaneous_mse, filtered_mse = eval_phaseanalysis(phasestep=0.1, amplitude=3.0, numpoints=1000, display=display)
    print(instantaneous_mse, filtered_mse)
    assert instantaneous_mse < msethresh
    assert filtered_mse < msethresh
    

def main():
    test_phaseanalysis(debug=True, display=True)


if __name__ == '__main__':
    main()
