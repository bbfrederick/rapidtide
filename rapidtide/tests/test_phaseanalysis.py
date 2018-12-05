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
#       $Date: 2016/07/11 14:50:43 $
#       $Id: showxcorr,v 1.41 2016/07/11 14:50:43 frederic Exp $
#
from __future__ import print_function, division

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


import matplotlib.pyplot as plt


def eval_phaseanalysis(rootname=None, display=False):
    # read in some data
    testwaveform = tide_io.readvec(rootname + '.txt')

    # now calculate the phase waveform
    instantaneous_phase, amplitude_envelope = tide_fit.phaseanalysis(testwaveform)
    tide_io.writevec(amplitude_envelope, rootname + '_ampenv.txt')
    tide_io.writevec(instantaneous_phase, rootname + '_instphase_unwrapped.txt')
    filtered_phase = tide_math.trendfilt(instantaneous_phase, order=3, ndevs=2.0)
    tide_io.writevec(filtered_phase, rootname + '_filtered_instphase_unwrapped.txt')
    initialphase = instantaneous_phase[0]

    if display:
        plt.figure()
        plt.plot(instantaneous_phase)
        plt.plot(filtered_phase)
        plt.show()

    return True


def test_phaseanalysis(display=False):
    eval_phaseanalysis(rootname='rapidtide/tests/phasetest', display=display)

def main():
    test_phaseanalysis(display=True)


if __name__ == '__main__':
    main()
