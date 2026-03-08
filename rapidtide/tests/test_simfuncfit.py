#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rapidtide.simfuncfit import onesimfuncfit


class _DummyFitter:
    corrtimeaxis = np.linspace(-5.0, 5.0, 101)
    lagmod = 1000.0

    def __init__(self):
        self.last_range = None

    def setguess(self, *_args, **_kwargs):
        return None

    def setrange(self, lagmin, lagmax):
        self.last_range = (lagmin, lagmax)

    def setlthresh(self, _val):
        return None

    def fit(self, correlationfunc):
        maxindex = int(np.argmax(correlationfunc))
        maxlag = float(self.corrtimeaxis[maxindex])
        maxval = float(correlationfunc[maxindex])
        return maxindex, maxlag, maxval, 1.0, 1, 0, maxindex, maxindex


def test_onesimfuncfit_sets_range_around_initiallag_with_despeckle_thresh():
    fitter = _DummyFitter()
    corr = np.zeros(101, dtype=float)
    corr[50] = 1.0
    onesimfuncfit(
        corr,
        fitter,
        initiallag=2.0,
        despeckle_thresh=6.0,
    )
    assert fitter.last_range == (-1.0, 5.0)
