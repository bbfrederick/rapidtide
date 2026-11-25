#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rapidtide.simfuncfit import onesimfuncfit


class _DummyFitter:
    corrtimeaxis = np.linspace(-5.0, 5.0, 101)
    lagmod = 1000.0

    def __init__(self):
        """Initialize the test double for ``_DummyFitter``."""
        self.last_range = None

    def setguess(self, *_args, **_kwargs):
        """
        Set up ``setguess`` for the test.

        Parameters
        ----------
        *_args : object
            Additional positional arguments passed by the caller.
        **_kwargs : object
            Additional keyword arguments passed by the caller.
        """
        return None

    def setrange(self, lagmin, lagmax):
        """
        Set up ``setrange`` for the test.

        Parameters
        ----------
        lagmin : object
            Value supplied as ``lagmin`` by the test case.
        lagmax : object
            Value supplied as ``lagmax`` by the test case.
        """
        self.last_range = (lagmin, lagmax)

    def setlthresh(self, _val):
        """
        Set up ``setlthresh`` for the test.

        Parameters
        ----------
        _val : object
            Value supplied as ``_val`` by the test case.
        """
        return None

    def fit(self, correlationfunc):
        """
        Set up ``fit`` for the test.

        Parameters
        ----------
        correlationfunc : object
            Value supplied as ``correlationfunc`` by the test case.

        Returns
        -------
        tuple
            Values supplied together for the test case.
        """
        maxindex = int(np.argmax(correlationfunc))
        maxlag = float(self.corrtimeaxis[maxindex])
        maxval = float(correlationfunc[maxindex])
        return maxindex, maxlag, maxval, 1.0, 1, 0, maxindex, maxindex


def test_onesimfuncfit_sets_range_around_initiallag_with_despeckle_thresh():
    """Verify onesimfuncfit sets range around initiallag with despeckle thresh."""
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
