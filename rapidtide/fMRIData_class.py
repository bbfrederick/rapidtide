#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
from statsmodels.robust import mad

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util


class fMRIData:
    header = None
    data = None
    data_byvoxel = None
    data_valid = None
    nativeshape = None
    mask = None
    validvoxels = None
    filename = None
    textio = False
    fileiscifti = False
    fileisnifti = False

    def __init__(
        self,
        filename=None,
        header=None,
        data=None,
        mask=None,
        validvoxels=None,
        dims=None,
        sizes=None,
        description=None,
    ):
        r"""

        Parameters
        ----------
        corrtimeaxis:  1D float array
            The time axis of the correlation function
        lagmin: float
            The minimum allowed lag time in seconds
        lagmax: float
            The maximum allowed lag time in seconds
        absmaxsigma: float
            The maximum allowed peak halfwidth in seconds
        hardlimit
        bipolar: boolean
            If true find the correlation peak with the maximum absolute value, regardless of sign
        threshval
        uthreshval
        debug
        zerooutbadfit
        maxguess
        useguess
        searchfrac
        lagmod
        enforcethresh
        displayplots

        Returns
        -------


        Methods
        -------
        fit(corrfunc):
            Fit the correlation function given in corrfunc and return the location of the peak in seconds, the maximum
            correlation value, the peak width
        setrange(lagmin, lagmax):
            Specify the search range for lag peaks, in seconds
        """

        self.filename = filename
        self.header = header
        self.data = data
        self.mask = mask
        self.validvoxels = validvoxels
        self.dims = dims
        self.sizes = sizes
        self.description = description

    def load(self, filename):
        self.filename = filename
        ####################################################
        #  Read data
        ####################################################
        # open the fmri datafil
        if tide_io.checkiftext(self.filename):
            self.textio = True
        else:
            self.textio = False

        if self.textio:
            self.data = tide_io.readvecs(self.filename)
            self.header = None
            theshape = np.shape(nim_data)
            self.xsize = theshape[0]
            self.ysize = 1
            self.numslices = 1
            self.fileiscifti = False
            self.timepoints = theshape[1]
            self.thesizes = [0, int(self.xsize), 1, 1, int(self.timepoints)]
            self.numspatiallocs = int(self.xsize)
            self.nativespaceshape = self.xsize
            self.cifti_hdr = None
        else:
            self.fileiscifti = tide_io.checkifcifti(self.filename)
            if self.fileiscifti:
                (
                    cifti,
                    cifti_hdr,
                    self.data,
                    self.header,
                    thedims,
                    thesizes,
                    dummy,
                ) = tide_io.readfromcifti(self.filename)
                self.isgrayordinate = True
                self.timepoints = nim_data.shape[1]
                numspatiallocs = nim_data.shape[0]
                LGR.debug(
                    f"cifti file has {timepoints} timepoints, {numspatiallocs} numspatiallocs"
                )
                slicesize = numspatiallocs
                nativespaceshape = (1, 1, 1, 1, numspatiallocs)
            else:
                LGR.debug("input file is NIFTI")
                nim, self.data, self.header, thedims, thesizes = tide_io.readfromnifti(
                    fmrifilename
                )
                optiondict["isgrayordinate"] = False
                xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
                numspatiallocs = int(xsize) * int(ysize) * int(numslices)
                cifti_hdr = None
                nativespaceshape = (xsize, ysize, numslices)
            xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
