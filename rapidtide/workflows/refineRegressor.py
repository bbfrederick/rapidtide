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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.refineregressor as tide_refineregressor
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
from rapidtide.tests.utils import mse

"""
A class to implement regressor refinement
"""


class RegressorRefiner:

    refinemaskvoxels = None

    def __init__(
        self,
        internalvalidfmrishape: Any,
        internalvalidpaddedfmrishape: Any,
        pid: Any,
        outputname: Any,
        initial_fmri_x: Any,
        paddedinitial_fmri_x: Any,
        os_fmri_x: Any,
        sharedmem: bool = False,
        offsettime: float = 0.0,
        ampthresh: float = 0.3,
        lagminthresh: float = 0.25,
        lagmaxthresh: float = 3.0,
        sigmathresh: float = 1000.0,
        cleanrefined: bool = False,
        bipolar: bool = False,
        fixdelay: bool = False,
        includemask: Optional[Any] = None,
        excludemask: Optional[Any] = None,
        LGR: Optional[Any] = None,
        nprocs: int = 1,
        detrendorder: int = 1,
        alwaysmultiproc: bool = False,
        showprogressbar: bool = True,
        chunksize: int = 50000,
        padtrs: int = 10,
        refineprenorm: str = "var",
        refineweighting: Optional[Any] = None,
        refinetype: str = "pca",
        pcacomponents: float = 0.8,
        dodispersioncalc: bool = False,
        dispersioncalc_lower: float = -5.0,
        dispersioncalc_upper: float = 5.0,
        dispersioncalc_step: float = 0.5,
        windowfunc: str = "hamming",
        passes: int = 3,
        maxpasses: int = 15,
        convergencethresh: Optional[Any] = None,
        interptype: str = "univariate",
        usetmask: bool = False,
        tmask_y: Optional[Any] = None,
        tmaskos_y: Optional[Any] = None,
        fastresamplerpadtime: float = 45.0,
        prewhitenregressor: bool = False,
        prewhitenlags: int = 10,
        debug: bool = False,
        rt_floattype: np.dtype = np.float64,
    ) -> None:
        """
        Initialize the object with configuration parameters for fMRI data processing.

        This constructor sets up internal attributes and performs initial setup tasks such as
        allocating memory and applying masks based on provided parameters.

        Parameters
        ----------
        internalvalidfmrishape : Any
            Shape of the internal valid fMRI data.
        internalvalidpaddedfmrishape : Any
            Shape of the padded internal valid fMRI data.
        pid : Any
            Process identifier used for memory allocation.
        outputname : Any
            Name of the output file or dataset.
        initial_fmri_x : Any
            Initial fMRI data array.
        paddedinitial_fmri_x : Any
            Padded version of the initial fMRI data.
        os_fmri_x : Any
            Oversampled fMRI data array.
        sharedmem : bool, optional
            Whether to use shared memory for processing (default is False).
        offsettime : float, optional
            Time offset in seconds (default is 0.0).
        ampthresh : float, optional
            Amplitude threshold for filtering (default is 0.3).
        lagminthresh : float, optional
            Minimum lag threshold for correlation analysis (default is 0.25).
        lagmaxthresh : float, optional
            Maximum lag threshold for correlation analysis (default is 3.0).
        sigmathresh : float, optional
            Significance threshold for statistical tests (default is 1000.0).
        cleanrefined : bool, optional
            Whether to apply refined cleaning steps (default is False).
        bipolar : bool, optional
            Whether to use bipolar filtering (default is False).
        fixdelay : bool, optional
            Whether to fix delay in the processing pipeline (default is False).
        includemask : Optional[Any], optional
            Mask to include specific regions (default is None).
        excludemask : Optional[Any], optional
            Mask to exclude specific regions (default is None).
        LGR : Optional[Any], optional
            Logarithmic gradient or related parameter (default is None).
        nprocs : int, optional
            Number of processes to use (default is 1).
        detrendorder : int, optional
            Order of detrending polynomial (default is 1).
        alwaysmultiproc : bool, optional
            Force multiprocessing even for small tasks (default is False).
        showprogressbar : bool, optional
            Show progress bar during processing (default is True).
        chunksize : int, optional
            Size of data chunks for processing (default is 50000).
        padtrs : int, optional
            Number of TRs to pad (default is 10).
        refineprenorm : str, optional
            Pre-normalization method for refinement ("var", "mean", etc.) (default is "var").
        refineweighting : Optional[Any], optional
            Weighting scheme for refinement (default is None).
        refinetype : str, optional
            Type of refinement to perform ("pca", "ica", etc.) (default is "pca").
        pcacomponents : float, optional
            Fraction of PCA components to retain (default is 0.8).
        dodispersioncalc : bool, optional
            Whether to perform dispersion calculation (default is False).
        dispersioncalc_lower : float, optional
            Lower bound for dispersion calculation (default is -5.0).
        dispersioncalc_upper : float, optional
            Upper bound for dispersion calculation (default is 5.0).
        dispersioncalc_step : float, optional
            Step size for dispersion calculation (default is 0.5).
        windowfunc : str, optional
            Window function used in spectral analysis (default is "hamming").
        passes : int, optional
            Number of filter passes (default is 3).
        maxpasses : int, optional
            Maximum allowed number of passes (default is 15).
        convergencethresh : Optional[Any], optional
            Convergence threshold for iterative algorithms (default is None).
        interptype : str, optional
            Interpolation type for resampling ("univariate", "multivariate") (default is "univariate").
        usetmask : bool, optional
            Whether to use temporal mask (default is False).
        tmask_y : Optional[Any], optional
            Temporal mask for y-axis (default is None).
        tmaskos_y : Optional[Any], optional
            Oversampled temporal mask for y-axis (default is None).
        fastresamplerpadtime : float, optional
            Padding time for fast resampling (default is 45.0).
        prewhitenregressor : bool, optional
            Apply pre-whitening to regressors (default is False).
        prewhitenlags : int, optional
            Number of lags for pre-whitening (default is 10).
        debug : bool, optional
            Enable debug mode (default is False).
        rt_floattype : np.dtype, optional
            Rapidtide floating-point data type (default is np.float64).

        Returns
        -------
        None
            This method initializes the object and does not return any value.

        Notes
        -----
        - The function internally calls `setmasks` and `_allocatemem` to initialize
          masks and allocate memory respectively.
        - The parameters are stored as instance attributes for use in subsequent processing steps.

        Examples
        --------
        >>> obj = MyClass(
        ...     internalvalidfmrishape=(64, 64, 30),
        ...     internalvalidpaddedfmrishape=(64, 64, 35),
        ...     pid=12345,
        ...     outputname="output.nii",
        ...     initial_fmri_x=np.random.rand(64, 64, 30),
        ...     paddedinitial_fmri_x=np.random.rand(64, 64, 35),
        ...     os_fmri_x=np.random.rand(64, 64, 60),
        ...     sharedmem=True,
        ...     offsettime=0.5,
        ...     ampthresh=0.5,
        ...     lagminthresh=0.3,
        ...     lagmaxthresh=2.0,
        ...     sigmathresh=500.0,
        ...     cleanrefined=True,
        ...     bipolar=False,
        ...     fixdelay=False,
        ...     includemask=None,
        ...     excludemask=None,
        ...     LGR=None,
        ...     nprocs=4,
        ...     detrendorder=2,
        ...     alwaysmultiproc=False,
        ...     showprogressbar=True,
        ...     chunksize=10000,
        ...     padtrs=5,
        ...     refineprenorm="mean",
        ...     refineweighting=None,
        ...     refinetype="pca",
        ...     pcacomponents=0.9,
        ...     dodispersioncalc=True,
        ...     dispersioncalc_lower=-4.0,
        ...     dispersioncalc_upper=4.0,
        ...     dispersioncalc_step=0.25,
        ...     windowfunc="hann",
        ...     passes=2,
        ...     maxpasses=10,
        ...     convergencethresh=None,
        ...     interptype="multivariate",
        ...     usetmask=True,
        ...     tmask_y=np.ones((64, 64)),
        ...     tmaskos_y=np.ones((64, 64)),
        ...     fastresamplerpadtime=30.0,
        ...     prewhitenregressor=True,
        ...     prewhitenlags=5,
        ...     debug=False,
        ...     rt_floattype=np.float32,
        ... )
        """
        self.internalvalidfmrishape = internalvalidfmrishape
        self.internalvalidpaddedfmrishape = internalvalidpaddedfmrishape
        self.sharedmem = sharedmem
        self.outputname = outputname
        self.initial_fmri_x = initial_fmri_x
        self.paddedinitial_fmri_x = paddedinitial_fmri_x
        self.os_fmri_x = os_fmri_x

        self.offsettime = offsettime
        self.ampthresh = ampthresh
        self.lagminthresh = lagminthresh
        self.lagmaxthresh = lagmaxthresh
        self.sigmathresh = sigmathresh
        self.cleanrefined = cleanrefined
        self.bipolar = bipolar
        self.fixdelay = fixdelay
        self.LGR = LGR
        self.nprocs = nprocs
        self.detrendorder = detrendorder
        self.alwaysmultiproc = alwaysmultiproc
        self.showprogressbar = showprogressbar
        self.chunksize = chunksize
        self.padtrs = padtrs
        self.refineprenorm = refineprenorm
        self.refineweighting = refineweighting
        self.refinetype = refinetype
        self.pcacomponents = pcacomponents
        self.dodispersioncalc = dodispersioncalc
        self.dispersioncalc_lower = dispersioncalc_lower
        self.dispersioncalc_upper = dispersioncalc_upper
        self.dispersioncalc_step = dispersioncalc_step
        self.windowfunc = windowfunc
        self.passes = passes
        self.maxpasses = maxpasses
        self.convergencethresh = convergencethresh
        self.interptype = interptype
        self.usetmask = usetmask
        self.tmask_y = tmask_y
        self.tmaskos_y = tmaskos_y
        self.fastresamplerpadtime = fastresamplerpadtime
        self.prewhitenregressor = prewhitenregressor
        self.prewhitenlags = prewhitenlags
        self.debug = debug
        self.rt_floattype = rt_floattype

        self.setmasks(includemask, excludemask)
        self.totalrefinementbytes = self._allocatemem(pid)

    def setmasks(self, includemask: Any, excludemask: Any) -> None:
        """
        Set the include and exclude masks for the object.

        Parameters
        ----------
        includemask : Any
            The mask to be used for including elements. Type and structure depends
            on the specific implementation and usage context.
        excludemask : Any
            The mask to be used for excluding elements. Type and structure depends
            on the specific implementation and usage context.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method assigns the provided masks to instance attributes `includemask`
        and `excludemask`. The masks are typically used for filtering or selection
        operations in data processing workflows.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.setmasks([1, 0, 1], [0, 1, 0])
        >>> print(obj.includemask)
        [1, 0, 1]
        >>> print(obj.excludemask)
        [0, 1, 0]
        """
        self.includemask = includemask
        self.excludemask = excludemask

    def _allocatemem(self, pid: Any) -> int:
        """
        Allocate memory for refinement arrays using shared memory if specified.

        This function allocates four arrays used in the refinement process:
        `shiftedtcs`, `weights`, `paddedshiftedtcs`, and `paddedweights`. These
        arrays are allocated with shapes determined by `internalvalidfmrishape` and
        `internalvalidpaddedfmrishape`, using the specified data type and memory
        sharing settings.

        Parameters
        ----------
        pid : Any
            Process identifier used to name shared memory segments.

        Returns
        -------
        int
            Total number of bytes allocated for the refinement arrays.

        Notes
        -----
        If `sharedmem` is True, the arrays are allocated in shared memory; otherwise,
        they are allocated locally. The function prints information about the
        allocation size and location, and logs memory usage after allocation.

        Examples
        --------
        >>> self._allocatemem(pid=1234)
        allocated 10.500 MB in shared memory for refinement
        11010048
        """
        self.shiftedtcs, self.shiftedtcs_shm = tide_util.allocarray(
            self.internalvalidfmrishape,
            self.rt_floattype,
            shared=self.sharedmem,
            name=f"shiftedtcs_{pid}",
        )
        self.weights, self.weights_shm = tide_util.allocarray(
            self.internalvalidfmrishape,
            self.rt_floattype,
            shared=self.sharedmem,
            name=f"weights_{pid}",
        )
        self.paddedshiftedtcs, self.paddedshiftedtcs_shm = tide_util.allocarray(
            self.internalvalidpaddedfmrishape,
            self.rt_floattype,
            shared=self.sharedmem,
            name=f"paddedshiftedtcs_{pid}",
        )
        self.paddedweights, self.paddedweights_shm = tide_util.allocarray(
            self.internalvalidpaddedfmrishape,
            self.rt_floattype,
            shared=self.sharedmem,
            name=f"paddedweights_{pid}",
        )
        if self.sharedmem:
            ramlocation = "in shared memory"
        else:
            ramlocation = "locally"
        totalrefinementbytes = (
            self.shiftedtcs.nbytes
            + self.weights.nbytes
            + self.paddedshiftedtcs.nbytes
            + self.paddedweights.nbytes
        )
        thesize, theunit = tide_util.format_bytes(totalrefinementbytes)
        print(f"allocated {thesize:.3f} {theunit} {ramlocation} for refinement")
        tide_util.logmem("after refinement array allocation")
        return totalrefinementbytes

    def cleanup(self) -> None:
        """
        Clean up memory resources by deleting internal attributes and shared memory segments.

        This method removes all internal arrays and their corresponding shared memory segments
        when shared memory is being used. It's designed to free up memory resources that were
        allocated during processing.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        - Deletes the following internal attributes: paddedshiftedtcs, paddedweights,
          shiftedtcs, weights
        - If shared memory is enabled (sharedmem=True), also cleans up the corresponding
          shared memory segments using tide_util.cleanup_shm()
        - This method should be called when the object is no longer needed to prevent
          memory leaks

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.cleanup()
        >>> # All internal memory resources are now freed
        """
        del self.paddedshiftedtcs
        del self.paddedweights
        del self.shiftedtcs
        del self.weights
        if self.sharedmem:
            tide_util.cleanup_shm(self.paddedshiftedtcs_shm)
            tide_util.cleanup_shm(self.paddedweights_shm)
            tide_util.cleanup_shm(self.shiftedtcs_shm)
            tide_util.cleanup_shm(self.weights_shm)

    def makemask(self, lagstrengths: Any, lagtimes: Any, lagsigma: Any, fitmask: Any) -> bool:
        """
        Create a refinement mask based on lag strength, lag time, and sigma thresholds.

        This function generates a mask for refining regressor parameters by evaluating
        the quality of lag estimates against specified thresholds. The mask determines
        which voxels should be included in the refinement process based on their
        lag strength, lag time, and sigma values.

        Parameters
        ----------
        lagstrengths : array-like
            Array containing lag strength values for each voxel
        lagtimes : array-like
            Array containing lag time values for each voxel
        lagsigma : array-like
            Array containing sigma values for each voxel
        fitmask : array-like
            Boolean mask indicating which voxels to consider for fitting

        Returns
        -------
        bool
            True if voxels are included in the refine mask, False if no voxels
            meet the refinement criteria

        Notes
        -----
        The function uses internal threshold parameters to determine which voxels
        should be included in the refinement process. These include:

        - ampthresh: amplitude threshold
        - lagminthresh: minimum lag threshold
        - lagmaxthresh: maximum lag threshold
        - sigmathresh: sigma threshold

        If no voxels meet the criteria, a critical log message is generated and
        the function returns False.

        Examples
        --------
        >>> # Assuming self is an instance of a class with the required attributes
        >>> result = self.makemask(lagstrengths, lagtimes, lagsigma, fitmask)
        >>> if result:
        ...     print("Refinement mask created successfully")
        ... else:
        ...     print("No voxels in refine mask")
        """
        # create the refinement mask
        (
            self.refinemaskvoxels,
            self.refinemask,
            self.locationfails,
            self.ampfails,
            self.lagfails,
            self.sigmafails,
            self.numinmask,
        ) = tide_refineregressor.makerefinemask(
            lagstrengths,
            lagtimes,
            lagsigma,
            fitmask,
            offsettime=self.offsettime,
            ampthresh=self.ampthresh,
            lagminthresh=self.lagminthresh,
            lagmaxthresh=self.lagmaxthresh,
            sigmathresh=self.sigmathresh,
            cleanrefined=self.cleanrefined,
            bipolar=self.bipolar,
            includemask=self.includemask,
            excludemask=self.excludemask,
            fixdelay=self.fixdelay,
            debug=self.debug,
        )

        if self.numinmask == 0:
            self.LGR.critical("No voxels in refine mask - adjust thresholds or external masks")
            return False
        else:
            return True

    def getrefinemask(self) -> NDArray:
        """
        Return the refinement mask.

        Returns
        -------
        NDArray
            The refinement mask stored in the instance.

        Notes
        -----
        This method provides access to the refinement mask attribute. The refinement
        mask is typically used in computational physics or data analysis workflows
        to identify regions of interest or apply specific processing to certain data
        points.

        Examples
        --------
        >>> mask = obj.getrefinemask()
        >>> print(mask)
        [True, False, True, False]
        """
        return self.refinemask

    def getpaddedshiftedtcs(self) -> NDArray:
        """
        Return the padded and shifted time-course data.

        This method retrieves the pre-computed padded and shifted time-course data
        that has been processed for analysis. The data is typically used in
        time-series analysis or signal processing applications where temporal
        alignment and padding are required.

        Returns
        -------
        NDArray
            The padded and shifted time-course data stored in the instance variable
            `paddedshiftedtcs`. The exact format depends on the data processing
            pipeline that generated this data.

        Notes
        -----
        This method serves as a simple getter function for the `paddedshiftedtcs`
        attribute. The actual computation of padded and shifted time-course data
        should be performed prior to calling this method.

        Examples
        --------
        >>> processor = TimeCourseProcessor()
        >>> processor.compute_padded_shifted_tcs()
        >>> result = processor.getpaddedshiftedtcs()
        >>> print(result.shape)
        (1000, 50)
        """
        return self.paddedshiftedtcs

    def alignvoxels(self, fmri_data_valid: Any, fmritr: Any, lagtimes: Any) -> int:
        """
        Align timecourses to prepare for refinement.

        This function aligns voxel timecourses for further refinement processing by
        coordinating the alignment of fMRI data with specified lag times and processing
        parameters.

        Parameters
        ----------
        fmri_data_valid : Any
            Valid fMRI data to be aligned
        fmritr : Any
            fMRI temporal resolution information
        lagtimes : Any
            Lag times to be used for alignment

        Returns
        -------
        int
            Returns the number of voxels processed

        Notes
        -----
        The function utilizes the `tide_refineregressor.alignvoxels` function internally
        and passes all relevant processing parameters including multiprocessing settings,
        detrending options, and padding parameters.

        Examples
        --------
        >>> alignvoxels(fmri_data_valid, fmritr, lagtimes)
        >>> # Function processes data and updates internal state
        """
        # align timecourses to prepare for refinement
        self.LGR.info("aligning timecourses")
        voxelsprocessed_rra = tide_refineregressor.alignvoxels(
            fmri_data_valid,
            fmritr,
            self.shiftedtcs,
            self.weights,
            self.paddedshiftedtcs,
            self.paddedweights,
            lagtimes,
            self.refinemask,
            nprocs=self.nprocs,
            detrendorder=self.detrendorder,
            offsettime=self.offsettime,
            alwaysmultiproc=self.alwaysmultiproc,
            showprogressbar=self.showprogressbar,
            chunksize=self.chunksize,
            padtrs=self.padtrs,
            rt_floattype=self.rt_floattype,
        )
        return voxelsprocessed_rra
        # self.LGR.info(f"align complete: {voxelsprocessed_rra=}")

    def prenormalize(self, lagtimes: Any, lagstrengths: Any, R2: Any) -> None:
        """
        Pre-normalize time series data for refinement regression.

        This function applies pre-normalization to the padded and shifted time series
        data using the specified lag times, lag strengths, and R2 values. The
        normalization is performed through the underlying tide_refineregressor.prenorm
        function with the appropriate internal parameters.

        Parameters
        ----------
        lagtimes : Any
            Array or list of lag times to be used in the pre-normalization process.
        lagstrengths : Any
            Array or list of lag strengths corresponding to the lag times.
        R2 : Any
            Array or list of R2 values used for the pre-normalization calculation.

        Returns
        -------
        None
            This function does not return any value. It modifies internal attributes
            in-place.

        Notes
        -----
        The function internally uses:
        - self.paddedshiftedtcs: padded and shifted time series data
        - self.refinemask: refinement mask for the normalization process
        - self.lagmaxthresh: maximum lag threshold
        - self.refineprenorm: refinement pre-normalization parameters
        - self.refineweighting: refinement weighting parameters

        Examples
        --------
        >>> prenormalize(lagtimes=[1, 2, 3], lagstrengths=[0.5, 0.3, 0.8], R2=[0.9, 0.85, 0.92])
        """
        tide_refineregressor.prenorm(
            self.paddedshiftedtcs,
            self.refinemask,
            lagtimes,
            self.lagmaxthresh,
            lagstrengths,
            R2,
            self.refineprenorm,
            self.refineweighting,
        )

    def refine(
        self,
        theprefilter: Any,
        fmritr: Any,
        thepass: Any,
        lagstrengths: Any,
        lagtimes: Any,
        previousnormoutputdata: Any,
        corrmasksize: Any,
    ) -> Tuple[int, dict, NDArray, NDArray, NDArray, bool, Optional[str], Any]:
        """
        Refine the regressor by iteratively applying filtering and resampling operations.

        This method performs iterative refinement of a regressor using a series of
        filtering, resampling, and normalization steps. It tracks convergence and
        updates internal state variables accordingly.

        Parameters
        ----------
        theprefilter : Any
            The prefilter to be applied to the data.
        fmritr : Any
            The fMRI temporal resolution (TR).
        thepass : Any
            The current pass number in the refinement process.
        lagstrengths : Any
            The lag strengths used in the refinement.
        lagtimes : Any
            The lag times used in the refinement.
        previousnormoutputdata : Any
            The normalized output data from the previous pass.
        corrmasksize : Any
            The size of the correlation mask.

        Returns
        -------
        tuple
            A tuple containing:
            - voxelsprocessed_rr : int
                Number of voxels processed in this pass.
            - outputdict : dict
                Dictionary of output statistics for this pass.
            - previousnormoutputdata : NDArray
                Updated normalized output data for the next pass.
            - resampref_y : NDArray
                Resampled refined regressor at oversampled frequency.
            - resampnonosref_y : NDArray
                Resampled refined regressor at original frequency.
            - stoprefining : bool
                Flag indicating whether refinement should stop.
            - refinestopreason : str or None
                Reason for stopping refinement.
            - genlagtc : FastResampler
                Generator for lag time courses.

        Notes
        -----
        This function modifies internal attributes of the class, such as `paddedoutputdata`,
        `locationfails`, `ampfails`, `lagfails`, and `sigmafails`. It also writes output
        files using `tide_io.writebidstsv`.

        Examples
        --------
        >>> refine(prefilter, fmritr, 1, lagstrengths, lagtimes, prev_data, mask_size)
        """
        (
            voxelsprocessed_rr,
            self.paddedoutputdata,
        ) = tide_refineregressor.dorefine(
            self.paddedshiftedtcs,
            self.refinemask,
            self.weights,
            theprefilter,
            fmritr,
            thepass,
            lagstrengths,
            lagtimes,
            self.refinetype,
            1.0 / fmritr,
            self.outputname,
            detrendorder=self.detrendorder,
            pcacomponents=self.pcacomponents,
            dodispersioncalc=self.dodispersioncalc,
            dispersioncalc_lower=self.dispersioncalc_lower,
            dispersioncalc_upper=self.dispersioncalc_upper,
            dispersioncalc_step=self.dispersioncalc_step,
            windowfunc=self.windowfunc,
            cleanrefined=self.cleanrefined,
            bipolar=self.bipolar,
            rt_floattype=self.rt_floattype,
        )
        outputdict = {}
        outputdict["refinemasksize_pass" + str(thepass)] = voxelsprocessed_rr
        outputdict["refinemaskpct_pass" + str(thepass)] = 100.0 * voxelsprocessed_rr / corrmasksize
        outputdict["refinelocationfails_pass" + str(thepass)] = self.locationfails
        outputdict["refineampfails_pass" + str(thepass)] = self.ampfails
        outputdict["refinelagfails_pass" + str(thepass)] = self.lagfails
        outputdict["refinesigmafails_pass" + str(thepass)] = self.sigmafails

        if self.prewhitenregressor:
            self.paddedoutputdata = tide_fit.prewhiten(
                self.paddedoutputdata, self.prewhitenlags, debug=self.debug
            )

        fmrifreq = 1.0 / fmritr
        if voxelsprocessed_rr > 0:
            paddednormoutputdata = tide_math.stdnormalize(
                theprefilter.apply(fmrifreq, self.paddedoutputdata)
            )
            outputdata = self.paddedoutputdata[self.padtrs : -self.padtrs]
            normoutputdata = tide_math.stdnormalize(theprefilter.apply(fmrifreq, outputdata))
            normunfilteredoutputdata = tide_math.stdnormalize(outputdata)
            tide_io.writebidstsv(
                f"{self.outputname}_desc-refinedmovingregressor_timeseries",
                normunfilteredoutputdata,
                fmrifreq,
                columns=["unfiltered_pass" + str(thepass)],
                extraheaderinfo={
                    "Description": "The raw and filtered probe regressor produced by the refinement procedure, at the time resolution of the data"
                },
                append=(thepass > 1),
            )
            tide_io.writebidstsv(
                f"{self.outputname}_desc-refinedmovingregressor_timeseries",
                normoutputdata,
                fmrifreq,
                columns=["filtered_pass" + str(thepass)],
                extraheaderinfo={
                    "Description": "The raw and filtered probe regressor produced by the refinement procedure, at the time resolution of the data"
                },
                append=True,
            )

        # check for convergence
        regressormse = mse(normoutputdata, previousnormoutputdata)
        outputdict["regressormse_pass" + str(thepass).zfill(2)] = regressormse
        self.LGR.info(f"regressor difference at end of pass {thepass:d} is {regressormse:.6f}")
        if self.convergencethresh is not None:
            if thepass >= self.maxpasses:
                self.LGR.info("refinement ended (maxpasses reached)")
                stoprefining = True
                refinestopreason = "maxpassesreached"
            elif regressormse < self.convergencethresh:
                self.LGR.info("refinement ended (refinement has converged")
                stoprefining = True
                refinestopreason = "convergence"
            else:
                stoprefining = False
        elif thepass >= self.passes:
            stoprefining = True
            refinestopreason = "passesreached"
        else:
            stoprefining = False
            refinestopreason = None
        outputdict["refinestopreason"] = refinestopreason

        if self.detrendorder > 0:
            resampnonosref_y = tide_fit.detrend(
                tide_resample.doresample(
                    self.paddedinitial_fmri_x,
                    paddednormoutputdata,
                    self.initial_fmri_x,
                    method=self.interptype,
                ),
                order=self.detrendorder,
                demean=True,
            )
            resampref_y = tide_fit.detrend(
                tide_resample.doresample(
                    self.paddedinitial_fmri_x,
                    paddednormoutputdata,
                    self.os_fmri_x,
                    method=self.interptype,
                ),
                order=self.detrendorder,
                demean=True,
            )
        else:
            resampnonosref_y = tide_resample.doresample(
                self.paddedinitial_fmri_x,
                paddednormoutputdata,
                self.initial_fmri_x,
                method=self.interptype,
            )
            resampref_y = tide_resample.doresample(
                self.paddedinitial_fmri_x,
                paddednormoutputdata,
                self.os_fmri_x,
                method=self.interptype,
            )
        if self.usetmask:
            resampnonosref_y *= self.tmask_y
            thefit, R2val = tide_fit.mlregress(self.tmask_y, resampnonosref_y)
            resampnonosref_y -= thefit[0, 1] * self.tmask_y
            resampref_y *= self.tmaskos_y
            thefit, R2val = tide_fit.mlregress(self.tmaskos_y, resampref_y)
            resampref_y -= thefit[0, 1] * self.tmaskos_y

        # reinitialize genlagtc for resampling
        previousnormoutputdata = np.zeros_like(normoutputdata)
        genlagtc = tide_resample.FastResampler(
            self.paddedinitial_fmri_x,
            paddednormoutputdata,
            padtime=self.fastresamplerpadtime,
        )
        genlagtc.save(f"{self.outputname}_desc-lagtcgenerator_timeseries")
        if self.debug:
            genlagtc.info()
        (
            outputdict[f"kurtosis_reference_pass{thepass + 1}"],
            outputdict[f"kurtosisz_reference_pass{thepass + 1}"],
            outputdict[f"kurtosisp_reference_pass{thepass + 1}"],
        ) = tide_stats.kurtosisstats(resampref_y)
        (
            outputdict[f"skewness_reference_pass{thepass + 1}"],
            outputdict[f"skewnessz_reference_pass{thepass + 1}"],
            outputdict[f"skewnessp_reference_pass{thepass + 1}"],
        ) = tide_stats.skewnessstats(resampref_y)
        if not stoprefining:
            tide_io.writebidstsv(
                f"{self.outputname}_desc-movingregressor_timeseries",
                tide_math.stdnormalize(resampnonosref_y),
                1.0 / fmritr,
                columns=["pass" + str(thepass + 1)],
                extraheaderinfo={
                    "Description": "The probe regressor used in each pass, at the time resolution of the data"
                },
                append=True,
            )
            oversampfreq = 1.0 / (self.os_fmri_x[1] - self.os_fmri_x[0])
            tide_io.writebidstsv(
                f"{self.outputname}_desc-oversampledmovingregressor_timeseries",
                tide_math.stdnormalize(resampref_y),
                oversampfreq,
                columns=["pass" + str(thepass + 1)],
                extraheaderinfo={
                    "Description": "The probe regressor used in each pass, at the time resolution used for calculating the similarity function"
                },
                append=True,
            )
        else:
            self.LGR.warning(f"refinement failed - terminating at end of pass {thepass}")
            stoprefining = True
            refinestopreason = "emptymask"

        return (
            voxelsprocessed_rr,
            outputdict,
            previousnormoutputdata,
            resampref_y,
            resampnonosref_y,
            stoprefining,
            refinestopreason,
            genlagtc,
        )


def refineRegressor(
    LGR: Any,
    TimingLGR: Any,
    thepass: Any,
    optiondict: Any,
    fitmask: NDArray,
    internaloffsetincludemask_valid: NDArray,
    internaloffsetexcludemask_valid: NDArray,
    internalrefineincludemask_valid: NDArray,
    internalrefineexcludemask_valid: NDArray,
    internaldespeckleincludemask: NDArray,
    validvoxels: NDArray,
    theRegressorRefiner: Any,
    lagtimes: NDArray,
    lagstrengths: NDArray,
    lagsigma: NDArray,
    fmri_data_valid: NDArray,
    fmritr: float,
    R2: Any,
    theprefilter: Any,
    previousnormoutputdata: Any,
    theinputdata: Any,
    numpadtrs: Any,
    outputname: Any,
    nativefmrishape: Any,
    bidsbasedict: Any,
    rt_floattype: np.dtype = np.dtype(np.float64),
    debug: bool = False,
) -> Tuple[NDArray, NDArray, bool, str, Any]:
    """
    Refine the regressor by adjusting masks, aligning timecourses, and performing refinement steps.

    This function performs regressor refinement during a specified pass, including:
    - Updating offset time based on lag properties
    - Managing masks for refinement and despeckling
    - Aligning timecourses
    - Pre-normalizing data
    - Executing the refinement step using a regressor refiner object

    Parameters
    ----------
    LGR : logging.Logger
        Logger instance for general logging.
    TimingLGR : logging.Logger
        Logger instance for timing-related messages.
    thepass : int
        Current pass number in the refinement process.
    optiondict : dict
        Dictionary containing various options and settings for the refinement process.
    fitmask : array_like
        Boolean mask indicating voxels to be considered in fitting.
    internaloffsetincludemask_valid : array_like or None
        Mask for including voxels in offset calculation.
    internaloffsetexcludemask_valid : array_like or None
        Mask for excluding voxels from offset calculation.
    internalrefineincludemask_valid : array_like or None
        Mask for including voxels in refinement.
    internalrefineexcludemask_valid : array_like or None
        Mask for excluding voxels from refinement.
    internaldespeckleincludemask : array_like
        Mask for including voxels in despeckling.
    validvoxels : array_like
        Indices of valid voxels.
    theRegressorRefiner : object
        Regressor refiner object with methods for mask setting, masking, alignment, etc.
    lagtimes : array_like
        Array of lag times.
    lagstrengths : array_like
        Array of lag strengths.
    lagsigma : array_like
        Array of lag sigma values.
    fmri_data_valid : array_like
        Valid fMRI data.
    fmritr : float
        fMRI repetition time.
    R2 : array_like
        R2 values.
    theprefilter : object
        Filter object used for preprocessing.
    previousnormoutputdata : array_like
        Previously normalized output data.
    theinputdata : object
        Input data object (e.g., Nifti1Image).
    numpadtrs : int
        Number of padded timepoints.
    outputname : str
        Base name for output files.
    nativefmrishape : tuple
        Shape of the native fMRI data.
    bidsbasedict : dict
        Dictionary for BIDS metadata.
    rt_floattype : numpy.dtype, optional
        Data type for floating-point operations, default is np.float64.
    debug : bool, optional
        Enable debug mode, default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - resampref_y : NDArray
          Resampled refined y values.
        - resampnonosref_y : NDArray
          Resampled non-oscillatory refined y values.
        - stoprefining : bool
          Flag indicating whether refinement should stop.
        - refinestopreason : str
          Reason for stopping refinement.
        - genlagtc : array_like
          Generated lag timecourses.

    Notes
    -----
    - This function modifies `optiondict` in-place, updating offset times and other parameters.
    - The function uses `theRegressorRefiner` to perform various refinement steps including:
      `setmasks`, `makemask`, `alignvoxels`, `prenormalize`, and `refine`.
    - If `refinedespeckled` is False and `despeckle_passes` > 0, the exclude mask for refinement
      is adjusted to include voxels being despeckled.
    - The function may exit early if no voxels qualify for refinement.

    Examples
    --------
    >>> refineRegressor(
    ...     LGR, TimingLGR, 1, optiondict, fitmask, offset_incl, offset_excl,
    ...     refine_incl, refine_excl, despeckle_incl, valid_voxels, refiner,
    ...     lagtimes, lagstrengths, lagsigma, fmri_data, fmritr, R2, prefilter,
    ...     prev_norm_data, input_data, numpadtrs, output_name, native_shape,
    ...     bids_dict, debug=True
    ... )
    """
    LGR.info(f"\n\nRegressor refinement, pass {thepass}")
    TimingLGR.info(f"Regressor refinement start, pass {thepass}")
    if optiondict["refineoffset"]:
        # check that we won't end up excluding all voxels from offset calculation before accepting mask
        offsetmask = np.uint16(fitmask)
        if internaloffsetincludemask_valid is not None:
            offsetmask[np.where(internaloffsetincludemask_valid == 0)] = np.uint16(0)
        if internaloffsetexcludemask_valid is not None:
            offsetmask[np.where(internaloffsetexcludemask_valid != 0.0)] = np.uint16(0)
        if tide_stats.getmasksize(offsetmask) == 0:
            LGR.warning(
                "NB: cannot exclude voxels from offset calculation mask - including for this pass"
            )
            offsetmask = fitmask + 0

        peaklag, dummy, dummy = tide_stats.gethistprops(
            lagtimes[np.where(offsetmask > 0)],
            optiondict["histlen"],
            pickleft=optiondict["pickleft"],
            peakthresh=optiondict["pickleftthresh"],
        )
        optiondict["offsettime"] = peaklag
        optiondict["offsettime_total"] += peaklag
        optiondict[f"offsettime_pass{thepass}"] = optiondict["offsettime"]
        optiondict[f"offsettime_total_pass{thepass}"] = optiondict["offsettime_total"]
        LGR.info(
            f"offset time set to {optiondict['offsettime']:.3f}, "
            f"total is {optiondict['offsettime_total']:.3f}"
        )

    if optiondict["refinedespeckled"] or (optiondict["despeckle_passes"] == 0):
        # if refinedespeckled is true, or there is no despeckling, masks are unaffected
        thisinternalrefineexcludemask_valid = internalrefineexcludemask_valid
    else:
        # if refinedespeckled is false and there is despeckling, need to make a proper mask
        if internalrefineexcludemask_valid is None:
            # if there is currently no exclude mask, set exclude mask = despeckle mask
            thisinternalrefineexcludemask_valid = np.where(
                internaldespeckleincludemask[validvoxels] == 0.0, 0, 1
            )
        else:
            # if there is a current exclude mask, add any voxels that are being despeckled
            thisinternalrefineexcludemask_valid = np.where(
                internalrefineexcludemask_valid > 0, 1, 0
            )
            thisinternalrefineexcludemask_valid[
                np.where(internaldespeckleincludemask[validvoxels] != 0.0)
            ] = 1

        # now check that we won't end up excluding all voxels from refinement before accepting mask
        overallmask = np.uint16(fitmask)
        if internalrefineincludemask_valid is not None:
            overallmask[np.where(internalrefineincludemask_valid == 0)] = np.uint16(0)
        if thisinternalrefineexcludemask_valid is not None:
            overallmask[np.where(thisinternalrefineexcludemask_valid != 0.0)] = np.uint16(0)
        if tide_stats.getmasksize(overallmask) == 0:
            LGR.warning(
                "NB: cannot exclude despeckled voxels from refinement - including for this pass"
            )
            thisinternalrefineexcludemask_valid = internalrefineexcludemask_valid
    theRegressorRefiner.setmasks(
        internalrefineincludemask_valid, thisinternalrefineexcludemask_valid
    )

    # regenerate regressor for next pass
    # create the refinement mask
    LGR.info("making refine mask")
    createdmask = theRegressorRefiner.makemask(lagstrengths, lagtimes, lagsigma, fitmask)
    print(f"Refine mask has {theRegressorRefiner.refinemaskvoxels} voxels")
    if not createdmask:
        print("no voxels qualify for refinement - exiting")
        sys.exit()

    # align timecourses to prepare for refinement
    LGR.info("aligning timecourses")
    tide_util.disablemkl(optiondict["nprocs_refine"], debug=optiondict["threaddebug"])
    voxelsprocessed_rra = theRegressorRefiner.alignvoxels(fmri_data_valid, fmritr, lagtimes)
    tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])
    LGR.info(f"align complete: {voxelsprocessed_rra=}")

    # prenormalize
    LGR.info("prenormalizing timecourses")
    theRegressorRefiner.prenormalize(lagtimes, lagstrengths, R2)

    # now doing the refinement
    (
        voxelsprocessed_rr,
        outputdict,
        previousnormoutputdata,
        resampref_y,
        resampnonosref_y,
        stoprefining,
        refinestopreason,
        genlagtc,
    ) = theRegressorRefiner.refine(
        theprefilter,
        fmritr,
        thepass,
        lagstrengths,
        lagtimes,
        previousnormoutputdata,
        optiondict["corrmasksize"],
    )
    TimingLGR.info(
        f"Regressor refinement end, pass {thepass}",
        {
            "message2": voxelsprocessed_rr,
            "message3": "voxels",
        },
    )
    for key, value in outputdict.items():
        optiondict[key] = value

    # Save shifted timecourses for César
    if optiondict["saveintermediatemaps"] and optiondict["savelagregressors"]:
        theheader = theinputdata.copyheader()
        bidspasssuffix = f"_intermediatedata-pass{thepass}"
        maplist = [
            (
                (theRegressorRefiner.getpaddedshiftedtcs())[:, numpadtrs:-numpadtrs],
                "shiftedtcs",
                "bold",
                None,
                "The filtered input fMRI data, in voxels used for refinement, time shifted by the negated delay in every voxel so that the moving blood component is aligned.",
            ),
        ]
        tide_io.savemaplist(
            f"{outputname}{bidspasssuffix}",
            maplist,
            validvoxels,
            nativefmrishape,
            theheader,
            bidsbasedict,
            filetype=theinputdata.filetype,
            rt_floattype=rt_floattype,
            cifti_hdr=theinputdata.cifti_hdr,
            debug=debug,
        )

    return resampref_y, resampnonosref_y, stoprefining, refinestopreason, genlagtc
