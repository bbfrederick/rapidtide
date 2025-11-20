#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2018-2025 Blaise Frederick
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
import copy
import time
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter, welch
from scipy.stats import kurtosis, pearsonr, skew
from statsmodels.robust import mad
from tqdm import tqdm

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.genericmultiproc as tide_genericmultiproc
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False


def rrifromphase(timeaxis: NDArray, thephase: NDArray) -> None:
    """
    Convert phase to range rate.

    This function converts phase measurements to range rate values using the
    provided time axis and phase data.

    Parameters
    ----------
    timeaxis : NDArray
        Time axis values corresponding to the phase measurements.
    thephase : NDArray
        Phase measurements to be converted to range rate.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function performs conversion from phase to range rate but does not
    return the result. The actual implementation details are not provided
    in the function signature.

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([0, 1, 2, 3])
    >>> phase = np.array([0.1, 0.2, 0.3, 0.4])
    >>> rrifromphase(time, phase)
    """
    return None


def calc_3d_optical_flow(
    video: NDArray,
    projmask: NDArray,
    flowhdr: dict,
    outputroot: str,
    window_size: int = 3,
    debug: bool = False,
) -> tuple[NDArray, NDArray]:
    """
    Compute 3D optical flow for a video volume using the Lucas-Kanade method.

    This function calculates optical flow in three dimensions (x, y, z) across
    a sequence of video frames. It uses a Lucas-Kanade approach to estimate
    motion vectors at each voxel, considering a local window around each pixel.
    The results are saved as NIfTI files for each frame.

    Parameters
    ----------
    video : NDArray
        4D array of shape (xsize, ysize, zsize, num_frames) representing the
        input video data.
    projmask : NDArray
        3D boolean or integer mask of shape (xsize, ysize, zsize) indicating
        which voxels to process for optical flow computation.
    flowhdr : dict
        Header dictionary for NIfTI output files, containing metadata for
        the optical flow results.
    outputroot : str
        Root name for output NIfTI files. Files will be saved with suffixes
        `_desc-flow_phase-XX_map` and `_desc-flowmag_phase-XX_map`.
    window_size : int, optional
        Size of the local window used for gradient computation. Default is 3.
    debug : bool, optional
        If True, print debug information during computation. Default is False.

    Returns
    -------
    tuple[NDArray, NDArray]
        A tuple containing:
        - `flow_vectors`: 5D array of shape (xsize, ysize, zsize, num_frames, 3)
          representing the computed optical flow vectors for each frame.
        - `None`: Placeholder return value; function currently returns only
          `flow_vectors` and saves outputs to disk.

    Notes
    -----
    - The optical flow is computed using a Lucas-Kanade method with spatial
      gradients in x, y, and z directions.
    - Temporal gradient is computed as the difference between consecutive frames.
    - Output files are saved using `tide_io.savetonifti`.
    - The function wraps around frames when reaching the end (i.e., next frame
      for the last frame is the first frame).

    Examples
    --------
    >>> import numpy as np
    >>> video = np.random.rand(64, 64, 32, 10)
    >>> mask = np.ones((64, 64, 32), dtype=bool)
    >>> header = {}
    >>> output_root = "flow_result"
    >>> flow_vectors = calc_3d_optical_flow(video, mask, header, output_root)
    >>> print(flow_vectors.shape)
    (64, 64, 32, 10, 3)
    """
    # window Define the window size for Lucas-Kanade method
    # Get the number of frames, height, and width of the video
    singlehdr = copy.deepcopy(flowhdr)
    singlehdr["dim"][4] = 1
    xsize, ysize, zsize, num_frames = video.shape

    # Create an empty array to store the optical flow vectors
    flow_vectors = np.zeros((xsize, ysize, zsize, num_frames, 3))

    if debug:
        print(
            f"calc_3d_optical_flow: calculating flow in {xsize}, {ysize}, {zsize}, {num_frames} array with window_size {window_size}"
        )

    # Loop over all pairs of consecutive frames
    for i in range(num_frames):
        if debug:
            print(f"calculating flow for time point {i}")
        prev_frame = video[:, :, :, i]
        next_frame = video[:, :, :, (i + 1) % num_frames]

        # Initialize the flow vectors to zero
        flow = np.zeros((xsize, ysize, zsize, 3))

        # Loop over each pixel in the image
        for z in range(window_size // 2, zsize - window_size // 2):
            if debug:
                print(f"\tz={z}")
            for y in range(window_size // 2, ysize - window_size // 2):
                for x in range(window_size // 2, zsize - window_size // 2):
                    if projmask[x, y, z] > 0:
                        # Define the window around the pixel
                        window_prev = prev_frame[
                            x - window_size // 2 : x + window_size // 2 + 1,
                            y - window_size // 2 : y + window_size // 2 + 1,
                            z - window_size // 2 : z + window_size // 2 + 1,
                        ]
                        window_next = next_frame[
                            x - window_size // 2 : x + window_size // 2 + 1,
                            y - window_size // 2 : y + window_size // 2 + 1,
                            z - window_size // 2 : z + window_size // 2 + 1,
                        ]

                        # Compute the gradient of the window in x, y, and z directions
                        grad_x = np.gradient(window_prev)[0]
                        grad_y = np.gradient(window_prev)[1]
                        grad_z = np.gradient(window_prev)[2]

                        # Compute the temporal gradient between two frames
                        grad_t = window_next - window_prev

                        # Compute the optical flow vector using Lucas-Kanade method
                        A = np.vstack((grad_x.ravel(), grad_y.ravel(), grad_z.ravel())).T
                        b = -grad_t.ravel()
                        flow_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                        # Store the optical flow vector in the result array
                        flow[x, y, z, 0] = flow_vec[0]
                        flow[x, y, z, 1] = flow_vec[1]
                        flow[x, y, z, 2] = flow_vec[2]

        # Store the optical flow vectors in the result array
        flow_vectors[:, :, :, i, 0] = flow[..., 0]
        flow_vectors[:, :, :, i, 1] = flow[..., 1]
        flow_vectors[:, :, :, i, 2] = flow[..., 2]
        thename = f"{outputroot}_desc-flow_phase-{str(i).zfill(2)}_map"
        tide_io.savetonifti(flow_vectors[:, :, :, i, :], flowhdr, thename)
        thename = f"{outputroot}_desc-flowmag_phase-{str(i).zfill(2)}_map"
        tide_io.savetonifti(
            np.sqrt(np.sum(np.square(flow_vectors[:, :, :, i, :]), axis=3)), singlehdr, thename
        )

    return flow_vectors


def phasejolt(phaseimage: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute phase gradient-based metrics including jump, jolt, and laplacian.

    This function calculates three important metrics from a phase image:
    - jump: average absolute gradient magnitude
    - jolt: average absolute second-order gradient magnitude
    - laplacian: sum of second-order partial derivatives

    Parameters
    ----------
    phaseimage : NDArray
        Input phase image array of arbitrary dimensions (typically 2D or 3D).

    Returns
    -------
    tuple of NDArray
        A tuple containing three arrays:
        - jump: array of same shape as input, representing average absolute gradient
        - jolt: array of same shape as input, representing average absolute second-order gradient
        - laplacian: array of same shape as input, representing Laplacian of the phase image

    Notes
    -----
    The function computes gradients using numpy's gradient function which applies
    central differences in the interior and first differences at the boundaries.
    All metrics are computed in a voxel-wise manner across the entire image.

    Examples
    --------
    >>> import numpy as np
    >>> phase_img = np.random.rand(10, 10)
    >>> jump, jolt, laplacian = phasejolt(phase_img)
    >>> print(jump.shape, jolt.shape, laplacian.shape)
    (10, 10) (10, 10) (10, 10)
    """

    # Compute the gradient of the window in x, y, and z directions
    grad_x, grad_y, grad_z = np.gradient(phaseimage)

    # Now compute the second order gradients of the window in x, y, and z directions
    grad_xx, grad_xy, grad_xz = np.gradient(grad_x)
    grad_yx, grad_yy, grad_yz = np.gradient(grad_y)
    grad_zx, grad_zy, grad_zz = np.gradient(grad_z)

    # Calculate our metrics of interest
    jump = (np.fabs(grad_x) + np.fabs(grad_y) + np.fabs(grad_z)) / 3.0
    jolt = (
        (np.fabs(grad_xx) + np.fabs(grad_xy) + np.fabs(grad_xz))
        + (np.fabs(grad_yx) + np.fabs(grad_yy) + np.fabs(grad_yz))
        + (np.fabs(grad_zx) + np.fabs(grad_zy) + np.fabs(grad_zz))
    ) / 9.0
    laplacian = grad_xx + grad_yy + grad_zz
    return (jump, jolt, laplacian)


def cardiacsig(
    thisphase: float | NDArray,
    amps: tuple | NDArray = (1.0, 0.0, 0.0),
    phases: NDArray | None = None,
    overallphase: float = 0.0,
) -> float | NDArray:
    """
    Generate a cardiac signal model using harmonic components.

    This function creates a cardiac signal by summing weighted cosine waves
    at different harmonic frequencies. The signal can be computed for
    scalar phase values or arrays of phase values.

    Parameters
    ----------
    thisphase : float or NDArray
        The phase value(s) at which to evaluate the cardiac signal.
        Can be a scalar or array of phase values.
    amps : tuple or NDArray, optional
        Amplitude coefficients for each harmonic component. Default is
        (1.0, 0.0, 0.0) representing the fundamental frequency with
        amplitude 1.0 and higher harmonics with amplitude 0.0.
    phases : NDArray or None, optional
        Phase shifts for each harmonic component. If None, all phase shifts
        are set to zero. Default is None.
    overallphase : float, optional
        Overall phase shift applied to the entire signal. Default is 0.0.

    Returns
    -------
    float or NDArray
        The computed cardiac signal value(s) at the given phase(s).
        Returns a scalar if input is scalar, or array if input is array.

    Notes
    -----
    The cardiac signal is computed as:
    .. math::
        s(t) = \\sum_{i=0}^{n-1} A_i \\cos((i+1)\\phi + \\phi_i + \\phi_{overall})

    where:
    - A_i are the amplitude coefficients
    - φ is the phase value
    - φ_i are the harmonic phase shifts
    - φ_{overall} is the overall phase shift

    Examples
    --------
    >>> import numpy as np
    >>> cardiacsig(0.5)
    1.0

    >>> cardiacsig(np.linspace(0, 2*np.pi, 100), amps=(1.0, 0.5, 0.2))
    array([...])

    >>> cardiacsig(1.0, amps=(2.0, 1.0, 0.5), phases=[0.0, np.pi/4, np.pi/2])
    -0.7071067811865476
    """
    total = 0.0
    if phases is None:
        phases = np.zeros_like(amps)
    for i in range(len(amps)):
        total += amps[i] * np.cos((i + 1) * thisphase + phases[i] + overallphase)
    return total


# Constants for signal processing
SIGN_NORMAL = 1.0
SIGN_INVERTED = -1.0
SIGNAL_INVERSION_FACTOR = -1.0


@dataclass
class CardiacExtractionConfig:
    """
    Configuration for cardiac signal extraction.

    Parameters
    ----------
    notchpct : float
        Percentage of notch bandwidth, default is 1.5.
    notchrolloff : float
        Notch filter rolloff, default is 0.5.
    invertphysiosign : bool
        If True, invert the physiological signal sign, default is False.
    madnorm : bool
        If True, use median absolute deviation normalization, default is True.
    nprocs : int
        Number of processes to use for computation, default is 1.
    arteriesonly : bool
        If True, only use arterial signal, default is False.
    fliparteries : bool
        If True, flip the arterial signal, default is False.
    debug : bool
        If True, enable debug output, default is False.
    verbose : bool
        If True, print verbose output, default is False.
    usemask : bool
        If True, use masking for valid voxels, default is True.
    multiplicative : bool
        If True, apply multiplicative normalization, default is True.
    """

    notchpct: float = 1.5
    notchrolloff: float = 0.5
    invertphysiosign: bool = False
    madnorm: bool = True
    nprocs: int = 1
    arteriesonly: bool = False
    fliparteries: bool = False
    debug: bool = False
    verbose: bool = False
    usemask: bool = True
    multiplicative: bool = True


@dataclass
class CardiacExtractionResult:
    """
    Results from cardiac signal extraction.

    This dataclass supports tuple unpacking for backward compatibility.

    Attributes
    ----------
    hirescardtc : NDArray
        High-resolution cardiac time course.
    cardnormfac : float
        Normalization factor for cardiac signal.
    hiresresptc : NDArray
        High-resolution respiratory time course.
    respnormfac : float
        Normalization factor for respiratory signal.
    slicesamplerate : float
        Slice sampling rate in Hz.
    numsteps : int
        Number of unique slice times.
    sliceoffsets : NDArray
        Slice offsets relative to TR.
    cycleaverage : NDArray
        Average signal per slice time step.
    slicenorms : NDArray
        Slice-wise normalization factors.
    """

    hirescardtc: NDArray
    cardnormfac: float
    hiresresptc: NDArray
    respnormfac: float
    slicesamplerate: float
    numsteps: int
    sliceoffsets: NDArray
    cycleaverage: NDArray
    slicenorms: NDArray

    def __iter__(self):
        """Support tuple unpacking for backward compatibility."""
        return iter(
            (
                self.hirescardtc,
                self.cardnormfac,
                self.hiresresptc,
                self.respnormfac,
                self.slicesamplerate,
                self.numsteps,
                self.sliceoffsets,
                self.cycleaverage,
                self.slicenorms,
            )
        )


def _validate_cardiacfromimage_inputs(
    normdata_byslice: NDArray,
    estweights_byslice: NDArray,
    numslices: int,
    timepoints: int,
    tr: float,
) -> None:
    """
    Validate input dimensions and values for cardiacfromimage.

    Parameters
    ----------
    normdata_byslice : NDArray
        Normalized fMRI data organized by slice.
    estweights_byslice : NDArray
        Estimated weights for each voxel and slice.
    numslices : int
        Number of slices in the acquisition.
    timepoints : int
        Number of time points in the fMRI time series.
    tr : float
        Repetition time (TR) in seconds.

    Raises
    ------
    ValueError
        If input dimensions or values are invalid.
    """
    if timepoints <= 0:
        raise ValueError(f"timepoints must be positive, got {timepoints}")

    if numslices <= 0:
        raise ValueError(f"numslices must be positive, got {numslices}")

    if tr <= 0:
        raise ValueError(f"tr must be positive, got {tr}")

    if normdata_byslice.shape[1] != numslices:
        raise ValueError(
            f"normdata_byslice slice dimension {normdata_byslice.shape[1]} "
            f"does not match numslices {numslices}"
        )

    if normdata_byslice.shape[2] != timepoints:
        raise ValueError(
            f"normdata_byslice timepoint dimension {normdata_byslice.shape[2]} "
            f"does not match timepoints {timepoints}"
        )

    if estweights_byslice.shape[1] != numslices:
        raise ValueError(
            f"estweights_byslice slice dimension {estweights_byslice.shape[1]} "
            f"does not match numslices {numslices}"
        )


def _prepare_weights(
    estweights_byslice: NDArray,
    appflips_byslice: NDArray | None,
    arteriesonly: bool,
    fliparteries: bool,
) -> tuple[NDArray, NDArray]:
    """
    Prepare appflips and weight arrays based on configuration.

    Parameters
    ----------
    estweights_byslice : NDArray
        Estimated weights for each voxel and slice.
    appflips_byslice : NDArray | None
        Array of application flips for each slice.
    arteriesonly : bool
        If True, only use arterial signal.
    fliparteries : bool
        If True, flip the arterial signal.

    Returns
    -------
    tuple[NDArray, NDArray]
        Processed appflips_byslice and theseweights_byslice arrays.
    """
    # Make sure there is an appflips array
    if appflips_byslice is None:
        appflips_byslice = np.ones_like(estweights_byslice)
    else:
        if arteriesonly:
            appflips_byslice[np.where(appflips_byslice > 0.0)] = 0.0

    # Prepare weights
    if fliparteries:
        theseweights_byslice = appflips_byslice.astype(np.float64) * estweights_byslice
    else:
        theseweights_byslice = estweights_byslice

    return appflips_byslice, theseweights_byslice


def _compute_slice_averages(
    normdata_byslice: NDArray,
    theseweights_byslice: NDArray,
    numslices: int,
    timepoints: int,
    numsteps: int,
    sliceoffsets: NDArray,
    signal_sign: float,
    madnorm: bool,
    usemask: bool,
    multiplicative: bool,
    verbose: bool,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute averaged signals for each slice with normalization.

    Parameters
    ----------
    normdata_byslice : NDArray
        Normalized fMRI data organized by slice.
    theseweights_byslice : NDArray
        Processed weights for each voxel and slice.
    numslices : int
        Number of slices in the acquisition.
    timepoints : int
        Number of time points in the fMRI time series.
    numsteps : int
        Number of unique slice times.
    sliceoffsets : NDArray
        Slice offsets relative to TR.
    signal_sign : float
        Sign factor for physiological signal (+1.0 or -1.0).
    madnorm : bool
        If True, use median absolute deviation normalization.
    usemask : bool
        If True, use masking for valid voxels.
    multiplicative : bool
        If True, apply multiplicative normalization.
    verbose : bool
        If True, print verbose output.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        - high_res_timecourse: High-resolution time course across all slices
        - cycleaverage: Average signal per slice time step
        - slicenorms: Normalization factors for each slice
    """
    high_res_timecourse = np.zeros((timepoints * numsteps), dtype=np.float64)
    cycleaverage = np.zeros((numsteps), dtype=np.float64)
    slice_averages = np.zeros((numslices, timepoints), dtype=np.float64)
    slicenorms = np.zeros((numslices), dtype=np.float64)

    if not verbose:
        print("Averaging slices...")

    for slice_idx in range(numslices):
        if verbose:
            print("Averaging slice", slice_idx)

        # Find valid voxels for this slice
        if usemask:
            valid_voxel_indices = np.where(np.abs(theseweights_byslice[:, slice_idx]) > 0)[0]
        else:
            valid_voxel_indices = np.where(np.abs(theseweights_byslice[:, slice_idx] >= 0))[0]

        if len(valid_voxel_indices) > 0:
            # Compute weighted average for this slice
            weighted_slice_data = np.mean(
                normdata_byslice[valid_voxel_indices, slice_idx, :]
                * theseweights_byslice[valid_voxel_indices, slice_idx, np.newaxis],
                axis=0,
            )

            # Apply normalization if requested
            if madnorm:
                slice_averages[slice_idx, :], slicenorms[slice_idx] = tide_math.madnormalize(
                    weighted_slice_data
                )
            else:
                slice_averages[slice_idx, :] = weighted_slice_data
                slicenorms[slice_idx] = 1.0

            # Build high-resolution time course
            for t in range(timepoints):
                high_res_timecourse[numsteps * t + sliceoffsets[slice_idx]] += (
                    signal_sign * slice_averages[slice_idx, t]
                )

    # Compute cycle average
    for i in range(numsteps):
        cycleaverage[i] = np.mean(high_res_timecourse[i:-1:numsteps])

    # Apply cycle average correction
    for t in range(len(high_res_timecourse)):
        if multiplicative:
            high_res_timecourse[t] /= cycleaverage[t % numsteps] + 1.0
        else:
            high_res_timecourse[t] -= cycleaverage[t % numsteps]

    if not verbose:
        print("done")

    return high_res_timecourse, cycleaverage, slicenorms


def _normalize_and_filter_signal(
    prefilter: tide_filt.NoncausalFilter,
    slicesamplerate: float,
    filtered_timecourse: NDArray,
    slicenorms: NDArray,
) -> tuple[NDArray, float]:
    """
    Apply filter and MAD normalization to signal.

    Parameters
    ----------
    prefilter : tide_filt.NoncausalFilter
        Prefilter object with an `apply` method for filtering physiological signals.
    slicesamplerate : float
        Slice sampling rate in Hz.
    filtered_timecourse : NDArray
        Input time course to filter and normalize.
    slicenorms : NDArray
        Slice-wise normalization factors.

    Returns
    -------
    tuple[NDArray, float]
        - Filtered and normalized signal
        - Normalization factor
    """
    signal, normfac = tide_math.madnormalize(prefilter.apply(slicesamplerate, filtered_timecourse))
    signal *= SIGNAL_INVERSION_FACTOR
    normfac *= np.mean(slicenorms)
    return signal, normfac


def _extract_physiological_signals(
    filtered_timecourse: NDArray,
    slicesamplerate: float,
    cardprefilter: tide_filt.NoncausalFilter,
    respprefilter: tide_filt.NoncausalFilter,
    slicenorms: NDArray,
) -> tuple[NDArray, float, NDArray, float]:
    """
    Extract and normalize cardiac and respiratory signals.

    Parameters
    ----------
    filtered_timecourse : NDArray
        Notch-filtered high-resolution time course.
    slicesamplerate : float
        Slice sampling rate in Hz.
    cardprefilter : tide_filt.NoncausalFilter
        Cardiac prefilter object.
    respprefilter : tide_filt.NoncausalFilter
        Respiratory prefilter object.
    slicenorms : NDArray
        Slice-wise normalization factors.

    Returns
    -------
    tuple[NDArray, float, NDArray, float]
        - hirescardtc: High-resolution cardiac time course
        - cardnormfac: Cardiac normalization factor
        - hiresresptc: High-resolution respiratory time course
        - respnormfac: Respiratory normalization factor
    """
    hirescardtc, cardnormfac = _normalize_and_filter_signal(
        cardprefilter, slicesamplerate, filtered_timecourse, slicenorms
    )

    hiresresptc, respnormfac = _normalize_and_filter_signal(
        respprefilter, slicesamplerate, filtered_timecourse, slicenorms
    )

    return hirescardtc, cardnormfac, hiresresptc, respnormfac


def cardiacfromimage(
    normdata_byslice: NDArray,
    estweights_byslice: NDArray,
    numslices: int,
    timepoints: int,
    tr: float,
    slicetimes: NDArray,
    cardprefilter: tide_filt.NoncausalFilter,
    respprefilter: tide_filt.NoncausalFilter,
    config: CardiacExtractionConfig,
    appflips_byslice: NDArray | None = None,
) -> CardiacExtractionResult:
    """
    Extract cardiac and respiratory signals from 4D fMRI data using slice timing.

    This function processes preprocessed fMRI data to isolate cardiac and respiratory
    physiological signals by leveraging slice timing information and filtering techniques.
    It applies normalization, averaging across slices, and harmonic notch filtering to
    extract clean physiological time series.

    Parameters
    ----------
    normdata_byslice : NDArray
        Normalized fMRI data organized by slice, shape (voxels, numslices, timepoints).
    estweights_byslice : NDArray
        Estimated weights for each voxel and slice, shape (voxels, numslices).
    numslices : int
        Number of slices in the acquisition.
    timepoints : int
        Number of time points in the fMRI time series.
    tr : float
        Repetition time (TR) in seconds.
    slicetimes : NDArray
        Slice acquisition times relative to the start of the TR, shape (numslices,).
    cardprefilter : tide_filt.NoncausalFilter
        Cardiac prefilter object with an `apply` method for filtering physiological signals.
    respprefilter : tide_filt.NoncausalFilter
        Respiratory prefilter object with an `apply` method for filtering physiological signals.
    config : CardiacExtractionConfig | None, optional
        Configuration object containing all processing parameters. If None, uses default config.
        If provided along with individual parameters, individual parameters override config values.
    appflips_byslice : NDArray | None, optional
        Array of application flips for each slice, default is None.

    Returns
    -------
    CardiacExtractionResult
        Dataclass containing:
        - hirescardtc: High-resolution cardiac time course
        - cardnormfac: Normalization factor for cardiac signal
        - hiresresptc: High-resolution respiratory time course
        - respnormfac: Normalization factor for respiratory signal
        - slicesamplerate: Slice sampling rate in Hz
        - numsteps: Number of unique slice times
        - sliceoffsets: Slice offsets relative to TR
        - cycleaverage: Average signal per slice time step
        - slicenorms: Slice-wise normalization factors

    Notes
    -----
    - The function assumes that `normdata_byslice` and `estweights_byslice` are properly
      preprocessed and aligned with slice timing information.
    - The cardiac and respiratory signals are extracted using harmonic notch filtering
      and prefiltering steps.
    - For backward compatibility, individual parameters can be passed instead of config.
      Individual parameters override config values when both are provided.

    Examples
    --------
    >>> # Using config object (recommended)
    >>> config = CardiacExtractionConfig(madnorm=True, verbose=False)
    >>> result = cardiacfromimage(
    ...     normdata_byslice, estweights_byslice, numslices, timepoints,
    ...     tr, slicetimes, cardprefilter, respprefilter, config=config
    ... )
    >>> print(result.slicesamplerate)

    >>> # Backward compatible usage (returns same result)
    >>> result = cardiacfromimage(
    ...     normdata_byslice, estweights_byslice, numslices, timepoints,
    ...     tr, slicetimes, cardprefilter, respprefilter
    ... )
    """

    # Validate inputs
    _validate_cardiacfromimage_inputs(
        normdata_byslice, estweights_byslice, numslices, timepoints, tr
    )

    # Find out what timepoints we have, and their spacing
    numsteps, minstep, sliceoffsets = tide_io.sliceinfo(slicetimes, tr)
    print(
        len(slicetimes),
        "slice times with",
        numsteps,
        "unique values - diff is",
        f"{minstep:.3f}",
    )

    # Determine signal sign
    signal_sign = SIGN_INVERTED if config.invertphysiosign else SIGN_NORMAL

    # Prepare weights
    appflips_byslice, theseweights_byslice = _prepare_weights(
        estweights_byslice,
        appflips_byslice,
        config.arteriesonly,
        config.fliparteries,
    )

    # Compute slice averages
    print("Making slice means...")
    high_res_timecourse, cycleaverage, slicenorms = _compute_slice_averages(
        normdata_byslice,
        theseweights_byslice,
        numslices,
        timepoints,
        numsteps,
        sliceoffsets,
        signal_sign,
        config.madnorm,
        config.usemask,
        config.multiplicative,
        config.verbose,
    )

    # Calculate slice sample rate
    slicesamplerate = 1.0 * numsteps / tr
    print(f"Slice sample rate is {slicesamplerate:.3f}")

    # Delete the TR frequency and the first subharmonic
    print("Notch filtering...")
    filtered_timecourse = tide_filt.harmonicnotchfilter(
        high_res_timecourse,
        slicesamplerate,
        1.0 / tr,
        notchpct=config.notchpct,
        debug=config.debug,
    )

    # Extract cardiac and respiratory waveforms
    hirescardtc, cardnormfac, hiresresptc, respnormfac = _extract_physiological_signals(
        filtered_timecourse,
        slicesamplerate,
        cardprefilter,
        respprefilter,
        slicenorms,
    )

    return CardiacExtractionResult(
        hirescardtc=hirescardtc,
        cardnormfac=cardnormfac,
        hiresresptc=hiresresptc,
        respnormfac=respnormfac,
        slicesamplerate=slicesamplerate,
        numsteps=numsteps,
        sliceoffsets=sliceoffsets,
        cycleaverage=cycleaverage,
        slicenorms=slicenorms,
    )


def theCOM(X: NDArray, data: NDArray) -> float:
    """
    Calculate the center of mass of a system of particles.

    Parameters
    ----------
    X : NDArray
        Array of positions (coordinates) of particles. Shape should be (n_particles, n_dimensions).
    data : NDArray
        Array of mass values for each particle. Shape should be (n_particles,).

    Returns
    -------
    float
        The center of mass of the system.

    Notes
    -----
    The center of mass is calculated using the formula:
    COM = Σ(m_i * x_i) / Σ(m_i)

    where m_i are the masses and x_i are the positions of particles.

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[1, 2], [3, 4], [5, 6]])
    >>> masses = np.array([1, 2, 3])
    >>> com = theCOM(positions, masses)
    >>> print(com)
    3.3333333333333335
    """
    # return the center of mass
    return np.sum(X * data) / np.sum(data)


def savgolsmooth(data: NDArray, smoothlen: int = 101, polyorder: int = 3) -> NDArray:
    """
    Apply Savitzky-Golay filter to smooth data.

    This function applies a Savitzky-Golay filter to smooth the input data using
    a polynomial fit. The filter preserves higher moments of the data better than
    simple moving averages, making it particularly useful for smoothing noisy data
    while preserving peak shapes and heights.

    Parameters
    ----------
    data : NDArray
        Input data to be smoothed. Can be 1D or 2D array.
    smoothlen : int, optional
        Length of the filter window (i.e., the number of coefficients).
        Must be a positive odd integer. Default is 101.
    polyorder : int, optional
        Order of the polynomial used to fit the samples. Must be less than
        `smoothlen`. Default is 3.

    Returns
    -------
    NDArray
        Smoothed data with the same shape as the input `data`.

    Notes
    -----
    The Savitzky-Golay filter is a digital filter that smooths data by fitting
    a polynomial of specified order to a sliding window of data points. It is
    particularly effective at preserving the shape and features of the original
    data while removing noise.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100)
    >>> smoothed = savgolsmooth(data, smoothlen=21, polyorder=3)

    >>> # For 2D data
    >>> data_2d = np.random.randn(50, 10)
    >>> smoothed_2d = savgolsmooth(data_2d, smoothlen=11, polyorder=2)
    """
    return savgol_filter(data, smoothlen, polyorder)


def getperiodic(
    inputdata: NDArray,
    Fs: float,
    fundfreq: float,
    ncomps: int = 1,
    width: float = 0.4,
    debug: bool = False,
) -> NDArray:
    """
    Apply a periodic filter to extract harmonic components from input data.

    This function applies a non-causal filter to isolate and extract periodic
    components of a signal based on a fundamental frequency and number of
    harmonics. It uses an arbitrary filter design to define stopband and passband
    frequencies for each harmonic component.

    Parameters
    ----------
    inputdata : NDArray
        Input signal data to be filtered.
    Fs : float
        Sampling frequency of the input signal (Hz).
    fundfreq : float
        Fundamental frequency of the periodic signal (Hz).
    ncomps : int, optional
        Number of harmonic components to extract. Default is 1.
    width : float, optional
        Width parameter controlling the bandwidth of each harmonic filter.
        Default is 0.4.
    debug : bool, optional
        If True, print debug information during processing. Default is False.

    Returns
    -------
    NDArray
        Filtered output signal containing the specified harmonic components.

    Notes
    -----
    The function reduces the number of components (`ncomps`) if the highest
    harmonic exceeds the Nyquist frequency (Fs/2). Each harmonic is filtered
    using an arbitrary filter with stopband and passband frequencies defined
    based on the `width` parameter.
    """
    outputdata = np.zeros_like(inputdata)
    lowerdist = fundfreq - fundfreq / (1.0 + width)
    upperdist = fundfreq * width
    if debug:
        print(f"GETPERIODIC: starting with fundfreq={fundfreq}, ncomps={ncomps}, Fs={Fs}")
    while ncomps * fundfreq >= Fs / 2.0:
        ncomps -= 1
        print(f"\tncomps reduced to {ncomps}")
    thefundfilter = tide_filt.NoncausalFilter(filtertype="arb")
    for component in range(ncomps):
        arb_lower = (component + 1) * fundfreq - lowerdist
        arb_upper = (component + 1) * fundfreq + upperdist
        arb_lowerstop = 0.9 * arb_lower
        arb_upperstop = 1.1 * arb_upper
        if debug:
            print(
                f"GETPERIODIC: component {component} - arb parameters:{arb_lowerstop}, {arb_lower}, {arb_upper}, {arb_upperstop}"
            )
        thefundfilter.setfreqs(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)
        outputdata += 1.0 * thefundfilter.apply(Fs, inputdata)
    return outputdata


def getcardcoeffs(
    cardiacwaveform: NDArray,
    slicesamplerate: float,
    minhr: float = 40.0,
    maxhr: float = 140.0,
    smoothlen: int = 101,
    debug: bool = False,
) -> float:
    """
    Compute the fundamental cardiac frequency from a cardiac waveform using spectral analysis.

    This function estimates the heart rate (in beats per minute) from a given cardiac waveform
    by performing a Welch periodogram and applying a smoothing filter to identify the dominant
    frequency component. The result is returned as a frequency value in Hz, which can be
    converted to BPM by multiplying by 60.

    Parameters
    ----------
    cardiacwaveform : NDArray
        Input cardiac waveform signal as a 1D numpy array.
    slicesamplerate : float
        Sampling rate of the input waveform in Hz.
    minhr : float, optional
        Minimum allowed heart rate in BPM. Default is 40.0.
    maxhr : float, optional
        Maximum allowed heart rate in BPM. Default is 140.0.
    smoothlen : int, optional
        Length of the Savitzky-Golay filter window for smoothing the spectrum.
        Default is 101.
    debug : bool, optional
        If True, print intermediate debug information including initial and final
        frequency estimates. Default is False.

    Returns
    -------
    float
        Estimated fundamental cardiac frequency in Hz.

    Notes
    -----
    The function applies a Hamming window to the input signal before spectral analysis.
    It removes spectral components outside the physiological range (defined by `minhr`
    and `maxhr`) and uses Savitzky-Golay smoothing to detect the peak frequency.

    Examples
    --------
    >>> import numpy as np
    >>> waveform = np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, 1000))
    >>> freq = getcardcoeffs(waveform, slicesamplerate=100)
    >>> print(f"Estimated heart rate: {freq * 60:.2f} BPM")
    """
    if len(cardiacwaveform) > 1024:
        thex, they = welch(cardiacwaveform, slicesamplerate, nperseg=1024)
    else:
        thex, they = welch(cardiacwaveform, slicesamplerate)
    initpeakfreq = np.round(thex[np.argmax(they)] * 60.0, 2)
    if initpeakfreq > maxhr:
        initpeakfreq = maxhr
    if initpeakfreq < minhr:
        initpeakfreq = minhr
    if debug:
        print("initpeakfreq:", initpeakfreq, "BPM")
    freqaxis, spectrum = tide_filt.spectrum(
        tide_filt.hamming(len(cardiacwaveform)) * cardiacwaveform,
        Fs=slicesamplerate,
        mode="complex",
    )
    # remove any spikes at zero frequency
    minbin = int(minhr // (60.0 * (freqaxis[1] - freqaxis[0])))
    maxbin = int(maxhr // (60.0 * (freqaxis[1] - freqaxis[0])))
    spectrum[:minbin] = 0.0
    spectrum[maxbin:] = 0.0

    # find the max
    ampspec = savgolsmooth(np.abs(spectrum), smoothlen=smoothlen)
    peakfreq = freqaxis[np.argmax(ampspec)]
    if debug:
        print("Cardiac fundamental frequency is", np.round(peakfreq * 60.0, 2), "BPM")
    normfac = np.sqrt(2.0) * tide_math.rms(cardiacwaveform)
    if debug:
        print("normfac:", normfac)
    return peakfreq


def _procOneVoxelDetrend(
    vox: int,
    voxelargs: tuple,
    **kwargs,
) -> tuple[int, NDArray]:
    """
    Detrend fMRI voxel data for a single voxel.

    This function applies detrending to fMRI voxel data using the tide_fit.detrend
    function. It supports both linear and polynomial detrending with optional
    mean centering.

    Parameters
    ----------
    vox : int
        Voxel index identifier.
    voxelargs : tuple
        Tuple containing fMRI voxel data as the first element. Expected format:
        (fmri_voxeldata,)
    **kwargs : dict
        Additional keyword arguments for detrending options:
        - detrendorder : int, optional
            Order of the detrend polynomial (default: 1 for linear detrend)
        - demean : bool, optional
            If True, remove the mean from the data (default: False)
        - debug : bool, optional
            If True, print debug information (default: False)

    Returns
    -------
    tuple
        A tuple containing:
        - vox : int
            The original voxel index
        - detrended_voxeldata : ndarray
            The detrended fMRI voxel data with the same shape as input

    Notes
    -----
    This function uses the tide_fit.detrend function internally for the actual
    detrending operation. The detrendorder parameter controls the polynomial order
    of the detrending (0 = mean removal only, 1 = linear detrend, 2 = quadratic detrend, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> from rapidtide.fit import detrend
    >>> data = np.random.randn(100)
    >>> result = _procOneVoxelDetrend(0, (data,), detrendorder=1, demean=True)
    >>> print(result[0])  # voxel index
    0
    >>> print(result[1].shape)  # detrended data shape
    (100,)
    """
    # unpack arguments
    options = {
        "detrendorder": 1,
        "demean": False,
        "debug": False,
    }
    options.update(kwargs)
    detrendorder = options["detrendorder"]
    demean = options["demean"]
    debug = options["debug"]
    [fmri_voxeldata] = voxelargs
    if debug:
        print(f"{vox=}, {detrendorder=}, {demean=}, {fmri_voxeldata.shape=}")

    detrended_voxeldata = tide_fit.detrend(fmri_voxeldata, order=detrendorder, demean=demean)

    return (
        vox,
        detrended_voxeldata,
    )


def _packDetrendvoxeldata(voxnum: int, voxelargs: list) -> list[NDArray]:
    """
    Extract voxel data for a specific voxel number from voxel arguments.

    Parameters
    ----------
    voxnum : int
        The voxel number to extract data for.
    voxelargs : tuple
        A tuple containing voxel data arrays, where the first element is
        expected to be a 2D array with voxel data indexed by [voxel, feature].

    Returns
    -------
    list
        A list containing a single element, which is a 1D array of feature
        values for the specified voxel number.

    Notes
    -----
    This function is designed to extract a single voxel's worth of data
    from a collection of voxel arguments for further processing in
    detrending operations.

    Examples
    --------
    >>> voxel_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> result = _packDetrendvoxeldata(1, (voxel_data,))
    >>> print(result)
    [[4, 5, 6]]
    """
    return [(voxelargs[0])[voxnum, :]]


def _unpackDetrendvoxeldata(retvals: tuple, voxelproducts: list) -> None:
    """
    Unpack detrend voxel data by assigning values to voxel products array.

    Parameters
    ----------
    retvals : tuple or list
        Contains two elements where retvals[0] is used as indices and retvals[1]
        contains the values to be assigned.
    voxelproducts : list
        List containing arrays where voxelproducts[0] is the target array that
        will be modified in-place with the assigned values.

    Returns
    -------
    None
        This function modifies voxelproducts[0] in-place and does not return anything.

    Notes
    -----
    This function performs an in-place assignment operation where values from
    retvals[1] are placed at the specified indices retvals[0] in the first
    element of voxelproducts list.

    Examples
    --------
    >>> retvals = ([0, 1, 2], [10, 20, 30])
    >>> voxelproducts = [np.zeros(5)]
    >>> _unpackDetrendvoxeldata(retvals, voxelproducts)
    >>> print(voxelproducts[0])
    [10. 20. 30.  0.  0.]
    """
    (voxelproducts[0])[retvals[0], :] = retvals[1]


def normalizevoxels(
    fmri_data: NDArray,
    detrendorder: int,
    validvoxels: NDArray,
    time: object,
    timings: list,
    LGR: object | None = None,
    mpcode: bool = True,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    debug: bool = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Normalize fMRI voxel data by detrending and z-scoring.

    This function applies detrending to fMRI data and then normalizes the data
    using mean and median-based scaling. It supports both single-threaded and
    multi-threaded processing for detrending.

    Parameters
    ----------
    fmri_data : NDArray
        2D array of fMRI data with shape (n_voxels, n_timepoints).
    detrendorder : int
        Order of detrending to apply. If 0, no detrending is performed.
    validvoxels : NDArray
        1D array of indices indicating which voxels are valid for processing.
    time : object
        Module or object with a `time.time()` method for timing operations.
    timings : list
        List to append timing information about processing steps.
    LGR : object, optional
        Logger object for debugging; default is None.
    mpcode : bool, optional
        If True, use multi-processing for detrending; default is True.
    nprocs : int, optional
        Number of processes to use in multi-processing; default is 1.
    alwaysmultiproc : bool, optional
        If True, always use multi-processing even for small datasets; default is False.
    showprogressbar : bool, optional
        If True, show progress bar during voxel processing; default is True.
    chunksize : int, optional
        Size of chunks for multi-processing; default is 1000.
    debug : bool, optional
        If True, enable debug output; default is False.

    Returns
    -------
    tuple of NDArray
        A tuple containing:
        - `normdata`: Normalized fMRI data (z-scored).
        - `demeandata`: Detrended and mean-centered data.
        - `means`: Mean values for each voxel.
        - `medians`: Median values for each voxel.
        - `mads`: Median absolute deviation for each voxel.

    Notes
    -----
    - The function modifies `fmri_data` in-place during detrending.
    - If `detrendorder` is greater than 0, detrending is applied using `tide_fit.detrend`.
    - Multi-processing is used when `mpcode=True` and the number of voxels exceeds a threshold.
    - Timing information is appended to the `timings` list.

    Examples
    --------
    >>> import numpy as np
    >>> from tqdm import tqdm
    >>> fmri_data = np.random.rand(100, 200)
    >>> validvoxels = np.arange(100)
    >>> timings = []
    >>> normdata, demeandata, means, medians, mads = normalizevoxels(
    ...     fmri_data, detrendorder=1, validvoxels=validvoxels,
    ...     time=time, timings=timings
    ... )
    """
    print("Normalizing voxels...")
    normdata = np.zeros_like(fmri_data)
    demeandata = np.zeros_like(fmri_data)
    starttime = time.time()
    # detrend if we are going to
    numspatiallocs = fmri_data.shape[0]
    # NB: fmri_data is detrended in place
    if detrendorder > 0:
        print("Detrending to order", detrendorder, "...")
        if mpcode:
            if debug:
                print(f"detrend multiproc path: {detrendorder=}")
            inputshape = fmri_data.shape
            voxelargs = [
                fmri_data,
            ]
            voxelfunc = _procOneVoxelDetrend
            packfunc = _packDetrendvoxeldata
            unpackfunc = _unpackDetrendvoxeldata
            voxelmask = np.zeros_like(fmri_data[:, 0])
            voxelmask[validvoxels] = 1
            voxeltargets = [fmri_data]

            numspatiallocs = tide_genericmultiproc.run_multiproc(
                voxelfunc,
                packfunc,
                unpackfunc,
                voxelargs,
                voxeltargets,
                inputshape,
                voxelmask,
                LGR,
                nprocs,
                alwaysmultiproc,
                showprogressbar,
                chunksize,
                debug=debug,
                detrendorder=detrendorder,
                demean=False,
            )
        else:
            if debug:
                print(f"detrend nonmultiproc path: {detrendorder=}")
            for idx, thevox in enumerate(
                tqdm(
                    validvoxels,
                    desc="Voxel",
                    unit="voxels",
                    disable=(not showprogressbar),
                )
            ):
                fmri_data[thevox, :] = tide_fit.detrend(
                    fmri_data[thevox, :], order=detrendorder, demean=False
                )
            timings.append(["Detrending finished", time.time(), numspatiallocs, "voxels"])
            print(" done")

        timings.append(["Detrending finished", time.time(), numspatiallocs, "voxels"])
        print(" done")

    means = np.mean(fmri_data[:, :], axis=1).flatten()
    demeandata[validvoxels, :] = fmri_data[validvoxels, :] - means[validvoxels, None]
    normdata[validvoxels, :] = np.nan_to_num(demeandata[validvoxels, :] / means[validvoxels, None])
    medians = np.median(normdata[:, :], axis=1).flatten()
    mads = mad(normdata[:, :], axis=1).flatten()
    timings.append(["Normalization finished", time.time(), numspatiallocs, "voxels"])
    print("Normalization took", "{:.3f}".format(time.time() - starttime), "seconds")
    return normdata, demeandata, means, medians, mads


def cleanphysio(
    Fs: float,
    physiowaveform: NDArray,
    cutoff: float = 0.4,
    thresh: float = 0.2,
    nyquist: float | None = None,
    iscardiac: bool = True,
    debug: bool = False,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """
    Apply filtering and normalization to a physiological waveform to extract a cleaned signal and envelope.

    This function performs bandpass filtering on a physiological signal to detect its envelope,
    then applies high-pass filtering to remove baseline drift. The waveform is normalized using
    the envelope to produce a cleaned and standardized signal.

    Parameters
    ----------
    Fs : float
        Sampling frequency of the input waveform in Hz.
    physiowaveform : NDArray
        Input physiological waveform signal (1D array).
    cutoff : float, optional
        Cutoff frequency for envelope detection, by default 0.4.
    thresh : float, optional
        Threshold for envelope normalization, by default 0.2.
    nyquist : float, optional
        Nyquist frequency to constrain the high-pass filter, by default None.
    iscardiac : bool, optional
        Flag indicating if the signal is cardiac; affects filter type, by default True.
    debug : bool, optional
        If True, print debug information during processing, by default False.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, float]
        A tuple containing:
        - `filtphysiowaveform`: The high-pass filtered waveform.
        - `normphysio`: The normalized waveform using the envelope.
        - `envelope`: The detected envelope of the signal.
        - `envmean`: The mean of the envelope.

    Notes
    -----
    - The function uses `tide_filt.NoncausalFilter` for filtering and `tide_math.envdetect` for envelope detection.
    - The waveform is normalized using median absolute deviation (MAD) normalization.
    - The envelope is thresholded to avoid very low values during normalization.

    Examples
    --------
    >>> import numpy as np
    >>> Fs = 100.0
    >>> signal = np.random.randn(1000)
    >>> filtered, normalized, env, env_mean = cleanphysio(Fs, signal)
    """
    # first bandpass the cardiac signal to calculate the envelope
    if debug:
        print("Entering cleanphysio")

    print("Filtering")
    physiofilter = tide_filt.NoncausalFilter("cardiac", debug=debug)

    print("Envelope detection")
    envelope = tide_math.envdetect(
        Fs,
        tide_math.madnormalize(physiofilter.apply(Fs, tide_math.madnormalize(physiowaveform)[0]))[
            0
        ],
        cutoff=cutoff,
    )
    envmean = np.mean(envelope)

    # now patch the envelope function to eliminate very low values
    envlowerlim = thresh * np.max(envelope)
    envelope = np.where(envelope >= envlowerlim, envelope, envlowerlim)

    # now high pass the waveform to eliminate baseline
    arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop = physiofilter.getfreqs()
    physiofilter.settype("arb")
    arb_upper = 10.0
    arb_upperstop = arb_upper * 1.1
    if nyquist is not None:
        if nyquist < arb_upper:
            arb_upper = nyquist
            arb_upperstop = nyquist
    physiofilter.setfreqs(arb_lowerstop, arb_lowerpass, arb_upperpass, arb_upperstop)
    filtphysiowaveform = tide_math.madnormalize(
        physiofilter.apply(Fs, tide_math.madnormalize(physiowaveform)[0])
    )[0]
    print("Normalizing")
    normphysio = tide_math.madnormalize(envmean * filtphysiowaveform / envelope)[0]

    # return the filtered waveform, the normalized waveform, and the envelope
    if debug:
        print("Leaving cleanphysio")
    return filtphysiowaveform, normphysio, envelope, envmean


def findbadpts(
    thewaveform: NDArray,
    nameroot: str,
    outputroot: str,
    samplerate: float,
    infodict: dict,
    thetype: str = "mad",
    retainthresh: float = 0.89,
    mingap: float = 2.0,
    outputlevel: int = 0,
    debug: bool = True,
) -> tuple[NDArray, float | tuple[float, float]]:
    """
    Identify bad points in a waveform based on statistical thresholding and gap filling.

    This function detects outliers in a waveform using either the Median Absolute Deviation (MAD)
    or a fractional value-based method. It then applies gap-filling logic to merge short
    sequences of bad points into longer ones, based on a minimum gap threshold.

    Parameters
    ----------
    thewaveform : NDArray
        Input waveform data as a 1D numpy array.
    nameroot : str
        Root name used for labeling output files and dictionary keys.
    outputroot : str
        Root path for writing output files if `outputlevel > 0`.
    samplerate : float
        Sampling rate of the waveform in Hz.
    infodict : dict
        Dictionary to store metadata about the thresholding method and value.
    thetype : str, optional
        Thresholding method to use. Options are:
        - "mad" (default): Uses Median Absolute Deviation.
        - "fracval": Uses percentile-based thresholds.
    retainthresh : float, optional
        Threshold for retaining data, between 0 and 1. Default is 0.89.
    mingap : float, optional
        Minimum gap (in seconds) to consider for merging bad point streaks. Default is 2.0.
    outputlevel : int, optional
        Level of output verbosity. If > 0, writes bad point vector to file. Default is 0.
    debug : bool, optional
        If True, prints debug information. Default is True.

    Returns
    -------
    tuple[NDArray, float | tuple[float, float]]
        A tuple containing:
        - `thebadpts`: A 1D numpy array of the same length as `thewaveform`, with 1.0 for bad points and 0.0 for good.
        - `thresh`: The calculated threshold value(s) used for bad point detection.
          - If `thetype == "mad"`, `thresh` is a float.
          - If `thetype == "fracval"`, `thresh` is a tuple of (lower_threshold, upper_threshold).

    Notes
    -----
    - The "mad" method uses the median and MAD to compute a sigma-based threshold.
    - The "fracval" method uses percentiles to define a range and marks values outside
      that range as bad.
    - Gap-filling logic merges bad point streaks that are closer than `mingap` seconds.

    Examples
    --------
    >>> import numpy as np
    >>> waveform = np.random.normal(0, 1, 1000)
    >>> info = {}
    >>> badpts, threshold = findbadpts(waveform, "test", "/tmp", 100.0, info, thetype="mad")
    >>> print(f"Threshold used: {threshold}")
    """
    # if thetype == 'triangle' or thetype == 'mad':
    if thetype == "mad":
        absdev = np.fabs(thewaveform - np.median(thewaveform))
        # if thetype == 'triangle':
        #    thresh = threshold_triangle(np.reshape(absdev, (len(absdev), 1)))
        medianval = np.median(thewaveform)
        sigma = mad(thewaveform, center=medianval)
        numsigma = np.sqrt(1.0 / (1.0 - retainthresh))
        thresh = numsigma * sigma
        thebadpts = np.where(absdev >= thresh, 1.0, 0.0)
        print(
            "Bad point threshold set to",
            "{:.3f}".format(thresh),
            "using the",
            thetype,
            "method for",
            nameroot,
        )
    elif thetype == "fracval":
        lower, upper = tide_stats.getfracvals(
            thewaveform,
            [(1.0 - retainthresh) / 2.0, (1.0 + retainthresh) / 2.0],
        )
        therange = upper - lower
        lowerthresh = lower - therange
        upperthresh = upper + therange
        thebadpts = np.where((lowerthresh <= thewaveform) & (thewaveform <= upperthresh), 0.0, 1.0)
        thresh = (lowerthresh, upperthresh)
        print(
            "Values outside of ",
            "{:.3f}".format(lowerthresh),
            "to",
            "{:.3f}".format(upperthresh),
            "marked as bad using the",
            thetype,
            "method for",
            nameroot,
        )
    else:
        raise ValueError("findbadpts error: Bad thresholding type")

    # now fill in gaps
    streakthresh = int(np.round(mingap * samplerate))
    lastbad = 0
    if thebadpts[0] == 1.0:
        isbad = True
    else:
        isbad = False
    for i in range(1, len(thebadpts)):
        if thebadpts[i] == 1.0:
            if not isbad:
                # streak begins
                isbad = True
                if i - lastbad < streakthresh:
                    thebadpts[lastbad:i] = 1.0
            lastbad = i
        else:
            isbad = False
    if len(thebadpts) - lastbad - 1 < streakthresh:
        thebadpts[lastbad:] = 1.0

    if outputlevel > 0:
        tide_io.writevec(thebadpts, outputroot + "_" + nameroot + "_badpts.txt")
    infodict[nameroot + "_threshvalue"] = thresh
    infodict[nameroot + "_threshmethod"] = thetype
    return thebadpts


def approximateentropy(waveform: NDArray, m: int, r: float) -> float:
    """
    Calculate the approximate entropy of a waveform.

    Approximate entropy is a measure of the complexity or irregularity of a time series.
    It quantifies the likelihood that similar patterns of observations will not be followed
    by additional similar observations.

    Parameters
    ----------
    waveform : array_like
        Input time series data as a 1D array or list of numerical values.
    m : int
        Length of compared run of data. Must be a positive integer.
    r : float
        Tolerance parameter. Defines the maximum difference between values to be considered
        similar. Should be a positive number, typically set to 0.1-0.2 times the standard
        deviation of the data.

    Returns
    -------
    float
        Approximate entropy value. Lower values indicate more regularity in the data,
        while higher values indicate more complexity or randomness.

    Notes
    -----
    The approximate entropy is calculated using the method described by Pincus (1991).
    The algorithm computes the logarithm of the ratio of the number of similar patterns
    of length m to those of length m+1, averaged over all possible patterns.

    This implementation assumes that the input waveform is a 1D array of numerical values.
    The function is sensitive to the choice of parameters m and r, and results may vary
    depending on the data characteristics.

    Examples
    --------
    >>> import numpy as np
    >>> waveform = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    >>> apen = approximateentropy(waveform, m=2, r=0.1)
    >>> print(apen)
    0.123456789

    >>> # For a more complex signal
    >>> np.random.seed(42)
    >>> noisy_signal = np.random.randn(100)
    >>> apen_noisy = approximateentropy(noisy_signal, m=2, r=0.1)
    >>> print(apen_noisy)
    0.456789123
    """

    def _maxdist(x_i, x_j):
        """
        Calculate the maximum absolute difference between corresponding elements of two sequences.

        Parameters
        ----------
        x_i : array-like
            First sequence of numbers.
        x_j : array-like
            Second sequence of numbers.

        Returns
        -------
        float
            The maximum absolute difference between corresponding elements of x_i and x_j.

        Notes
        -----
        This function computes the Chebyshev distance (also known as the maximum metric) between two vectors.
        Both sequences must have the same length, otherwise the function will raise a ValueError.

        Examples
        --------
        >>> _maxdist([1, 2, 3], [4, 1, 2])
        3
        >>> _maxdist([0, 0], [1, 1])
        1
        """
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        """
        Calculate phi value for approximate entropy calculation.

        Parameters
        ----------
        m : int
            Length of template vectors for comparison.

        Returns
        -------
        float
            Phi value representing the approximate entropy.

        Notes
        -----
        This function computes the phi value used in approximate entropy calculations.
        It compares template vectors of length m and calculates the proportion of
        vectors that are within a tolerance threshold r of each other.

        Examples
        --------
        >>> _phi(2)
        0.5703489003472879
        """
        x = [[waveform[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(waveform)

    return abs(_phi(m + 1) - _phi(m))


def summarizerun(theinfodict: dict, getkeys: bool = False) -> str:
    """
    Summarize physiological signal quality metrics from a dictionary.

    This function extracts specific signal quality indices from a dictionary
    containing physiological monitoring data. It can either return the metric
    values or the corresponding keys depending on the getkeys parameter.

    Parameters
    ----------
    theinfodict : dict
        Dictionary containing physiological signal quality metrics with keys
        including 'corrcoeff_raw2pleth', 'corrcoeff_filt2pleth', 'E_sqi_mean_pleth',
        'E_sqi_mean_bold', 'S_sqi_mean_pleth', 'S_sqi_mean_bold', 'K_sqi_mean_pleth',
        and 'K_sqi_mean_bold'.
    getkeys : bool, optional
        If True, returns a comma-separated string of all metric keys.
        If False (default), returns a comma-separated string of metric values
        corresponding to the keys in the dictionary. If a key is missing, an
        empty string is returned for that position.

    Returns
    -------
    str
        If getkeys=True: comma-separated string of all metric keys.
        If getkeys=False: comma-separated string of metric values from the dictionary,
        with empty strings for missing keys.

    Notes
    -----
    The function handles missing keys gracefully by returning empty strings
    for missing metrics rather than raising exceptions.

    Examples
    --------
    >>> data = {
    ...     "corrcoeff_raw2pleth": 0.85,
    ...     "E_sqi_mean_pleth": 0.92
    ... }
    >>> summarizerun(data)
    '0.85,,0.92,,,,,'

    >>> summarizerun(data, getkeys=True)
    'corrcoeff_raw2pleth,corrcoeff_filt2pleth,E_sqi_mean_pleth,E_sqi_mean_bold,S_sqi_mean_pleth,S_sqi_mean_bold,K_sqi_mean_pleth,K_sqi_mean_bold'
    """
    keylist = [
        "corrcoeff_raw2pleth",
        "corrcoeff_filt2pleth",
        "E_sqi_mean_pleth",
        "E_sqi_mean_bold",
        "S_sqi_mean_pleth",
        "S_sqi_mean_bold",
        "K_sqi_mean_pleth",
        "K_sqi_mean_bold",
    ]
    if getkeys:
        return ",".join(keylist)
    else:
        outputline = []
        for thekey in keylist:
            try:
                outputline.append(str(theinfodict[thekey]))
            except KeyError:
                outputline.append("")
        return ",".join(outputline)


def entropy(waveform: NDArray) -> float:
    """
    Calculate the entropy of a waveform.

    Parameters
    ----------
    waveform : array-like
        Input waveform data. Should be a numeric array-like object containing
        the waveform samples.

    Returns
    -------
    float
        The entropy value of the waveform, computed as -∑(x² * log₂(x²)) where
        x represents the waveform samples.

    Notes
    -----
    This function computes the entropy using the formula -∑(x² * log₂(x²)),
    where x² represents the squared waveform values. The np.nan_to_num function
    is used to handle potential NaN values in the logarithm calculation.

    Examples
    --------
    >>> import numpy as np
    >>> waveform = np.array([0.5, 0.5, 0.5, 0.5])
    >>> entropy(waveform)
    0.0
    """
    return -np.sum(np.square(waveform) * np.nan_to_num(np.log2(np.square(waveform))))


def calcplethquality(
    waveform: NDArray,
    Fs: float,
    infodict: dict,
    suffix: str,
    outputroot: str,
    S_windowsecs: float = 5.0,
    K_windowsecs: float = 60.0,
    E_windowsecs: float = 1.0,
    detrendorder: int = 8,
    outputlevel: int = 0,
    initfile: bool = True,
    debug: bool = False,
) -> None:
    """
    Calculate windowed skewness, kurtosis, and entropy quality metrics for a plethysmogram.

    This function computes three quality metrics — skewness (S), kurtosis (K), and entropy (E) —
    over sliding windows of the input waveform. These metrics are used to assess the quality
    of photoplethysmogram (PPG) signals based on the method described in Elgendi (2016).

    Parameters
    ----------
    waveform : array-like
        The cardiac waveform to be assessed.
    Fs : float
        The sample rate of the data in Hz.
    infodict : dict
        Dictionary to store computed quality metrics.
    suffix : str
        Suffix to append to metric keys in `infodict`.
    outputroot : str
        Root name for output files if `outputlevel > 1`.
    S_windowsecs : float, optional
        Skewness window duration in seconds. Default is 5.0 seconds.
    K_windowsecs : float, optional
        Kurtosis window duration in seconds. Default is 60.0 seconds.
    E_windowsecs : float, optional
        Entropy window duration in seconds. Default is 1.0 seconds.
    detrendorder : int, optional
        Order of the detrending polynomial applied to the plethysmogram. Default is 8.
    outputlevel : int, optional
        Level of output verbosity. If > 1, time-series data will be written to files.
    initfile : bool, optional
        Whether to initialize output files. Default is True.
    debug : bool, optional
        If True, print debug information. Default is False.

    Returns
    -------
    None
        All generated values are returned in infodict
    tuple
        A tuple containing the following elements in order:

        - S_sqi_mean : float
            Mean value of the skewness quality index over all time.
        - S_sqi_std : float
            Standard deviation of the skewness quality index over all time.
        - S_waveform : array
            The skewness quality metric over all timepoints.
        - K_sqi_mean : float
            Mean value of the kurtosis quality index over all time.
        - K_sqi_std : float
            Standard deviation of the kurtosis quality index over all time.
        - K_waveform : array
            The kurtosis quality metric over all timepoints.
        - E_sqi_mean : float
            Mean value of the entropy quality index over all time.
        - E_sqi_std : float
            Standard deviation of the entropy quality index over all time.
        - E_waveform : array
            The entropy quality metric over all timepoints.

    Notes
    -----
    The function applies a detrending polynomial to the input waveform before computing
    the quality metrics. Window sizes are rounded to the nearest odd number of samples
    to ensure symmetric windows.

    The following values are put into infodict:
        - S_sqi_mean : float
            Mean value of the skewness quality index over all time.
        - S_sqi_std : float
            Standard deviation of the skewness quality index over all time.
        - S_waveform : array
            The skewness quality metric over all timepoints.
        - K_sqi_mean : float
            Mean value of the kurtosis quality index over all time.
        - K_sqi_std : float
            Standard deviation of the kurtosis quality index over all time.
        - K_waveform : array
            The kurtosis quality metric over all timepoints.
        - E_sqi_mean : float
            Mean value of the entropy quality index over all time.
        - E_sqi_std : float
            Standard deviation of the entropy quality index over all time.
        - E_waveform : array
            The entropy quality metric over all timepoints.

    References
    ----------
    Elgendi, M. "Optimal Signal Quality Index for Photoplethysmogram Signals".
    Bioengineering 2016, Vol. 3, Page 21 (2016).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import skew, kurtosis
    >>> waveform = np.random.randn(1000)
    >>> Fs = 100.0
    >>> infodict = {}
    >>> suffix = "_test"
    >>> outputroot = "test_output"
    >>> S_mean, S_std, S_wave, K_mean, K_std, K_wave, E_mean, E_std, E_wave = calcplethquality(
    ...     waveform, Fs, infodict, suffix, outputroot
    ... )
    """
    # detrend the waveform
    dt_waveform = tide_fit.detrend(waveform, order=detrendorder, demean=True)

    # calculate S_sqi and K_sqi over a sliding window.  Window size should be an odd number of points.
    S_windowpts = int(np.round(S_windowsecs * Fs, 0))
    S_windowpts += 1 - S_windowpts % 2
    S_waveform = np.zeros_like(dt_waveform)
    K_windowpts = int(np.round(K_windowsecs * Fs, 0))
    K_windowpts += 1 - K_windowpts % 2
    K_waveform = np.zeros_like(dt_waveform)
    E_windowpts = int(np.round(E_windowsecs * Fs, 0))
    E_windowpts += 1 - E_windowpts % 2
    E_waveform = np.zeros_like(dt_waveform)

    if debug:
        print("S_windowsecs, S_windowpts:", S_windowsecs, S_windowpts)
        print("K_windowsecs, K_windowpts:", K_windowsecs, K_windowpts)
        print("E_windowsecs, E_windowpts:", E_windowsecs, E_windowpts)
    for i in range(0, len(dt_waveform)):
        startpt = np.max([0, i - S_windowpts // 2])
        endpt = np.min([i + S_windowpts // 2, len(dt_waveform)])
        S_waveform[i] = skew(dt_waveform[startpt : endpt + 1], nan_policy="omit")

        startpt = np.max([0, i - K_windowpts // 2])
        endpt = np.min([i + K_windowpts // 2, len(dt_waveform)])
        K_waveform[i] = kurtosis(dt_waveform[startpt : endpt + 1], fisher=False)

        startpt = np.max([0, i - E_windowpts // 2])
        endpt = np.min([i + E_windowpts // 2, len(dt_waveform)])
        # E_waveform[i] = entropy(dt_waveform[startpt:endpt + 1])
        r = 0.2 * np.std(dt_waveform[startpt : endpt + 1])
        E_waveform[i] = approximateentropy(dt_waveform[startpt : endpt + 1], 2, r)

    S_sqi_mean = np.mean(S_waveform)
    S_sqi_median = np.median(S_waveform)
    S_sqi_std = np.std(S_waveform)
    K_sqi_mean = np.mean(K_waveform)
    K_sqi_median = np.median(K_waveform)
    K_sqi_std = np.std(K_waveform)
    E_sqi_mean = np.mean(E_waveform)
    E_sqi_median = np.median(E_waveform)
    E_sqi_std = np.std(E_waveform)

    infodict["S_sqi_mean" + suffix] = S_sqi_mean
    infodict["S_sqi_median" + suffix] = S_sqi_median
    infodict["S_sqi_std" + suffix] = S_sqi_std
    infodict["K_sqi_mean" + suffix] = K_sqi_mean
    infodict["K_sqi_median" + suffix] = K_sqi_median
    infodict["K_sqi_std" + suffix] = K_sqi_std
    infodict["E_sqi_mean" + suffix] = E_sqi_mean
    infodict["E_sqi_median" + suffix] = E_sqi_median
    infodict["E_sqi_std" + suffix] = E_sqi_std

    if outputlevel > 1:
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            S_waveform,
            Fs,
            columns=["S_sqi" + suffix],
            append=(not initfile),
            debug=debug,
        )
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            K_waveform,
            Fs,
            columns=["K_sqi" + suffix],
            append=True,
            debug=debug,
        )
        tide_io.writebidstsv(
            outputroot + "_desc-qualitymetrics" + str(Fs) + "Hz_timeseries",
            E_waveform,
            Fs,
            columns=["E_sqi" + suffix],
            append=True,
            debug=debug,
        )


def getphysiofile(
    waveformfile: str,
    inputfreq: float,
    inputstart: float | None,
    slicetimeaxis: NDArray,
    stdfreq: float,
    stdpoints: int,
    envcutoff: float,
    envthresh: float,
    timings: list,
    outputroot: str,
    slop: float = 0.25,
    outputlevel: int = 0,
    iscardiac: bool = True,
    debug: bool = False,
) -> tuple[NDArray, NDArray, float, int]:
    """
    Read, process, and resample physiological waveform data.

    This function reads a physiological signal from a text file, filters and normalizes
    the signal, and resamples it to both slice-specific and standard time resolutions.
    It supports cardiac and non-cardiac signal processing, with optional debugging and
    output writing.

    Parameters
    ----------
    waveformfile : str
        Path to the input physiological waveform file.
    inputfreq : float
        Sampling frequency of the input waveform. If negative, the frequency is
        inferred from the file.
    inputstart : float or None
        Start time of the input waveform. If None, defaults to 0.0.
    slicetimeaxis : array_like
        Time axis corresponding to slice acquisition times.
    stdfreq : float
        Standard sampling frequency for resampling.
    stdpoints : int
        Number of points for the standard time axis.
    envcutoff : float
        Cutoff frequency for envelope filtering.
    envthresh : float
        Threshold for envelope normalization.
    timings : list
        List to append timing information for logging.
    outputroot : str
        Root name for output files.
    slop : float, optional
        Tolerance for time alignment check (default is 0.25).
    outputlevel : int, optional
        Level of output writing (default is 0).
    iscardiac : bool, optional
        Flag indicating if the signal is cardiac (default is True).
    debug : bool, optional
        Enable debug printing (default is False).

    Returns
    -------
    waveform_sliceres : NDArray
        Physiological signal resampled to slice time resolution.
    waveform_stdres : NDArray
        Physiological signal resampled to standard time resolution.
    inputfreq : float
        The actual input sampling frequency used.
    len(waveform_fullres) : int
        Length of the original waveform data.

    Notes
    -----
    - The function reads the waveform file using `tide_io.readvectorsfromtextfile`.
    - Signal filtering and normalization are performed using `cleanphysio`.
    - Resampling is done using `tide_resample.doresample`.
    - If `iscardiac` is True, raw and cleaned signals are saved to files when `outputlevel > 1`.

    Examples
    --------
    >>> waveform_sliceres, waveform_stdres, freq, length = getphysiofile(
    ...     waveformfile="physio.txt",
    ...     inputfreq=100.0,
    ...     inputstart=0.0,
    ...     slicetimeaxis=np.linspace(0, 10, 50),
    ...     stdfreq=25.0,
    ...     stdpoints=100,
    ...     envcutoff=0.5,
    ...     envthresh=0.1,
    ...     timings=[],
    ...     outputroot="output",
    ...     debug=False
    ... )
    """
    if debug:
        print("Entering getphysiofile")
    print("Reading physiological signal from file")

    # check file type
    filefreq, filestart, dummy, waveform_fullres, dummy, dummy = tide_io.readvectorsfromtextfile(
        waveformfile, onecol=True, debug=debug
    )
    if inputfreq < 0.0:
        if filefreq is not None:
            inputfreq = filefreq
        else:
            inputfreq = -inputfreq

    if inputstart is None:
        if filestart is not None:
            inputstart = filestart
        else:
            inputstart = 0.0

    if debug:
        print("inputfreq:", inputfreq)
        print("inputstart:", inputstart)
        print("waveform_fullres:", waveform_fullres)
        print("waveform_fullres.shape:", waveform_fullres.shape)
    inputtimeaxis = (
        np.linspace(
            0.0,
            (1.0 / inputfreq) * len(waveform_fullres),
            num=len(waveform_fullres),
            endpoint=False,
        )
        + inputstart
    )
    stdtimeaxis = (
        np.linspace(0.0, (1.0 / stdfreq) * stdpoints, num=stdpoints, endpoint=False) + inputstart
    )

    if debug:
        print("getphysiofile: input time axis start, stop, step, freq, length")
        print(
            inputtimeaxis[0],
            inputtimeaxis[-1],
            inputtimeaxis[1] - inputtimeaxis[0],
            1.0 / (inputtimeaxis[1] - inputtimeaxis[0]),
            len(inputtimeaxis),
        )
        print("getphysiofile: slice time axis start, stop, step, freq, length")
        print(
            slicetimeaxis[0],
            slicetimeaxis[-1],
            slicetimeaxis[1] - slicetimeaxis[0],
            1.0 / (slicetimeaxis[1] - slicetimeaxis[0]),
            len(slicetimeaxis),
        )
    if (inputtimeaxis[0] > slop) or (inputtimeaxis[-1] < slicetimeaxis[-1] - slop):
        print("\tinputtimeaxis[0]:", inputtimeaxis[0])
        print("\tinputtimeaxis[-1]:", inputtimeaxis[-1])
        print("\tslicetimeaxis[0]:", slicetimeaxis[0])
        print("\tslicetimeaxis[-1]:", slicetimeaxis[-1])
        if inputtimeaxis[0] > slop:
            print("\tfailed condition 1:", inputtimeaxis[0], ">", slop)
        if inputtimeaxis[-1] < slicetimeaxis[-1] - slop:
            print(
                "\tfailed condition 2:",
                inputtimeaxis[-1],
                "<",
                slicetimeaxis[-1] - slop,
            )
        raise ValueError("getphysiofile: error - waveform file does not cover the fmri time range")
    if debug:
        print("waveform_fullres: len=", len(waveform_fullres), "vals=", waveform_fullres)
        print("inputfreq =", inputfreq)
        print("inputstart =", inputstart)
        print("inputtimeaxis: len=", len(inputtimeaxis), "vals=", inputtimeaxis)
    timings.append(["Cardiac signal from physiology data read in", time.time(), None, None])

    # filter and amplitude correct the waveform to remove gain fluctuations
    cleanwaveform_fullres, normwaveform_fullres, waveformenv_fullres, envmean = cleanphysio(
        inputfreq,
        waveform_fullres,
        iscardiac=iscardiac,
        cutoff=envcutoff,
        thresh=envthresh,
        nyquist=inputfreq / 2.0,
        debug=debug,
    )

    if iscardiac:
        if outputlevel > 1:
            tide_io.writevec(waveform_fullres, outputroot + "_rawpleth_native.txt")
            tide_io.writevec(cleanwaveform_fullres, outputroot + "_pleth_native.txt")
            tide_io.writevec(waveformenv_fullres, outputroot + "_cardenvelopefromfile_native.txt")
        timings.append(["Cardiac signal from physiology data cleaned", time.time(), None, None])

    # resample to slice time resolution and save
    waveform_sliceres = tide_resample.doresample(
        inputtimeaxis, cleanwaveform_fullres, slicetimeaxis, method="univariate", padlen=0
    )

    # resample to standard resolution and save
    waveform_stdres = tide_math.madnormalize(
        tide_resample.doresample(
            inputtimeaxis,
            cleanwaveform_fullres,
            stdtimeaxis,
            method="univariate",
            padlen=0,
        )
    )[0]

    timings.append(
        [
            "Cardiac signal from physiology data resampled to slice resolution and saved",
            time.time(),
            None,
            None,
        ]
    )

    if debug:
        print("Leaving getphysiofile")
    return waveform_sliceres, waveform_stdres, inputfreq, len(waveform_fullres)


def readextmask(
    thefilename: str,
    nim_hdr: dict,
    xsize: int,
    ysize: int,
    numslices: int,
    debug: bool = False,
) -> NDArray:
    """
    Read and validate external mask from NIfTI file.

    This function reads a mask from a NIfTI file and performs validation checks
    to ensure compatibility with the input fMRI data dimensions. The mask must
    have exactly 3 dimensions and match the spatial dimensions of the fMRI data.

    Parameters
    ----------
    thefilename : str
        Path to the NIfTI file containing the mask
    nim_hdr : dict
        Header information from the fMRI data
    xsize : int
        X dimension size of the fMRI data
    ysize : int
        Y dimension size of the fMRI data
    numslices : int
        Number of slices in the fMRI data
    debug : bool, optional
        If True, print debug information about mask dimensions (default is False)

    Returns
    -------
    NDArray
        The mask data array with shape (xsize, ysize, numslices)

    Raises
    ------
    ValueError
        If mask dimensions do not match fMRI data dimensions or if mask has
        more than 3 dimensions

    Notes
    -----
    The function performs the following validation checks:
    1. Reads mask from NIfTI file using tide_io.readfromnifti
    2. Parses NIfTI dimensions using tide_io.parseniftidims
    3. Validates that mask spatial dimensions match fMRI data dimensions
    4. Ensures mask has exactly 3 dimensions (no time dimension allowed)

    Examples
    --------
    >>> import numpy as np
    >>> mask_data = readextmask('mask.nii', fmri_header, 64, 64, 30)
    >>> print(mask_data.shape)
    (64, 64, 30)
    """
    (
        extmask,
        extmask_data,
        extmask_hdr,
        theextmaskdims,
        theextmasksizes,
    ) = tide_io.readfromnifti(thefilename)
    (
        xsize_extmask,
        ysize_extmask,
        numslices_extmask,
        timepoints_extmask,
    ) = tide_io.parseniftidims(theextmaskdims)
    if debug:
        print(
            f"Mask dimensions: {xsize_extmask}, {ysize_extmask}, {numslices_extmask}, {timepoints_extmask}"
        )
    if not tide_io.checkspacematch(nim_hdr, extmask_hdr):
        raise ValueError("Dimensions of mask do not match the fmri data - exiting")
    if timepoints_extmask > 1:
        raise ValueError("Mask must have only 3 dimensions - exiting")
    return extmask_data


def checkcardmatch(
    reference: NDArray,
    candidate: NDArray,
    samplerate: float,
    refine: bool = True,
    zeropadding: int = 0,
    debug: bool = False,
) -> tuple[float, float, str]:
    """
    Compare two cardiac waveforms using cross-correlation and peak fitting.

    This function performs a cross-correlation between a reference and a candidate
    cardiac waveform after applying a non-causal cardiac filter. It then fits a
    Gaussian to the cross-correlation peak to estimate the time delay and
    correlation strength.

    Parameters
    ----------
    reference : 1D numpy array
        The cardiac waveform to compare to.
    candidate : 1D numpy array
        The cardiac waveform to be assessed.
    samplerate : float
        The sample rate of the data in Hz.
    refine : bool, optional
        Whether to refine the peak fit. Default is True.
    zeropadding : int, optional
        Specify the length of correlation padding to use. Default is 0.
    debug : bool, optional
        Output additional information for debugging. Default is False.

    Returns
    -------
    maxval : float
        The maximum value of the crosscorrelation function.
    maxdelay : float
        The time, in seconds, where the maximum crosscorrelation occurs.
    failreason : int
        Reason why the fit failed (0 if no failure).

    Notes
    -----
    The function applies a cardiac filter to both waveforms before computing
    the cross-correlation. A Gaussian fit is used to estimate the peak location
    and strength within a predefined search range of ±2 seconds around the
    initial peak.

    Examples
    --------
    >>> import numpy as np
    >>> reference = np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, 1000))
    >>> candidate = np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, 1000) + 0.1)
    >>> maxval, maxdelay, failreason = checkcardmatch(reference, candidate, 100)
    >>> print(f"Max correlation: {maxval}, Delay: {maxdelay}s")
    """
    thecardfilt = tide_filt.NoncausalFilter(filtertype="cardiac")
    trimlength = np.min([len(reference), len(candidate)])
    thexcorr = tide_corr.fastcorrelate(
        tide_math.corrnormalize(
            thecardfilt.apply(samplerate, reference),
            detrendorder=3,
            windowfunc="hamming",
        )[:trimlength],
        tide_math.corrnormalize(
            thecardfilt.apply(samplerate, candidate),
            detrendorder=3,
            windowfunc="hamming",
        )[:trimlength],
        usefft=True,
        zeropadding=zeropadding,
    )
    xcorrlen = len(thexcorr)
    sampletime = 1.0 / samplerate
    xcorr_x = np.r_[0.0:xcorrlen] * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    searchrange = 5.0
    trimstart = tide_util.valtoindex(xcorr_x, -2.0 * searchrange)
    trimend = tide_util.valtoindex(xcorr_x, 2.0 * searchrange)
    (
        maxindex,
        maxdelay,
        maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    ) = tide_fit.findmaxlag_gauss(
        xcorr_x[trimstart:trimend],
        thexcorr[trimstart:trimend],
        -searchrange,
        searchrange,
        3.0,
        refine=refine,
        zerooutbadfit=False,
        useguess=False,
        fastgauss=False,
        displayplots=False,
    )
    if debug:
        print(
            "CORRELATION: maxindex, maxdelay, maxval, maxsigma, maskval, failreason, peakstart, peakend:",
            maxindex,
            maxdelay,
            maxval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        )
    return maxval, maxdelay, failreason


def cardiaccycleaverage(
    sourcephases: NDArray,
    destinationphases: NDArray,
    waveform: NDArray,
    procpoints: int,
    congridbins: int,
    gridkernel: str,
    centric: bool,
    cache: bool = True,
    cyclic: bool = True,
) -> NDArray:
    """
    Compute the average waveform over a cardiac cycle using phase-based resampling.

    This function performs phase-resolved averaging of a waveform signal over a
    cardiac cycle. It uses a resampling technique to map source phase values to
    destination phases, accumulating weighted contributions to produce an averaged
    waveform. The result is normalized and adjusted to remove artifacts from low
    weight regions.

    Parameters
    ----------
    sourcephases : array-like
        Array of source phase values (in radians) corresponding to the waveform data.
    destinationphases : array-like
        Array of destination phase values (in radians) where the averaged waveform
        will be computed.
    waveform : array-like
        Array of waveform values to be averaged.
    procpoints : array-like
        Array of indices indicating which points in `waveform` and `sourcephases`
        should be processed.
    congridbins : int
        Number of bins used in the resampling process.
    gridkernel : callable
        Kernel function used for interpolation during resampling.
    centric : bool
        If True, phase values are treated as centric (e.g., centered around 0).
        If False, phase values are treated as cyclic (e.g., 0 to 2π).
    cache : bool, optional
        If True, use cached results for repeated computations (default is True).
    cyclic : bool, optional
        If True, treat phase values as cyclic (default is True).

    Returns
    -------
    tuple of ndarray
        A tuple containing:
        - `rawapp_bypoint`: The normalized averaged waveform values for each
          destination phase.
        - `weight_bypoint`: The total weight for each destination phase.

    Notes
    -----
    The function applies a threshold to weights: only points with weights greater
    than 1/50th of the maximum weight are considered valid. These points are then
    normalized and shifted to start from zero.

    Examples
    --------
    >>> import numpy as np
    >>> sourcephases = np.linspace(0, 2*np.pi, 100)
    >>> destinationphases = np.linspace(0, 2*np.pi, 50)
    >>> waveform = np.sin(sourcephases)
    >>> procpoints = np.arange(100)
    >>> congridbins = 10
    >>> gridkernel = lambda x: np.exp(-x**2 / 2)
    >>> centric = False
    >>> avg_waveform, weights = cardiaccycleaverage(
    ...     sourcephases, destinationphases, waveform, procpoints,
    ...     congridbins, gridkernel, centric
    ... )
    """
    rawapp_bypoint = np.zeros(len(destinationphases), dtype=np.float64)
    weight_bypoint = np.zeros(len(destinationphases), dtype=np.float64)
    for t in procpoints:
        thevals, theweights, theindices = tide_resample.congrid(
            destinationphases,
            tide_math.phasemod(sourcephases[t], centric=centric),
            1.0,
            congridbins,
            kernel=gridkernel,
            cache=cache,
            cyclic=cyclic,
        )
        for i in range(len(theindices)):
            weight_bypoint[theindices[i]] += theweights[i]
            rawapp_bypoint[theindices[i]] += theweights[i] * waveform[t]
    rawapp_bypoint = np.where(
        weight_bypoint > (np.max(weight_bypoint) / 50.0),
        np.nan_to_num(rawapp_bypoint / weight_bypoint),
        0.0,
    )
    minval = np.min(rawapp_bypoint[np.where(weight_bypoint > np.max(weight_bypoint) / 50.0)])
    rawapp_bypoint = np.where(
        weight_bypoint > np.max(weight_bypoint) / 50.0, rawapp_bypoint - minval, 0.0
    )
    return rawapp_bypoint, weight_bypoint


def circularderivs(timecourse: NDArray) -> tuple[NDArray, float, float]:
    """
    Compute circular first derivatives and their extremal values.

    This function calculates the circular first derivative of a time course,
    which is the difference between consecutive elements with the last element
    wrapped around to the first. It then returns the maximum and minimum values
    of these derivatives along with their indices.

    Parameters
    ----------
    timecourse : array-like
        Input time course data as a 1D array or sequence of numerical values.

    Returns
    -------
    tuple
        A tuple containing four elements:
        - max_derivative : float
            The maximum value of the circular first derivative
        - argmax_index : int
            The index of the maximum derivative value
        - min_derivative : float
            The minimum value of the circular first derivative
        - argmin_index : int
            The index of the minimum derivative value

    Notes
    -----
    The circular first derivative is computed as:
    ``first_deriv[i] = timecourse[i+1] - timecourse[i]`` for i < n-1,
    and ``first_deriv[n-1] = timecourse[0] - timecourse[n-1]``.

    Examples
    --------
    >>> import numpy as np
    >>> timecourse = [1, 2, 3, 2, 1]
    >>> max_val, max_idx, min_val, min_idx = circularderivs(timecourse)
    >>> print(f"Max derivative: {max_val} at index {max_idx}")
    >>> print(f"Min derivative: {min_val} at index {min_idx}")
    """
    firstderiv = np.diff(timecourse, append=[timecourse[0]])
    return (
        np.max(firstderiv),
        np.argmax(firstderiv),
        np.min(firstderiv),
        np.argmin(firstderiv),
    )


def _procOnePhaseProject(slice, sliceargs, **kwargs):
    """
    Process a single phase project for fMRI data resampling and averaging.

    This function performs temporal resampling of fMRI data along the phase dimension
    using a congrid-based interpolation scheme. It updates weight, raw application,
    and cine data arrays based on the resampled values.

    Parameters
    ----------
    slice : int
        The slice index to process.
    sliceargs : tuple
        A tuple containing the following elements:
        - validlocslist : list of arrays
          List of valid location indices for each slice.
        - proctrs : array-like
          Time indices to process.
        - demeandata_byslice : ndarray
          Demeaned fMRI data organized by slice and time.
        - fmri_data_byslice : ndarray
          Raw fMRI data organized by slice and time.
        - outphases : array-like
          Output phase values for resampling.
        - cardphasevals : ndarray
          Cardinality of phase values for each slice and time.
        - congridbins : int
          Number of bins for congrid interpolation.
        - gridkernel : str
          Interpolation kernel to use.
        - weights_byslice : ndarray
          Weight array to be updated.
        - cine_byslice : ndarray
          Cine data array to be updated.
        - destpoints : int
          Number of destination points.
        - rawapp_byslice : ndarray
          Raw application data array to be updated.
    **kwargs : dict
        Additional options to override default settings:
        - cache : bool, optional
          Whether to use caching in congrid (default: True).
        - debug : bool, optional
          Whether to enable debug mode (default: False).

    Returns
    -------
    tuple
        A tuple containing:
        - slice : int
          The input slice index.
        - rawapp_byslice : ndarray
          Updated raw application data for the slice.
        - cine_byslice : ndarray
          Updated cine data for the slice.
        - weights_byslice : ndarray
          Updated weights for the slice.
        - validlocs : array-like
          Valid location indices for the slice.

    Notes
    -----
    This function modifies the input arrays `weights_byslice`, `rawapp_byslice`,
    and `cine_byslice` in-place. The function assumes that the data has already
    been preprocessed and organized into slices and time points.

    Examples
    --------
    >>> slice_idx = 0
    >>> args = (validlocslist, proctrs, demeandata_byslice, fmri_data_byslice,
    ...         outphases, cardphasevals, congridbins, gridkernel,
    ...         weights_byslice, cine_byslice, destpoints, rawapp_byslice)
    >>> result = _procOnePhaseProject(slice_idx, args, cache=False)
    """
    options = {
        "cache": True,
        "debug": False,
    }
    options.update(kwargs)
    cache = options["cache"]
    debug = options["debug"]
    (
        validlocslist,
        proctrs,
        demeandata_byslice,
        fmri_data_byslice,
        outphases,
        cardphasevals,
        congridbins,
        gridkernel,
        weights_byslice,
        cine_byslice,
        destpoints,
        rawapp_byslice,
    ) = sliceargs
    # now smooth the projected data along the time dimension
    validlocs = validlocslist[slice]
    if len(validlocs) > 0:
        for t in proctrs:
            filteredmr = -demeandata_byslice[validlocs, slice, t]
            cinemr = fmri_data_byslice[validlocs, slice, t]
            thevals, theweights, theindices = tide_resample.congrid(
                outphases,
                cardphasevals[slice, t],
                1.0,
                congridbins,
                kernel=gridkernel,
                cache=cache,
                cyclic=True,
            )
            for i in range(len(theindices)):
                weights_byslice[validlocs, slice, theindices[i]] += theweights[i]
                rawapp_byslice[validlocs, slice, theindices[i]] += filteredmr
                cine_byslice[validlocs, slice, theindices[i]] += theweights[i] * cinemr
        for d in range(destpoints):
            if weights_byslice[validlocs[0], slice, d] == 0.0:
                weights_byslice[validlocs, slice, d] = 1.0
        rawapp_byslice[validlocs, slice, :] = np.nan_to_num(
            rawapp_byslice[validlocs, slice, :] / weights_byslice[validlocs, slice, :]
        )
        cine_byslice[validlocs, slice, :] = np.nan_to_num(
            cine_byslice[validlocs, slice, :] / weights_byslice[validlocs, slice, :]
        )
    else:
        rawapp_byslice[:, slice, :] = 0.0
        cine_byslice[:, slice, :] = 0.0

    return (
        slice,
        rawapp_byslice[:, slice, :],
        cine_byslice[:, slice, :],
        weights_byslice[:, slice, :],
        validlocs,
    )


def _packslicedataPhaseProject(slicenum, sliceargs):
    """
    Pack slice data for phase projection.

    This function takes a slice number and slice arguments, then returns a
    flattened list containing all the slice arguments in order.

    Parameters
    ----------
    slicenum : int
        The slice number identifier.
    sliceargs : list or tuple
        Collection of slice arguments to be packed into a flat list.

    Returns
    -------
    list
        A list containing all elements from sliceargs in the same order.

    Notes
    -----
    This function essentially performs a flattening operation on the slice
    arguments, converting them into a fixed-length list format.

    Examples
    --------
    >>> _packslicedataPhaseProject(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    """
    return [
        sliceargs[0],
        sliceargs[1],
        sliceargs[2],
        sliceargs[3],
        sliceargs[4],
        sliceargs[5],
        sliceargs[6],
        sliceargs[7],
        sliceargs[8],
        sliceargs[9],
        sliceargs[10],
        sliceargs[11],
    ]


def _unpackslicedataPhaseProject(retvals, voxelproducts):
    """
    Unpack slice data for phase project operation.

    This function assigns sliced data from retvals to corresponding voxelproducts
    based on index mappings. It performs three simultaneous assignments using
    slicing operations on 3D arrays.

    Parameters
    ----------
    retvals : tuple of array-like
        A tuple containing 5 elements where:
        - retvals[0], retvals[1], retvals[2], retvals[3], retvals[4]
        - retvals[4] is used as row index for slicing
        - retvals[0] is used as column index for slicing
    voxelproducts : list of array-like
        A list of 3 arrays that will be modified in-place with the sliced data.
        Each array is expected to be 3D and will be indexed using retvals[4] and retvals[0].

    Returns
    -------
    None
        This function modifies voxelproducts in-place and does not return any value.

    Notes
    -----
    The function performs three assignments:
    1. voxelproducts[0][retvals[4], retvals[0], :] = retvals[1][retvals[4], :]
    2. voxelproducts[1][retvals[4], retvals[0], :] = retvals[2][retvals[4], :]
    3. voxelproducts[2][retvals[4], retvals[0], :] = retvals[3][retvals[4], :]

    All arrays must be compatible for the specified slicing operations.

    Examples
    --------
    >>> retvals = (np.array([0, 1]), np.array([[1, 2], [3, 4]]),
    ...            np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]]),
    ...            np.array([0, 1]))
    >>> voxelproducts = [np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2))]
    >>> _unpackslicedataPhaseProject(retvals, voxelproducts)
    """
    (voxelproducts[0])[retvals[4], retvals[0], :] = (retvals[1])[retvals[4], :]
    (voxelproducts[1])[retvals[4], retvals[0], :] = (retvals[2])[retvals[4], :]
    (voxelproducts[2])[retvals[4], retvals[0], :] = (retvals[3])[retvals[4], :]


def preloadcongrid(
    outphases: NDArray,
    congridbins: int,
    gridkernel: str = "kaiser",
    cyclic: bool = True,
    debug: bool = False,
) -> None:
    """
    Preload congrid interpolation cache for efficient subsequent calls.

    This function preloads the congrid interpolation cache by performing a series
    of interpolation operations with different phase values. This avoids the
    computational overhead of cache initialization during subsequent calls to
    tide_resample.congrid with the same parameters.

    Parameters
    ----------
    outphases : array-like
        Output phase values for the interpolation grid.
    congridbins : array-like
        Binning parameters for the congrid interpolation.
    gridkernel : str, optional
        Interpolation kernel to use. Default is "kaiser".
    cyclic : bool, optional
        Whether to treat the data as cyclic. Default is True.
    debug : bool, optional
        Enable debug output. Default is False.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function is designed to improve performance when calling tide_resample.congrid
    multiple times with the same parameters. By preloading the cache with various
    phase values, subsequent calls will be faster as the cache is already populated.

    Examples
    --------
    >>> import numpy as np
    >>> outphases = np.linspace(0, 2*np.pi, 100)
    >>> congridbins = [10, 20]
    >>> preloadcongrid(outphases, congridbins, gridkernel="kaiser", cyclic=True)
    """
    outphasestep = outphases[1] - outphases[0]
    outphasecenter = outphases[int(len(outphases) / 2)]
    fillargs = outphasestep * (
        np.linspace(-0.5, 0.5, 10001, endpoint=True, dtype=float) + outphasecenter
    )
    for thearg in fillargs:
        dummy, dummy, dummy = tide_resample.congrid(
            outphases,
            thearg,
            1.0,
            congridbins,
            kernel=gridkernel,
            cyclic=cyclic,
            cache=True,
            debug=debug,
        )


def phaseprojectpass(
    numslices,
    demeandata_byslice,
    fmri_data_byslice,
    validlocslist,
    proctrs,
    weights_byslice,
    cine_byslice,
    rawapp_byslice,
    outphases,
    cardphasevals,
    congridbins,
    gridkernel,
    destpoints,
    mpcode=False,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    cache=True,
    debug=False,
):
    """
    Perform phase-encoding projection for fMRI data across slices.

    This function projects fMRI data onto a set of phase values using congrid
    resampling, accumulating results in `rawapp_byslice` and `cine_byslice` arrays.
    It supports both single-threaded and multi-processed execution.

    Parameters
    ----------
    numslices : int
        Number of slices to process.
    demeandata_byslice : ndarray
        Demeaned fMRI data, shape (nvoxels, nslices, ntr).
    fmri_data_byslice : ndarray
        Raw fMRI data, shape (nvoxels, nslices, ntr).
    validlocslist : list of ndarray
        List of valid voxel indices for each slice.
    proctrs : ndarray
        Timepoints to process.
    weights_byslice : ndarray
        Weight array, shape (nvoxels, nslices, ndestpoints).
    cine_byslice : ndarray
        Cine data array, shape (nvoxels, nslices, ndestpoints).
    rawapp_byslice : ndarray
        Raw application data array, shape (nvoxels, nslices, ndestpoints).
    outphases : ndarray
        Output phase values.
    cardphasevals : ndarray
        Cardinal phase values for each slice and timepoint, shape (nslices, ntr).
    congridbins : int
        Number of bins for congrid resampling.
    gridkernel : str
        Kernel to use for congrid resampling.
    destpoints : int
        Number of destination points.
    mpcode : bool, optional
        If True, use multiprocessing. Default is False.
    nprocs : int, optional
        Number of processes to use if `mpcode` is True. Default is 1.
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for small datasets. Default is False.
    showprogressbar : bool, optional
        If True, show progress bar. Default is True.
    cache : bool, optional
        If True, enable caching for congrid. Default is True.
    debug : bool, optional
        If True, enable debug output. Default is False.

    Returns
    -------
    None
        The function modifies `weights_byslice`, `cine_byslice`, and `rawapp_byslice` in-place.

    Notes
    -----
    This function is typically used in the context of phase-encoded fMRI analysis.
    It applies a congrid-based resampling technique to project data onto a specified
    phase grid, accumulating weighted contributions in the output arrays.

    Examples
    --------
    >>> phaseprojectpass(
    ...     numslices=10,
    ...     demeandata_byslice=demean_data,
    ...     fmri_data_byslice=fmri_data,
    ...     validlocslist=valid_locs_list,
    ...     proctrs=tr_list,
    ...     weights_byslice=weights,
    ...     cine_byslice=cine_data,
    ...     rawapp_byslice=rawapp_data,
    ...     outphases=phase_vals,
    ...     cardphasevals=card_phase_vals,
    ...     congridbins=100,
    ...     gridkernel='gaussian',
    ...     destpoints=50,
    ...     mpcode=False,
    ...     nprocs=4,
    ...     showprogressbar=True,
    ...     cache=True,
    ...     debug=False,
    ... )
    """
    if mpcode:
        inputshape = rawapp_byslice.shape
        sliceargs = [
            validlocslist,
            proctrs,
            demeandata_byslice,
            fmri_data_byslice,
            outphases,
            cardphasevals,
            congridbins,
            gridkernel,
            weights_byslice,
            cine_byslice,
            destpoints,
            rawapp_byslice,
        ]
        slicefunc = _procOnePhaseProject
        packfunc = _packslicedataPhaseProject
        unpackfunc = _unpackslicedataPhaseProject
        slicetargets = [rawapp_byslice, cine_byslice, weights_byslice]
        slicemask = np.ones_like(rawapp_byslice[0, :, 0])

        slicetotal = tide_genericmultiproc.run_multiproc(
            slicefunc,
            packfunc,
            unpackfunc,
            sliceargs,
            slicetargets,
            inputshape,
            slicemask,
            None,
            nprocs,
            alwaysmultiproc,
            showprogressbar,
            8,
            indexaxis=1,
            procunit="slices",
            cache=cache,
            debug=debug,
        )
    else:
        for theslice in tqdm(
            range(numslices),
            desc="Slice",
            unit="slices",
            disable=(not showprogressbar),
        ):
            validlocs = validlocslist[theslice]
            if len(validlocs) > 0:
                for t in proctrs:
                    filteredmr = -demeandata_byslice[validlocs, theslice, t]
                    cinemr = fmri_data_byslice[validlocs, theslice, t]
                    thevals, theweights, theindices = tide_resample.congrid(
                        outphases,
                        cardphasevals[theslice, t],
                        1.0,
                        congridbins,
                        kernel=gridkernel,
                        cyclic=True,
                        cache=cache,
                        debug=debug,
                    )
                    for i in range(len(theindices)):
                        weights_byslice[validlocs, theslice, theindices[i]] += theweights[i]
                        rawapp_byslice[validlocs, theslice, theindices[i]] += filteredmr
                        cine_byslice[validlocs, theslice, theindices[i]] += theweights[i] * cinemr
                for d in range(destpoints):
                    if weights_byslice[validlocs[0], theslice, d] == 0.0:
                        weights_byslice[validlocs, theslice, d] = 1.0
                rawapp_byslice[validlocs, theslice, :] = np.nan_to_num(
                    rawapp_byslice[validlocs, theslice, :]
                    / weights_byslice[validlocs, theslice, :]
                )
                cine_byslice[validlocs, theslice, :] = np.nan_to_num(
                    cine_byslice[validlocs, theslice, :] / weights_byslice[validlocs, theslice, :]
                )
            else:
                rawapp_byslice[:, theslice, :] = 0.0
                cine_byslice[:, theslice, :] = 0.0


def _procOneSliceSmoothing(slice, sliceargs, **kwargs):
    """
    Apply smoothing filter to a single slice of projected data along time dimension.

    This function processes a single slice of data by applying a smoothing filter
    to the raw application data and computing circular derivatives for the
    specified slice. The smoothing is applied only to valid locations within the slice.

    Parameters
    ----------
    slice : int
        The slice index to process.
    sliceargs : tuple
        A tuple containing the following elements:

        - validlocslist : list of arrays
          List of arrays containing valid location indices for each slice
        - rawapp_byslice : ndarray
          Array containing raw application data by slice [locations, slices, time_points]
        - appsmoothingfilter : object
          Smoothing filter object with an apply method
        - phaseFs : array-like
          Frequency values for smoothing filter application
        - derivatives_byslice : ndarray
          Array to store computed derivatives [locations, slices, time_points]
    **kwargs : dict
        Additional keyword arguments:
        - debug : bool, optional
          Enable debug mode (default: False)

    Returns
    -------
    tuple
        A tuple containing:

        - slice : int
          The input slice index
        - rawapp_byslice : ndarray
          Smoothed raw application data for the specified slice [locations, time_points]
        - derivatives_byslice : ndarray
          Computed circular derivatives for the specified slice [locations, time_points]

    Notes
    -----
    - The function only processes slices with valid locations (len(validlocs) > 0)
    - Smoothing is applied using the provided smoothing filter's apply method
    - Circular derivatives are computed using the `circularderivs` function
    - The function modifies the input arrays in-place

    Examples
    --------
    >>> slice_idx = 5
    >>> sliceargs = (validlocslist, rawapp_byslice, appsmoothingfilter, phaseFs, derivatives_byslice)
    >>> result = _procOneSliceSmoothing(slice_idx, sliceargs, debug=True)
    """
    options = {
        "debug": False,
    }
    options.update(kwargs)
    debug = options["debug"]
    (validlocslist, rawapp_byslice, appsmoothingfilter, phaseFs, derivatives_byslice) = sliceargs
    # now smooth the projected data along the time dimension
    validlocs = validlocslist[slice]
    if len(validlocs) > 0:
        for loc in validlocs:
            rawapp_byslice[loc, slice, :] = appsmoothingfilter.apply(
                phaseFs, rawapp_byslice[loc, slice, :]
            )
            derivatives_byslice[loc, slice, :] = circularderivs(rawapp_byslice[loc, slice, :])
    return slice, rawapp_byslice[:, slice, :], derivatives_byslice[:, slice, :]


def _packslicedataSliceSmoothing(slicenum, sliceargs):
    """Pack slice data for slice smoothing operation.

    Parameters
    ----------
    slicenum : int
        The slice number identifier.
    sliceargs : list
        List containing slice arguments with at least 5 elements.

    Returns
    -------
    list
        A list containing the first 5 elements from sliceargs in the same order.

    Notes
    -----
    This function extracts the first five elements from the sliceargs parameter
    and returns them as a new list. It's typically used as part of a slice
    smoothing pipeline where slice arguments need to be packed for further processing.

    Examples
    --------
    >>> _packslicedataSliceSmoothing(1, [10, 20, 30, 40, 50, 60])
    [10, 20, 30, 40, 50]
    """
    return [
        sliceargs[0],
        sliceargs[1],
        sliceargs[2],
        sliceargs[3],
        sliceargs[4],
    ]


def _unpackslicedataSliceSmoothing(retvals, voxelproducts):
    """
    Unpack slice data for smoothing operation.

    This function assigns smoothed slice data back to the voxel products array
    based on the provided retvals structure.

    Parameters
    ----------
    retvals : tuple of array-like
        A tuple containing:
        - retvals[0] : array-like
            Index array for slice selection
        - retvals[1] : array-like
            First set of smoothed data to assign
        - retvals[2] : array-like
            Second set of smoothed data to assign
    voxelproducts : list of array-like
        A list containing two array-like objects where:
        - voxelproducts[0] : array-like
            First voxel product array to be modified
        - voxelproducts[1] : array-like
            Second voxel product array to be modified

    Returns
    -------
    None
        This function modifies the voxelproducts arrays in-place and does not return anything.

    Notes
    -----
    The function performs in-place assignment operations on the voxelproducts arrays.
    The first dimension of voxelproducts arrays is modified using retvals[0] as indices,
    while the second and third dimensions are directly assigned from retvals[1] and retvals[2].

    Examples
    --------
    >>> import numpy as np
    >>> retvals = (np.array([0, 1, 2]), np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8], [9, 10], [11, 12]]))
    >>> voxelproducts = [np.zeros((3, 3, 2)), np.zeros((3, 3, 2))]
    >>> _unpackslicedataSliceSmoothing(retvals, voxelproducts)
    >>> print(voxelproducts[0])
    >>> print(voxelproducts[1])
    """
    (voxelproducts[0])[:, retvals[0], :] = retvals[1]
    (voxelproducts[1])[:, retvals[0], :] = retvals[2]


def tcsmoothingpass(
    numslices,
    validlocslist,
    rawapp_byslice,
    appsmoothingfilter,
    phaseFs,
    derivatives_byslice,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    debug=False,
):
    """
    Apply smoothing to time course data across slices using multiprocessing.

    This function performs smoothing operations on time course data organized by slices,
    utilizing multiprocessing for improved performance when processing large datasets.

    Parameters
    ----------
    numslices : int
        Number of slices in the dataset
    validlocslist : list
        List of valid locations for processing
    rawapp_byslice : NDArray
        Raw application data organized by slice
    appsmoothingfilter : NDArray
        Smoothing filter to be applied
    phaseFs : float
        Phase frequency parameter for smoothing operations
    derivatives_byslice : NDArray
        Derivative data organized by slice
    nprocs : int, optional
        Number of processors to use for multiprocessing (default is 1)
    alwaysmultiproc : bool, optional
        Whether to always use multiprocessing regardless of data size (default is False)
    showprogressbar : bool, optional
        Whether to display progress bar during processing (default is True)
    debug : bool, optional
        Enable debug mode for additional logging (default is False)

    Returns
    -------
    NDArray
        Processed data after smoothing operations have been applied

    Notes
    -----
    This function uses the `tide_genericmultiproc.run_multiproc` utility to distribute
    the smoothing workload across multiple processors. The function handles data organization
    and processing for each slice individually, then combines results.

    Examples
    --------
    >>> result = tcsmoothingpass(
    ...     numslices=10,
    ...     validlocslist=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ...     rawapp_byslice=raw_data,
    ...     appsmoothingfilter=smoothing_filter,
    ...     phaseFs=100.0,
    ...     derivatives_byslice=derivatives,
    ...     nprocs=4
    ... )
    """
    inputshape = rawapp_byslice.shape
    sliceargs = [validlocslist, rawapp_byslice, appsmoothingfilter, phaseFs, derivatives_byslice]
    slicefunc = _procOneSliceSmoothing
    packfunc = _packslicedataSliceSmoothing
    unpackfunc = _unpackslicedataSliceSmoothing
    slicetargets = [rawapp_byslice, derivatives_byslice]
    slicemask = np.ones_like(rawapp_byslice[0, :, 0])

    slicetotal = tide_genericmultiproc.run_multiproc(
        slicefunc,
        packfunc,
        unpackfunc,
        sliceargs,
        slicetargets,
        inputshape,
        slicemask,
        None,
        nprocs,
        alwaysmultiproc,
        showprogressbar,
        16,
        indexaxis=1,
        procunit="slices",
        debug=debug,
    )


def phaseproject(
    input_data,
    demeandata_byslice,
    means_byslice,
    rawapp_byslice,
    app_byslice,
    normapp_byslice,
    weights_byslice,
    cine_byslice,
    projmask_byslice,
    derivatives_byslice,
    proctrs,
    thispass,
    args,
    sliceoffsets,
    cardphasevals,
    outphases,
    appsmoothingfilter,
    phaseFs,
    thecorrfunc_byslice,
    waveamp_byslice,
    wavedelay_byslice,
    wavedelayCOM_byslice,
    corrected_rawapp_byslice,
    corrstartloc,
    correndloc,
    thealiasedcorrx,
    theAliasedCorrelator,
):
    """
    Perform phase projection and related processing on fMRI data across slices.

    This function performs phase projection on fMRI data, optionally smoothing
    timecourses, and applying flips based on derivative information. It also
    computes wavelet-based correlation measures and updates relevant arrays
    in-place for further processing.

    Parameters
    ----------
    input_data : object
        Input fMRI data container with `getdims()` and `byslice()` methods.
    demeandata_byslice : array_like
        Demeaned fMRI data by slice.
    means_byslice : array_like
        Mean values by slice for normalization.
    rawapp_byslice : array_like
        Raw APP (Arterial Spin Labeling) data by slice.
    app_byslice : array_like
        APP data after initial processing.
    normapp_byslice : array_like
        Normalized APP data.
    weights_byslice : array_like
        Weights by slice for processing.
    cine_byslice : array_like
        Cine data by slice.
    projmask_byslice : array_like
        Projection mask by slice.
    derivatives_byslice : array_like
        Derivative data by slice, used for determining flips.
    proctrs : array_like
        Processing timepoints or transformation parameters.
    thispass : int
        Current processing pass number.
    args : argparse.Namespace
        Command-line arguments controlling processing behavior.
    sliceoffsets : array_like
        Slice offset values.
    cardphasevals : array_like
        Cardiac phase values.
    outphases : array_like
        Output phases.
    appsmoothingfilter : array_like
        Smoothing filter for timecourses.
    phaseFs : float
        Sampling frequency for phase processing.
    thecorrfunc_byslice : array_like
        Correlation function by slice.
    waveamp_byslice : array_like
        Wave amplitude by slice.
    wavedelay_byslice : array_like
        Wave delay by slice.
    wavedelayCOM_byslice : array_like
        Center of mass of wave delay by slice.
    corrected_rawapp_byslice : array_like
        Corrected raw APP data by slice.
    corrstartloc : int
        Start location for correlation computation.
    correndloc : int
        End location for correlation computation.
    thealiasedcorrx : array_like
        Aliased correlation x-axis values.
    theAliasedCorrelator : object
        Correlator object for aliased correlation computation.

    Returns
    -------
    appflips_byslice : array_like
        Flip values applied to the APP data by slice.

    Notes
    -----
    - The function modifies several input arrays in-place.
    - If `args.smoothapp` is True, smoothing is applied to the raw APP data.
    - If `args.fliparteries` is True, flips are applied to correct arterial
      orientation.
    - If `args.doaliasedcorrelation` is True, aliased correlation is computed
      and stored in `thecorrfunc_byslice`.

    Examples
    --------
    >>> phaseproject(
    ...     input_data, demeandata_byslice, means_byslice, rawapp_byslice,
    ...     app_byslice, normapp_byslice, weights_byslice, cine_byslice,
    ...     projmask_byslice, derivatives_byslice, proctrs, thispass, args,
    ...     sliceoffsets, cardphasevals, outphases, appsmoothingfilter,
    ...     phaseFs, thecorrfunc_byslice, waveamp_byslice, wavedelay_byslice,
    ...     wavedelayCOM_byslice, corrected_rawapp_byslice, corrstartloc,
    ...     correndloc, thealiasedcorrx, theAliasedCorrelator
    ... )
    """
    xsize, ysize, numslices, timepoints = input_data.getdims()
    fmri_data_byslice = input_data.byslice()

    # first find the validlocs for each slice
    validlocslist = []
    if args.verbose:
        print("Finding validlocs")
    for theslice in range(numslices):
        validlocslist.append(np.where(projmask_byslice[:, theslice] > 0)[0])

    # phase project each slice
    print("Phase projecting")
    phaseprojectpass(
        numslices,
        demeandata_byslice,
        fmri_data_byslice,
        validlocslist,
        proctrs,
        weights_byslice,
        cine_byslice,
        rawapp_byslice,
        outphases,
        cardphasevals,
        args.congridbins,
        args.gridkernel,
        args.destpoints,
        cache=args.congridcache,
        mpcode=args.mpphaseproject,
        nprocs=args.nprocs,
        showprogressbar=args.showprogressbar,
    )

    # smooth the phase projection, if requested
    if args.smoothapp:
        print("Smoothing timecourses")
        tcsmoothingpass(
            numslices,
            validlocslist,
            rawapp_byslice,
            appsmoothingfilter,
            phaseFs,
            derivatives_byslice,
            nprocs=args.nprocs,
            showprogressbar=args.showprogressbar,
        )

    # now do the flips
    print("Doing flips")
    for theslice in tqdm(
        range(numslices),
        desc="Slice",
        unit="slices",
        disable=(not args.showprogressbar),
    ):
        # now do the flips
        validlocs = validlocslist[theslice]
        if len(validlocs) > 0:
            appflips_byslice = np.where(
                -derivatives_byslice[:, :, 2] > derivatives_byslice[:, :, 0], -1.0, 1.0
            )
            timecoursemean = np.mean(rawapp_byslice[validlocs, theslice, :], axis=1).reshape(
                (-1, 1)
            )
            if args.fliparteries:
                corrected_rawapp_byslice[validlocs, theslice, :] = (
                    rawapp_byslice[validlocs, theslice, :] - timecoursemean
                ) * appflips_byslice[validlocs, theslice, None] + timecoursemean
                if args.doaliasedcorrelation and (thispass > 0):
                    for theloc in validlocs:
                        thecorrfunc_byslice[theloc, theslice, :] = theAliasedCorrelator.apply(
                            -appflips_byslice[theloc, theslice]
                            * demeandata_byslice[theloc, theslice, :],
                            int(sliceoffsets[theslice]),
                        )[corrstartloc : correndloc + 1]
                        maxloc = np.argmax(thecorrfunc_byslice[theloc, theslice, :])
                        wavedelay_byslice[theloc, theslice] = (
                            thealiasedcorrx[corrstartloc : correndloc + 1]
                        )[maxloc]
                        waveamp_byslice[theloc, theslice] = np.fabs(
                            thecorrfunc_byslice[theloc, theslice, maxloc]
                        )
                        wavedelayCOM_byslice[theloc, theslice] = theCOM(
                            thealiasedcorrx[corrstartloc : correndloc + 1],
                            np.fabs(thecorrfunc_byslice[theloc, theslice, :]),
                        )
            else:
                corrected_rawapp_byslice[validlocs, theslice, :] = rawapp_byslice[
                    validlocs, theslice, :
                ]
                if args.doaliasedcorrelation and (thispass > 0):
                    for theloc in validlocs:
                        thecorrfunc_byslice[theloc, theslice, :] = theAliasedCorrelator.apply(
                            -demeandata_byslice[theloc, theslice, :],
                            int(sliceoffsets[theslice]),
                        )[corrstartloc : correndloc + 1]
                        maxloc = np.argmax(np.abs(thecorrfunc_byslice[theloc, theslice, :]))
                        wavedelay_byslice[theloc, theslice] = (
                            thealiasedcorrx[corrstartloc : correndloc + 1]
                        )[maxloc]
                        waveamp_byslice[theloc, theslice] = np.fabs(
                            thecorrfunc_byslice[theloc, theslice, maxloc]
                        )
            timecoursemin = np.min(
                corrected_rawapp_byslice[validlocs, theslice, :], axis=1
            ).reshape((-1, 1))
            app_byslice[validlocs, theslice, :] = (
                corrected_rawapp_byslice[validlocs, theslice, :] - timecoursemin
            )
            normapp_byslice[validlocs, theslice, :] = np.nan_to_num(
                app_byslice[validlocs, theslice, :] / means_byslice[validlocs, theslice, None]
            )
    return appflips_byslice


def findvessels(
    app,
    normapp,
    validlocs,
    numspatiallocs,
    outputroot,
    unnormvesselmap,
    destpoints,
    softvesselfrac,
    histlen,
    outputlevel,
    debug=False,
):
    """
    Find vessel thresholds and generate vessel masks from app data.

    This function processes app data to identify vessel thresholds and optionally
    generates histograms for visualization. It handles both normalized and
    unnormalized vessel maps based on the input parameters.

    Parameters
    ----------
    app : NDArray
        Raw app data array
    normapp : NDArray
        Normalized app data array
    validlocs : NDArray
        Array of valid locations for processing
    numspatiallocs : int
        Number of spatial locations
    outputroot : str
        Root directory path for output files
    unnormvesselmap : bool
        Flag indicating whether to use unnormalized vessel map
    destpoints : int
        Number of destination points
    softvesselfrac : float
        Fractional multiplier for soft vessel threshold
    histlen : int
        Length of histogram bins
    outputlevel : int
        Level of output generation (0 = no histogram, 1 = histogram only)
    debug : bool, optional
        Debug flag for additional logging (default is False)

    Returns
    -------
    tuple
        Tuple containing (hardvesselthresh, softvesselthresh) threshold values

    Notes
    -----
    The function performs the following steps:
    1. Reshapes app data based on unnormvesselmap flag
    2. Extracts valid locations from the reshaped data
    3. Generates histogram if outputlevel > 0
    4. Calculates hard and soft vessel thresholds based on 98th percentile
    5. Prints threshold values to console

    Examples
    --------
    >>> hard_thresh, soft_thresh = findvessels(
    ...     app=app_data,
    ...     normapp=norm_app_data,
    ...     validlocs=valid_indices,
    ...     numspatiallocs=100,
    ...     outputroot='/path/to/output',
    ...     unnormvesselmap=True,
    ...     destpoints=50,
    ...     softvesselfrac=0.5,
    ...     histlen=100,
    ...     outputlevel=1
    ... )
    """
    if unnormvesselmap:
        app2d = app.reshape((numspatiallocs, destpoints))
    else:
        app2d = normapp.reshape((numspatiallocs, destpoints))
    histinput = app2d[validlocs, :].reshape((len(validlocs), destpoints))
    if outputlevel > 0:
        namesuffix = "_desc-apppeaks_hist"
        tide_stats.makeandsavehistogram(
            histinput,
            histlen,
            0,
            outputroot + namesuffix,
            debug=debug,
        )

    # find vessel thresholds
    tide_util.logmem("before making vessel masks")
    hardvesselthresh = tide_stats.getfracvals(np.max(histinput, axis=1), [0.98])[0] / 2.0
    softvesselthresh = softvesselfrac * hardvesselthresh
    print(
        "hard, soft vessel thresholds set to",
        "{:.3f}".format(hardvesselthresh),
        "{:.3f}".format(softvesselthresh),
    )


def upsampleimage(input_data, numsteps, sliceoffsets, slicesamplerate, outputroot):
    """
    Upsample fMRI data along the temporal and slice dimensions.

    This function takes fMRI data and upsamples it by a factor of `numsteps` along
    the temporal dimension, and interpolates across slices to align with specified
    slice offsets. The resulting upsampled data is saved as a NIfTI file.

    Parameters
    ----------
    input_data : object
        Input fMRI data object with attributes: `byvol()`, `timepoints`, `xsize`,
        `ysize`, `numslices`, and `copyheader()`.
    numsteps : int
        Upsampling factor along the temporal dimension.
    sliceoffsets : array-like of int
        Slice offset indices indicating where each slice's data should be placed
        in the upsampled volume.
    slicesamplerate : float
        Sampling rate of the slice acquisition (used to set the TR in the output header).
    outputroot : str
        Root name for the output NIfTI file (will be suffixed with "_upsampled").

    Returns
    -------
    None
        The function saves the upsampled data to a NIfTI file and does not return any value.

    Notes
    -----
    - The function demeanes the input data before upsampling.
    - Interpolation is performed along the slice direction using linear interpolation.
    - The output file is saved using `tide_io.savetonifti`.

    Examples
    --------
    >>> upsampleimage(fmri_data, numsteps=2, sliceoffsets=[0, 1], slicesamplerate=2.0, outputroot='output')
    Upsamples the fMRI data by a factor of 2 and saves to 'output_upsampled.nii'.
    """
    fmri_data = input_data.byvol()
    timepoints = input_data.timepoints
    xsize = input_data.xsize
    ysize = input_data.ysize
    numslices = input_data.numslices

    # allocate the image
    print(f"upsampling fmri data by a factor of {numsteps}")
    upsampleimage = np.zeros((xsize, ysize, numslices, numsteps * timepoints), dtype=float)

    # demean the raw data
    meanfmri = fmri_data.mean(axis=1)
    demeaned_data = fmri_data - meanfmri[:, None]

    # drop in the raw data
    for theslice in range(numslices):
        upsampleimage[
            :, :, theslice, sliceoffsets[theslice] : timepoints * numsteps : numsteps
        ] = demeaned_data.reshape((xsize, ysize, numslices, timepoints))[:, :, theslice, :]

    upsampleimage_byslice = upsampleimage.reshape(xsize * ysize, numslices, timepoints * numsteps)

    # interpolate along the slice direction
    thedstlocs = np.linspace(0, numslices, num=len(sliceoffsets), endpoint=False)
    print(f"len(destlocst), destlocs: {len(thedstlocs)}, {thedstlocs}")
    for thetimepoint in range(0, timepoints * numsteps):
        thestep = thetimepoint % numsteps
        print(f"interpolating step {thestep}")
        thesrclocs = np.where(sliceoffsets == thestep)[0]
        print(f"timepoint: {thetimepoint}, sourcelocs: {thesrclocs}")
        for thexyvoxel in range(xsize * ysize):
            theinterps = np.interp(
                thedstlocs,
                1.0 * thesrclocs,
                upsampleimage_byslice[thexyvoxel, thesrclocs, thetimepoint],
            )
            upsampleimage_byslice[thexyvoxel, :, thetimepoint] = 1.0 * theinterps

    theheader = input_data.copyheader(
        numtimepoints=(timepoints * numsteps), tr=(1.0 / slicesamplerate)
    )
    tide_io.savetonifti(upsampleimage, theheader, outputroot + "_upsampled")
    print("upsampling complete")


def wrightmap(
    input_data,
    demeandata_byslice,
    rawapp_byslice,
    projmask_byslice,
    outphases,
    cardphasevals,
    proctrs,
    congridbins,
    gridkernel,
    destpoints,
    iterations=100,
    nprocs=-1,
    verbose=False,
    debug=False,
):
    """
    Compute a vessel map using Wright's method by performing phase correlation
    analysis across randomized subsets of timecourses.

    This function implements Wright's method for estimating vessel maps by
    splitting the timecourse data into two random halves, projecting each half
    separately, and computing the Pearson correlation between the resulting
    projections for each voxel and slice. The final map is derived as the mean
    of these correlations across iterations.

    Parameters
    ----------
    input_data : object
        Input data container with attributes `xsize`, `ysize`, and `numslices`.
    demeandata_byslice : array_like
        Demeaned data organized by slice, shape ``(nvoxels, numslices)``.
    rawapp_byslice : array_like
        Raw application data by slice, shape ``(nvoxels, numslices)``.
    projmask_byslice : array_like
        Projection mask by slice, shape ``(nvoxels, numslices)``.
    outphases : array_like
        Output phases, shape ``(nphases,)``.
    cardphasevals : array_like
        Cardinal phase values, shape ``(nphases,)``.
    proctrs : array_like
        Timecourse indices to be processed, shape ``(ntimepoints,)``.
    congridbins : array_like
        Binning information for congrid interpolation.
    gridkernel : array_like
        Kernel for grid interpolation.
    destpoints : array_like
        Destination points for projection.
    iterations : int, optional
        Number of iterations for random splitting (default is 100).
    nprocs : int, optional
        Number of processes to use for parallel computation; -1 uses all
        available cores (default is -1).
    verbose : bool, optional
        If True, print progress messages (default is False).
    debug : bool, optional
        If True, print additional debug information (default is False).

    Returns
    -------
    wrightcorrs : ndarray
        Computed vessel map with shape ``(xsize, ysize, numslices)``.

    Notes
    -----
    This function performs a bootstrap-like procedure where the input timecourse
    is randomly split into two halves, and phase projections are computed for
    each half. Pearson correlation is computed between the two projections for
    each voxel and slice. The result is averaged over all iterations to produce
    the final vessel map.

    Examples
    --------
    >>> wrightcorrs = wrightmap(
    ...     input_data,
    ...     demeandata_byslice,
    ...     rawapp_byslice,
    ...     projmask_byslice,
    ...     outphases,
    ...     cardphasevals,
    ...     proctrs,
    ...     congridbins,
    ...     gridkernel,
    ...     destpoints,
    ...     iterations=50,
    ...     verbose=True
    ... )
    """
    xsize = input_data.xsize
    ysize = input_data.ysize
    numslices = input_data.numslices
    # make a vessel map using Wright's method
    wrightcorrs_byslice = np.zeros((xsize * ysize, numslices, iterations))
    # first find the validlocs for each slice
    validlocslist = []
    if verbose:
        print("Finding validlocs")
    for theslice in range(numslices):
        validlocslist.append(np.where(projmask_byslice[:, theslice] > 0)[0])
    for theiteration in range(iterations):
        print(f"wright iteration: {theiteration + 1} of {iterations}")
        # split timecourse into two sets
        scrambledprocs = np.random.permutation(proctrs)
        proctrs1 = scrambledprocs[: int(len(scrambledprocs) // 2)]
        proctrs2 = scrambledprocs[int(len(scrambledprocs) // 2) :]
        if debug:
            print(f"{proctrs1=}, {proctrs2=}")

        # phase project each slice
        rawapp_byslice1 = np.zeros_like(rawapp_byslice)
        cine_byslice1 = np.zeros_like(rawapp_byslice)
        weights_byslice1 = np.zeros_like(rawapp_byslice)
        phaseprojectpass(
            numslices,
            demeandata_byslice,
            input_data.byslice(),
            validlocslist,
            proctrs1,
            weights_byslice1,
            cine_byslice1,
            rawapp_byslice1,
            outphases,
            cardphasevals,
            congridbins,
            gridkernel,
            destpoints,
            nprocs=nprocs,
            showprogressbar=False,
        )
        rawapp_byslice2 = np.zeros_like(rawapp_byslice)
        cine_byslice2 = np.zeros_like(rawapp_byslice)
        weights_byslice2 = np.zeros_like(rawapp_byslice)
        phaseprojectpass(
            numslices,
            demeandata_byslice,
            input_data.byslice(),
            validlocslist,
            proctrs2,
            weights_byslice2,
            cine_byslice2,
            rawapp_byslice2,
            outphases,
            cardphasevals,
            congridbins,
            gridkernel,
            destpoints,
            nprocs=nprocs,
            showprogressbar=False,
        )
        for theslice in range(numslices):
            for thepoint in validlocslist[theslice]:
                theresult = pearsonr(
                    rawapp_byslice1[thepoint, theslice, :],
                    rawapp_byslice2[thepoint, theslice, :],
                )
                theRvalue = theresult.statistic
                if debug:
                    print("theRvalue = ", theRvalue)
                wrightcorrs_byslice[thepoint, theslice, theiteration] = theRvalue
    wrightcorrs = np.mean(wrightcorrs_byslice, axis=2).reshape(xsize, ysize, numslices)
    return wrightcorrs
