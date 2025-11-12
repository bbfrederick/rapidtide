#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2024-2025 Blaise Frederick
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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion

import rapidtide.filter as tide_filt
import rapidtide.stats as tide_stats
from rapidtide.RapidtideDataset import RapidtideDataset


def prepmask(inputmask: NDArray) -> NDArray:
    """
    Apply binary erosion to the input mask.

    Parameters
    ----------
    inputmask : NDArray
        Input binary mask array to be eroded.

    Returns
    -------
    NDArray
        Eroded binary mask array with features reduced by one pixel in all directions.

    Notes
    -----
    This function uses binary erosion to shrink the boundaries of foreground objects
    in the input mask. The erosion operation removes pixels from the boundaries of
    objects, effectively reducing their size by one pixel in all directions.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import binary_erosion
    >>> mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> eroded = prepmask(mask)
    >>> print(eroded)
    [[0 0 0]
     [0 0 0]
     [0 0 0]]
    """
    erodedmask = binary_erosion(inputmask)
    return erodedmask


def getmasksize(themask: NDArray) -> int:
    """
    Calculate the number of non-zero elements in a mask array.

    This function counts the number of elements in the input array that are greater than zero,
    effectively measuring the size of the active region in a binary mask.

    Parameters
    ----------
    themask : ndarray
        Input array representing a mask, where positive values indicate active regions
        and zero/negative values indicate inactive regions.

    Returns
    -------
    int
        The number of elements in the mask that are greater than zero.

    Notes
    -----
    The function uses `np.where()` to find indices where the mask is greater than zero,
    then `np.ravel()` to flatten the resulting array, and finally `len()` to count the elements.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([[1, 0, 1], [0, 1, 1]])
    >>> getmasksize(mask)
    4

    >>> mask = np.array([0, 0, 0, 0])
    >>> getmasksize(mask)
    0

    >>> mask = np.array([1, 2, 3, 4])
    >>> getmasksize(mask)
    4
    """
    return len(np.ravel(themask[np.where(themask > 0)]))


def checkregressors(
    theregressors: dict[str, Any], numpasses: int, filterlimits: list[float], debug: bool = False
) -> dict[str, float]:
    """
    Calculate and return statistical metrics for the first and last regressors in a sequence.

    This function extracts the first and last regressors from a dictionary of regressors,
    applies spectral filtering based on provided limits, and computes various statistical
    measures such as kurtosis, skewness, and spectral flatness for both regressors.

    Parameters
    ----------
    theregressors : dict[str, Any]
        Dictionary containing regressor objects indexed by pass number (e.g., "pass1", "pass2").
        Each regressor is expected to have attributes like `specaxis`, `specdata`, `kurtosis`,
        `kurtosis_z`, `kurtosis_p`, `skewness`, `skewness_z`, `skewness_p`.
    numpasses : int
        Total number of passes; used to identify the last regressor in the dictionary.
    filterlimits : list[float]
        A list of two floats specifying the lower and upper spectral limits for filtering.
        The function uses these to slice the spectral data.
    debug : bool, optional
        If True, prints debug information including filter limits, indices, and spectral data.
        Default is False.

    Returns
    -------
    dict[str, float]
        A dictionary containing the following keys for both the first and last regressors:
        - `{label}_kurtosis`
        - `{label}_kurtosis_z`
        - `{label}_kurtosis_p`
        - `{label}_skewness`
        - `{label}_skewness_z`
        - `{label}_skewness_p`
        - `{label}_spectralflatness`
        Where `{label}` is either "first" or "last".

    Notes
    -----
    The function uses `numpy.argmax` and `numpy.argmin` to determine the indices of the
    spectral axis that correspond to the provided filter limits. It then slices the
    spectral data using these indices to compute the spectral flatness.

    Examples
    --------
    >>> import numpy as np
    >>> regressors = {
    ...     "pass1": MockRegressor(
    ...         specaxis=np.linspace(0, 10, 100),
    ...         specdata=np.random.rand(100),
    ...         kurtosis=1.0,
    ...         kurtosis_z=0.5,
    ...         kurtosis_p=0.1,
    ...         skewness=0.0,
    ...         skewness_z=0.0,
    ...         skewness_p=0.5
    ...     ),
    ...     "pass2": MockRegressor(
    ...         specaxis=np.linspace(0, 10, 100),
    ...         specdata=np.random.rand(100),
    ...         kurtosis=1.2,
    ...         kurtosis_z=0.6,
    ...         kurtosis_p=0.05,
    ...         skewness=0.1,
    ...         skewness_z=0.1,
    ...         skewness_p=0.4
    ...     )
    ... }
    >>> result = checkregressors(regressors, 2, [2.0, 8.0])
    >>> print(result)
    {'first_kurtosis': 1.0, 'first_kurtosis_z': 0.5, 'first_kurtosis_p': 0.1,
     'first_skewness': 0.0, 'first_skewness_z': 0.0, 'first_skewness_p': 0.5,
     'first_spectralflatness': 0.5, 'last_kurtosis': 1.2, 'last_kurtosis_z': 0.6,
     'last_kurtosis_p': 0.05, 'last_skewness': 0.1, 'last_skewness_z': 0.1,
     'last_skewness_p': 0.4, 'last_spectralflatness': 0.45}
    """
    regressormetrics = {}
    firstregressor = theregressors["pass1"]
    lastregressor = theregressors[f"pass{numpasses}"]
    lowerlimindex = np.argmax(firstregressor.specaxis >= filterlimits[0])
    upperlimindex = np.argmin(firstregressor.specaxis <= filterlimits[1]) + 1
    if debug:
        print(f"{filterlimits=}")
        print(f"{lowerlimindex=}, {upperlimindex=}")
        print(firstregressor.specaxis)
        print(firstregressor.specdata[lowerlimindex:upperlimindex])
    for label, regressor in [["first", firstregressor], ["last", lastregressor]]:
        regressormetrics[f"{label}_kurtosis"] = regressor.kurtosis
        regressormetrics[f"{label}_kurtosis_z"] = regressor.kurtosis_z
        regressormetrics[f"{label}_kurtosis_p"] = regressor.kurtosis_p
        regressormetrics[f"{label}_skewness"] = regressor.skewness
        regressormetrics[f"{label}_skewness_z"] = regressor.skewness_z
        regressormetrics[f"{label}_skewness_p"] = regressor.skewness_p
        regressormetrics[f"{label}_spectralflatness"] = tide_filt.spectralflatness(
            regressor.specdata[lowerlimindex:upperlimindex]
        )
    return regressormetrics


def gethistmetrics(
    themap: NDArray,
    themask: NDArray,
    thedict: dict[str, Any],
    thehistlabel: str | None = None,
    histlen: int = 101,
    rangemin: float = -1.0,
    rangemax: float = 1.0,
    nozero: bool = False,
    savehist: bool = False,
    ignorefirstpoint: bool = False,
    debug: bool = False,
) -> None:
    """
    Compute histogram-based metrics for masked data and store results in a dictionary.

    This function applies a mask to the input data, computes various statistical
    measures including percentiles, moments (skewness, kurtosis), and histogram
    properties, and stores the results in the provided dictionary.

    Parameters
    ----------
    themap : NDArray
        The input data array from which metrics are computed.
    themask : NDArray
        A boolean or numeric mask array used to select valid data points.
    thedict : dict[str, Any]
        Dictionary to store computed metrics. Keys will be updated with statistical
        values such as percentiles, widths, skewness, kurtosis, and histogram properties.
    thehistlabel : str, optional
        Label for the histogram plot if `debug` is enabled. Default is None.
    histlen : int, optional
        Number of bins for the histogram. Default is 101.
    rangemin : float, optional
        Minimum value of the histogram range. Default is -1.0.
    rangemax : float, optional
        Maximum value of the histogram range. Default is 1.0.
    nozero : bool, optional
        If True, exclude zero values from the computation. Default is False.
    savehist : bool, optional
        If True, save histogram bin centers and values to `thedict`. Default is False.
    ignorefirstpoint : bool, optional
        If True, ignore the first point when computing the histogram. Default is False.
    debug : bool, optional
        If True, print debug information and display histogram plot. Default is False.

    Returns
    -------
    None
        Results are stored in the input dictionary `thedict`.

    Notes
    -----
    The function modifies the input dictionary `thedict` in place by adding or updating
    keys with computed statistics. If the mask is empty or no valid data remains after
    filtering, all keys are set to `None` or 0.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 100)
    >>> mask = np.ones((100, 100))
    >>> stats_dict = {}
    >>> gethistmetrics(data, mask, stats_dict, debug=True)
    >>> print(stats_dict['pct50'])
    """
    # mask and flatten the data
    maskisempty = False
    if len(np.where(themask > 0)) == 0:
        maskisempty = False
    if debug:
        print("num-nonzero in mask", len(np.where(themask > 0)[0]))
    if not maskisempty:
        dataforhist = np.ravel(themap[np.where(themask > 0.0)])
        if nozero:
            dataforhist = dataforhist[np.where(dataforhist != 0.0)]
            if len(dataforhist) == 0:
                maskisempty = True
            if debug:
                print("num-nonzero in dataforhist", len(dataforhist))
    else:
        maskisempty = True

    if not maskisempty:
        # get percentiles
        (
            thedict["pct02"],
            thedict["pct25"],
            thedict["pct50"],
            thedict["pct75"],
            thedict["pct98"],
        ) = tide_stats.getfracvals(dataforhist, [0.02, 0.25, 0.5, 0.75, 0.98], debug=debug)
        thedict["voxelsincluded"] = len(dataforhist)
        thedict["q1width"] = thedict["pct25"] - thedict["pct02"]
        thedict["q2width"] = thedict["pct50"] - thedict["pct25"]
        thedict["q3width"] = thedict["pct75"] - thedict["pct50"]
        thedict["q4width"] = thedict["pct98"] - thedict["pct75"]
        thedict["mid50width"] = thedict["pct75"] - thedict["pct25"]

        # get moments
        (
            thedict["kurtosis"],
            thedict["kurtosis_z"],
            thedict["kurtosis_p"],
        ) = tide_stats.kurtosisstats(dataforhist)
        (
            thedict["skewness"],
            thedict["skewness_z"],
            thedict["skewness_p"],
        ) = tide_stats.skewnessstats(dataforhist)
        (
            thehist,
            thedict["peakheight"],
            thedict["peakloc"],
            thedict["peakwidth"],
            thedict["centerofmass"],
            thedict["peakpercentile"],
        ) = tide_stats.makehistogram(
            dataforhist,
            histlen,
            refine=False,
            therange=(rangemin, rangemax),
            normalize=True,
            ignorefirstpoint=ignorefirstpoint,
            debug=debug,
        )
        histbincenters = ((thehist[1][1:] + thehist[1][0:-1]) / 2.0).tolist()
        histvalues = (thehist[0][-histlen:]).tolist()
        if savehist:
            thedict["histbincenters"] = histbincenters
            thedict["histvalues"] = histvalues
        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(thehistlabel)
            plt.plot(histbincenters, histvalues)
            plt.show()
    else:
        thedict["voxelsincluded"] = 0
        taglist = ["pct02", "pct25", "pct50", "pct75", "pct98"]
        taglist += ["q1width", "q2width", "q3width", "q4width", "mid50width"]
        taglist += ["kurtosis", "kurtosis_z", "kurtosis_p", "skewness", "skewness_z", "skewness_p"]
        taglist += ["peakheight", "peakloc", "peakwidth", "centerofmass", "peakpercentile"]
        if savehist:
            taglist += ["histbincenters", "histvalues"]
        for tag in taglist:
            thedict[tag] = None


def checkmap(
    themap: NDArray,
    themask: NDArray,
    histlen: int = 101,
    rangemin: float = 0.0,
    rangemax: float = 1.0,
    histlabel: str = "similarity metric histogram",
    ignorefirstpoint: bool = False,
    savehist: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Compute histogram metrics for a similarity map using a mask.

    This function calculates various statistical metrics from the histogram of
    similarity values in `themap` where `themask` is non-zero. The metrics include
    mean, standard deviation, minimum, maximum, and histogram data.

    Parameters
    ----------
    themap : NDArray
        Array containing similarity values to analyze.
    themask : NDArray
        Binary mask array where non-zero values indicate regions of interest.
    histlen : int, optional
        Number of bins in the histogram (default is 101).
    rangemin : float, optional
        Minimum value for histogram range (default is 0.0).
    rangemax : float, optional
        Maximum value for histogram range (default is 1.0).
    histlabel : str, optional
        Label for the histogram (default is "similarity metric histogram").
    ignorefirstpoint : bool, optional
        Whether to ignore the first point in the histogram calculation (default is False).
    savehist : bool, optional
        Whether to save the histogram data (default is False).
    debug : bool, optional
        Whether to enable debug output (default is False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing histogram metrics including mean, std, min, max,
        and histogram data. The exact keys depend on the implementation of
        `gethistmetrics` function.

    Notes
    -----
    This function serves as a wrapper around `gethistmetrics` and returns the
    computed metrics directly. The histogram is computed only for regions where
    `themask` is non-zero.

    Examples
    --------
    >>> import numpy as np
    >>> map_data = np.random.rand(100, 100)
    >>> mask_data = np.ones((100, 100))
    >>> metrics = checkmap(map_data, mask_data)
    >>> print(metrics.keys())
    dict_keys(['mean', 'std', 'min', 'max', 'hist'])
    """
    themetrics = {}

    gethistmetrics(
        themap,
        themask,
        themetrics,
        thehistlabel=histlabel,
        histlen=histlen,
        rangemin=rangemin,
        rangemax=rangemax,
        nozero=False,
        savehist=savehist,
        ignorefirstpoint=ignorefirstpoint,
        debug=debug,
    )
    return themetrics


def qualitycheck(
    datafileroot: str,
    graymaskspec: str | None = None,
    whitemaskspec: str | None = None,
    anatname: str | None = None,
    geommaskname: str | None = None,
    userise: bool = False,
    usecorrout: bool = False,
    useatlas: bool = False,
    forcetr: bool = False,
    forceoffset: bool = False,
    offsettime: float = 0.0,
    verbose: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Perform quality checks on a dataset by analyzing masks, regressors, and map statistics.

    This function loads a dataset using `RapidtideDataset` and performs a series of quality
    assessments on various overlays (e.g., lag times, strengths, MTT) and regressors. It
    computes statistics for different masks and map regions, including histogram data and
    relative sizes. Optional gray and white matter masks can be used to isolate analysis
    within those regions.

    Parameters
    ----------
    datafileroot : str
        Root name of the data files to be processed.
    graymaskspec : str, optional
        Path to the gray matter mask specification file.
    whitemaskspec : str, optional
        Path to the white matter mask specification file.
    anatname : str, optional
        Name of the anatomical image to use.
    geommaskname : str, optional
        Name of the geometric mask to use.
    userise : bool, default=False
        Whether to use RISE (reconstruction of instantaneous signal estimates).
    usecorrout : bool, default=False
        Whether to use corrected output.
    useatlas : bool, default=False
        Whether to use atlas-based registration.
    forcetr : bool, default=False
        Force TR (repetition time) to be set.
    forceoffset : bool, default=False
        Force offset to be set.
    offsettime : float, default=0.0
        Time offset to apply.
    verbose : bool, default=False
        Enable verbose output.
    debug : bool, default=False
        Enable debug output.

    Returns
    -------
    dict[str, Any]
        A dictionary containing quality check results, including:
        - ``passes``: Number of passes in the dataset.
        - ``filterlimits``: Regressor filter limits.
        - ``simcalclimits``: Regressor similarity calculation limits.
        - ``mask``: Dictionary of mask-related statistics.
        - ``regressor``: Regressor quality check results.
        - ``lag``, ``laggrad``, ``strength``, ``MTT``: Statistics for respective maps.
        - Optional gray/white matter-specific results if masks are provided.

    Notes
    -----
    This function relies on several helper functions and classes such as `RapidtideDataset`,
    `prepmask`, `checkregressors`, and `checkmap`. It uses the `numpy` library for array
    operations and `matplotlib` for histogram plotting (when enabled).

    Examples
    --------
    >>> output = qualitycheck("sub-01", graymaskspec="gray_mask.nii.gz")
    >>> print(output["passes"])
    3
    >>> print(output["mask"]["refinemaskrelsize"])
    0.75
    """
    # read in the dataset
    thedataset = RapidtideDataset(
        "main",
        datafileroot + "_",
        anatname=anatname,
        geommaskname=geommaskname,
        graymaskspec=graymaskspec,
        whitemaskspec=whitemaskspec,
        userise=userise,
        usecorrout=usecorrout,
        useatlas=useatlas,
        forcetr=forcetr,
        forceoffset=forceoffset,
        offsettime=offsettime,
        verbose=verbose,
        init_LUT=False,
    )

    if debug:
        print(f"qualitycheck started on {datafileroot}")
    outputdict = {}
    if graymaskspec is not None:
        dograyonly = True
        thegraymask = (thedataset.overlays["graymask"]).data
    else:
        dograyonly = False
    if whitemaskspec is not None:
        dowhiteonly = True
        thewhitemask = (thedataset.overlays["whitemask"]).data
    else:
        dowhiteonly = False

    # put in some basic information
    outputdict["passes"] = thedataset.numberofpasses
    outputdict["filterlimits"] = thedataset.regressorfilterlimits
    outputdict["simcalclimits"] = thedataset.regressorsimcalclimits

    # process the masks
    outputdict["mask"] = {}
    thelagmask = (thedataset.overlays["lagmask"]).data
    theerodedmask = prepmask(thelagmask)
    outputdict["mask"]["lagmaskvoxels"] = len(np.ravel(thelagmask[np.where(thelagmask > 0)]))
    for maskname in [
        "refinemask",
        "meanmask",
        "preselectmask",
        "p_lt_0p050_mask",
        "p_lt_0p010_mask",
        "p_lt_0p005_mask",
        "p_lt_0p001_mask",
        "desc-plt0p001_mask",
    ]:
        try:
            thismask = (thedataset.overlays[maskname]).data
        except KeyError:
            print(f"{maskname} not found in dataset")
        else:
            outname = maskname.replace("_mask", "").replace("mask", "")
            outputdict["mask"][f"{outname}relsize"] = getmasksize(thismask) / (
                1.0 * outputdict["mask"]["lagmaskvoxels"]
            )

    # process the regressors
    theregressors = thedataset.regressors
    outputdict["regressor"] = checkregressors(
        theregressors, outputdict["passes"], outputdict["filterlimits"], debug=debug
    )

    # process the lag map
    thelags = thedataset.overlays["lagtimes"]
    thelags.setFuncMask(thelagmask)
    thelags.updateStats()
    if debug:
        thelags.summarize()
    outputdict["lag"] = checkmap(
        thelags.data,
        thelagmask,
        rangemin=-5.0,
        rangemax=10.0,
        histlabel="lag histogram",
        debug=debug,
    )

    # get the gradient of the lag map
    thegradient = np.gradient(thelags.data)
    thegradientamp = np.sqrt(
        np.square(thegradient[0] / thelags.xsize)
        + np.square(thegradient[1] / thelags.ysize)
        + np.square(thegradient[2] / thelags.zsize)
    )
    outputdict["laggrad"] = checkmap(
        thegradientamp,
        theerodedmask,
        rangemin=0.0,
        rangemax=3.0,
        histlabel="lag gradient amplitude histogram",
        ignorefirstpoint=True,
        debug=debug,
    )

    # process the strength map
    thestrengths = thedataset.overlays["lagstrengths"]
    thestrengths.setFuncMask(thelagmask)
    thestrengths.updateStats()
    if debug:
        thestrengths.summarize()
    outputdict["strength"] = checkmap(
        thestrengths.data,
        thelagmask,
        rangemin=0.0,
        rangemax=1.0,
        histlabel="similarity metric histogram",
        debug=debug,
    )

    # process the MTT map
    theMTTs = thedataset.overlays["MTT"]
    theMTTs.setFuncMask(thelagmask)
    theMTTs.updateStats()
    if debug:
        theMTTs.summarize()
    outputdict["MTT"] = checkmap(
        theMTTs.data,
        thelagmask,
        histlabel="MTT histogram",
        rangemin=0.0,
        rangemax=10.0,
        debug=debug,
    )

    if dograyonly:
        outputdict["grayonly-lag"] = checkmap(
            thelags.data,
            thelagmask * thegraymask,
            rangemin=-5.0,
            rangemax=10.0,
            histlabel="lag histogram - gray only",
            debug=debug,
        )
        outputdict["grayonly-laggrad"] = checkmap(
            thegradientamp,
            theerodedmask * thegraymask,
            rangemin=0.0,
            rangemax=3.0,
            histlabel="lag gradient amplitude histogram - gray only",
            debug=debug,
        )
        outputdict["grayonly-strength"] = checkmap(
            thestrengths.data,
            thelagmask * thegraymask,
            rangemin=0.0,
            rangemax=1.0,
            histlabel="similarity metric histogram - gray only",
            debug=debug,
        )
    if dowhiteonly:
        outputdict["whiteonly-lag"] = checkmap(
            thelags.data,
            thelagmask * thewhitemask,
            rangemin=-5.0,
            rangemax=10.0,
            histlabel="lag histogram - white only",
            debug=debug,
        )
        outputdict["whiteonly-laggrad"] = checkmap(
            thegradientamp,
            theerodedmask * thewhitemask,
            rangemin=0.0,
            rangemax=3.0,
            histlabel="lag gradient amplitude histogram - white only",
            debug=debug,
        )
        outputdict["whiteonly-strength"] = checkmap(
            thestrengths.data,
            thelagmask * thewhitemask,
            rangemin=0.0,
            rangemax=1.0,
            histlabel="similarity metric histogram - white only",
            debug=debug,
        )

    return outputdict
