#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick (except for some routines listed below)
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
import math
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter1d,
    distance_transform_edt,
    map_coordinates,
)
from skimage.filters import threshold_multiotsu
from skimage.segmentation import flood_fill

import rapidtide.io as tide_io


def interpolate_masked_voxels(data, mask):
    """
    Interpolates the values in a 3D numpy data array at the locations specified by the mask.
    Data in the masked region is replaced by interpolated values derived from the unmasked region.

    Parameters:
        data (np.ndarray): A 3D numpy array containing the data.
        mask (np.ndarray): A 3D binary numpy array of the same shape as `data`, where
                           masked voxels are 1, and unmasked voxels are 0.

    Returns:
        np.ndarray: A new 3D array with interpolated values replacing masked regions.
    """
    if data.shape != mask.shape:
        raise ValueError("Data and mask must have the same shape.")

    # Ensure mask is binary
    mask = mask.astype(bool)

    # Compute the distance transform (distance to nearest unmasked voxel)
    distance, indices = distance_transform_edt(~mask, return_indices=True)

    # Use indices of nearest unmasked voxels to replace masked values
    interpolated_data = data[tuple(indices)]

    return interpolated_data


def flood3d(
    image,
    newvalue,
):
    filledim = image * 0
    for slice in range(image.shape[2]):
        filledim[:, :, slice] = flood_fill(image[:, :, slice], (0, 0), newvalue, connectivity=1)
    return filledim


def invertedflood3D(image, newvalue):
    return image + newvalue - flood3d(image, newvalue)


def growregion(image, location, value, separatedimage, regionsize, debug=False):
    separatedimage[location[0], location[1], location[2]] = value
    regionsize += 1
    if debug:
        print(f"{location=}, {value=}")
    xstart = np.max([location[0] - 1, 0])
    xend = np.min([location[0] + 2, image.shape[0]])
    ystart = np.max([location[1] - 1, 0])
    yend = np.min([location[1] + 2, image.shape[1]])
    zstart = np.max([location[2] - 1, 0])
    zend = np.min([location[2] + 2, image.shape[2]])
    if debug:
        print(f"{xstart=}, {xend=}, {ystart=}, {yend=}, {zstart=}, {zend=}")
    for x in range(xstart, xend):
        for y in range(ystart, yend):
            for z in range(zstart, zend):
                if (x != location[0]) or (y != location[1]) or (z != location[2]):
                    if separatedimage[x, y, z] == 0 and image[x, y, z] == 1:
                        regionsize = growregion(
                            image, (x, y, z), value, separatedimage, regionsize
                        )
    return regionsize


def separateclusters(image, sizethresh=0, debug=False):
    separatedclusters = image * 0
    stop = False
    value = 1
    while not stop:
        regionsize = 0
        searchvoxels = image * np.where(separatedclusters == 0, 1, 0)
        seedvoxels = np.where(searchvoxels > 0)
        if debug:
            print(f"{seedvoxels=}")
        if len(seedvoxels[0]) > 0:
            location = (seedvoxels[0][0], seedvoxels[1][0], seedvoxels[2][0])
            if debug:
                print(f"growing from {location}")
            regionsize = growregion(
                image, location, value, separatedclusters, regionsize, debug=debug
            )
            if regionsize >= sizethresh:
                print(f"region:{value}: {regionsize=} - retained")
                value += 1
            else:
                # print(f"{regionsize=} - discarded")
                image[np.where(separatedclusters == value)] = 0
                separatedclusters[np.where(separatedclusters == value)] = 0
        else:
            stop = True
    return separatedclusters


#
#  The following functions (to the end of the file) were adapted from the PyDog library on GitHub
#  This is the license information for that library
#
#  BSD 2-Clause License
#
# Copyright (c) 2021, Chris Rorden
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def clamp(low, high, value):
    """bound an integer to a range

    Parameters
    ----------
    low : int
    high : int
    value : int

    Returns
    -------
    result : int
    """
    return max(low, min(high, value))


def dehaze(fdata, level, verbose=0):
    """use Otsu to threshold https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
        n.b. threshold used to mask image: dark values are zeroed, but result is NOT binary
    Parameters
    ----------
    fdata : numpy.memmap from Niimg-like object
        Image(s) to run DoG on (see :ref:`extracting_data`
        for a detailed description of the valid input types).
    level : int
        value 1..5 with larger values preserving more bright voxels
        dark_classes/total_classes
            1: 3/4
            2: 2/3
            3: 1/2
            4: 1/3
            5: 1/4
    verbose : :obj:`int`, optional
        Controls the amount of verbosity: higher numbers give more messages
        (0 means no messages). Default=0.
    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image`
    """
    level = clamp(1, 5, level)
    n_classes = abs(3 - level) + 2
    dark_classes = 4 - level
    dark_classes = clamp(1, 3, dark_classes)
    thresholds = threshold_multiotsu(fdata, n_classes)
    thresh = thresholds[dark_classes - 1]
    if verbose > 0:
        print("Zeroing voxels darker than {}".format(thresh))
    fdata[fdata < thresh] = 0
    return fdata


# https://github.com/nilearn/nilearn/blob/1607b52458c28953a87bbe6f42448b7b4e30a72f/nilearn/image/image.py#L164
def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of `arr`.

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        4D array, with image number as last dimension. 3D arrays are also
        accepted.

    affine : :class:`numpy.ndarray`
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
        If `fwhm='fast'`, the affine is not used and can be None.

    fwhm : scalar, :class:`numpy.ndarray`/:obj:`tuple`/:obj:`list`, 'fast' or None, optional
        Smoothing strength, as a full-width at half maximum, in millimeters.
        If a nonzero scalar is given, width is identical in all 3 directions.
        A :class:`numpy.ndarray`, :obj:`tuple`, or :obj:`list` must have 3 elements,
        giving the FWHM along each axis.
        If any of the elements is zero or None, smoothing is not performed
        along that axis.
        If  `fwhm='fast'`, a fast smoothing will be performed with a filter
        [0.2, 1, 0.2] in each direction and a normalisation
        to preserve the local average value.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed).

    ensure_finite : :obj:`bool`, optional
        If True, replace every non-finite values (like NaNs) by zero before
        filtering. Default=True.

    copy : :obj:`bool`, optional
        If True, input array is not modified. True by default: the filtering
        is not performed in-place. Default=True.

    Returns
    -------
    :class:`numpy.ndarray`
        Filtered `arr`.

    Notes
    -----
    This function is most efficient with arr in C order.

    """
    # Here, we have to investigate use cases of fwhm. Particularly, if fwhm=0.
    # See issue #1537
    if isinstance(fwhm, (int, float)) and (fwhm == 0.0):
        warnings.warn(
            "The parameter 'fwhm' for smoothing is specified "
            "as {0}. Setting it to None "
            "(no smoothing will be performed)".format(fwhm)
        )
        fwhm = None
    if arr.dtype.kind == "i":
        if arr.dtype == np.int64:
            arr = arr.astype(np.float64)
        else:
            arr = arr.astype(np.float32)  # We don't need crazy precision.
    if copy:
        arr = arr.copy()
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0
    if fwhm is not None:
        fwhm = np.asarray([fwhm]).ravel()
        fwhm = np.asarray([0.0 if elem is None else elem for elem in fwhm])
        affine = affine[:3, :3]  # Keep only the scale part.
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
        vox_size = np.sqrt(np.sum(affine**2, axis=0))
        # n.b. FSL specifies blur in sigma, SPM in FWHM
        # FWHM = sigma*sqrt(8*ln(2)) = sigma*2.3548.
        # convert fwhm to sd in voxels see https://github.com/0todd0000/spm1d
        fwhmvox = fwhm / vox_size
        sd = fwhmvox / math.sqrt(8 * math.log(2))
        for n, s in enumerate(sd):
            if s > 0.0:
                gaussian_filter1d(arr, s, output=arr, axis=n)
    return arr


def binary_zero_crossing(fdata):
    """binarize (negative voxels are zero)
    Parameters
    ----------
    fdata : numpy.memmap from Niimg-like object
    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image`
    """
    edge = np.where(fdata > 0.0, 1, 0)
    edge = distance_transform_edt(edge)
    edge[edge > 1] = 0
    edge[edge > 0] = 1
    edge = edge.astype("uint8")
    return edge


def difference_of_gaussian(fdata, affine, fwhmNarrow, verbose=0):
    """Apply Difference of Gaussian (DoG) filter.
    https://en.wikipedia.org/wiki/Difference_of_Gaussians
    https://en.wikipedia.org/wiki/Marr–Hildreth_algorithm
    D. Marr and E. C. Hildreth. Theory of edge detection. Proceedings of the Royal Society, London B, 207:187-217, 1980
    Parameters
    ----------
    fdata : numpy.memmap from Niimg-like object
    affine : :class:`numpy.ndarray`
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
    fwhmNarrow : int
        Narrow kernel width, in millimeters. Is an arbitrary ratio of wide to narrow kernel.
            human cortex about 2.5mm thick
            Large values yield smoother results
    verbose : :obj:`int`, optional
        Controls the amount of verbosity: higher numbers give more messages
        (0 means no messages). Default=0.
    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image`
    """

    # Hardcode 1.6 as ratio of wide versus narrow FWHM
    # Marr and Hildreth (1980) suggest narrow to wide ratio of 1.6
    # Wilson and Giese (1977) suggest narrow to wide ratio of 1.5
    fwhmWide = fwhmNarrow * 1.6
    # optimization: we will use the narrow Gaussian as the input to the wide filter
    fwhmWide = math.sqrt((fwhmWide * fwhmWide) - (fwhmNarrow * fwhmNarrow))
    if verbose > 0:
        print("Narrow/Wide FWHM {} / {}".format(fwhmNarrow, fwhmWide))
    imgNarrow = _smooth_array(fdata, affine, fwhmNarrow)
    imgWide = _smooth_array(imgNarrow, affine, fwhmWide)
    img = imgNarrow - imgWide
    img = binary_zero_crossing(img)
    return img


# for rapidtide purposes, this is the main entry point to the DOG calculation.
# We are operating on data in memory that are closely associated with the source
# NIFTI files, so the affine and sizes fields are easy to come by, but unlike the
# original library, we are not working directly with NIFTI images.
def calc_DoG(thedata, theaffine, thesizes, fwhm=3, verbose=0):
    """Find edges of a NIfTI image using the Difference of Gaussian (DoG).
    Parameters
    ----------
    thedata : 3D data array
        Image(s) to run DoG on (see :ref:`extracting_data`
        for a detailed description of the valid input types).
    fwhm : int
        Edge detection strength, as a full-width at half maximum, in millimeters.
    verbose : :obj:`int`, optional
        Controls the amount of verbosity: higher numbers give more messages
        (0 means no messages). Default=0.
    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image`
    """

    if verbose > 0:
        print("Input intensity range {}..{}".format(np.nanmin(thedata), np.nanmax(thedata)))
        print("Image shape {}x{}x{}".format(thesizes[1], thesizes[2], thesizes[3]))

    dehazed_data = dehaze(thedata, 3, verbose)
    return difference_of_gaussian(dehazed_data, theaffine, fwhm, verbose)


def getclusters(theimage, theaffine, thesizes, sizethresh=10, debug=False):
    if debug:
        print("Detecting clusters..")
        print(f"\t{theimage.shape=}")
        print(f"\t{theaffine=}")
        print(f"\t{thesizes=}")
        print(f"\t{sizethresh=}")
    return separateclusters(
        invertedflood3D(calc_DoG(theimage, theaffine, thesizes, fwhm=3), 1), sizethresh=sizethresh
    )


if __name__ == "__main__":
    """Apply Gaussian smooth to image
    Parameters
    ----------
    fnm : str
        NIfTI image to convert
    """
    if len(sys.argv) < 2:
        print("No filename provided: I do not know which image to convert!")
        sys.exit()
    fnm = sys.argv[1]
    img, img_data, img_hdr, thedims, thesizes = tide_io.readfromnifti(fnm)
    img_data = img_data.astype(np.float32)
    theaffine = img.affine
    dog = calc_DoG(img_data, theaffine, thesizes, fwhm=5, verbose=1)

    # update header
    out_hdr = copy.deepcopy(img_hdr)
    out_hdr.set_data_dtype(np.uint8)
    out_hdr["intent_code"] = 0
    out_hdr["scl_slope"] = 1.0
    out_hdr["scl_inter"] = 0.0
    out_hdr["cal_max"] = 0.0
    out_hdr["cal_min"] = 0.0
    pth, nm = os.path.split(fnm)
    if nm.endswith(".nii") or nm.endswith(".nii.gz"):
        if nm.endswith(".nii"):
            nm = nm[:-4]
        elif nm.endswith(".nii.gz"):
            nm = nm[:-7]
    if not pth:
        pth = "."
    outnm = pth + os.path.sep + "z" + nm
    tide_io.savetonifti(dog, out_hdr, outnm)

    filledim = invertedflood3D(dog, 1)
    outnm = pth + os.path.sep + "zfill" + nm
    tide_io.savetonifti(filledim, out_hdr, outnm)

    separatedimage = separateclusters(filledim, sizethresh=10)
    outnm = pth + os.path.sep + "zsep" + nm
    tide_io.savetonifti(separatedimage, out_hdr, outnm)

    otherseparatedimage = getclusters(img_data, theaffine, thesizes, sizethresh=10, debug=True)
    outnm = pth + os.path.sep + "zother" + nm
    tide_io.savetonifti(otherseparatedimage, out_hdr, outnm)
