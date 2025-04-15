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
import numpy as np
from tf_keras.src.layers.preprocessing.benchmarks.index_lookup_forward_benchmark import (
    get_vocab,
)
from tqdm import tqdm

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io


class VoxelData:
    nim = None
    nim_data = None
    nim_hdr = None
    nim_affine = None
    theshape = None
    xsize = None
    ysize = None
    numslices = None
    timepoints = None
    xdim = None
    ydim = None
    slicethickness = None
    timestep = None
    thesizes = None
    thedims = None
    numspatiallocs = None
    nativespaceshape = None
    validvoxels = None
    cifti_hdr = None
    filetype = None
    resident = False

    def __init__(
        self,
        filename,
        timestep=0.0,
        validstart=None,
        validend=None,
    ):

        self.filename = filename
        self.readdata(timestep, validstart, validend)

    def readdata(self, timestep, validstart, validend):
        self.load()

        if tide_io.checkiftext(self.filename):
            self.filetype = "text"
            self.nim_hdr = None
            self.nim_affine = None
            self.theshape = np.shape(self.nim_data)
            self.xsize = self.theshape[0]
            self.ysize = 1
            self.numslices = 1
            self.timepoints = self.theshape[1]
            self.thesizes = [0, int(self.xsize), 1, 1, int(self.timepoints)]
            self.toffset = 0.0
            self.numspatiallocs = int(self.xsize)
            self.nativespaceshape = self.xsize
            self.cifti_hdr = None
        else:
            if tide_io.checkifcifti(self.filename):
                self.filetype = "cifti"
                self.nim_affine = None
                self.timepoints = self.nim_data.shape[1]
                self.numspatiallocs = self.nim_data.shape[0]
                self.nativespaceshape = (1, 1, 1, 1, self.numspatiallocs)
            else:
                self.filetype = "nifti"
                self.nim_affine = self.nim.affine
                self.xsize, self.ysize, self.numslices, self.timepoints = tide_io.parseniftidims(
                    self.thedims
                )
                self.numspatiallocs = int(self.xsize) * int(self.ysize) * int(self.numslices)
                self.cifti_hdr = None
                self.nativespaceshape = (self.xsize, self.ysize, self.numslices)
            self.xdim, self.ydim, self.slicethickness, dummy = tide_io.parseniftisizes(
                self.thesizes
            )

        # correct some fields if necessary
        if self.filetype == "cifti":
            self.timestep = 0.72  # this is wrong and is a hack until I can parse CIFTI XML
            self.toffset = 0.0
        else:
            if self.filetype == "text":
                if timestep <= 0.0:
                    raise ValueError(
                        "for text file data input, you must use the -t option to set the timestep"
                    )
            else:
                if self.nim_hdr.get_xyzt_units()[1] == "msec":
                    self.timestep = self.thesizes[4] / 1000.0
                    self.toffset = self.nim_hdr["toffset"] / 1000.0
                else:
                    self.timestep = self.thesizes[4]
                    self.toffset = self.nim_hdr["toffset"]
        if timestep > 0.0:
            self.timestep = timestep

        self.setvalidtimes(validstart, validend)
        self.resident = True

    def getsizes(self):
        return self.xdim, self.ydim, self.slicethickness, self.timestep

    def getdims(self):
        return self.xsize, self.ysize, self.numslices, self.timepoints

    def unload(self):
        del self.nim_data
        del self.nim
        self.resident = False

    def load(self):
        if self.filetype is not None:
            print("reloading non-resident data")
        else:
            print(f"loading data from {self.filename}")
        if tide_io.checkiftext(self.filename):
            self.nim_data = tide_io.readvecs(self.filename)
            self.nim = None
        else:
            if tide_io.checkifcifti(self.filename):
                self.filetype = "cifti"
                (
                    dummy,
                    self.cifti_hdr,
                    self.nim_data,
                    self.nim_hdr,
                    self.thedims,
                    self.thesizes,
                    dummy,
                ) = tide_io.readfromcifti(self.filename)
                self.nim = None
            else:
                self.nim, self.nim_data, self.nim_hdr, self.thedims, self.thesizes = (
                    tide_io.readfromnifti(self.filename)
                )
        self.resident = True

    def setvalidtimes(self, validstart, validend):
        if validstart is None:
            self.validstart = 0
        else:
            self.validstart = validstart
        if validend is None:
            self.validend = self.timepoints - 1
        else:
            self.validend = validend

    def setvalidvoxels(self, validvoxels):
        self.validvoxels = validvoxels
        self.numvalidspatiallocs = np.shape(self.validvoxels)[0]

    def native(self):
        if not self.resident:
            self.load()
        return self.nim_data

    def nativetrimmed(self):
        if not self.resident:
            self.load()
        if self.filetype == "nifti":
            return self.nim_data[:, :, :, self.validstart : self.validend + 1]
        else:
            return self.nim_data[:, self.validstart: self.validend + 1]

    def voxelbytime(self):
        return self.nativetrimmed().reshape(self.numspatiallocs, -1)

    def getvalidvoxels(self):
        if self.validvoxels is None:
            return self.voxelbytime()
        else:
            return self.voxelbytime()[self.validvoxels, :]

    def smooth(
        self,
        gausssigma,
        brainmask=None,
        graymask=None,
        whitemask=None,
        premask=False,
        premasktissueonly=False,
        showprogressbar=False,
    ):
        # do spatial filtering if requested
        if self.filetype == "cifti" or self.filetype == "text":
            gausssigma = 0.0
        if gausssigma < 0.0:
            # set gausssigma automatically
            gausssigma = np.mean([self.xdim, self.ydim, self.slicethickness]) / 2.0
        if gausssigma > 0.0:
            # premask data if requested
            if premask:
                if premasktissueonly:
                    if (graymask is not None) and (whitemask is not None):
                        multmask = graymask + whitemask
                    else:
                        raise ValueError(
                            "ERROR: graymask and whitemask must be defined to use premasktissueonly - exiting"
                        )
                else:
                    if brainmask is not None:
                        multmask = brainmask
                    else:
                        raise ValueError(
                            "ERROR: brainmask must be defined to use premask - exiting"
                        )
                print(f"premasking timepoints {self.validstart} to {self.validend}")
                for i in tqdm(
                    range(self.validstart, self.validend + 1),
                    desc="Timepoint",
                    unit="timepoints",
                    disable=(not showprogressbar),
                ):
                    self.nim_data[:, :, :, i] *= multmask

            # now apply the filter
            print(
                f"applying gaussian spatial filter to timepoints {self.validstart} "
                f"to {self.validend} with sigma={gausssigma}"
            )
            sourcedata = self.native()
            for i in tqdm(
                range(self.validstart, self.validend + 1),
                desc="Timepoint",
                unit="timepoints",
                disable=(not showprogressbar),
            ):
                self.nim_data[:, :, :, i] = tide_filt.ssmooth(
                    self.xdim,
                    self.ydim,
                    self.slicethickness,
                    gausssigma,
                    sourcedata[:, :, :, i],
                )
        return gausssigma

    def summarize(self):
        print("Voxel data summary:")
        print(f"\t{self.nim=}")
        print(f"\t{self.nim_data.shape=}")
        print(f"\t{self.nim_hdr=}")
        print(f"\t{self.nim_affine=}")
        print(f"\t{self.theshape=}")
        print(f"\t{self.xsize=}")
        print(f"\t{self.ysize=}")
        print(f"\t{self.numslices=}")
        print(f"\t{self.timepoints=}")
        print(f"\t{self.timestep=}")
        print(f"\t{self.thesizes=}")
        print(f"\t{self.thedims=}")
        print(f"\t{self.numspatiallocs=}")
        print(f"\t{self.nativespaceshape=}")
        print(f"\t{self.cifti_hdr=}")
        print(f"\t{self.filetype=}")
        print(f"\t{self.resident=}")
