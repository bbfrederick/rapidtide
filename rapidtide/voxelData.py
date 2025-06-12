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
import copy

import numpy as np
from tqdm import tqdm

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.util as tide_util


class dataVolume:
    xsize = None
    ysize = None
    numslices = None
    numspatiallocs = None
    timepoints = None
    dtype = None
    dimensions = None
    data = None
    data_shm = None
    thepid = None

    def __init__(
        self,
        shape,
        shared=False,
        dtype=np.float64,
        thepid=0,
    ):
        if len(shape) == 3:
            self.xsize = int(shape[0])
            self.ysize = int(shape[1])
            self.numslices = int(shape[2])
            self.timepoints = 1
            self.dimensions = 3
        elif len(shape) == 4:
            self.xsize = int(shape[0])
            self.ysize = int(shape[1])
            self.numslices = int(shape[2])
            self.timepoints = int(shape[3])
            self.dimensions = 4
        else:
            print(f"illegal shape: {shape}")
        self.numspatiallocs = self.xsize * self.ysize * self.numslices
        self.dtype = dtype
        if not shared:
            self.data = np.zeros(shape, dtype=dtype)
        else:
            self.data, self.data_shm = tide_util.allocshared(
                shape, self.dtype, name=f"filtereddata_{thepid}"
            )
        return self.data

    def byvol(self):
        return self.data

    def byslice(self):
        if self.dimensions == 3:
            return self.data.reshape(self.xsize * self.ysize, -1)
        else:
            return self.data.reshape(self.xsize * self.ysize, self.numslices, -1)

    def byvoxel(self):
        if self.dimensions == 3:
            return self.data.reshape(self.numspatiallocs)
        else:
            return self.data.reshape(self.numspatiallocs, -1)

    def destroy(self):
        del self.data
        if self.data_shm is not None:
            tide_util.cleanup_shm(self.data_shm)


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
    dimensions = None
    realtimepoints = None
    xdim = None
    ydim = None
    slicethickness = None
    timestep = None
    thesizes = None
    thedims = None
    numslicelocs = None
    numspatiallocs = None
    nativespaceshape = None
    nativefmrishape = None
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
        # load the data
        self.load()

        if tide_io.checkiftext(self.filename):
            self.filetype = "text"
            self.nim_hdr = None
            self.nim_affine = None
            self.theshape = np.shape(self.nim_data)
            self.xsize = self.theshape[0]
            self.ysize = 1
            self.numslices = 1
            self.numslicelocs = None
            self.timepoints = int(self.theshape[1])
            self.thesizes = [0, int(self.xsize), 1, 1, int(self.timepoints)]
            self.toffset = 0.0
            self.numspatiallocs = int(self.xsize)
            self.nativespaceshape = self.xsize
            self.cifti_hdr = None
        else:
            if tide_io.checkifcifti(self.filename):
                self.filetype = "cifti"
                self.nim_affine = None
                self.numslicelocs = None
                self.timepoints = int(self.nim_data.shape[1])
                self.numspatiallocs = self.nim_data.shape[0]
                self.nativespaceshape = (1, 1, 1, 1, self.numspatiallocs)
            else:
                self.filetype = "nifti"
                self.nim_affine = self.nim.affine
                self.xsize, self.ysize, self.numslices, self.timepoints = tide_io.parseniftidims(
                    self.thedims
                )
                if self.timepoints == 1:
                    self.dimensions = 3
                else:
                    self.dimensions = 4
                self.numslicelocs = int(self.xsize) * int(self.ysize)
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

    def copyheader(self, numtimepoints=None, tr=None, toffset=None):
        if self.filetype == "text":
            return None
        else:
            thisheader = copy.deepcopy(self.nim_hdr)
            if self.filetype == "cifti":
                timeindex = thisheader["dim"][0] - 1
                spaceindex = thisheader["dim"][0]
                thisheader["dim"][timeindex] = numtimepoints
                thisheader["dim"][spaceindex] = self.numspatiallocs
            else:
                if numtimepoints is not None:
                    thisheader["dim"][4] = numtimepoints
                    if numtimepoints > 1:
                        thisheader["dim"][0] = 4
                    else:
                        thisheader["dim"][0] = 3
                        thisheader["pixdim"][4] = 1.0
                if toffset is not None:
                    thisheader["toffset"] = toffset
                if tr is not None:
                    thisheader["pixdim"][4] = tr
            return thisheader

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
        self.realtimepoints = self.validend - self.validstart + 1
        if self.filetype == "nifti":
            self.nativefmrishape = (self.xsize, self.ysize, self.numslices, self.realtimepoints)
        elif self.filetype == "cifti":
            self.nativefmrishape = (1, 1, 1, self.realtimepoints, self.numspatiallocs)
        else:
            self.nativefmrishape = (self.xsize, self.realtimepoints)

    def setvalidvoxels(self, validvoxels):
        self.validvoxels = validvoxels
        self.numvalidspatiallocs = np.shape(self.validvoxels)[0]

    def byvol(self):
        if not self.resident:
            self.load()
        return self.nim_data

    def byvoltrimmed(self):
        if not self.resident:
            self.load()
        if self.filetype == "nifti":
            if self.dimensions == 4 or self.filetype == "cifti" or self.filetype == "text":
                return self.nim_data[:, :, :, self.validstart : self.validend + 1]
            else:
                return self.nim_data[:, :, :]
        else:
            return self.nim_data[:, self.validstart : self.validend + 1]

    def byvoxel(self):
        if self.dimensions == 4 or self.filetype == "cifti" or self.filetype == "text":
            return self.byvoltrimmed().reshape(self.numspatiallocs, -1)
        else:
            return self.byvoltrimmed().reshape(self.numspatiallocs)

    def byslice(self):
        if self.dimensions == 4 or self.filetype == "cifti" or self.filetype == "text":
            return self.byvoltrimmed().reshape(self.numslicelocs, self.numslices, -1)
        else:
            return self.byvoltrimmed().reshape(self.numslicelocs, self.numslices)

    def validdata(self):
        if self.validvoxels is None:
            return self.byvoxel()
        else:
            return self.byvoxel()[self.validvoxels, :]

    # def validdatabyslice(self):
    #    if self.validvoxels is None:
    #       return self.byslice()
    #    else:
    #        return self.byvoxel()[self.validvoxels, :].reshape(self.numslicelocs, self.numslices, -1)

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
            sourcedata = self.byvol()
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
        print(f"\t{self.numslicelocs=}")
        print(f"\t{self.numspatiallocs=}")
        print(f"\t{self.nativespaceshape=}")
        print(f"\t{self.cifti_hdr=}")
        print(f"\t{self.filetype=}")
        print(f"\t{self.resident=}")
