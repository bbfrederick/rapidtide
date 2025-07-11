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
import os
import sys

import nibabel as nib
import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
from rapidtide.Colortables import *
from rapidtide.stats import neglogpfromr_interpolator

atlases = {
    "ASPECTS": {"atlasname": "ASPECTS"},
    "ATT": {"atlasname": "ATTbasedFlowTerritories_split"},
    "JHU1": {"atlasname": "JHU-ArterialTerritoriesNoVent-LVL1"},
    "JHU2": {"atlasname": "JHU-ArterialTerritoriesNoVent-LVL2"},
}


def check_rt_spatialmatch(dataset1, dataset2):
    if (
        (dataset1.xdim == dataset2.xdim)
        and (dataset1.ydim == dataset2.ydim)
        and (dataset1.zdim == dataset2.zdim)
    ):
        dimmatch = True
    else:
        dimmatch = False
    if (
        (dataset1.xsize == dataset2.xsize)
        and (dataset1.ysize == dataset2.ysize)
        and (dataset1.zsize == dataset2.zsize)
    ):
        sizematch = True
    else:
        sizematch = False
    if dataset1.space == dataset2.space:
        spacematch = True
    else:
        spacematch = False
    if dataset1.affine == dataset2.affine:
        affinematch = True
    else:
        affinematch = False
    return dimmatch, sizematch, spacematch, affinematch


class Timecourse:
    "Store a timecourse and some information about it"

    def __init__(
        self,
        name,
        filename,
        namebase,
        samplerate,
        displaysamplerate,
        starttime=0.0,
        label=None,
        report=False,
        isbids=False,
        limits=None,
        verbose=0,
    ):
        self.verbose = verbose
        self.name = name
        self.filename = filename
        self.namebase = namebase
        self.samplerate = samplerate
        self.displaysamplerate = displaysamplerate
        self.starttime = starttime
        self.isbids = isbids
        self.limits = limits

        if label is None:
            self.label = name
        else:
            self.label = label
        self.report = report
        if self.verbose > 1:
            print("reading Timecourse ", self.name, " from ", self.filename, "...")
        self.readTimeData(self.label)

    def readTimeData(self, thename):
        if self.isbids:
            dummy, dummy, columns, indata, dummy, dummy = tide_io.readbidstsv(self.filename)
            try:
                self.timedata = indata[columns.index(thename), :]
            except ValueError:
                if self.verbose > 1:
                    print("no column named", thename, "in", columns)
                self.timedata = None
                return
        else:
            self.timedata = tide_io.readvec(self.filename)
        self.length = len(self.timedata)
        self.timeaxis = (
            np.linspace(0.0, self.length, num=self.length, endpoint=False) / self.samplerate
        ) - self.starttime
        if self.limits is not None:
            startpoint = np.max((int(np.round(self.limits[0] * self.samplerate, 0)), 0))
            endpoint = np.min((int(np.round(self.limits[1] * self.samplerate, 0)), self.length))
        else:
            startpoint = 0
            endpoint = self.length
        self.specaxis, self.specdata = tide_filt.spectrum(
            tide_math.corrnormalize(self.timedata[startpoint:endpoint]), self.samplerate
        )
        self.kurtosis, self.kurtosis_z, self.kurtosis_p = tide_stats.kurtosisstats(
            self.timedata[startpoint:endpoint]
        )
        self.skewness, self.skewness_z, self.skewness_p = tide_stats.skewnessstats(
            self.timedata[startpoint:endpoint]
        )

        if self.verbose > 1:
            print("Timecourse data range:", np.min(self.timedata), np.max(self.timedata))
            print("sample rate:", self.samplerate)
            print("Timecourse length:", self.length)
            print("timeaxis length:", len(self.timeaxis))
            print("kurtosis:", self.kurtosis)
            print("kurtosis_z:", self.kurtosis_z)
            print("kurtosis_p:", self.kurtosis_p)

            print()

    def summarize(self):
        print()
        print("Timecourse name:      ", self.name)
        print("    label:            ", self.label)
        print("    filename:         ", self.filename)
        print("    namebase:         ", self.namebase)
        print("    samplerate:       ", self.samplerate)
        print("    length:           ", self.length)
        print("    kurtosis:         ", self.kurtosis)
        print("    kurtosis_z:       ", self.kurtosis_z)
        print("    kurtosis_p:       ", self.kurtosis_p)


class Overlay:
    "Store a data overlay and some information about it"

    LUTname = None

    def __init__(
        self,
        name,
        filespec,
        namebase,
        funcmask=None,
        geommask=None,
        label=None,
        report=False,
        lut_state=gen_gray_state(),
        alpha=128,
        endalpha=0,
        display_state=True,
        invertonload=False,
        isaMask=False,
        init_LUT=True,
        verbose=1,
    ):
        self.verbose = verbose
        self.name = name
        if label is None:
            self.label = name
        else:
            self.label = label
        self.report = report
        self.filename, self.filevals = tide_io.processnamespec(
            filespec, "Including voxels where ", "in mask"
        )
        self.namebase = namebase
        if self.verbose > 1:
            print("reading map ", self.name, " from ", self.filename, "...")
        self.maskhash = 0
        self.invertonload = invertonload
        self.readImageData(isaMask=isaMask)
        self.mask = None
        self.maskeddata = None
        self.setFuncMask(funcmask, maskdata=False)
        self.setGeomMask(geommask, maskdata=False)
        self.maskData()
        self.updateStats()
        self.dispmin = self.robustmin
        self.dispmax = self.robustmax
        if init_LUT:
            self.gradient = getagradient()
            self.lut_state = lut_state
            self.display_state = display_state
            self.theLUT = None
            self.alpha = alpha
            self.endalpha = endalpha
            self.setLUT(self.lut_state, alpha=self.alpha, endalpha=self.endalpha)
        self.space = "unspecified"
        if (self.header["sform_code"] == 4) or (self.header["qform_code"] == 4):
            if ((self.xdim == 61) and (self.ydim == 73) and (self.zdim == 61)) or (
                (self.xdim == 91) and (self.ydim == 109) and (self.zdim == 91)
            ):
                self.space = "MNI152"
            else:
                self.space = "MNI152NLin2009cAsym"
        if self.header["sform_code"] != 0:
            self.affine = self.header.get_sform()
        elif self.header["qform_code"] != 0:
            self.affine = self.header.get_qform()
        else:
            self.affine = self.header.get_base_affine()
        self.invaffine = np.linalg.inv(self.affine)
        if self.verbose > 1:
            print("affine matrix:")
            print(self.affine)
        if self.affine[0][0] < 0.0:
            self.RLfactor = -1.0
            if self.verbose > 1:
                print("Overlay appears to be in neurological orientation")
        elif self.affine[0][0] > 0.0:
            self.RLfactor = 1.0
            if self.verbose > 1:
                print("Overlay appears to be in radiological orientation")
        else:
            self.RLfactor = 0.0
            if self.verbose > 1:
                print("Overlay has indeterminate orientation")
        self.xpos = 0
        self.ypos = 0
        self.zpos = 0
        self.tpos = 0
        self.xcoord = 0.0
        self.ycoord = 0.0
        self.zcoord = 0.0
        self.tcoord = 0.0

        if self.verbose > 1:
            print(
                "Overlay initialized:",
                self.name,
                self.filename,
                self.minval,
                self.dispmin,
                self.dispmax,
                self.maxval,
            )
        if self.verbose > 0:
            self.summarize()

    def duplicate(self, newname, newlabel, init_LUT=True):
        return Overlay(
            newname,
            self.filename,
            self.namebase,
            funcmask=self.funcmask,
            geommask=self.geommask,
            label=newlabel,
            report=self.report,
            init_LUT=init_LUT,
            verbose=self.verbose,
        )

    def updateStats(self):
        calcmaskeddata = self.data[np.where(self.mask != 0)]

        self.minval = calcmaskeddata.min()
        self.maxval = calcmaskeddata.max()
        (
            self.robustmin,
            self.pct25,
            self.pct50,
            self.pct75,
            self.robustmax,
        ) = tide_stats.getfracvals(calcmaskeddata, [0.02, 0.25, 0.5, 0.75, 0.98], nozero=False)
        self.histy, self.histx = np.histogram(
            calcmaskeddata, bins=np.linspace(self.minval, self.maxval, 200)
        )
        self.quartiles = [self.pct25, self.pct50, self.pct75]

        if self.verbose > 1:
            print(
                self.name,
                ":",
                self.minval,
                self.maxval,
                self.robustmin,
                self.robustmax,
                self.quartiles,
            )

    def setData(self, data, isaMask=False):
        self.data = data.copy()
        if isaMask:
            self.data[np.where(self.data < 0.5)] = 0.0
            self.data[np.where(self.data > 0.5)] = 1.0
        self.updateStats()

    def readImageData(self, isaMask=False):
        self.nim, self.data, self.header, self.dims, self.sizes = tide_io.readfromnifti(
            self.filename
        )
        if self.invertonload:
            self.data *= -1.0
        if isaMask:
            if self.filevals is None:
                self.data[np.where(self.data < 0.5)] = 0.0
                self.data[np.where(self.data > 0.5)] = 1.0
            else:
                tempmask = (0 * self.data).astype("uint16")
                for theval in self.filevals:
                    tempmask[np.where(self.data - theval == 0.0)] += 1
                self.data = np.where(tempmask > 0, 1, 0)
        if self.verbose > 1:
            print("Overlay data range:", np.min(self.data), np.max(self.data))
            print("header", self.header)
        self.xdim, self.ydim, self.zdim, self.tdim = tide_io.parseniftidims(self.dims)
        self.xsize, self.ysize, self.zsize, self.tr = tide_io.parseniftisizes(self.sizes)
        self.toffset = self.header["toffset"]
        if self.verbose > 1:
            print("Overlay dims:", self.xdim, self.ydim, self.zdim, self.tdim)
            print("Overlay sizes:", self.xsize, self.ysize, self.zsize, self.tr)
            print("Overlay toffset:", self.toffset)

    def setLabel(self, label):
        self.label = label

    def real2tr(self, time):
        return np.round((time - self.toffset) / self.tr, 0)

    def tr2real(self, tpos):
        return self.toffset + self.tr * tpos

    def real2vox(self, xcoord, ycoord, zcoord, time):
        x, y, z = nib.apply_affine(self.invaffine, [xcoord, ycoord, zcoord])
        t = self.real2tr(time)
        return (
            int(np.round(x, 0)),
            int(np.round(y, 0)),
            int(np.round(z, 0)),
            int(np.round(t, 0)),
        )

    def vox2real(self, xpos, ypos, zpos, tpos):
        return np.concatenate(
            (nib.apply_affine(self.affine, [xpos, ypos, zpos]), [self.tr2real(tpos)]),
            axis=0,
        )

    def setXYZpos(self, xpos, ypos, zpos):
        self.xpos = int(xpos)
        self.ypos = int(ypos)
        self.zpos = int(zpos)

    def setTpos(self, tpos):
        if tpos > self.tdim - 1:
            self.tpos = int(self.tdim - 1)
        else:
            self.tpos = int(tpos)

    def getFocusVal(self):
        if self.tdim > 1:
            return self.maskeddata[self.xpos, self.ypos, self.zpos, self.tpos]
        else:
            return self.maskeddata[self.xpos, self.ypos, self.zpos]

    def setFuncMask(self, funcmask, maskdata=True):
        self.funcmask = funcmask
        if self.funcmask is None:
            if self.tdim == 1:
                self.funcmask = 1.0 + 0.0 * self.data
            else:
                self.funcmask = 1.0 + 0.0 * self.data[:, :, :, 0]
        else:
            self.funcmask = funcmask.copy()
        if maskdata:
            self.maskData()

    def setGeomMask(self, geommask, maskdata=True):
        self.geommask = geommask
        if self.geommask is None:
            if self.tdim == 1:
                self.geommask = 1.0 + 0.0 * self.data
            else:
                self.geommask = 1.0 + 0.0 * self.data[:, :, :, 0]
        else:
            self.geommask = geommask.copy()
        if maskdata:
            self.maskData()

    def maskData(self):
        self.mask = self.geommask * self.funcmask
        maskhash = hash(self.mask.tobytes())
        # these operations are expensive, so only do them if the mask is changed
        if (maskhash == self.maskhash) and (self.verbose > 1):
            print("mask has not changed")
        else:
            if self.verbose > 1:
                print("mask changed - recalculating")
            self.maskeddata = self.data.copy()
            self.maskeddata[np.where(self.mask < 0.5)] = 0.0
            self.updateStats()
            self.maskhash = maskhash

    def setReport(self, report):
        self.report = report

    def setTR(self, trval):
        self.tr = trval

    def settoffset(self, toffset):
        self.toffset = toffset

    def setLUT(self, lut_state, alpha=255, endalpha=128):
        if alpha is not None:
            theticks = [lut_state["ticks"][0]]
            for theelement in lut_state["ticks"][1:-1]:
                theticks.append(
                    (
                        theelement[0],
                        (theelement[1][0], theelement[1][1], theelement[1][2], alpha),
                    )
                )
            theticks.append(lut_state["ticks"][-1])
            if self.verbose > 1:
                print("setLUT alpha adjustment:\n", theticks)
            self.lut_state = setendalpha({"ticks": theticks, "mode": lut_state["mode"]}, endalpha)
        else:
            self.lut_state = setendalpha(lut_state, endalpha)
        self.gradient.restoreState(self.lut_state)
        self.theLUT = self.gradient.getLookupTable(512, alpha=True)
        self.LUTname = lut_state["name"]

    def setisdisplayed(self, display_state):
        self.display_state = display_state

    def summarize(self):
        print()
        print("Overlay name:         ", self.name)
        print("    label:            ", self.label)
        print("    filename:         ", self.filename)
        print("    namebase:         ", self.namebase)
        print("    xdim:             ", self.xdim)
        print("    ydim:             ", self.ydim)
        print("    zdim:             ", self.zdim)
        print("    tdim:             ", self.tdim)
        print("    space:            ", self.space)
        if self.RLfactor < 0.0:
            print("    orientation:       neurological")
        elif self.RLfactor > 0.0:
            print("    orientation:       radiological")
        else:
            print("    orientation:       indeterminate")
        print("    toffset:          ", self.toffset)
        print("    tr:               ", self.tr)
        print("    min:              ", self.minval)
        print("    max:              ", self.maxval)
        print("    robustmin:        ", self.robustmin)
        print("    robustmax:        ", self.robustmax)
        print("    dispmin:          ", self.dispmin)
        print("    dispmax:          ", self.dispmax)
        print("    data shape:       ", np.shape(self.data))
        print("    masked data shape:", np.shape(self.maskeddata))
        if self.geommask is None:
            print("    geometric mask not set")
        else:
            print("    geometric mask is set")
        if self.funcmask is None:
            print("    functional mask not set")
        else:
            print("    functional mask is set")


class RapidtideDataset:
    "Store all the data associated with a rapidtide dataset"

    fileroot = None
    focusregressor = None
    regressorfilterlimits = None
    regressorsimcalclimits = None
    focusmap = None
    dispmaps = None
    allloadedmaps = None
    loadedfuncmasks = None
    loadedfuncmaps = None
    atlaslabels = None
    atlasname = None
    useatlas = False
    xdim = 0
    ydim = 0
    zdim = 0
    tdim = 0
    xsize = 0.0
    ysize = 0.0
    zsize = 0.0
    tr = 0.0
    space = None
    affine = None

    def __init__(
        self,
        name,
        fileroot,
        anatname=None,
        geommaskname=None,
        funcmaskname=None,
        graymaskspec=None,
        whitemaskspec=None,
        userise=False,
        usecorrout=False,
        useatlas=False,
        minimal=False,
        forcetr=False,
        forceoffset=False,
        coordinatespace="unspecified",
        offsettime=0.0,
        init_LUT=True,
        verbose=0,
    ):
        self.verbose = verbose
        self.name = name
        self.fileroot = fileroot
        self.anatname = anatname
        self.geommaskname = geommaskname
        self.funcmaskname = funcmaskname
        self.graymaskspec = graymaskspec
        self.whitemaskspec = whitemaskspec
        self.userise = userise
        self.usecorrout = usecorrout
        self.useatlas = useatlas
        self.forcetr = forcetr
        self.forceoffset = forceoffset
        self.coordinatespace = coordinatespace
        self.offsettime = offsettime
        self.init_LUT = init_LUT
        self.referencedir = tide_util.findreferencedir()

        # check which naming style the dataset has
        if os.path.isfile(self.fileroot + "desc-maxtime_map.nii.gz"):
            self.bidsformat = True
            self.newstylenames = True
        else:
            self.bidsformat = False
            if os.path.isfile(self.fileroot + "fitmask.nii.gz"):
                self.newstylenames = True
            else:
                self.newstylenames = False
        if self.verbose > 1:
            print(
                "RapidtideDataset init: self.bidsformat=",
                self.bidsformat,
                "self.newstylenames=",
                self.newstylenames,
            )

        self.setupregressors()
        self.setupoverlays()

    def _loadregressors(self):
        self.focusregressor = None
        for thisregressor in self.regressorspecs:
            if os.path.isfile(self.fileroot + thisregressor[2]):
                if self.verbose > 1:
                    print("file: ", self.fileroot + thisregressor[2], " exists - reading...")
                thepath, thebase = os.path.split(self.fileroot + thisregressor[2])
                theregressor = Timecourse(
                    thisregressor[0],
                    self.fileroot + thisregressor[2],
                    thebase,
                    thisregressor[3],
                    thisregressor[4],
                    label=thisregressor[1],
                    starttime=thisregressor[5],
                    isbids=self.bidsformat,
                    limits=self.regressorsimcalclimits,
                    verbose=self.verbose,
                )
                if theregressor.timedata is not None:
                    self.regressors[thisregressor[0]] = copy.deepcopy(theregressor)
                    if self.verbose > 0:
                        theregressor.summarize()
                if self.focusregressor is None:
                    self.focusregressor = thisregressor[0]
            else:
                if thisregressor[6]:
                    raise FileNotFoundError(f"regressor file {self.fileroot + thisregressor[2]} does not exist")
                else:
                    if self.verbose > 1:
                        print(
                            "file: ",
                            self.fileroot + thisregressor[2],
                            " does not exist - skipping...",
                        )

    def _loadfuncmaps(self):
        mapstoinvert = ["varChange"]
        self.loadedfuncmaps = []
        xdim = 0
        ydim = 0
        zdim = 0
        for mapname, mapfilename in self.funcmaps:
            if self.verbose > 1:
                print(f"loading {mapname} from {mapfilename}")
            if os.path.isfile(self.fileroot + mapfilename + ".nii.gz"):
                if self.verbose > 1:
                    print(
                        "file: ",
                        self.fileroot + mapfilename + ".nii.gz",
                        " exists - reading...",
                    )
                thepath, thebase = os.path.split(self.fileroot)
                if mapname in mapstoinvert:
                    invertthismap = True
                else:
                    invertthismap = False
                self.overlays[mapname] = Overlay(
                    mapname,
                    self.fileroot + mapfilename + ".nii.gz",
                    thebase,
                    init_LUT=self.init_LUT,
                    report=True,
                    invertonload=invertthismap,
                    verbose=self.verbose,
                )
                if xdim == 0:
                    xdim = self.overlays[mapname].xdim
                    ydim = self.overlays[mapname].ydim
                    zdim = self.overlays[mapname].zdim
                    tdim = self.overlays[mapname].tdim
                    xsize = self.overlays[mapname].xsize
                    ysize = self.overlays[mapname].ysize
                    zsize = self.overlays[mapname].zsize
                    tr = self.overlays[mapname].tr
                else:
                    if (
                        xdim != self.overlays[mapname].xdim
                        or ydim != self.overlays[mapname].ydim
                        or zdim != self.overlays[mapname].zdim
                    ):
                        print("overlay dimensions do not match!")
                        sys.exit()
                    if (
                        xsize != self.overlays[mapname].xsize
                        or ysize != self.overlays[mapname].ysize
                        or zsize != self.overlays[mapname].zsize
                    ):
                        print("overlay voxel sizes do not match!")
                        sys.exit()
                self.loadedfuncmaps.append(mapname)
            else:
                if self.verbose > 1:
                    print("map: ", self.fileroot + mapfilename + ".nii.gz", " does not exist!")
        if self.verbose > 1:
            print("functional maps loaded:", self.loadedfuncmaps)

    def _loadfuncmasks(self):
        self.loadedfuncmasks = []
        for maskname, maskfilename in self.funcmasks:
            if self.verbose > 1:
                print(f"loading {maskname} from {maskfilename}")
            if os.path.isfile(self.fileroot + maskfilename + ".nii.gz"):
                thepath, thebase = os.path.split(self.fileroot)
                self.overlays[maskname] = Overlay(
                    maskname,
                    self.fileroot + maskfilename + ".nii.gz",
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                self.loadedfuncmasks.append(maskname)
            else:
                if self.verbose > 1:
                    print(
                        "mask: ",
                        self.fileroot + maskfilename + ".nii.gz",
                        " does not exist!",
                    )
        if self.verbose > 1:
            print(self.loadedfuncmasks)

    def _genpmasks(self, pvals=[0.05, 0.01, 0.005, 0.001]):
        for thepval in pvals:
            maskname = f"p_lt_{thepval:.3f}_mask".replace("0.0", "0p0")
            nlpthresh = -np.log10(thepval)
            if self.verbose > 1:
                print(f"generating {maskname} from neglog10p")
            self.overlays[maskname] = self.overlays[self.loadedfuncmasks[-1]].duplicate(
                maskname, None, self.init_LUT
            )
            self.overlays[maskname].setData(
                np.where(self.overlays["neglog10p"].data > nlpthresh, 1.0, 0.0), isaMask=True
            )
            self.loadedfuncmasks.append(maskname)
        if self.verbose > 1:
            print(self.loadedfuncmasks)

    def _loadgeommask(self):
        if self.geommaskname is not None:
            if os.path.isfile(self.geommaskname):
                thepath, thebase = os.path.split(self.geommaskname)
                self.overlays["geommask"] = Overlay(
                    "geommask",
                    self.geommaskname,
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.geommaskname, " as geometric mask")
                # allloadedmaps.append('geommask')
                return True
        elif self.coordinatespace == "MNI152":
            try:
                fsldir = os.environ["FSLDIR"]
            except KeyError:
                fsldir = None
            if self.verbose > 1:
                print("fsldir set to ", fsldir)
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                if fsldir is not None:
                    self.geommaskname = os.path.join(
                        fsldir, "data", "standard", "MNI152_T1_2mm_brain_mask.nii.gz"
                    )
            elif self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                self.geommaskname = os.path.join(
                    self.referencedir, "MNI152_T1_3mm_brain_mask_bin.nii.gz"
                )
            if os.path.isfile(self.geommaskname):
                thepath, thebase = os.path.split(self.geommaskname)
                self.overlays["geommask"] = Overlay(
                    "geommask",
                    self.geommaskname,
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.geommaskname, " as background")
                # allloadedmaps.append('geommask')
                return True
            else:
                if self.verbose > 1:
                    print("no geometric mask loaded")
                return False
        else:
            if self.verbose > 1:
                print("no geometric mask loaded")
            return False

    def _loadanatomics(self):
        try:
            fsldir = os.environ["FSLDIR"]
        except KeyError:
            fsldir = None

        if self.anatname is not None:
            if self.verbose > 1:
                print("using user input anatomic name")
            if os.path.isfile(self.anatname):
                thepath, thebase = os.path.split(self.anatname)
                self.overlays["anatomic"] = Overlay(
                    "anatomic",
                    self.anatname,
                    thebase,
                    init_LUT=self.init_LUT,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.anatname, " as background")
                # allloadedmaps.append('anatomic')
                return True
            else:
                if self.verbose > 1:
                    print("specified file does not exist!")
                return False
        elif os.path.isfile(self.fileroot + "highres_head.nii.gz"):
            if self.verbose > 1:
                print("using hires_head anatomic name")
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "highres_head.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print("using ", self.fileroot + "highres_head.nii.gz", " as background")
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + "highres.nii.gz"):
            if self.verbose > 1:
                print("using hires anatomic name")
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "highres.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print("using ", self.fileroot + "highres.nii.gz", " as background")
            # allloadedmaps.append('anatomic')
            return True
        elif self.coordinatespace == "MNI152":
            mniname = ""
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                if self.verbose > 1:
                    print("using 2mm MNI anatomic name")
                if fsldir is not None:
                    mniname = os.path.join(fsldir, "data", "standard", "MNI152_T1_2mm.nii.gz")
            elif self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                if self.verbose > 1:
                    print("using 3mm MNI anatomic name")
                mniname = os.path.join(self.referencedir, "MNI152_T1_3mm.nii.gz")
            if os.path.isfile(mniname):
                self.overlays["anatomic"] = Overlay(
                    "anatomic",
                    mniname,
                    "MNI152",
                    init_LUT=self.init_LUT,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", mniname, " as background")
                # allloadedmaps.append('anatomic')
                return True
            else:
                if self.verbose > 1:
                    print("xsize, ysize, zsize=", self.xsize, self.ysize, self.zsize)
                    print("MNI template brain ", mniname, " not loaded")
        elif self.coordinatespace == "MNI152NLin2009cAsym":
            mniname = ""
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                if self.verbose > 1:
                    print("using 2mm MNI anatomic name")
                if fsldir is not None:
                    mniname = os.path.join(
                        self.referencedir, "mni_icbm152_nlin_asym_09c_2mm.nii.gz"
                    )
            elif self.xsize == 1.0 and self.ysize == 1.0 and self.zsize == 1.0:
                if self.verbose > 1:
                    print("using 1mm MNI anatomic name")
                mniname = os.path.join(self.referencedir, "mni_icbm152_nlin_asym_09c_1mm.nii.gz")
            if os.path.isfile(mniname):
                self.overlays["anatomic"] = Overlay(
                    "anatomic",
                    mniname,
                    "MNI152NLin2009cAsym",
                    init_LUT=self.init_LUT,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", mniname, " as background")
                # allloadedmaps.append('anatomic')
                return True
            else:
                if self.verbose > 1:
                    print("xsize, ysize, zsize=", self.xsize, self.ysize, self.zsize)
                    print("MNI template brain ", mniname, " not loaded")
        elif os.path.isfile(self.fileroot + "mean.nii.gz"):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "mean.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print("using ", self.fileroot + "mean.nii.gz", " as background")
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + "meanvalue.nii.gz"):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "meanvalue.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print("using ", self.fileroot + "meanvalue.nii.gz", " as background")
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + "desc-unfiltmean_map.nii.gz"):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "desc-unfiltmean_map.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print(
                    "using ",
                    self.fileroot + "desc-unfiltmean_map.nii.gz",
                    " as background",
                )
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + "desc-mean_map.nii.gz"):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays["anatomic"] = Overlay(
                "anatomic",
                self.fileroot + "desc-mean_map.nii.gz",
                thebase,
                init_LUT=self.init_LUT,
                verbose=self.verbose,
            )
            if self.verbose > 1:
                print(
                    "using ",
                    self.fileroot + "desc-mean_map.nii.gz",
                    " as background",
                )
            # allloadedmaps.append('anatomic')
            return True
        else:
            if self.verbose > 1:
                print("no anatomic image loaded")
            return False

    def _loadgraymask(self):
        if self.graymaskspec is not None:
            filename, dummy = tide_io.parsefilespec(self.graymaskspec)
            if os.path.isfile(filename):
                thepath, thebase = os.path.split(self.graymaskspec)
                self.overlays["graymask"] = Overlay(
                    "graymask",
                    self.graymaskspec,
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.graymaskspec, " as gray matter mask")
                # allloadedmaps.append('geommask')
                return True
        else:
            if self.verbose > 1:
                print("no gray mask loaded")
            return False

    def _loadwhitemask(self):
        if self.whitemaskspec is not None:
            filename, dummy = tide_io.parsefilespec(self.whitemaskspec)
            if os.path.isfile(filename):
                thepath, thebase = os.path.split(self.whitemaskspec)
                self.overlays["whitemask"] = Overlay(
                    "whitemask",
                    self.whitemaskspec,
                    thebase,
                    init_LUT=self.init_LUT,
                    isaMask=True,
                    verbose=self.verbose,
                )
                if self.verbose > 1:
                    print("using ", self.whitemaskspec, " as white matter mask")
                # allloadedmaps.append('geommask')
                return True
        else:
            if self.verbose > 1:
                print("no white mask loaded")
            return False

    def setupregressors(self):
        # load the regressors
        self.regressors = {}
        self.therunoptions = tide_io.readoptionsfile(self.fileroot + "desc-runoptions_info")
        if self.verbose > 1:
            print("regressor similarity calculation limits:", self.regressorsimcalclimits)
        try:
            self.regressorfilterlimits = (
                float(self.therunoptions["lowerpass"]),
                float(self.therunoptions["upperpass"]),
            )
        except KeyError:
            self.regressorfilterlimits = (0.0, 100.0)
        if self.verbose > 1:
            print("regressor filter limits:", self.regressorfilterlimits)
        try:
            self.fmrifreq = float(self.therunoptions["fmrifreq"])
        except KeyError:
            self.fmrifreq = 1.0
        try:
            self.inputfreq = float(self.therunoptions["inputfreq"])
        except KeyError:
            self.inputfreq = 1.0
        try:
            self.inputstarttime = float(self.therunoptions["inputstarttime"])
        except KeyError:
            self.inputstarttime = 0.0
        try:
            self.oversampfactor = int(self.therunoptions["oversampfactor"])
        except KeyError:
            self.oversampfactor = 1
        try:
            self.similaritymetric = self.therunoptions["similaritymetric"]
        except KeyError:
            self.similaritymetric = "correlation"
        try:
            self.regressorsimcalclimits = (
                float((1.0 / self.fmrifreq) * self.therunoptions["validsimcalcstart"]),
                float((1.0 / self.fmrifreq) * self.therunoptions["validsimcalcend"]),
            )
        except KeyError:
            self.regressorsimcalclimits = (0.0, 10000000.0)
        try:
            self.numberofpasses = int(self.therunoptions["actual_passes"])
        except KeyError:
            self.numberofpasses = int(self.therunoptions["passes"])
        if self.numberofpasses > 4:
            secondtolast = self.numberofpasses - 1
            last = self.numberofpasses
        else:
            secondtolast = 3
            last = 4
        if self.bidsformat:
            self.regressorspecs = [
                [
                    "prefilt",
                    "prefilt",
                    "desc-initialmovingregressor_timeseries.json",
                    self.inputfreq,
                    self.inputfreq,
                    self.inputstarttime,
                    True,
                ],
                [
                    "postfilt",
                    "postfilt",
                    "desc-initialmovingregressor_timeseries.json",
                    self.inputfreq,
                    self.inputfreq,
                    self.inputstarttime,
                    True,
                ],
                [
                    "pass1",
                    "pass1",
                    "desc-oversampledmovingregressor_timeseries.json",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    True,
                ],
                [
                    "pass2",
                    "pass2",
                    "desc-oversampledmovingregressor_timeseries.json",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
                [
                    "pass3",
                    "pass{:d}".format(secondtolast),
                    "desc-oversampledmovingregressor_timeseries.json",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
                [
                    "pass4",
                    "pass{:d}".format(last),
                    "desc-oversampledmovingregressor_timeseries.json",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
            ]
        else:
            self.regressorspecs = [
                [
                    "prefilt",
                    "prefilt",
                    "reference_origres_prefilt.txt",
                    self.inputfreq,
                    self.inputfreq,
                    self.inputstarttime,
                    True,
                ],
                [
                    "postfilt",
                    "postfilt",
                    "reference_origres.txt",
                    self.inputfreq,
                    self.inputfreq,
                    self.inputstarttime,
                    True,
                ],
                [
                    "pass1",
                    "pass1",
                    "reference_resampres_pass1.txt",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    True,
                ],
                [
                    "pass2",
                    "pass2",
                    "reference_resampres_pass2.txt",
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
                [
                    "pass3",
                    "pass{:d}".format(secondtolast),
                    "reference_resampres_pass{:d}.txt".format(secondtolast),
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
                [
                    "pass4",
                    "pass{:d}".format(last),
                    "reference_resampres_pass{:d}.txt".format(last),
                    self.fmrifreq * self.oversampfactor,
                    self.fmrifreq,
                    0.0,
                    False,
                ],
            ]
        self._loadregressors()

    def getregressors(self):
        return self.regressors

    def setfocusregressor(self, whichregressor):
        try:
            testregressor = self.regressors[whichregressor]
            self.focusregressor = whichregressor
        except KeyError:
            self.focusregressor = "prefilt"

    def setupoverlays(self):
        # load the overlays
        self.overlays = {}

        # first the functional maps
        if self.bidsformat:
            self.funcmaps = [
                ["lagtimes", "desc-maxtime_map"],
                ["lagtimesrefined", "desc-maxtimerefined_map"],
                ["timepercentile", "desc-timepercentile_map"],
                ["lagstrengths", "desc-maxcorr_map"],
                ["lagstrengthsrefined", "desc-maxcorrrefined_map"],
                ["lagsigma", "desc-maxwidth_map"],
                ["MTT", "desc-MTT_map"],
                ["R2", "desc-lfofilterR2_map"],
                ["CoV", "desc-CoV_map"],
                ["confoundR2", "desc-confoundfilterR2_map"],
                ["varBefore", "desc-lfofilterInbandVarianceBefore_map"],
                ["varAfter", "desc-lfofilterInbandVarianceAfter_map"],
                ["varChange", "desc-lfofilterInbandVarianceChange_map"],
                ["fitNorm", "desc-lfofilterNorm_map"],
                ["fitcoff", "desc-lfofilterCoeff_map"],
                ["neglog10p", "desc-neglog10p_map"],
                ["delayoffset", "desc-delayoffset_map"],
            ]
            if self.usecorrout:
                self.funcmaps += [["corrout", "desc-corrout_info"]]
                # self.funcmaps += [['gaussout', 'desc-gaussout_info']]
                self.funcmaps += [["failimage", "desc-corrfitfailreason_info"]]

        else:
            if self.newstylenames:
                self.funcmaps = [
                    ["lagtimes", "lagtimes"],
                    ["lagstrengths", "lagstrengths"],
                    ["lagsigma", "lagsigma"],
                    ["MTT", "MTT"],
                    ["R2", "R2"],
                    ["fitNorm", "fitNorm"],
                    ["fitcoff", "fitCoeff"],
                ]
                if self.usecorrout:
                    self.funcmaps += [["corrout", "corrout"]]
                    # self.funcmaps += [['gaussout', 'gaussout']]
                    self.funcmaps += [["failimage", "corrfitfailreason"]]

            else:
                self.funcmaps = [
                    ["lagtimes", "lagtimes"],
                    ["lagstrengths", "lagstrengths"],
                    ["lagsigma", "lagsigma"],
                    ["MTT", "MTT"],
                    ["R2", "R2"],
                    ["fitNorm", "fitNorm"],
                    ["fitcoff", "fitcoff"],
                ]
                if self.userise:
                    self.funcmaps = [
                        ["lagtimes", "lagtimes"],
                        ["lagstrengths", "lagstrengths"],
                        ["lagsigma", "lagsigma"],
                        ["MTT", "MTT"],
                        ["R2", "R2"],
                        ["risetime_epoch_0", "risetime_epoch_0"],
                        ["starttime_epoch_0", "starttime_epoch_0"],
                        ["maxamp_epoch_0", "maxamp_epoch_0"],
                    ]
                if self.usecorrout:
                    self.funcmaps += [["corrout", "corrout"]]
                    # self.funcmaps += [['gaussout', 'gaussout']]
                    self.funcmaps += [["failimage", "failimage"]]

        self._loadfuncmaps()
        for themap in self.loadedfuncmaps:
            if self.forcetr:
                self.overlays[themap].setTR(self.trval)
            if self.forceoffset:
                self.overlays[themap].settoffset(self.offsettime)
            if self.overlays[themap].space == "MNI152":
                self.coordinatespace = "MNI152"
            elif self.overlays[themap].space == "MNI152NLin2009cAsym":
                self.coordinatespace = "MNI152NLin2009cAsym"

        # report results of load
        if self.verbose > 1:
            print("loaded functional maps: ", self.loadedfuncmaps)

        self.allloadedmaps = list(self.loadedfuncmaps)
        self.dispmaps = list(self.loadedfuncmaps)

        # extract some useful information about this dataset from the focusmap
        self.focusmap = "lagtimes"

        self.xdim = self.overlays[self.focusmap].xdim
        self.ydim = self.overlays[self.focusmap].ydim
        self.zdim = self.overlays[self.focusmap].zdim
        self.tdim = self.overlays[self.focusmap].tdim
        self.xsize = self.overlays[self.focusmap].xsize
        self.ysize = self.overlays[self.focusmap].ysize
        self.zsize = self.overlays[self.focusmap].zsize
        self.tr = self.overlays[self.focusmap].tr

        # then load the anatomics
        if self._loadanatomics():
            self.allloadedmaps.append("anatomic")

        # then the functional masks
        if self.bidsformat:
            self.funcmasks = [
                ["lagmask", "desc-corrfit_mask"],
                ["refinemask", "desc-refine_mask"],
                ["meanmask", "desc-globalmean_mask"],
                ["brainmask", "desc-brainmask_mask"],
                ["preselectmask", "desc-globalmeanpreselect_mask"],
            ]
            if not ("neglog10p" in self.loadedfuncmaps):
                # load p maps manually
                self.funcmasks += [
                    ["p_lt_0p050_mask", "desc-plt0p050_mask"],
                    ["p_lt_0p010_mask", "desc-plt0p010_mask"],
                    ["p_lt_0p005_mask", "desc-plt0p005_mask"],
                    ["p_lt_0p001_mask", "desc-plt0p001_mask"],
                ]
        else:
            if self.newstylenames:
                self.funcmasks = [
                    ["lagmask", "fitmask"],
                    ["refinemask", "refinemask"],
                    ["meanmask", "meanmask"],
                    ["p_lt_0p050_mask", "p_lt_0p050_mask"],
                    ["p_lt_0p010_mask", "p_lt_0p010_mask"],
                    ["p_lt_0p005_mask", "p_lt_0p005_mask"],
                    ["p_lt_0p001_mask", "p_lt_0p001_mask"],
                ]
            else:
                self.funcmasks = [
                    ["lagmask", "lagmask"],
                    ["refinemask", "refinemask"],
                    ["meanmask", "meanmask"],
                    ["p_lt_0p050_mask", "p_lt_0p050_mask"],
                    ["p_lt_0p010_mask", "p_lt_0p010_mask"],
                    ["p_lt_0p005_mask", "p_lt_0p005_mask"],
                    ["p_lt_0p001_mask", "p_lt_0p001_mask"],
                ]
        self._loadfuncmasks()
        if "neglog10p" in self.loadedfuncmaps:
            # generate p maps on the fly
            self._genpmasks()

        # then the geometric masks
        if self._loadgeommask():
            self.allloadedmaps.append("geommask")

        # then the tissue masks
        if self._loadgraymask():
            self.allloadedmaps.append("graymask")
        if self._loadwhitemask():
            self.allloadedmaps.append("whitemask")

        if self.useatlas and (
            (self.coordinatespace == "MNI152")
            or (self.coordinatespace == "MNI152NLin6")
            or (self.coordinatespace == "MNI152NLin2009cAsym")
        ):
            self.atlasshortname = "JHU1"
            self.atlasname = atlases[self.atlasshortname]["atlasname"]
            self.atlaslabels = tide_io.readlabels(
                os.path.join(self.referencedir, self.atlasname + "_regions.txt")
            )
            if self.verbose > 1:
                print(self.atlaslabels)
            self.atlasniftiname = None
            if self.coordinatespace == "MNI152":
                spacename = "_space-MNI152NLin6Asym"
                if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                    self.atlasniftiname = os.path.join(
                        self.referencedir, self.atlasname + spacename + "_2mm.nii.gz"
                    )
                    self.atlasmaskniftiname = os.path.join(
                        self.referencedir, self.atlasname + spacename + "_2mm_mask.nii.gz"
                    )
                if self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                    self.atlasniftiname = os.path.join(
                        self.referencedir, self.atlasname + spacename + "_3mm.nii.gz"
                    )
                    self.atlasmaskniftiname = os.path.join(
                        self.referencedir, self.atlasname + spacename + "_3mm_mask.nii.gz"
                    )
            else:
                pass
                """if xsize == 2.0 and ysize == 2.0 and zsize == 2.0:
                    atlasniftiname = os.path.join(referencedir, atlasname + '_nlin_asym_09c_2mm.nii.gz')
                    atlasmaskniftiname = os.path.join(referencedir, atlasname + '_nlin_asym_09c_2mm_mask.nii.gz')"""
            if self.atlasniftiname is not None:
                if os.path.isfile(self.atlasniftiname):
                    self.overlays["atlas"] = Overlay(
                        "atlas",
                        self.atlasniftiname,
                        self.atlasname,
                        report=True,
                        init_LUT=self.init_LUT,
                        verbose=self.verbose,
                    )
                    self.overlays["atlasmask"] = Overlay(
                        "atlasmask",
                        self.atlasmaskniftiname,
                        self.atlasname,
                        init_LUT=self.init_LUT,
                        report=True,
                        verbose=self.verbose,
                    )
                    self.allloadedmaps.append("atlas")
                    self.dispmaps.append("atlas")
                else:
                    print(
                        self.atlasname + " template: ",
                        self.atlasniftiname,
                        " does not exist!",
                    )

        try:
            test = self.overlays["atlas"]
            if self.verbose > 1:
                print("there is an atlas")
        except KeyError:
            if self.verbose > 1:
                print("there is not an atlas")
        if self.verbose > 1:
            print("done")

    def getoverlays(self):
        return self.overlays

    def setfocusmap(self, whichmap):
        try:
            testmap = self.overlays[whichmap]
            self.focusmap = whichmap
        except KeyError:
            self.focusmap = "lagtimes"

    def setFuncMaskName(self, maskname):
        self.funcmaskname = maskname
