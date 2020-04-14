#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
# $Author: frederic $
# $Date: 2016/04/07 21:46:54 $
# $Id: OrthoImageItem.py,v 1.13 2016/04/07 21:46:54 frederic Exp $
#
# -*- coding: utf-8 -*-
import os
from rapidtide.Colortables import *
import rapidtide.stats as tide_stats
import rapidtide.io as tide_io
import rapidtide.filter as tide_filt
import rapidtide.miscmath as tide_math
import numpy as np
import pyqtgraph as pg

atlases = {'ASPECTS': {'atlasname': 'ASPECTS'},
           'ATT': {'atlasname': 'ATTbasedFlowTerritories_split'}}

class timecourse:
    "Store a timecourse and some information about it"

    def __init__(self, name, filename, namebase, samplerate, displaysamplerate, label=None, report=False, verbose=False):
        self.verbose = verbose
        self.name = name
        self.filename = filename
        self.namebase = namebase
        self.samplerate = samplerate
        self.displaysamplerate = displaysamplerate

        if label is None:
            self.label = name
        else:
            self.label = label
        self.report = report
        if self.verbose:
            print('reading timecourse ', self.name, ' from ', self.filename, '...')
        self.readTimeData()

    def readTimeData(self):
        self.timedata = tide_io.readvec(self.filename)
        self.length = len(self.timedata)
        self.timeaxis = np.linspace(0.0, self.length, num=self.length, endpoint=False) / self.samplerate
        self.specaxis, self.specdata = tide_filt.spectrum(tide_math.corrnormalize(self.timedata), self.samplerate)
        self.kurtosis, self.kurtosis_z, self.kurtosis_p = tide_stats.kurtosisstats(self.timedata)

        if self.verbose:
            print('timecourse data range:', np.min(self.timedata), np.max(self.timedata))
            print('sample rate:', self.samplerate)
            print('timecourse length:', self.length)
            print('timeaxis length:', len(self.timeaxis))
            print('kurtosis:', self.kurtosis)
            print('kurtosis_z:', self.kurtosis_z)
            print('kurtosis_p:', self.kurtosis_p)

            print()

    def summarize(self):
        print()
        print('timecourse name:      ', self.name)
        print('    label:            ', self.label)
        print('    filename:         ', self.filename)
        print('    namebase:         ', self.namebase)
        print('    samplerate:       ', self.samplerate)
        print('    length:           ', self.length)
        print('    kurtosis:         ', self.kurtosis)
        print('    kurtosis_z:       ', self.kurtosis_z)
        print('    kurtosis_p:       ', self.kurtosis_p)


class overlay:
    "Store a data overlay and some information about it"

    def __init__(self, name, filename, namebase, funcmask=None, geommask=None, label=None, report=False,
                 lut_state=gray_state, isaMask=False, verbose=False):
        self.verbose = verbose
        self.name = name
        if label is None:
            self.label = name
        else:
            self.label = label
        self.report = report
        self.filename = filename
        self.namebase = namebase
        if self.verbose:
            print('reading map ', self.name, ' from ', self.filename, '...')
        self.readImageData(isaMask=isaMask)
        self.mask = None
        self.maskeddata = None
        self.setFuncMask(funcmask, maskdata=False)
        self.setGeomMask(geommask, maskdata=False)
        self.maskData()
        self.updateStats()
        self.gradient = pg.GradientWidget(orientation='right', allowAdd=True)
        self.lut_state = lut_state
        self.theLUT = None
        self.setLUT(self.lut_state)
        self.space = 'unspecified'
        if (self.header['sform_code'] == 4) or (self.header['qform_code'] == 4):
            if ((self.xdim == 61) and (self.ydim == 73) and (self.zdim == 61)) or \
                ((self.xdim == 91) and (self.ydim == 109) and (self.zdim == 91)):
                self.space = "MNI152"
            else:
                self.space = "MNI152NLin2009cAsym"
        if self.header['sform_code'] != 0:
            self.affine = self.header.get_sform()
        elif self.header['qform_code'] != 0:
            self.affine = self.header.get_qform()
        else:
            self.affine = self.header.get_base_affine()
        self.invaffine = np.linalg.inv(self.affine)
        self.xpos = 0
        self.ypos = 0
        self.zpos = 0
        self.tpos = 0
        self.xcoord = 0.0
        self.ycoord = 0.0
        self.zcoord = 0.0
        self.tcoord = 0.0

        if self.verbose:
            print('overlay initialized:', self.name, self.filename, self.minval, self.dispmin, self.dispmax,
                  self.maxval)
        self.summarize()

    def duplicate(self, newname, newlabel):
        return overlay(newname, self.filename, self.namebase, funcmask=self.funcmask, geommask=self.geommask,
                       label=newlabel, report=self.report,
                       lut_state=self.lut_state)

    def updateStats(self):
        calcmaskeddata = self.data[np.where(self.mask != 0)]
        self.minval = calcmaskeddata.min()
        self.maxval = calcmaskeddata.max()
        self.robustmin, self.pct25, self.pct50, self.pct75, self.robustmax = tide_stats.getfracvals(calcmaskeddata, [0.02, 0.25, 0.5, 0.75, 0.98], nozero=False)
        self.dispmin = self.robustmin
        self.dispmax = self.robustmax
        self.histy, self.histx = np.histogram(calcmaskeddata,
                                              bins=np.linspace(self.minval, self.maxval, 200))
        self.quartiles = [self.pct25, self.pct50, self.pct75]
        print(self.name,':',self.minval, self.maxval, self.robustmin, self.robustmax, self.quartiles)

    def setData(self, data, isaMask=False):
        self.data = data.copy()
        if isaMask:
            self.data[np.where(self.data < 0.5)] = 0.0
            self.data[np.where(self.data > 0.5)] = 1.0
        self.updateStats()

    def readImageData(self, isaMask=False):
        self.nim, self.data, self.header, self.dims, self.sizes = tide_io.readfromnifti(self.filename)
        if isaMask:
            self.data[np.where(self.data < 0.5)] = 0.0
            self.data[np.where(self.data > 0.5)] = 1.0
        if self.verbose:
            print('overlay data range:', np.min(self.data), np.max(self.data))
            print('header', self.header)
        self.xdim, self.ydim, self.zdim, self.tdim = \
            tide_io.parseniftidims(self.dims)
        self.xsize, self.ysize, self.zsize, self.tr = \
            tide_io.parseniftisizes(self.sizes)
        self.toffset = self.header['toffset']
        if self.verbose:
            print('overlay dims:', self.xdim, self.ydim, self.zdim, self.tdim)
            print('overlay sizes:', self.xsize, self.ysize, self.zsize, self.tr)
            print('overlay toffset:', self.toffset)

    def setLabel(self, label):
        self.label = label

    def real2tr(self, time):
        return np.round((time - self.toffset) / self.tr, 0)

    def tr2real(self, tpos):
        return self.toffset + self.tr * tpos

    def real2vox(self, xcoord, ycoord, zcoord, time):
        x, y, z = apply_affine(self.invaffine, [xcoord, ycoord, zcoord])
        t = self.real2tr(time)
        return int(np.round(x, 0)), int(np.round(y, 0)), int(np.round(z, 0)), int(np.round(t, 0))

    def vox2real(self, xpos, ypos, zpos, tpos):
        return np.concatenate((apply_affine(self.affine, [xpos, ypos, zpos]), [self.tr2real(tpos)]), axis=0)

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
        self.maskeddata = self.data.copy()
        self.maskeddata[np.where(self.mask < 0.5)] = 0.0
        self.updateStats()

    def setReport(self, report):
        self.report = report

    def setTR(self, trval):
        self.tr = trval

    def settoffset(self, toffset):
        self.toffset = toffset

    def setLUT(self, lut_state):
        self.lut_state = lut_state
        self.gradient.restoreState(lut_state)
        self.theLUT = self.gradient.getLookupTable(512, alpha=True)

    def summarize(self):
        print()
        print('overlay name:         ', self.name)
        print('    label:            ', self.label)
        print('    filename:         ', self.filename)
        print('    namebase:         ', self.namebase)
        print('    xdim:             ', self.xdim)
        print('    ydim:             ', self.ydim)
        print('    zdim:             ', self.zdim)
        print('    tdim:             ', self.tdim)
        print('    space:            ', self.space)
        print('    toffset:          ', self.toffset)
        print('    tr:               ', self.tr)
        print('    min:              ', self.minval)
        print('    max:              ', self.maxval)
        print('    robustmin:        ', self.robustmin)
        print('    robustmax:        ', self.robustmax)
        print('    dispmin:          ', self.dispmin)
        print('    dispmax:          ', self.dispmax)
        print('    data shape:       ', np.shape(self.data))
        print('    masked data shape:', np.shape(self.maskeddata))
        if self.geommask is None:
            print('    geometric mask not set')
        else:
            print('    geometric mask is set')
        if self.funcmask is None:
            print('    functional mask not set')
        else:
            print('    functional mask is set')




class RapidtideDataset:
    "Store all the data associated with a rapidtide dataset"
    focusregressor = None
    regressorfilterlimits = None
    focusmap = None
    dispmaps = None
    allloadedmaps = None
    loadedfuncmasks = None
    loadedfuncmaps = None
    atlaslabels = None
    atlasname = None
    xdim = 0
    ydim = 0
    zdim = 0
    tdim = 0
    xsize = 0.0
    ysize = 0.0
    zsize = 0.0
    tr = 0.0

    def __init__(self,
                 name,
                 fileroot,
                 anatname=None,
                 geommaskname=None,
                 userise=False,
                 usecorrout=False,
                 forcetr=False,
                 forceoffset=False,
                 coordinatespace='unspecified',
                 offsettime=0.0,
                 verbose=False):
        self.verbose = verbose
        self.name = name
        self.fileroot = fileroot
        self.anatname = anatname
        self.geommaskname = geommaskname
        self.userise = userise
        self.usecorrout = usecorrout
        self.forcetr = forcetr
        self.forceoffset = forceoffset
        self.coordinatespace = coordinatespace
        self.offsettime = offsettime
        self.referencedir = os.path.join(os.path.split(os.path.split(__file__)[0])[0], 'rapidtide', 'data',
                                    'reference')
        self.setupregressors()
        self.setupoverlays()

    def _loadregressors(self):
        self.focusregressor = None
        for thisregressor in self.regressorspecs:
            if os.path.isfile(self.fileroot + thisregressor[1]):
                print('file: ', self.fileroot + thisregressor[1], ' exists - reading...')
                thepath, thebase = os.path.split(self.fileroot + thisregressor[1])
                self.regressors[thisregressor[0]] = timecourse(thisregressor[0],
                                                             self.fileroot + thisregressor[1],
                                                             thebase,
                                                             thisregressor[2],
                                                             thisregressor[3],
                                                             verbose=self.verbose)
                if self.focusregressor is None:
                    self.focusregressor = thisregressor[0]
            else:
                print('file: ', self.fileroot + thisregressor[1], ' does not exist - skipping...')

    def _loadfuncmaps(self):
        self.loadedfuncmaps = []
        xdim = 0
        ydim = 0
        zdim = 0
        for themap in self.funcmaps:
            if os.path.isfile(self.fileroot + themap + '.nii.gz'):
                print('file: ', self.fileroot + themap + '.nii.gz', ' exists - reading...')
                thepath, thebase = os.path.split(self.fileroot)
                self.overlays[themap] = overlay(themap, self.fileroot + themap + '.nii.gz', thebase, report=True)
                if xdim == 0:
                    xdim = self.overlays[themap].xdim
                    ydim = self.overlays[themap].ydim
                    zdim = self.overlays[themap].zdim
                    tdim = self.overlays[themap].tdim
                    xsize = self.overlays[themap].xsize
                    ysize = self.overlays[themap].ysize
                    zsize = self.overlays[themap].zsize
                    tr = self.overlays[themap].tr
                else:
                    if xdim != self.overlays[themap].xdim or \
                            ydim != self.overlays[themap].ydim or \
                            zdim != self.overlays[themap].zdim:
                        print('overlay dimensions do not match!')
                        sys.exit()
                    if xsize != self.overlays[themap].xsize or \
                            ysize != self.overlays[themap].ysize \
                            or zsize != self.overlays[themap].zsize:
                        print('overlay voxel sizes do not match!')
                        sys.exit()
                self.loadedfuncmaps.append(themap)
            else:
                print('map: ', self.fileroot + themap + '.nii.gz', ' does not exist!')
        print('functional maps loaded:', self.loadedfuncmaps)

    def _loadfuncmasks(self):
        self.loadedfuncmasks = []
        for themap in self.funcmasks:
            if os.path.isfile(self.fileroot + themap + '.nii.gz'):
                thepath, thebase = os.path.split(self.fileroot)
                self.overlays[themap] = overlay(themap, self.fileroot + themap + '.nii.gz', thebase, isaMask=True)
                self.loadedfuncmasks.append(themap)
            else:
                print('mask: ', self.fileroot + themap + '.nii.gz', ' does not exist!')
        print(self.loadedfuncmasks)


    def _loadgeommask(self):
        if self.geommaskname != '':
            if os.path.isfile(self.geommaskname):
                thepath, thebase = os.path.split(self.geommaskname)
                self.theoverlays['geommask'] = overlay('geommask', self.geommaskname, thebase, isaMask=True)
                print('using ', self.geommaskname, ' as geometric mask')
                # allloadedmaps.append('geommask')
                return True
        elif self.coordinatespace == 'MNI152':
            try:
                fsldir = os.environ['FSLDIR']
            except KeyError:
                fsldir = None
            print('fsldir set to ', fsldir)
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                if fsldir is not None:
                    self.geommaskname = os.path.join(fsldir, 'data', 'standard', 'MNI152_T1_2mm_brain_mask.nii.gz')
            elif self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                self.geommaskname = os.path.join(self.referencedir, 'MNI152_T1_3mm_brain_mask_bin.nii.gz')
            if os.path.isfile(self.geommaskname):
                thepath, thebase = os.path.split(self.geommaskname)
                self.overlays['geommask'] = overlay('geommask', self.geommaskname, thebase, isaMask=True)
                print('using ', self.geommaskname, ' as background')
                # allloadedmaps.append('geommask')
                return True
            else:
                print('no geometric mask loaded')
                return False
        else:
            print('no geometric mask loaded')
            return False

    def _loadanatomics(self):
        try:
            fsldir = os.environ['FSLDIR']
        except KeyError:
            fsldir = None

        if self.anatname != '':
            print('using user input anatomic name')
            if os.path.isfile(self.anatname):
                thepath, thebase = os.path.split(self.anatname)
                self.overlays['anatomic'] = overlay('anatomic', self.anatname, thebase)
                print('using ', self.anatname, ' as background')
                # allloadedmaps.append('anatomic')
                return True
            else:
                print('specified file does not exist!')
                return False
        elif os.path.isfile(self.fileroot + 'highres_head.nii.gz'):
            print('using hires_head anatomic name')
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays['anatomic'] = overlay('anatomic', self.fileroot + 'highres_head.nii.gz', thebase)
            print('using ', self.fileroot + 'highres_head.nii.gz', ' as background')
            # allloadedmaps.append('anatomic')
            return True
        elif os.path.isfile(self.fileroot + 'highres.nii.gz'):
            print('using hires anatomic name')
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays['anatomic'] = overlay('anatomic', self.fileroot + 'highres.nii.gz', thebase)
            print('using ', self.fileroot + 'highres.nii.gz', ' as background')
            # allloadedmaps.append('anatomic')
            return True
        elif self.coordinatespace == 'MNI152':
            mniname = ''
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                print('using 2mm MNI anatomic name')
                if fsldir is not None:
                    mniname = os.path.join(fsldir, 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
            elif self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                print('using 3mm MNI anatomic name')
                mniname = os.path.join(self.referencedir, 'MNI152_T1_3mm.nii.gz')
            if os.path.isfile(mniname):
                self.overlays['anatomic'] = overlay('anatomic', mniname, 'MNI152')
                print('using ', mniname, ' as background')
                # allloadedmaps.append('anatomic')
                return True
            else:
                print('xsize, ysize, zsize=', self.xsize, self.ysize, self.zsize)
                print('MNI template brain ', mniname, ' not loaded')
        elif self.coordinatespace == 'MNI152NLin2009cAsym':
            mniname = ''
            if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                print('using 2mm MNI anatomic name')
                if fsldir is not None:
                    mniname = os.path.join(self.referencedir, 'mni_icbm152_nlin_asym_09c_2mm.nii.gz')
            elif self.xsize == 1.0 and self.ysize == 1.0 and self.zsize == 1.0:
                print('using 1mm MNI anatomic name')
                mniname = os.path.join(self.referencedir, 'mni_icbm152_nlin_asym_09c_1mm.nii.gz')
            if os.path.isfile(mniname):
                self.overlays['anatomic'] = overlay('anatomic', mniname, 'MNI152NLin2009cAsym')
                print('using ', mniname, ' as background')
                # allloadedmaps.append('anatomic')
                return True
            else:
                print('xsize, ysize, zsize=', self.xsize, self.ysize, self.zsize)
                print('MNI template brain ', mniname, ' not loaded')
        elif os.path.isfile(self.fileroot + 'mean.nii.gz'):
            thepath, thebase = os.path.split(self.fileroot)
            self.overlays['anatomic'] = overlay('anatomic', self.fileroot + 'mean.nii.gz', thebase)
            print('using ', self.fileroot + 'mean.nii.gz', ' as background')
            # allloadedmaps.append('anatomic')
            return True
        else:
            print('no anatomic image loaded')
            return False

    def setupregressors(self):
        # load the regressors
        self.regressors = {}
        self.therunoptions = tide_io.readoptionsfile(self.fileroot + 'options')
        try:
            self.regressorfilterlimits = (float(self.therunoptions['lowerpass']), float(self.therunoptions['upperpass']))
        except KeyError:
            self.regressorfilterlimits = (0.0, 100.0)
        print('regressor filter limits:', self.regressorfilterlimits)
        try:
            self.fmrifreq = float(self.therunoptions['fmrifreq'])
        except KeyError:
            self.fmrifreq = 1.0
        try:
            self.inputfreq = float(self.therunoptions['inputfreq'])
        except KeyError:
            self.inputfreq = 1.0
        try:
            self.oversampfactor = int(self.therunoptions['oversampfactor'])
        except KeyError:
            self.oversampfactor = 1
        self.regressorspecs = [['prefilt', 'reference_origres_prefilt.txt', self.inputfreq, self.inputfreq],
                          ['postfilt', 'reference_origres.txt', self.inputfreq, self.inputfreq],
                          ['pass1', 'reference_resampres_pass1.txt', self.fmrifreq * self.oversampfactor, self.fmrifreq],
                          ['pass2', 'reference_resampres_pass2.txt', self.fmrifreq * self.oversampfactor, self.fmrifreq],
                          ['pass3', 'reference_resampres_pass3.txt', self.fmrifreq * self.oversampfactor, self.fmrifreq],
                          ['pass4', 'reference_resampres_pass4.txt', self.fmrifreq * self.oversampfactor, self.fmrifreq]]
        self._loadregressors()

    def getregressors(self):
        return self.regressors

    def setfocusregressor(self, whichregressor):
        self.focusregressor = whichregressor

    def setupoverlays(self):
        # load the overlays
        self.overlays = {}

        # first the functional maps
        self.funcmaps = ['lagtimes', 'lagstrengths', 'lagsigma', 'MTT', 'R2', 'fitNorm', 'fitcoff']
        if self.userise:
            self.funcmaps = ['lagtimes', 'lagstrengths', 'lagsigma', 'MTT', 'R2', 'risetime_epoch_0', 'starttime_epoch_0',
                        'maxamp_epoch_0']
        if self.usecorrout:
            self.funcmaps += ['corrout']
            self.funcmaps += ['gaussout']
            self.funcmaps += ['failreason']
        self._loadfuncmaps()
        for themap in self.loadedfuncmaps:
            if self.forcetr:
                self.overlays[themap].setTR(self.trval)
            if self.forceoffset:
                self.overlays[themap].settoffset(self.offsettime)
            if self.overlays[themap].space == "MNI152":
                self.coordinatespace = 'MNI152'
            elif self.overlays[themap].space == "MNI152NLin2009cAsym":
                self.coordinatespace = 'MNI152NLin2009cAsym'

        # report results of load
        print('loaded functional maps: ', self.loadedfuncmaps)

        self.allloadedmaps = list(self.loadedfuncmaps)
        self.dispmaps = list(self.loadedfuncmaps)

        # extract some useful information about this dataset from the focusmap
        self.focusmap = 'lagtimes'

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
            self.allloadedmaps.append('anatomic')

        # then the functional masks
        self.funcmasks = ['lagmask', 'refinemask', 'meanmask', 'p_lt_0p050_mask', 'p_lt_0p010_mask', 'p_lt_0p005_mask',
                     'p_lt_0p001_mask']
        self._loadfuncmasks()

        # then the geometric masks
        if self._loadgeommask():
            self.allloadedmaps.append('geommask')

        if (self.coordinatespace == 'MNI152') or (self.coordinatespace == 'MNI152NLin6')or (self.coordinatespace == 'MNI152NLin2009cAsym'):
            # atlasname = 'ASPECTS'
            self.atlasshortname = 'ATT'
            self.atlasname = atlases[self.atlasshortname]['atlasname']
            self.atlaslabels = tide_io.readlabels(os.path.join(self.referencedir, self.atlasname + '_regions.txt'))
            print(self.atlaslabels)
            self.atlasniftiname = None
            if self.coordinatespace == 'MNI152':
                if self.xsize == 2.0 and self.ysize == 2.0 and self.zsize == 2.0:
                    self.atlasniftiname = os.path.join(self.referencedir, self.atlasname + '_2mm.nii.gz')
                    self.atlasmaskniftiname = os.path.join(self.referencedir, self.atlasname + '_2mm_mask.nii.gz')
                if self.xsize == 3.0 and self.ysize == 3.0 and self.zsize == 3.0:
                    self.atlasniftiname = os.path.join(self.referencedir, self.atlasname + '_3mm.nii.gz')
                    self.atlasmaskniftiname = os.path.join(referencedir, self.atlasname + '_3mm_mask.nii.gz')
            else:
                pass
                '''if xsize == 2.0 and ysize == 2.0 and zsize == 2.0:
                    atlasniftiname = os.path.join(referencedir, atlasname + '_nlin_asym_09c_2mm.nii.gz')
                    atlasmaskniftiname = os.path.join(referencedir, atlasname + '_nlin_asym_09c_2mm_mask.nii.gz')'''
            if self.atlasniftiname is not None:
                if os.path.isfile(self.atlasniftiname):
                    self.overlays['atlas'] = overlay('atlas', self.atlasniftiname, self.atlasname, report=True)
                    self.overlays['atlasmask'] = overlay('atlasmask', self.atlasmaskniftiname, self.atlasname, report=True)
                    self.allloadedmaps.append('atlas')
                    self.dispmaps.append('atlas')
                else:
                    print(self.atlasname + ' template: ', self.atlasniftiname, ' does not exist!')

        try:
            test = self.overlays['atlas']
            print('there is an atlas')
            # ui.report_pushButton.show()
            # ui.report_pushButton.setDisabled(False)
        except KeyError:
            print('there is not an atlas')
            # ui.report_pushButton.hide()
            # ui.report_pushButton.setDisabled(True)
        print('done')

    def getoverlays(self):
        return self.overlays

    def setfocusmap(self, whichmap):
        self.focusmap = whichmap


