#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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

"""
A widget for orthographically displaying 3 and 4 dimensional data
"""

import copy
import os

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

try:
    from PIL import Image

    PILexists = True
except ImportError:
    PILexists = False


def newColorbar(left, top, impixpervoxx, impixpervoxy, imgsize):
    cb_xdim = imgsize // 10
    cb_ydim = imgsize
    theviewbox = pg.ViewBox(enableMouse=False)
    theviewbox.setRange(
        QtCore.QRectF(0, 0, cb_xdim, cb_ydim),
        xRange=(0, cb_xdim - 1),
        yRange=(0, cb_ydim - 1),
        padding=0.0,
        disableAutoRange=True,
    )
    theviewbox.setBackgroundColor([50, 50, 50])
    theviewbox.setAspectLocked()

    thecolorbarfgwin = pg.ImageItem()
    theviewbox.addItem(thecolorbarfgwin)
    thecolorbarfgwin.setZValue(10)
    tr = QtGui.QTransform()  # prepare ImageItem transformation:
    tr.translate(left, top)  # move
    tr.scale(impixpervoxx, impixpervoxy)  # scale horizontal and vertical axes
    thecolorbarfgwin.setTransform(tr)

    thecolorbarbgwin = pg.ImageItem()
    theviewbox.addItem(thecolorbarbgwin)
    thecolorbarbgwin.setZValue(0)
    thecolorbarbgwin.setTransform(tr)

    colorbarvals = np.zeros((cb_xdim, cb_ydim), dtype=np.float64)
    for i in range(0, cb_ydim):
        colorbarvals[:, i] = i * (1.0 / (cb_ydim - 1.0))
    return thecolorbarfgwin, thecolorbarbgwin, theviewbox, colorbarvals


def newViewWindow(view, left, top, impixpervoxx, impixpervoxy, imgsize, enableMouse=False):
    theviewbox = view.addViewBox(enableMouse=enableMouse, enableMenu=False, lockAspect=1.0)
    theviewbox.setAspectLocked()
    theviewbox.setRange(QtCore.QRectF(0, 0, imgsize, imgsize), padding=0.0, disableAutoRange=True)
    theviewbox.setBackgroundColor([50, 50, 50])

    theviewbgwin = pg.ImageItem()
    tr = QtGui.QTransform()  # prepare ImageItem transformation:
    tr.translate(left, top)  # move 3x3 image to locate center at axis origin
    tr.scale(impixpervoxx, impixpervoxy)  # scale horizontal and vertical axes
    theviewbgwin.setTransform(tr)

    theviewbgwin.setZValue(0)
    theviewbox.addItem(theviewbgwin)

    theviewfgwin = pg.ImageItem()
    theviewfgwin.setTransform(tr)

    theviewfgwin.setZValue(10)
    theviewbox.addItem(theviewfgwin)

    theviewvLine = pg.InfiniteLine(angle=90, movable=False, pen="g")
    theviewvLine.setZValue(20)
    theviewbox.addItem(theviewvLine)
    theviewhLine = pg.InfiniteLine(angle=0, movable=False, pen="g")
    theviewhLine.setZValue(20)
    theviewbox.addItem(theviewhLine)

    return theviewfgwin, theviewbgwin, theviewvLine, theviewhLine, theviewbox


class OrthoImageItem(QtWidgets.QWidget):
    updated = QtCore.pyqtSignal()

    def __init__(
        self,
        map,
        axview,
        corview,
        sagview,
        enableMouse=False,
        button=None,
        imgsize=64,
        arrangement=0,
        bgmap=None,
        verbose=0,
    ):
        QtWidgets.QWidget.__init__(self)
        self.map = map
        self.bgmap = bgmap
        self.axview = axview
        self.corview = corview
        self.sagview = sagview
        self.button = button
        self.verbose = verbose
        self.enableMouse = enableMouse
        self.xdim = self.map.xdim  # this is the number of voxels along this axis
        self.ydim = self.map.ydim  # this is the number of voxels along this axis
        self.zdim = self.map.zdim  # this is the number of voxels along this axis
        self.tdim = self.map.tdim  # this is the number of voxels along this axis
        self.xsize = self.map.xsize  # this is the mapping between voxel and physical space
        self.ysize = self.map.ysize  # this is the mapping between voxel and physical space
        self.zsize = self.map.zsize  # this is the mapping between voxel and physical space
        self.imgsize = imgsize
        self.xfov = self.xdim * self.xsize
        self.yfov = self.ydim * self.ysize
        self.zfov = self.zdim * self.zsize
        self.xpos = int(self.xdim // 2)
        self.ypos = int(self.ydim // 2)
        self.zpos = int(self.zdim // 2)
        self.tpos = int(0)
        self.maxfov = np.max([self.xfov, self.yfov, self.zfov])
        self.impixpervoxx = self.imgsize * (self.xfov / self.maxfov) / self.xdim
        self.impixpervoxy = self.imgsize * (self.yfov / self.maxfov) / self.ydim
        self.impixpervoxz = self.imgsize * (self.zfov / self.maxfov) / self.zdim
        self.offsetx = self.imgsize * (0.5 - self.xfov / (2.0 * self.maxfov))
        self.offsety = self.imgsize * (0.5 - self.yfov / (2.0 * self.maxfov))
        self.offsetz = self.imgsize * (0.5 - self.zfov / (2.0 * self.maxfov))

        if self.verbose > 1:
            print("OrthoImageItem intialization:")
            print("    Dimensions:", self.xdim, self.ydim, self.zdim)
            print("    Voxel sizes:", self.xsize, self.ysize, self.zsize)
            print("    FOVs:", self.xfov, self.yfov, self.zfov)
            print("    Maxfov, imgsize:", self.maxfov, self.imgsize)
            print(
                "    Scale factors:",
                self.impixpervoxx,
                self.impixpervoxy,
                self.impixpervoxz,
            )
            print("    Offsets:", self.offsetx, self.offsety, self.offsetz)
        self.buttonisdown = False

        self.arrangement = arrangement
        self.axview.setBackground(None)
        self.axview.setRange(padding=0.0)
        self.axview.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.axview.ci.layout.setSpacing(5)
        self.corview.setBackground(None)
        self.corview.setRange(padding=0.0)
        self.corview.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.corview.ci.layout.setSpacing(5)
        self.sagview.setBackground(None)
        self.sagview.setRange(padding=0.0)
        self.sagview.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.sagview.ci.layout.setSpacing(5)

        (
            self.axviewwin,
            self.axviewbgwin,
            self.axviewvLine,
            self.axviewhLine,
            self.axviewbox,
        ) = newViewWindow(
            self.axview,
            self.offsetx,
            self.offsety,
            self.impixpervoxx,
            self.impixpervoxy,
            self.imgsize,
            enableMouse=self.enableMouse,
        )
        (
            self.corviewwin,
            self.corviewbgwin,
            self.corviewvLine,
            self.corviewhLine,
            self.corviewbox,
        ) = newViewWindow(
            self.corview,
            self.offsetx,
            self.offsetz,
            self.impixpervoxx,
            self.impixpervoxz,
            self.imgsize,
            enableMouse=self.enableMouse,
        )
        (
            self.sagviewwin,
            self.sagviewbgwin,
            self.sagviewvLine,
            self.sagviewhLine,
            self.sagviewbox,
        ) = newViewWindow(
            self.sagview,
            self.offsety,
            self.offsetz,
            self.impixpervoxy,
            self.impixpervoxz,
            self.imgsize,
            enableMouse=self.enableMouse,
        )
        if self.enableMouse:
            self.axviewbox.keyPressEvent = self.handleaxkey
            self.axviewbox.mousePressEvent = self.handleaxclick
            self.axviewbox.mouseMoveEvent = self.handleaxmousemove
            self.axviewbox.mouseReleaseEvent = self.handlemouseup
            self.corviewbox.mousePressEvent = self.handlecorclick
            self.corviewbox.mouseMoveEvent = self.handlecormousemove
            self.corviewbox.mouseReleaseEvent = self.handlemouseup
            self.sagviewbox.mousePressEvent = self.handlesagclick
            self.sagviewbox.mouseMoveEvent = self.handlesagmousemove
            self.sagviewbox.mouseReleaseEvent = self.handlemouseup

        self.enableView()
        self.updateAllViews()

    def xvox2pix(self, xpos):
        return int(np.round(self.offsetx + self.impixpervoxx * xpos))

    def yvox2pix(self, ypos):
        return int(np.round(self.offsety + self.impixpervoxy * ypos))

    def zvox2pix(self, zpos):
        return int(np.round(self.offsetz + self.impixpervoxz * zpos))

    def xpix2vox(self, xpix):
        thepos = (xpix - self.offsetx) / self.impixpervoxx
        if thepos > self.xdim - 1:
            thepos = self.xdim - 1
        if thepos < 0:
            thepos = 0
        return int(np.round(thepos))

    def ypix2vox(self, ypix):
        thepos = (ypix - self.offsety) / self.impixpervoxy
        if thepos > self.ydim - 1:
            thepos = self.ydim - 1
        if thepos < 0:
            thepos = 0
        return int(np.round(thepos))

    def zpix2vox(self, zpix):
        thepos = (zpix - self.offsetz) / self.impixpervoxz
        if thepos > self.zdim - 1:
            thepos = self.zdim - 1
        if thepos < 0:
            thepos = 0
        return int(np.round(thepos))

    def updateAllViews(self):
        if self.tdim == 1:
            axdata = self.map.maskeddata[:, :, self.zpos]
        else:
            axdata = self.map.maskeddata[:, :, self.zpos, self.tpos]
        if not (self.map.mask is None):
            axmask = self.map.mask[:, :, self.zpos]
        else:
            axmask = 0.0 * self.map.maskeddata[:, :, self.zpos] + 1.0
        if self.bgmap is None:
            axbg = None
        else:
            axbg = self.bgmap.data[:, :, self.zpos]
        self.updateOneView(axdata, axmask, axbg, self.map.theLUT, self.axviewwin, self.axviewbgwin)
        self.axviewvLine.setValue(self.xvox2pix(self.xpos))
        self.axviewhLine.setValue(self.yvox2pix(self.ypos))

        if self.tdim == 1:
            cordata = self.map.maskeddata[:, self.ypos, :]
        else:
            cordata = self.map.maskeddata[:, self.ypos, :, self.tpos]
        if not (self.map.mask is None):
            cormask = self.map.mask[:, self.ypos, :]
        else:
            cormask = 0.0 * self.map.maskeddata[:, self.ypos, :] + 1.0
        if self.bgmap is None:
            corbg = None
        else:
            corbg = self.bgmap.data[:, self.ypos, :]
        self.updateOneView(
            cordata, cormask, corbg, self.map.theLUT, self.corviewwin, self.corviewbgwin
        )
        self.corviewvLine.setValue(self.xvox2pix(self.xpos))
        self.corviewhLine.setValue(self.zvox2pix(self.zpos))

        if self.tdim == 1:
            sagdata = self.map.maskeddata[self.xpos, :, :]
        else:
            sagdata = self.map.maskeddata[self.xpos, :, :, self.tpos]
        if not (self.map.mask is None):
            sagmask = self.map.mask[self.xpos, :, :]
        else:
            sagmask = 0.0 * self.map.maskeddata[self.xpos, :, :] + 1.0
        if self.bgmap is None:
            sagbg = None
        else:
            sagbg = self.bgmap.data[self.xpos, :, :]
        self.updateOneView(
            sagdata, sagmask, sagbg, self.map.theLUT, self.sagviewwin, self.sagviewbgwin
        )
        self.sagviewvLine.setValue(self.yvox2pix(self.ypos))
        self.sagviewhLine.setValue(self.zvox2pix(self.zpos))

    def updateOneView(self, data, mask, background, theLUT, thefgwin, thebgwin):
        im = self.applyLUT(data, mask, theLUT, self.map.dispmin, self.map.dispmax)
        thefgwin.setImage(im.astype("float"))
        if background is not None:
            thebgwin.setImage(background.astype("float"), autoLevels=True)

    def setMap(self, themap):
        self.map = themap
        self.tdim = self.map.tdim

    def enableView(self):
        if self.button is not None:
            self.button.setText(self.map.label)
            self.button.setDisabled(False)
            self.button.show()
        self.axview.show()
        self.corview.show()
        self.sagview.show()

    def applyLUT(self, theimage, mask, theLUT, dispmin, dispmax):
        offset = dispmin
        if dispmax - dispmin > 0:
            scale = len(theLUT) / (dispmax - dispmin)
        else:
            scale = 0.0
        scaleddata = np.rint((theimage - offset) * scale).astype("int32")
        scaleddata[np.where(scaleddata < 0)] = 0
        scaleddata[np.where(scaleddata > (len(theLUT) - 1))] = len(theLUT) - 1
        mappeddata = theLUT[scaleddata]
        mappeddata[:, :, 3][np.where(mask < 1)] = 0
        return mappeddata

    def updateCursors(self):
        xpix = self.xvox2pix(self.xpos)
        ypix = self.yvox2pix(self.ypos)
        zpix = self.zvox2pix(self.zpos)
        self.axviewvLine.setValue(xpix)
        self.axviewhLine.setValue(ypix)
        self.corviewvLine.setValue(xpix)
        self.corviewhLine.setValue(zpix)
        self.sagviewvLine.setValue(ypix)
        self.sagviewhLine.setValue(zpix)

    def handlemouseup(self, event):
        self.buttonisdown = False
        self.updateCursors()
        self.updateAllViews()

    def handleaxmousemove(self, event):
        if self.buttonisdown:
            self.xpos = self.xpix2vox(event.pos().x() - 1)
            self.ypos = self.ypix2vox(self.imgsize - event.pos().y() + 1)
            self.updateAllViews()
            self.updated.emit()

    def handlecormousemove(self, event):
        if self.buttonisdown:
            self.xpos = self.xpix2vox(event.pos().x() - 1)
            self.zpos = self.zpix2vox(self.imgsize - event.pos().y() + 1)
            self.updateAllViews()
            self.updated.emit()

    def handlesagmousemove(self, event):
        if self.buttonisdown:
            self.ypos = self.ypix2vox(event.pos().x() - 1)
            self.zpos = self.zpix2vox(self.imgsize - event.pos().y() + 1)
            self.updateAllViews()
            self.updated.emit()

    def handleaxkey(self, event):
        if self.verbose > 1:
            print(event)
        self.updateAllViews()
        self.updated.emit()

    def handleaxclick(self, event):
        self.xpos = self.xpix2vox(event.pos().x() - 1)
        self.ypos = self.ypix2vox(self.imgsize - event.pos().y() + 1)
        self.buttonisdown = True
        self.updateAllViews()
        self.updated.emit()

    def handlecorclick(self, event):
        self.xpos = self.xpix2vox(event.pos().x() - 1)
        self.zpos = self.zpix2vox(self.imgsize - event.pos().y() + 1)
        self.buttonisdown = True
        self.updateAllViews()
        self.updated.emit()

    def handlesagclick(self, event):
        self.ypos = self.ypix2vox(event.pos().x() - 1)
        self.zpos = self.zpix2vox(self.imgsize - event.pos().y() + 1)
        self.buttonisdown = True
        self.updateAllViews()
        self.updated.emit()

    def setXYZpos(self, xpos, ypos, zpos, emitsignal=True):
        self.xpos = int(xpos)
        self.ypos = int(ypos)
        self.zpos = int(zpos)
        self.updateAllViews()
        if emitsignal:
            self.updated.emit()

    def setTpos(self, tpos, emitsignal=True):
        if tpos > self.tdim - 1:
            self.tpos = int(self.tdim - 1)
        else:
            self.tpos = int(tpos)

        self.updateAllViews()
        if emitsignal:
            self.updated.emit()

    def getFocusVal(self):
        if self.tdim > 1:
            return self.map.maskeddata[self.xpos, self.ypos, self.zpos, self.tpos]
        else:
            return self.map.maskeddata[self.xpos, self.ypos, self.zpos]

    def saveandcomposite(self, square_img, fg_img, bg_img, name, savedir, scalefach, scalefacv):
        if PILexists:
            if self.verbose > 1:
                print("using PIL to save ", name)
            squarename = os.path.join(savedir, name + "_square.png")
            fgname = os.path.join(savedir, name + "_foreground.png")
            bgname = os.path.join(savedir, name + "_background.png")
            compositename = os.path.join(savedir, name + ".jpg")

            # make the individual layers
            square_img.save(squarename)
            fg_img.save(fgname)
            bg_img.save(bgname)
            square = Image.open(squarename)
            background = Image.open(bgname)
            foreground = Image.open(fgname)
            if self.verbose > 1:
                print(foreground.getbands())

            # now composite
            background.paste(foreground, None, foreground)
            flipped = background.transpose(Image.FLIP_TOP_BOTTOM)

            # scale
            if self.verbose > 1:
                print("scaling")
            mulfac = 8
            hsize = int(mulfac * scalefach)
            vsize = int(mulfac * scalefacv)
            if self.verbose > 1:
                print("scaling to ", hsize, vsize)
            flipped = flipped.resize((hsize, vsize), Image.NEAREST)

            # save and clean up
            if self.verbose > 1:
                print("saving to ", compositename)
            flipped.save(compositename, "jpeg")
            if self.verbose > 1:
                print("cleaning")
            os.remove(fgname)
            os.remove(bgname)
            os.remove(squarename)
        else:
            if self.verbose > 1:
                print("saving ", name)
            square_img.save(os.path.join(savedir, name + "_square.png"))
            fg_img.save(os.path.join(savedir, name + "_fg.png"))
            bg_img.save(os.path.join(savedir, name + "_bg.png"))

    def saveDisp(self):
        if self.verbose > 1:
            print("saving main window")
        mydialog = QtWidgets.QFileDialog()
        options = mydialog.Options()
        thedir = str(
            mydialog.getExistingDirectory(options=options, caption="Image output directory")
        )
        if self.verbose > 1:
            print("thedir=", thedir)
        thename = self.map.namebase + self.map.name

        # make a square background
        thesquarewin = pg.ImageItem()
        maximpervox = np.max([self.impixpervoxx, self.impixpervoxy, self.impixpervoxz])
        maxdim = np.max([self.xdim, self.ydim, self.zdim])
        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        tr.translate(0, 0)  # move 3x3 image to locate center at axis origin
        tr.scale(maximpervox, maximpervox)
        thesquarewin.setTransform(tr)

        thesquarewin.setImage(np.zeros((maxdim, maxdim), dtype=float), autoLevels=True)

        # make a rectangular background
        therectwin = pg.ImageItem()
        therectwin.setTransform(tr)

        therectwin.setImage(np.zeros((maxdim // 10, maxdim), dtype=float), autoLevels=True)

        (
            thecolorbarfgwin,
            thecolorbarbgwin,
            thecolorbarviewbox,
            colorbarvals,
        ) = newColorbar(0, 0, maximpervox, maximpervox, maxdim)
        cbim = self.applyLUT(
            colorbarvals,
            (colorbarvals * 0 + 1).astype("int"),
            self.map.theLUT,
            0.0,
            1.0,
        )
        thecolorbarfgwin.setImage(cbim.astype("float"))
        thecolorbarbgwin.setImage(cbim.astype("float"), autoLevels=True)
        if self.verbose > 1:
            print(thecolorbarfgwin)
            print(thecolorbarbgwin)
            print(thecolorbarviewbox)

        self.saveandcomposite(
            thesquarewin,
            self.axviewwin,
            self.axviewbgwin,
            thename + "_ax",
            thedir,
            self.xdim * self.xsize,
            self.ydim * self.ysize,
        )
        self.saveandcomposite(
            thesquarewin,
            self.corviewwin,
            self.corviewbgwin,
            thename + "_cor",
            thedir,
            self.xdim * self.xsize,
            self.zdim * self.zsize,
        )
        self.saveandcomposite(
            thesquarewin,
            self.sagviewwin,
            self.sagviewbgwin,
            thename + "_sag",
            thedir,
            self.ydim * self.ysize,
            self.zdim * self.zsize,
        )
        """self.saveandcomposite(therectwin,
                              thecolorbarfgwin, thecolorbarbgwin,
                              thename + '_colorbar', thedir,
                              maximpervox * maxdim // 10,
                              maximpervox * maxdim)"""

        with open(os.path.join(thedir, thename + "_lims.txt"), "w") as FILE:
            FILE.writelines(str(self.map.dispmin) + "\t" + str(self.map.dispmax))
            # img_colorbar.save(thedir + self.map.name + '_colorbar.png')

    def summarize(self):
        if self.map is not None:
            # print('OrthoImageItem[', self.map.name, ']: map is set')
            pass
