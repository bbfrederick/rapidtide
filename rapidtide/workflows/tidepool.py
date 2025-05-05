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
"""
A simple GUI for looking at the results of a rapidtide analysis
"""

import argparse
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import pyqtgraph as pg
from nibabel.affines import apply_affine
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import rapidtide.util as tide_util
from rapidtide.Colortables import *
from rapidtide.helper_classes import SimilarityFunctionFitter
from rapidtide.OrthoImageItem import OrthoImageItem
from rapidtide.RapidtideDataset import RapidtideDataset, check_rt_spatialmatch
from rapidtide.workflows.atlasaverage import summarizevoxels

try:
    from PyQt6.QtCore import QT_VERSION_STR
except ImportError:
    pyqtversion = 5
else:
    pyqtversion = 6
print(f"using {pyqtversion=}")

os.environ["QT_MAC_WANTS_LAYER"] = "1"


def _get_parser():
    parser = argparse.ArgumentParser(
        prog="tidepool",
        description="A program to display the results of a time delay analysis",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--offsettime",
        help="Set lag offset",
        dest="offsettime",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--risetime",
        help="enable risetime display",
        dest="userise",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--useatlas",
        help="enable atlas processing",
        dest="useatlas",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--settr",
        help="Set similarity function TR",
        dest="trval",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        help=(
            "Specify one or more dataset root names (skip initial selection step).  The root name is the entire path "
            "to the rapidtide output data (including the underscore) that precedes 'desc-maxtime_map.nii.gz'"
        ),
        dest="datafileroot",
        default=None,
    )
    parser.add_argument(
        "--anatname",
        help="Set anatomic background image",
        dest="anatname",
        default=None,
    )
    parser.add_argument(
        "--maskname", help="Set geometric mask image", dest="geommaskname", default=None
    )
    parser.add_argument(
        "--uistyle",
        action="store",
        type=str,
        choices=["normal", "big"],
        help=(
            "Set the window layout style.  The 'normal' uistyle loads 8 data maps into display panes, "
            "and fits comfortably on the screen of a 14in MacbookPro screen.  The 'big' uistyle has 16 display "
            "panes, and fits on a 16in MBP screen.  Default is 'normal'."
        ),
        default="normal",
    )
    parser.add_argument(
        "--verbosity",
        help="Specify level of truly devastatingly boring console messages.  Default is 0",
        dest="verbose",
        metavar="VERBOSITY",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ignoredimmatch",
        help="Do not check to see if dataset sizes match.  This is almost certainly a terrible idea.",
        dest="ignoredimmatch",
        action="store_true",
        default=False,
    )

    return parser


def addDataset(
    thisdatafileroot,
    anatname=None,
    geommaskname=None,
    userise=False,
    usecorrout=True,
    useatlas=False,
    forcetr=False,
    forceoffset=False,
    offsettime=0.0,
    ignoredimmatch=False,
):
    global currentdataset, thesubjects, whichsubject, datafileroots
    global verbosity

    print("Loading", thisdatafileroot)
    thissubject = RapidtideDataset(
        "main",
        thisdatafileroot,
        anatname=anatname,
        geommaskname=geommaskname,
        userise=userise,
        usecorrout=usecorrout,
        useatlas=useatlas,
        forcetr=forcetr,
        forceoffset=forceoffset,
        offsettime=offsettime,
        verbose=verbosity,
    )
    if len(thesubjects) > 0:
        # check to see that the dimensions match
        dimmatch, sizematch, spacematch, affinematch = check_rt_spatialmatch(
            thissubject, thesubjects[0]
        )
        if dimmatch or ignoredimmatch:
            thesubjects.append(thissubject)
        else:
            print(f"dataset {thisdatafileroot} does not match loaded data - skipping")
    else:
        thesubjects.append(thissubject)

    # list the datasets
    for idx, subject in enumerate(thesubjects):
        print(f"subject {idx}: {subject.fileroot}")


def updateFileMenu():
    global thesubjects, whichsubject
    global fileMenu, sel_open
    global sel_files

    if pyqtversion == 5:
        qactionfunc = QtWidgets.QAction
    else:
        qactionfunc = QtGui.QAction

    # scrub file menu
    if sel_files is not None:
        for sel_file in sel_files:
            fileMenu.removeAction(sel_file)
            del sel_file

    # now build it back
    if len(thesubjects) > 0:
        sel_files = []
        for idx, subject in enumerate(thesubjects):
            if idx == whichsubject:
                indicator = "\u2714 "
            else:
                indicator = "  "
            sel_files.append(qactionfunc(indicator + subject.fileroot, win))
            sel_files[-1].triggered.connect(partial(selectDataset, idx))
            fileMenu.addAction(sel_files[-1])


def datasetPicker():
    global currentdataset, thesubjects, whichsubject, datafileroots
    global ui, win, defaultdict, overlagGraphicsViews
    global verbosity

    mydialog = QtWidgets.QFileDialog()
    if pyqtversion == 5:
        options = mydialog.Options()
    else:
        options = mydialog.options()
    lagfilename = mydialog.getOpenFileName(
        options=options,
        filter="Lag time files (*_lagtimes.nii.gz *_desc-maxtime_map.nii.gz)",
    )[0]
    # check to see which file type we got back
    bidsstartloc = lagfilename.find("desc-maxtime_map.nii.gz")
    if bidsstartloc > 0:
        datafileroot = str(lagfilename[:bidsstartloc])
    else:
        datafileroot = str(lagfilename[: lagfilename.find("lagtimes.nii.gz")])
    datafileroots.append(datafileroot)
    addDataset(datafileroots[-1])
    whichsubject = len(thesubjects) - 1
    selectDataset(whichsubject)

    # update the file menu
    updateFileMenu()


def selectDataset(thesubject):
    global currentdataset, thesubjects, whichsubject, datafileroots
    global ui, win, defaultdict, overlagGraphicsViews
    global verbosity, uiinitialized

    whichsubject = thesubject
    if uiinitialized:
        thesubjects[whichsubject].setfocusregressor(currentdataset.focusregressor)
        thesubjects[whichsubject].setfocusmap(currentdataset.focusmap)
    currentdataset = thesubjects[whichsubject]
    activateDataset(
        currentdataset,
        ui,
        win,
        defaultdict,
        overlayGraphicsViews,
        verbosity=verbosity,
    )

    # update the file menu
    updateFileMenu()


class xyztlocation(QtWidgets.QWidget):
    "Manage a location in time and space"

    updatedXYZ = QtCore.pyqtSignal()
    updatedT = QtCore.pyqtSignal()
    movieTimer = QtCore.QTimer()

    def __init__(
        self,
        xpos,
        ypos,
        zpos,
        tpos,
        xdim,
        ydim,
        zdim,
        tdim,
        toffset,
        tr,
        affine,
        XPosSpinBox=None,
        YPosSpinBox=None,
        ZPosSpinBox=None,
        TPosSpinBox=None,
        XCoordSpinBox=None,
        YCoordSpinBox=None,
        ZCoordSpinBox=None,
        TCoordSpinBox=None,
        TimeSlider=None,
        runMovieButton=None,
    ):
        QtWidgets.QWidget.__init__(self)

        self.XPosSpinBox = XPosSpinBox
        self.YPosSpinBox = YPosSpinBox
        self.ZPosSpinBox = ZPosSpinBox
        self.TPosSpinBox = TPosSpinBox
        self.XCoordSpinBox = XCoordSpinBox
        self.YCoordSpinBox = YCoordSpinBox
        self.ZCoordSpinBox = ZCoordSpinBox
        self.TCoordSpinBox = TCoordSpinBox
        self.TimeSlider = TimeSlider
        self.runMovieButton = runMovieButton

        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.setXYZInfo(xdim, ydim, zdim, affine)

        self.tpos = tpos
        self.setTInfo(tdim, tr, toffset)

        self.frametime = 25
        self.movierunning = False
        self.movieTimer.timeout.connect(self.updateMovie)

    def setXYZInfo(self, xdim, ydim, zdim, affine):
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.affine = affine
        self.invaffine = np.linalg.inv(self.affine)
        self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
        self.setupSpinBox(self.XPosSpinBox, self.getXpos, 0, self.xdim - 1, 1, self.xpos)
        self.setupSpinBox(self.YPosSpinBox, self.getYpos, 0, self.ydim - 1, 1, self.ypos)
        self.setupSpinBox(self.ZPosSpinBox, self.getZpos, 0, self.zdim - 1, 1, self.zpos)
        xllcoord, yllcoord, zllcoord = self.vox2real(0, 0, 0)
        xulcoord, yulcoord, zulcoord = self.vox2real(self.xdim - 1, self.ydim - 1, self.zdim - 1)
        xmin = np.min([xllcoord, xulcoord])
        xmax = np.max([xllcoord, xulcoord])
        ymin = np.min([yllcoord, yulcoord])
        ymax = np.max([yllcoord, yulcoord])
        zmin = np.min([zllcoord, zulcoord])
        zmax = np.max([zllcoord, zulcoord])
        self.setupSpinBox(
            self.XCoordSpinBox,
            self.getXcoord,
            xmin,
            xmax,
            (xmax - xmin) / self.xdim,
            self.xcoord,
        )
        self.setupSpinBox(
            self.YCoordSpinBox,
            self.getYcoord,
            ymin,
            ymax,
            (ymax - ymin) / self.ydim,
            self.ycoord,
        )
        self.setupSpinBox(
            self.ZCoordSpinBox,
            self.getZcoord,
            zmin,
            zmax,
            (zmax - zmin) / self.zdim,
            self.zcoord,
        )

    def setTInfo(self, tdim, tr, toffset):
        self.tdim = tdim
        self.toffset = toffset
        self.tr = tr
        self.tcoord = self.tr2real(self.tpos)
        self.setupSpinBox(self.TPosSpinBox, self.getTpos, 0, self.tdim - 1, 1, self.tpos)
        tllcoord = self.tr2real(0)
        tulcoord = self.tr2real(self.tdim - 1)
        tmin = np.min([tllcoord, tulcoord])
        tmax = np.max([tllcoord, tulcoord])
        self.setupSpinBox(
            self.TCoordSpinBox,
            self.getTcoord,
            tmin,
            tmax,
            (tmax - tmin) / self.tdim,
            self.tcoord,
        )
        self.setupTimeSlider(self.TimeSlider, self.getTimeSlider, 0, self.tdim - 1, self.tpos)
        self.setupRunMovieButton(self.runMovieButton, self.runMovieToggle)

    def setupRunMovieButton(self, thebutton, thehandler):
        global verbosity

        if thebutton is not None:
            if verbosity > 1:
                print("initializing movie button")
            thebutton.setText("Start Movie")
            thebutton.clicked.connect(thehandler)

    def setupTimeSlider(self, theslider, thehandler, minval, maxval, currentval):
        if theslider is not None:
            theslider.setRange(minval, maxval)
            theslider.setSingleStep(1)
            theslider.valueChanged.connect(thehandler)

    def setupSpinBox(self, thespinbox, thehandler, minval, maxval, stepsize, currentval):
        if thespinbox is not None:
            thespinbox.setRange(minval, maxval)
            thespinbox.setSingleStep(stepsize)
            thespinbox.setValue(currentval)
            thespinbox.setWrapping(True)
            thespinbox.setKeyboardTracking(False)
            thespinbox.valueChanged.connect(thehandler)

    def updateXYZValues(self, emitsignal=True):
        # print('resetting XYZ spinbox values')
        if self.XPosSpinBox is not None:
            self.XPosSpinBox.setValue(self.xpos)
        if self.YPosSpinBox is not None:
            self.YPosSpinBox.setValue(self.ypos)
        if self.ZPosSpinBox is not None:
            self.ZPosSpinBox.setValue(self.zpos)
        if self.XCoordSpinBox is not None:
            self.XCoordSpinBox.setValue(self.xcoord)
        if self.YCoordSpinBox is not None:
            self.YCoordSpinBox.setValue(self.ycoord)
        if self.ZCoordSpinBox is not None:
            self.ZCoordSpinBox.setValue(self.zcoord)
        # print('done resetting XYZ spinbox values')
        if emitsignal:
            self.updatedXYZ.emit()

    def updateTValues(self):
        # print('resetting T spinbox values')
        if self.TPosSpinBox is not None:
            self.TPosSpinBox.setValue(self.tpos)
            self.TPosSpinBox.setRange(0, self.tdim - 1)
        if self.TCoordSpinBox is not None:
            self.TCoordSpinBox.setValue(self.tcoord)
            tllcoord = self.tr2real(0)
            tulcoord = self.tr2real(self.tdim - 1)
            tmin = np.min([tllcoord, tulcoord])
            tmax = np.max([tllcoord, tulcoord])
            self.TCoordSpinBox.setRange(tmin, tmax)
        if self.TimeSlider is not None:
            self.TimeSlider.setValue((self.tpos))
            self.TimeSlider.setRange(0, self.tdim - 1)
        # print('done resetting T spinbox values')
        self.updatedT.emit()

    def real2tr(self, time):
        return int(np.round((time - self.toffset) / self.tr, 0))

    def tr2real(self, tpos):
        return self.toffset + self.tr * tpos

    def real2vox(self, xcoord, ycoord, zcoord):
        x, y, z = apply_affine(self.invaffine, [xcoord, ycoord, zcoord])
        return int(np.round(x, 0)), int(np.round(y, 0)), int(np.round(z, 0))

    def vox2real(self, xpos, ypos, zpos):
        return apply_affine(self.affine, [xpos, ypos, zpos])

    def setXYZpos(self, xpos, ypos, zpos, emitsignal=True):
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
        self.updateXYZValues(emitsignal=emitsignal)

    def setXYZcoord(self, xcoord, ycoord, zcoord, emitsignal=True):
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.zcoord = zcoord
        self.xpos, self.ypos, self.zpos = self.real2vox(self.xcoord, self.ycoord, self.zcoord)
        self.updateXYZValues(emitsignal=emitsignal)

    def setTpos(self, tpos):
        if tpos > self.tdim:
            tpos = self.tdim
        if self.tpos != tpos:
            self.tpos = tpos
            self.tcoord = self.tr2real(self.tpos)
            self.updateTValues()

    def setTcoord(self, tcoord):
        if self.tcoord != tcoord:
            self.tcoord = tcoord
            self.tpos = self.real2tr(self.tcoord)
            self.updateTValues()

    def getXpos(self, event):
        # print('entering getXpos')
        newx = int(self.XPosSpinBox.value())
        if self.xpos != newx:
            self.xpos = newx
            self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
            self.updateXYZValues(emitsignal=True)

    def getYpos(self, event):
        # print('entering getYpos')
        newy = int(self.YPosSpinBox.value())
        if self.ypos != newy:
            self.ypos = newy
            self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
            self.updateXYZValues(emitsignal=True)

    def getZpos(self, event):
        # print('entering getZpos')
        newz = int(self.ZPosSpinBox.value())
        if self.zpos != newz:
            self.zpos = newz
            self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
            self.updateXYZValues(emitsignal=True)

    def getTpos(self, event):
        # print('entering getTpos')
        newt = int(self.TPosSpinBox.value())
        if self.tpos != newt:
            self.tpos = newt
            self.tcoord = self.tr2real(self.tpos)
            if self.movierunning:
                self.movieTimer.stop()
                self.movieTimer.start(int(self.tpos))
            self.updateTValues()

    def getXcoord(self, event):
        newxcoord = self.XCoordSpinBox.value()
        if self.xcoord != newxcoord:
            self.xcoord = newxcoord
            self.xpos, self.ypos, self.zpos = self.real2vox(self.xcoord, self.ycoord, self.zcoord)
            self.updateXYZValues(emitsignal=True)

    def getYcoord(self, event):
        newycoord = self.YCoordSpinBox.value()
        if self.ycoord != newycoord:
            self.ycoord = newycoord
            self.xpos, self.ypos, self.zpos = self.real2vox(self.xcoord, self.ycoord, self.zcoord)
            self.updateXYZValues(emitsignal=True)

    def getZcoord(self, event):
        newzcoord = self.ZCoordSpinBox.value()
        if self.zcoord != newzcoord:
            self.zcoord = newzcoord
            self.xpos, self.ypos, self.zpos = self.real2vox(self.xcoord, self.ycoord, self.zcoord)
            self.updateXYZValues(emitsignal=True)

    def getTcoord(self, event):
        newtcoord = self.TCoordSpinBox.value()
        if self.tcoord != newtcoord:
            self.tcoord = newtcoord
            self.tpos = self.real2tr(self.tcoord)
            self.updateTValues()

    def getTimeSlider(self):
        self.tpos = self.TimeSlider.value()
        self.tcoord = self.tr2real(self.tpos)
        self.updateTValues()

    def updateMovie(self):
        # self.tpos = (self.tpos + 1) % self.tdim
        self.setTpos((self.tpos + 1) % self.tdim)
        self.updateTValues()
        if verbosity > 1:
            print(f"Tpos, t: {self.tpos}, {self.TCoordSpinBox.value()}")

    def stopMovie(self):
        self.movierunning = False
        self.movieTimer.stop()
        if self.runMovieButton is not None:
            self.runMovieButton.setText("Start Movie")

    def startMovie(self):
        self.movierunning = True
        if self.runMovieButton is not None:
            self.runMovieButton.setText("Stop Movie")
        self.movieTimer.start(int(self.frametime))

    def runMovieToggle(self, event):
        if verbosity > 1:
            print("entering runMovieToggle")
        if self.movierunning:
            if verbosity > 1:
                print("movie is running - turning off")
            self.stopMovie()
        else:
            if verbosity > 1:
                print("movie is not running - turning on")
            self.startMovie()
        if verbosity > 1:
            print("leaving runMovieToggle")

    def setFrametime(self, frametime):
        if self.movierunning:
            self.movieTimer.stop()
        self.frametime = frametime
        if self.movierunning:
            self.movieTimer.start(int(self.frametime))


def logStatus(thetextbox, thetext):
    if pyqtversion == 5:
        thetextbox.moveCursor(QtGui.QTextCursor.End)
    thetextbox.insertPlainText(thetext + "\n")
    sb = thetextbox.verticalScrollBar()
    sb.setValue(sb.maximum())


def getMinDispLimit():
    global ui, overlays, currentdataset
    overlays[currentdataset.focusmap].dispmin = ui.dispmin_doubleSpinBox.value()
    updateDispLimits()


def getMaxDispLimit():
    global ui, overlays, currentdataset
    overlays[currentdataset.focusmap].dispmax = ui.dispmax_doubleSpinBox.value()
    updateDispLimits()


def updateDispLimits():
    global ui, overlays, currentdataset
    ui.dispmin_doubleSpinBox.setRange(
        overlays[currentdataset.focusmap].minval,
        overlays[currentdataset.focusmap].maxval,
    )
    ui.dispmax_doubleSpinBox.setRange(
        overlays[currentdataset.focusmap].minval,
        overlays[currentdataset.focusmap].maxval,
    )
    ui.dispmin_doubleSpinBox.setSingleStep(
        (overlays[currentdataset.focusmap].maxval - overlays[currentdataset.focusmap].minval)
        / 100.0
    )
    ui.dispmax_doubleSpinBox.setSingleStep(
        (overlays[currentdataset.focusmap].maxval - overlays[currentdataset.focusmap].minval)
        / 100.0
    )
    ui.dispmin_doubleSpinBox.setValue(overlays[currentdataset.focusmap].dispmin)
    ui.dispmax_doubleSpinBox.setValue(overlays[currentdataset.focusmap].dispmax)
    updateUI(callingfunc="updateDispLimits", orthoimages=True)


def resetDispLimits():
    global overlays, currentdataset
    overlays[currentdataset.focusmap].dispmin = overlays[currentdataset.focusmap].minval
    overlays[currentdataset.focusmap].dispmax = overlays[currentdataset.focusmap].maxval
    updateDispLimits()


def resetDispSmart():
    global overlays, currentdataset
    overlays[currentdataset.focusmap].dispmin = overlays[currentdataset.focusmap].robustmin
    overlays[currentdataset.focusmap].dispmax = overlays[currentdataset.focusmap].robustmax
    updateDispLimits()


def updateSimilarityFunc():
    global overlays, timeaxis, currentloc
    global currentdataset
    global simfunc_ax, simfuncCurve, simfuncfitCurve, simfuncTLine, simfuncPeakMarker, simfuncCurvePoint, simfuncCaption, simfuncFitter
    global verbosity

    if simfunc_ax is not None:
        corrvals = overlays["corrout"].data[currentloc.xpos, currentloc.ypos, currentloc.zpos, :]
        # fitvals = overlays['gaussout'].data[currentloc.xpos, currentloc.ypos, currentloc.zpos, :]
        peakmaxtime = overlays["lagtimes"].maskeddata[
            currentloc.xpos, currentloc.ypos, currentloc.zpos
        ]
        peakmaxval = overlays["lagstrengths"].maskeddata[
            currentloc.xpos, currentloc.ypos, currentloc.zpos
        ]
        maxval = np.max(corrvals)
        minval = np.min(corrvals)
        valrange = maxval - minval
        try:
            failreason = overlays["failimage"].data[
                currentloc.xpos, currentloc.ypos, currentloc.zpos
            ]
        except KeyError:
            failreason = 0

        # update the plot
        simfuncCurve.setData(timeaxis, corrvals)
        # simfuncfitCurve.setData(timeaxis, fitvals)
        if overlays["corrout"].mask[currentloc.xpos, currentloc.ypos, currentloc.zpos] > 0:
            if (currentdataset.similaritymetric == "correlation") or (
                currentdataset.similaritymetric == "hybrid"
            ):
                thecaption = "lag={1:.2f}, R={0:.2f}".format(peakmaxval, peakmaxtime)
            else:
                thecaption = "lag={1:.2f}, MI={0:.2f}".format(peakmaxval, peakmaxtime)
        else:
            peakmaxtime = float(
                tide_util.valtoindex(
                    timeaxis,
                    overlays["lagtimes"].data[currentloc.xpos, currentloc.ypos, currentloc.zpos],
                )
            )
            thefailreason = simfuncFitter.diagnosefail(np.uint32(failreason))
            if verbosity > 0:
                print(thefailreason)
            thecaption = "No valid fit"
        simfunc_ax.setYRange(minval - 0.05 * valrange, maxval + 0.3 * valrange, padding=0)
        simfuncCaption.setText(thecaption)
        if verbosity > 1:
            print("current tpos:", currentloc.tpos)
            print("setting line location:", timeaxis[currentloc.tpos])
        simfuncTLine.setPos(timeaxis[currentloc.tpos])
        simfuncCurvePoint.setPos(
            float(tide_util.valtoindex(timeaxis, peakmaxtime)) / (len(timeaxis) - 1)
        )
        if overlays["corrout"].mask[currentloc.xpos, currentloc.ypos, currentloc.zpos] > 0:
            simfuncPeakMarker.setData(x=[peakmaxtime], y=[peakmaxval], symbol="d")
        else:
            simfuncPeakMarker.setData(x=[], y=[], symbol="d")


def updateHistogram():
    global hist_ax, overlays, currentdataset
    hist_ax.plot(
        overlays[currentdataset.focusmap].histx,
        overlays[currentdataset.focusmap].histy,
        title="Histogram",
        stepMode=True,
        fillLevel=0,
        brush=(0, 0, 255, 80),
        clear=True,
    )

    histtop = 1.25 * np.max(overlays[currentdataset.focusmap].histy)
    hist_ax.setYRange(0.0, histtop, padding=0)
    pct02 = overlays[currentdataset.focusmap].robustmin
    pct25 = overlays[currentdataset.focusmap].quartiles[0]
    pct50 = overlays[currentdataset.focusmap].quartiles[1]
    pct75 = overlays[currentdataset.focusmap].quartiles[2]
    pct98 = overlays[currentdataset.focusmap].robustmax
    hist_ax.addLine(x=pct02, pen="#008800")
    hist_ax.addLine(x=pct25, pen="g")
    hist_ax.addLine(x=pct50, pen="g")
    hist_ax.addLine(x=pct75, pen="g")
    hist_ax.addLine(x=pct98, pen="#008800")
    updateDispLimits()


# found this routine at https://github.com/abhilb/pyqtgraphutils/blob/master/pyqtgraphutils.py
class RectangleItem(pg.GraphicsObject):
    def __init__(self, topLeft, size, color=(0, 128, 0, 128)):
        pg.GraphicsObject.__init__(self)
        self.topLeft = topLeft
        self.size = size
        self.color = color
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen(self.color))
        p.setBrush(pg.mkBrush(self.color))
        tl = QtCore.QPointF(self.topLeft[0], self.topLeft[1])
        size = QtCore.QSizeF(self.size[0], self.size[1])
        p.drawRect(QtCore.QRectF(tl, size))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


def updateRegressor():
    global regressor_ax, liveplots, currentdataset
    focusregressor = currentdataset.focusregressor
    regressors = currentdataset.getregressors()
    if currentdataset.focusregressor is not None:
        theplot = regressor_ax.plot(
            regressors[focusregressor].timeaxis,
            regressors[focusregressor].timedata,
            stepMode=False,
            fillLevel=0,
            pen=pg.mkPen("w", width=1),
            clear=True,
        )
        regressor_ax.setXRange(0.0, regressors["pass1"].timeaxis[-1])
        thelabel = "kurtosis: {0:.2f}, (z={0:.2f}, p<{0:.2f})".format(
            regressors[focusregressor].kurtosis,
            regressors[focusregressor].kurtosis_z,
            regressors[focusregressor].kurtosis_p,
        )
        text = pg.TextItem(text=thelabel, anchor=(0.0, 2.0), angle=0, fill=(0, 0, 0, 100))
        # regressor_ax.addItem(text)

        # add in calculation limits
        regressorsimcalclimits = currentdataset.regressorsimcalclimits
        tctop = 1.25 * np.max(regressors[focusregressor].timedata)
        tcbottom = 1.25 * np.min(regressors[focusregressor].timedata)
        lowerlim = regressorsimcalclimits[0]
        if regressorsimcalclimits[1] == -1:
            upperlim = regressors[focusregressor].timeaxis[-1]
        else:
            upperlim = regressorsimcalclimits[1]
        bottomleft = [lowerlim, tcbottom]
        size = [upperlim - lowerlim, tctop - tcbottom]
        therectangle = RectangleItem(bottomleft, size)
        regressor_ax.addItem(therectangle)
    else:
        print("currentdataset.focusregressor is None!")


def updateRegressorSpectrum():
    global regressorspectrum_ax, liveplots, currentdataset
    focusregressor = currentdataset.focusregressor
    regressorfilterlimits = currentdataset.regressorfilterlimits
    regressors = currentdataset.getregressors()
    if focusregressor is not None:
        regressorspectrum_ax.plot(
            regressors[focusregressor].specaxis,
            regressors[focusregressor].specdata,
            stepMode=False,
            fillLevel=0,
            pen=pg.mkPen("w", width=1),
            clear=True,
        )
        regressorspectrum_ax.setXRange(
            0.0, 0.5 * regressors[focusregressor].displaysamplerate, padding=0
        )
        spectop = 1.25 * np.max(regressors[focusregressor].specdata)
        regressorspectrum_ax.setYRange(0.0, spectop, padding=0)
        lowerlim = regressorfilterlimits[0]
        if regressorfilterlimits[1] == -1:
            upperlim = regressors[focusregressor].specaxis[-1]
        else:
            upperlim = regressorfilterlimits[1]
        bottomleft = [lowerlim, 0.0]
        size = [upperlim - lowerlim, spectop]
        therectangle = RectangleItem(bottomleft, size)
        regressorspectrum_ax.addItem(therectangle)


def calcAtlasStats():
    global overlays, atlasstats, averagingmode, currentdataset
    global atlasaveragingdone
    print("in calcAtlasStats")
    methodlist = ["mean", "median", "std", "MAD", "CoV"]
    if "atlas" in overlays:
        atlasstats = {}
        print("performing atlas averaging")
        for idx, themap in enumerate(currentdataset.loadedfuncmaps):
            if overlays[themap].display_state and (overlays[themap].tdim == 1):
                atlasstats[themap] = {}
                for regnum, region in enumerate(currentdataset.atlaslabels):
                    print("calculating stats for region", regnum + 1, "(", region, ")")
                    atlasstats[themap][region] = {}
                    maskedregion = np.where(
                        (overlays["atlas"].data == (regnum + 1)) & (overlays[themap].mask > 0)
                    )
                    for themethod in methodlist:
                        atlasstats[themap][region][themethod] = summarizevoxels(
                            overlays[themap].data[maskedregion],
                            method=themethod,
                        )
                atlasstatmap = overlays[themap].duplicate(
                    themap + "_atlasstat", overlays[themap].label
                )
                atlasstatmap.funcmask = overlays["atlasmask"].data
                atlasstatmap.data *= 0.0
                atlasstatmap.maskData()
                atlasstatmap.updateStats()
                overlays[themap + "_atlasstat"] = atlasstatmap
        print("done performing atlas averaging")
        for thestat in methodlist:
            d = {}
            cols = []
            d["Region"] = np.asarray(currentdataset.atlaslabels)
            cols.append("Region")
            for idx, themap in enumerate(currentdataset.loadedfuncmaps):
                if overlays[themap].display_state and (overlays[themap].tdim == 1):
                    templist = []
                    for regnum, region in enumerate(currentdataset.atlaslabels):
                        templist.append(
                            atlasstats[themap][currentdataset.atlaslabels[regnum]][thestat]
                        )
                    d[themap] = np.asarray(templist)
                    cols.append(themap)
            df = pd.DataFrame(data=d)
            df = df[cols]
            df.to_csv(
                currentdataset.fileroot + currentdataset.atlasname + "_" + thestat + ".txt",
                sep="\t",
                index=False,
            )
        atlasaveragingdone = True
    else:
        print("cannot perform average - no atlas map found")


def updateAtlasStats():
    global overlays, atlasstats, averagingmode, currentdataset
    print("in updateAtlasStats")
    if "atlas" in overlays and (averagingmode is not None):
        for idx, themap in enumerate(currentdataset.loadedfuncmaps):
            print("loading", averagingmode, "into", themap + "_atlasstat")
            for regnum, region in enumerate(currentdataset.atlaslabels):
                if themap != "atlas":
                    overlays[themap + "_atlasstat"].data[
                        np.where(overlays["atlas"].data == (regnum + 1))
                    ] = atlasstats[themap][region][averagingmode]
                overlays[themap + "_atlasstat"].maskData()
                overlays[themap + "_atlasstat"].updateStats()


def doAtlasAveraging(state):
    global atlasaveragingdone
    print("in doAtlasAveraging")
    if state == QtCore.Qt.CheckState.Checked:
        atlasaveragingdone = True
        print("atlas averaging is turned on")
    else:
        atlasaveragingdone = False
        print("atlas averaging is turned off")
    updateOrthoImages()


def updateAveragingMode():
    global averagingmode, focusmap
    global atlasaveragingdone
    global overlays
    global currentdataset
    print("in updateAveragingMode")
    if ("atlas" in overlays) and (not atlasaveragingdone):
        calcAtlasStats()
        setAtlasMask()
    if ("atlas" in overlays) and False:
        updateAtlasStats()
    if averagingmode is not None:
        currentdataset.focusmap = currentdataset.focusmap.replace("_atlasstat", "") + "_atlasstat"
    else:
        currentdataset.focusmap = currentdataset.focusmap.replace("_atlasstat", "")
    resetDispLimits()
    resetDispSmart()
    print("averaging mode set to ", averagingmode)
    updateUI(
        callingfunc="updateAverageingMode()",
        orthoimages=True,
        histogram=True,
        LUT=True,
        focusvals=True,
    )


def raw_radioButton_clicked(enabled):
    global averagingmode
    if enabled:
        print("in raw_radioButton_clicked")
        averagingmode = None
        updateAveragingMode()


def mean_radioButton_clicked(enabled):
    global averagingmode
    if enabled:
        print("in mean_radioButton_clicked")
        averagingmode = "mean"
        updateAveragingMode()


def median_radioButton_clicked(enabled):
    global averagingmode
    if enabled:
        print("in median_radioButton_clicked")
        averagingmode = "median"
        updateAveragingMode()


def CoV_radioButton_clicked(enabled):
    global averagingmode
    if enabled:
        print("in CoV_radioButton_clicked")
        averagingmode = "CoV"
        updateAveragingMode()


def std_radioButton_clicked(enabled):
    global averagingmode
    if enabled:
        print("in std_radioButton_clicked")
        averagingmode = "std"
        updateAveragingMode()


def MAD_radioButton_clicked(enabled):
    global averagingmode
    if enabled:
        print("in MAD_radioButton_clicked")
        averagingmode = "MAD"
        updateAveragingMode()


def transparencyCheckboxClicked():
    global LUT_alpha, LUT_endalpha, ui, overlays, currentdataset
    global verbosity

    if ui.transparency_checkBox.isChecked():
        LUT_endalpha = 0
    else:
        LUT_endalpha = 255
    if verbosity > 1:
        print("LUT_endalpha=", LUT_endalpha)
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setLUT(overlays[themap].lut_state, alpha=LUT_alpha, endalpha=LUT_endalpha)
        overlays[themap].gradient.restoreState(overlays[themap].lut_state)
        imageadj.restoreState(overlays[themap].lut_state)
    updateLUT()


def gray_radioButton_clicked(enabled):
    global imageadj, overlays, currentdataset, LUT_alpha, LUT_endalpha

    if enabled:
        overlays[currentdataset.focusmap].setLUT(
            gen_gray_state(), alpha=LUT_alpha, endalpha=LUT_endalpha
        )
        overlays[currentdataset.focusmap].gradient.restoreState(
            overlays[currentdataset.focusmap].lut_state
        )
        imageadj.restoreState(overlays[currentdataset.focusmap].lut_state)
        updateLUT()


def thermal_radioButton_clicked(enabled):
    global imageadj, overlays, currentdataset, LUT_alpha, LUT_endalpha
    if enabled:
        overlays[currentdataset.focusmap].setLUT(
            gen_thermal_state(), alpha=LUT_alpha, endalpha=LUT_endalpha
        )
        overlays[currentdataset.focusmap].gradient.restoreState(
            overlays[currentdataset.focusmap].lut_state
        )
        imageadj.restoreState(overlays[currentdataset.focusmap].lut_state)
        updateLUT()


def plasma_radioButton_clicked(enabled):
    global imageadj, overlays, currentdataset, LUT_alpha, LUT_endalpha
    if enabled:
        overlays[currentdataset.focusmap].setLUT(
            gen_plasma_state(), alpha=LUT_alpha, endalpha=LUT_endalpha
        )
        overlays[currentdataset.focusmap].gradient.restoreState(
            overlays[currentdataset.focusmap].lut_state
        )
        imageadj.restoreState(overlays[currentdataset.focusmap].lut_state)
        updateLUT()


def viridis_radioButton_clicked(enabled):
    global imageadj, overlays, currentdataset, LUT_alpha, LUT_endalpha
    if enabled:
        overlays[currentdataset.focusmap].setLUT(
            gen_viridis_state(), alpha=LUT_alpha, endalpha=LUT_endalpha
        )
        overlays[currentdataset.focusmap].gradient.restoreState(
            overlays[currentdataset.focusmap].lut_state
        )
        imageadj.restoreState(overlays[currentdataset.focusmap].lut_state)
        updateLUT()


def turbo_radioButton_clicked(enabled):
    global imageadj, overlays, currentdataset, LUT_alpha, LUT_endalpha
    if enabled:
        overlays[currentdataset.focusmap].setLUT(
            gen_turbo_state(), alpha=LUT_alpha, endalpha=LUT_endalpha
        )
        overlays[currentdataset.focusmap].gradient.restoreState(
            overlays[currentdataset.focusmap].lut_state
        )
        imageadj.restoreState(overlays[currentdataset.focusmap].lut_state)
        updateLUT()


def rainbow_radioButton_clicked(enabled):
    global imageadj, overlays, currentdataset, LUT_alpha, LUT_endalpha
    if enabled:
        overlays[currentdataset.focusmap].setLUT(
            gen_spectrum_state(), alpha=LUT_alpha, endalpha=LUT_endalpha
        )
        overlays[currentdataset.focusmap].gradient.restoreState(
            overlays[currentdataset.focusmap].lut_state
        )
        imageadj.restoreState(overlays[currentdataset.focusmap].lut_state)
        updateLUT()


def setMask(maskname):
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    maskinfodicts = {}
    maskinfodicts["nomask"] = {
        "msg": "Disabling functional mask",
        "label": "No mask",
    }
    maskinfodicts["meanmask"] = {
        "msg": "Mean regressor seed mask",
        "label": "Mean mask",
    }
    maskinfodicts["lagmask"] = {
        "msg": "Using valid fit points as functional mask",
        "label": "Valid mask",
    }
    maskinfodicts["brainmask"] = {
        "msg": "Externally provided brain mask",
        "label": "Brain mask",
    }
    maskinfodicts["refinemask"] = {
        "msg": "Voxel refinement mask",
        "label": "Refine mask",
    }
    maskinfodicts["preselectmask"] = {
        "msg": "Preselected mean regressor seed mask",
        "label": "Preselect mask",
    }
    for pval in [0.05, 0.01, 0.005, 0.001]:
        maskinfodicts[f"p_lt_{(str(pval) + '0').replace('.','p')[0:5]}_mask"] = {
            "msg": f"Setting functional mask to p<{str(pval)}",
            "label": f"p<{str(pval)}",
        }
    print(maskinfodicts[maskname]["msg"])
    ui.setMask_Button.setText(maskinfodicts[maskname]["label"])
    currentdataset.setFuncMaskName(maskname)
    for themap in currentdataset.loadedfuncmaps:
        if maskname == "nomask":
            overlays[themap].setFuncMask(None)
        else:
            overlays[themap].setFuncMask(overlays[maskname].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc=f"setMask({maskname})", orthoimages=True, histogram=True)


def setAtlasMask():
    global overlays, loadedfuncmaps, ui, currentdataset
    print("Using all defined atlas regions as functional mask")
    ui.setMask_Button.setText("Valid mask")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["atlasmask"].data)
    updateUI(callingfunc="setAtlasMask", orthoimages=True, histogram=True)


def overlay_radioButton_clicked(which, enabled):
    global imageadj, overlays, currentdataset, panetomap, ui, overlaybuttons
    global currentloc, atlasaveragingdone
    global verbosity

    if enabled:
        overlaybuttons[which].setChecked(True)
        currentloc.stopMovie()
        if panetomap[which] != "":
            if atlasaveragingdone and (panetomap[which] != "atlas"):
                currentdataset.setfocusmap(panetomap[which] + "_atlasstat")
            else:
                currentdataset.setfocusmap(panetomap[which])
                thedispmin = overlays[currentdataset.focusmap].dispmin
                thedispmax = overlays[currentdataset.focusmap].dispmax
            if verbosity > 1:
                print(f"currentdataset.focusmap set to {currentdataset.focusmap}")
                print(
                    f"overlays[currentdataset.focusmap].LUTname set to {overlays[currentdataset.focusmap].LUTname}"
                )
            if overlays[currentdataset.focusmap].LUTname == "gray":
                ui.gray_radioButton.setChecked(True)
            elif overlays[currentdataset.focusmap].LUTname == "thermal":
                ui.thermal_radioButton.setChecked(True)
            elif overlays[currentdataset.focusmap].LUTname == "plasma":
                ui.plasma_radioButton.setChecked(True)
            elif overlays[currentdataset.focusmap].LUTname == "viridis":
                ui.viridis_radioButton.setChecked(True)
            elif overlays[currentdataset.focusmap].LUTname == "turbo":
                ui.turbo_radioButton.setChecked(True)
            else:
                ui.rainbow_radioButton.setChecked(True)

            if verbosity > 1:
                print(
                    "setting t position limits to ",
                    0,
                    overlays[currentdataset.focusmap].tdim - 1,
                )
            currentloc.setTInfo(
                overlays[currentdataset.focusmap].tdim,
                overlays[currentdataset.focusmap].tr,
                overlays[currentdataset.focusmap].toffset,
            )
            overlays[currentdataset.focusmap].dispmin = thedispmin
            overlays[currentdataset.focusmap].dispmax = thedispmax
            updateDispLimits()
            updateUI(
                callingfunc="overlay_radioButton_clicked",
                histogram=True,
                LUT=True,
                orthoimages=True,
            )


def updateTimepoint(event):
    global data, overlays, simfunc_ax, tr, tpos, timeaxis, currentloc, ui
    global currentdataset
    global vLine
    global verbosity

    if verbosity > 1:
        print("arrived in updateTimepoint")
    pos = event.pos()  ## using signal proxy turns original arguments into a tuple
    mousePoint = simfunc_ax.vb.mapToView(pos)
    tval = mousePoint.x()
    if timeaxis[0] <= tval <= timeaxis[-1]:
        currentloc.tpos = int(tide_util.valtoindex(timeaxis, tval))
        ui.TimeSlider.setValue(currentloc.tpos)
        updateUI(
            callingfunc="updateTimepoint",
            orthoimages=True,
            similarityfunc=True,
            focusvals=True,
        )


def updateLUT():
    global img_colorbar
    global overlays, currentdataset
    global harvestcolormaps
    theLUT = overlays[currentdataset.focusmap].theLUT
    img_colorbar.setLookupTable(theLUT)
    if harvestcolormaps:
        print(imageadj.saveState())
    updateUI(callingfunc="updateLUT", orthoimages=True)


"""def updateTimecoursePlot():
    global roisize
    global timeaxis
    global currentdataset
    global xpos, ypos, zpos
    global overlays, timecourse_ax
    if overlays['corrout'].tdim > 1:
        selected = overlays[currentdataset.focusmap].data[xpos, ypos, zpos, :]
        timecourse_ax.plot(timeaxis, selected, clear=True)"""


def mapwithLUT(theimage, themask, theLUT, dispmin, dispmax):
    offset = dispmin
    scale = len(theLUT) / (dispmax - dispmin)
    scaleddata = np.rint((theimage - offset) * scale).astype("int32")
    scaleddata[np.where(scaleddata < 0)] = 0
    scaleddata[np.where(scaleddata > (len(theLUT) - 1))] = len(theLUT) - 1
    mappeddata = theLUT[scaleddata]
    mappeddata[:, :, 3][np.where(themask < 1)] = 0
    return mappeddata


def updateTFromControls():
    global mainwin, currentloc
    mainwin.setTpos(currentloc.tpos, emitsignal=False)
    updateUI(
        callingfunc="updateTFromControls",
        orthoimages=True,
        similarityfunc=True,
        focusvals=True,
    )


def updateXYZFromControls():
    global mainwin, currentloc
    mainwin.setXYZpos(currentloc.xpos, currentloc.ypos, currentloc.zpos, emitsignal=False)
    updateUI(
        callingfunc="updateXYZFromControls",
        orthoimages=True,
        similarityfunc=True,
        focusvals=True,
    )


def updateXYZFromMainWin():
    global mainwin, currentloc
    currentloc.setXYZpos(mainwin.xpos, mainwin.ypos, mainwin.zpos, emitsignal=False)
    updateUI(callingfunc="updateXYZFromMainWin", similarityfunc=True, focusvals=True)


def updateUI(
    orthoimages=False,
    histogram=False,
    LUT=False,
    similarityfunc=False,
    focusvals=False,
    callingfunc=None,
    verbose=0,
):
    if verbose > 1:
        if callingfunc is None:
            print("updateUI called with:")
        else:
            print("updateUI called from", callingfunc, " with:")
        print("\torthoimages=", orthoimages)
        print("\thistogram=", histogram)
        print("\tLUT=", LUT)
        print("\tsimilarityfunc=", similarityfunc)
        print("\tfocusvals=", focusvals)

    if histogram:
        updateHistogram()
    if LUT:
        updateLUT()
        orthoimages = True
    if similarityfunc:
        updateSimilarityFunc()
    if orthoimages:
        updateOrthoImages()
    if focusvals:
        printfocusvals()


def updateOrthoImages():
    global hist
    global currentdataset
    global maps
    global panetomap
    global ui
    global mainwin, orthoimagedict
    global currentloc
    # global updateTimecoursePlot
    global xdim, ydim, zdim
    global overlays
    global imagadj

    for thismap in panetomap:
        if thismap != "":
            orthoimagedict[thismap].setXYZpos(currentloc.xpos, currentloc.ypos, currentloc.zpos)
            orthoimagedict[thismap].setTpos(currentloc.tpos)
    mainwin.setMap(overlays[currentdataset.focusmap])
    mainwin.setTpos(currentloc.tpos)
    mainwin.updateAllViews()

    ui.TimeSlider.setValue(currentloc.tpos)


def printfocusvals():
    global ui, overlays, currentdataset
    global currentloc
    global simfuncFitter
    logStatus(
        ui.logOutput,
        "\n\nValues at location "
        + "{0},{1},{2}".format(currentloc.xpos, currentloc.ypos, currentloc.zpos),
    )
    indentstring = "   "
    for key in overlays:
        # print(key, overlays[key].report)
        if key != "mnibrainmask":
            # print('key=', key)
            if overlays[key].report:
                overlays[key].setXYZpos(currentloc.xpos, currentloc.ypos, currentloc.zpos)
                overlays[key].setTpos(currentloc.tpos)
                focusval = overlays[key].getFocusVal()
                if key != "atlas":
                    if key != "failimage":
                        outstring = (
                            indentstring
                            + str(overlays[key].label.ljust(26))
                            + str(":")
                            + "{:.3f}".format(round(focusval, 3))
                        )
                        logStatus(ui.logOutput, outstring)
                    else:
                        if focusval > 0.0:
                            if simfuncFitter is not None:
                                failstring = simfuncFitter.diagnosefail(np.uint32(focusval))
                                outstring = (
                                    indentstring
                                    + str(overlays[key].label.ljust(26))
                                    + str(":\n\t    ")
                                    + failstring.replace(", ", "\n\t    ")
                                )
                                logStatus(ui.logOutput, outstring)
                else:
                    outstring = (
                        indentstring
                        + str(overlays[key].label.ljust(26))
                        + str(":")
                        + str(currentdataset.atlaslabels[int(focusval) - 1])
                    )
                    logStatus(ui.logOutput, outstring)


def regressor_radioButton_clicked(theregressor, enabled):
    global currentdataset
    currentdataset.setfocusregressor(theregressor)
    updateRegressor()
    updateRegressorSpectrum()


def activateDataset(currentdataset, ui, win, defaultdict, overlayGraphicsViews, verbosity=0):
    global regressors, overlays
    global mainwin
    global xdim, ydim, zdim, tdim, xpos, ypos, zpos, tpos
    global timeaxis
    global usecorrout
    global orthoimagedict
    global panesinitialized, uiinitialized
    global currentloc

    if uiinitialized:
        currentloc.xdim = currentdataset.xdim
        currentloc.ydim = currentdataset.ydim
        currentloc.xdim = currentdataset.zdim
        currentloc.tdim = currentdataset.tdim
        currentloc.setTpos(currentloc.tpos)

    if verbosity > 1:
        print("getting regressors")
    regressors = currentdataset.getregressors()

    if verbosity > 1:
        print("getting overlays")
    overlays = currentdataset.getoverlays()
    try:
        test = overlays["corrout"].display_state
    except KeyError:
        usecorrout = False

    # activate the appropriate regressor radio buttons
    if verbosity > 1:
        print("activating radio buttons")
    if "prefilt" in regressors.keys():
        ui.prefilt_radioButton.setDisabled(False)
        ui.prefilt_radioButton.show()
    else:
        ui.prefilt_radioButton.setDisabled(True)
        ui.prefilt_radioButton.hide()
    if "postfilt" in regressors.keys():
        ui.postfilt_radioButton.setDisabled(False)
        ui.postfilt_radioButton.show()
    else:
        ui.postfilt_radioButton.setDisabled(True)
        ui.postfilt_radioButton.hide()
    if "pass1" in regressors.keys():
        ui.pass1_radioButton.setDisabled(False)
        ui.pass1_radioButton.show()
    else:
        ui.pass1_radioButton.setDisabled(True)
        ui.pass1_radioButton.hide()
    if "pass2" in regressors.keys():
        ui.pass2_radioButton.setDisabled(False)
        ui.pass2_radioButton.show()
    else:
        ui.pass2_radioButton.setDisabled(True)
        ui.pass2_radioButton.hide()
    if "pass3" in regressors.keys():
        ui.pass3_radioButton.setDisabled(False)
        ui.pass3_radioButton.setText("Pass " + regressors["pass3"].label[4:])
        ui.pass3_radioButton.show()
    else:
        ui.pass3_radioButton.setDisabled(True)
        ui.pass3_radioButton.setText("")
        ui.pass3_radioButton.hide()
    if "pass4" in regressors.keys():
        ui.pass4_radioButton.setDisabled(False)
        ui.pass4_radioButton.setText("Pass " + regressors["pass4"].label[4:])
        ui.pass4_radioButton.show()
    else:
        ui.pass4_radioButton.setDisabled(True)
        ui.pass4_radioButton.setText("")
        ui.pass4_radioButton.hide()

    win.setWindowTitle("TiDePool - " + currentdataset.fileroot[:-1])

    # read in the significance distribution
    if os.path.isfile(currentdataset.fileroot + "sigfit.txt"):
        sighistfitname = currentdataset.fileroot + "sigfit.txt"
    else:
        sighistfitname = None

    #  This is currently very broken, so it's disabled
    # if sighistfitname is not None:
    #    thepcts = np.array([0.95, 0.99, 0.995, 0.999])
    #    thervals = tide_stats.rfromp(sighistfitname, thepcts)
    #    tide_stats.printthresholds(thepcts, thervals, 'Crosscorrelation significance thresholds from data:')

    # set the background image
    if "anatomic" in overlays:
        bgmap = "anatomic"
    else:
        bgmap = None

    # set up the timecourse plot window
    if verbosity > 1:
        print("setting up timecourse plot window")
    xpos = int(currentdataset.xdim) // 2
    ypos = int(currentdataset.ydim) // 2
    zpos = int(currentdataset.zdim) // 2
    if usecorrout:
        timeaxis = (
            np.linspace(
                0.0,
                overlays["corrout"].tdim * overlays["corrout"].tr,
                num=overlays["corrout"].tdim,
                endpoint=False,
            )
            + overlays["corrout"].toffset
        )
    else:
        timeaxis = (
            np.linspace(
                0.0,
                overlays[currentdataset.focusmap].tdim * overlays[currentdataset.focusmap].tr,
                num=overlays[currentdataset.focusmap].tdim,
                endpoint=False,
            )
            + overlays[currentdataset.focusmap].toffset
        )
    tpos = 0

    # set position and scale of images
    if verbosity > 1:
        print("setting position and scale of images")
    lg_imgsize = 256.0
    sm_imgsize = 32.0
    xfov = currentdataset.xdim * currentdataset.xsize
    yfov = currentdataset.ydim * currentdataset.ysize
    zfov = currentdataset.zdim * currentdataset.zsize
    maxfov = np.max([xfov, yfov, zfov])
    # scalefacx = (lg_imgsize / maxfov) * currentdataset.xsize
    # scalefacy = (lg_imgsize / maxfov) * currentdataset.ysize
    # scalefacz = (lg_imgsize / maxfov) * currentdataset.zsize

    if verbosity > 1:
        print("setting overlay defaults")
    for themap in currentdataset.allloadedmaps + currentdataset.loadedfuncmasks:
        overlays[themap].setLUT(
            defaultdict[themap]["colormap"], alpha=LUT_alpha, endalpha=LUT_endalpha
        )
        overlays[themap].setisdisplayed(defaultdict[themap]["display"])
        overlays[themap].setLabel(defaultdict[themap]["label"])
    if verbosity > 1:
        print("done setting overlay defaults")

    if verbosity > 1:
        print("setting geometric masks")
    if "geommask" in overlays:
        thegeommask = overlays["geommask"].data
        if verbosity > 1:
            print("setting geometric mask")
    else:
        thegeommask = None
        if verbosity > 1:
            print("setting geometric mask to None")

    for theoverlay in currentdataset.loadedfuncmaps:
        overlays[theoverlay].setGeomMask(thegeommask)
    if verbosity > 1:
        print("done setting geometric masks")

    if verbosity > 1:
        print("setting functional masks")
        print(currentdataset.loadedfuncmaps)
    for theoverlay in currentdataset.loadedfuncmaps:
        if theoverlay != "failimage":
            if "p_lt_0p050_mask" in overlays and False:  # disable this BBF 2/8/18
                overlays[theoverlay].setFuncMask(overlays["p_lt_0p050_mask"].data)
                ui.setMask_Button.setText("p<0.05")
            else:
                overlays[theoverlay].setFuncMask(overlays["lagmask"].data)
                ui.setMask_Button.setText("Valid mask")
    if verbosity > 1:
        print("done setting functional masks")

    if "anatomic" in overlays:
        overlays["anatomic"].setFuncMask(None)
        overlays["anatomic"].setGeomMask(None)

    if "atlas" in overlays:
        overlays["atlas"].setGeomMask(thegeommask)
        overlays["atlas"].setFuncMask(overlays["atlasmask"].data)

    if not panesinitialized:
        if verbosity > 0:
            for theoverlay in overlays:
                overlays[theoverlay].summarize()

    if verbosity > 1:
        print("focusmap is:", currentdataset.focusmap, "bgmap is:", bgmap)
    if not panesinitialized:
        if bgmap is None:
            mainwin = OrthoImageItem(
                overlays[currentdataset.focusmap],
                ui.main_graphicsView_ax,
                ui.main_graphicsView_cor,
                ui.main_graphicsView_sag,
                imgsize=lg_imgsize,
                enableMouse=True,
                verbose=verbosity,
            )
        else:
            mainwin = OrthoImageItem(
                overlays[currentdataset.focusmap],
                ui.main_graphicsView_ax,
                ui.main_graphicsView_cor,
                ui.main_graphicsView_sag,
                bgmap=overlays[bgmap],
                imgsize=lg_imgsize,
                enableMouse=True,
                verbose=verbosity,
            )
    else:
        mainwin.setMap(overlays[currentdataset.focusmap])

    availablepanes = len(overlayGraphicsViews)
    if verbosity > 0:
        print(f"loading {availablepanes} available panes")
    numnotloaded = 0
    numloaded = 0
    if not panesinitialized:
        orthoimagedict = {}
        for idx, themap in enumerate(currentdataset.dispmaps):
            if overlays[themap].display_state:
                if (numloaded > availablepanes - 1) or (
                    (numloaded > availablepanes - 2) and (themap != "corrout")
                ):
                    if verbosity > 1:
                        print(
                            f"skipping map {themap}({idx}): out of display panes ({numloaded=}, {availablepanes=})"
                        )
                    numnotloaded += 1
                else:
                    if verbosity > 1:
                        print(
                            f"loading map {themap}=({idx}) into pane {numloaded} of {availablepanes}"
                        )
                    if bgmap is None:
                        loadpane(
                            overlays[themap],
                            numloaded,
                            overlayGraphicsViews,
                            overlaybuttons,
                            panetomap,
                            orthoimagedict,
                            sm_imgsize=sm_imgsize,
                        )
                    else:
                        loadpane(
                            overlays[themap],
                            numloaded,
                            overlayGraphicsViews,
                            overlaybuttons,
                            panetomap,
                            orthoimagedict,
                            bgmap=overlays[bgmap],
                            sm_imgsize=sm_imgsize,
                        )
                    numloaded += 1
            else:
                if verbosity > 1:
                    print("not loading map ", themap, "(", idx, "): display_state is False")
    else:
        for thismap in panetomap:
            if thismap != "":
                try:
                    orthoimagedict[thismap].setMap(overlays[thismap])
                except KeyError:
                    pass
    if verbosity > 1:
        print("done loading panes")
    if numnotloaded > 0:
        print("WARNING:", numnotloaded, "maps could not be loaded - not enough panes")

    # record that we've been through once
    panesinitialized = True

    if uiinitialized:
        # update the windows
        updateUI(
            callingfunc="activateDataset",
            orthoimages=True,
            histogram=True,
        )

        # update the regressor
        updateRegressor()
        updateRegressorSpectrum()

        # update the mask menu
        if currentdataset.funcmaskname is not None:
            print(f"updating the mask menu to {currentdataset.funcmaskname}")
            setMask(currentdataset.funcmaskname)


def loadpane(
    themap,
    thepane,
    view,
    button,
    panemap,
    orthoimagedict,
    bgmap=None,
    sm_imgsize=32.0,
):
    if themap.display_state:
        if bgmap is None:
            orthoimagedict[themap.name] = OrthoImageItem(
                themap,
                view[thepane],
                view[thepane],
                view[thepane],
                button=button[thepane],
                imgsize=sm_imgsize,
                verbose=verbosity,
            )
        else:
            orthoimagedict[themap.name] = OrthoImageItem(
                themap,
                view[thepane],
                view[thepane],
                view[thepane],
                button=button[thepane],
                bgmap=bgmap,
                imgsize=sm_imgsize,
                verbose=verbosity,
            )
        panemap[thepane] = themap.name


def tidepool(args):
    global vLine
    global ui, win
    global fileMenu, sel_open, sel_files
    global movierunning
    global focusmap, bgmap, focusregressor
    global maps
    global roi
    global overlays, regressors, regressorfilterlimits, regressorsimcalclimits, loadedfuncmaps, atlasstats, averagingmode
    global mainwin, orthoimagedict, overlaybuttons, panetomap
    global img_colorbar, LUT_alpha, LUT_endalpha
    global lg_imgsize, scalefacx, scalefacy, scalefacz
    global xdim, ydim, zdim, tdim, xpos, ypos, zpos, tpos
    global xsize, ysize, zsize, tr
    global timeaxis
    global usecorrout
    global buttonisdown
    global imageadj
    global harvestcolormaps
    global atlasaveragingdone
    global currentdataset, thesubjects, whichsubject, datafileroots
    global defaultdict, overlayGraphicsViews
    global verbosity
    global panesinitialized, uiinitialized
    global simfuncFitter
    global simfunc_ax, simfuncCurve, simfuncfitCurve, simfuncTLine, simfuncPeakMarker, simfuncCurvePoint, simfuncCaption

    # initialize default values
    averagingmode = None
    atlasaveragingdone = False
    forcetr = False
    buttonisdown = False
    harvestcolormaps = False
    LUT_alpha = 255
    LUT_endalpha = 255
    xdim = 0
    ydim = 0
    zdim = 0
    tdim = 0
    xsize = 0
    ysize = 0
    zsize = 0
    tr = 0
    xpos = 0
    ypos = 0
    zpos = 0
    tpos = 0
    verbosity = 0
    simfuncFitter = None
    datafileroots = []
    panesinitialized = False
    uiinitialized = False
    sel_files = None

    if pyqtversion == 5:
        if args.uistyle == "normal":
            import rapidtide.tidepoolTemplate_alt as uiTemplate
        elif args.uistyle == "big":
            import rapidtide.tidepoolTemplate_big as uiTemplate
        else:
            import rapidtide.tidepoolTemplate as uiTemplate
    else:
        if args.uistyle == "normal":
            import rapidtide.tidepoolTemplate_alt_qt6 as uiTemplate
        elif args.uistyle == "big":
            import rapidtide.tidepoolTemplate_big_qt6 as uiTemplate
        else:
            import rapidtide.tidepoolTemplate_qt6 as uiTemplate

    verbosity = args.verbose
    print(f"verbosity: {verbosity}")

    anatname = args.anatname
    if anatname is not None:
        print("using ", anatname, " as the anatomic background image")
    else:
        anatname = None

    geommaskname = args.geommaskname
    if geommaskname is not None:
        print("using ", geommaskname, " as the geometric mask")
    else:
        geommaskname = None

    if args.datafileroot is not None:
        print("using ", args.datafileroot, " as the root file name list")
        datafileroots = args.datafileroot

    if args.offsettime is not None:
        forceoffset = True
        offsettime = args.offsettime
        print("forcing offset time to ", offsettime)
    else:
        forceoffset = False
        offsettime = 0.0

    if args.trval is not None:
        trval = args.trval
        print("forcing similarity function TR to ", trval)

    userise = args.userise
    if args.userise:
        print("enabling risetime display")

    usecorrout = True
    useatlas = args.useatlas

    # make the main window
    app = QtWidgets.QApplication([])
    print("setting up output window")
    win = QtWidgets.QMainWindow()
    ui = uiTemplate.Ui_MainWindow()
    ui.setupUi(win)
    win.show()
    win.setWindowTitle("TiDePool")

    # create the menu bar
    print("creating menu bar")
    menuBar = win.menuBar()
    fileMenu = menuBar.addMenu("File")
    if pyqtversion == 5:
        qactionfunc = QtWidgets.QAction
    else:
        qactionfunc = QtGui.QAction
    sel_open = qactionfunc("Add dataset...", win)
    sel_open.triggered.connect(datasetPicker)
    fileMenu.addAction(sel_open)
    fileMenu.addSeparator()
    print("done creating menu bar")

    # wire up the ortho image windows for mouse interaction
    vb_colorbar = pg.ViewBox(enableMouse=False)
    vb_colorbar.setRange(
        QtCore.QRectF(0, 0, 25, 256),
        xRange=(0, 25),
        yRange=(0, 255),
        padding=0.0,
        disableAutoRange=True,
    )
    ui.graphicsView_colorbar.setCentralItem(vb_colorbar)
    vb_colorbar.setAspectLocked()
    img_colorbar = pg.ImageItem()
    vb_colorbar.addItem(img_colorbar)

    colorbar = np.zeros((25, 256), dtype=np.float64)
    for i in range(0, 256):
        colorbar[:, i] = i * (1.0 / 255.0)
    # img_colorbar.setImage(colorbar.astype(np.float64), levels=[0.0, 1.0])
    img_colorbar.setImage(colorbar, levels=[0.0, 1.0])

    # configure the gradient scale
    imageadj = pg.GradientWidget(orientation="right", allowAdd=True)
    imageadj.restoreState(gen_gray_state())
    if harvestcolormaps:
        ui.largeimage_horizontalLayout.addWidget(imageadj)

    if args.uistyle == "big":
        extramaps = True
    else:
        extramaps = False
    defaultdict = {
        "lagmask": {
            "colormap": gen_gray_state(),
            "label": "Lag mask",
            "display": False,
            "funcmask": None,
        },
        "geommask": {
            "colormap": gen_gray_state(),
            "label": "Geometric mask",
            "display": False,
            "funcmask": None,
        },
        "refinemask": {
            "colormap": gen_gray_state(),
            "label": "Refine mask",
            "display": False,
            "funcmask": None,
        },
        "meanmask": {
            "colormap": gen_gray_state(),
            "label": "Global mean mask",
            "display": False,
            "funcmask": None,
        },
        "preselectmask": {
            "colormap": gen_gray_state(),
            "label": "Global mean preselect mask",
            "display": False,
            "funcmask": None,
        },
        "brainmask": {
            "colormap": gen_gray_state(),
            "label": "Brain mask",
            "display": False,
            "funcmask": None,
        },
        "p_lt_0p050_mask": {
            "colormap": gen_gray_state(),
            "label": "p<0.05",
            "display": False,
            "funcmask": None,
        },
        "p_lt_0p010_mask": {
            "colormap": gen_gray_state(),
            "label": "p<0.01",
            "display": False,
            "funcmask": None,
        },
        "p_lt_0p005_mask": {
            "colormap": gen_gray_state(),
            "label": "p<0.005",
            "display": False,
            "funcmask": None,
        },
        "p_lt_0p001_mask": {
            "colormap": gen_gray_state(),
            "label": "p<0.001",
            "display": False,
            "funcmask": None,
        },
        "risetime_epoch_0": {
            "colormap": gen_viridis_state(),
            "label": "Rise times",
            "display": False,
            "funcmask": "p_lt_0p050_mask",
        },
        "maxamp_epoch_0": {
            "colormap": gen_viridis_state(),
            "label": "CVR",
            "display": False,
            "funcmask": "p_lt_0p050_mask",
        },
        "starttime_epoch_0": {
            "colormap": gen_viridis_state(),
            "label": "Start times",
            "display": False,
            "funcmask": "p_lt_0p050_mask",
        },
        "lagtimes": {
            "colormap": gen_viridis_state(),
            "label": "Lag time",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "lagtimesrefined": {
            "colormap": gen_viridis_state(),
            "label": "Refined lag time",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "timepercentile": {
            "colormap": gen_viridis_state(),
            "label": "Lag percentile",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "lagstrengths": {
            "colormap": gen_thermal_state(),
            "label": "Similarity coefficient",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "lagstrengthsrefined": {
            "colormap": gen_thermal_state(),
            "label": "Refined similarity coefficient",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "lagsigma": {
            "colormap": gen_plasma_state(),
            "label": "Similarity width",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "MTT": {
            "colormap": gen_plasma_state(),
            "label": "MTT",
            "display": extramaps,
            "funcmask": "p_lt_0p050_mask",
        },
        "R2": {
            "colormap": gen_thermal_state(),
            "label": "GLM Fit R2",
            "display": extramaps,
            "funcmask": "p_lt_0p050_mask",
        },
        "CoV": {
            "colormap": gen_thermal_state(),
            "label": "Coefficient of variation",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "confoundR2": {
            "colormap": gen_thermal_state(),
            "label": "Confound Fit R2",
            "display": extramaps,
            "funcmask": "p_lt_0p050_mask",
        },
        "varBefore": {
            "colormap": gen_thermal_state(),
            "label": "LFO variance before GLM",
            "display": extramaps,
            "funcmask": "p_lt_0p050_mask",
        },
        "varAfter": {
            "colormap": gen_thermal_state(),
            "label": "LFO variance after GLM",
            "display": extramaps,
            "funcmask": "p_lt_0p050_mask",
        },
        "varChange": {
            "colormap": gen_thermal_state(),
            "label": "LFO variance decrease %",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "corrout": {
            "colormap": gen_thermal_state(),
            "label": "Similarity function",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "fitNorm": {
            "colormap": gen_thermal_state(),
            "label": "fitNorm",
            "display": False,
            "funcmask": "p_lt_0p050_mask",
        },
        "gaussout": {
            "colormap": gen_thermal_state(),
            "label": "Gauss fit",
            "display": False,
            "funcmask": "p_lt_0p050_mask",
        },
        "failimage": {
            "colormap": gen_spectrum_state(),
            "label": "Fit failure reason",
            "display": extramaps,
            "funcmask": None,
        },
        "anatomic": {
            "colormap": gen_gray_state(),
            "label": "Anatomic image",
            "display": False,
            "funcmask": None,
        },
        "atlas": {
            "colormap": gen_spectrum_state(),
            "label": "Atlas territories",
            "display": True,
            "funcmask": None,
        },
        "fitcoff": {
            "colormap": gen_thermal_state(),
            "label": "GLM fit coefficient",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "neglog10p": {
            "colormap": gen_thermal_state(),
            "label": "Correlation fit -log10p",
            "display": extramaps,
            "funcmask": "None",
        },
        "delayoffset": {
            "colormap": gen_viridis_state(),
            "label": "Lag time adjustments",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
    }

    # set up the timecourse plot window
    """if usetime:
        tcwin = ui.timecourse_graphicsView
        global timecourse_ax
        timecourse_ax = tcwin.addPlot()
        timecourse_ax.setZValue(10)
        vLine = pg.InfiniteLine(angle=90, movable=False, pen='g')
        vLine.setZValue(20)
        timecourse_ax.addItem(vLine)
        timecourse_ax.setTitle('Timecourse')
        timecourse_ax.enableAutoRange()"""

    # wire up the atlas averaging checkboxes
    if args.useatlas:
        ui.raw_radioButton.setDisabled(False)
        ui.mean_radioButton.setDisabled(False)
        ui.median_radioButton.setDisabled(False)
        ui.CoV_radioButton.setDisabled(False)
        ui.std_radioButton.setDisabled(False)
        ui.MAD_radioButton.setDisabled(False)
    else:
        ui.raw_radioButton.setDisabled(True)
        ui.mean_radioButton.setDisabled(True)
        ui.median_radioButton.setDisabled(True)
        ui.CoV_radioButton.setDisabled(True)
        ui.std_radioButton.setDisabled(True)
        ui.MAD_radioButton.setDisabled(True)

    ui.raw_radioButton.clicked.connect(raw_radioButton_clicked)
    ui.mean_radioButton.clicked.connect(mean_radioButton_clicked)
    ui.median_radioButton.clicked.connect(median_radioButton_clicked)
    ui.CoV_radioButton.clicked.connect(CoV_radioButton_clicked)
    ui.std_radioButton.clicked.connect(std_radioButton_clicked)
    ui.MAD_radioButton.clicked.connect(MAD_radioButton_clicked)

    # wire up the colormap radio buttons
    ui.gray_radioButton.clicked.connect(gray_radioButton_clicked)
    ui.thermal_radioButton.clicked.connect(thermal_radioButton_clicked)
    ui.plasma_radioButton.clicked.connect(plasma_radioButton_clicked)
    ui.viridis_radioButton.clicked.connect(viridis_radioButton_clicked)
    ui.turbo_radioButton.clicked.connect(turbo_radioButton_clicked)
    ui.rainbow_radioButton.clicked.connect(rainbow_radioButton_clicked)

    # wire up the transparency checkbox
    ui.transparency_checkBox.stateChanged.connect(transparencyCheckboxClicked)

    overlaybuttons = [
        ui.overlay_radioButton_01,
        ui.overlay_radioButton_02,
        ui.overlay_radioButton_03,
        ui.overlay_radioButton_04,
        ui.overlay_radioButton_05,
        ui.overlay_radioButton_06,
        ui.overlay_radioButton_07,
        ui.overlay_radioButton_08,
    ]
    if args.uistyle == "big":
        overlaybuttons += [
            ui.overlay_radioButton_09,
            ui.overlay_radioButton_10,
            ui.overlay_radioButton_11,
            ui.overlay_radioButton_12,
            ui.overlay_radioButton_13,
            ui.overlay_radioButton_14,
            ui.overlay_radioButton_15,
            ui.overlay_radioButton_16,
        ]
    for i in range(len(overlaybuttons)):
        clickfunc = partial(overlay_radioButton_clicked, i)
        overlaybuttons[i].clicked.connect(clickfunc)

    for button in overlaybuttons:
        button.setDisabled(True)
        button.hide()

    overlayGraphicsViews = [
        ui.overlay_graphicsView_01,
        ui.overlay_graphicsView_02,
        ui.overlay_graphicsView_03,
        ui.overlay_graphicsView_04,
        ui.overlay_graphicsView_05,
        ui.overlay_graphicsView_06,
        ui.overlay_graphicsView_07,
        ui.overlay_graphicsView_08,
    ]
    if args.uistyle == "big":
        overlayGraphicsViews += [
            ui.overlay_graphicsView_09,
            ui.overlay_graphicsView_10,
            ui.overlay_graphicsView_11,
            ui.overlay_graphicsView_12,
            ui.overlay_graphicsView_13,
            ui.overlay_graphicsView_14,
            ui.overlay_graphicsView_15,
            ui.overlay_graphicsView_16,
        ]
    panetomap = []

    for theview in overlayGraphicsViews:
        panetomap.append("")
        theview.hide()

    # define things for the popup mask menu
    popMaskMenu = QtWidgets.QMenu(win)
    if pyqtversion == 5:
        qactionfunc = QtWidgets.QAction
    else:
        qactionfunc = QtGui.QAction
    sel_nomask = qactionfunc("No mask", win)
    sel_lagmask = qactionfunc("Valid fit", win)
    sel_brainmask = qactionfunc("Externally provided brain mask", win)
    sel_refinemask = qactionfunc("Voxels used in refine", win)
    sel_meanmask = qactionfunc("Voxels used in mean regressor calculation", win)
    sel_preselectmask = qactionfunc(
        "Voxels chosen for the mean regressor calculation in the preselect pass", win
    )
    sel_0p05 = qactionfunc("p<0.05", win)
    sel_0p01 = qactionfunc("p<0.01", win)
    sel_0p005 = qactionfunc("p<0.005", win)
    sel_0p001 = qactionfunc("p<0.001", win)

    sel_nomask.triggered.connect(partial(setMask, "nomask"))
    sel_lagmask.triggered.connect(partial(setMask, "lagmask"))
    sel_brainmask.triggered.connect(partial(setMask, "brainmask"))
    sel_refinemask.triggered.connect(partial(setMask, "refinemask"))
    sel_meanmask.triggered.connect(partial(setMask, "meanmask"))
    sel_preselectmask.triggered.connect(partial(setMask, "preselectmask"))
    sel_0p05.triggered.connect(partial(setMask, "p_lt_0p050_mask"))
    sel_0p01.triggered.connect(partial(setMask, "p_lt_0p010_mask"))
    sel_0p005.triggered.connect(partial(setMask, "p_lt_0p005_mask"))
    sel_0p001.triggered.connect(partial(setMask, "p_lt_0p001_mask"))
    popMaskMenu.addAction(sel_nomask)
    numspecial = 0

    # configure the mask selection popup menu
    def on_mask_context_menu(point):
        # show context menu
        popMaskMenu.exec(ui.setMask_Button.mapToGlobal(point))

    ui.setMask_Button.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
    ui.setMask_Button.customContextMenuRequested.connect(on_mask_context_menu)

    # configure the file selection popup menu
    def on_file_context_menu(point):
        # show context menu
        popMaskMenu.exec(ui.setFile_Button.mapToGlobal(point))

    try:
        ui.setFile_Button.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        ui.setFile_Button.customContextMenuRequested.connect(on_file_context_menu)
        setfilebuttonexists = True
    except AttributeError:
        setfilebuttonexists = False

    # wire up the regressor selection radio buttons
    regressorbuttons = [
        ui.prefilt_radioButton,
        ui.postfilt_radioButton,
        ui.pass1_radioButton,
        ui.pass2_radioButton,
        ui.pass3_radioButton,
        ui.pass4_radioButton,
    ]
    ui.prefilt_radioButton.clicked.connect(partial(regressor_radioButton_clicked, "prefilt"))
    ui.postfilt_radioButton.clicked.connect(partial(regressor_radioButton_clicked, "postfilt"))
    ui.pass1_radioButton.clicked.connect(partial(regressor_radioButton_clicked, "pass1"))
    ui.pass2_radioButton.clicked.connect(partial(regressor_radioButton_clicked, "pass2"))
    ui.pass3_radioButton.clicked.connect(partial(regressor_radioButton_clicked, "pass3"))
    ui.pass4_radioButton.clicked.connect(partial(regressor_radioButton_clicked, "pass4"))

    for thebutton in regressorbuttons:
        thebutton.setDisabled(True)
        thebutton.hide()

    # read in all the datasets
    thesubjects = []
    whichsubject = 0
    if len(datafileroots) > 0:
        print("loading prespecified datasets...")
        for thisdatafileroot in datafileroots:
            addDataset(
                thisdatafileroot,
                anatname=anatname,
                geommaskname=geommaskname,
                userise=userise,
                usecorrout=usecorrout,
                useatlas=useatlas,
                forcetr=forcetr,
                forceoffset=forceoffset,
                offsettime=offsettime,
                ignoredimmatch=args.ignoredimmatch,
            )
        currentdataset = thesubjects[whichsubject]
        activateDataset(
            currentdataset,
            ui,
            win,
            defaultdict,
            overlayGraphicsViews,
            verbosity=verbosity,
        )
        # update the file menu
        updateFileMenu()
    else:
        # get inputfile root name if necessary
        datasetPicker()

    # check to see that something is loaded
    if len(thesubjects) == 0:
        print("No input datasets specified - exiting.")
        sys.exit()

    # wire up the display range controls
    ui.resetDispLimits_Button.clicked.connect(resetDispLimits)
    ui.resetDispSmart_Button.clicked.connect(resetDispSmart)
    ui.saveDisp_Button.clicked.connect(mainwin.saveDisp)
    ui.dispmin_doubleSpinBox.valueChanged.connect(getMinDispLimit)
    ui.dispmax_doubleSpinBox.valueChanged.connect(getMaxDispLimit)

    if verbosity > 1:
        print("done setting up histogram window")

    # set up the histogram
    if verbosity > 1:
        print("about to set up the histogram")
    global hist_ax
    histtitle = ui.histogram_groupBox.title()
    if verbosity > 1:
        print("current histtitle", histtitle)
    ui.histogram_groupBox.setWindowTitle("blah")
    histtitle = ui.histogram_groupBox.title()
    if verbosity > 1:
        print("current histtitle", histtitle)
    histwin = ui.histogram_graphicsView
    hist_ax = histwin.addPlot()

    # set up the similarity function window
    if verbosity > 1:
        print("about to set up the similarity function window")

    try:
        thesimfunc_groupBox = ui.simfunc_groupBox
        simfuncwindowexists = True
    except AttributeError:
        print("no similarity function window - cancelling setup")
        simfuncwindowexists = False

    if usecorrout and simfuncwindowexists:
        if verbosity > 1:
            print("setting up similarity window")
        simfunctitle = ui.simfunc_groupBox.title()
        if verbosity > 1:
            print("current simfunctitle", simfunctitle)
        ui.simfunc_groupBox.setWindowTitle("blah")
        simfunctitle = ui.simfunc_groupBox.title()
        if verbosity > 1:
            print("current simfunctitle", simfunctitle)
        simfuncwin = ui.simfunc_graphicsView
        simfunc_ax = simfuncwin.addPlot()
        simfuncCurve = simfunc_ax.plot(
            timeaxis,
            0.0 * timeaxis,
            title="Similarity function",
            stepMode=False,
            fillLevel=0,
            pen=pg.mkPen("w", width=1),
            clear=True,
            clickable=True,
        )
        """simfuncfitCurve = simfunc_ax.plot(timeaxis, 0.0 * timeaxis,
                                        stepMode=False,
                                        fillLevel=0,
                                        pen=pg.mkPen('r', width=1),
                                        clear=True,
                                        clickable=True)"""
        simfuncTLine = simfunc_ax.addLine(x=timeaxis[0], pen="g")
        simfuncCurvePoint = pg.CurvePoint(simfuncCurve)
        simfuncPeakMarker = pg.ScatterPlotItem(
            x=[], y=[], pen=pg.mkPen("w", width=1), brush="b", size=10, pxMode=True
        )
        simfunc_ax.addItem(simfuncPeakMarker)
        simfuncCaption = pg.TextItem("placeholder", anchor=(0.5, 1.0))
        simfuncCaption.setParentItem(simfuncCurvePoint)
        simfuncCurve.scene().sigMouseClicked.connect(updateTimepoint)
        simfuncFitter = SimilarityFunctionFitter()
        print("simfuncFitter has been defined")
    else:
        simfunc_ax = None

    # update the main window now that simfuncFitter has been defined
    mainwin.updated.connect(updateXYZFromMainWin)

    # set up the regressor timecourse window
    if verbosity > 1:
        print("about to set up the timecourse")
    global regressor_ax
    regressorwin = ui.regressortimecourse_graphicsView
    regressor_ax = regressorwin.addPlot()
    updateRegressor()

    # set up the regressor spectrum window
    if verbosity > 1:
        print("about to set up the spectrum")
    global regressorspectrum_ax
    regressorspectrumwin = ui.regressorspectrum_graphicsView
    regressorspectrum_ax = regressorspectrumwin.addPlot()
    updateRegressorSpectrum()

    # set up the mask selection popup window
    if verbosity > 1:
        print("loadedfuncmasks", currentdataset.loadedfuncmasks)
    if len(currentdataset.loadedfuncmasks) > 0:
        popMaskMenu.addSeparator()
        if "lagmask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_lagmask)
            numspecial += 1
        if "brainmask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_brainmask)
            numspecial += 1
        if "refinemask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_refinemask)
            numspecial += 1
        if "meanmask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_meanmask)
            numspecial += 1
        if "preselectmask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_preselectmask)
            numspecial += 1
        if numspecial > 0:
            popMaskMenu.addSeparator()
        if "p_lt_0p050_mask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_0p05)
        if "p_lt_0p010_mask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_0p01)
        if "p_lt_0p005_mask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_0p005)
        if "p_lt_0p001_mask" in currentdataset.loadedfuncmasks:
            popMaskMenu.addAction(sel_0p001)

    # initialize the location picker
    global currentloc
    currentloc = xyztlocation(
        xpos,
        ypos,
        zpos,
        tpos,
        currentdataset.xdim,
        currentdataset.ydim,
        currentdataset.zdim,
        currentdataset.tdim,
        overlays[currentdataset.focusmap].toffset,
        overlays[currentdataset.focusmap].tr,
        overlays[currentdataset.focusmap].affine,
        XPosSpinBox=ui.pixnumX_doubleSpinBox,
        YPosSpinBox=ui.pixnumY_doubleSpinBox,
        ZPosSpinBox=ui.pixnumZ_doubleSpinBox,
        TPosSpinBox=ui.pixnumT_doubleSpinBox,
        XCoordSpinBox=ui.coordX_doubleSpinBox,
        YCoordSpinBox=ui.coordY_doubleSpinBox,
        ZCoordSpinBox=ui.coordZ_doubleSpinBox,
        TCoordSpinBox=ui.coordT_doubleSpinBox,
        TimeSlider=ui.TimeSlider,
        runMovieButton=ui.runMovieButton,
    )
    currentloc.updatedXYZ.connect(updateXYZFromControls)
    currentloc.updatedT.connect(updateTFromControls)
    if usecorrout:
        updateSimilarityFunc()
    updateHistogram()

    updateLUT()

    # zoom to fit imageo
    # timecourse_ax.enableAutoRange()

    # select the first pane
    overlay_radioButton_clicked(0, True)

    # have to do this after the windows are created
    imageadj.sigGradientChanged.connect(updateLUT)

    updateUI(callingfunc="main thread", orthoimages=True, focusvals=True)
    updateRegressor()
    updateRegressorSpectrum()
    uiinitialized = True

    # for profiling
    """for i in range(20):
        selectDataset((i + 1) % 2)
    sys.exit(0)"""

    QtWidgets.QApplication.instance().exec()
