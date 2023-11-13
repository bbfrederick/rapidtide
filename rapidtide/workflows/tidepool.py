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
"""
A simple GUI for looking at the results of a rapidtide analysis
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
from nibabel.affines import apply_affine
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from statsmodels.robust.scale import mad

import rapidtide.util as tide_util
from rapidtide.Colortables import *
from rapidtide.helper_classes import SimilarityFunctionFitter
from rapidtide.OrthoImageItem import OrthoImageItem
from rapidtide.RapidtideDataset import RapidtideDataset

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
        help="Use this dataset (skip initial selection step)",
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
        "--oldstyle",
        help="Use the old style layout",
        dest="compact",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--verbosity",
        help="Specify level of truly devastatingly boring console messages.  Default is 0",
        dest="verbose",
        metavar="VERBOSITY",
        type=int,
        default=1,
    )

    return parser


def selectFile():
    global datafileroot
    mydialog = QtWidgets.QFileDialog()
    options = mydialog.Options()
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
        if self.TCoordSpinBox is not None:
            self.TCoordSpinBox.setValue(self.tcoord)
        if self.TimeSlider is not None:
            self.TimeSlider.setValue((self.tpos))
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


def logstatus(thetextbox, thetext):
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
    global overlays, datafileroot, atlasstats, averagingmode, currentdataset
    global atlasaveragingdone
    print("in calcAtlasStats")
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
                    atlasstats[themap][region]["mean"] = np.mean(
                        overlays[themap].data[maskedregion]
                    )
                    atlasstats[themap][region]["median"] = np.median(
                        overlays[themap].data[maskedregion]
                    )
                    atlasstats[themap][region]["robustmean"] = mad(
                        overlays[themap].data[maskedregion]
                    )
                    atlasstats[themap][region]["std"] = np.std(overlays[themap].data[maskedregion])
                    atlasstats[themap][region]["MAD"] = mad(overlays[themap].data[maskedregion])
                atlasstatmap = overlays[themap].duplicate(
                    themap + "_atlasstat", overlays[themap].label
                )
                atlasstatmap.funcmask = overlays["atlasmask"].data
                atlasstatmap.data *= 0.0
                atlasstatmap.maskData()
                atlasstatmap.updateStats()
                overlays[themap + "_atlasstat"] = atlasstatmap
        print("done performing atlas averaging")
        for thestat in ["mean", "std", "robustmean", "median", "MAD"]:
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
                datafileroot + currentdataset.atlasname + "_" + thestat + ".txt",
                sep="\t",
                index=False,
            )
        atlasaveragingdone = True
    else:
        print("cannot perform average - no atlas map found")


def updateAtlasStats():
    global overlays, datafileroot, atlasstats, averagingmode, currentdataset
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


"""
def doAtlasAveraging(state):
    global atlasaveragingdone
    print('in doAtlasAveraging')
    if state == QtCore.Qt.Checked:
        atlasaveragingdone = True
        print('atlas averaging is turned on')
    else:
        atlasaveragingdone = False
        print('atlas averaging is turned off')
    updateOrthoImages()
"""


def updateAveragingMode():
    global averagingmode, focusmap
    global atlasaveragingdone
    global overlays
    global currentdataset
    print("in updateAveragingMode")
    if ("atlas" in overlays) and (not atlasaveragingdone) and False:
        calcAtlasStats()
        set_atlasmask()
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


def robustmean_radioButton_clicked(enabled):
    global averagingmode
    if enabled:
        print("in robustmean_radioButton_clicked")
        averagingmode = "robustmean"
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


def transparency_checkbox_clicked():
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


def flame_radioButton_clicked(enabled):
    global imageadj, overlays, currentdataset, LUT_alpha, LUT_endalpha
    if enabled:
        overlays[currentdataset.focusmap].setLUT(
            gen_flame_state(), alpha=LUT_alpha, endalpha=LUT_endalpha
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


def jet_radioButton_clicked(enabled):
    global imageadj, overlays, currentdataset, LUT_alpha, LUT_endalpha
    if enabled:
        overlays[currentdataset.focusmap].setLUT(
            gen_cyclic_state(), alpha=LUT_alpha, endalpha=LUT_endalpha
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


def set_atlasmask():
    global overlays, loadedfuncmaps, ui, currentdataset
    print("Using all defined atlas regions as functional mask")
    ui.setMask_Button.setText("Valid mask")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["atlasmask"].data)
    updateUI(callingfunc="set_atlasmask", orthoimages=True, histogram=True)


def set_lagmask():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("Using valid fit points as functional mask")
    ui.setMask_Button.setText("Valid mask")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["lagmask"].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_lagmask()", orthoimages=True, histogram=True)


def set_refinemask():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("Voxel refinement mask")
    ui.setMask_Button.setText("Refine mask")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["refinemask"].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_refinemask", orthoimages=True, histogram=True)


def set_meanmask():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("Mean regressor seed mask")
    ui.setMask_Button.setText("Mean mask")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["meanmask"].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_meanmask", orthoimages=True, histogram=True)


def set_preselectmask():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("Preselected mean regressor seed mask")
    ui.setMask_Button.setText("Preselect mask")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["preselectmask"].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_preselectmask", orthoimages=True, histogram=True)


def set_nomask():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("disabling functional mask")
    ui.setMask_Button.setText("No Mask")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(None)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_nomask", orthoimages=True, histogram=True)


def set_0p05():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("setting functional mask to p<0.05")
    ui.setMask_Button.setText("p<0.05")
    # overlays['jit_mask'] = tide_stats.makepmask(overlays['lagstrengths'], 0.05, sighistfit, onesided=True)
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["p_lt_0p050_mask"].data)
        # overlays[themap].setFuncMask(overlays['jit_mask'].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_0p05", orthoimages=True, histogram=True)


def set_0p01():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("setting functional mask to p<0.01")
    ui.setMask_Button.setText("p<0.01")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["p_lt_0p010_mask"].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_0p01", orthoimages=True, histogram=True)


def set_0p005():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("setting functional mask to p<0.005")
    ui.setMask_Button.setText("p<0.005")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["p_lt_0p005_mask"].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_0p005", orthoimages=True, histogram=True)


def set_0p001():
    global overlays, loadedfuncmaps, ui, atlasaveragingdone, currentdataset
    print("setting functional mask to p<0.001")
    ui.setMask_Button.setText("p<0.001")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["p_lt_0p001_mask"].data)
    atlasaveragingdone = False
    updateAveragingMode()
    updateUI(callingfunc="set_0p001", orthoimages=True, histogram=True)


def overlay_radioButton_01_clicked(enabled):
    overlay_radioButton_clicked(0, enabled)


def overlay_radioButton_02_clicked(enabled):
    overlay_radioButton_clicked(1, enabled)


def overlay_radioButton_03_clicked(enabled):
    overlay_radioButton_clicked(2, enabled)


def overlay_radioButton_04_clicked(enabled):
    overlay_radioButton_clicked(3, enabled)


def overlay_radioButton_05_clicked(enabled):
    overlay_radioButton_clicked(4, enabled)


def overlay_radioButton_06_clicked(enabled):
    overlay_radioButton_clicked(5, enabled)


def overlay_radioButton_07_clicked(enabled):
    overlay_radioButton_clicked(6, enabled)


def overlay_radioButton_08_clicked(enabled):
    overlay_radioButton_clicked(7, enabled)


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
            if verbosity > 1:
                print("currentdataset.focusmap set to ", currentdataset.focusmap)
            if overlays[currentdataset.focusmap].lut_state == gen_gray_state():
                ui.gray_radioButton.setChecked(True)
            elif overlays[currentdataset.focusmap].lut_state == gen_thermal_state():
                ui.thermal_radioButton.setChecked(True)
            elif overlays[currentdataset.focusmap].lut_state == gen_flame_state():
                ui.flame_radioButton.setChecked(True)
            elif overlays[currentdataset.focusmap].lut_state == gen_viridis_state():
                ui.viridis_radioButton.setChecked(True)
            elif overlays[currentdataset.focusmap].lut_state == gen_cyclic_state():
                ui.jet_radioButton.setChecked(True)
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
    global mainwin, orthoimages
    global currentloc
    # global updateTimecoursePlot
    global xdim, ydim, zdim
    global overlays
    global imagadj

    for thismap in panetomap:
        if thismap != "":
            orthoimages[thismap].setXYZpos(currentloc.xpos, currentloc.ypos, currentloc.zpos)
            orthoimages[thismap].setTpos(currentloc.tpos)
    mainwin.setMap(overlays[currentdataset.focusmap])
    mainwin.setTpos(currentloc.tpos)
    mainwin.updateAllViews()

    ui.TimeSlider.setValue(currentloc.tpos)


def printfocusvals():
    global ui, overlays, currentdataset
    global currentloc
    global simfuncFitter
    logstatus(
        ui.logOutput,
        "\n\nValues at location "
        + "{0},{1},{2}".format(currentloc.xpos, currentloc.ypos, currentloc.zpos),
    )
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
                            "\t"
                            + str(overlays[key].label.ljust(26))
                            + str(":")
                            + "{:.3f}".format(round(focusval, 3))
                        )
                        logstatus(ui.logOutput, outstring)
                    else:
                        if focusval > 0.0:
                            failstring = simfuncFitter.diagnosefail(np.uint32(focusval))
                            outstring = (
                                "\t"
                                + str(overlays[key].label.ljust(26))
                                + str(":\n\t    ")
                                + failstring.replace(", ", "\n\t    ")
                            )
                            logstatus(ui.logOutput, outstring)
                else:
                    outstring = (
                        "\t"
                        + str(overlays[key].label.ljust(26))
                        + str(":")
                        + str(currentdataset.atlaslabels[int(focusval) - 1])
                    )
                    logstatus(ui.logOutput, outstring)


def prefilt_radioButton_clicked(enabled):
    global currentdataset
    currentdataset.setfocusregressor("prefilt")
    updateRegressor()
    updateRegressorSpectrum()


def postfilt_radioButton_clicked(enabled):
    global currentdataset
    currentdataset.setfocusregressor("postfilt")
    updateRegressor()
    updateRegressorSpectrum()


def pass1_radioButton_clicked(enabled):
    global currentdataset
    currentdataset.setfocusregressor("pass1")
    updateRegressor()
    updateRegressorSpectrum()


def pass2_radioButton_clicked(enabled):
    global currentdataset
    currentdataset.setfocusregressor("pass2")
    updateRegressor()
    updateRegressorSpectrum()


def pass3_radioButton_clicked(enabled):
    global currentdataset
    currentdataset.setfocusregressor("pass3")
    updateRegressor()
    updateRegressorSpectrum()


def pass4_radioButton_clicked(enabled):
    global currentdataset
    currentdataset.setfocusregressor("pass4")
    updateRegressor()
    updateRegressorSpectrum()


def tidepool(args):
    global vLine
    global ui, win
    global movierunning
    global focusmap, bgmap
    global maps
    global roi
    global overlays, regressors, regressorfilterlimits, loadedfuncmaps, atlasstats, averagingmode
    global mainwin, orthoimages, overlaybuttons, panetomap
    global img_colorbar, LUT_alpha, LUT_endalpha
    global lg_imgsize, scalefacx, scalefacy, scalefacz
    global xdim, ydim, zdim, tdim, xpos, ypos, zpos, tpos
    global xsize, ysize, zsize, tr
    global timeaxis
    global buttonisdown
    global imageadj
    global harvestcolormaps
    global datafileroot
    global atlasaveragingdone
    global currentdataset
    global verbosity

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

    if args.compact:
        import rapidtide.tidepoolTemplate_alt as uiTemplate
    else:
        import rapidtide.tidepoolTemplate as uiTemplate

    verbosity = args.verbose
    print(f"verbosity: {args.verbose}")

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

    datafileroot = args.datafileroot
    if datafileroot is not None:
        print("using ", datafileroot, " as the root file name ")
    else:
        datafileroot = ""

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

    # get inputfile root name if necessary
    if datafileroot == "":
        selectFile()
    if datafileroot == "":
        print("No input file specified - exiting.")
        sys.exit()

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
            "label": "Lag times",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "lagstrengths": {
            "colormap": gen_thermal_state(),
            "label": "Similarity coefficient",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "lagsigma": {
            "colormap": gen_spectrum_state(),
            "label": "Similarity width",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "MTT": {
            "colormap": gen_spectrum_state(),
            "label": "MTT",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "fitNorm": {
            "colormap": gen_thermal_state(),
            "label": "fitNorm",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "R2": {
            "colormap": gen_thermal_state(),
            "label": "R2",
            "display": True,
            "funcmask": "p_lt_0p050_mask",
        },
        "corrout": {
            "colormap": gen_thermal_state(),
            "label": "Similarity function",
            "display": True,
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
            "display": False,
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
    ui.raw_radioButton.setDisabled(True)
    ui.mean_radioButton.setDisabled(True)
    ui.median_radioButton.setDisabled(True)
    ui.robustmean_radioButton.setDisabled(True)
    ui.std_radioButton.setDisabled(True)
    ui.MAD_radioButton.setDisabled(True)

    """ui.raw_radioButton.hide()
    ui.mean_radioButton.hide()
    ui.median_radioButton.hide()
    ui.robustmean_radioButton.hide()
    ui.std_radioButton.hide()
    ui.MAD_radioButton.hide()"""

    ui.raw_radioButton.clicked.connect(raw_radioButton_clicked)
    ui.mean_radioButton.clicked.connect(mean_radioButton_clicked)
    ui.median_radioButton.clicked.connect(median_radioButton_clicked)
    ui.robustmean_radioButton.clicked.connect(robustmean_radioButton_clicked)
    ui.std_radioButton.clicked.connect(std_radioButton_clicked)
    ui.MAD_radioButton.clicked.connect(MAD_radioButton_clicked)

    # wire up the colormap radio buttons
    ui.gray_radioButton.clicked.connect(gray_radioButton_clicked)
    ui.thermal_radioButton.clicked.connect(thermal_radioButton_clicked)
    ui.flame_radioButton.clicked.connect(flame_radioButton_clicked)
    ui.viridis_radioButton.clicked.connect(viridis_radioButton_clicked)
    ui.jet_radioButton.clicked.connect(jet_radioButton_clicked)
    ui.rainbow_radioButton.clicked.connect(rainbow_radioButton_clicked)

    # wire up the transparency checkbox
    ui.transparency_checkBox.stateChanged.connect(transparency_checkbox_clicked)

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
    overlaybuttons[0].clicked.connect(overlay_radioButton_01_clicked)
    overlaybuttons[1].clicked.connect(overlay_radioButton_02_clicked)
    overlaybuttons[2].clicked.connect(overlay_radioButton_03_clicked)
    overlaybuttons[3].clicked.connect(overlay_radioButton_04_clicked)
    overlaybuttons[4].clicked.connect(overlay_radioButton_05_clicked)
    overlaybuttons[5].clicked.connect(overlay_radioButton_06_clicked)
    overlaybuttons[6].clicked.connect(overlay_radioButton_07_clicked)
    overlaybuttons[7].clicked.connect(overlay_radioButton_08_clicked)
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

    panetomap = []

    for theview in overlayGraphicsViews:
        panetomap.append("")
        theview.hide()

    # define things for the popup mask menu
    popMenu = QtWidgets.QMenu(win)
    sel_nomask = QtWidgets.QAction("No mask", win)
    sel_nomask.triggered.connect(set_nomask)
    sel_lagmask = QtWidgets.QAction("Valid fit", win)
    sel_lagmask.triggered.connect(set_lagmask)
    sel_refinemask = QtWidgets.QAction("Voxels used in refine", win)
    sel_meanmask = QtWidgets.QAction("Voxels used in mean regressor calculation", win)
    sel_preselectmask = QtWidgets.QAction(
        "Voxels chosen for the mean regressor calculation in the preselect pass", win
    )
    sel_refinemask.triggered.connect(set_refinemask)
    sel_meanmask.triggered.connect(set_meanmask)
    sel_preselectmask.triggered.connect(set_preselectmask)
    sel_0p05 = QtWidgets.QAction("p<0.05", win)
    sel_0p05.triggered.connect(set_0p05)
    sel_0p01 = QtWidgets.QAction("p<0.01", win)
    sel_0p01.triggered.connect(set_0p01)
    sel_0p005 = QtWidgets.QAction("p<0.005", win)
    sel_0p005.triggered.connect(set_0p005)
    sel_0p001 = QtWidgets.QAction("p<0.001", win)
    sel_0p001.triggered.connect(set_0p001)
    popMenu.addAction(sel_nomask)
    numspecial = 0

    def on_context_menu(point):
        # show context menu
        popMenu.exec_(ui.setMask_Button.mapToGlobal(point))

    ui.setMask_Button.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    ui.setMask_Button.customContextMenuRequested.connect(on_context_menu)

    # wire up the regressor selection radio buttons
    regressorbuttons = [
        ui.prefilt_radioButton,
        ui.postfilt_radioButton,
        ui.pass1_radioButton,
        ui.pass2_radioButton,
        ui.pass3_radioButton,
        ui.pass4_radioButton,
    ]

    ui.prefilt_radioButton.clicked.connect(prefilt_radioButton_clicked)
    ui.postfilt_radioButton.clicked.connect(postfilt_radioButton_clicked)
    ui.pass1_radioButton.clicked.connect(pass1_radioButton_clicked)
    ui.pass2_radioButton.clicked.connect(pass2_radioButton_clicked)
    ui.pass3_radioButton.clicked.connect(pass3_radioButton_clicked)
    ui.pass4_radioButton.clicked.connect(pass4_radioButton_clicked)

    for thebutton in regressorbuttons:
        thebutton.setDisabled(True)
        thebutton.hide()

    # read in all the datasets
    thesubjects = []

    thesubjects.append(
        RapidtideDataset(
            "main",
            datafileroot,
            anatname=anatname,
            geommaskname=geommaskname,
            userise=userise,
            usecorrout=usecorrout,
            useatlas=useatlas,
            forcetr=forcetr,
            forceoffset=forceoffset,
            offsettime=offsettime,
            verbose=args.verbose,
        )
    )
    currentdataset = thesubjects[-1]
    print("loading datasets...")

    regressors = currentdataset.getregressors()
    overlays = currentdataset.getoverlays()
    try:
        test = overlays["corrout"].display_state
    except KeyError:
        usecorrout = False

    # activate the appropriate regressor radio buttons
    if "prefilt" in regressors.keys():
        ui.prefilt_radioButton.setDisabled(False)
        ui.prefilt_radioButton.show()
    if "postfilt" in regressors.keys():
        ui.postfilt_radioButton.setDisabled(False)
        ui.postfilt_radioButton.show()
    if "pass1" in regressors.keys():
        ui.pass1_radioButton.setDisabled(False)
        ui.pass1_radioButton.show()
    if "pass2" in regressors.keys():
        ui.pass2_radioButton.setDisabled(False)
        ui.pass2_radioButton.show()
    if "pass3" in regressors.keys():
        ui.pass3_radioButton.setDisabled(False)
        ui.pass3_radioButton.setText("Pass " + regressors["pass3"].label[4:])
        ui.pass3_radioButton.show()
    if "pass4" in regressors.keys():
        ui.pass4_radioButton.setDisabled(False)
        ui.pass4_radioButton.setText("Pass " + regressors["pass4"].label[4:])
        ui.pass4_radioButton.show()

    win.setWindowTitle("TiDePool - " + datafileroot[:-1])

    # read in the significance distribution
    if os.path.isfile(datafileroot + "sigfit.txt"):
        sighistfitname = datafileroot + "sigfit.txt"
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
    lg_imgsize = 256.0
    sm_imgsize = 32.0
    xfov = currentdataset.xdim * currentdataset.xsize
    yfov = currentdataset.ydim * currentdataset.ysize
    zfov = currentdataset.zdim * currentdataset.zsize
    maxfov = np.max([xfov, yfov, zfov])
    scalefacx = (lg_imgsize / maxfov) * currentdataset.xsize
    scalefacy = (lg_imgsize / maxfov) * currentdataset.ysize
    scalefacz = (lg_imgsize / maxfov) * currentdataset.zsize

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

    if args.verbose > 0:
        for theoverlay in overlays:
            overlays[theoverlay].summarize()

    def loadpane(themap, thepane, view, button, panemap, orthoimages, bgmap=None):
        if themap.display_state:
            if bgmap is None:
                orthoimages[themap.name] = OrthoImageItem(
                    themap,
                    view[thepane],
                    view[thepane],
                    view[thepane],
                    button=button[thepane],
                    imgsize=sm_imgsize,
                    verbose=verbosity,
                )
            else:
                orthoimages[themap.name] = OrthoImageItem(
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

    if verbosity > 1:
        print("focusmap is:", currentdataset.focusmap, "bgmap is:", bgmap)
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
    mainwin.updated.connect(updateXYZFromMainWin)

    orthoimages = {}
    if verbosity > 1:
        print("loading panes")
    availablepanes = len(overlayGraphicsViews)
    numnotloaded = 0
    numloaded = 0
    for idx, themap in enumerate(currentdataset.dispmaps):
        if overlays[themap].display_state:
            if numloaded > availablepanes - 1:
                if verbosity > 1:
                    print("skipping map ", themap, "(", idx, "): out of display panes")
                numnotloaded += 1
            else:
                if verbosity > 1:
                    print("loading map ", themap, "(", idx, ") into pane ", numloaded)
                if bgmap is None:
                    loadpane(
                        overlays[themap],
                        numloaded,
                        overlayGraphicsViews,
                        overlaybuttons,
                        panetomap,
                        orthoimages,
                    )
                else:
                    loadpane(
                        overlays[themap],
                        numloaded,
                        overlayGraphicsViews,
                        overlaybuttons,
                        panetomap,
                        orthoimages,
                        bgmap=overlays[bgmap],
                    )
                numloaded += 1
        else:
            if verbosity > 1:
                print("not loading map ", themap, "(", idx, "): display_state is False")
    if verbosity > 1:
        print("done loading panes")
    if numnotloaded > 0:
        print("WARNING:", numnotloaded, "maps could not be loaded - not enough panes")

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
    global simfunc_ax, simfuncCurve, simfuncfitCurve, simfuncTLine, simfuncPeakMarker, simfuncCurvePoint, simfuncCaption, simfuncFitter

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
    else:
        simfunc_ax = None

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
        popMenu.addSeparator()
        if "lagmask" in currentdataset.loadedfuncmasks:
            popMenu.addAction(sel_lagmask)
            numspecial += 1
        if "refinemask" in currentdataset.loadedfuncmasks:
            popMenu.addAction(sel_refinemask)
            numspecial += 1
        if "meanmask" in currentdataset.loadedfuncmasks:
            popMenu.addAction(sel_meanmask)
            numspecial += 1
        if "preselectmask" in currentdataset.loadedfuncmasks:
            popMenu.addAction(sel_preselectmask)
            numspecial += 1
        if numspecial > 0:
            popMenu.addSeparator()
        if "p_lt_0p050_mask" in currentdataset.loadedfuncmasks:
            popMenu.addAction(sel_0p05)
        if "p_lt_0p010_mask" in currentdataset.loadedfuncmasks:
            popMenu.addAction(sel_0p01)
        if "p_lt_0p005_mask" in currentdataset.loadedfuncmasks:
            popMenu.addAction(sel_0p005)
        if "p_lt_0p001_mask" in currentdataset.loadedfuncmasks:
            popMenu.addAction(sel_0p001)

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
    overlay_radioButton_01_clicked(True)

    # have to do this after the windows are created
    imageadj.sigGradientChanged.connect(updateLUT)

    updateUI(callingfunc="main thread", orthoimages=True, focusvals=True)
    updateRegressor()

    QtWidgets.QApplication.instance().exec_()
