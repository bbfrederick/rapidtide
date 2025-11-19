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
from argparse import Namespace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from nibabel.affines import apply_affine
from numpy.typing import NDArray
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import rapidtide.util as tide_util
from rapidtide.Colortables import *
from rapidtide.OrthoImageItem import OrthoImageItem
from rapidtide.RapidtideDataset import RapidtideDataset, check_rt_spatialmatch
from rapidtide.simFuncClasses import SimilarityFunctionFitter
from rapidtide.workflows.atlasaverage import summarizevoxels

try:
    from PySide6.QtCore import __version__
except ImportError:
    try:
        from PyQt6.QtCore import QT_VERSION_STR
    except ImportError:
        pyqtbinding = "pyqt5"
    else:
        pyqtbinding = "pyqt6"
else:
    pyqtbinding = "pyside6"
print(f"using {pyqtbinding=}")

os.environ["QT_MAC_WANTS_LAYER"] = "1"

thesubjects: List[RapidtideDataset] = []
currentdataset: RapidtideDataset = None
whichsubject: int = 0
datafileroots: List[str] = []
verbosity: int = 0
defaultdict: dict = {}
overlays: dict = None
timeaxis: NDArray = None
averagingmode: str = None
atlasaveragingdone: bool = False


def _get_parser() -> Any:
    """
    Create and configure an argument parser for the tidepool program.

    This function constructs an `argparse.ArgumentParser` object with a set of
    predefined command-line arguments used to control the behavior of the tidepool
    time delay analysis visualization tool. The parser is configured with a
    descriptive help text and default values for various options.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all supported command-line options.

    Notes
    -----
    The parser is designed to be used in a larger application workflow where
    command-line arguments are parsed and used to configure the visualization
    and processing of time delay analysis results.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['--risetime', '--uistyle', 'big'])
    >>> print(args.risetime)
    True
    >>> print(args.uistyle)
    'big'
    """
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


def keyPressed(evt: Any) -> None:
    """
    Handle key press events for dataset navigation.

    This function processes keyboard input to navigate through different datasets
    in a subjects list. It supports navigation using arrow keys and shift modifier
    for special behavior.

    Parameters
    ----------
    evt : Any
        The key press event object containing key and modifier information.
        Expected to have methods `key()` and `modifiers()` similar to Qt events.

    Returns
    -------
    None
        This function does not return any value but modifies global variables
        `currentdataset`, `thesubjects`, and `whichsubject`.

    Notes
    -----
    - Up/Left arrow keys navigate to the previous dataset
    - Down/Right arrow keys navigate to the next dataset
    - Shift modifier reverses the navigation direction:
        * Shift + Up/Left: jumps to last dataset (index = numelements - 1)
        * Shift + Down/Right: jumps to first dataset (index = 0)
    - The function prints the current dataset information to console
    - Global variables modified: `currentdataset`, `thesubjects`, `whichsubject`

    Examples
    --------
    >>> keyPressed(event)
    Dataset set to subject_001 (0)

    >>> keyPressed(shift_event)
    Dataset set to subject_100 (99)
    """
    global currentdataset, thesubjects, whichsubject

    numelements = len(thesubjects)

    keymods = None
    if evt.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
        keymods = "shift"

    if (evt.key() == QtCore.Qt.Key.Key_Up) or (evt.key() == QtCore.Qt.Key.Key_Left):
        if keymods == "shift":
            whichsubject = numelements - 1
        else:
            whichsubject = (whichsubject - 1) % numelements
        selectDataset(whichsubject)
        print(f"Dataset set to {currentdataset.fileroot[:-1]} ({whichsubject})")
    elif (evt.key() == QtCore.Qt.Key.Key_Down) or (evt.key() == QtCore.Qt.Key.Key_Right):
        if keymods == "shift":
            whichsubject = 0
        else:
            whichsubject = (whichsubject + 1) % numelements
        selectDataset(whichsubject)
        print(f"Dataset set to {currentdataset.fileroot[:-1]} ({whichsubject})")
    else:
        print(evt.key())


class KeyPressWindow(QtWidgets.QMainWindow):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the object by calling the parent class constructor.

        This method serves as a constructor that delegates initialization to the
        parent class, allowing for proper inheritance chain execution.

        Parameters
        ----------
        *args : tuple
            Variable length argument list passed to the parent class constructor.
        **kwargs : dict
            Arbitrary keyword arguments passed to the parent class constructor.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This is a standard constructor implementation that ensures proper
        initialization of the parent class when subclassing.

        Examples
        --------
        >>> class ChildClass(ParentClass):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...
        >>> obj = ChildClass(param1, param2, kwarg1=value1)
        """
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev: Any) -> None:
        """
        Handle key press events and emit signal.

        This method is called when a key press event occurs. It emits the
        ``sigKeyPress`` signal with the event object as parameter.

        Parameters
        ----------
        ev : Any
            The key press event object containing event information.

        Returns
        -------
        None
            This method does not return a value.

        Notes
        -----
        This method is typically used in GUI applications to handle keyboard input
        and propagate key press events to connected signal handlers.

        Examples
        --------
        >>> def handle_key_press(event):
        ...     print(f"Key pressed: {event.key()}")
        ...
        >>> widget.sigKeyPress.connect(handle_key_press)
        >>> # When a key is pressed, the signal will be emitted
        """
        self.sigKeyPress.emit(ev)


def addDataset(
    thisdatafileroot: Any,
    anatname: Optional[Any] = None,
    geommaskname: Optional[Any] = None,
    userise: bool = False,
    usecorrout: bool = True,
    useatlas: bool = False,
    forcetr: bool = False,
    forceoffset: bool = False,
    offsettime: float = 0.0,
    ignoredimmatch: bool = False,
) -> None:
    """
    Load and add a dataset to the global list of subjects for processing.

    This function initializes a `RapidtideDataset` object from the specified data root
    and appends it to the global list `thesubjects`. It performs dimension matching
    checks against previously loaded datasets unless `ignoredimmatch` is set to `True`.

    Parameters
    ----------
    thisdatafileroot : Any
        Root path to the dataset to be loaded.
    anatname : Optional[Any], optional
        Anatomical image filename, by default None.
    geommaskname : Optional[Any], optional
        Geometric mask filename, by default None.
    userise : bool, optional
        Whether to use RISE (respiratory-induced signal enhancement), by default False.
    usecorrout : bool, optional
        Whether to use correlation output, by default True.
    useatlas : bool, optional
        Whether to use atlas-based processing, by default False.
    forcetr : bool, optional
        Whether to force TR (repetition time) correction, by default False.
    forceoffset : bool, optional
        Whether to force offset correction, by default False.
    offsettime : float, optional
        Time offset to apply, by default 0.0.
    ignoredimmatch : bool, optional
        If True, skip dimension matching checks, by default False.

    Returns
    -------
    None
        This function does not return a value but modifies global variables.

    Notes
    -----
    - The function modifies global variables: `currentdataset`, `thesubjects`, `whichsubject`, `datafileroots`.
    - If a dataset is not dimensionally compatible with the first loaded dataset and `ignoredimmatch` is False,
      the dataset is skipped.
    - Prints information about the loaded dataset and the list of all loaded subjects.

    Examples
    --------
    >>> addDataset('/path/to/dataset1')
    Loading /path/to/dataset1
    subject 0: /path/to/dataset1

    >>> addDataset('/path/to/dataset2', ignoredimmatch=True)
    Loading /path/to/dataset2
    subject 0: /path/to/dataset1
    subject 1: /path/to/dataset2
    """
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


def updateFileMenu() -> None:
    """
    Update the file menu with subject options based on current subjects list.

    This function updates the file menu by removing existing subject actions and
    rebuilding it with current subjects from the global `thesubjects` list. Each
    subject is displayed with a checkmark indicator for the currently selected
    subject (`whichsubject`).

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function relies on global variables:
    - `thesubjects`: List of subject objects with `fileroot` attribute
    - `whichsubject`: Index of currently selected subject
    - `fileMenu`: QMenu object to be updated
    - `sel_files`: List of QAction objects representing menu items
    - `pyqtbinding`: String indicating the PyQt binding being used
    - `win`: Main window object for QAction parent

    The function handles different PyQt bindings (PyQt5, PyQt6, PySide6) by
    selecting the appropriate QAction constructor based on the binding.

    Examples
    --------
    >>> updateFileMenu()
    # Updates the file menu with current subjects and their selection status

    See Also
    --------
    selectDataset : Function called when a subject menu item is triggered
    """
    global thesubjects, whichsubject
    global fileMenu, sel_open
    global sel_files

    if pyqtbinding == "pyqt5":
        qactionfunc = QtWidgets.QAction
    elif pyqtbinding == "pyqt6":
        qactionfunc = QtGui.QAction
    elif pyqtbinding == "pyside6":
        qactionfunc = QtGui.QAction
    else:
        print("unsupported")

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


def datasetPicker() -> None:
    """
    Open a file dialog to select a lag time file and initialize the dataset.

    This function opens a Qt file dialog to select a lag time file in BIDS format,
    extracts the data file root path, adds the dataset to the global dataset list,
    selects the newly added dataset, and updates the file menu.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value but modifies global variables.

    Notes
    -----
    The function supports PyQt5, PyQt6, and PySide6 bindings. It expects lag time
    files in the format *_lagtimes.nii.gz or *_desc-maxtime_map.nii.gz.

    Global variables modified:
    - currentdataset: Current dataset identifier
    - thesubjects: List of subject identifiers
    - whichsubject: Index of currently selected subject
    - datafileroots: List of data file roots
    - ui: User interface object
    - win: Window object
    - overlagGraphicsViews: Graphics views for overlap visualization
    - verbosity: Verbosity level for logging

    Examples
    --------
    >>> datasetPicker()
    # Opens file dialog and initializes dataset selection
    """
    global currentdataset, thesubjects, whichsubject, datafileroots
    global ui, win, defaultdict, overlagGraphicsViews
    global verbosity

    mydialog = QtWidgets.QFileDialog()
    if pyqtbinding == "pyqt5":
        options = mydialog.Options()
    elif pyqtbinding == "pyqt6":
        options = mydialog.options()
    elif pyqtbinding == "pyside6":
        options = mydialog.options()
    else:
        print("unsupported")
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


def selectDataset(thesubject: Any) -> None:
    """
    Select a dataset for viewing and activate it in the user interface.

    This function updates the current dataset selection and activates the specified
    dataset in the user interface. It handles focus updates for regressor and map
    components if the UI has been initialized, and triggers necessary updates
    to the file menu.

    Parameters
    ----------
    thesubject : Any
        The subject identifier or reference to select as the current dataset.
        This parameter determines which dataset from the global `thesubjects`
        collection will be activated.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables including:
    - `currentdataset`: Set to the selected subject
    - `whichsubject`: Set to the input parameter `thesubject`
    - Global UI-related variables: `ui`, `win`, `defaultdict`, `overlayGraphicsViews`

    The function assumes that global variables `thesubjects`, `uiinitialized`,
    `verbosity`, and other required globals are properly initialized before calling.

    Examples
    --------
    >>> selectDataset('subject_001')
    >>> selectDataset(42)
    """
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
        xpos: Any,
        ypos: Any,
        zpos: Any,
        tpos: Any,
        xdim: Any,
        ydim: Any,
        zdim: Any,
        tdim: Any,
        toffset: Any,
        tr: Any,
        affine: Any,
        XPosSpinBox: Optional[Any] = None,
        YPosSpinBox: Optional[Any] = None,
        ZPosSpinBox: Optional[Any] = None,
        TPosSpinBox: Optional[Any] = None,
        XCoordSpinBox: Optional[Any] = None,
        YCoordSpinBox: Optional[Any] = None,
        ZCoordSpinBox: Optional[Any] = None,
        TCoordSpinBox: Optional[Any] = None,
        TimeSlider: Optional[Any] = None,
        runMovieButton: Optional[Any] = None,
    ) -> None:
        """
        Initialize the widget with position and dimension information.

        This constructor sets up the widget's internal state by initializing
        position, dimension, and time-related attributes. It also connects
        the movie timer to update the display during animation.

        Parameters
        ----------
        xpos : Any
            X-axis position value.
        ypos : Any
            Y-axis position value.
        zpos : Any
            Z-axis position value.
        tpos : Any
            Time position value.
        xdim : Any
            X-axis dimension.
        ydim : Any
            Y-axis dimension.
        zdim : Any
            Z-axis dimension.
        tdim : Any
            Time dimension.
        toffset : Any
            Time offset value.
        tr : Any
            Repetition time (TR) value.
        affine : Any
            Affine transformation matrix.
        XPosSpinBox : Optional[Any], optional
            Spin box for X position, by default None
        YPosSpinBox : Optional[Any], optional
            Spin box for Y position, by default None
        ZPosSpinBox : Optional[Any], optional
            Spin box for Z position, by default None
        TPosSpinBox : Optional[Any], optional
            Spin box for T position, by default None
        XCoordSpinBox : Optional[Any], optional
            Spin box for X coordinate, by default None
        YCoordSpinBox : Optional[Any], optional
            Spin box for Y coordinate, by default None
        ZCoordSpinBox : Optional[Any], optional
            Spin box for Z coordinate, by default None
        TCoordSpinBox : Optional[Any], optional
            Spin box for T coordinate, by default None
        TimeSlider : Optional[Any], optional
            Time slider widget, by default None
        runMovieButton : Optional[Any], optional
            Button to run the movie, by default None

        Returns
        -------
        None
            This method does not return a value.

        Notes
        -----
        The widget initializes internal state variables including `frametime`,
        `movierunning`, and connects the `movieTimer.timeout` signal to the
        `updateMovie` method for animation control.

        Examples
        --------
        >>> widget = MyClass(xpos=10, ypos=20, zpos=30, tpos=40,
        ...                  xdim=100, ydim=100, zdim=100, tdim=10,
        ...                  toffset=0, tr=2.5, affine=np.eye(4))
        """
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

    def setXYZInfo(self, xdim: Any, ydim: Any, zdim: Any, affine: Any) -> None:
        """
        Set voxel and coordinate information based on dimensions and affine transformation.

        This function initializes voxel dimensions (`xdim`, `ydim`, `zdim`) and an affine
        transformation matrix (`affine`). It computes the inverse of the affine matrix,
        converts voxel coordinates to real-world coordinates, and sets up spin boxes
        for position and coordinate controls.

        Parameters
        ----------
        xdim : Any
            The number of voxels along the x-axis.
        ydim : Any
            The number of voxels along the y-axis.
        zdim : Any
            The number of voxels along the z-axis.
        affine : Any
            The 4x4 affine transformation matrix mapping voxel coordinates to real-world
            coordinates.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        - The function assumes the existence of helper methods such as `vox2real`, `setupSpinBox`,
          `getXpos`, `getYpos`, `getZpos`, `getXcoord`, `getYcoord`, and `getZcoord`.
        - Spin boxes for voxel positions (`XPosSpinBox`, `YPosSpinBox`, `ZPosSpinBox`) and
          coordinate positions (`XCoordSpinBox`, `YCoordSpinBox`, `ZCoordSpinBox`) are initialized
          with appropriate ranges and step sizes.

        Examples
        --------
        >>> setXYZInfo(64, 64, 32, affine_matrix)
        # Initializes voxel and coordinate settings based on given dimensions and affine.
        """
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

    def setTInfo(self, tdim: Any, tr: Any, toffset: Any) -> None:
        """
        Set time information and configure time-related UI elements.

        This function initializes time coordinates and configures spin boxes, sliders,
        and movie controls based on the provided time dimensions and transformation
        parameters.

        Parameters
        ----------
        tdim : Any
            Time dimension parameter used for setting up UI elements and coordinate
            calculations.
        tr : Any
            Time transformation parameter used for coordinate transformations.
        toffset : Any
            Time offset parameter used for setting up time coordinates.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The function performs the following operations:
        1. Sets instance variables tdim, toffset, and tr
        2. Converts time position to real coordinates using tr2real method
        3. Configures TPosSpinBox with appropriate range and initial value
        4. Calculates time coordinate bounds and configures TCoordSpinBox
        5. Sets up TimeSlider with time position range
        6. Configures runMovieButton for movie playback control

        Examples
        --------
        >>> setTInfo(100, 0.1, 0.0)
        >>> # Configures time information for 100 time steps with transformation 0.1
        """
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

    def setupRunMovieButton(self, thebutton: Any, thehandler: Any) -> None:
        """
        Set up the movie button with appropriate text and click handler.

        This function configures a button to serve as a movie control button by
        setting its text to "Start Movie" and connecting it to the provided handler.
        The button initialization is conditional based on verbosity level.

        Parameters
        ----------
        thebutton : Any
            The button widget to be configured. Expected to have setText() and
            clicked.connect() methods.
        thehandler : Any
            The callback function or method to be connected to the button's clicked signal.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The button configuration only occurs if `thebutton` is not None. When
        verbosity level is greater than 1, a message is printed to indicate
        the button initialization.

        Examples
        --------
        >>> setupRunMovieButton(button_widget, movie_handler)
        >>> # Button text set to "Start Movie" and connected to handler
        """
        global verbosity

        if thebutton is not None:
            if verbosity > 1:
                print("initializing movie button")
            thebutton.setText("Start Movie")
            thebutton.clicked.connect(thehandler)

    def setupTimeSlider(
        self, theslider: Any, thehandler: Any, minval: Any, maxval: Any, currentval: Any
    ) -> None:
        """
        Set up a time slider with specified parameters and connect event handler.

        This function configures a slider widget for time selection by setting its range,
        step size, and connecting a value change handler.

        Parameters
        ----------
        theslider : Any
            The slider widget to be configured. Expected to have methods setRange(),
            setSingleStep(), and valueChanged.connect().
        thehandler : Any
            The event handler function to be connected to the slider's valueChanged signal.
        minval : Any
            The minimum value for the slider range.
        maxval : Any
            The maximum value for the slider range.
        currentval : Any
            The current value of the slider (not used in this implementation).

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The function only configures the slider if it is not None. The slider's single
        step is set to 1, and the valueChanged signal is connected to the provided handler.

        Examples
        --------
        >>> slider = QSlider()
        >>> def handler(value):
        ...     print(f"Slider value: {value}")
        >>> setupTimeSlider(self, slider, handler, 0, 100, 50)
        """
        if theslider is not None:
            theslider.setRange(minval, maxval)
            theslider.setSingleStep(1)
            theslider.valueChanged.connect(thehandler)

    def setupSpinBox(
        self,
        thespinbox: Any,
        thehandler: Any,
        minval: Any,
        maxval: Any,
        stepsize: Any,
        currentval: Any,
    ) -> None:
        """
        Configure a spin box widget with specified parameters and connect its signal.

        This function sets up a QSpinBox widget with the provided range, step size, and
        initial value. It also configures wrapping behavior and connects the valueChanged
        signal to the specified handler function.

        Parameters
        ----------
        thespinbox : Any
            The spin box widget to be configured. Expected to be a QSpinBox or compatible
            widget with setRange, setSingleStep, setValue, setWrapping, and valueChanged
            methods.
        thehandler : Any
            The callback function to be connected to the spin box's valueChanged signal.
            This function will be called whenever the spin box value changes.
        minval : Any
            The minimum value allowed in the spin box.
        maxval : Any
            The maximum value allowed in the spin box.
        stepsize : Any
            The step size for incrementing/decrementing the spin box value.
        currentval : Any
            The initial value to set in the spin box.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The function performs no validation on input parameters. It assumes that the
        spin box widget has the required methods and that the handler is callable.
        The spin box is configured with wrapping enabled and keyboard tracking disabled.

        Examples
        --------
        >>> setupSpinBox(spinbox, on_value_changed, 0, 100, 1, 50)
        >>> # Configures spinbox to range from 0 to 100 with step size 1,
        >>> # initial value 50, wrapping enabled, and keyboard tracking disabled.
        """
        if thespinbox is not None:
            thespinbox.setRange(minval, maxval)
            thespinbox.setSingleStep(stepsize)
            thespinbox.setValue(currentval)
            thespinbox.setWrapping(True)
            thespinbox.setKeyboardTracking(False)
            thespinbox.valueChanged.connect(thehandler)

    def updateXYZValues(self, emitsignal: bool = True) -> None:
        """
        Update the values of XYZ spinbox widgets with current coordinate values.

        This method updates the displayed values in the XYZ position and coordinate spinbox
        widgets with the current coordinate values stored in the instance. It can optionally
        emit a signal to notify other components of the update.

        Parameters
        ----------
        emitsignal : bool, optional
            If True (default), emits the updatedXYZ signal after updating the spinbox values.
            If False, updates the spinbox values without emitting the signal.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method checks if each spinbox widget exists (is not None) before attempting
        to set its value. This prevents errors when widgets are not initialized or
        have been deleted.

        Examples
        --------
        >>> updateXYZValues()
        # Updates all XYZ spinbox values and emits signal

        >>> updateXYZValues(emitsignal=False)
        # Updates all XYZ spinbox values without emitting signal
        """
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

    def updateTValues(self) -> None:
        """
        Update T values and ranges for spinboxes and slider based on current time coordinates.

        This method updates the state of T-related UI elements (spinboxes and slider)
        based on the current time position and coordinate values. It ensures that the
        UI elements reflect the correct current values and valid ranges for time
        positioning and coordinate display.

        Parameters
        ----------
        self : object
            The instance of the class containing the method, which should have the
            following attributes:
            - TPosSpinBox: QSpinBox for time position
            - TCoordSpinBox: QSpinBox for time coordinate
            - TimeSlider: QSlider for time positioning
            - tpos: Current time position value
            - tcoord: Current time coordinate value
            - tdim: Total time dimension
            - tr2real: Method to convert time coordinates to real values

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method checks if each UI element exists before attempting to update it.
        For TCoordSpinBox, the range is calculated based on the minimum and maximum
        real values of the time coordinate range.

        Examples
        --------
        >>> updateTValues()
        # Updates all T-related UI elements with current time values and ranges

        See Also
        --------
        emit : Emits the updatedT signal after updating UI elements
        """
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

    def real2tr(self, time: Any) -> int:
        """
        Convert real time to trigger time.

        Parameters
        ----------
        time : Any
            The real time value to be converted to trigger time.

        Returns
        -------
        int
            The converted trigger time value.

        Notes
        -----
        This function performs a linear transformation from real time to trigger time
        using the formula: tr = round((time - toffset) / tr), where toffset and tr
        are attributes of the class instance.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.toffset = 10.5
        >>> obj.tr = 0.1
        >>> obj.real2tr(11.0)
        45
        """
        return int(np.round((time - self.toffset) / self.tr, 0))

    def tr2real(self, tpos: Any) -> float:
        """
        Convert time position to real time.

        Parameters
        ----------
        tpos : Any
            Time position to be converted to real time. Type can vary depending on
            the specific implementation context.

        Returns
        -------
        None
            This method returns the result of the calculation: `self.toffset + self.tr * tpos`
            but does not explicitly return a value in the function body.

        Notes
        -----
        This function performs a linear transformation from time position to real time
        using the formula: real_time = toffset + tr * tpos

        Examples
        --------
        >>> obj.tr2real(5)
        # Returns: self.toffset + self.tr * 5
        """
        return self.toffset + self.tr * tpos

    def real2vox(self, xcoord: Any, ycoord: Any, zcoord: Any) -> Tuple[int, int, int]:
        """
        Convert real coordinates to voxel coordinates using inverse affine transformation.

        This function transforms continuous real-world coordinates into discrete voxel coordinates
        by applying the inverse affine transformation matrix. The result is rounded to the nearest
        integer values to obtain valid voxel indices.

        Parameters
        ----------
        xcoord : Any
            X-coordinate in real-world space
        ycoord : Any
            Y-coordinate in real-world space
        zcoord : Any
            Z-coordinate in real-world space

        Returns
        -------
        tuple of int
            Tuple containing (x, y, z) voxel coordinates as integers

        Notes
        -----
        The transformation uses the inverse affine matrix stored in ``self.invaffine``.
        Coordinates are rounded to the nearest integer using ``np.round``.

        Examples
        --------
        >>> voxel_coords = img.real2vox(10.7, 15.2, 20.8)
        >>> print(voxel_coords)
        (11, 15, 21)
        """
        x, y, z = apply_affine(self.invaffine, [xcoord, ycoord, zcoord])
        return int(np.round(x, 0)), int(np.round(y, 0)), int(np.round(z, 0))

    def vox2real(self, xpos: Any, ypos: Any, zpos: Any) -> NDArray:
        """
        Convert voxel coordinates to real-world coordinates using the affine transformation.

        This function applies the affine transformation matrix to convert voxel coordinates
        to real-world (scanner) coordinates. The affine matrix contains the necessary
        scaling, rotation, and translation information to map voxel indices to physical space.

        Parameters
        ----------
        xpos : Any
            X coordinate in voxel space
        ypos : Any
            Y coordinate in voxel space
        zpos : Any
            Z coordinate in voxel space

        Returns
        -------
        None
            The function returns the result of apply_affine function which contains
            the real-world coordinates corresponding to the input voxel coordinates

        Notes
        -----
        The affine transformation matrix is stored in ``self.affine`` and contains
        the spatial transformation parameters needed to map voxel coordinates to
        scanner coordinates. This is commonly used in neuroimaging applications
        where voxel indices need to be converted to real-world coordinates for
        anatomical localization.

        Examples
        --------
        >>> # Assuming self.affine is properly initialized
        >>> real_coords = vox2real(10, 20, 30)
        >>> print(real_coords)
        [x_real, y_real, z_real]
        """
        return apply_affine(self.affine, [xpos, ypos, zpos])

    def setXYZpos(self, xpos: Any, ypos: Any, zpos: Any, emitsignal: bool = True) -> None:
        """
        Set the XYZ position coordinates and update corresponding values.

        Parameters
        ----------
        xpos : Any
            X coordinate position value
        ypos : Any
            Y coordinate position value
        zpos : Any
            Z coordinate position value
        emitsignal : bool, default=True
            Whether to emit a signal after updating coordinates

        Returns
        -------
        None
            This method does not return any value

        Notes
        -----
        This method updates the internal position coordinates and converts voxel
        coordinates to real-world coordinates using the `vox2real` method. The
        `updateXYZValues` method is called to propagate the changes.

        Examples
        --------
        >>> obj.setXYZpos(10, 20, 30)
        >>> obj.setXYZpos(10, 20, 30, emitsignal=False)
        """
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
        self.updateXYZValues(emitsignal=emitsignal)

    def setXYZcoord(self, xcoord: Any, ycoord: Any, zcoord: Any, emitsignal: bool = True) -> None:
        """
        Set the XYZ coordinates and update corresponding voxel positions.

        This method assigns the provided XYZ coordinates to the object's coordinate attributes
        and converts them to voxel coordinates using the real2vox transformation. It then
        updates the XYZ values and optionally emits a signal.

        Parameters
        ----------
        xcoord : Any
            The x-coordinate value to be set
        ycoord : Any
            The y-coordinate value to be set
        zcoord : Any
            The z-coordinate value to be set
        emitsignal : bool, optional
            Whether to emit a signal after updating coordinates (default is True)

        Returns
        -------
        None
            This method does not return any value

        Notes
        -----
        The method internally calls `real2vox` to convert real coordinates to voxel coordinates
        and `updateXYZValues` to propagate the changes. The `emitsignal` parameter controls
        whether the signal emission is skipped for performance reasons when multiple updates
        are needed.

        Examples
        --------
        >>> obj.setXYZcoord(10.5, 20.3, 5.0)
        >>> obj.setXYZcoord(10.5, 20.3, 5.0, emitsignal=False)
        """
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.zcoord = zcoord
        self.xpos, self.ypos, self.zpos = self.real2vox(self.xcoord, self.ycoord, self.zcoord)
        self.updateXYZValues(emitsignal=emitsignal)

    def setTpos(self, tpos: Any) -> None:
        """
        Set the temporal position and update related coordinates and values.

        This method updates the temporal position (`tpos`) of the object, ensuring it
        does not exceed the maximum temporal dimension (`tdim`). When the position
        changes, the corresponding real coordinates are calculated and the temporal
        values are updated accordingly.

        Parameters
        ----------
        tpos : Any
            The temporal position to set. If greater than `self.tdim`, it will be
            capped at `self.tdim`.

        Returns
        -------
        None
            This method modifies the object in-place and does not return any value.

        Notes
        -----
        The method performs an equality check before updating to avoid unnecessary
        computations when the position remains unchanged.

        Examples
        --------
        >>> obj.setTpos(5)
        >>> obj.tpos
        5

        >>> obj.setTpos(10)
        >>> obj.tpos
        5  # capped at tdim = 5
        """
        if tpos > self.tdim:
            tpos = self.tdim
        if self.tpos != tpos:
            self.tpos = tpos
            self.tcoord = self.tr2real(self.tpos)
            self.updateTValues()

    def setTcoord(self, tcoord: Any) -> None:
        """
        Set the t-coordinate and update related values.

        Parameters
        ----------
        tcoord : Any
            The new t-coordinate value to be set. Type can vary depending on the implementation.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        If the provided t-coordinate differs from the current one, this method will:
        1. Update the internal `tcoord` attribute
        2. Convert the new t-coordinate to tr-coordinates using `real2tr` method
        3. Update all related t-values by calling `updateTValues()`

        Examples
        --------
        >>> obj.setTcoord(5.0)
        >>> obj.tcoord
        5.0
        >>> obj.tpos
        [converted_tr_coordinates]
        """
        if self.tcoord != tcoord:
            self.tcoord = tcoord
            self.tpos = self.real2tr(self.tcoord)
            self.updateTValues()

    def getXpos(self, event: Any) -> None:
        """
        Update the X coordinate position and related values based on spin box input.

        This method retrieves the current value from the X position spin box, compares
        it with the stored X position, and updates the position if changed. When updated,
        it converts the voxel coordinates to real-world coordinates and triggers an
        update of all coordinate values.

        Parameters
        ----------
        event : Any
            Event object passed to the method (typically from GUI interaction).
            This parameter is not used within the method implementation.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method performs coordinate conversion using the `vox2real` method and
        triggers an update of all coordinate values through `updateXYZValues` with
        `emitsignal=True` to propagate changes to connected components.

        Examples
        --------
        >>> getXpos(event)
        # Updates X position and related coordinates when spin box value changes
        """
        # print('entering getXpos')
        newx = int(self.XPosSpinBox.value())
        if self.xpos != newx:
            self.xpos = newx
            self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
            self.updateXYZValues(emitsignal=True)

    def getYpos(self, event: Any) -> None:
        """
        Update Y position coordinate and related values.

        This method retrieves the current Y position from the spin box widget,
        updates the internal Y position attribute if it has changed, and then
        recalculates the corresponding real-world coordinates using the voxel-to-real
        transformation. Finally, it updates the XYZ coordinate values and emits a
        signal to notify other components of the change.

        Parameters
        ----------
        event : Any
            Event object passed by the GUI framework when the spin box value changes.
            This parameter is typically not used within the method but is required
            for compatibility with Qt's signal-slot mechanism.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method performs a comparison between the current Y position and the
        new value from the spin box. If they differ, the method updates the internal
        state and triggers coordinate recalculation. The voxel-to-real transformation
        is handled by the `vox2real` method which converts voxel coordinates to
        real-world coordinates.

        Examples
        --------
        >>> getYpos(event)
        # Updates Y position and related coordinates when spin box value changes
        """
        # print('entering getYpos')
        newy = int(self.YPosSpinBox.value())
        if self.ypos != newy:
            self.ypos = newy
            self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
            self.updateXYZValues(emitsignal=True)

    def getZpos(self, event: Any) -> None:
        """
        Update Z position and corresponding coordinates based on spin box value.

        This method retrieves the current Z position value from the Z position spin box,
        updates the internal zpos attribute if the value has changed, and calculates
        the corresponding real-world coordinates using the vox2real transformation.
        It then updates the XYZ coordinate display values.

        Parameters
        ----------
        event : Any
            Event object passed to the method (typically from GUI interaction).
            The specific content of this parameter is not used within the method.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method only updates coordinates and emits signals when the Z position
        value actually changes from the current stored value.

        Examples
        --------
        >>> getZpos(event=None)
        # Updates zpos and corresponding coordinates if value changed
        """
        # print('entering getZpos')
        newz = int(self.ZPosSpinBox.value())
        if self.zpos != newz:
            self.zpos = newz
            self.xcoord, self.ycoord, self.zcoord = self.vox2real(self.xpos, self.ypos, self.zpos)
            self.updateXYZValues(emitsignal=True)

    def getTpos(self, event: Any) -> None:
        """
        Update time position and related properties based on spin box value.

        This method retrieves the current value from the time position spin box,
        updates the internal time position tracking, converts it to real coordinates,
        and manages the movie timer if the movie is running.

        Parameters
        ----------
        event : Any
            Event object passed to the method (typically from GUI interaction).
            The specific content of this parameter depends on the calling context.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        - If the new time position differs from the current one, the method updates
          both `tpos` and `tcoord` attributes
        - When the movie is running, the movie timer is stopped and restarted with
          the new time position value
        - The `updateTValues()` method is called to refresh all time-related displays

        Examples
        --------
        >>> getTpos(event)
        # Updates time position and related properties based on spin box value
        """
        # print('entering getTpos')
        newt = int(self.TPosSpinBox.value())
        if self.tpos != newt:
            self.tpos = newt
            self.tcoord = self.tr2real(self.tpos)
            if self.movierunning:
                self.movieTimer.stop()
                self.movieTimer.start(int(self.tpos))
            self.updateTValues()

    def getXcoord(self, event: Any) -> None:
        """
        Update the X coordinate value and corresponding voxel coordinates.

        This method retrieves the current value from the X coordinate spin box and updates
        the internal X coordinate value if it has changed. When the coordinate changes,
        the corresponding voxel coordinates are recalculated and the XYZ values are updated.

        Parameters
        ----------
        event : Any
            The event object associated with the coordinate change. This parameter is
            typically provided by Qt event handling mechanisms but is not used within
            the method implementation.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method triggers an update of voxel coordinates using the `real2vox` conversion
        function and emits a signal to update the XYZ display values when the coordinate
        changes.

        Examples
        --------
        >>> getXcoord(event)
        # Updates X coordinate and corresponding voxel values
        """
        newxcoord = self.XCoordSpinBox.value()
        if self.xcoord != newxcoord:
            self.xcoord = newxcoord
            self.xpos, self.ypos, self.zpos = self.real2vox(self.xcoord, self.ycoord, self.zcoord)
            self.updateXYZValues(emitsignal=True)

    def getYcoord(self, event: Any) -> None:
        """
        Update Y coordinate and related voxel coordinates when Y coordinate spin box value changes.

        This method is typically called when the Y coordinate spin box widget value is changed.
        It updates the internal Y coordinate value and recalculates the corresponding voxel coordinates
        using the real-to-voxel conversion function. The XYZ values are then updated and signal is emitted
        to notify other components of the change.

        Parameters
        ----------
        event : Any
            Event object triggered by the spin box value change. This parameter is typically
            provided by Qt's signal-slot mechanism but is not used within the method.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method only updates the coordinates and emits signals when the new Y coordinate
        value differs from the current stored value. This prevents unnecessary updates when
        the spin box value hasn't actually changed.

        Examples
        --------
        >>> getYcoord(event)
        # Updates self.ycoord and recalculates voxel coordinates when spin box value changes
        """
        newycoord = self.YCoordSpinBox.value()
        if self.ycoord != newycoord:
            self.ycoord = newycoord
            self.xpos, self.ypos, self.zpos = self.real2vox(self.xcoord, self.ycoord, self.zcoord)
            self.updateXYZValues(emitsignal=True)

    def getZcoord(self, event: Any) -> None:
        """
        Update Z coordinate and related position values.

        This method retrieves the current Z coordinate value from the spin box widget,
        updates the internal Z coordinate attribute if it has changed, and then
        converts the real coordinates to voxel coordinates. Finally, it updates the
        XYZ value displays and emits a signal to notify other components of the change.

        Parameters
        ----------
        event : Any
            The event that triggered this method call. Typically a Qt event object
            or None if called programmatically.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method performs coordinate conversion using the `real2vox` method and
        updates the internal position attributes (`xpos`, `ypos`, `zpos`). It also
        emits a signal through `updateXYZValues` to notify other components of the
        coordinate change.

        Examples
        --------
        >>> getZcoord(event=None)
        # Updates Z coordinate and related position values
        """
        newzcoord = self.ZCoordSpinBox.value()
        if self.zcoord != newzcoord:
            self.zcoord = newzcoord
            self.xpos, self.ypos, self.zpos = self.real2vox(self.xcoord, self.ycoord, self.zcoord)
            self.updateXYZValues(emitsignal=True)

    def getTcoord(self, event: Any) -> None:
        """
        Update time coordinate values based on spin box changes.

        This method retrieves the current value from the TCoordSpinBox widget and
        updates the internal time coordinate tracking when a change is detected.
        It also calculates the corresponding real position and updates related values.

        Parameters
        ----------
        event : Any
            Event object passed to the method (typically from GUI interactions).
            The specific content of this parameter depends on the calling context.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method only updates internal state when the new coordinate value differs
        from the current stored value. This prevents unnecessary recalculations.

        Examples
        --------
        >>> getTcoord(event)
        # Updates internal tcoord, tpos, and related values when spin box value changes
        """
        newtcoord = self.TCoordSpinBox.value()
        if self.tcoord != newtcoord:
            self.tcoord = newtcoord
            self.tpos = self.real2tr(self.tcoord)
            self.updateTValues()

    def getTimeSlider(self) -> None:
        """
        Retrieve current time slider value and update related coordinates.

        This method reads the current value from the time slider widget, converts
        it to real-time coordinates using the tr2real transformation function,
        and updates the time-related values in the system.

        Returns
        -------
        None
            This method does not return any value but updates instance attributes
            self.tpos, self.tcoord, and triggers self.updateTValues().

        Notes
        -----
        The method assumes that self.TimeSlider is a valid slider widget with a
        value() method, and that self.tr2real is a defined transformation function.
        The updateTValues() method is called after coordinate calculation to
        synchronize dependent time values.

        Examples
        --------
        >>> getTimeSlider()
        # Updates self.tpos, self.tcoord, and calls self.updateTValues()
        """
        self.tpos = self.TimeSlider.value()
        self.tcoord = self.tr2real(self.tpos)
        self.updateTValues()

    def updateMovie(self) -> None:
        """
        Update the movie position and related values.

        This method advances the movie position by one step, wrapping around to the
        beginning when the end is reached. It also updates the temporal values and
        optionally prints debug information based on verbosity level.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method uses modulo arithmetic to cycle through temporal positions.
        When verbosity level is greater than 1, debug information including
        current position (tpos) and T coordinate value is printed to console.

        Examples
        --------
        >>> updateMovie()
        # Advances movie position and updates related values
        """
        # self.tpos = (self.tpos + 1) % self.tdim
        self.setTpos((self.tpos + 1) % self.tdim)
        self.updateTValues()
        if verbosity > 1:
            print(f"Tpos, t: {self.tpos}, {self.TCoordSpinBox.value()}")

    def stopMovie(self) -> None:
        """
        Stop the movie playback and reset the UI state.

        This method stops the currently running movie by setting the movie running flag to False,
        stopping the movie timer, and updating the button text to "Start Movie" if the button exists.

        Notes
        -----
        This method assumes that the class has the following attributes:
        - movierunning: boolean flag indicating if movie is currently running
        - movieTimer: timer object with a stop() method
        - runMovieButton: button widget with setText() method

        Examples
        --------
        >>> player = MoviePlayer()
        >>> player.startMovie()
        >>> player.stopMovie()
        >>> player.movierunning
        False
        """
        self.movierunning = False
        self.movieTimer.stop()
        if self.runMovieButton is not None:
            self.runMovieButton.setText("Start Movie")

    def startMovie(self) -> None:
        """
        Start movie playback and update UI controls.

        This method initiates movie playback by setting the movie running flag to True,
        updating the button text to "Stop Movie" if the button exists, and starting
        the movie timer with the specified frame time.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method assumes that `self.movierunning`, `self.runMovieButton`, and
        `self.movieTimer` attributes are properly initialized before calling this method.
        The `self.frametime` attribute should contain the frame time in milliseconds.

        Examples
        --------
        >>> movie_player.startMovie()
        >>> # Movie playback started, button text changed to "Stop Movie"
        """
        self.movierunning = True
        if self.runMovieButton is not None:
            self.runMovieButton.setText("Stop Movie")
        self.movieTimer.start(int(self.frametime))

    def runMovieToggle(self, event: Any) -> None:
        """
        Toggle movie playback state.

        Toggles the movie playback state between running and stopped. If the movie is currently
        running, it will be stopped. If the movie is stopped, it will be started.

        Parameters
        ----------
        event : Any
            The event that triggered the toggle action. Type and content may vary depending
            on the event source (e.g., button click, keyboard shortcut).

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This function uses a global `verbosity` variable to control debug output. When
        verbosity level is greater than 1, informational messages are printed to the console
        indicating the current state and action being performed.

        Examples
        --------
        >>> # Assuming self.movierunning is False
        >>> runMovieToggle(event=None)
        entering runMovieToggle
        movie is not running - turning on
        leaving runMovieToggle
        >>> # Assuming self.movierunning is True
        >>> runMovieToggle(event=None)
        entering runMovieToggle
        movie is running - turning off
        leaving runMovieToggle
        """
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

    def setFrametime(self, frametime: Any) -> None:
        """
        Set the frame time for the movie timer.

        Parameters
        ----------
        frametime : Any
            The frame time to set. Typically an integer or float representing
            the time interval in milliseconds between frames.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        If the movie is currently running, this method will stop the movie timer,
        update the frame time, and then restart the timer with the new frame time.
        If the movie is not running, only the frame time will be updated.

        Examples
        --------
        >>> player.setFrametime(100)
        >>> player.setFrametime(50.5)
        """
        if self.movierunning:
            self.movieTimer.stop()
        self.frametime = frametime
        if self.movierunning:
            self.movieTimer.start(int(self.frametime))


def logStatus(thetextbox: Any, thetext: Any) -> None:
    """
    Log text to a text box and scroll to the bottom.

    This function inserts text into a text box widget and ensures the scroll bar
    is positioned at the bottom to show the newly added text. The behavior differs
    slightly depending on the PyQt binding being used.

    Parameters
    ----------
    thetextbox : Any
        A text box widget (typically QTextEdit or QPlainTextEdit) where the text
        will be inserted.
    thetext : Any
        The text string to be inserted into the text box.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    - For PyQt5, the cursor is moved to the end before inserting text
    - For PyQt6 and PySide6, no special cursor handling is performed
    - The text is always appended with a newline character
    - The scroll bar is automatically set to the maximum value to show the latest text

    Examples
    --------
    >>> logStatus(textbox_widget, "Processing completed")
    >>> logStatus(textbox_widget, "Error: File not found")
    """
    if pyqtbinding == "pyqt5":
        thetextbox.moveCursor(QtGui.QTextCursor.End)
    elif pyqtbinding == "pyqt6":
        pass
    elif pyqtbinding == "pyside6":
        pass
    else:
        print("unsupported")
    thetextbox.insertPlainText(thetext + "\n")
    sb = thetextbox.verticalScrollBar()
    sb.setValue(sb.maximum())


def getMinDispLimit() -> None:
    """
    Update the minimum display limit for the current focus map from the UI spin box.

    This function retrieves the current value from the display minimum double spin box
    in the user interface and sets it as the minimum display limit for the currently
    focused map in the overlays dictionary. It then calls updateDispLimits() to refresh
    the display limits.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables: ui, overlays, and currentdataset.
    The function assumes that ui.dispmin_doubleSpinBox exists and has a value() method.
    The function assumes that overlays and currentdataset are properly initialized
    global variables with the expected structure.

    Examples
    --------
    >>> getMinDispLimit()
    # Updates overlays[currentdataset.focusmap].dispmin with ui.dispmin_doubleSpinBox.value()
    # and calls updateDispLimits()
    """
    global ui, overlays, currentdataset
    overlays[currentdataset.focusmap].dispmin = ui.dispmin_doubleSpinBox.value()
    updateDispLimits()


def getMaxDispLimit() -> None:
    """
    Update the maximum display limit for the current focus map.

    This function retrieves the current value from the display maximum spin box
    in the user interface and sets it as the maximum display limit for the
    currently focused map in the overlays dictionary. It then calls the
    updateDispLimits function to refresh the display settings.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables: ui, overlays, and currentdataset.
    The function assumes that ui.dispmax_doubleSpinBox exists and has a value() method.
    The function assumes that overlays and currentdataset are properly initialized
    global variables with the expected structure.

    Examples
    --------
    >>> getMaxDispLimit()
    # Updates the display maximum limit for the current focus map
    """
    global ui, overlays, currentdataset
    overlays[currentdataset.focusmap].dispmax = ui.dispmax_doubleSpinBox.value()
    updateDispLimits()


def updateDispLimits() -> None:
    """
    Update display limits for the current dataset's focus map.

    This function configures the minimum and maximum display range spin boxes
    based on the current dataset's focus map values. It sets the range, single
    step size, and current values for both the minimum and maximum display
    spin boxes, then triggers an UI update.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function modifies global variables:
    - ui: User interface object containing display spin boxes
    - overlays: Data structure containing overlay information
    - currentdataset: Current dataset object with focusmap attribute

    The single step size is calculated as 1/100th of the range between
    minimum and maximum values of the current focus map.

    Examples
    --------
    >>> updateDispLimits()
    # Updates the display limits UI elements based on current dataset
    """
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


def resetDispLimits() -> None:
    """
    Reset display limits for the current focus map to their minimum and maximum values.

    This function resets the display minimum and maximum values of the currently
    focused map in the overlays dictionary to match the actual minimum and maximum
    values of that map. After resetting the limits, it updates the display.

    Notes
    -----
    This function modifies global variables `overlays` and `currentdataset`.
    The `dispmin` and `dispmax` attributes of the focus map are updated in-place.

    Examples
    --------
    >>> resetDispLimits()
    # Resets display limits for the current focus map
    """
    global overlays, currentdataset
    overlays[currentdataset.focusmap].dispmin = overlays[currentdataset.focusmap].minval
    overlays[currentdataset.focusmap].dispmax = overlays[currentdataset.focusmap].maxval
    updateDispLimits()


def resetDispSmart() -> None:
    """
    Reset display limits to robust minimum and maximum values for the current focus map.

    This function resets the display minimum and maximum values of the currently
    focused map to their robust counterparts, effectively restoring default display
    limits. It then updates the display limits in the interface.

    Notes
    -----
    This function modifies global variables `overlays` and `currentdataset`.
    The reset operation affects only the currently focused map as determined
    by `currentdataset.focusmap`.

    Examples
    --------
    >>> resetDispSmart()
    # Resets display limits for the current focus map to robust values
    """
    global overlays, currentdataset
    overlays[currentdataset.focusmap].dispmin = overlays[currentdataset.focusmap].robustmin
    overlays[currentdataset.focusmap].dispmax = overlays[currentdataset.focusmap].robustmax
    updateDispLimits()


def updateSimilarityFunc() -> None:
    """
    Update the similarity function plot based on the current dataset and location.

    This function updates various plot elements (curves, lines, markers, captions)
    to reflect the similarity function data at the current spatial and temporal location.
    It handles both correlation and mutual information similarity metrics, and displays
    fitting results or failure reasons accordingly.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value. It modifies global plot elements in place.

    Notes
    -----
    The function relies on several global variables:
    - `overlays`: Dictionary containing data arrays such as 'corrout', 'gaussout', 'lagtimes', etc.
    - `timeaxis`: Array of time values used for plotting.
    - `currentloc`: Object with attributes `xpos`, `ypos`, `zpos`, and `tpos` indicating current location.
    - `currentdataset`: Object with attribute `similaritymetric` indicating the metric used.
    - `simfunc_ax`: The matplotlib axes object for the similarity function plot.
    - `simfuncCurve`, `simfuncfitCurve`, `simfuncTLine`, `simfuncPeakMarker`, `simfuncCurvePoint`,
      `simfuncCaption`, `simfuncFitter`: Plot elements managed by the function.
    - `verbosity`: Controls the level of printed output for debugging.

    Examples
    --------
    Assuming all global variables are properly initialized, calling this function will update
    the similarity function plot to show the current data:

    >>> updateSimilarityFunc()
    """
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


def updateHistogram() -> None:
    """
    Update the histogram display with current dataset statistics.

    This function plots the histogram data for the currently focused dataset,
    adds statistical reference lines, and updates the display limits.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function modifies global variables:
    - hist_ax: The histogram axis object to be updated
    - overlays: Container holding histogram data and statistics
    - currentdataset: Current dataset object with focusmap attribute

    The histogram includes:
    - Green reference lines at 2nd, 25th, 50th, 75th, and 98th percentiles
    - Blue brush fill for the histogram area
    - Automatic y-axis scaling based on maximum histogram value

    Examples
    --------
    >>> updateHistogram()
    # Updates the histogram display with current dataset data

    See Also
    --------
    updateDispLimits : Function called after histogram update to refresh display
    """
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
        """
        Initialize a graphics object with specified position, size, and color.

        Parameters
        ----------
        topLeft : tuple of int
            The top-left corner coordinates (x, y) of the graphics object.
        size : tuple of int
            The dimensions (width, height) of the graphics object.
        color : tuple of int, optional
            The RGBA color values (r, g, b, a) for the graphics object.
            Default is (0, 128, 0, 128) which represents green with 50% opacity.

        Returns
        -------
        None
            This method initializes the object and does not return any value.

        Notes
        -----
        This constructor calls the parent class initializer from pg.GraphicsObject
        and generates the initial picture representation of the graphics object.

        Examples
        --------
        >>> obj = GraphicsObject((10, 20), (100, 50))
        >>> obj = GraphicsObject((0, 0), (800, 600), (255, 0, 0, 255))
        """
        pg.GraphicsObject.__init__(self)
        self.topLeft = topLeft
        self.size = size
        self.color = color
        self.generatePicture()

    def generatePicture(self) -> None:
        """
        Generate a picture representation of the object using Qt graphics primitives.

        This method creates a Qt picture object containing a rectangle shape based on
        the object's color, top-left position, and size properties.

        Returns
        -------
        None
            This method does not return any value but stores the generated picture
            in the instance variable `self.picture`.

        Notes
        -----
        The generated picture is created using Qt's QPainter and QPicture classes.
        The rectangle is drawn with both pen and brush set to the object's color,
        and the rectangle is positioned according to `self.topLeft` and sized according
        to `self.size`.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.color = 'red'
        >>> obj.topLeft = [10, 20]
        >>> obj.size = [100, 50]
        >>> obj.generatePicture()
        >>> # Picture is now stored in obj.picture
        """
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen(self.color))
        p.setBrush(pg.mkBrush(self.color))
        tl = QtCore.QPointF(self.topLeft[0], self.topLeft[1])
        size = QtCore.QSizeF(self.size[0], self.size[1])
        p.drawRect(QtCore.QRectF(tl, size))
        p.end()

    def paint(self, p: Any, *args: Any) -> None:
        """
        Draw the picture using the provided painter object.

        This method renders the stored picture onto the given painter object
        at position (0, 0).

        Parameters
        ----------
        p : Any
            The painter object that will be used to draw the picture.
            This object must have a `drawPicture` method compatible with the
            expected interface.
        *args : Any
            Additional arguments that may be passed to the drawing operation.
            These are forwarded directly to the underlying drawing method.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The picture is drawn at coordinates (0, 0) relative to the painter's
        coordinate system. The actual drawing behavior depends on the
        implementation of the `drawPicture` method in the provided painter object.

        Examples
        --------
        >>> painter = SomePainter()
        >>> obj.paint(painter)
        >>> obj.paint(painter, additional_param)
        """
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self) -> QtCore.QRectF:
        """
        Return the bounding rectangle of the picture.

        Returns
        -------
        QtCore.QRectF
            The bounding rectangle of the picture as a QRectF object.

        Notes
        -----
        This method retrieves the bounding rectangle that encompasses the entire picture
        content. The returned rectangle defines the minimal area that contains all
        graphical elements of the picture.

        Examples
        --------
        >>> rect = picture.boundingRect()
        >>> print(rect)
        QRectF(0, 0, 100, 100)
        """
        return QtCore.QRectF(self.picture.boundingRect())


def updateRegressor() -> None:
    """
    Update the regressor plot with data from the currently focused regressor.

    This function retrieves the currently focused regressor from the dataset and
    plots its time data on the global `regressor_ax` axis. It also adds a text
    label with kurtosis statistics and a rectangle highlighting the calculation
    limits defined in the dataset.

    Notes
    -----
    The function modifies global variables: `regressor_ax`, `liveplots`, and
    `currentdataset`. It assumes that `currentdataset` has a `focusregressor`
    attribute and a `getregressors()` method returning a dictionary of regressors.

    Examples
    --------
    Assuming `currentdataset` is properly initialized and contains regressor data,
    calling this function will update the plot on `regressor_ax` with the current
    regressor's data and statistics.

    See Also
    --------
    currentdataset : The dataset object containing regressor information.
    regressor_ax : The PyQtGraph axis used for plotting.
    """
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


def updateRegressorSpectrum() -> None:
    """
    Update the regressor spectrum plot with the currently focused regressor data.

    This function retrieves the regressor spectrum data for the currently focused
    regressor and updates the plot in the global `regressorspectrum_ax` axis. It
    plots the spectrum data as a line with filled area under the curve, sets the
    x-axis range based on the samplerate, and adjusts the y-axis range to fit the
    data with a 25% padding. A rectangle is added to highlight the filter limits
    of the regressor.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value. It modifies the global plot axis
        `regressorspectrum_ax` in place.

    Notes
    -----
    The function relies on global variables:
    - `regressorspectrum_ax`: The PyQtGraph axis object for plotting
    - `liveplots`: Global variable controlling live plotting behavior
    - `currentdataset`: Global dataset object containing regressor data

    The function uses the following attributes from `currentdataset`:
    - `focusregressor`: Index of the currently focused regressor
    - `regressorfilterlimits`: Tuple of (lower, upper) filter limits
    - `getregressors()`: Method returning the regressor data

    Examples
    --------
    >>> updateRegressorSpectrum()
    # Updates the regressor spectrum plot with the current focus regressor data
    """
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


def calcAtlasStats() -> None:
    """
    Calculate statistical measures for each region in the atlas across functional maps.

    This function performs atlas-based averaging of functional data, computing statistics
    such as mean, median, standard deviation, median absolute deviation (MAD), and coefficient
    of variation (CoV) for each region in the atlas. It also saves the results to CSV files
    and updates the overlay dictionary with new statistical maps.

    The function requires the presence of an atlas map in the `overlays` dictionary and
    assumes that `currentdataset` contains the necessary data including functional maps
    and atlas labels. The results are saved with filenames based on the dataset's root
    and atlas name.

    Notes
    -----
    This function modifies global variables:
        - `atlasstats`: Dictionary storing computed statistics for each map and region.
        - `atlasaveragingdone`: Boolean flag indicating completion of averaging.
        - `overlays`: Updated with new statistical map overlays.

    Examples
    --------
    >>> calcAtlasStats()
    Performing atlas averaging...
    Calculating stats for region 1 ( Region1 )
    ...
    Done performing atlas averaging
    """
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


def updateAtlasStats() -> None:
    """
    Update atlas statistics for all loaded functional maps.

    This function updates the atlas statistics data for all functional maps
    in the current dataset. It iterates through loaded functional maps and
    assigns atlas statistics values based on the current averaging mode.

    Notes
    -----
    The function modifies global variables: overlays, atlasstats, averagingmode,
    and currentdataset. It only processes maps when an "atlas" overlay exists
    and averagingmode is not None.

    Examples
    --------
    >>> updateAtlasStats()
    in updateAtlasStats
    loading mean into map1_atlasstat
    loading mean into map2_atlasstat
    """
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


def doAtlasAveraging(state: Any) -> None:
    """
    Toggle atlas averaging functionality based on the provided state.

    This function controls whether atlas averaging is enabled or disabled.
    When enabled, it sets a global flag and updates ortho images. When disabled,
    it clears the flag and updates ortho images accordingly.

    Parameters
    ----------
    state : Any
        The state indicating whether atlas averaging should be enabled.
        Typically a QtCore.Qt.CheckState value (Checked or Unchecked).

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies the global variable `atlasaveragingdone` and calls
    `updateOrthoImages()` to refresh the ortho images display.

    Examples
    --------
    >>> doAtlasAveraging(QtCore.Qt.CheckState.Checked)
    in doAtlasAveraging
    atlas averaging is turned on

    >>> doAtlasAveraging(QtCore.Qt.CheckState.Unchecked)
    in doAtlasAveraging
    atlas averaging is turned off
    """
    global atlasaveragingdone
    print("in doAtlasAveraging")
    if state == QtCore.Qt.CheckState.Checked:
        atlasaveragingdone = True
        print("atlas averaging is turned on")
    else:
        atlasaveragingdone = False
        print("atlas averaging is turned off")
    updateOrthoImages()


def updateAveragingMode() -> None:
    """
    Update the averaging mode for the current dataset and refresh display components.

    This function handles the updating of averaging mode settings, including:
    - Calculating atlas statistics when needed
    - Updating focus map paths with atlas statistics suffix
    - Resetting display limits and smart settings
    - Updating the user interface components

    The function checks if atlas overlays are present and if atlas averaging has been completed,
    then performs necessary calculations and updates. It also manages the focus map path
    based on the current averaging mode setting.

    Notes
    -----
    This function modifies global variables including:
    - averagingmode
    - focusmap
    - atlasaveragingdone
    - overlays
    - currentdataset

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value but modifies global state.

    Examples
    --------
    >>> updateAveragingMode()
    in updateAveragingMode
    averaging mode set to  None

    See Also
    --------
    calcAtlasStats : Calculate atlas statistics
    setAtlasMask : Set atlas mask based on current settings
    updateAtlasStats : Update atlas statistics
    resetDispLimits : Reset display limits
    resetDispSmart : Reset display smart settings
    updateUI : Update user interface components
    """
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


def raw_radioButton_clicked(enabled: Any) -> None:
    """
    Handle raw radio button click event.

    This function is triggered when the raw radio button is clicked. It sets the
    averaging mode to None and updates the averaging mode display.

    Parameters
    ----------
    enabled : Any
        Boolean value indicating whether the radio button is selected. When True,
        the function executes the averaging mode reset and update operations.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies the global variable `averagingmode` and calls
    `updateAveragingMode()` function. The function prints a debug message
    "in raw_radioButton_clicked" when executed.

    Examples
    --------
    >>> raw_radioButton_clicked(True)
    in raw_radioButton_clicked
    """
    global averagingmode
    if enabled:
        print("in raw_radioButton_clicked")
        averagingmode = None
        updateAveragingMode()


def mean_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the mean radio button.

    This function is triggered when the mean radio button is selected. It updates
    the global averaging mode to "mean" and calls the updateAveragingMode function
    to apply the change.

    Parameters
    ----------
    enabled : Any
        Boolean value indicating whether the radio button is selected. When True,
        the averaging mode is set to "mean".

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies the global variable `averagingmode` and calls
    `updateAveragingMode()` to propagate the change.

    Examples
    --------
    >>> mean_radioButton_clicked(True)
    in mean_radioButton_clicked
    >>> print(averagingmode)
    'mean'
    """
    global averagingmode
    if enabled:
        print("in mean_radioButton_clicked")
        averagingmode = "mean"
        updateAveragingMode()


def median_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the median radio button.

    This function is called when the median radio button is clicked. It updates
    the global averaging mode to "median" and triggers an update of the averaging
    mode in the application.

    Parameters
    ----------
    enabled : Any
        A boolean or boolean-like value indicating whether the radio button is selected.
        When True, the median averaging mode is activated.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies the global variable `averagingmode` and calls `updateAveragingMode()`.
    The function prints a debug message "in median_radioButton_clicked" when executed.

    Examples
    --------
    >>> median_radioButton_clicked(True)
    in median_radioButton_clicked
    >>> median_radioButton_clicked(False)
    # No output, but averagingmode remains unchanged
    """
    global averagingmode
    if enabled:
        print("in median_radioButton_clicked")
        averagingmode = "median"
        updateAveragingMode()


def CoV_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the CoV radio button.

    This function is triggered when the CoV radio button is clicked. It updates
    the global averaging mode to "CoV" and calls the updateAveragingMode function
    to refresh the display.

    Parameters
    ----------
    enabled : Any
        Boolean value indicating whether the radio button is selected. When True,
        the averaging mode is set to "CoV".

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies the global variable `averagingmode` and calls
    `updateAveragingMode()` to update the user interface.

    Examples
    --------
    >>> CoV_radioButton_clicked(True)
    in CoV_radioButton_clicked
    >>> print(averagingmode)
    'CoV'
    """
    global averagingmode
    if enabled:
        print("in CoV_radioButton_clicked")
        averagingmode = "CoV"
        updateAveragingMode()


def std_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the standard deviation radio button.

    This function is called when the standard deviation radio button is clicked.
    It updates the global averaging mode to "std" and triggers an update of the
    averaging mode in the application.

    Parameters
    ----------
    enabled : Any
        Boolean value indicating whether the radio button is selected. Typically
        True when the button is clicked and becomes active.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies the global variable `averagingmode` and calls
    `updateAveragingMode()` to propagate the change throughout the application.

    Examples
    --------
    >>> std_radioButton_clicked(True)
    in std_radioButton_clicked
    """
    global averagingmode
    if enabled:
        print("in std_radioButton_clicked")
        averagingmode = "std"
        updateAveragingMode()


def MAD_radioButton_clicked(enabled: Any) -> None:
    """
    Handle radio button click event for MAD (Median Absolute Deviation) averaging mode.

    This function is triggered when the MAD radio button is clicked. It updates the global
    averaging mode to "MAD" and calls the updateAveragingMode function to refresh the UI.

    Parameters
    ----------
    enabled : Any
        Boolean value indicating whether the radio button is selected. Typically True
        when the button is clicked and becomes active.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies the global variable `averagingmode` and calls `updateAveragingMode()`.
    The function prints a debug message "in MAD_radioButton_clicked" when executed.

    Examples
    --------
    >>> MAD_radioButton_clicked(True)
    in MAD_radioButton_clicked
    """
    global averagingmode
    if enabled:
        print("in MAD_radioButton_clicked")
        averagingmode = "MAD"
        updateAveragingMode()


def transparencyCheckboxClicked() -> None:
    """
    Handle transparency checkbox click event to update layer transparency settings.

    This function is triggered when the transparency checkbox in the user interface
    is clicked. It updates the endalpha value based on the checkbox state and
    applies the new transparency settings to all loaded function maps in the current dataset.

    The function modifies global variables and updates the visualization state
    of all overlays in the current dataset.

    Notes
    -----
    This function assumes that the global variables ui, overlays, currentdataset,
    LUT_alpha, LUT_endalpha, and verbosity are properly initialized before calling.

    Examples
    --------
    >>> transparencyCheckboxClicked()
    # Updates transparency settings for all overlays in current dataset
    """
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


def gray_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the gray radio button in the image visualization interface.

    This function updates the lookup table (LUT) of the current focus map to a grayscale
    representation when the gray radio button is selected. It restores the previous LUT
    state and updates the visualization accordingly.

    Parameters
    ----------
    enabled : Any
        Boolean value indicating whether the gray radio button is selected. When True,
        the focus map is updated to display in grayscale.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables including `imageadj`, `overlays`, `currentdataset`,
    `LUT_alpha`, and `LUT_endalpha`. The function assumes that `gen_gray_state()` is a
    predefined function that generates the grayscale lookup table state.

    Examples
    --------
    >>> gray_radioButton_clicked(True)
    # Sets the current focus map to grayscale display

    >>> gray_radioButton_clicked(False)
    # Does not change the display (assuming other radio buttons handle this case)
    """
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


def thermal_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the thermal radio button.

    This function is called when the thermal radio button is clicked. It updates
    the lookup table (LUT) for the current focus map overlay when the thermal
    visualization is enabled, and restores the previous LUT state.

    Parameters
    ----------
    enabled : Any
        A boolean or boolean-like value indicating whether the thermal
        visualization is enabled. When True, the thermal LUT is applied to
        the current overlay.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables including `imageadj`, `overlays`,
    `currentdataset`, `LUT_alpha`, and `LUT_endalpha`. The function assumes
    that `gen_thermal_state()` returns a valid LUT state and that the necessary
    overlay and dataset objects exist.

    Examples
    --------
    >>> thermal_radioButton_clicked(True)
    # Applies thermal LUT to current overlay and updates display

    >>> thermal_radioButton_clicked(False)
    # No action taken if enabled is False
    """
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


def plasma_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the plasma radio button in the visualization interface.

    This function updates the lookup table (LUT) of the current focus map overlay
    when the plasma radio button is selected. It applies a plasma color scheme
    and restores the previous LUT state for proper visualization.

    Parameters
    ----------
    enabled : Any
        Boolean or equivalent value indicating whether the plasma radio button is selected.
        When True, the plasma LUT is applied to the current overlay.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables:
    - imageadj: Image adjustment object
    - overlays: Dictionary of overlay objects
    - currentdataset: Current dataset object
    - LUT_alpha: Alpha value for LUT
    - LUT_endalpha: End alpha value for LUT

    The function relies on the global `gen_plasma_state()` function to generate
    the plasma color mapping state.

    Examples
    --------
    >>> plasma_radioButton_clicked(True)
    # Applies plasma LUT to current overlay and updates visualization

    >>> plasma_radioButton_clicked(False)
    # No action taken (function only processes when enabled is True)
    """
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


def viridis_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the viridis colormap radio button.

    This function updates the colormap of the current focus map to viridis when
    the radio button is enabled. It restores the lookup table state and updates
    the color mapping accordingly.

    Parameters
    ----------
    enabled : Any
        Flag indicating whether the viridis radio button is selected. When True,
        the viridis colormap is applied to the current focus map.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables including imageadj, overlays,
    currentdataset, LUT_alpha, and LUT_endalpha. The function assumes that
    the necessary global state has been properly initialized before calling.

    Examples
    --------
    >>> viridis_radioButton_clicked(True)
    # Applies viridis colormap to current focus map

    >>> viridis_radioButton_clicked(False)
    # No action taken (function does nothing when enabled is False)
    """
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


def turbo_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the turbo radio button in the visualization interface.

    This function updates the lookup table (LUT) for the current focus map when the turbo
    radio button is enabled. It applies the turbo color scheme with specified alpha values
    and restores the previous LUT state for proper visualization continuity.

    Parameters
    ----------
    enabled : Any
        Boolean or equivalent value indicating whether the turbo radio button is selected.
        When True, the turbo LUT is applied to the current focus map.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables:
    - imageadj: Image adjustment object
    - overlays: Dictionary of overlay objects
    - currentdataset: Current dataset object
    - LUT_alpha: Alpha value for LUT
    - LUT_endalpha: End alpha value for LUT

    The function restores the LUT state from the saved state and updates the visualization
    through the updateLUT() function call.

    Examples
    --------
    >>> turbo_radioButton_clicked(True)
    # Applies turbo LUT to current focus map and updates visualization

    >>> turbo_radioButton_clicked(False)
    # No action taken (function only processes when enabled is True)
    """
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


def rainbow_radioButton_clicked(enabled: Any) -> None:
    """
    Handle the click event for the rainbow radio button in the GUI.

    This function is triggered when the rainbow radio button is selected. It updates
    the lookup table (LUT) of the current focus map overlay with a spectral color
    gradient and restores the previous LUT state.

    Parameters
    ----------
    enabled : Any
        Boolean or equivalent value indicating whether the rainbow mode is enabled.
        When True, the function applies the spectral LUT to the current overlay.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables including:
    - imageadj: Image adjustment object
    - overlays: Dictionary of overlay objects
    - currentdataset: Current dataset object
    - LUT_alpha: Alpha value for LUT
    - LUT_endalpha: End alpha value for LUT

    The function calls `gen_spectrum_state()` to generate the spectral color
    gradient and `updateLUT()` to refresh the display.

    Examples
    --------
    >>> rainbow_radioButton_clicked(True)
    # Applies rainbow color mapping to current overlay

    >>> rainbow_radioButton_clicked(False)
    # No effect (function only acts when enabled is True)
    """
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


def setMask(maskname: Any) -> None:
    """
    Set the functional mask for the current dataset and update associated UI elements.

    This function configures the functional mask based on the provided mask name,
    updates the user interface, and triggers re-computation of averaging modes.
    It supports various predefined mask types including 'nomask', 'meanmask', 'lagmask',
    'brainmask', 'refinemask', 'preselectmask', and p-value-based masks.

    Parameters
    ----------
    maskname : Any
        Name of the mask to be applied. Supported values include:
        - 'nomask': Disables functional mask
        - 'meanmask': Mean regressor seed mask
        - 'lagmask': Uses valid fit points as functional mask
        - 'brainmask': Externally provided brain mask
        - 'refinemask': Voxel refinement mask
        - 'preselectmask': Preselected mean regressor seed mask
        - 'p_lt_X_mask': P-value threshold masks (e.g., 'p_lt_0p05_mask' for p < 0.05)

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function modifies global variables:
    - `overlays`: Updates mask data for each functional map.
    - `loadedfuncmaps`: Iterates over loaded functional maps to apply the mask.
    - `ui`: Updates the mask button text in the user interface.
    - `atlasaveragingdone`: Resets to False to trigger re-computation.
    - `currentdataset`: Sets the functional mask name in the dataset.

    Examples
    --------
    >>> setMask("meanmask")
    Mean regressor seed mask
    >>> setMask("p_lt_0p01_mask")
    Setting functional mask to p<0.01
    """
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


def setAtlasMask() -> None:
    """
    Set functional mask using all defined atlas regions.

    This function activates the atlas mask for all loaded functional maps in the current dataset.
    It updates the UI to reflect that a valid mask has been applied and applies the atlas mask
    data to all functional maps in the overlays dictionary.

    Notes
    -----
    This function modifies global variables: overlays, loadedfuncmaps, ui, and currentdataset.
    The function calls updateUI after applying the mask to refresh the visualization.

    See Also
    --------
    updateUI : Updates the user interface after mask application.
    overlays : Global dictionary containing all loaded map objects.
    currentdataset : Global object containing current dataset information.

    Examples
    --------
    >>> setAtlasMask()
    Using all defined atlas regions as functional mask
    """
    global overlays, loadedfuncmaps, ui, currentdataset
    print("Using all defined atlas regions as functional mask")
    ui.setMask_Button.setText("Valid mask")
    for themap in currentdataset.loadedfuncmaps:
        overlays[themap].setFuncMask(overlays["atlasmask"].data)
    updateUI(callingfunc="setAtlasMask", orthoimages=True, histogram=True)


def overlay_radioButton_clicked(which: Any, enabled: Any) -> None:
    """
    Handle the click event of an overlay radio button.

    This function is triggered when a user clicks on an overlay radio button. It updates
    the current dataset's focus map, adjusts display limits, and synchronizes the UI
    with the selected overlay settings. It also manages the visibility and state of
    related UI elements such as color lookup table (LUT) radio buttons and time position
    information.

    Parameters
    ----------
    which : Any
        Index or identifier of the clicked radio button.
    enabled : Any
        Boolean or equivalent indicating whether the radio button is selected.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function modifies global variables including `imageadj`, `overlays`, `currentdataset`,
    `panetomap`, `ui`, `overlaybuttons`, `currentloc`, `atlasaveragingdone`, and `verbosity`.
    The function assumes that `currentdataset`, `currentloc`, and `overlays` are properly initialized
    and contain valid data structures.

    Examples
    --------
    >>> overlay_radioButton_clicked(0, True)
    # Sets the focus map to the first overlay and updates UI accordingly.
    """
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


def updateTimepoint(event: Any) -> None:
    """
    Update the current timepoint based on mouse event coordinates.

    This function handles mouse events to update the current timepoint in the
    visualization. It maps the mouse position to the time axis, validates the
    time value, updates the current location, and refreshes the UI components.

    Parameters
    ----------
    event : Any
        Mouse event containing position information. The event is expected to
        have a `pos()` method that returns the mouse coordinates.

    Returns
    -------
    None
        This function does not return any value but modifies global variables
        including `currentloc.tpos`, `ui.TimeSlider.value`, and triggers UI updates.

    Notes
    -----
    This function modifies global state variables including:
    - `currentloc.tpos`: Updates the current timepoint index
    - `ui.TimeSlider`: Updates the slider position
    - `data`, `overlays`, `simfunc_ax`, `tr`, `tpos`, `timeaxis`, `currentloc`, `ui`, `currentdataset`, `vLine`, `verbosity`

    The function uses `tide_util.valtoindex()` to convert time values to array indices
    and calls `updateUI()` to refresh the visualization components.

    Examples
    --------
    >>> # Assuming a mouse event is generated
    >>> updateTimepoint(mouse_event)
    >>> # Updates current timepoint and refreshes UI
    """
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


def updateLUT() -> None:
    """
    Update the lookup table for the current dataset and associated UI elements.

    This function updates the color lookup table for the currently focused dataset
    and synchronizes it with the colorbar display. If harvest colormaps are enabled,
    it also saves the current image adjustment state. The function triggers a UI update
    to reflect the changes in the orthographic images.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function relies on global variables:
    - img_colorbar: The colorbar object to update
    - overlays: Collection of overlay data structures
    - currentdataset: Current dataset object containing focusmap information
    - harvestcolormaps: Flag indicating whether to save colormap states

    The function updates the lookup table for the current focusmap and calls
    updateUI with specific parameters to refresh the user interface.

    Examples
    --------
    >>> updateLUT()
    # Updates the lookup table and refreshes the UI without returning any value
    """
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


def mapwithLUT(
    theimage: NDArray, themask: NDArray, theLUT: Any, dispmin: Any, dispmax: Any
) -> None:
    """
    Map image data using a lookup table with optional masking.

    This function applies a lookup table to map input image data to output values,
    with optional masking and scaling based on display range parameters.

    Parameters
    ----------
    theimage : NDArray
        Input image data to be mapped
    themask : NDArray
        Mask array where values less than 1 will be set to 0 in the output
    theLUT : array-like
        Lookup table array used for mapping
    dispmin : float
        Minimum display value used for scaling
    dispmax : float
        Maximum display value used for scaling

    Returns
    -------
    NDArray
        Mapped data with shape matching input image, where values are
        mapped using the lookup table and masked according to themask parameter

    Notes
    -----
    - The function performs linear scaling of input data to index the lookup table
    - Values outside the valid range are clipped to the lookup table bounds
    - The alpha channel (4th channel) of the output is set to 0 where mask values are less than 1

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([[0, 128, 255], [64, 192, 32]])
    >>> mask = np.array([[1, 1, 0], [1, 0, 1]])
    >>> lut = np.array([0, 50, 100, 150, 200, 255])
    >>> result = mapwithLUT(image, mask, lut, 0, 255)
    """
    offset = dispmin
    scale = len(theLUT) / (dispmax - dispmin)
    scaleddata = np.rint((theimage - offset) * scale).astype("int32")
    scaleddata[np.where(scaleddata < 0)] = 0
    scaleddata[np.where(scaleddata > (len(theLUT) - 1))] = len(theLUT) - 1
    mappeddata = theLUT[scaleddata]
    mappeddata[:, :, 3][np.where(themask < 1)] = 0
    return mappeddata


def updateTFromControls() -> None:
    """
    Update the time position from control values and refresh the UI.

    This function synchronizes the time position displayed in the main window
    with the current location's time position, then updates various UI components
    including orthoimages, similarity functions, and focus values.

    Notes
    -----
    This function modifies global variables `mainwin` and `currentloc`.
    The time position update does not emit a signal to prevent recursive updates.

    See Also
    --------
    updateUI : Function called to refresh UI components after time position update.

    Examples
    --------
    >>> updateTFromControls()
    # Updates main window time position and refreshes UI components
    """
    global mainwin, currentloc
    mainwin.setTpos(currentloc.tpos, emitsignal=False)
    updateUI(
        callingfunc="updateTFromControls",
        orthoimages=True,
        similarityfunc=True,
        focusvals=True,
    )


def updateXYZFromControls() -> None:
    """
    Update XYZ position coordinates from control inputs and refresh UI components.

    This function synchronizes the global current location coordinates with the main
    window's position display and triggers updates to various UI elements including
    orthographic images, similarity functions, and focus values.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function relies on global variables `mainwin` and `currentloc` which must be
    properly initialized before calling this function. The position update is performed
    without emitting signals to prevent recursive updates.

    Examples
    --------
    >>> updateXYZFromControls()
    # Updates the main window's XYZ position and refreshes related UI components
    """
    global mainwin, currentloc
    mainwin.setXYZpos(currentloc.xpos, currentloc.ypos, currentloc.zpos, emitsignal=False)
    updateUI(
        callingfunc="updateXYZFromControls",
        orthoimages=True,
        similarityfunc=True,
        focusvals=True,
    )


def updateXYZFromMainWin() -> None:
    """
    Update XYZ position from main window coordinates and refresh UI.

    This function synchronizes the current location's XYZ position with the
    coordinates stored in the main window, then updates the user interface
    to reflect these changes. The synchronization is performed without emitting
    a signal to prevent recursive updates.

    Notes
    -----
    This function relies on global variables `mainwin` and `currentloc`. The
    `mainwin` object must have `xpos`, `ypos`, and `zpos` attributes, while
    `currentloc` must have a `setXYZpos` method.

    Examples
    --------
    >>> updateXYZFromMainWin()
    # Updates current location with main window coordinates and refreshes UI
    """
    global mainwin, currentloc
    currentloc.setXYZpos(mainwin.xpos, mainwin.ypos, mainwin.zpos, emitsignal=False)
    updateUI(callingfunc="updateXYZFromMainWin", similarityfunc=True, focusvals=True)


def updateUI(
    orthoimages: bool = False,
    histogram: bool = False,
    LUT: bool = False,
    similarityfunc: bool = False,
    focusvals: bool = False,
    callingfunc: Optional[Any] = None,
    verbose: int = 0,
) -> None:
    """
    Update the user interface components based on specified flags.

    This function updates various UI elements including orthoimages, histogram,
    lookup table (LUT), similarity function, and focus values. The function
    supports verbose output for debugging purposes and can be called from
    different functions to track execution flow.

    Parameters
    ----------
    orthoimages : bool, optional
        Flag to update orthoimages display. Default is False.
    histogram : bool, optional
        Flag to update histogram display. Default is False.
    LUT : bool, optional
        Flag to update lookup table display. Default is False.
    similarityfunc : bool, optional
        Flag to update similarity function display. Default is False.
    focusvals : bool, optional
        Flag to update focus values display. Default is False.
    callingfunc : Any, optional
        The function that called this function, used for verbose output.
        Default is None.
    verbose : int, optional
        Verbosity level. If greater than 1, prints detailed information
        about the function call. Default is 0.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    When LUT flag is True, orthoimages flag is automatically set to True
    as LUT updates require orthoimages to be updated as well.

    Examples
    --------
    >>> updateUI(histogram=True, LUT=True)
    >>> updateUI(verbose=2, callingfunc="main")
    """
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


def updateOrthoImages() -> None:
    """
    Update orthographic image displays with current location and time position.

    This function updates all orthographic images in the visualization system
    with the current spatial coordinates and time position. It also updates
    the main window's map display and time slider to reflect the current state.

    Notes
    -----
    This function relies on several global variables that must be properly
    initialized before calling this function:

    - hist: Historical data container
    - currentdataset: Current dataset information
    - maps: Map data structure
    - panetomap: Mapping between panes and maps
    - ui: User interface components
    - mainwin: Main window object
    - orthoimagedict: Dictionary of orthographic images
    - currentloc: Current location object with xpos, ypos, zpos, and tpos attributes
    - overlays: Overlay data structure
    - xdim, ydim, zdim: Dimension parameters
    - imagadj: Image adjustment parameters

    Examples
    --------
    >>> updateOrthoImages()
    # Updates all orthographic images with current position and time
    """
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


def printfocusvals() -> None:
    """
    Print focus values for all overlays at the current location.

    This function logs the focus values of all overlays (except 'mnibrainmask')
    at the current spatial and temporal position. The output includes labels
    and corresponding focus values, formatted with indentation. Special handling
    is applied for 'atlas' and 'failimage' overlays to display atlas labels or
    diagnostic failure messages, respectively.

    Notes
    -----
    This function modifies the global state by accessing and updating global
    variables: `ui`, `overlays`, `currentdataset`, `currentloc`, and `simfuncFitter`.

    Examples
    --------
    Assuming `currentloc` is set and `overlays` contains valid data:

    >>> printfocusvals()
    Values at location 10, 20, 30
       Overlay1: 45.678
       Overlay2: 12.345
       atlas:  BrainRegionA
       failimage:
           Error: Invalid data
           Reason: Out of bounds
    """
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


def regressor_radioButton_clicked(theregressor: Any, enabled: Any) -> None:
    """
    Handle radio button click event for regressor selection.

    This function updates the current dataset's regressor focus and triggers
    updates to the regressor and its spectrum visualization.

    Parameters
    ----------
    theregressor : Any
        The regressor object or identifier that was clicked
    enabled : Any
        Boolean or indicator showing whether the regressor is enabled

    Returns
    -------
    None
        This function does not return any value

    Notes
    -----
    This function modifies global state through `currentdataset.setfocusregressor()`
    and calls two update functions: `updateRegressor()` and `updateRegressorSpectrum()`.

    Examples
    --------
    >>> regressor_radioButton_clicked('linear_regression', True)
    >>> regressor_radioButton_clicked('polynomial_regression', False)
    """
    global currentdataset
    currentdataset.setfocusregressor(theregressor)
    updateRegressor()
    updateRegressorSpectrum()


def activateDataset(
    currentdataset: Any,
    ui: Any,
    win: Any,
    defaultdict: Any,
    overlayGraphicsViews: Any,
    verbosity: int = 0,
) -> None:
    """
    Initialize and activate dataset for visualization and analysis.

    This function sets up the global state and UI elements based on the provided dataset,
    configures overlay displays, initializes image viewers, and updates the user interface
    to reflect the current dataset's properties.

    Parameters
    ----------
    currentdataset : Any
        Object containing dataset metadata and data, including dimensions, regressors,
        overlays, and file paths.
    ui : Any
        User interface object containing widgets such as radio buttons, buttons, and
        graphics views for display.
    win : Any
        Main window object used to set the window title.
    defaultdict : Any
        Dictionary containing default display settings for overlays, such as colormap,
        display state, and labels.
    overlayGraphicsViews : Any
        List or collection of graphics views used for displaying overlay images.
    verbosity : int, optional
        Level of verbosity for logging output. Default is 0 (no output).

    Returns
    -------
    None
        This function does not return a value but modifies global state and UI elements.

    Notes
    -----
    This function modifies global variables including `regressors`, `overlays`, `mainwin`,
    `xdim`, `ydim`, `zdim`, `tdim`, `xpos`, `ypos`, `zpos`, `tpos`, `timeaxis`,
    `usecorrout`, `orthoimagedict`, `panesinitialized`, `uiinitialized`, and `currentloc`.

    Examples
    --------
    >>> activateDataset(dataset, ui, window, defaults, graphics_views, verbosity=1)
    """
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
    themap: Any,
    thepane: Any,
    view: Any,
    button: Any,
    panemap: Any,
    orthoimagedict: Any,
    bgmap: Optional[Any] = None,
    sm_imgsize: float = 32.0,
) -> None:
    """
    Load and initialize an ortho image item for a given map pane.

    This function creates an OrthoImageItem object and stores it in the
    orthoimagedict using the map name as the key. It also updates the panemap
    to associate the pane with the map name.

    Parameters
    ----------
    themap : Any
        The map object to be loaded into the pane
    thepane : Any
        The pane identifier where the map will be displayed
    view : Any
        The view configuration for the map display
    button : Any
        The button configuration for the map interaction
    panemap : Any
        Dictionary mapping panes to map names
    orthoimagedict : Any
        Dictionary storing ortho image items keyed by map names
    bgmap : Any, optional
        Background map object, default is None
    sm_imgsize : float, optional
        Size of the image items, default is 32.0

    Returns
    -------
    None
        This function modifies the orthoimagedict and panemap in-place

    Notes
    -----
    The function checks if the map has a display state before creating the
    OrthoImageItem. The OrthoImageItem is initialized with the same view
    values for all three parameters, which may need to be adjusted based on
    specific requirements.

    Examples
    --------
    >>> loadpane(map_obj, pane_id, view_config, button_config, panemap, orthoimagedict)
    >>> loadpane(map_obj, pane_id, view_config, button_config, panemap, orthoimagedict, bgmap=bg_obj)
    """
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


def tidepool(args: argparse.Namespace) -> None:
    """
    Initialize and run the TiDePool GUI application for rapidtide data analysis.

    This function sets up the main graphical user interface for the TiDePool
    application, initializes global variables, configures the UI based on
    command-line arguments, and starts the Qt event loop for user interaction.

    Parameters
    ----------
    args : Any
        Command-line arguments parsed by argparse. Expected to contain attributes
        such as uistyle, anatname, geommaskname, datafileroot, offsettime, trval,
        userise, useatlas, verbose, and ignoredimmatch.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function initializes numerous global variables related to the UI state,
    image data, and analysis parameters. It dynamically imports UI templates
    based on the Qt binding being used (PyQt5, PyQt6, or PySide6) and the
    specified UI style (normal or big).

    The function sets up various Qt widgets including main window, menu bar,
    image views, colorbars, and plot windows for displaying analysis results.
    It also configures event handlers for user interactions and initializes
    the data loading process.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--uistyle', default='normal')
    >>> parser.add_argument('--verbose', action='store_true')
    >>> args = parser.parse_args()
    >>> tidepool(args)
    # Starts the TiDePool GUI with specified settings
    """
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

    if pyqtbinding == "pyqt5":
        if args.uistyle == "normal":
            import rapidtide.tidepoolTemplate_alt as uiTemplate
        elif args.uistyle == "big":
            import rapidtide.tidepoolTemplate_big as uiTemplate
        else:
            import rapidtide.tidepoolTemplate as uiTemplate
    elif pyqtbinding == "pyqt6":
        if args.uistyle == "normal":
            import rapidtide.tidepoolTemplate_alt_qt6 as uiTemplate
        elif args.uistyle == "big":
            import rapidtide.tidepoolTemplate_big_qt6 as uiTemplate
        else:
            import rapidtide.tidepoolTemplate_qt6 as uiTemplate
    elif pyqtbinding == "pyside6":
        if args.uistyle == "normal":
            import rapidtide.tidepoolTemplate_alt_qt6 as uiTemplate
        elif args.uistyle == "big":
            import rapidtide.tidepoolTemplate_big_qt6 as uiTemplate
        else:
            import rapidtide.tidepoolTemplate_qt6 as uiTemplate
    else:
        print("unsupported")

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
    win = KeyPressWindow()
    win.sigKeyPress.connect(keyPressed)
    # win = QtWidgets.QMainWindow()
    ui = uiTemplate.Ui_MainWindow()
    ui.setupUi(win)
    win.show()
    win.setWindowTitle("TiDePool")

    # create the menu bar
    print("creating menu bar")
    menuBar = win.menuBar()
    fileMenu = menuBar.addMenu("File")
    if pyqtbinding == "pyqt5":
        qactionfunc = QtWidgets.QAction
    elif pyqtbinding == "pyqt6":
        qactionfunc = QtGui.QAction
    elif pyqtbinding == "pyside6":
        qactionfunc = QtGui.QAction
    else:
        print("unsupported")
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
            "label": "sLFO Fit R2",
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
            "label": "LFO var. before filtering",
            "display": extramaps,
            "funcmask": "p_lt_0p050_mask",
        },
        "varAfter": {
            "colormap": gen_thermal_state(),
            "label": "LFO var. after filtering",
            "display": extramaps,
            "funcmask": "p_lt_0p050_mask",
        },
        "varChange": {
            "colormap": gen_thermal_state(),
            "label": "LFO var. decrease %",
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
            "label": "sLFO fit coefficient",
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
    if pyqtbinding == "pyqt5":
        qactionfunc = QtWidgets.QAction
    elif pyqtbinding == "pyqt6":
        qactionfunc = QtGui.QAction
    elif pyqtbinding == "pyside6":
        qactionfunc = QtGui.QAction
    else:
        print("unsupported")
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
    def on_mask_context_menu(point: Any) -> None:
        """
        Show context menu for mask operations.

        This function displays a context menu at the specified point coordinates
        for mask-related operations. The menu is positioned relative to the
        setMask_Button widget.

        Parameters
        ----------
        point : Any
            The point coordinates where the context menu should be displayed.
            Typically a QPoint object or similar coordinate representation.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The context menu is displayed using the exec() method on the popMaskMenu
        object, with the point converted to global coordinates using the
        mapToGlobal() method of the setMask_Button widget.

        Examples
        --------
        >>> on_mask_context_menu(QPoint(100, 150))
        # Displays context menu at global coordinates (100, 150)
        """
        # show context menu
        popMaskMenu.exec(ui.setMask_Button.mapToGlobal(point))

    ui.setMask_Button.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
    ui.setMask_Button.customContextMenuRequested.connect(on_mask_context_menu)

    # configure the file selection popup menu
    def on_file_context_menu(point: Any) -> None:
        """
        Show context menu at specified point.

        This function displays a context menu when a user right-clicks on a file
        element in the UI. The menu is positioned at the global coordinates
        corresponding to the provided point relative to the file button.

        Parameters
        ----------
        point : Any
            The point coordinates where the context menu should be displayed.
            Typically this is a QPoint object from Qt's event system.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The context menu is displayed using the `exec_` method of the popup menu,
        which blocks execution until the menu is closed. The menu position is
        converted from local coordinates to global coordinates using the
        `mapToGlobal` method of the file button widget.

        Examples
        --------
        >>> on_file_context_menu(QPoint(100, 150))
        # Displays context menu at global position (100, 150)
        """
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
