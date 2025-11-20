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
A widget for orthographically displaying 3 and 4 dimensional data
"""
import os
from typing import Any

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

try:
    from PIL import Image

    PILexists = True
except ImportError:
    PILexists = False

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


def newColorbar(
    left: float, top: float, impixpervoxx: float, impixpervoxy: float, imgsize: int
) -> tuple[Any, Any, Any, NDArray[np.float64]]:
    """
    Create a colorbar widget with foreground and background image items for plotting.

    This function generates a colorbar using PyGraphQt (pg) components, including
    a foreground and background `ImageItem`, a `ViewBox` for layout control, and
    a 2D array of color values. The colorbar is scaled and positioned according
    to the provided parameters.

    Parameters
    ----------
    left : float
        The x-coordinate of the top-left corner of the colorbar in the scene.
    top : float
        The y-coordinate of the top-left corner of the colorbar in the scene.
    impixpervoxx : float
        Scaling factor for the horizontal axis (pixel per unit in x-direction).
    impixpervoxy : float
        Scaling factor for the vertical axis (pixel per unit in y-direction).
    imgsize : int
        The size of the colorbar image in pixels, used to determine dimensions.

    Returns
    -------
    tuple[Any, Any, Any, NDArray[np.float64]]
        A tuple containing:
        - `thecolorbarfgwin`: The foreground `ImageItem` for the colorbar.
        - `thecolorbarbgwin`: The background `ImageItem` for the colorbar.
        - `theviewbox`: The `ViewBox` that contains the colorbar items.
        - `colorbarvals`: A 2D NumPy array of shape `(cb_xdim, cb_ydim)` with
          color values ranging from 0.0 to 1.0, used for rendering the colorbar.

    Notes
    -----
    The colorbar uses a linear gradient from black (0.0) to white (1.0) along the
    vertical axis. The horizontal dimension is set to 1/10th of `imgsize`, and
    the vertical dimension equals `imgsize`. The `ViewBox` is configured to disable
    auto-ranging and mouse interaction.

    Examples
    --------
    >>> fg_item, bg_item, view_box, color_vals = newColorbar(
    ...     left=100, top=50, impixpervoxx=1.0, impixpervoxy=1.0, imgsize=256
    ... )
    >>> view_box.addItem(fg_item)
    >>> view_box.addItem(bg_item)
    """
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


def setupViewWindow(
    view: Any,
    left: float,
    top: float,
    impixpervoxx: float,
    impixpervoxy: float,
    imgsize: int,
    enableMouse: bool = False,
) -> tuple[Any, Any, Any, Any, Any]:
    """
    Set up a view window with background and foreground image items, and crosshair lines.

    This function configures a PyGraphQt view box with specified image transformation
    parameters and adds background and foreground image items along with vertical and
    horizontal crosshair lines for visualization purposes.

    Parameters
    ----------
    view : Any
        The parent view object to which the view box will be added.
    left : float
        The x-coordinate offset for the image transformation.
    top : float
        The y-coordinate offset for the image transformation.
    impixpervoxx : float
        The scaling factor for the horizontal axis of the image.
    impixpervoxy : float
        The scaling factor for the vertical axis of the image.
    imgsize : int
        The size of the image in pixels, used to set the view range.
    enableMouse : bool, optional
        Whether to enable mouse interaction with the view box. Default is False.

    Returns
    -------
    tuple[Any, Any, Any, Any, Any]
        A tuple containing:
        - theviewfgwin: The foreground image item.
        - theviewbgwin: The background image item.
        - theviewvLine: The vertical crosshair line.
        - theviewhLine: The horizontal crosshair line.
        - theviewbox: The configured view box object.

    Notes
    -----
    The view box is locked to a 1:1 aspect ratio and initialized with a gray background.
    The transformation applied to both image items includes translation and scaling
    to align the image properly within the view.

    Examples
    --------
    >>> import pyqtgraph as pg
    >>> from PyQt5 import QtCore, QtGui
    >>> view = pg.GraphicsLayoutWidget()
    >>> fgwin, bgwin, vline, hline, vbox = setupViewWindow(
    ...     view, 0, 0, 1.0, 1.0, 256, enableMouse=True
    ... )
    """

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
        map: Any,
        axview: Any,
        corview: Any,
        sagview: Any,
        enableMouse: bool = False,
        button: Any | None = None,
        imgsize: int = 64,
        arrangement: int = 0,
        bgmap: Any | None = None,
        verbose: int = 0,
    ) -> None:
        """
        Initialize the OrthoImageItem widget for displaying 3D medical images in orthogonal views.

        This constructor sets up the necessary attributes and configurations for rendering
        a 3D image volume in three orthogonal views (axial, coronal, and sagittal) using
        PyQt and PySide. It handles coordinate transformations, view setup, and mouse interaction
        if enabled.

        Parameters
        ----------
        map : Any
            The 3D image data map object containing image dimensions and voxel information.
        axview : Any
            The axial view widget (e.g., a PySide QGraphicsView).
        corview : Any
            The coronal view widget (e.g., a PySide QGraphicsView).
        sagview : Any
            The sagittal view widget (e.g., a PySide QGraphicsView).
        enableMouse : bool, optional
            Whether to enable mouse interaction for navigating the views, by default False.
        button : Any | None, optional
            Mouse button used for interaction, by default None.
        imgsize : int, optional
            Size of the image display in pixels, by default 64.
        arrangement : int, optional
            Layout arrangement for the views, by default 0.
        bgmap : Any | None, optional
            Background image map for overlay, by default None.
        verbose : int, optional
            Verbosity level for printing debug information, by default 0.

        Returns
        -------
        None
            This method initializes the object and does not return any value.

        Notes
        -----
        The method performs coordinate transformations to map voxel indices to physical space
        and sets up the view ranges and layouts for each of the three orthogonal views.
        If `enableMouse` is True, mouse event handlers are attached to allow navigation.

        Examples
        --------
        >>> ortho_item = OrthoImageItem(
        ...     map=volume_map,
        ...     axview=axial_view,
        ...     corview=coronal_view,
        ...     sagview=sagittal_view,
        ...     enableMouse=True
        ... )
        """
        QtWidgets.QWidget.__init__(self)
        self.map = map
        self.mapname = self.map.label
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
        self.axviewbox = None
        self.corviewbox = None
        self.sagviewbox = None
        self.axviewwin = None
        self.corviewwin = None
        self.sagviewwin = None
        self.axviewbgwin = None
        self.corviewbgwin = None
        self.sagviewbgwin = None
        self.debug = True
        self.arrangement = arrangement

        if self.verbose > 1:
            print("OrthoImageItem initialization:")
            print("    Map name:", self.mapname)
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
        ) = setupViewWindow(
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
        ) = setupViewWindow(
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
        ) = setupViewWindow(
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

    def xvox2pix(self, xpos: int) -> int:
        """
        Convert voxel position to pixel position along x-axis.

        Parameters
        ----------
        xpos : int
            Voxel position along x-axis to be converted to pixel coordinates.

        Returns
        -------
        int
            Corresponding pixel position along x-axis.

        Notes
        -----
        This function performs a linear transformation from voxel coordinates to pixel coordinates
        using the formula: pixel = offsetx + impixpervoxx * voxel_position

        Examples
        --------
        >>> obj.xvox2pix(10)
        15
        >>> obj.xvox2pix(0)
        5
        """
        return int(np.round(self.offsetx + self.impixpervoxx * xpos))

    def yvox2pix(self, ypos: int) -> int:
        """
        Convert voxel y-coordinate to pixel y-coordinate.

        This function transforms a y-coordinate from voxel space to pixel space using
        the transformation parameters stored in the object.

        Parameters
        ----------
        ypos : int
            Y-coordinate in voxel space to be converted to pixel space.

        Returns
        -------
        int
            Y-coordinate in pixel space corresponding to the input voxel y-coordinate.

        Notes
        -----
        The conversion follows the formula: pixel_y = offsety + impixpervoxy * voxel_y
        where offsety and impixpervoxy are attributes of the object.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.offsety = 10
        >>> obj.impixpervoxy = 2.5
        >>> obj.yvox2pix(4)
        20
        """
        return int(np.round(self.offsety + self.impixpervoxy * ypos))

    def zvox2pix(self, zpos: int) -> int:
        """
        Convert z-voxel position to pixel position.

        This function transforms a z-coordinate in voxel space to its corresponding
        position in pixel space using the camera's offset and pixel-per-voxel ratio.

        Parameters
        ----------
        zpos : int
            Z-coordinate in voxel space to be converted to pixel space.

        Returns
        -------
        int
            Corresponding z-position in pixel space.

        Notes
        -----
        The conversion follows the formula: pixel_position = offsetz + impixpervoxz * zpos
        where:
        - offsetz: base offset in pixel space
        - impixpervoxz: pixels per voxel in z-direction

        Examples
        --------
        >>> zvox2pix(10)
        150
        """
        return int(np.round(self.offsetz + self.impixpervoxz * zpos))

    def xpix2vox(self, xpix: float) -> int:
        """
        Convert pixel coordinate to voxel coordinate in x-direction.

        This function transforms a pixel coordinate in the x-direction to its
        corresponding voxel coordinate, taking into account the image offset and
        pixel-to-voxel conversion factors.

        Parameters
        ----------
        xpix : float
            The pixel coordinate in x-direction to be converted to voxel coordinate.

        Returns
        -------
        int
            The corresponding voxel coordinate in x-direction, clamped to the
            valid range [0, self.xdim-1].

        Notes
        -----
        The conversion is performed using the formula: voxel = (pixel - offset) / pixels_per_voxel.
        Edge cases are handled by clamping values to the valid voxel range.

        Examples
        --------
        >>> # Assuming self.offsetx = 10, self.impixpervoxx = 2.0, self.xdim = 100
        >>> result = self.xpix2vox(20.0)
        >>> print(result)
        5
        """
        thepos = (xpix - self.offsetx) / self.impixpervoxx
        if thepos > self.xdim - 1:
            thepos = self.xdim - 1
        if thepos < 0:
            thepos = 0
        return int(np.round(thepos))

    def ypix2vox(self, ypix: float) -> int:
        """
        Convert y pixel coordinate to voxel coordinate.

        This function transforms a y pixel coordinate in the image space to
        the corresponding voxel coordinate in the volume space, taking into
        account the image offset and pixel spacing.

        Parameters
        ----------
        ypix : float
            Y pixel coordinate in image space

        Returns
        -------
        int
            Corresponding y voxel coordinate in volume space

        Notes
        -----
        The conversion uses the formula: voxel = (pixel - offset) / pixels_per_voxel
        Boundary conditions are enforced to ensure the result stays within valid
        voxel range [0, ydim-1].

        Examples
        --------
        >>> # Assuming self.offsety = 10, self.impixpervoxy = 2.0, self.ydim = 100
        >>> ypix2vox(12)  # Returns 1
        >>> ypix2vox(200) # Returns 99 (clamped to maximum valid voxel)
        """
        thepos = (ypix - self.offsety) / self.impixpervoxy
        if thepos > self.ydim - 1:
            thepos = self.ydim - 1
        if thepos < 0:
            thepos = 0
        return int(np.round(thepos))

    def zpix2vox(self, zpix: float) -> int:
        """
        Convert z-pixel coordinate to z-voxel coordinate.

        This function transforms a z-pixel coordinate to the corresponding z-voxel
        coordinate by applying the inverse transformation using the offset and
        pixel-per-voxel parameters.

        Parameters
        ----------
        zpix : float
            Z-pixel coordinate to be converted to voxel coordinate.

        Returns
        -------
        int
            Corresponding z-voxel coordinate. The result is clamped to the
            valid range [0, self.zdim-1].

        Notes
        -----
        The conversion is performed using the formula:
        voxel = (zpix - offsetz) / impixpervoxz

        If the resulting voxel coordinate exceeds the valid range [0, self.zdim-1],
        it is clamped to the nearest boundary value.

        Examples
        --------
        >>> # Assuming self.offsetz = 10, self.impixpervoxz = 2, self.zdim = 100
        >>> zpix2vox(12)  # Returns 1
        >>> zpix2vox(10)  # Returns 0
        >>> zpix2vox(200) # Returns 99 (clamped to max)
        """
        thepos = (zpix - self.offsetz) / self.impixpervoxz
        if thepos > self.zdim - 1:
            thepos = self.zdim - 1
        if thepos < 0:
            thepos = 0
        return int(np.round(thepos))

    def updateAllViews(self) -> None:
        """
        Update all three views (axial, coronal, and sagittal) of the visualization.

        This function updates the axial, coronal, and sagittal views based on the current
        position settings (`xpos`, `ypos`, `zpos`) and time index (`tpos`), using the
        underlying data, mask, and background map if available. It also updates the
        position lines in each view to reflect the current voxel coordinates.

        Notes
        -----
        The function handles both 2D and 3D data depending on the value of `self.tdim`.
        If `self.tdim == 1`, the data is treated as 2D; otherwise, it is treated as 3D
        with a time dimension.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return any value.

        Examples
        --------
        >>> updateAllViews()
        Updates all three views (axial, coronal, sagittal) with current data and position.
        """
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

    def updateOneView(
        self,
        data: NDArray,
        mask: NDArray,
        background: NDArray | None,
        theLUT: NDArray,
        thefgwin: Any,
        thebgwin: Any,
    ) -> None:
        """
        Update the visualization view with processed data and optional background.

        This function applies a lookup table to the input data, displays it in the
        foreground window, and optionally displays a background image in the background window.

        Parameters
        ----------
        data : NDArray
            The main data array to be processed and displayed.
        mask : NDArray
            The mask array used for data processing.
        background : NDArray | None
            The background data array to be displayed, or None if no background.
        theLUT : NDArray
            The lookup table array used for color mapping.
        thefgwin : Any
            The foreground window object where the processed data will be displayed.
        thebgwin : Any
            The background window object where the background data will be displayed.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        The function applies the lookup table using `self.applyLUT` method and sets
        the image in the foreground window with float data type. If background is
        provided, it is displayed in the background window with automatic level
        adjustment enabled.

        Examples
        --------
        >>> updateOneView(data, mask, background, theLUT, fg_window, bg_window)
        """
        im = self.applyLUT(data, mask, theLUT, self.map.dispmin, self.map.dispmax)
        thefgwin.setImage(im.astype("float"))
        if background is not None:
            thebgwin.setImage(background.astype("float"), autoLevels=True)

    def setMap(self, themap: Any) -> None:
        """
        Set the map attribute and update related properties.

        This method assigns the provided map to the instance and updates
        the dimensionality and label properties based on the map's attributes.

        Parameters
        ----------
        themap : Any
            The map object to be assigned to the instance. Expected to have
            attributes 'tdim' (dimensionality) and 'label' (name/label).

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method assumes that the input map object has 'tdim' and 'label'
        attributes. If these attributes are not present, an AttributeError
        will be raised.

        Examples
        --------
        >>> instance.setMap(my_map)
        >>> print(instance.tdim)
        2
        >>> print(instance.mapname)
        'my_map_label'
        """
        self.map = themap
        self.tdim = self.map.tdim
        self.mapname = self.map.label

    def enableView(self) -> None:
        """
        Enable and display all view components.

        This method enables the main button and displays all three view components
        (axial, coronal, and sagittal) by setting their visibility to True.

        Parameters
        ----------
        self : object
            The instance of the class containing the view components and button.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method first checks if the button exists before attempting to modify it.
        If the button exists, it updates the button text with the map label and enables
        the button. All three view components (axview, corview, and sagview) are then
        made visible.

        Examples
        --------
        >>> viewer = Viewer()
        >>> viewer.enableView()
        >>> # All view components are now visible and button is enabled
        """
        if self.button is not None:
            self.button.setText(self.map.label)
            self.button.setDisabled(False)
            self.button.show()
        self.axview.show()
        self.corview.show()
        self.sagview.show()

    def applyLUT(
        self, theimage: NDArray, mask: NDArray, theLUT: NDArray, dispmin: float, dispmax: float
    ) -> NDArray:
        """
        Apply a lookup table to an image with optional masking and scaling.

        This function maps image values to colors using a lookup table, applying
        scaling based on the display range and masking invalid regions.

        Parameters
        ----------
        theimage : NDArray
            Input image array to be mapped using the lookup table
        mask : NDArray
            Mask array where values less than 1 will be set to transparent (alpha = 0)
        theLUT : NDArray
            Lookup table array for color mapping, typically with shape (N, 4) for RGBA values
        dispmin : float
            Minimum display value for scaling the input image
        dispmax : float
            Maximum display value for scaling the input image

        Returns
        -------
        NDArray
            Mapped image array with the same shape as input image, with colors applied
            from the lookup table and masked regions set to transparent

        Notes
        -----
        The function performs the following operations:
        1. Scales input image values to index range of the lookup table
        2. Clamps scaled values to valid lookup table indices
        3. Maps image values to colors using the lookup table
        4. Applies mask to set transparent pixels where mask < 1

        Examples
        --------
        >>> # Apply LUT to image with display range [0, 255]
        >>> result = applyLUT(image, mask, lut, 0.0, 255.0)
        """
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

    def updateCursors(self) -> None:
        """
        Update cursor positions in all view axes based on voxel coordinates.

        This method converts voxel coordinates to pixel coordinates for each view
        and updates the corresponding cursor lines in the axial, coronal, and
        sagittal views.

        Parameters
        ----------
        self : object
            The instance containing the cursor update functionality

        Returns
        -------
        None
            This method does not return any value

        Notes
        -----
        The method uses the following coordinate conversion methods:
        - xvox2pix: converts x voxel coordinate to x pixel coordinate
        - yvox2pix: converts y voxel coordinate to y pixel coordinate
        - zvox2pix: converts z voxel coordinate to z pixel coordinate

        The cursor lines are updated in the following views:
        - axviewvLine: axial view vertical line
        - axviewhLine: axial view horizontal line
        - corviewvLine: coronal view vertical line
        - corviewhLine: coronal view horizontal line
        - sagviewvLine: sagittal view vertical line
        - sagviewhLine: sagittal view horizontal line

        Examples
        --------
        >>> viewer = Viewer()
        >>> viewer.xpos = 10
        >>> viewer.ypos = 15
        >>> viewer.zpos = 20
        >>> viewer.updateCursors()
        >>> # Cursor positions updated in all three views
        """
        xpix = self.xvox2pix(self.xpos)
        ypix = self.yvox2pix(self.ypos)
        zpix = self.zvox2pix(self.zpos)
        self.axviewvLine.setValue(xpix)
        self.axviewhLine.setValue(ypix)
        self.corviewvLine.setValue(xpix)
        self.corviewhLine.setValue(zpix)
        self.sagviewvLine.setValue(ypix)
        self.sagviewhLine.setValue(zpix)

    def handlemouseup(self, event: Any) -> None:
        """
        Handle mouse button up event.

        This method is called when a mouse button is released. It updates the internal
        state to reflect that the button is no longer pressed and refreshes all views
        to ensure proper cursor representation and visual feedback.

        Parameters
        ----------
        event : Any
            The mouse event object containing information about the mouse button release.
            This typically includes coordinates and button information.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method updates the cursor state and triggers updates across all connected views
        to ensure consistent visual representation of the mouse state.

        Examples
        --------
        >>> widget.handlemouseup(event)
        >>> # Mouse button state updated and views refreshed
        """
        self.buttonisdown = False
        self.updateCursors()
        self.updateAllViews()

    def handleaxmousemove(self, event: Any) -> None:
        """
        Handle mouse move events for axis navigation.

        This method is called when the mouse is moved while a button is pressed,
        updating the position coordinates and triggering view updates.

        Parameters
        ----------
        event : Any
            Mouse event object containing position information. The event object
            should have a `pos()` method that returns the mouse position.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method only processes mouse movement when `self.buttonisdown` is True.
        Position coordinates are converted from pixel to voxel space using
        `xpix2vox` and `ypix2vox` conversion methods. The image size is used to
        properly map y-coordinates from pixel space to voxel space.

        Examples
        --------
        >>> # This method is typically called internally by the GUI framework
        >>> # when mouse events occur
        >>> self.handleaxmousemove(mouse_event)
        >>> # Updates self.xpos and self.ypos coordinates
        >>> # Triggers view updates and emits updated signal
        """
        if self.buttonisdown:
            self.xpos = self.xpix2vox(event.pos().x() - 1)
            self.ypos = self.ypix2vox(self.imgsize - event.pos().y() + 1)
            self.updateAllViews()
            self.updated.emit()

    def handlecormousemove(self, event: Any) -> None:
        """
        Handle mouse move events for correlation function view navigation.

        This method updates the x and z position coordinates based on mouse movement
        when a button is pressed. It converts pixel coordinates to voxel coordinates
        and triggers view updates and signal emission.

        Parameters
        ----------
        event : Any
            Mouse event object containing position information. Expected to have
            a `pos()` method that returns a position object with `x()` and `y()` methods.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method only processes mouse movement when `self.buttonisdown` is True.
        Pixel coordinates are converted to voxel coordinates using:
        - x position: `xpix2vox(event.pos().x() - 1)`
        - z position: `zpix2vox(self.imgsize - event.pos().y() + 1)`

        Examples
        --------
        >>> # This method is typically called internally by the GUI framework
        >>> # when mouse events occur in the correlative view
        >>> handlecormousemove(mouse_event)
        """
        if self.buttonisdown:
            self.xpos = self.xpix2vox(event.pos().x() - 1)
            self.zpos = self.zpix2vox(self.imgsize - event.pos().y() + 1)
            self.updateAllViews()
            self.updated.emit()

    def handlesagmousemove(self, event: Any) -> None:
        """
        Handle mouse move events for sagittal view navigation.

        This method updates the y and z position coordinates when the mouse is moved
        while the button is pressed, and triggers view updates and signals.

        Parameters
        ----------
        event : Any
            Mouse event object containing position information. Expected to have
            a `pos()` method that returns a position object with `x()` and `y()` methods.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method only processes mouse movement when `self.buttonisdown` is True.
        Position coordinates are converted from pixel to voxel space using
        `self.ypix2vox()` and `self.zpix2vox()` conversion methods.

        Examples
        --------
        >>> self.handlesagmousemove(mouse_event)
        # Updates position coordinates and triggers view updates
        """
        if self.buttonisdown:
            self.ypos = self.ypix2vox(event.pos().x() - 1)
            self.zpos = self.zpix2vox(self.imgsize - event.pos().y() + 1)
            self.updateAllViews()
            self.updated.emit()

    def handleaxkey(self, event: Any) -> None:
        """
        Handle axis key events and update views.

        This method is called when an axis key event occurs. It prints the event
        if verbose mode is enabled (verbose > 1), updates all views in the
        application, and emits an updated signal.

        Parameters
        ----------
        event : Any
            The axis key event object containing event information and data.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method is typically used as an event handler for axis key events.
        The verbose flag controls whether event information is printed to stdout.

        Examples
        --------
        >>> # Typically called internally by event system
        >>> self.handleaxkey(some_event_object)
        >>> # Output will be printed if self.verbose > 1
        """
        if self.verbose > 1:
            print(event)
        self.updateAllViews()
        self.updated.emit()

    def handleaxclick(self, event: Any) -> None:
        """
        Handle mouse click events on the axis.

        This method processes mouse click events to convert pixel coordinates to voxel coordinates,
        updates the button state, and triggers view updates.

        Parameters
        ----------
        event : Any
            Mouse event object containing position information. Expected to have a `pos()` method
            that returns a position object with `x()` and `y()` methods.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method converts pixel coordinates to voxel coordinates using the following transformations:
        - x-coordinate: xpix2vox(event.pos().x() - 1)
        - y-coordinate: ypix2vox(self.imgsize - event.pos().y() + 1)

        The y-coordinate transformation accounts for the difference between pixel coordinate systems
        where (0,0) is typically at the top-left corner, and voxel coordinate systems where (0,0)
        is at the bottom-left corner.

        Examples
        --------
        >>> handleaxclick(event)
        # Processes the mouse click event and updates internal state
        """
        self.xpos = self.xpix2vox(event.pos().x() - 1)
        self.ypos = self.ypix2vox(self.imgsize - event.pos().y() + 1)
        self.buttonisdown = True
        self.updateAllViews()
        self.updated.emit()

    def handlecorclick(self, event: Any) -> None:
        """
        Handle mouse click event for coordinate conversion and view update.

        This method processes mouse click events to convert pixel coordinates to voxel coordinates,
        updates the button state, and triggers view updates and signals.

        Parameters
        ----------
        event : Any
            Mouse event object containing position information. Expected to have a `pos()` method
            that returns a point with `x()` and `y()` coordinate methods.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method converts pixel coordinates to voxel coordinates using:
        - x coordinate: `xpix2vox(event.pos().x() - 1)`
        - z coordinate: `zpix2vox(self.imgsize - event.pos().y() + 1)`

        The coordinate transformation accounts for pixel indexing differences and image orientation.

        Examples
        --------
        >>> # Typical usage in event handling context
        >>> self.handlecorclick(mouse_event)
        >>> # Updates self.xpos, self.zpos, self.buttonisdown and triggers updates
        """
        self.xpos = self.xpix2vox(event.pos().x() - 1)
        self.zpos = self.zpix2vox(self.imgsize - event.pos().y() + 1)
        self.buttonisdown = True
        self.updateAllViews()
        self.updated.emit()

    def handlesagclick(self, event: Any) -> None:
        """
        Handle mouse click events for sagittal view navigation.

        This method processes mouse click events in the sagittal view, converting pixel
        coordinates to voxel coordinates and updating the view accordingly.

        Parameters
        ----------
        event : Any
            Mouse event object containing position information. Expected to have a
            `pos()` method that returns a position object with `x()` and `y()` methods.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The method updates internal position variables (ypos, zpos) and triggers
        view updates through the updateAllViews() method and updated.emit() signal.
        The pixel coordinates are adjusted by subtracting 1 from x and adding 1 to y
        for proper coordinate system alignment.

        Examples
        --------
        >>> viewer = ImageViewer()
        >>> event = MouseEvent(pos=QPoint(100, 150))
        >>> viewer.handlesagclick(event)
        >>> print(viewer.ypos, viewer.zpos)
        (0.5, 0.75)
        """
        self.ypos = self.ypix2vox(event.pos().x() - 1)
        self.zpos = self.zpix2vox(self.imgsize - event.pos().y() + 1)
        self.buttonisdown = True
        self.updateAllViews()
        self.updated.emit()

    def setXYZpos(self, xpos: int, ypos: int, zpos: int, emitsignal: bool = True) -> None:
        """
        Set the XYZ position coordinates and update views.

        Parameters
        ----------
        xpos : int
            The x-coordinate position value.
        ypos : int
            The y-coordinate position value.
        zpos : int
            The z-coordinate position value.
        emitsignal : bool, optional
            If True, emit the updated signal after position change (default is True).

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method converts all position parameters to integers and updates
        all views through the updateAllViews() call. The updated signal is
        only emitted when emitsignal is True.

        Examples
        --------
        >>> obj.setXYZpos(10, 20, 30)
        >>> obj.setXYZpos(5, 15, 25, emitsignal=False)
        """
        self.xpos = int(xpos)
        self.ypos = int(ypos)
        self.zpos = int(zpos)
        self.updateAllViews()
        if emitsignal:
            self.updated.emit()

    def setTpos(self, tpos: int, emitsignal: bool = True) -> None:
        """
        Set the current time position and update all views.

        This method updates the internal time position counter and triggers
        view updates. If emitsignal is True, it also emits the updated signal.

        Parameters
        ----------
        tpos : int
            The new time position to set. If tpos exceeds the maximum allowed
            time position (self.tdim - 1), it will be clamped to the maximum value.
        emitsignal : bool, optional
            If True, emit the updated signal after updating the time position.
            Default is True.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The time position is automatically clamped to the valid range [0, self.tdim - 1].
        This ensures that the time position never exceeds the dimension limits.

        Examples
        --------
        >>> obj.setTpos(5)
        >>> obj.setTpos(10, emitsignal=False)
        """
        if tpos > self.tdim - 1:
            self.tpos = int(self.tdim - 1)
        else:
            self.tpos = int(tpos)

        self.updateAllViews()
        if emitsignal:
            self.updated.emit()

    def getFocusVal(self) -> float:
        """
        Get the focus value at the current position.

        This method retrieves the data value from the masked data array at the
        current position coordinates. The method handles both 3D and 4D data arrays
        by checking the time dimension.

        Parameters
        ----------
        self : object
            The instance containing the data and position attributes.

        Returns
        -------
        float
            The focus value at the current position. Returns a scalar value
            from the masked data array.

        Notes
        -----
        The method uses the following attributes from the instance:
        - self.tdim: Time dimension flag (1 for 3D, >1 for 4D)
        - self.map.maskeddata: The data array containing the values
        - self.xpos, self.ypos, self.zpos, self.tpos: Position coordinates

        Examples
        --------
        >>> value = obj.getFocusVal()
        >>> print(value)
        0.567
        """
        if self.tdim > 1:
            return self.map.maskeddata[self.xpos, self.ypos, self.zpos, self.tpos]
        else:
            return self.map.maskeddata[self.xpos, self.ypos, self.zpos]

    def saveandcomposite(
        self,
        square_img: Any,
        fg_img: Any,
        bg_img: Any,
        name: str,
        savedir: str,
        scalefach: float,
        scalefacv: float,
    ) -> None:
        """
        Save and composite image layers into a final output file.

        This function saves individual image layers (square, foreground, and background)
        and composites them into a final image. If PIL is available, it performs
        additional operations such as flipping, scaling, and saving the composite
        as a JPEG file. Temporary files are removed after processing.

        Parameters
        ----------
        square_img : Any
            The square image to be saved.
        fg_img : Any
            The foreground image to be saved and composited.
        bg_img : Any
            The background image to be saved and composited.
        name : str
            Base name for the output files.
        savedir : str
            Directory where output files will be saved.
        scalefach : float
            Horizontal scaling factor for the final image.
        scalefacv : float
            Vertical scaling factor for the final image.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        - If PIL is available, the function will save the images in PNG format
          and composite them into a JPEG file.
        - Temporary files are deleted after processing.
        - The function uses `Image.NEAREST` for resizing to preserve pixel integrity.
        - Verbose logging is enabled based on `self.verbose` level.

        Examples
        --------
        >>> saveandcomposite(
        ...     square_img=square,
        ...     fg_img=foreground,
        ...     bg_img=background,
        ...     name="test_image",
        ...     savedir="/path/to/save",
        ...     scalefach=1.0,
        ...     scalefacv=1.0
        ... )
        """
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

    def saveDisp(self) -> None:
        """
        Save display windows for axial, coronal, and sagittal views of the image data.

        This function saves three orthogonal views (axial, coronal, and sagittal) of the
        image data to the specified directory. It also saves a colorbar and writes
        display limits to a text file. The saved images are composed using the
        `saveandcomposite` method, and the colorbar is generated using `newColorbar`.

        Parameters
        ----------
        self : object
            The instance of the class containing this method. Expected to have attributes:
            - `verbose`: int, controls verbosity of output.
            - `map`: object with attributes:
                - `namebase`: str, base name for output files.
                - `name`: str, name of the map.
                - `theLUT`: lookup table for color mapping.
                - `dispmin`: float, minimum display value.
                - `dispmax`: float, maximum display value.
            - `impixpervoxx`, `impixpervoxy`, `impixpervoxz`: float, voxel sizes in each dimension.
            - `xdim`, `ydim`, `zdim`: int, dimensions of the image in each axis.
            - `xsize`, `ysize`, `zsize`: float, physical size per voxel in each axis.
            - `axviewwin`, `axviewbgwin`, `corviewwin`, `corviewbgwin`, `sagviewwin`, `sagviewbgwin`:
              ImageItem objects for the respective views.
            - `applyLUT`: method to apply lookup table to data.
            - `saveandcomposite`: method to composite and save image views.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        - The function uses Qt bindings (PyQt5, PyQt6, or PySide6) to open a file dialog
          for selecting the output directory.
        - The saved images are named with the base name from `self.map.namebase` and
          appended with `_ax`, `_cor`, and `_sag` for the respective views.
        - A colorbar is generated but currently commented out in the code.
        - The display limits are saved to a file named `<namebase><name>_lims.txt`.

        Examples
        --------
        Assuming `obj` is an instance of the class containing this method:

        >>> obj.saveDisp()
        # Saves axial, coronal, and sagittal views to the selected directory.
        """
        if self.verbose > 1:
            print("saving main window")
        mydialog = QtWidgets.QFileDialog()
        if pyqtbinding == "pyqt5":
            options = mydialog.Options()
        elif pyqtbinding == "pyqt6":
            options = mydialog.options()
        elif pyqtbinding == "pyside6":
            options = mydialog.options()
        else:
            print("unsupported qt binding")
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
            np.ones_like(colorbarvals, "int"),
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

    def summarize(self) -> None:
        """
        Summarize the orthoimage item information.

        This method performs a summary operation on the orthoimage item, typically
        used to validate or prepare the item for further processing. When a map is
        associated with the item, it may perform additional operations or validation.

        Notes
        -----
        The method currently contains a placeholder implementation that only checks
        if a map is set. In a complete implementation, this method would likely
        generate summary statistics, validate data integrity, or prepare the item
        for display or export operations.

        Examples
        --------
        >>> item = OrthoImageItem()
        >>> item.summarize()
        # Performs summary operation on the item

        See Also
        --------
        OrthoImageItem : Main class for orthoimage items
        """
        if self.map is not None:
            # print('OrthoImageItem[', self.map.name, ']: map is set')
            pass
