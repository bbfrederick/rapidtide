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


def setendalpha(thestate: dict, alpha: int, debug: bool = False) -> dict:
    """
    Adjust the alpha values of the first and last color ticks in a gradient state.

    This function modifies the alpha channel of the first and last color stops in a
    gradient defined by `thestate`. It ensures that the first tick starts with the
    specified alpha value and the last tick ends with the specified alpha value,
    while preserving the original color values and intermediate ticks.

    Parameters
    ----------
    thestate : dict
        A dictionary representing the current gradient state. It must contain:
        - "ticks": list of tuples (float, tuple), where the first element is the
          position (0.0 to 1.0) and the second is the RGBA color tuple.
        - "mode": string, either "rgb" or "hsv", indicating the color mode.
    alpha : int
        The alpha value (0-255) to set for the first and last color ticks.
    debug : bool, optional
        If True, prints debug information about the process. Default is False.

    Returns
    -------
    dict
        A new gradient state dictionary with updated ticks. The returned dictionary
        has the same structure as the input `thestate`, but with modified alpha
        values at the start and end of the gradient.

    Notes
    -----
    - The function ensures that the first tick starts at 0.0000 and the last tick
      ends at 1.0000.
    - If the original first tick is not at 0.0, a new tick at 0.0000 is added with
      the specified alpha and a black color (0, 0, 0).
    - If the original last tick is not at 1.0, a new tick at 1.0000 is added with
      the specified alpha and a white color (255, 255, 255) in RGB mode or red (255, 0, 0)
      in HSV mode.
    - Intermediate ticks are preserved as-is.

    Examples
    --------
    >>> thestate = {
    ...     "ticks": [(0.2, (255, 0, 0, 255)), (0.8, (0, 255, 0, 255))],
    ...     "mode": "rgb"
    ... }
    >>> result = setendalpha(thestate, alpha=128)
    >>> print(result["ticks"])
    [(0.0, (0, 0, 0, 128)), (0.0001, (0, 0, 0, 255)), (0.2, (255, 0, 0, 255)), (0.8, (0, 255, 0, 255)), (0.9999, (255, 255, 255, 255)), (1.0, (255, 255, 255, 128))]
    """
    if debug:
        print("color mode:", thestate["mode"])
    sortedticks = sorted(thestate["ticks"], key=lambda x: x[0])
    newticks = []
    starttuple = sortedticks[0][1]
    if float(sortedticks[0][0]) == 0.0:
        if debug:
            print("first element is 0.0000")
        newticks.append((0.0000, (starttuple[0], starttuple[1], starttuple[2], alpha)))
        newticks.append((0.0001, starttuple))
        startloc = 1
    else:
        if debug:
            print("first element is", sortedticks[0][0], "not 0.0000")
        newticks.append((0.0000, (0, 0, 0, alpha)))
        newticks.append((0.0001, (0, 0, 0, starttuple[3])))
        startloc = 0
    newticks += sortedticks[startloc:-1]
    starttuple = sortedticks[-1][1]
    if float(sortedticks[-1][0]) == 1.0:
        if debug:
            print("last element is 1.0000")
        newticks.append((0.9999, starttuple))
        newticks.append((1.0000, (starttuple[0], starttuple[1], starttuple[2], alpha)))
    else:
        if debug:
            print("last element is", sortedticks[-1][0], "not 1.0000")
        if thestate["mode"] == "hsv":
            newticks.append((0.9999, (255, 0, 0, starttuple[3])))
            newticks.append((1.0000, (255, 0, 0, alpha)))
        else:
            newticks.append((0.9999, (255, 255, 255, starttuple[3])))
            newticks.append((1.0000, (255, 255, 255, alpha)))

    if debug:
        print("original ticks:", sortedticks)
        print("final ticks:", newticks)

    adjustedgradient = {"ticks": newticks, "mode": thestate["mode"]}

    return adjustedgradient


def gen_thermal_state() -> dict:
    """
    Generate and return a thermal gradient dictionary for use in pyqtgraph.

    This function retrieves the built-in "thermal" gradient from pyqtgraph's
    GradientEditorItem and ensures it has the correct name attribute set.

    Returns
    -------
    dict
        A dictionary representing the thermal gradient with the following structure:
        - 'name': str, gradient name ("thermal")
        - Gradient color stops and their positions

    Notes
    -----
    The thermal gradient is a predefined color map that transitions from blue
    through purple to red, commonly used for thermal imaging and heat maps.

    Examples
    --------
    >>> thermal_gradient = gen_thermal_state()
    >>> print(thermal_gradient['name'])
    'thermal'
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["thermal"]
    thegradient["name"] = "thermal"
    return thegradient


def gen_flame_state() -> dict:
    """
    Generate and return a flame color gradient dictionary.

    This function creates a flame-colored gradient using the pyqtgraph GradientEditorItem
    gradients collection. The returned gradient can be used for styling plots, widgets,
    or other graphical elements that support gradient color schemes.

    Returns
    -------
    dict
        A dictionary containing the flame gradient definition with the following structure:
        - 'name': str, gradient name ("flame")
        - Gradient color stops and their corresponding RGB values
        - Other gradient properties as defined in pyqtgraph's GradientEditorItem

    Notes
    -----
    The function relies on pyqtgraph's built-in gradients collection. The "flame" gradient
    typically represents a color scheme transitioning from dark colors to bright orange/red
    colors, resembling a flame or fire effect.

    Examples
    --------
    >>> flame_gradient = gen_flame_state()
    >>> print(flame_gradient['name'])
    'flame'

    See Also
    --------
    pyqtgraph.graphicsItems.GradientEditorItem.Gradients :
        The source of built-in gradients in pyqtgraph
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["flame"]
    thegradient["name"] = "flame"
    return thegradient


def gen_yellowy_state() -> dict:
    """
    Generate and return the yellowy gradient definition.

    This function retrieves the predefined 'yellowy' gradient from pyqtgraph's
    GradientEditorItem Gradients collection and returns it as a dictionary with
    the gradient data and name.

    Returns
    -------
    dict
        A dictionary containing the yellowy gradient definition with the following structure:
        - Keys: Gradient color point definitions
        - 'name': str, the name of the gradient ('yellowy')

    Notes
    -----
    The yellowy gradient is predefined in pyqtgraph's GradientEditorItem module.
    This function provides a convenient way to access and return the gradient data
    for use in plotting and visualization applications.

    Examples
    --------
    >>> gradient = gen_yellowy_state()
    >>> print(gradient['name'])
    'yellowy'
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["yellowy"]
    thegradient["name"] = "yellowy"
    return thegradient


def gen_bipolar_state() -> dict:
    """
    Generate and return the bipolar gradient configuration.

    This function retrieves the bipolar gradient from pyqtgraph's GradientEditorItem
    and returns it as a dictionary with the gradient name set to "bipolar".

    Returns
    -------
    dict
        A dictionary containing the bipolar gradient configuration with the following structure:
        - 'name': str, gradient name ("bipolar")
        - Gradient color stops and values as defined in pyqtgraph's bipolar gradient

    Notes
    -----
    The bipolar gradient is commonly used for visualizing data with both positive and negative values,
    typically displaying negative values in one color (e.g., blue) and positive values in another (e.g., red).
    This gradient is particularly useful for scientific visualization and data analysis applications.

    Examples
    --------
    >>> gradient = gen_bipolar_state()
    >>> print(gradient['name'])
    'bipolar'
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["bipolar"]
    thegradient["name"] = "bipolar"
    return thegradient


def gen_spectrum_state() -> dict:
    """Generate and return the spectrum gradient dictionary.

    This function retrieves the 'spectrum' gradient from pyqtgraph's GradientEditorItem
    and ensures it has the correct name attribute set.

    Returns
    -------
    dict
        A dictionary containing the spectrum gradient definition with keys:
        - 'name': str, set to "spectrum"
        - Gradient color stops and other properties from pyqtgraph's spectrum gradient

    Notes
    -----
    The function relies on pyqtgraph's built-in gradients and specifically uses
    the "spectrum" gradient which provides a rainbow-like color mapping.

    Examples
    --------
    >>> spectrum_grad = gen_spectrum_state()
    >>> print(spectrum_grad['name'])
    'spectrum'
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["spectrum"]
    thegradient["name"] = "spectrum"
    return thegradient


def gen_turbo_state() -> dict:
    """
    Generate and return the turbo color gradient definition.

    This function retrieves the 'turbo' color gradient from pyqtgraph's GradientEditorItem
    and returns it as a dictionary with the gradient data and name.

    Returns
    -------
    dict
        A dictionary containing the turbo color gradient definition with the following structure:
        - 'colorMap': list of color stops
        - 'name': string identifier "turbo"
        - Other gradient properties as defined in pyqtgraph's Gradients

    Notes
    -----
    The turbo gradient is a perceptually uniform color map designed for scientific visualization.
    It provides better color distinction across the entire range of values compared to traditional
    colormaps like 'jet' or 'hsv'.

    Examples
    --------
    >>> gradient = gen_turbo_state()
    >>> print(gradient['name'])
    'turbo'
    >>> print(len(gradient['colorMap']))
    256
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["turbo"]
    thegradient["name"] = "turbo"
    return thegradient


def gen_gray_state() -> dict:
    """
    Generate a gray color state configuration.

    Returns
    -------
    dict
        A dictionary containing color state configuration with the following keys:
        - ticks: List of tuples containing time values and RGBA color tuples
        - mode: Color mode string, currently "rgb"
        - name: Name of the color state, currently "gray"

    Notes
    -----
    This function returns a predefined gray color state configuration with:
    - Start color: Black (0, 0, 0, 255)
    - End color: White (255, 255, 255, 255)
    - Time values: 0.0 and 1.0

    Examples
    --------
    >>> state = gen_gray_state()
    >>> print(state['name'])
    'gray'
    >>> print(state['ticks'])
    [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))]
    """
    return {
        "ticks": [
            (0.0000, (0, 0, 0, 255)),
            (1.0000, (255, 255, 255, 255)),
        ],
        "mode": "rgb",
        "name": "gray",
    }


def gen_grey_state() -> dict:
    """
    Generate a grey color state configuration.

    Returns
    -------
    dict
        A dictionary containing grey color state configuration with the following keys:
        - ticks: list of tuples containing time and RGBA color values
        - mode: string indicating color mode ('rgb')
        - name: string identifier for the color state

    Notes
    -----
    This function returns a predefined grey color state where:
    - Time 0.0 corresponds to black (0, 0, 0, 255)
    - Time 1.0 corresponds to white (255, 255, 255, 255)

    Examples
    --------
    >>> state = gen_grey_state()
    >>> print(state['name'])
    'grey'
    >>> print(state['ticks'])
    [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))]
    """
    return {
        "ticks": [
            (0.0000, (0, 0, 0, 255)),
            (1.0000, (255, 255, 255, 255)),
        ],
        "mode": "rgb",
        "name": "grey",
    }


def gen_viridis_state() -> dict:
    """Generate and return the viridis color gradient configuration.

    This function retrieves the viridis color gradient from pyqtgraph's GradientEditorItem
    and returns it as a dictionary with the gradient data and name.

    Returns
    -------
    dict
        A dictionary containing the viridis color gradient data with the following structure:
        - 'colorMap': list of QColor objects representing the gradient colors
        - 'positions': list of float values representing the color positions (0.0 to 1.0)
        - 'name': str, the name of the gradient ('viridis')

    Notes
    -----
    The viridis gradient is a perceptually uniform colormap designed to be
    visible to colorblind individuals and optimized for scientific visualization.

    Examples
    --------
    >>> gradient_dict = gen_viridis_state()
    >>> print(gradient_dict['name'])
    'viridis'
    >>> print(len(gradient_dict['colorMap']))
    256
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["viridis"]
    thegradient["name"] = "viridis"
    return thegradient


def gen_inferno_state() -> dict:
    """Generate and return the inferno color gradient configuration.

    This function retrieves the inferno color gradient from pyqtgraph's GradientEditorItem
    and returns it as a dictionary with the gradient data and name.

    Returns
    -------
    dict
        A dictionary containing the inferno color gradient data with the following structure:
        - 'name': str, gradient name ("inferno")
        - Gradient color points and their corresponding colors

    Notes
    -----
    The inferno gradient is a perceptually uniform color map designed for scientific visualization.
    It transitions from dark to bright colors, making it suitable for heat maps and other data visualizations.

    Examples
    --------
    >>> gradient = gen_inferno_state()
    >>> print(gradient['name'])
    'inferno'
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["inferno"]
    thegradient["name"] = "inferno"
    return thegradient


def gen_plasma_state() -> dict:
    """
    Generate and return the plasma color gradient definition.

    This function retrieves the plasma gradient from PyQtd's GradientEditorItem
    and returns it as a dictionary with the gradient data and name.

    Returns
    -------
    dict
        Dictionary containing the plasma color gradient definition with keys:
        - 'colorMap': The color mapping data for the gradient
        - 'name': String identifier "plasma"

    Notes
    -----
    The plasma gradient is a popular color scheme that maps values to colors
    in a way that resembles plasma physics visualization. It typically transitions
    from blue through purple to red as values increase.

    Examples
    --------
    >>> gradient = gen_plasma_state()
    >>> print(gradient['name'])
    'plasma'
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["plasma"]
    thegradient["name"] = "plasma"
    return thegradient


def gen_magma_state() -> dict:
    """
    Generate and return the magma color gradient definition.

    This function retrieves the magma color gradient from pyqtgraph's gradient collection
    and returns it as a dictionary with the gradient data and name.

    Returns
    -------
    dict
        A dictionary containing the magma color gradient definition with the following structure:
        - Keys: Color stop positions (float values between 0 and 1)
        - Values: RGBA color tuples (float values between 0 and 1)
        - Additional key: "name" with value "magma"

    Notes
    -----
    The magma gradient is a perceptually uniform colormap designed for scientific visualization.
    It transitions from dark purple to bright yellow, making it suitable for displaying
    data with varying magnitudes.

    Examples
    --------
    >>> magma_gradient = gen_magma_state()
    >>> print(magma_gradient["name"])
    'magma'
    >>> print(len(magma_gradient) > 1)
    True
    """
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    thegradient = Gradients["magma"]
    thegradient["name"] = "magma"
    return thegradient


def gen_g2y2r_state() -> dict:
    """
    Generate a color transition state configuration for green-to-yellow-to-red transition.

    This function returns a dictionary containing color transition data that defines
    a smooth color transition from green to yellow to red, with intermediate states
    and timing information for animation or visualization purposes.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - ticks: list of tuples containing (time, rgba_color) pairs
        - mode: str, color mode identifier ("rgb")
        - name: str, transition name ("g2y2r")

    Notes
    -----
    The color transition follows this sequence:
    - Start: (0, 0, 0, 0) - Transparent black
    - Green: (0, 255, 0, 255) - Pure green
    - Yellow: (255, 255, 0, 255) - Pure yellow
    - Red: (255, 0, 0, 255) - Pure red
    - End: (255, 0, 0, 0) - Transparent red

    Examples
    --------
    >>> state = gen_g2y2r_state()
    >>> print(state['name'])
    'g2y2r'
    >>> print(len(state['ticks']))
    5
    """
    return {
        "ticks": [
            (0.0000, (0, 0, 0, 0)),
            (0.0001, (0, 255, 0, 255)),
            (0.5000, (255, 255, 0, 255)),
            (0.9999, (255, 0, 0, 255)),
            (1.0000, (255, 0, 0, 0)),
        ],
        "mode": "rgb",
        "name": "g2y2r",
    }


def gen_mask_state() -> dict:
    """Generate a mask state configuration dictionary.

    This function creates and returns a dictionary containing mask state
    configuration parameters including tick values, color mode, and name.

    Returns
    -------
    dict
        A dictionary containing mask state configuration with the following keys:
        - ticks: list of tuples containing (time, rgba_color) values
        - mode: string indicating color mode ('rgb')
        - name: string identifier for the mask

    Notes
    -----
    The returned dictionary represents a simple mask state with two ticks:
    - First tick at time 0.0 with black color (0, 0, 0, 255)
    - Second tick at time 1.0 with white color (255, 255, 255, 0)

    Examples
    --------
    >>> mask_state = gen_mask_state()
    >>> print(mask_state['name'])
    'mask'
    >>> print(mask_state['ticks'])
    [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 0))]
    """
    return {
        "ticks": [(0.0000, (0, 0, 0, 255)), (1.0000, (255, 255, 255, 0))],
        "mode": "rgb",
        "name": "mask",
    }


def gen_greyclip_state() -> dict:
    """
    Generate a greyclip color state configuration.

    This function returns a dictionary containing color state information for a greyclip
    color mapping. The color state defines a gradient transition from black to white
    to red, with specific tick points and color values.

    Returns
    -------
    dict
        A dictionary containing the color state configuration with the following keys:
        - ticks: list of tuples containing (position, rgba_color) where position is
          a float between 0.0 and 1.0, and rgba_color is a tuple of (r, g, b, a)
          values in the range 0-255
        - mode: string indicating the color mode, always "rgb" for this function
        - name: string identifier for the color state, always "greyclip"

    Notes
    -----
    The returned color state defines a gradient that starts at black (0, 0, 0, 255),
    transitions to white (255, 255, 255, 255) at 99% position, and ends at red
    (255, 0, 0, 255) at 100% position.

    Examples
    --------
    >>> state = gen_greyclip_state()
    >>> print(state['name'])
    'greyclip'
    >>> print(len(state['ticks']))
    3
    """
    return {
        "ticks": [
            (0.0, (0, 0, 0, 255)),
            (0.99, (255, 255, 255, 255)),
            (1.0, (255, 0, 0, 255)),
        ],
        "mode": "rgb",
        "name": "greyclip",
    }


def getagradient():  # type: ignore
    """Create and return a gradient widget for color selection.

    This function creates a PyQtGraph GradientWidget configured for right-oriented
    gradient editing with the ability to add new color points.

    Returns
    -------
    pyqtgraph.GradientWidget
        A configured GradientWidget instance with:
        - orientation="right" (vertical gradient)
        - allowAdd=True (allows adding new color points)

    Notes
    -----
    The returned widget can be used to create color gradients for visualization
    purposes. The gradient widget allows users to interactively modify color
    stops and create custom color mappings.

    Examples
    --------
    >>> gradient_widget = getagradient()
    >>> # Use the gradient widget in a PyQt application
    >>> layout.addWidget(gradient_widget)
    """
    from pyqtgraph import GradientWidget

    return GradientWidget(orientation="right", allowAdd=True)
