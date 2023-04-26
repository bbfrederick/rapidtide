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
def setendalpha(thestate, alpha, debug=False):
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


def gen_thermal_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["thermal"]


def gen_flame_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["flame"]


def gen_yellowy_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["yellowy"]


def gen_bipolar_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["bipolar"]


def gen_spectrum_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["spectrum"]


def gen_cyclic_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["cyclic"]


def gen_gray_state():
    return {
        "ticks": [
            (0.0000, (0, 0, 0, 255)),
            (1.0000, (255, 255, 255, 255)),
        ],
        "mode": "rgb",
    }


def gen_grey_state():
    return {
        "ticks": [
            (0.0000, (0, 0, 0, 255)),
            (1.0000, (255, 255, 255, 255)),
        ],
        "mode": "rgb",
    }


def gen_viridis_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["viridis"]


def gen_inferno_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["inferno"]


def gen_plasma_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["plasma"]


def gen_magma_state():
    from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

    return Gradients["magma"]


def gen_g2y2r_state():
    return {
        "ticks": [
            (0.0000, (0, 0, 0, 0)),
            (0.0001, (0, 255, 0, 255)),
            (0.5000, (255, 255, 0, 255)),
            (0.9999, (255, 0, 0, 255)),
            (1.0000, (255, 0, 0, 0)),
        ],
        "mode": "rgb",
    }


def gen_mask_state():
    return {
        "ticks": [(0.0000, (0, 0, 0, 255)), (1.0000, (255, 255, 255, 0))],
        "mode": "rgb",
    }


def gen_greyclip_state():
    return {
        "ticks": [
            (0.0, (0, 0, 0, 255)),
            (0.99, (255, 255, 255, 255)),
            (1.0, (255, 0, 0, 255)),
        ],
        "mode": "rgb",
    }


def getagradient():
    from pyqtgraph import GradientWidget

    return GradientWidget(orientation="right", allowAdd=True)
