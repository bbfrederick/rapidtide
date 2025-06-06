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

# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# vispy: gallery 2

"""
Example volume rendering
Controls:
* 1  - toggle camera between first person (fly), regular 3D (turntable) and
       arcball
* 2  - toggle between volume rendering methods
* 3  - toggle between stent-CT / brain-MRI image
* 4  - toggle between colormaps
* 0  - reset cameras
* [] - decrease/increase isosurface threshold
With fly camera:
* WASD or arrow keys - move around
* SPACE - brake
* FC - move up-down
* IJKL or mouse - look around
"""

import os
import sys
from itertools import cycle

import numpy as np
from vispy import app, scene
from vispy.color import BaseColormap, get_colormaps
from vispy.visuals.transforms import STTransform

import rapidtide.io as tide_io
import rapidtide.stats as tide_stats


def usage():
    print(os.path.basename(sys.argv[0]), "- render 3 (or 4) d images")
    print("")
    print("usage: ", os.path.basename(sys.argv[0]), " niftifile")
    print("")
    print("required arguments:")
    print("    niftifile:                3 or 4 dimensional nifti file containing MR data")
    print("")
    # print("optional arguments:")
    # print("    --debug                    - turn on debugging messages")
    # print("    --glm                      - generate voxelwise aliased synthetic cardiac regressors and filter them out")


def incrementvolume(volumelist):
    global activevol

    volumelist[activevol].visible = False
    activevol = (activevol + 1) % len(volumelist)
    volumelist[activevol].visible = True


### control flow starts here
global activevol
activevol = 0

nargs = len(sys.argv)
if nargs < 2:
    usage()
    exit()
thefilename = sys.argv[1]

# Read volume
nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(thefilename)
xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
vollist = []
if timepoints > 1:
    for i in range(timepoints):
        vollist.append(np.swapaxes(nim_data[:, :, :, i], 0, 2))
else:
    vollist.append(np.swapaxes(nim_data[:, :, :], 0, 2))

# Prepare canvas
canvas = scene.SceneCanvas(keys="interactive", size=(800, 600), show=True)
canvas.measure_fps()

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Set whether we are emulating a 3D texture
emulate_texture = False

# Create the volume visuals, only one is visible
minsize = np.min([xdim, ydim, slicethickness])
volumelist = []
transfacs = (0, 0, 0)
scalefacs = (xdim / minsize, ydim / minsize, slicethickness / minsize)
datarobustmax = tide_stats.getfracvals(nim_data, [0.98])[0]
fracval = 0.2
clims = (datarobustmax * fracval, 2.0 * datarobustmax)

for thevol in vollist:
    volumelist.append(
        scene.visuals.Volume(
            thevol,
            parent=view.scene,
            threshold=0.2,
            emulate_texture=emulate_texture,
            clim=clims,
        )
    )
    volumelist[-1].transform = scene.STTransform(translate=transfacs)
    volumelist[-1].transform = scene.STTransform(scale=scalefacs)
    volumelist[-1].visible = False
volumelist[activevol].visible = True

# Create three cameras (Fly, Turntable and Arcball)
fov = 60.0
cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name="Fly")
cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov, name="Turntable")
cam3 = scene.cameras.ArcballCamera(parent=view.scene, fov=fov, name="Arcball")
view.camera = cam2  # Select turntable at first

# Create an XYZAxis visual
axis = scene.visuals.XYZAxis(parent=view)
s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
affine = s.to_numpy()
axis.transform = affine


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """


# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransFire(), TransGrays()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)


# Implement axis connection with cam2
@canvas.events.mouse_move.connect
def on_mouse_move(event):
    if event.button == 1 and event.is_dragging:
        axis.transform.reset()

        axis.transform.rotate(cam2.roll, (0, 0, 1))
        axis.transform.rotate(cam2.elevation, (1, 0, 0))
        axis.transform.rotate(cam2.azimuth, (0, 1, 0))

        axis.transform.scale((50, 50, 0.001))
        axis.transform.translate((50.0, 50.0))
        axis.update()


# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global opaque_cmap, translucent_cmap, activevol
    if event.text == "1":
        cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
        view.camera = cam_toggle.get(view.camera, cam2)
        print(view.camera.name + " camera")
        if view.camera is cam2:
            axis.visible = True
        else:
            axis.visible = False
    elif event.text == "2":
        methods = ["mip", "translucent", "iso", "additive"]
        method = methods[(methods.index(volumelist[activevol].method) + 1) % 4]
        print("Volume render method: %s" % method)
        cmap = opaque_cmap if method in ["mip", "iso"] else translucent_cmap
        for thevolume in volumelist:
            thevolume.method = method
            thevolume.cmap = cmap
    elif event.text == "3":
        incrementvolume(volumelist)
        # volumelist[activevol].visible = False
        # activevol = (activevol + 1) % len(volumelist)
        # print('activevol set to', activevol)
        # volumelist[activevol].visible = True
    elif event.text == "4":
        if volumelist[activevol].method in ["mip", "iso"]:
            cmap = opaque_cmap = next(opaque_cmaps)
        else:
            cmap = translucent_cmap = next(translucent_cmaps)
        for thevolume in volumelist:
            thevolume.cmap = cmap
    elif event.text == "0":
        cam1.set_range()
        cam3.set_range()
    elif event.text != "" and event.text in "[]":
        s = -0.025 if event.text == "[" else 0.025
        for thevolume in volumelist:
            thevolume.threshold += s
        th = volumelist[activevol].threshold
        print("Isosurface threshold: %0.3f" % th)


# for testing performance
# @canvas.connect
# def on_draw(ev):
# canvas.update()


def entrypoint():
    print(__doc__)
    app.run()


if __name__ == "__main__":
    entrypoint()
