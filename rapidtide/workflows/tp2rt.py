#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026 Blaise Frederick
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
import argparse
from typing import Any

import cv2
import numpy as np

import rapidtide.io as tide_io


def _get_parser() -> Any:
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="tp2rt",
        description="Converts a TelePlethy output file to a rapidtide compatible NIFTI image.",
        allow_abbrev=False,
    )
    parser.add_argument("infilename", help="An mp4 file output by TelePlethy")
    parser.add_argument("outfileroot", help="Root file for NIFTI files")

    parser.add_argument(
        "--framerate", type=float, default=30.0, help="The frame rate of the input video"
    )

    return parser


def main(args):
    # 1. Open the video file using OpenCV
    cap = cv2.VideoCapture(args.infilename)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Convert frame to r, g and b
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        g_frame = rgb_frame[::-1, :, 1]
        frames.append(np.transpose(g_frame))

    cap.release()

    # 3. Stack frames into a 3D numpy array (Height, Width, time)
    # Shape will be (NHeight, Width, Number of frames)
    init_array = np.stack(frames, axis=2)
    initshape = init_array.shape
    print(f"{init_array.shape=}")
    data_array = init_array.reshape(initshape[0], 1, initshape[1], initshape[2])
    print(f"{data_array.shape=}")

    # 4. Create a NIfTI image using NiBabel
    # We use an identity matrix for the affine as MP4 lacks spatial metadata
    thehdr = tide_io.niftihdrfromarray(data_array)
    thehdr.set_xyzt_units(xyz="mm", t="sec")
    thehdr["pixdim"][4] = 1.0 / args.framerate
    tide_io.savetonifti(data_array, thehdr, args.outfileroot)

    # affine = np.eye(4)
    # nifti_img = nib.Nifti1Image(data_array, affine)

    # 5. Save the NIfTI file
    # nib.save(nifti_img, args.outfileroot)
    print(f"Successfully converted {args.infilename} to {args.outfileroot}")
