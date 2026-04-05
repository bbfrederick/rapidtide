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
import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def _get_parser() -> Any:
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="tp2rt",
        description="Converts a TelePlethy output file to a rapidtide compatible NIFTI image.",
        allow_abbrev=False,
    )
    parser.add_argument("infilename", help="An mp4 file output by TelePlethy")
    parser.add_argument("outfilename", help="Root file for NIFTI files")

    return parser


def main(args):
    # 1. Open the video file using OpenCV
    cap = cv2.VideoCapture(args.inputfile)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Convert frame to r, g and b
        # r_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(g_frame)

    cap.release()

    # 3. Stack frames into a 3D numpy array (Depth, Height, Width)
    # Shape will be (NHeight, Width, Number of frames)
    data_array = np.stack(frames, axis=2)

    # 4. Create a NIfTI image using NiBabel
    # We use an identity matrix for the affine as MP4 lacks spatial metadata
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data_array, affine)

    # 5. Save the NIfTI file
    nib.save(nifti_img, args.outputfile)
    print(f"Successfully converted {args.inputfile} to {args.outputfile}")
