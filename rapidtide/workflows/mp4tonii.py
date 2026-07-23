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
from numpy.typing import NDArray

import rapidtide.io as tide_io


def _get_parser() -> Any:
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="mp4tonii",
        description="Converts an mp4 video to a 4D NIFTI file with color in the z channel.",
        allow_abbrev=False,
    )
    parser.add_argument("infilename", help="An mp4 file output by TelePlethy")
    parser.add_argument("outfileroot", help="Root file for NIFTI files")
    parser.add_argument(
        "--framerate", type=float, default=30.0, help="The frame rate of the input video"
    )
    parser.add_argument(
        "--showframes",
        dest="showframes",
        action="store_true",
        help="Show frames while processing them.",
        default=False,
    )

    return parser


def showminmax(thearray, thelabel):
    themin = np.min(thearray)
    themax = np.max(thearray)
    print(f"\t{thelabel}: {themin=}, {themax=}")


def POSproc(theframe: NDArray, xloc, yloc):

    posframe = theframe[:, :, 1]
    # print(posframe.shape, xloc, yloc)
    posframe[xloc - 5 : xloc + 5, yloc - 5 : yloc + 5] = 255
    return theframe[:, :, 1]


def main(args):
    # first find the shape of the frames
    cap = cv2.VideoCapture(args.infilename)
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameshape = rgb_frame.shape

    # get the video properties, then close the file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"{frame_width=}, {frame_height=}, {fps=}")
    cap.release()

    # set up the output files
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")


    # 1. Open the video file using OpenCV
    cap = cv2.VideoCapture(args.infilename)
    rawframes = []

    framenumber = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Convert frame to r, g and b
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"{rgb_frame.shape=}")
        rawframes.append(np.transpose(rgb_frame[::-1, :], axes=(1, 0, 2)))
        print(f"{rawframes[-1].shape=}")
        print(f"{framenumber=}")

        if args.showframes:
            cv2.imshow(
                "rawframe",
                cv2.cvtColor((rawframes[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
        cv2.waitKey(1)

        framenumber += 1
        print()
        # if framenumber > 300:
        #    break

    cap.release()
    cv2.destroyAllWindows()

    # 3. Stack frames into a 3D numpy array (Height, Width, time)
    # Shape will be (NHeight, Width, Number of frames)
    data_array = np.stack(rawframes, axis=3)
    print(f"{data_array.shape=}")

    # 4. Create a NIfTI image using NiBabel
    # We use an identity matrix for the affine as MP4 lacks spatial metadata
    thehdr = tide_io.niftihdrfromarray(data_array)
    thehdr.set_xyzt_units(xyz="mm", t="sec")
    thehdr["pixdim"][4] = 1.0 / args.framerate
    tide_io.savetonifti(data_array, thehdr, args.outfileroot, nifti2=True)

    print(f"Successfully converted {args.infilename} to {args.outfileroot}")
