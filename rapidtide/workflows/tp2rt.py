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
import sys
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

import rapidtide.io as tide_io


def _get_parser() -> Any:
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="tp2rt",
        description="Converts a TelePlethy output file to a rapidtide compatible NIFTI image.",
        allow_abbrev=False,
    )
    parser.add_argument("infilename", help="An mp4 file output by TelePlethy")
    parser.add_argument("tsvfilename", help="A tsv file output by TelePlethy")
    parser.add_argument("outfileroot", help="Root file for NIFTI files")

    parser.add_argument(
        "--framerate", type=float, default=30.0, help="The frame rate of the input video"
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
    # 0. Open the tsv file to get the region location
    samplerate, starttime, columns, regionpos, compressed, filetype = (
        tide_io.readvectorsfromtextfile(f"{args.tsvfilename}:focusX,focusY")
    )

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
    #out_filtframes = cv2.VideoWriter("filtframes.mp4", fourcc, fps, (frame_width, frame_height))
    #out_meanframes = cv2.VideoWriter("meanframes.mp4", fourcc, fps, (frame_width, frame_height))
    #out_grframes = cv2.VideoWriter("GRframes.mp4", fourcc, fps, (frame_width, frame_height))
    #out_gbframes = cv2.VideoWriter("GBframes.mp4", fourcc, fps, (frame_width, frame_height))
    out_grgbframes = cv2.VideoWriter("GRGBframes.mp4", fourcc, fps, (frame_width, frame_height))
    #out_hemoframes = cv2.VideoWriter("hemoframes.mp4", fourcc, fps, (frame_width, frame_height))
    #out_normframes = cv2.VideoWriter("normframes.mp4", fourcc, fps, (frame_width, frame_height))
    #out_colorframes = cv2.VideoWriter("colorframes.mp4", fourcc, fps, (frame_width, frame_height))

    # now make a buffer
    windowsize = 48
    rawframe_buffer = np.zeros(
        (frameshape[0], frameshape[1], frameshape[2], windowsize), dtype=int
    )
    normframe_buffer = np.zeros(
        (frameshape[0], frameshape[1], frameshape[2], windowsize), dtype=float
    )
    colorframe_buffer = np.zeros(
        (frameshape[0], frameshape[1], frameshape[2], windowsize), dtype=float
    )

    # 1. Open the video file using OpenCV
    cap = cv2.VideoCapture(args.infilename)
    filtframes = []
    normframes = []
    grframes = []
    gbframes = []
    grgbframes = []
    colorframes = []
    hemoframes = []

    framenumber = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Convert frame to r, g and b
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        filtframe = gaussian_filter(
            rgb_frame, axes=(0, 1), sigma=7.5
        )
        rawframe_buffer[:, :, :, framenumber % windowsize] = filtframe
        filtframes.append(rawframe_buffer[:, :, :, framenumber % windowsize])
        #out_filtframes.write(cv2.cvtColor(filtframes[-1].astype(np.uint8), cv2.COLOR_RGB2BGR))
        if framenumber < windowsize:
            meanframe = np.mean(rawframe_buffer[:, :, :, : framenumber + 1], axis=3)
        else:
            meanframe = np.mean(rawframe_buffer[:, :, :, :], axis=3)
        print(f"{framenumber=}")
        #showminmax(meanframe, "meanframe")
        # out_meanframes.write(cv2.cvtColor(meanframe.astype(np.uint8), cv2.COLOR_RGB2BGR))
        LOWERTHRESH = 0.2
        UPPERTHRESH = 1.0 / LOWERTHRESH
        normframe_buffer[:, :, :, framenumber % windowsize] = np.nan_to_num(rawframe_buffer[:, :, :, framenumber % windowsize] / meanframe)
        #normframe_buffer[:, :, :, framenumber % windowsize] = np.where(
        #   normframe_buffer[:, :, :, framenumber % windowsize] > LOWERTHRESH,
        #    normframe_buffer[:, :, :, framenumber % windowsize],
        #    0.0,
        #)
        normframes.append(normframe_buffer[:, :, :, framenumber % windowsize])
        #showminmax(normframes[-1], "normframe")
        grframes.append(np.nan_to_num(normframes[-1][:, :, 1] / normframes[-1][:, :, 0]))
        #showminmax(grframes[-1], "GR")
        #out_grframes.write(
        #    cv2.cvtColor((128.0 * grframes[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #)
        gbframes.append(np.nan_to_num(normframes[-1][:, :, 1] / normframes[-1][:, :, 2]))
        #showminmax(gbframes[-1], "GB")
        #out_gbframes.write(
        #    cv2.cvtColor((128.0 * gbframes[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #)
        grgbframes.append(np.transpose((grframes[-1] + gbframes[-1])[::-1,:]))
        #showminmax(grgbframes[-1], "GRGB")
        out_grgbframes.write(
            cv2.cvtColor((64.0 * grgbframes[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        """
        out_normframes.write(
            cv2.cvtColor((192.0 * normframes[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        modulusframe = np.sum(
            normframe_buffer[:, :, :, framenumber % windowsize]
            * normframe_buffer[:, :, :, framenumber % windowsize],
            axis=2,
        )
        colorframe_buffer[:, :, :, framenumber % windowsize] = np.nan_to_num(
            normframe_buffer[:, :, :, framenumber % windowsize] / modulusframe[:, :, None]
        )
        colorframes.append(colorframe_buffer[:, :, :, framenumber % windowsize])
        out_colorframes.write(
            cv2.cvtColor((255.0 * colorframes[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        """
        # cv2.imshow("normframe", cv2.cvtColor((255.0 * normframe_buffer[:, :, :, framenumber % windowsize]).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imshow(
            "normframe",
            cv2.cvtColor((64.0 * normframes[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        cv2.waitKey(1)
        """
        ypos = int(np.round((regionpos[0, framenumber]), 0))
        xpos = int(np.round(rgb_frame.shape[0] - (regionpos[1, framenumber]), 0))
        hemoframe = POSproc(meanframe[::-1, :, :], xpos, ypos)
        hemoframes.append(np.transpose(hemoframe))
        out_hemoframes.write(
            cv2.cvtColor((255.0 * hemoframes[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        """
        framenumber += 1
        print()
        #if framenumber > 300:
        #    break

    cap.release()
    #out_normframes.release()
    #out_colorframes.release()
    #out_hemoframes.release()
    #ut_grframes.release()
    #out_gbframes.release()
    out_grgbframes.release()
    cv2.destroyAllWindows()

    # 3. Stack frames into a 3D numpy array (Height, Width, time)
    # Shape will be (NHeight, Width, Number of frames)
    init_array = np.stack(grgbframes, axis=2)
    initshape = init_array.shape
    print(f"{init_array.shape=}")
    data_array = init_array.reshape(initshape[0], 1, initshape[1], initshape[2])
    data_array_mean = np.mean(data_array, axis=2)
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
