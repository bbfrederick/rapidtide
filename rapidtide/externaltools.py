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
# $Date: 2016/07/11 14:50:43 $
# $Id: rapidtide,v 1.161 2016/07/11 14:50:43 frederic Exp $
#
#
#
import os
import subprocess
from os.path import join as pjoin

import rapidtide.io as tide_io
import rapidtide.util as tide_util

fsldir = os.environ.get("FSLDIR")
if fsldir is not None:
    fslsubcmd = os.path.join(fsldir, "bin", "fsl_sub")
    flirtcmd = os.path.join(fsldir, "bin", "flirt")
    applywarpcmd = os.path.join(fsldir, "bin", "applywarp")
    fslexists = True
else:
    fslexists = False

c3dexists = tide_util.isexecutable("c3d_affine_tool")
print("c3dexists =", c3dexists)
antsexists = tide_util.isexecutable("antsApplyTransforms")
print("antsexists =", antsexists)


def whatexists():
    return fslexists, c3dexists, antsexists


def runcmd(thecmd, fake=False, debug=False):
    if debug:
        print(thecmd)
    if fake:
        print(" ".join(thecmd))
        print()
    else:
        subprocess.call(thecmd)


def makeflirtcmd(inputfile, targetname, xform, outputname, warpfile=None, cluster=False):
    thecommand = []
    if warpfile is None:
        print("doing linear transformation")
        if cluster:
            thecommand.append(fslsubcmd)
        thecommand.append(flirtcmd)
        thecommand.append("-in")
        thecommand.append(inputfile)
        thecommand.append("-ref")
        thecommand.append(targetname)
        thecommand.append("-applyxfm")
        thecommand.append("-init")
        thecommand.append(xform)
        thecommand.append("-out")
        thecommand.append(outputname)
    else:
        print("doing nonlinear transformation")
        if cluster:
            thecommand.append(fslsubcmd)
        thecommand.append(applywarpcmd)
        thecommand.append("--ref=" + targetname)
        thecommand.append("--in=" + inputfile)
        thecommand.append("--out=" + outputname)
        thecommand.append("--warp=" + warpfile)

    return thecommand


def runflirt(inputfile, targetname, xform, outputname, warpfile=None, fake=False, debug=False):
    thecommand = makeflirtcmd(inputfile, targetname, xform, outputname, warpfile=warpfile)
    runcmd(thecommand, fake=fake, debug=debug)


def n4correct(inputfile, outputdir, fake=False, debug=False):
    thename, theext = tide_io.niftisplitext(inputfile)
    n4cmd = []
    n4cmd += ["N4BiasFieldCorrection"]
    n4cmd += ["-d", "3"]
    n4cmd += ["-i", inputfile]
    n4cmd += ["-o", pjoin(outputdir, thename + "_n4" + theext)]
    runcmd(n4cmd, fake=fake, debug=debug)


def antsapply(inputname, targetname, outputroot, transforms, fake=False, debug=False):
    applyxfmcmd = []
    applyxfmcmd += ["antsApplyTransforms"]
    applyxfmcmd += ["--default-value", "0"]
    applyxfmcmd += ["-d", "3"]
    applyxfmcmd += ["-i", inputname]
    applyxfmcmd += ["-o", outputroot]
    applyxfmcmd += ["-r", targetname]
    for thetransform in transforms:
        applyxfmcmd += ["--transform", thetransform]
    runcmd(applyxfmcmd, fake=fake, debug=debug)
