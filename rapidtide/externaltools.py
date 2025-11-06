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
import os
import subprocess
from os.path import join as pjoin

import rapidtide.io as tide_io
import rapidtide.util as tide_util


def fslinfo() -> str | None:
    """
    Get FSL directory path from environment variable.

    This function retrieves the FSL directory path from the FSLDIR environment
    variable. It also sets a global flag `fslexists` to indicate whether FSL
    is available in the current environment.

    Returns
    -------
    str or None
        The path to the FSL directory if FSLDIR environment variable is set,
        otherwise None.

    Notes
    -----
    This function modifies the global variable `fslexists` which is set to
    True if FSLDIR is found, False otherwise. The function does not perform
    any validation of the FSL directory existence or integrity.

    Examples
    --------
    >>> fslinfo()
    '/usr/local/fsl'  # if FSLDIR is set
    >>> fslinfo()
    None  # if FSLDIR is not set
    """
    global fslexists
    fsldir = os.environ.get("FSLDIR")
    if fsldir is not None:
        fslexists = True
    else:
        fslexists = False
    return fsldir


def whatexists() -> tuple[bool, bool, bool]:
    """
    Check for existence of FSL, c3d, and ANTS executables.

    This function checks whether the FSL directory is set in the environment
    and verifies the existence of c3d_affine_tool and antsApplyTransforms
    executables in the system PATH.

    Returns
    -------
    tuple[bool, bool, bool]
        A tuple containing three boolean values:
        - First value: True if FSLDIR environment variable is set, False otherwise
        - Second value: True if c3d_affine_tool executable is found, False otherwise
        - Third value: True if antsApplyTransforms executable is found, False otherwise

    Notes
    -----
    The function uses os.environ.get() to check for FSLDIR and tide_util.isexecutable()
    to check for executable existence. The results are printed to stdout for debugging
    purposes.

    Examples
    --------
    >>> whatexists()
    c3dexists = True
    antsexists = False
    (True, True, False)
    """
    fsldir = os.environ.get("FSLDIR")
    if fsldir is not None:
        fslexists = True
    else:
        fslexists = False

    c3dexists = tide_util.isexecutable("c3d_affine_tool")
    print("c3dexists =", c3dexists)
    antsexists = tide_util.isexecutable("antsApplyTransforms")
    print("antsexists =", antsexists)
    return fslexists, c3dexists, antsexists


def getfslcmds() -> tuple[str | None, str | None, str | None]:
    """
    Get paths to FSL command executables.

    This function retrieves the paths to three core FSL commands (fsl_sub, flirt, and applywarp)
    by looking up the FSLDIR environment variable. If FSLDIR is not set, all returned paths will be None.

    Returns
    -------
    tuple[str | None, str | None, str | None]
        A tuple containing three elements:
        - fslsubcmd: Path to fsl_sub command, or None if FSLDIR is not set
        - flirtcmd: Path to flirt command, or None if FSLDIR is not set
        - applywarpcmd: Path to applywarp command, or None if FSLDIR is not set

    Notes
    -----
    The function expects the FSLDIR environment variable to be set to the FSL installation directory.
    The returned paths are constructed by joining the FSLDIR with "bin/fsl_sub", "bin/flirt", and "bin/applywarp".

    Examples
    --------
    >>> fslsub, flirt, applywarp = getfslcmds()
    >>> print(fslsub)
    '/usr/local/fsl/bin/fsl_sub'  # if FSLDIR is set
    >>> print(flirt)
    '/usr/local/fsl/bin/flirt'    # if FSLDIR is set
    >>> print(applywarp)
    '/usr/local/fsl/bin/applywarp' # if FSLDIR is set
    """
    fsldir = os.environ.get("FSLDIR")
    if fsldir is not None:
        fslsubcmd = os.path.join(fsldir, "bin", "fsl_sub")
        flirtcmd = os.path.join(fsldir, "bin", "flirt")
        applywarpcmd = os.path.join(fsldir, "bin", "applywarp")
    else:
        fslsubcmd = None
        flirtcmd = None
        applywarpcmd = None
    return fslsubcmd, flirtcmd, applywarpcmd


def runcmd(thecmd: list[str], fake: bool = False, debug: bool = False) -> None:
    """
    Execute a command using subprocess.call or simulate execution.

    This function executes a command represented as a list of strings using
    subprocess.call. It can also simulate command execution for testing purposes
    when the 'fake' parameter is set to True.

    Parameters
    ----------
    thecmd : list[str]
        Command to execute, represented as a list of strings where the first
        element is the command and subsequent elements are arguments.
    fake : bool, optional
        If True, print the command that would be executed without actually
        running it. Default is False.
    debug : bool, optional
        If True, print the command before executing or simulating. Default is False.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function uses subprocess.call to execute commands, which means it will
    block until the command completes. The 'fake' mode is useful for testing
    without actually executing commands.

    Examples
    --------
    >>> runcmd(['ls', '-l'])
    # Executes 'ls -l' command

    >>> runcmd(['echo', 'hello'], fake=True)
    # Prints: echo hello
    # But does not actually execute the echo command

    >>> runcmd(['python', '--version'], debug=True)
    # Prints: ['python', '--version']
    # Then executes the command
    """
    if debug:
        print(thecmd)
    if fake:
        print(" ".join(thecmd))
        print()
    else:
        subprocess.call(thecmd)


def makeflirtcmd(
    inputfile: str,
    targetname: str,
    xform: str,
    outputname: str,
    warpfile: str | None = None,
    cluster: bool = False,
    debug: bool = False,
) -> list[str]:
    """
    Create FSL FLIRT command for image registration.

    This function generates a command list for FSL's FLIRT tool to perform either
    linear or nonlinear image registration depending on whether a warp file is provided.

    Parameters
    ----------
    inputfile : str
        Path to the input image file to be registered
    targetname : str
        Path to the target reference image
    xform : str
        Path to the transformation matrix file (for linear registration)
    outputname : str
        Path where the registered output image will be saved
    warpfile : str, optional
        Path to the warp field file for nonlinear registration. If None,
        linear registration is performed (default is None)
    cluster : bool, optional
        If True, prefix the command with fslsub for cluster execution (default is False)
    debug : bool, optional
        If True, print the constructed command for debugging purposes (default is False)

    Returns
    -------
    list[str]
        List of command arguments that can be executed using subprocess or similar

    Notes
    -----
    - For linear registration, the function uses FLIRT with the -applyxfm flag
    - For nonlinear registration, the function uses applywarp instead of FLIRT
    - The function prints transformation type information to stdout

    Examples
    --------
    >>> cmd = makeflirtcmd('input.nii', 'target.nii', 'xform.mat', 'output.nii')
    >>> print(' '.join(cmd))
    flirt -in input.nii -ref target.nii -applyxfm -init xform.mat -out output.nii

    >>> cmd = makeflirtcmd('input.nii', 'target.nii', 'xform.mat', 'output.nii', 'warp.nii')
    >>> print(' '.join(cmd))
    applywarp --ref=target.nii --in=input.nii --out=output.nii --warp=warp.nii
    """
    fslsubcmd, flirtcmd, applywarpcmd = getfslcmds()
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
    if debug:
        print(f"MAKEFILIRTCMD: {thecommand}")
    return thecommand


def runflirt(
    inputfile: str,
    targetname: str,
    xform: str,
    outputname: str,
    warpfile: str | None = None,
    fake: bool = False,
    debug: bool = False,
) -> None:
    """
    Run FLIRT (FMRIB's Linear Image Registration Tool) for image registration.

    This function constructs and executes a FLIRT command for linear image registration
    between an input file and a target file. It supports various transformation types
    and can optionally generate warp fields for non-linear registration.

    Parameters
    ----------
    inputfile : str
        Path to the input image file to be registered
    targetname : str
        Path to the target image file for registration
    xform : str
        Transformation type to use (e.g., 'rigid', 'affine', 'syn')
    outputname : str
        Path where the registered output image will be saved
    warpfile : str, optional
        Path to save the warp field file (default is None)
    fake : bool, optional
        If True, only print the command without executing it (default is False)
    debug : bool, optional
        If True, enable debug mode with additional output (default is False)

    Returns
    -------
    None
        This function does not return any value

    Notes
    -----
    This function relies on the `makeflirtcmd` function to construct the command
    and `runcmd` function to execute it. The actual FLIRT command execution
    is delegated to these helper functions.

    Examples
    --------
    >>> runflirt('input.nii.gz', 'target.nii.gz', 'affine', 'output.nii.gz')
    >>> runflirt('input.nii.gz', 'target.nii.gz', 'rigid', 'output.nii.gz',
    ...          warpfile='warpfield.nii.gz', fake=True)
    """
    thecommand = makeflirtcmd(inputfile, targetname, xform, outputname, warpfile=warpfile)
    runcmd(thecommand, fake=fake, debug=debug)


def n4correct(inputfile: str, outputdir: str, fake: bool = False, debug: bool = False) -> None:
    """
    Apply N4 bias field correction to MRI images.

    This function performs N4 bias field correction using the ANTs N4BiasFieldCorrection
    tool. It corrects intensity inhomogeneities in MRI images by estimating and removing
    the bias field component from the input image.

    Parameters
    ----------
    inputfile : str
        Path to the input NIfTI image file to be corrected
    outputdir : str
        Directory path where the corrected output file will be saved
    fake : bool, optional
        If True, only print the command that would be executed without actually running it
        Default is False
    debug : bool, optional
        If True, enable debug mode to print additional information during execution
        Default is False

    Returns
    -------
    None
        This function does not return any value but saves the corrected image to the
        specified output directory

    Notes
    -----
    The output file will be saved with the naming convention: {input_filename}_n4{file_extension}

    Examples
    --------
    >>> n4correct('subject1.nii.gz', '/path/to/output')
    >>> n4correct('subject1.nii.gz', '/path/to/output', fake=True, debug=True)

    See Also
    --------
    tide_io.niftisplitext : Function used to split filename and extension
    runcmd : Function used to execute the N4 command
    """
    thename, theext = tide_io.niftisplitext(inputfile)
    n4cmd = []
    n4cmd += ["N4BiasFieldCorrection"]
    n4cmd += ["-d", "3"]
    n4cmd += ["-i", inputfile]
    n4cmd += ["-o", pjoin(outputdir, thename + "_n4" + theext)]
    runcmd(n4cmd, fake=fake, debug=debug)


def antsapply(
    inputname: str,
    targetname: str,
    outputroot: str,
    transforms: list[str],
    fake: bool = False,
    debug: bool = False,
) -> None:
    """
    Apply ANTs transforms to an input image using antsApplyTransforms.

    This function constructs and executes an antsApplyTransforms command to apply
    a series of transforms to an input image, resampling it to match a target image.

    Parameters
    ----------
    inputname : str
        Path to the input image to be transformed
    targetname : str
        Path to the target image (reference space) for resampling
    outputroot : str
        Root name for the output file (without extension)
    transforms : list[str]
        List of transform files to apply in order
    fake : bool, optional
        If True, only print the command without executing it (default: False)
    debug : bool, optional
        If True, print additional debug information (default: False)

    Returns
    -------
    None
        This function does not return a value but executes the antsApplyTransforms command

    Notes
    -----
    The function uses antsApplyTransforms with the following fixed parameters:
    - Default value set to 0
    - Dimensionality set to 3
    - Uses the target image as reference for resampling

    Examples
    --------
    >>> antsapply('input.nii.gz', 'target.nii.gz', 'output', ['transform1.nii.gz', 'transform2.nii.gz'])
    >>> antsapply('input.nii.gz', 'target.nii.gz', 'output', ['transform1.nii.gz'], fake=True)
    """
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
