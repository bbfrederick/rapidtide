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
import bisect
import logging
import os
import platform
import resource
import site
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import shared_memory
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import rapidtide._version as tide_versioneer
import rapidtide.io as tide_io
from rapidtide.decorators import getdecoratorvars

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False

LGR = logging.getLogger(__name__)
TimingLGR = logging.getLogger("TIMING")
MemoryLGR = logging.getLogger("MEMORY")


# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder: int = 6
MAXLINES: int = 10000000
donotusenumba: bool = False


def disablenumba() -> None:
    """
    Set a global variable to disable numba.

    This function sets the global variable `donotusenumba` to `True`, which
    effectively disables the use of numba in subsequent operations that check
    this variable.

    Notes
    -----
    This function modifies a global variable. The variable `donotusenumba` should
    be checked by other functions in the codebase to determine whether to use
    numba or not.

    Examples
    --------
    >>> disablenumba()
    >>> print(donotusenumba)
    True
    """
    global donotusenumba
    donotusenumba = True


# ----------------------------------------- Conditional imports ---------------------------------------
try:
    import pyfftw
except ImportError:
    pyfftwpresent = False
else:
    pyfftwpresent = True


def checkimports(optiondict: dict[str, Any]) -> None:
    """
    Check availability of optional dependencies and optimization settings.

    This function verifies the presence of optional packages and optimization
    settings, printing status messages and updating the provided dictionary with
    the results. It checks for pyfftw, aggressive optimization flags, and numba
    usage settings.

    Parameters
    ----------
    optiondict : dict[str, Any]
        Dictionary to be updated with boolean values indicating the status of
        optional dependencies and optimization settings. The dictionary will be
        modified in-place with the following keys:

        - "pfftwexists": bool, True if pyfftw is available, False otherwise
        - "donotbeaggressive": bool, True if aggressive optimization is disabled,
          False if enabled
        - "donotusenumba": bool, True if numba usage is disabled, False if numba
          will be used when available

    Returns
    -------
    None
        This function does not return a value but modifies the input dictionary
        in-place.

    Notes
    -----
    The function relies on global variables:
    - `pyfftwpresent`: Indicates if pyfftw is available
    - `donotbeaggressive`: Controls aggressive optimization flag
    - `donotusenumba`: Controls numba usage flag

    Examples
    --------
    >>> options = {}
    >>> checkimports(options)
    pfftw does not exist
    aggressive optimization
    using numba if present
    >>> print(options)
    {'pfftwexists': False, 'donotbeaggressive': False, 'donotusenumba': False}
    """
    donotusenumba, donotbeaggressive = getdecoratorvars()
    if pyfftwpresent:
        print("pfftw exists")
    else:
        print("pfftw does not exist")
    optiondict["pfftwexists"] = pyfftwpresent

    if donotbeaggressive:
        print("no aggressive optimization")
    else:
        print("aggressive optimization")
    optiondict["donotbeaggressive"] = donotbeaggressive

    if donotusenumba:
        print("will not use numba even if present")
    else:
        print("using numba if present")
    optiondict["donotusenumba"] = donotusenumba


def disablemkl(numprocs: int, debug: bool = False) -> None:
    """
    Disable MKL threading for parallel execution.

    This function configures Intel MKL (Math Kernel Library) to use only a single
    thread when the number of processes exceeds 1. This is useful for avoiding
    oversubscription of CPU resources in parallel computing environments.

    Parameters
    ----------
    numprocs : int
        Number of processes to check against. If greater than 1, MKL threading
        will be disabled by setting the number of threads to 1.
    debug : bool, optional
        If True, prints debug information about the threading configuration
        (default is False).

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function only has an effect if MKL is available (mklexists is True).
    The function uses mkl.set_num_threads(1) to disable parallel threading in MKL.

    Examples
    --------
    >>> disablemkl(numprocs=4, debug=True)
    disablemkl: setting threads to 1

    >>> disablemkl(numprocs=1)
    # No output, no threading changes
    """
    if mklexists:
        if numprocs > 1:
            if debug:
                print("disablemkl: setting threads to 1")
            mkl.set_num_threads(1)


def enablemkl(numthreads: int, debug: bool = False) -> None:
    """
    Enable Intel MKL threading with specified number of threads.

    This function configures the Intel MKL (Math Kernel Library) to use the
    specified number of threads for parallel execution. It only has an effect
    if MKL is available in the current environment.

    Parameters
    ----------
    numthreads : int
        Number of threads to use for MKL operations. Must be a positive integer.
    debug : bool, optional
        If True, print debug information about the thread setting operation.
        Default is False.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function only has an effect if MKL is available (mklexists is True).
    The function uses mkl.set_num_threads() internally to configure the threading.

    Examples
    --------
    >>> enablemkl(4)
    >>> enablemkl(8, debug=True)
    """
    if mklexists:
        if debug:
            print(f"enablemkl: setting threads to {numthreads}")
        mkl.set_num_threads(numthreads)


def configurepyfftw(threads: int = 1, debug: bool = False) -> Optional[str]:
    if pyfftwpresent:
        if threads < 1:
            if os.environ.get("PYFFTW_NUM_THREADS") is not None:
                pyfftw.config.NUM_THREADS = os.environ.get("PYFFTW_NUM_THREADS")
        else:
            pyfftw.config.NUM_THREADS = threads

        if os.environ.get("PYFFTW_PLANNER_EFFORT") is None:
            pyfftw.config.PLANNER_EFFORT = "FFTW_ESTIMATE"

        # check for wisdom file, load it if it exist
        wisdomfilename = os.path.join(
            os.environ.get("HOME"), ".config", f"rapidtide_wisdom_{pyfftw.config.PLANNER_EFFORT}.txt"
        )
        if os.path.isfile(wisdomfilename):
            # load the wisdom
            # You need to parse the string
            # For simple cases, eval() can work but is generally not recommended for untrusted input.
            # For more complex cases, manual parsing or using a library like ast.literal_eval is safer.
            with open(wisdomfilename, "r") as file:
                loaded_string = file.read()
                # Example using eval (use with caution)
                thewisdom = eval(loaded_string)
                if debug:
                    print("----------------------Loaded wisdom---------------------------------")
                    print(thewisdom)
                    print("----------------------Loaded wisdom---------------------------------")
                pyfftw.import_wisdom(thewisdom)
                print(f"Loaded pyfftw wisdom from {wisdomfilename}")
        return wisdomfilename
    else:
        return None

def savewisdom(wisdomfilename: str, debug: bool = False) -> None:
    if pyfftwpresent and (wisdomfilename is not None):
        thewisdom = pyfftw.export_wisdom()
        makeadir(os.path.split(wisdomfilename)[0])

        if debug:
            print("----------------------Saved wisdom---------------------------------")
            print(thewisdom)
            print("----------------------Saved wisdom---------------------------------")

        # Save the tuple as a string to a text file
        with open(wisdomfilename, "w") as file:
            file.write(str(thewisdom))

# --------------------------- Utility functions -------------------------------------------------
def findavailablemem() -> tuple[int, int]:
    """
    Get available memory information from system resources.

    This function retrieves memory information from either cgroup limits or system
    free memory statistics. It returns a tuple containing the memory limit and
    swap information, both in bytes.

    Returns
    -------
    tuple[int, int]
        A tuple containing two integers:
        - First integer: Memory limit in bytes (from cgroup or total memory)
        - Second integer: Swap available in bytes (from cgroup or swap memory)

    Notes
    -----
    The function first checks for cgroup memory limits at "/sys/fs/cgroup/memory/memory.limit_in_bytes".
    If found, it returns the limit for both values in the tuple. Otherwise, it uses the "free" command
    to retrieve system memory information, specifically the free memory and swap memory values.

    Examples
    --------
    >>> findavailablemem()
    (8589934592, 2147483648)
    """
    if os.path.isfile("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as limit:
            mem = int(limit.read())
            return mem, mem
    else:
        retdata = subprocess.run(["free", "-m"], capture_output=True).stdout.decode().split("\n")
        free = int((retdata[1].split())[3]) * 1024 * 1024
        swap = int((retdata[2].split())[3]) * 1024 * 1024
        return free, swap


def checkifincontainer() -> str | None:
    """
    Determine if the program is running in a container and identify the container type.

    This function checks environment variables to detect whether the program is running
    inside a container environment. It specifically looks for indicators of Docker,
    Singularity, and CircleCI environments. The function returns the container type
    as a string, or None if running outside any container.

    Returns
    -------
    str or None
        Container type if running in a container, otherwise None. Possible return values:
        - "Docker": Running in a Docker container (indicated by RUNNING_IN_CONTAINER env var)
        - "Singularity": Running in a Singularity container (indicated by SINGULARITY_CONTAINER env var)
        - "CircleCI": Running in CircleCI environment (indicated by CIRCLECI env var)
        - None: Not running in any container environment

    Notes
    -----
    The function prioritizes detection in the following order:
    1. Singularity containers (SINGULARITY_CONTAINER env var)
    2. Docker containers (RUNNING_IN_CONTAINER env var)
    3. CircleCI environment (CIRCLECI env var)

    CircleCI detection takes precedence over other container types, as CircleCI
    environments may not handle container parameter adjustments properly.

    Examples
    --------
    >>> checkifincontainer()
    'Docker'

    >>> checkifincontainer()
    'Singularity'

    >>> checkifincontainer()
    None
    """
    if os.environ.get("SINGULARITY_CONTAINER") is not None:
        containertype = "Singularity"
    elif os.environ.get("RUNNING_IN_CONTAINER") is not None:
        containertype = "Docker"
    else:
        containertype = None
    if os.environ.get("CIRCLECI") is not None:
        containertype = "CircleCI"
    return containertype


def setmemlimit(memlimit: int) -> None:
    """
    Set the memory limit for the current process.

    This function sets the virtual memory limit (RLIMIT_AS) for the current process
    using the resource module. The limit is specified in bytes and applies to both
    soft and hard limits.

    Parameters
    ----------
    memlimit : int
        The memory limit in bytes. Setting this to -1 will remove the limit.
        Values should be non-negative integers.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    - This function uses `resource.setrlimit()` with `resource.RLIMIT_AS`
    - The memory limit is enforced by the operating system
    - Setting memlimit to -1 removes any existing memory limit
    - This function may raise `ValueError` or `OSError` if the limit cannot be set
    - The limit applies to the current process and its children

    Examples
    --------
    >>> setmemlimit(1024 * 1024 * 100)  # Set limit to 100 MB
    >>> setmemlimit(-1)                 # Remove memory limit
    """
    resource.setrlimit(resource.RLIMIT_AS, (memlimit, memlimit))


def formatmemamt(meminbytes: int) -> str:
    """
    Format memory amount in bytes to human readable format.

    Convert a memory size in bytes to a human readable string with appropriate units
    (B, kB, MB, GB, TB).

    Parameters
    ----------
    meminbytes : int
        Memory amount in bytes to be formatted.

    Returns
    -------
    str
        Formatted memory amount with appropriate unit. The result is rounded to 3
        decimal places and includes the unit suffix.

    Notes
    -----
    The function uses binary units (1024-based) rather than decimal units (1000-based).
    Units are: B (bytes), kB (kilobytes), MB (megabytes), GB (gigabytes), TB (terabytes).

    Examples
    --------
    >>> formatmemamt(1024)
    '1.000kB'

    >>> formatmemamt(1048576)
    '1.000MB'

    >>> formatmemamt(1073741824)
    '1.000GB'
    """
    units = ["B", "kB", "MB", "GB", "TB"]
    index = 0
    unitnumber = np.uint64(1)
    while True:
        if meminbytes < np.uint64(1024) * unitnumber:
            return f"{round(meminbytes/unitnumber, 3):.3f}{units[index]}"
        unitnumber *= np.uint64(1024)
        index += 1
        if index >= len(units):
            break
    return f"{round(meminbytes/unitnumber, 3):.3f}{units[-1]}"


def format_bytes(size: float) -> tuple[float, str]:
    """
    Convert a size in bytes to a human-readable format with appropriate units.

    Convert a size in bytes to a more readable format by scaling it to the
    appropriate unit (bytes, kilobytes, megabytes, gigabytes, terabytes).

    Parameters
    ----------
    size : float
        The size in bytes to be converted. Should be a non-negative number.

    Returns
    -------
    tuple[float, str]
        A tuple containing the scaled size (float) and the corresponding unit (str).
        The unit will be one of: 'bytes', 'kilobytes', 'megabytes', 'gigabytes', 'terabytes'.

    Notes
    -----
    This function uses base-2 (binary) units where 1 kilobyte = 1024 bytes.
    The conversion continues until the size is less than 1024, at which point
    the appropriate unit is returned.

    Examples
    --------
    >>> format_bytes(512)
    (512.0, 'bytes')

    >>> format_bytes(2048)
    (2.0, 'kilobytes')

    >>> format_bytes(1048576)
    (1.0, 'megabytes')

    >>> format_bytes(1073741824)
    (1.0, 'gigabytes')
    """
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "kilo", 2: "mega", 3: "giga", 4: "tera"}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n] + "bytes"


def logmem(msg: str | None = None) -> None:
    """
    Log memory usage with a logging object.

    This function logs detailed memory usage statistics for the current process
    and its children, including resident set size (RSS), shared and unshared memory,
    page faults, and swap usage. On Windows, memory statistics are not available
    and a placeholder message is logged instead.

    Parameters
    ----------
    msg : str, optional
        A message to include in the first column of the logged output.
        If None, column headers are logged instead. Default is None.

    Returns
    -------
    None
        This function does not return any value; it logs information to a global
        logger named `MemoryLGR`.

    Notes
    -----
    - On Unix-like systems (Linux, macOS), this function uses `resource.getrusage`
      to retrieve memory usage details.
    - On Windows, memory statistics are not supported and a placeholder message
      is logged.
    - The function maintains internal state (`lastmaxrss_parent`, `lastmaxrss_child`)
      to compute differences in memory usage between calls.

    Examples
    --------
    >>> logmem("Before loop")
    # Logs memory usage with "Before loop" as the first column

    >>> logmem()
    # Logs column headers for memory usage statistics
    """
    global lastmaxrss_parent, lastmaxrss_child
    if platform.system() != "Windows":
        import resource

        if msg is None:
            outvals = [
                "",
                "Self Max RSS",
                "Self Diff RSS",
                "Self Shared Mem",
                "Self Unshared Mem",
                "Self Unshared Stack",
                "Self Non IO Page Fault",
                "Self IO Page Fault",
                "Self Swap Out",
                "Children Max RSS",
                "Children Diff RSS",
                "Children Shared Mem",
                "Children Unshared Mem",
                "Children Unshared Stack",
                "Children Non IO Page Fault",
                "Children IO Page Fault",
                "Children Swap Out",
            ]
            lastmaxrss_parent = 0
            lastmaxrss_child = 0
        else:
            rcusage = resource.getrusage(resource.RUSAGE_SELF)
            outvals = [msg]
            outvals.append(formatmemamt(rcusage.ru_maxrss))
            outvals.append(formatmemamt(rcusage.ru_maxrss - lastmaxrss_parent))
            lastmaxrss_parent = rcusage.ru_maxrss
            outvals.append(formatmemamt(rcusage.ru_ixrss))
            outvals.append(formatmemamt(rcusage.ru_idrss))
            outvals.append(formatmemamt(rcusage.ru_isrss))
            outvals.append(formatmemamt(rcusage.ru_minflt))
            outvals.append(formatmemamt(rcusage.ru_majflt))
            outvals.append(formatmemamt(rcusage.ru_nswap))
            rcusage = resource.getrusage(resource.RUSAGE_CHILDREN)
            outvals.append(formatmemamt(rcusage.ru_maxrss))
            outvals.append(formatmemamt(rcusage.ru_maxrss - lastmaxrss_child))
            lastmaxrss_child = rcusage.ru_maxrss
            outvals.append(formatmemamt(rcusage.ru_ixrss))
            outvals.append(formatmemamt(rcusage.ru_idrss))
            outvals.append(formatmemamt(rcusage.ru_isrss))
            outvals.append(formatmemamt(rcusage.ru_minflt))
            outvals.append(formatmemamt(rcusage.ru_majflt))
            outvals.append(formatmemamt(rcusage.ru_nswap))
    else:
        outvals = ["Not available on Windows"]

    MemoryLGR.info("\t".join(outvals))


def findexecutable(command: str) -> str | None:
    """
    Locate an executable file in the system PATH.

    This function searches for an executable file with the given name in the
    system's PATH environment variable. It uses the most appropriate method
    based on the Python version.

    Parameters
    ----------
    command : str
        The name of the executable command to search for.

    Returns
    -------
    str or None
        The full path to the executable if found, None otherwise.

    Notes
    -----
    For Python 3.3 and later, this function uses `shutil.which()` which is the
    recommended approach. For earlier Python versions, it manually searches
    through the PATH environment variable and checks execute permissions.

    Examples
    --------
    >>> findexecutable('python')
    '/usr/bin/python'

    >>> findexecutable('nonexistent_command')
    None
    """
    import shutil

    theversion = sys.version_info
    if (theversion[0] >= 3) and (theversion[1] >= 3):
        return shutil.which(command)
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            if os.access(os.path.join(path, command), os.X_OK):
                return os.path.join(path, command)
        return None


def isexecutable(command: str) -> bool:
    """
    Check if a command is executable in the system's PATH.

    This function determines whether a given command can be executed by checking
    if it exists in the system's PATH and has execute permissions. For Python 3.3+
    the function uses shutil.which() for cross-platform compatibility, while for
    older versions it manually checks execute permissions in each PATH directory.

    Parameters
    ----------
    command : str
        The name of the command to check for executability.

    Returns
    -------
    bool
        True if the command is executable, False otherwise.

    Notes
    -----
    This function provides cross-platform compatibility by using different
    approaches depending on the Python version. For Python 3.3 and later,
    shutil.which() is used which handles platform-specific path searching.
    For older Python versions, the function manually checks execute permissions
    in each directory listed in the PATH environment variable.

    Examples
    --------
    >>> isexecutable('python')
    True
    >>> isexecutable('nonexistent_command')
    False
    """
    import shutil

    theversion = sys.version_info
    if (theversion[0] >= 3) and (theversion[1] >= 3):
        if shutil.which(command) is not None:
            return True
        else:
            return False
    else:
        return any(
            os.access(os.path.join(path, command), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )


def makeadir(pathname: str) -> bool:
    """
    Create a directory if it doesn't already exist.

    This function attempts to create a directory at the specified path. If the
    directory already exists, it returns True without raising an error. If the
    directory cannot be created due to a permissions error or other OS-related
    issues, it returns False.

    Parameters
    ----------
    pathname : str
        The path of the directory to create. Can be a relative or absolute path.

    Returns
    -------
    bool
        True if the directory exists or was successfully created, False otherwise.

    Notes
    -----
    This function uses `os.makedirs()` which creates all intermediate-level
    directories needed to contain the leaf directory. If the directory already
    exists, no error is raised.

    Examples
    --------
    >>> makeadir('test_directory')
    True

    >>> makeadir('path/to/new/directory')
    True

    >>> makeadir('/root/protected_directory')
    False  # Will fail due to insufficient permissions
    """
    try:
        os.makedirs(pathname)
    except OSError:
        if os.path.exists(pathname):
            # We are nearly safe
            return True
        else:
            # There was an error on creation, so make sure we know about it
            print("ERROR: ", pathname, " does not exist, and could not create it")
            return False
    return True


def findreferencedir() -> str:
    """
    Find and return the path to the rapidtide reference data directory.

    This function locates the site-packages directory and constructs the path to
    the rapidtide reference data folder. It searches through all site-packages
    directories to find the one ending with "site-packages" and then builds
    the reference directory path relative to that location.

    Returns
    -------
    str
        Absolute path to the rapidtide reference data directory, typically
        structured as: {site-packages-dir}/rapidtide/data/reference/

    Notes
    -----
    This function is designed to work within the rapidtide package environment
    and assumes that rapidtide is installed in a standard Python site-packages
    location. The function will return None if no site-packages directory is found.

    Examples
    --------
    >>> ref_dir = findreferencedir()
    >>> print(ref_dir)
    '/usr/local/lib/python3.8/site-packages/rapidtide/data/reference'
    """
    # Get the list of directories
    site_packages_dirs = site.getsitepackages()

    # Find the "site-packages" directory in the list
    for dir in site_packages_dirs:
        if dir.endswith("site-packages"):
            sitepackages_dir = dir
            break
        else:
            sitepackages_dir = None
    referencedir = os.path.join(
        sitepackages_dir,
        "rapidtide",
        "data",
        "reference",
    )
    return referencedir


def savecommandline(theargs: list[str], thename: str) -> None:
    """
    Save command line arguments to a text file.

    This function takes a list of command line arguments and saves them
    as a single line in a text file with a specified name.

    Parameters
    ----------
    theargs : list[str]
        List of command line arguments to be saved
    thename : str
        Base name for the output file (without extension)

    Returns
    -------
    None
        This function does not return any value

    Notes
    -----
    The function creates a file named ``{thename}_commandline.txt`` containing
    the command line arguments joined by spaces on a single line.

    Examples
    --------
    >>> savecommandline(['python', 'script.py', '--verbose'], 'myrun')
    # Creates file 'myrun_commandline.txt' with content: "python script.py --verbose"
    """
    tide_io.writevec([" ".join(theargs)], thename + "_commandline.txt")


def startendcheck(timepoints: int, startpoint: int, endpoint: int) -> tuple[int, int]:
    """
    Validate and adjust start and end points for time series processing.

    This function checks if the provided start and end points are within valid
    bounds for a time series with the specified number of time points. It handles
    edge cases by adjusting values to reasonable defaults and raises errors for
    invalid configurations.

    Parameters
    ----------
    timepoints : int
        Total number of time points in the series. Must be positive.
    startpoint : int
        Starting index for the time series segment. If negative, set to 0.
        If greater than timepoints-1, the program exits with an error.
    endpoint : int
        Ending index for the time series segment. If -1, set to a large default value.
        If greater than timepoints-1, set to timepoints-1.

    Returns
    -------
    tuple[int, int]
        A tuple containing (realstart, realend) where both values are valid
        indices for the time series. realstart <= realend and both are within
        the valid range [0, timepoints-1].

    Notes
    -----
    - If startpoint is negative, it's automatically set to 0
    - If endpoint is -1, it's set to 100000000 (large default value)
    - If endpoint exceeds timepoints-1, it's set to timepoints-1
    - The function exits with sys.exit() if startpoint >= endpoint or if
      startpoint exceeds the maximum valid index

    Examples
    --------
    >>> startendcheck(10, 2, 5)
    startpoint set to 2
    endpoint set to 5
    (2, 5)

    >>> startendcheck(5, -1, 3)
    startpoint set to minimum, (0)
    endpoint set to 3
    (0, 3)

    >>> startendcheck(5, 2, -1)
    startpoint set to 2
    endpoint set to maximum, (4)
    (2, 4)
    """
    if startpoint > timepoints - 1:
        print("startpoint is too large (maximum is ", timepoints - 1, ")")
        sys.exit()
    if startpoint < 0:
        realstart = 0
        print("startpoint set to minimum, (0)")
    else:
        realstart = startpoint
        print("startpoint set to ", startpoint)
    if endpoint == -1:
        endpoint = 100000000
    if endpoint > timepoints - 1:
        realend = timepoints - 1
        print("endpoint set to maximum, (", timepoints - 1, ")")
    else:
        realend = endpoint
        print("endpoint set to ", endpoint)
    if realstart >= realend:
        print("endpoint (", realend, ") must be greater than startpoint (", realstart, ")")
        sys.exit()
    return realstart, realend


def valtoindex(
    thearray: NDArray,
    thevalue: float,
    evenspacing: bool = True,
    discrete: bool = True,
    discretization: str = "round",
    debug: bool = False,
) -> int:
    """
    Find the index of the closest value in an ordered array to a given value.

    This function computes the index of the element in `thearray` that is closest
    to `thevalue`. It supports both evenly spaced and unevenly spaced arrays,
    with options for discrete or continuous index output and different rounding
    methods.

    Parameters
    ----------
    thearray : array-like
        An ordered list of values (does not need to be equally spaced).
    thevalue : float
        The value to search for in the array.
    evenspacing : bool, optional
        If True (default), assume data is evenly spaced for faster calculation.
    discrete : bool, optional
        If True (default), the returned index is an integer.
    discretization : str, optional
        Select rounding method when `discrete=True`. Options are:
        - "round" (default): round to nearest integer
        - "floor": round down to nearest integer
        - "ceiling": round up to nearest integer
    debug : bool, optional
        If True, print debug information during execution.

    Returns
    -------
    int or float
        The index of the closest value in `thearray` to `thevalue`. If `discrete=False`,
        the index may be a float.

    Notes
    -----
    When `evenspacing=True`, the function assumes uniform spacing between elements
    and calculates the index using a linear interpolation formula. This is faster
    than the default method but only accurate for evenly spaced data.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([0, 1, 2, 3, 4])
    >>> valtoindex(arr, 2.3)
    2
    >>> valtoindex(arr, 2.7, discretization="ceil")
    3
    >>> valtoindex(arr, 2.5, evenspacing=False)
    2
    """
    if evenspacing:
        limval = np.max([thearray[0], np.min([thearray[-1], thevalue])])
        position = (limval - thearray[0]) / (thearray[1] - thearray[0])
        if debug:
            print("valtoindex:")
            print("\tthevalue:", thevalue)
            print("\tarraymin:", thearray[0])
            print("\tarraymax:", thearray[-1])
            print("\tlimval:", limval)
            print(
                "\tindex:",
                int(np.round((limval - thearray[0]) / (thearray[1] - thearray[0]), 0)),
            )
        if discrete:
            if discretization == "round":
                position = int(np.round(position, 0))
            elif discretization == "floor":
                position = int(np.floor(position))
            elif discretization == "ceiling":
                position = int(np.ceil(position))
            else:
                print("valtoindex - illegal discretization mode")
                position = None
            position = int(np.min([len(thearray) - 1, np.max([0, position])]))
        return position
    else:
        return int((np.abs(thearray - thevalue)).argmin())


def progressbar(thisval: int, end_val: int, label: str = "Percent", barsize: int = 60) -> None:
    """
    Display a progress bar in the terminal.

    This function creates a visual progress indicator that updates in place
    on the terminal. It shows a bar filled according to the progress percentage
    and displays the percentage value.

    Parameters
    ----------
    thisval : int
        Current progress value. Should be less than or equal to ``end_val``.
    end_val : int
        Total value representing 100% progress.
    label : str, optional
        Label to display before the progress bar (default is "Percent").
    barsize : int, optional
        Size of the progress bar in characters (default is 60).

    Returns
    -------
    None
        This function does not return any value. It prints directly to stdout.

    Notes
    -----
    The progress bar updates in place using carriage return (`\\r`) to overwrite
    the previous output. The function uses ``sys.stdout.flush()`` to ensure
    immediate display updates.

    Examples
    --------
    >>> progressbar(25, 100, "Loading", 30)
    Loading: [##############################                          ] 25.00%

    >>> progressbar(50, 50)
    Percent: [##################################################] 100.00%
    """
    percent = float(thisval) / end_val
    hashes = "#" * int(round(percent * barsize))
    spaces = " " * (barsize - len(hashes))
    sys.stdout.write("\r{0}: [{1}] {2:.2f}%".format(label, hashes + spaces, 100.0 * percent))
    sys.stdout.flush()


def makelaglist(lagstart: float, lagend: float, lagstep: float) -> NDArray:
    """
    Create a list of lag values from start to end with specified step size.

    This function generates an array of evenly spaced lag values starting from
    `lagstart` up to (and including) `lagend` with increments of `lagstep`.

    Parameters
    ----------
    lagstart : float
        The starting value of the lag sequence.
    lagend : float
        The ending value of the lag sequence (inclusive).
    lagstep : float
        The step size between consecutive lag values.

    Returns
    -------
    NDArray
        Array of lag values from `lagstart` to `lagend` with step size `lagstep`.

    Notes
    -----
    The function adjusts the `lagend` value to ensure that the last value in the
    sequence is exactly `lagend` if it's a valid step from `lagstart`. The actual
    number of steps is calculated as ``(lagend - lagstart) // lagstep + 1``.

    Examples
    --------
    >>> makelaglist(0.0, 1.0, 0.2)
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])

    >>> makelaglist(1.0, 5.0, 1.5)
    array([1. , 2.5, 4. ])
    """
    numsteps = int((lagend - lagstart) // lagstep + 1)
    lagend = lagstart + lagstep * (numsteps - 1)
    print(
        "creating list of ",
        numsteps,
        " lag steps (",
        lagstart,
        " to ",
        lagend,
        " in steps of ",
        lagstep,
        ")",
    )
    # thelags = np.r_[0.0:1.0 * numsteps] * lagstep + lagstart
    thelags = np.arange(0.0, 1.0 * numsteps) * lagstep + lagstart
    return thelags


# ------------------------------------------ Version function ----------------------------------
def version() -> tuple[str, str, str, bool | str]:
    """
    Retrieve version information for the package, including version string,
    Git SHA, commit date, and dirty status.

    This function attempts to retrieve version information from environment
    variables when running inside a container. If not in a container, it falls
    back to using `tide_versioneer.get_versions()` to obtain version details
    from the Git repository.

    Returns
    -------
    tuple of (str, str, str, bool or str)
        A tuple containing:
        - version (str): The version string, potentially modified for container builds.
        - sha (str): The Git commit SHA, or "UNKNOWN" if not available.
        - date (str): The Git commit date, or "UNKNOWN" if not available.
        - isdirty (bool or str): Indicates whether the working directory is dirty
          (i.e., has uncommitted changes). Returns `True`, `False`, or `"UNKNOWN"`
          if the information is not available.

    Notes
    -----
    - In containerized environments, version information is expected to be
      provided via environment variables: `GITVERSION`, `GITDIRECTVERSION`,
      `GITSHA`, and `GITDATE`.
    - If the environment variable `RUNNING_IN_CONTAINER` is not set, the function
      attempts to use `tide_versioneer` to extract version information from the
      Git repository.
    - If `tide_versioneer` is not available or fails, the function returns
      `("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN")`.

    Examples
    --------
    >>> version()
    ('1.2.3', 'a1b2c3d', '2023-04-05', False)
    """
    try:
        dummy = os.environ["RUNNING_IN_CONTAINER"]
    except KeyError:
        iscontainer = False
    else:
        iscontainer = True

    if iscontainer:
        try:
            theversion = os.environ["GITVERSION"]
            if theversion.find("+") < 0:
                theverion = theversion.split(".")[0]
        except KeyError:
            theversion = "UNKNOWN"
        try:
            thedirectversion = os.environ["GITDIRECTVERSION"]
            directversionparts = thedirectversion.split("-")
            if len(directversionparts) == 3:
                thedirectversion = (
                    directversionparts[0]
                    + "."
                    + directversionparts[1]
                    + "+"
                    + directversionparts[2]
                )
                isdirty = True
            elif len(directversionparts) == 2:
                thedirectversion = directversionparts[0] + "." + directversionparts[1]
                isdirty = True
            elif len(directversionparts) == 1:
                thedirectversion = directversionparts[0]
                isdirty = False
            else:
                pass
        except KeyError:
            thedirectversion = "UNKNOWN"
        try:
            thesha = os.environ["GITSHA"]
        except KeyError:
            thesha = "UNKNOWN"
        try:
            thedate = os.environ["GITDATE"]
        except KeyError:
            thedate = "UNKNOWN"
        if thedirectversion != "UNKNOWN":
            theversion = thedirectversion
    else:
        try:
            versioninfo = tide_versioneer.get_versions()
        except:
            return "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"
        isdirty = versioninfo["dirty"]
        if isdirty is None:
            isdirty = "UNKNOWN"
        theversion = versioninfo["version"]
        if theversion is None:
            theversion = "UNKNOWN"
        else:
            splitversion = theversion.split("+")
            if len(splitversion) > 1:
                resplit = splitversion[1].split(".")
                if len(resplit) == 3:
                    if resplit[0] == "0":
                        theversion = splitversion[0]
        thesha = versioninfo["full-revisionid"]
        if thesha is None:
            thesha = "UNKNOWN"
        thedate = versioninfo["date"]
        if thedate is None:
            thedate = "UNKNOWN"

    return theversion, thesha, thedate, isdirty


# --------------------------- timing functions -------------------------------------------------
def timefmt(thenumber: float) -> str:
    """
    Format a floating-point number as a string with fixed width and 2 decimal places.

    Parameters
    ----------
    thenumber : float
        The numeric value to be formatted as a string.

    Returns
    -------
    str
        A string representation of the input number formatted to 2 decimal places
        with a minimum width of 10 characters, right-aligned.

    Notes
    -----
    The formatting uses "{:10.2f}".format() which ensures:
    - Fixed width of 10 characters
    - 2 decimal places
    - Right alignment (default for numeric formats)

    Examples
    --------
    >>> timefmt(123.456)
    '   123.46'
    >>> timefmt(1.234)
    '     1.23'
    >>> timefmt(0.0)
    '     0.00'
    """
    return "{:10.2f}".format(thenumber)


def proctiminglogfile(logfilename: str, timewidth: int = 10) -> tuple[list[str], float]:
    """
    Process a timing log file and return formatted timing information.

    This function reads a timing log file, calculates cumulative and incremental
    time differences from the start time, and formats the output into a list of
    strings. If numerical data and units are present in the log, they are used
    to compute and display processing speeds.

    Parameters
    ----------
    logfilename : str
        Path to the timing log file. The file should be a CSV with columns:
        'time', 'description', 'number', 'units'.
    timewidth : int, optional
        Width for right-justifying time values in the output (default is 10).

    Returns
    -------
    tuple[list[str], float]
        A tuple containing:
        - List of formatted timing lines as strings.
        - Total elapsed time in seconds as a float.

    Notes
    -----
    The log file is expected to follow the format:
    `YYYYMMDDTHHMMSS.ffffff` for timestamps.
    The function assumes the first row is the starting point for all time calculations.

    Examples
    --------
    >>> lines, total_time = proctiminglogfile('timing.log', timewidth=12)
    >>> for line in lines:
    ...     print(line)
    Total (s)    Diff. (s)    Description
    0.00         0.00         Start process
    5.20         5.20         Load data
    12.40        7.20         Process data (1000 items @ 138.89 items/s)
    """
    timingdata = pd.read_csv(
        logfilename,
        sep=None,
        header=None,
        names=["time", "description", "number", "units"],
        engine="python",
    )
    starttime = datetime.strptime(timingdata["time"].iloc[0], "%Y%m%dT%H%M%S.%f")
    outputlines = [f"{'Total (s)'.rjust(timewidth)}\t{'Diff. (s)'.rjust(timewidth)}\tDescription"]
    outputlines += [
        f"{'0.0'.rjust(timewidth)}\t{'0.0'.rjust(timewidth)}\t{timingdata['description'].iloc[0]}"
    ]
    for therow in range(1, timingdata.shape[0]):
        thistime = datetime.strptime(timingdata["time"].iloc[therow], "%Y%m%dT%H%M%S.%f")
        prevtime = datetime.strptime(timingdata["time"].iloc[therow - 1], "%Y%m%dT%H%M%S.%f")
        totaldiff = (thistime - starttime).total_seconds()
        incdiff = (thistime - prevtime).total_seconds()
        totaldiffstr = f"{totaldiff:.2f}".rjust(timewidth)
        incdiffstr = f"{incdiff:.2f}".rjust(timewidth)
        theoutputline = f"{totaldiffstr}\t{incdiffstr}\t{timingdata['description'].iloc[therow]}"
        try:
            dummy = np.isnan(timingdata["number"].iloc[therow])
        except:
            pass
        else:
            if not np.isnan(timingdata["number"].iloc[therow]):
                speedunit = f"{timingdata['units'].iloc[therow]}/s"
                if incdiff == 0.0:
                    speed = "undefined"
                else:
                    speed = f"{float(timingdata['number'].iloc[therow]) / incdiff:.2f}"
                theoutputline += f" ({timingdata['number'].iloc[therow]} {timingdata['units'].iloc[therow]} @ {speed} {speedunit})"
        outputlines += [theoutputline]

    return outputlines, totaldiff


def proctiminginfo(
    thetimings: list[tuple[str, float, float | None, str | None]],
    outputfile: str = "",
    extraheader: str | None = None,
) -> None:
    """
    Process and display timing information for program execution.

    This function takes a list of timing events and displays them in a formatted table
    showing clock time, program time, duration, and event descriptions. Optional
    output to file and additional header information can also be specified.

    Parameters
    ----------
    thetimings : list of tuple of (str, float, float | None, str | None)
        List of timing events where each event is a tuple containing:
        - Event description (str)
        - Timestamp (float)
        - Events per second (float or None)
        - Unit of measurement (str or None)
    outputfile : str, optional
        Path to output file for writing timing information (default is "")
    extraheader : str, optional
        Additional header text to be printed before timing information (default is None)

    Returns
    -------
    None
        This function does not return any value but prints timing information to stdout
        and optionally writes to a file.

    Notes
    -----
    The function formats timestamps using YYYYMMDDTHHMMSS format and calculates
    durations between consecutive events. If event rate information is provided,
    it will be displayed in the format "(rate unit/second)".

    Examples
    --------
    >>> timings = [
    ...     ("Start", 1640995200.0, None, None),
    ...     ("Process A", 1640995205.5, 100.0, "events"),
    ...     ("End", 1640995210.0, None, None)
    ... ]
    >>> proctiminginfo(timings, "timing_output.txt", "Execution Timing Report")
    """
    theinfolist = []
    start = thetimings[0]
    starttime = float(start[1])
    lasteventtime = starttime
    if extraheader is not None:
        print(extraheader)
        theinfolist.append(extraheader)
    headerstring = "Clock time\tProgram time\tDuration\tDescription"
    print(headerstring)
    theinfolist.append(headerstring)
    for theevent in thetimings:
        theduration = float(theevent[1] - lasteventtime)
        outstring = (
            time.strftime("%Y%m%dT%H%M%S", time.localtime(theevent[1]))
            + timefmt(float(theevent[1]) - starttime)
            + "\t"
            + timefmt(theduration)
            + "\t"
            + theevent[0]
        )
        if theevent[2] is not None:
            outstring += " ({0:.2f} {1}/second)".format(
                float(theevent[2]) / theduration, theevent[3]
            )
        print(outstring)
        theinfolist.append(outstring)
        lasteventtime = float(theevent[1])
    if outputfile != "":
        tide_io.writevec(theinfolist, outputfile)


# timecourse functions
def maketcfrom3col(
    inputdata: NDArray, timeaxis: NDArray, outputvector: NDArray, debug: bool = False
) -> NDArray:
    """
    Create temporal output vector from 3-column input data.

    This function processes input data containing start times, durations, and values,
    and maps these to an output vector based on a time axis. Each input row defines
    a time interval [start_time, start_time + duration] that is mapped to the output
    vector by setting the corresponding elements to the specified value.

    Parameters
    ----------
    inputdata : array-like
        3-column input data where:
        - First column: start times
        - Second column: durations
        - Third column: values to assign
    timeaxis : array-like
        Time axis defining the temporal resolution of the output vector
    outputvector : array-like
        Output vector to be populated with values from inputdata
    debug : bool, optional
        If True, displays a plot of the output vector (default is False)

    Returns
    -------
    ndarray
        The populated output vector with values assigned according to input intervals

    Notes
    -----
    - Intervals are clipped to the bounds of the time axis
    - Only intervals that overlap with the time axis (0 to max(timeaxis)) are processed
    - The function modifies the outputvector in-place

    Examples
    --------
    >>> import numpy as np
    >>> timeaxis = np.linspace(0, 10, 11)
    >>> inputdata = np.array([[1, 2, 5], [3, 1, 10]])
    >>> outputvector = np.zeros(11)
    >>> result = maketcfrom3col(inputdata, timeaxis, outputvector)
    >>> print(result)
    [0. 5. 5. 10. 10.  0.  0.  0.  0.  0.  0.]
    """
    theshape = np.shape(inputdata)
    for idx in range(0, theshape[1]):
        starttime = inputdata[0, idx]
        endtime = starttime + inputdata[1, idx]
        if (starttime <= timeaxis[-1]) and (endtime >= 0.0) and (endtime > starttime):
            startindex = np.max((bisect.bisect_left(timeaxis, starttime), 0))
            endindex = np.min((bisect.bisect_right(timeaxis, endtime), len(outputvector)))
            outputvector[startindex:endindex] = inputdata[2, idx]
            print(starttime, startindex, endtime, endindex)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("temporal output vector")
        plt.plot(timeaxis, outputvector)
        plt.show()
    return outputvector


def maketcfrom2col(
    inputdata: NDArray, timeaxis: NDArray, outputvector: NDArray, debug: bool = False
) -> NDArray:
    """
    Create a temporal output vector from 2-column input data.

    This function processes input data consisting of two columns where the first column
    represents time ranges and the second column represents the value to be assigned to
    corresponding time intervals in the output vector. The function iterates through
    the input data and assigns values to contiguous ranges in the output vector.

    Parameters
    ----------
    inputdata : NDArray
        2D array with shape (2, n) where first row contains start/end time indices
        and second row contains corresponding values to assign.
    timeaxis : NDArray
        1D array representing time values for plotting (used only in debug mode).
    outputvector : NDArray
        1D array to be populated with values from inputdata. This array is modified
        in-place and returned.
    debug : bool, optional
        If True, enables debug output including range assignments and plots the
        resulting output vector. Default is False.

    Returns
    -------
    NDArray
        The modified outputvector with values assigned from inputdata.

    Notes
    -----
    The function processes inputdata by iterating through columns and assigning
    values to ranges in outputvector. Each column represents a time interval
    [start, end) where start is taken from inputdata[0, i-1] and end from
    inputdata[0, i]. The value assigned is from inputdata[1, i-1].

    Examples
    --------
    >>> import numpy as np
    >>> inputdata = np.array([[0, 5, 10, 15], [1, 2, 3, 4]])
    >>> timeaxis = np.arange(20)
    >>> outputvector = np.zeros(20)
    >>> result = maketcfrom2col(inputdata, timeaxis, outputvector, debug=False)
    >>> print(result[:15])
    [1. 1. 1. 1. 1. 2. 2. 2. 2. 2. 3. 3. 3. 3. 3.]
    """
    theshape = np.shape(inputdata)
    rangestart = int(inputdata[0, 0])
    for i in range(1, theshape[1]):
        if rangestart < len(outputvector) - 1:
            rangeend = int(np.min([inputdata[0, i], len(outputvector)]))
            if rangeend > rangestart:
                theval = inputdata[1, i - 1]
                if debug:
                    print(f"{i}: setting outputvector[{rangestart}:{rangeend}] to {theval}")
                outputvector[rangestart:rangeend] = theval
                rangestart = rangeend
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("temporal output vector")
        plt.plot(timeaxis, outputvector)
        plt.show()
    return outputvector


# --------------------------- simulation functions ----------------------------------------------
def makeslicetimes(
    numslices: int,
    sliceordertype: str,
    tr: float = 1.0,
    multibandfac: int = 1,
    debug: bool = False,
) -> NDArray | None:
    """
    Generate slice timing list for MRI data acquisition based on slice ordering type.

    Parameters
    ----------
    numslices : int
        Number of slices in the volume.
    sliceordertype : str
        Type of slice ordering. Valid options are:
        - 'ascending': slices acquired in ascending order
        - 'descending': slices acquired in descending order
        - 'ascending_interleaved': interleaved ascending order
        - 'descending_interleaved': interleaved descending order
        - 'ascending_sparkplug': sparkplug ascending order
        - 'descending_sparkplug': sparkplug descending order
        - 'ascending_interleaved_siemens': Siemens-style interleaved ascending
        - 'descending_interleaved_siemens': Siemens-style interleaved descending
        - 'ascending_interleaved_philips': Philips-style interleaved ascending
        - 'descending_interleaved_philips': Philips-style interleaved descending
    tr : float, optional
        Repetition time in seconds (default is 1.0).
    multibandfac : int, optional
        Multiband factor (default is 1).
    debug : bool, optional
        If True, print debug information (default is False).

    Returns
    -------
    NDArray | None
        Array of slice times in seconds. Returns None if an error occurs.

    Notes
    -----
    The function computes slice acquisition times based on the specified slice order
    and multiband factor. It supports various slice ordering strategies commonly used
    in MRI pulse sequences.

    Examples
    --------
    >>> makeslicetimes(32, 'ascending_interleaved', tr=2.0, multibandfac=2)
    array([0.   , 0.25 , 0.5  , 0.75 , 1.   , 1.25 , 1.5  , 1.75 , 2.   ,
           2.25 , 2.5  , 2.75 , 3.   , 3.25 , 3.5  , 3.75 , 4.   , 4.25 ,
           4.5  , 4.75 , 5.   , 5.25 , 5.5  , 5.75 , 6.   , 6.25 , 6.5  ,
           6.75 , 7.   , 7.25 , 7.5  , 7.75 ])

    >>> makeslicetimes(16, 'descending_sparkplug', multibandfac=2)
    array([0.   , 0.5  , 1.   , 1.5  , 2.   , 2.5  , 3.   , 3.5  , 4.   ,
           4.5  , 5.   , 5.5  , 6.   , 6.5  , 7.   , 7.5  ])
    """
    outlist = np.zeros((numslices), dtype=np.float)
    if (numslices % multibandfac) != 0:
        print("ERROR: numslices is not evenly divisible by multband factor")
        return None
    mbcycle = int(numslices / multibandfac)
    normal = True
    if sliceordertype == "ascending":
        controllist = [[0, 1]]
    elif sliceordertype == "descending":
        controllist = [[mbcycle - 1, -1]]
    elif sliceordertype == "ascending_interleaved":
        controllist = [[0, 2], [1, 2]]
    elif sliceordertype == "descending_interleaved":
        controllist = [[mbcycle - 1, -2], [mbcycle - 2, -2]]
    elif sliceordertype == "ascending_sparkplug":
        normal = False
        controllist = [[0, int(mbcycle // 2) - 1]]
    elif sliceordertype == "descending_sparkplug":
        normal = False
        controllist = [[mbcycle - 1, -int(mbcycle // 2) - 1]]
    elif sliceordertype == "ascending_interleaved_siemens":
        if numslices % 2 == 0:
            controllist = [[0, 2], [1, 2]]
        else:
            controllist = [[1, 2], [0, 2]]
    elif sliceordertype == "descending_interleaved_siemens":
        if numslices % 2 == 0:
            controllist = [[mbcycle - 1, -2], [mbcycle - 2, -2]]
        else:
            controllist = [[mbcycle - 2, -2], [mbcycle - 1, -2]]
    elif sliceordertype == "ascending_interleaved_philips":
        controllist = []
        numgroups = int(np.floor(np.sqrt(numslices)))
        for i in range(numgroups):
            controllist.append([i, numgroups])
    elif sliceordertype == "descending_interleaved_philips":
        controllist = []
        numgroups = int(np.floor(np.sqrt(numslices)))
        for i in range(numgroups):
            controllist.append([mbcycle - i - 1, -numgroups])
    else:
        print("ERROR: illegal sliceordertype")
        return None

    # now make the slicetimes
    timelist = np.linspace(0, tr, num=mbcycle, endpoint=False)
    slicelist = []
    if debug:
        print("sliceordertype:", sliceordertype)
        print("number of mbcycles:", numslices // mbcycle)
        print("size of mbcycles:", mbcycle)
    for thecontrollist in controllist:
        start = thecontrollist[0]
        step = thecontrollist[1]
        theindex = start
        if normal:
            while 0 <= theindex < mbcycle:
                slicelist.append(theindex)
                theindex += step
        else:
            while len(slicelist) < mbcycle:
                slicelist.append(theindex)
                theindex = (theindex + step) % mbcycle

    if debug:
        print(slicelist)
    for index in range(numslices):
        posinmbcycle = index % mbcycle
        outlist[index] = timelist[slicelist[posinmbcycle]] + 0.0
    return outlist


# --------------------------- testing functions -------------------------------------------------
def comparemap(
    map1: NDArray, map2: NDArray, mask: NDArray | None = None, debug: bool = False
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Compare two arrays (maps) and compute various difference statistics.

    This function computes multiple metrics comparing two input arrays, `map1` and `map2`.
    It supports optional masking to focus comparisons on specific regions of the arrays.
    The function handles both 1D and multi-dimensional arrays, with support for different
    mask dimensions (either matching the map dimensions or one less).

    Parameters
    ----------
    map1 : NDArray
        First input array to compare. Can be 1D or multi-dimensional.
    map2 : NDArray
        Second input array to compare. Must have the same shape as `map1`.
    mask : NDArray, optional
        A boolean or numeric mask to select valid voxels for comparison.
        If provided, its shape must either match `map1` or be one dimension smaller.
        If `None`, all voxels are compared.
    debug : bool, optional
        If True, print debug information during execution. Default is False.

    Returns
    -------
    tuple of float
        A tuple containing the following statistics in order:
        - `mindiff`: Minimum absolute difference between `map1` and `map2`.
        - `maxdiff`: Maximum absolute difference between `map1` and `map2`.
        - `meandiff`: Mean absolute difference between `map1` and `map2`.
        - `mse`: Mean squared error between `map1` and `map2`.
        - `minreldiff`: Minimum relative difference (relative to `map1`).
        - `maxreldiff`: Maximum relative difference (relative to `map1`).
        - `meanreldiff`: Mean relative difference (relative to `map1`).
        - `relmse`: Mean squared relative error between `map1` and `map2`.

    Notes
    -----
    - If `map1` contains zero values, relative differences are set to 0 to avoid division by zero.
    - When `mask` is provided and has one fewer dimension than `map1`, it is reshaped to match
      the first dimension of `map1` before comparison.
    - The function exits with an error if shapes are incompatible or if masks are not valid.

    Examples
    --------
    >>> import numpy as np
    >>> map1 = np.array([1.0, 2.0, 3.0])
    >>> map2 = np.array([1.1, 2.2, 2.9])
    >>> result = comparemap(map1, map2)
    >>> print(result)
    (-0.1, 0.1, 0.0, 0.006666666666666667, -0.1, 0.1, 0.0, 0.006666666666666667)

    >>> mask = np.array([1, 1, 0])
    >>> result = comparemap(map1, map2, mask=mask)
    >>> print(result)
    (-0.1, 0.1, 0.0, 0.006666666666666667, -0.1, 0.1, 0.0, 0.006666666666666667)
    """
    ndims = len(map1.shape)
    if debug:
        print("map has", ndims, "axes")
    if map1.shape != map2.shape:
        print("comparemap: maps do not have the same shape - aborting")
        sys.exit()
    if ndims == 1:
        if debug:
            print("dealing with ndims == 1 case")
        map1valid = map1
        map2valid = map2
    else:
        if mask is None:
            map1valid = map1
            map2valid = map2
        else:
            if debug:
                print("mask is not None")
            ndims_mask = len(mask.shape)
            if debug:
                print("mask has", ndims_mask, "axes")
            if ndims_mask == ndims:
                if debug:
                    print("dealing with ndims == ndims_mask case")
                if map1.shape != mask.shape:
                    print("comparemap: mask does not have the same shape as the maps - aborting")
                    sys.exit()
                validvoxels = np.where(mask > 0)[0]
                map1valid = map1[validvoxels, :]
                map2valid = map2[validvoxels, :]
            elif ndims_mask == ndims - 1:
                # need to make expanded mask
                if debug:
                    print("dealing with ndims == ndims_mask + 1 case")
                    print("shape of map:", map1.shape)
                    print("shape of mask:", mask.shape)
                numvox = 1
                for i in range(ndims - 1):
                    numvox *= mask.shape[i]
                reshapemask = mask.reshape(numvox)
                reshapemap1 = map1.reshape(numvox, -1)
                reshapemap2 = map2.reshape(numvox, -1)
                validvoxels = np.where(reshapemask > 0)[0]
                map1valid = reshapemap1[validvoxels, :]
                map2valid = reshapemap2[validvoxels, :]
            else:
                print("mask is not compatible with map")
                sys.exit()

    # at this point, map2valid and map1valid are the same dimensions
    diff = map2valid - map1valid
    reldiff = np.where(map1valid != 0.0, diff / map1valid, 0.0)
    maxdiff = np.max(diff)
    mindiff = np.min(diff)
    meandiff = np.mean(diff)
    mse = np.mean(np.square(diff))

    maxreldiff = np.max(reldiff)
    minreldiff = np.min(reldiff)
    meanreldiff = np.mean(reldiff)
    relmse = np.mean(np.square(reldiff))

    return mindiff, maxdiff, meandiff, mse, minreldiff, maxreldiff, meanreldiff, relmse


def comparerapidtideruns(root1: str, root2: str, debug: bool = False) -> dict[str, Any]:
    """
    Compare results from two rapidtide runs by evaluating corresponding maps and timecourses.

    This function compares NIfTI maps and text-based timecourses from two different rapid tide
    processing runs. It evaluates differences between the corresponding files using various
    statistical measures such as mean difference, max difference, mean squared error, and
    relative versions of these metrics.

    Parameters
    ----------
    root1 : str
        The base filename (without extension) for the first rapid tide run.
    root2 : str
        The base filename (without extension) for the second rapid tide run.
    debug : bool, optional
        If True, print detailed debug information during execution. Default is False.

    Returns
    -------
    dict[str, Any]
        A dictionary containing comparison results for each map and timecourse.
        Keys are map or timecourse names, and values are dictionaries with the following keys:
        - 'mindiff': minimum difference
        - 'maxdiff': maximum difference
        - 'meandiff': mean difference
        - 'mse': mean squared error
        - 'relmindiff': relative minimum difference
        - 'relmaxdiff': relative maximum difference
        - 'relmeandiff': relative mean difference
        - 'relmse': relative mean squared error

    Notes
    -----
    - The function assumes that both runs have corresponding mask files named
      ``<root>_desc-corrfit_mask.nii.gz``.
    - For each map, the function checks if the corresponding NIfTI files exist and match
      in spatial dimensions.
    - For each timecourse, the function reads from JSON files and compares the time series
      only if both files are present and have matching lengths.
    - If spatial or temporal dimensions do not match, the function exits with an error.

    Examples
    --------
    >>> results = comparerapidtideruns("run1", "run2", debug=True)
    >>> print(results["maxtime"])
    {'mindiff': -0.01, 'maxdiff': 0.02, 'meandiff': 0.005, 'mse': 0.0001, ...}
    """
    results = {}
    maskname1 = f"{root1}_desc-corrfit_mask.nii.gz"
    (
        masknim1,
        maskdata1,
        maskhdr1,
        themaskdims1,
        themasksizes1,
    ) = tide_io.readfromnifti(maskname1)
    maskname2 = f"{root2}_desc-corrfit_mask.nii.gz"
    (
        masknim2,
        maskdata2,
        maskhdr2,
        themaskdims2,
        themasksizes2,
    ) = tide_io.readfromnifti(maskname2)

    # compare maps
    for map in [
        "maxtime",
        "maxcorr",
        "maxwidth",
        "MTT",
        "mean",
        "lfofilterCoeff",
        "lfofilterMean",
        "lfofilterNorm",
        "lfofilterR",
        "lfofilterR2",
        "lfofilterInbandVarianceChange",
    ]:
        if debug:
            print(f"checking map {map}")
        filename1 = f"{root1}_desc-{map}_map.nii.gz"
        filename2 = f"{root2}_desc-{map}_map.nii.gz"
        if tide_io.checkspacematch(maskhdr1, maskhdr2):
            mask = maskdata1 * maskdata2
            if os.path.isfile(filename1) and os.path.isfile(filename2):
                # files exist - read them in and process them
                nim1, data1, hdr1, thedims1, thesizes1 = tide_io.readfromnifti(filename1)
                nim2, data2, hdr2, thedims2, thesizes2 = tide_io.readfromnifti(filename2)
                if tide_io.checkspacematch(hdr1, hdr2) and tide_io.checkspacematch(hdr1, maskhdr1):
                    # files match in size
                    results[map] = {}
                    (
                        results[map]["mindiff"],
                        results[map]["maxdiff"],
                        results[map]["meandiff"],
                        results[map]["mse"],
                        results[map]["relmindiff"],
                        results[map]["relmaxdiff"],
                        results[map]["relmeandiff"],
                        results[map]["relmse"],
                    ) = comparemap(data1, data2, mask=mask, debug=debug)
                    if debug:
                        print(results[map])
                else:
                    print("mask dimensions don't match - aborting")
                    sys.exit()
            else:
                print("map", map, "does not exist - skipping")
        else:
            print("mask dimensions don't match - aborting")
            sys.exit()
    for timecourse in [
        "initialmovingregressor_timeseries.json:prefilt",
        "initialmovingregressor_timeseries.json:postfilt",
        "oversampledmovingregressor_timeseries.json:pass1",
        "oversampledmovingregressor_timeseries.json:pass2",
        "oversampledmovingregressor_timeseries.json:pass3",
        "oversampledmovingregressor_timeseries.json:pass4",
    ]:
        if debug:
            print(f"checking timecourse {timecourse}")
        filespec1 = f"{root1}_desc-{timecourse}"
        filespec2 = f"{root2}_desc-{timecourse}"
        allpresent = True
        try:
            dummy, dummy, dummy, timecourse1, dummy, dummy = tide_io.readvectorsfromtextfile(
                filespec1, onecol=True
            )
        except FileNotFoundError:
            if debug:
                print(f"{filespec2} file not found")
            allpresent = False
        except ValueError:
            if debug:
                print(f"{filespec2} column not found")
            allpresent = False

        try:
            dummy, dummy, dummy, timecourse2, dummy, dummy = tide_io.readvectorsfromtextfile(
                filespec2, onecol=True
            )
        except FileNotFoundError:
            if debug:
                print(f"{filespec2} file not found")
            allpresent = False
        except ValueError:
            if debug:
                print(f"{filespec2} column not found")
            allpresent = False

        if allpresent:
            tcname = timecourse.replace("_timeseries.json:", "_")
            if len(timecourse1) == len(timecourse2):
                results[tcname] = {}
                (
                    results[tcname]["mindiff"],
                    results[tcname]["maxdiff"],
                    results[tcname]["meandiff"],
                    results[tcname]["mse"],
                    results[tcname]["relmindiff"],
                    results[tcname]["relmaxdiff"],
                    results[tcname]["relmeandiff"],
                    results[tcname]["relmse"],
                ) = comparemap(timecourse1, timecourse2, debug=debug)
                if debug:
                    print(results[tcname])
            else:
                print("timecourse dimensions don't match - skipping")
        else:
            print(f"{timecourse} not present in both datasets - skipping")
    return results


def comparehappyruns(root1: str, root2: str, debug: bool = False) -> dict[str, Any]:
    """
    Compare results from two happy runs by comparing output maps and timecourses.

    This function compares neuroimaging maps (e.g., app_info, vessels_mask) and
    cardiac timecourses (e.g., cardfromfmri_25.0Hz.txt) between two datasets
    identified by their root names. It performs spatial alignment checks and
    computes various statistical differences between corresponding files.

    Parameters
    ----------
    root1 : str
        Root name of the first dataset (e.g., 'subject01_run1').
    root2 : str
        Root name of the second dataset (e.g., 'subject01_run2').
    debug : bool, optional
        If True, print debug information during execution. Default is False.

    Returns
    -------
    dict[str, Any]
        A dictionary containing comparison results for each processed map and
        timecourse. Each entry includes:
        - mindiff: minimum absolute difference
        - maxdiff: maximum absolute difference
        - meandiff: mean absolute difference
        - mse: mean squared error
        - relmindiff: relative minimum difference
        - relmaxdiff: relative maximum difference
        - relmeandiff: relative mean difference
        - relmse: relative mean squared error

    Notes
    -----
    - The function expects specific file naming conventions for both maps and
      timecourses.
    - Spatial dimensions of masks and data must match for comparison to proceed.
    - If any file is missing or mismatched, the function will exit with an error.

    Examples
    --------
    >>> results = comparehappyruns('subject01_run1', 'subject01_run2', debug=True)
    >>> print(results['app_info']['meandiff'])
    """
    results = {}
    if debug:
        print("comparehappyruns rootnames:", root1, root2)
    for map in ["app_info", "vessels_mask"]:
        filename1 = f"{root1}_desc-{map}.nii.gz"
        maskname1 = f"{root1}_processvoxels_mask.nii.gz"
        filename2 = f"{root2}_desc-{map}.nii.gz"
        maskname2 = f"{root2}_processvoxels_mask.nii.gz"
        (
            masknim1,
            maskdata1,
            maskhdr1,
            themaskdims1,
            themasksizes1,
        ) = tide_io.readfromnifti(maskname1)
        (
            masknim2,
            maskdata2,
            maskhdr2,
            themaskdims2,
            themasksizes2,
        ) = tide_io.readfromnifti(maskname2)
        if tide_io.checkspacematch(maskhdr1, maskhdr2):
            mask = maskdata1 * maskdata2
            if os.path.isfile(filename1) and os.path.isfile(filename2):
                # files exist - read them in and process them
                if debug:
                    print("comparing maps:")
                    print("\t", filename1)
                    print("\t", filename2)
                nim1, data1, hdr1, thedims1, thesizes1 = tide_io.readfromnifti(filename1)
                nim2, data2, hdr2, thedims2, thesizes2 = tide_io.readfromnifti(filename2)
                if tide_io.checkspacematch(hdr1, hdr2) and tide_io.checkspacematch(hdr1, maskhdr1):
                    # files match in size
                    results[map] = {}
                    (
                        results[map]["mindiff"],
                        results[map]["maxdiff"],
                        results[map]["meandiff"],
                        results[map]["mse"],
                        results[map]["relmindiff"],
                        results[map]["relmaxdiff"],
                        results[map]["relmeandiff"],
                        results[map]["relmse"],
                    ) = comparemap(data1, data2, mask=mask, debug=debug)
                else:
                    print("mask dimensions don't match - aborting")
                    sys.exit()
            else:
                print("map", map, "does not exist - skipping")
        else:
            print("mask dimensions don't match - aborting")
            sys.exit()
        if debug:
            print("done processing", map)
    for timecourse in [
        "cardfromfmri_25.0Hz.txt",
        "cardfromfmri_dlfiltered_25.0Hz.txt",
        "cardfromfmrienv_25.0Hz.txt",
    ]:
        filename1 = root1 + "_" + timecourse
        filename2 = root2 + "_" + timecourse
        if os.path.isfile(filename1) and os.path.isfile(filename2):
            if debug:
                print("comparing timecourses:")
                print("\t", filename1)
                print("\t", filename2)
            data1 = np.transpose(tide_io.readvecs(filename1))
            data2 = np.transpose(tide_io.readvecs(filename2))
            if len(data1) == len(data2):
                # files match in size
                results[timecourse] = {}
                (
                    results[timecourse]["mindiff"],
                    results[timecourse]["maxdiff"],
                    results[timecourse]["meandiff"],
                    results[timecourse]["mse"],
                    results[timecourse]["relmindiff"],
                    results[timecourse]["relmaxdiff"],
                    results[timecourse]["relmeandiff"],
                    results[timecourse]["relmse"],
                ) = comparemap(data1, data2, debug=debug)
            else:
                print("timecourse lengths don't match - aborting")
                sys.exit()
        else:
            print("timecourse", timecourse, "does not exist - skipping")
        if debug:
            print("done processing", timecourse)

    return results


# shared memory routines
def numpy2shared(
    inarray: NDArray, theouttype: np.dtype, name: str | None = None
) -> tuple[NDArray, shared_memory.SharedMemory]:
    """
    Convert a numpy array to a shared memory array.

    This function creates a shared memory block and copies the data from the input
    numpy array to the shared memory array. The returned array and shared memory
    object can be used for inter-process communication or memory sharing.

    Parameters
    ----------
    inarray : NDArray
        Input numpy array to be converted to shared memory.
    theouttype : dtype
        Data type of the output shared memory array.
    name : str, optional
        Name of the shared memory block. If None, an anonymous shared memory
        block is created.

    Returns
    -------
    tuple[NDArray, multiprocessing.shared_memory.SharedMemory]
        A tuple containing:
        - The shared memory array with the same shape as input array
        - The shared memory object that manages the memory block

    Notes
    -----
    The returned shared memory object must be explicitly closed and unlink
    when no longer needed to free system resources. The shared memory block
    will be automatically unlinked upon creation but can be accessed by other
    processes using the same name.

    Examples
    --------
    >>> import numpy as np
    >>> from multiprocessing import shared_memory
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> shared_arr, shm = numpy2shared(arr, np.int32)
    >>> print(shared_arr)
    [1 2 3 4 5]
    >>> # Clean up when done
    >>> shared_arr.close()
    >>> shm.close()
    >>> shm.unlink()
    """
    # Create a shared memory block to store the array data
    outnbytes = np.dtype(theouttype).itemsize * inarray.size
    shm = shared_memory.SharedMemory(name=None, create=True, size=outnbytes)
    shm.unlink()
    inarray_shared = np.ndarray(inarray.shape, dtype=theouttype, buffer=shm.buf)
    np.copyto(inarray_shared, inarray)  # Copy data to shared memory array
    return inarray_shared, shm  # Return both the array and the shared memory object


def allocshared(
    theshape: tuple[int, ...], thetype: np.dtype, name: str | None = None
) -> tuple[NDArray, shared_memory.SharedMemory]:
    """
    Allocate shared memory for a numpy array.

    This function creates a shared memory block and returns both the numpy array
    backed by this shared memory and the shared memory object itself. The array
    can be accessed from different processes, making it useful for inter-process
    communication.

    Parameters
    ----------
    theshape : tuple of int
        The shape of the array to be created. Must be a tuple of integers.
    thetype : type
        The data type of the array elements. Can be any numpy-compatible dtype.
    name : str, optional
        Name of existing shared memory object. If None, a new shared memory
        block is created. Default is None.

    Returns
    -------
    tuple[NDArray, shared_memory.SharedMemory]
        A tuple containing:
        - The numpy array backed by shared memory
        - The shared_memory.SharedMemory object

    Notes
    -----
    The returned shared memory object should be explicitly closed and unlink
    when no longer needed to free system resources. The array can be accessed
    from multiple processes, but care should be taken to avoid race conditions.

    Examples
    --------
    >>> import numpy as np
    >>> from multiprocessing import shared_memory
    >>> arr, shm = allocshared((3, 4), np.float64)
    >>> arr[0, 0] = 42.0
    >>> print(arr)
    [[42.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    >>> # Don't forget to clean up
    >>> shm.close()
    >>> shm.unlink()
    """
    # Calculate size based on shape
    thesize = np.prod(theshape)
    # Determine the data type size
    dtype_size = np.dtype(thetype).itemsize
    # Create a shared memory block of the required size
    shm = shared_memory.SharedMemory(name=None, create=True, size=thesize * dtype_size)
    shm.unlink()
    outarray = np.ndarray(theshape, dtype=thetype, buffer=shm.buf)
    return outarray, shm  # Return both the array and the shared memory object


def allocarray(
    theshape: tuple[int, ...], thetype: np.dtype, shared: bool = False, name: str | None = None
) -> tuple[NDArray, shared_memory.SharedMemory | None]:
    """
    Allocate and return a numpy array with specified shape and type.

    Parameters
    ----------
    theshape : tuple[int, ...]
        Shape of the array to be allocated.
    thetype : type
        Data type of the array elements.
    shared : bool, optional
        If True, allocate the array in shared memory. Default is False.
    name : str | None, optional
        Name for the shared memory segment. Required if shared=True. Default is None.

    Returns
    -------
    tuple[NDArray, shared_memory.SharedMemory | None]
        A tuple containing:
        - The allocated numpy array filled with zeros
        - The shared memory object if shared=True, otherwise None

    Notes
    -----
    When ``shared=True``, the function delegates to ``allocshared`` to create
    a shared memory array. Otherwise, it creates a regular numpy array using
    ``np.zeros`` with the specified shape and dtype.

    Examples
    --------
    >>> import numpy as np
    >>> arr, shm = allocarray((3, 4), np.float64)
    >>> print(arr.shape)
    (3, 4)
    >>> print(arr.dtype)
    float64

    >>> # For shared memory allocation
    >>> arr, shm = allocarray((2, 3), np.int32, shared=True, name="my_array")
    >>> print(shm is not None)
    True
    """
    if shared:
        return allocshared(theshape, thetype, name)
    else:
        return np.zeros(theshape, dtype=thetype), None


def cleanup_shm(shm: shared_memory.SharedMemory | None) -> None:
    """
    Clean up shared memory object.

    Parameters
    ----------
    shm : shared_memory.SharedMemory or None
        Shared memory object to clean up. If None, no action is taken.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function is designed to properly release shared memory resources.
    It should be called to clean up shared memory objects to prevent resource leaks.
    If the shared memory object is None, the function performs no operation.

    Examples
    --------
    >>> from multiprocessing import shared_memory
    >>> shm = shared_memory.SharedMemory(create=True, size=1024)
    >>> cleanup_shm(shm)
    >>> # Shared memory is now cleaned up
    """
    # Cleanup
    pass
    # if shm is not None:
    #    shm.close()
    #    shm.unlink()
