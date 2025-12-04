#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2018-2025 Blaise Frederick
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
"""
Functions for parsers.
"""
import argparse
import os.path as op
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.util as tide_util


class IndicateSpecifiedAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        """
        Store the parsed values in the namespace and mark as non-default.

        This method is called when an argument parser encounters a command-line
        argument that matches this action. It stores the parsed values in the
        namespace and sets a corresponding flag indicating the value was explicitly
        provided (not default).

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The ArgumentParser object which contains this action.
        namespace : argparse.Namespace
            The namespace object to which the parsed values will be stored.
        values : Any
            The parsed values for the argument.
        option_string : str, optional
            The command-line option string that triggered this action.

        Returns
        -------
        None
            This method does not return a value.

        Notes
        -----
        This action stores the values using `setattr(namespace, self.dest, values)`
        and additionally sets `setattr(namespace, self.dest + "_nondefault", True)`
        to indicate that the value was explicitly provided rather than using a default.

        Examples
        --------
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--verbose', action='store_true')
        >>> args = parser.parse_args(['--verbose'])
        >>> args.verbose
        True
        """
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + "_nondefault", True)


def detailedversion() -> None:
    """
    Print detailed version information including release version, git metadata, and Python version.

    This function retrieves version information from the tide_util.version() function and
    prints comprehensive details about the current build, including release version, git
    commit hash, git commit date, whether the working directory is dirty, and the Python
    version being used. The function then exits the program with sys.exit().

    Returns
    -------
    None
        This function does not return any value and exits the program after printing
        the version information.

    Notes
    -----
    The function relies on the tide_util.version() function to retrieve git and release
    information. The output includes:
    - release_version: The official release version string
    - git_sha: The git commit hash
    - git_date: The git commit date
    - git_isdirty: Boolean indicating if the working directory has uncommitted changes
    - python_version: The Python version information

    Examples
    --------
    >>> detailedversion()
    release version: 1.2.3
    git_sha: a1b2c3d4e5f6
    git_date: 2023-10-15 14:30:45
    git_isdirty: False
    python_version: (3, 9, 7, 'final', 0)
    """
    (
        release_version,
        git_sha,
        git_date,
        git_isdirty,
    ) = tide_util.version()
    python_version = str(sys.version_info)
    print(f"release version: {release_version}")
    print(f"git_sha: {git_sha}")
    print(f"git_date: {git_date}")
    print(f"git_isdirty: {git_isdirty}")
    print(f"python_version: {python_version}")
    sys.exit()


def setifnotset(thedict: Dict[str, Any], thekey: str, theval: Any) -> None:
    """
    Set a value in dictionary if key with "_nondefault" suffix is not present.

    This function checks if a key with the suffix "_nondefault" exists in the
    dictionary. If not found, it prints a message and sets the specified key
    to the given value.

    Parameters
    ----------
    thedict : dict of str to Any
        The dictionary to modify
    thekey : str
        The key to set in the dictionary
    theval : Any
        The value to set for thekey

    Returns
    -------
    None
        This function modifies the dictionary in-place and returns None

    Notes
    -----
    The function uses the convention that keys with "_nondefault" suffix indicate
    that a default value has already been overridden. This is useful for tracking
    which settings have been explicitly set versus those that remain at default values.

    Examples
    --------
    >>> config = {}
    >>> setifnotset(config, 'debug', True)
    overriding debug
    >>> print(config)
    {'debug': True}

    >>> config = {'debug_nondefault': True}
    >>> setifnotset(config, 'debug', False)
    >>> print(config)
    {'debug_nondefault': True}
    """
    if (thekey + "_nondefault") not in thedict.keys():
        print("overriding " + thekey)
        thedict[thekey] = theval


def is_valid_file(parser: argparse.ArgumentParser, arg: Optional[str]) -> Optional[str]:
    """
    Check if argument is an existing file.

    This function validates that the provided argument corresponds to an existing file.
    If the argument is None, the function returns None without further validation.
    If the argument is not None, it parses the file specification and checks if the file exists.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser object used to raise errors when file validation fails.
    arg : str, optional
        The file path to validate. If None, no validation is performed.

    Returns
    -------
    str, optional
        Returns the original argument if it is a valid file path, or None if the argument is None.

    Raises
    ------
    SystemExit
        If the file specified in arg does not exist, the parser will raise a SystemExit error.

    Notes
    -----
    This function uses `tide_io.parsefilespec()` to parse the file specification and
    `os.path.isfile()` to check file existence.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> is_valid_file(parser, "existing_file.txt")
    'existing_file.txt'

    >>> is_valid_file(parser, "nonexistent_file.txt")
    # Raises SystemExit with error message
    """
    if arg is not None:
        thefilename, colspec = tide_io.parsefilespec(arg)
    else:
        thefilename = None

    if not op.isfile(thefilename) and (thefilename is not None):
        parser.error("The file {0} does not exist!".format(thefilename))

    return arg


def invert_float(parser: argparse.ArgumentParser, arg: Union[str, float]) -> Union[str, float]:
    """
    Check if argument is float or auto.

    This function validates if the input argument is a float value or the string "auto".
    If the argument is a valid float, it computes the multiplicative inverse (1/x).
    If the argument is "auto", it returns "auto" unchanged.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser object used for validation.
    arg : str or float
        The argument to be checked and inverted. Can be a string representation
        of a float or the string "auto".

    Returns
    -------
    str or float
        If arg is "auto", returns "auto". Otherwise, returns the multiplicative
        inverse of the float value.

    Notes
    -----
    This function relies on the `is_float` helper function for validation.
    The multiplicative inverse is computed as 1.0 / arg, so the input must be
    non-zero to avoid division by zero errors.

    Examples
    --------
    >>> invert_float(parser, 2.0)
    0.5

    >>> invert_float(parser, "auto")
    "auto"

    >>> invert_float(parser, "0.5")
    2.0
    """
    arg = is_float(parser, arg)

    if arg != "auto":
        arg = 1.0 / arg
    return arg


def is_float(
    parser: argparse.ArgumentParser,
    arg: Union[str, float],
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
) -> Union[str, float]:
    """
    Check if argument is float or auto.

    This function validates that the input argument can be converted to a float
    or is the string "auto". It also checks that the value falls within specified
    bounds if minval or maxval are provided.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser object used for error reporting.
    arg : str or float
        The argument to validate. Can be a string representation of a float
        or the string "auto".
    minval : float, optional
        The minimum allowed value. If specified, the argument must be greater than
        or equal to this value.
    maxval : float, optional
        The maximum allowed value. If specified, the argument must be less than
        or equal to this value.

    Returns
    -------
    str or float
        Returns the validated argument as a float if it's not "auto", otherwise
        returns the string "auto".

    Raises
    ------
    SystemExit
        If the argument cannot be converted to a float or violates the min/max
        constraints, the parser will call sys.exit() with an error message.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> is_float(parser, "3.14")
    3.14

    >>> is_float(parser, "auto")
    'auto'

    >>> is_float(parser, "5", minval=0, maxval=10)
    5.0

    >>> is_float(parser, "-1", minval=0)
    SystemExit: Value -1.0 is smaller than 0
    """
    if arg != "auto":
        try:
            arg = float(arg)
        except parser.error:
            parser.error('Value {0} is not a float or "auto"'.format(arg))
        if minval is not None and arg < minval:
            parser.error("Value {0} is smaller than {1}".format(arg, minval))
        if maxval is not None and arg > maxval:
            parser.error("Value {0} is larger than {1}".format(arg, maxval))

    return arg


def is_valid_file_or_float(
    parser: argparse.ArgumentParser, arg: Optional[str]
) -> Union[str, float]:
    """
    Check if argument is an existing file or a valid float value.

    This function validates whether the input argument is either:
    1. An existing file path, or
    2. A valid float number

    If the argument is not a valid file but can be converted to a float,
    the function returns the float value. Otherwise, it raises an error
    through the provided parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser object used to raise errors when validation fails.
    arg : str, optional
        The argument to validate. Can be a file path or a string representation
        of a float number. If None, the function returns None.

    Returns
    -------
    str or float
        If argument is a valid file path, returns the original string.
        If argument is a valid float, returns the float value.
        If argument is None, returns None.

    Raises
    ------
    SystemExit
        Raised by the parser when the argument is neither a valid file nor
        a valid float number.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> is_valid_file_or_float(parser, "data.txt")
    "data.txt"

    >>> is_valid_file_or_float(parser, "3.14")
    3.14

    >>> is_valid_file_or_float(parser, "invalid_file.txt")
    SystemExit: Value invalid_file.txt is not a float or a valid filename
    """
    if arg is not None:
        thefilename, colspec = tide_io.parsefilespec(arg)
    else:
        thefilename = None

    if not op.isfile(thefilename) and thefilename is not None:
        # this is not a file - is it a float?
        try:
            arg = float(arg)
        except ValueError:
            parser.error("Value {0} is not a float or a valid filename".format(arg))

    return arg


def is_int(
    parser: argparse.ArgumentParser,
    arg: Union[str, int],
    minval: Optional[int] = None,
    maxval: Optional[int] = None,
) -> Union[str, int]:
    """
    Check if argument is int or auto.

    This function validates that the input argument is either an integer or the string "auto".
    If the argument is not "auto", it attempts to convert it to an integer. The function also
    checks that the integer value falls within the specified bounds.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser object used to raise errors when validation fails.
    arg : str or int
        The argument to validate. Can be an integer or the string "auto".
    minval : int, optional
        The minimum allowed value for the integer. If None, no minimum check is performed.
    maxval : int, optional
        The maximum allowed value for the integer. If None, no maximum check is performed.

    Returns
    -------
    str or int
        Returns the original argument if it is "auto", otherwise returns the converted integer.

    Raises
    ------
    SystemExit
        Raised by the parser.error() method when validation fails.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> is_int(parser, "42")
    42

    >>> is_int(parser, "auto")
    'auto'

    >>> is_int(parser, "42", minval=0, maxval=100)
    42

    >>> is_int(parser, "42", minval=50)  # raises parser.error
    SystemExit: Value 42 is smaller than 50
    """
    if arg != "auto":
        try:
            arg = int(arg)
        except parser.error:
            parser.error('Value {0} is not an int or "auto"'.format(arg))
        if minval is not None and arg < minval:
            parser.error("Value {0} is smaller than {1}".format(arg, minval))
        if maxval is not None and arg > maxval:
            parser.error("Value {0} is larger than {1}".format(arg, maxval))

    return arg


def is_range(parser: argparse.ArgumentParser, arg: Optional[List[Any]]) -> Optional[List[Any]]:
    """
    Check if argument is min/max pair.

    This function validates that the provided argument is a list containing exactly two elements
    that can be converted to floats, with the first element being less than or equal to the second.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser object used for error reporting
    arg : list of any, optional
        List containing two elements representing min and max values, or None

    Returns
    -------
    list of any, optional
        The original argument if validation passes, or None if arg is None

    Raises
    ------
    SystemExit
        If argument is not None and does not contain exactly two elements, or if min > max

    Notes
    -----
    This function is typically used as a type validator for argparse arguments.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> is_range(parser, ['1', '10'])
    ['1', '10']

    >>> is_range(parser, ['10', '1'])
    SystemExit: Argument min must be lower than max.

    >>> is_range(parser, ['1', '2', '3'])
    SystemExit: Argument must be min/max pair.
    """
    if arg is not None and len(arg) != 2:
        parser.error("Argument must be min/max pair.")
    elif arg is not None and float(arg[0]) > float(arg[1]):
        parser.error("Argument min must be lower than max.")

    return arg


def is_valid_tag(parser: argparse.ArgumentParser, arg: Optional[str]) -> Tuple[str, str]:
    """
    Check if argument is existing file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser object used for error reporting.
    arg : str, optional
        The argument string to validate, expected to be in the format "tagname,value1,value2,...".
        If None, the function will return empty strings.

    Returns
    -------
    tuple of (str, str)
        A tuple containing (tagname, tagval) where tagname is the first part
        before the first comma and tagval is the rest of the string joined by commas.

    Notes
    -----
    If the argument contains fewer than 2 comma-separated parts, the parser will
    raise an error with the message "No tag value specified."

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> is_valid_tag(parser, "environment,production,us-east")
    ('environment', 'production,us-east')

    >>> is_valid_tag(parser, "type,web,app")
    ('type', 'web,app')
    """
    if arg is not None:
        argparts = arg.split(",")
        if len(argparts) < 2:
            parser.error("No tag value specified.")
        tagname = argparts[0]
        tagval = ",".join(argparts[1:])

    return (tagname, tagval)


DEFAULT_FILTER_ORDER = 6
DEFAULT_PAD_SECONDS = 30.0
DEFAULT_PREFILTERPADTYPE = "reflect"
DEFAULT_PERMUTATIONMETHOD = "shuffle"
DEFAULT_NORMTYPE = "stddev"
DEFAULT_FILTERBAND = "lfo"
DEFAULT_FILTERTYPE = "trapezoidal"
DEFAULT_PADVAL = 0
DEFAULT_WINDOWFUNC = "hamming"


def addreqinputniftifile(
    parser: argparse.ArgumentParser, varname: str, addedtext: str = ""
) -> None:
    """
    Add a required input NIFTI file argument to an ArgumentParser.

    This function adds a command-line argument for specifying a required input
    NIFTI file to the provided ArgumentParser. The argument includes validation
    to ensure the file exists and is readable.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser object to which the argument will be added.
    varname : str
        The name of the command-line argument (e.g., '--input' or '-i').
    addedtext : str, optional
        Additional text to append to the help message. Default is empty string.

    Returns
    -------
    None
        This function modifies the parser in-place and returns nothing.

    Notes
    -----
    The function uses a lambda function for type validation that calls
    `is_valid_file(parser, x)` to ensure the specified file exists and is
    readable before accepting it as a valid input.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addreqinputniftifile(parser, '--input', 'Path to the input NIFTI file.')
    >>> args = parser.parse_args(['--input', 'data.nii.gz'])
    """
    parser.add_argument(
        varname,
        type=lambda x: is_valid_file(parser, x),
        help="Input NIFTI file name.  " + addedtext,
    )


def addreqoutputniftifile(
    parser: argparse.ArgumentParser, varname: str, addedtext: str = ""
) -> None:
    """
    Add a required argument for specifying output NIFTI file name to an argument parser.

    This function adds a command-line argument to the provided ArgumentParser instance
    that specifies the output NIFTI file name. The argument is required and accepts
    string input.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the output NIFTI file argument will be added.
    varname : str
        The name of the command-line argument (without dashes). This will be used
        as the argument name in the parser.
    addedtext : str, optional
        Additional text to append to the help message. Default is empty string.

    Returns
    -------
    None
        This function modifies the parser in-place and does not return any value.

    Notes
    -----
    The added argument will be a required positional argument that accepts a string
    representing the output NIFTI file path. The help message will include the
    provided addedtext to give additional context to users.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addreqoutputniftifile(parser, "output_file", "Path to the output NIFTI file.")
    >>> args = parser.parse_args(['--output_file', 'output.nii.gz'])
    >>> print(args.output_file)
    output.nii.gz
    """
    parser.add_argument(
        varname,
        type=str,
        help="Output NIFTI file name.  " + addedtext,
    )


def addreqinputtextfile(
    parser: argparse.ArgumentParser, varname: str, onecol: bool = False
) -> None:
    """
    Add a required text file argument to an argument parser for timeseries data.

    This function adds a command-line argument to specify a text file containing
    one or more timeseries columns. The argument supports column selection options
    depending on whether single or multiple columns are expected.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the text file argument will be added.
    varname : str
        The name of the argument variable (used as the argument flag).
    onecol : bool, optional
        If True, the argument expects a single column selection.
        If False, the argument accepts multiple column specifications.
        Default is False.

    Returns
    -------
    None
        This function modifies the parser in-place and returns None.

    Notes
    -----
    The text file can be in BIDS format, which allows column names to be used
    in addition to integer column indices. Column selection syntax varies based
    on the `onecol` parameter:

    - When `onecol=True`: Use `[:COLUMN]` where COLUMN is an integer or column name
    - When `onecol=False`: Use `[:COLSPEC]` where COLSPEC is an integer, a column
      separated list of ranges, or a comma separated set of column names

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addreqinputtextfile(parser, 'timeseries_file', onecol=False)
    >>> args = parser.parse_args()
    """
    if onecol:
        colspecline = (
            "Use [:COLUMN] to select which column to use, where COLUMN is an "
            "integer or a column name (if input file is BIDS)."
        )
    else:
        colspecline = (
            "Use [:COLSPEC] to select which column(s) to use, where COLSPEC is an "
            "integer, a column separated list of ranges, or a comma "
            "separated set of column names (if input file is BIDS).  Default is to use all columns"
        )
    parser.add_argument(
        varname,
        type=lambda x: is_valid_file(parser, x),
        help="Text file containing one or more timeseries columns. " + colspecline,
    )


def addreqinputtextfiles(
    parser: argparse.ArgumentParser,
    varname: str,
    numreq: str = "Two",
    nargs: str = "*",
    onecol: bool = False,
) -> None:
    """
    Add required input text files argument to an ArgumentParser.

    This function adds a command-line argument to specify one or more text files
    containing timeseries data. The argument supports various column selection
    options depending on the `onecol` parameter.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser object to which the argument will be added.
    varname : str
        The name of the command-line argument (without dashes).
    numreq : str, optional
        Number of required input files (default is "Two").
    nargs : str, optional
        Argument specification for the number of arguments (default is "*").
    onecol : bool, optional
        If True, single column selection is enabled; if False, multiple column
        selection is enabled (default is False).

    Returns
    -------
    None
        This function modifies the parser in-place and returns None.

    Notes
    -----
    The added argument accepts text files with timeseries data. When `onecol` is
    True, column selection uses the format [:COLUMN] where COLUMN is an integer
    or column name (if input file is BIDS). When `onecol` is False, column
    selection uses the format [:COLSPEC] where COLSPEC can be an integer, a
    comma-separated list of ranges, or a comma-separated set of column names
    (if input file is BIDS). Default behavior is to use all columns.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addreqinputtextfiles(parser, "input_files", numreq="Three", onecol=True)
    >>> args = parser.parse_args()
    """
    if onecol:
        colspecline = (
            "Use [:COLUMN] to select which column to use, where COLUMN is an "
            "integer or a column name (if input file is BIDS)."
        )
    else:
        colspecline = (
            "Use [:COLSPEC] to select which column(s) to use, where COLSPEC is an "
            "integer, a column separated list of ranges, or a comma "
            "separated set of column names (if input file is BIDS).  Default is to use all columns."
        )
    parser.add_argument(
        varname,
        nargs=nargs,
        type=lambda x: is_valid_file(parser, x),
        help=numreq + " text files containing one or more timeseries columns. " + colspecline,
    )


def addreqoutputtextfile(
    parser: argparse.ArgumentParser, varname: str, rootname: bool = False
) -> None:
    """
    Add a required argument for specifying output text file name to an argument parser.

    This function adds a command-line argument to the provided ArgumentParser instance
    that specifies the name of an output text file. The argument is required and
    accepts string values.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser instance to which the output file argument will be added.
    varname : str
        The name of the command-line argument (e.g., '--output-file' or '-o').
    rootname : bool, optional
        If True, the help text will indicate this is a root name for output files
        (default is False, which indicates a single output text file name).

    Returns
    -------
    None
        This function modifies the parser in-place and does not return any value.

    Notes
    -----
    This function is typically used when setting up command-line interfaces for
    tools that require a specified output text file name. The argument added is
    required, meaning the parser will raise an error if this argument is not provided.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addreqoutputtextfile(parser, '--output-file')
    >>> args = parser.parse_args(['--output-file', 'results.txt'])
    >>> print(args.output_file)
    'results.txt'

    >>> parser = argparse.ArgumentParser()
    >>> addreqoutputtextfile(parser, '--root-name', rootname=True)
    >>> args = parser.parse_args(['--root-name', 'data'])
    >>> print(args.root_name)
    'data'
    """
    if rootname:
        helpline = "Root name for the output files"
    else:
        helpline = "Name of the output text file."
    parser.add_argument(
        varname,
        type=str,
        help=helpline,
    )


def addtagopts(
    opt_group: argparse._ArgumentGroup,
    helptext: str = "Additional key, value pairs to add to the options json file (useful for tracking analyses).",
) -> None:
    """
    Add infotag argument to an argument group for adding key-value pairs to options JSON.

    This function adds an --infotag command line argument to the provided argument group.
    The argument accepts key-value pairs that are stored as additional metadata in the
    options JSON file, which is useful for tracking analysis runs and experiments.

    Parameters
    ----------
    opt_group : argparse._ArgumentGroup
        The argument group to which the infotag argument will be added
    helptext : str, optional
        Help text to display for the infotag argument (default: "Additional key, value pairs to add to the options json file (useful for tracking analyses).")

    Returns
    -------
    None
        This function modifies the argument group in-place and returns None

    Notes
    -----
    The --infotag argument uses action="append" with nargs=2, allowing multiple
    key-value pairs to be specified on the command line. Each pair is stored as
    a tuple (key, value) in the argument group's parsed arguments.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> group = parser.add_argument_group('Options')
    >>> addtagopts(group)
    >>> args = parser.parse_args(['--infotag', 'experiment', 'run1', '--infotag', 'version', '1.0'])
    >>> args.infotag
    [['experiment', 'run1'], ['version', '1.0']]
    """
    opt_group.add_argument(
        "--infotag",
        action="append",
        nargs=2,
        metavar=("tagkey", "tagvalue"),
        help=helptext,
        default=None,
    )


def postprocesstagopts(args: Namespace) -> Namespace:
    """
    Process infotag options and convert them to INFO_ prefixed environment variables.

    This function takes a Namespace object containing command line arguments and
    processes any infotag options by converting them into environment variable
    assignments with INFO_ prefix. The original infotag argument is removed from
    the returned namespace.

    Parameters
    ----------
    args : Namespace
        Command line arguments namespace containing potentially 'infotag' attribute
        which is a list of tuples where each tuple contains (tag_name, tag_value).

    Returns
    -------
    Namespace
        Updated namespace with infotag options converted to INFO_ prefixed
        environment variables. If infotag is None, returns the original args.

    Notes
    -----
    The function modifies the input namespace in-place by converting infotag
    entries from tuples of (tag_name, tag_value) to environment variable
    assignments with INFO_ prefix. For example, if infotag contains
    [('version', '1.0')], the result will have INFO_version='1.0' as a new
    attribute in the namespace.

    Examples
    --------
    >>> from argparse import Namespace
    >>> args = Namespace(infotag=[('version', '1.0'), ('build', '2023')])
    >>> result = postprocesstagopts(args)
    >>> print(result.INFO_version)
    '1.0'
    >>> print(result.INFO_build)
    '2023'
    >>> print(hasattr(result, 'infotag'))
    False
    """
    if args.infotag is not None:
        argvars = vars(args)
        for thetag in argvars["infotag"]:
            argvars[f"INFO_{thetag[0]}"] = thetag[1]
        del argvars["infotag"]
        return Namespace(**argvars)
    else:
        return args


def addnormalizationopts(
    parser: argparse.ArgumentParser,
    normtarget: str = "timecourse",
    defaultmethod: str = DEFAULT_NORMTYPE,
) -> None:
    """
    Add normalization options to an argument parser.

    This function adds a new argument group titled "Normalization options" to the
    provided argument parser. It includes an argument for specifying the normalization
    method to be applied to the specified target.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the normalization options will be added.
    normtarget : str, optional
        The target data type being normalized (default is "timecourse").
    defaultmethod : str, optional
        The default normalization method to use (default is DEFAULT_NORMTYPE).

    Returns
    -------
    None
        This function modifies the parser in-place and returns None.

    Notes
    -----
    The available normalization methods are:

    - "None" - demean only
    - "percent" - divide by mean
    - "variance" - divide by variance
    - "stddev" or "z" - divide by standard deviation
    - "p2p" - divide by range
    - "mad" - divide by median absolute deviation

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addnormalizationopts(parser)
    >>> args = parser.parse_args()
    """
    norm_opts = parser.add_argument_group("Normalization options")
    norm_opts.add_argument(
        "--normmethod",
        dest="normmethod",
        action="store",
        type=str,
        choices=["None", "percent", "variance", "stddev", "z", "p2p", "mad"],
        help=(
            f"Demean and normalize {normtarget} "
            "using one of the following methods: "
            '"None" - demean only; '
            '"percent" - divide by mean; '
            '"variance" - divide by variance; '
            '"stddev" or "z" - divide by standard deviation; '
            '"p2p" - divide by range; '
            '"mad" - divide by median absolute deviation. '
            f'Default is "{defaultmethod}".'
        ),
        default=defaultmethod,
    )


def addversionopts(parser: argparse.ArgumentParser) -> None:
    """
    Add version-related command line arguments to the given argument parser.

    This function adds two version options to the provided argument parser:
    '--version' for simplified version information and '--detailedversion'
    for detailed version information.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which version options will be added.

    Returns
    -------
    None
        This function modifies the parser in-place and does not return anything.

    Notes
    -----
    The version information is retrieved using `tide_util.version()` function.
    The '--version' option displays only the major version number, while
    '--detailedversion' shows the complete version information including
    minor and patch versions.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addversionopts(parser)
    >>> args = parser.parse_args(['--version'])
    """
    version_opts = parser.add_argument_group("Version options")
    version_opts.add_argument(
        "--version",
        action="version",
        help="Show simplified version information and exit",
        version=f"%(prog)s {tide_util.version()[0]}",
    )
    version_opts.add_argument(
        "--detailedversion",
        action="version",
        help="Show detailed version information and exit",
        version=f"%(prog)s {tide_util.version()}",
    )


def addsamplerateopts(parser: argparse.ArgumentParser, details: bool = False) -> None:
    """
    Add sample rate related arguments to an argument parser.

    This function adds mutually exclusive arguments for specifying the sample rate
    of data files. The user can either specify the sample rate directly using
    '--samplerate' or define it indirectly using '--sampletime' (which sets the
    sample rate to 1.0/TSTEP).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the sample rate arguments will be added.
    details : bool, optional
        If True, additional detailed help text may be included (default is False).

    Returns
    -------
    None
        This function modifies the parser in-place and does not return any value.

    Notes
    -----
    The arguments are mutually exclusive - only one of '--samplerate' or '--sampletime'
    can be specified. If neither is specified, the default sample rate is 1.0.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addsamplerateopts(parser)
    >>> args = parser.parse_args(['--samplerate', '44100'])
    >>> args.samplerate
    44100.0

    >>> parser = argparse.ArgumentParser()
    >>> addsamplerateopts(parser)
    >>> args = parser.parse_args(['--sampletime', '0.001'])
    >>> args.samplerate
    1000.0
    """
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        metavar="FREQ",
        type=lambda x: is_float(parser, x),
        help=(
            "Set the sample rate of the data file to FREQ. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )
    sampling.add_argument(
        "--sampletime",
        dest="samplerate",
        action="store",
        metavar="TSTEP",
        type=lambda x: invert_float(parser, x),
        help=(
            "Set the sample rate of the data file to 1.0/TSTEP. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )


def addfilteropts(
    parser: argparse.ArgumentParser,
    filtertarget: str = "timecourses",
    defaultmethod: str = DEFAULT_FILTERBAND,
    details: bool = False,
) -> None:
    """
    Add filtering options to an argument parser for configuring signal filtering.

    This function adds a group of arguments to the provided `argparse.ArgumentParser`
    that allow users to specify filtering parameters for timecourses or other signals.
    It supports predefined filter bands, custom pass/stop frequencies, and various
    filter types and settings.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the filtering options will be added.
    filtertarget : str, optional
        The target of the filtering operation (e.g., "timecourses"). Default is "timecourses".
    defaultmethod : str, optional
        The default filter band to use. Default is `DEFAULT_FILTERBAND`.
    details : bool, optional
        If True, adds additional detailed filtering options such as filter type,
        Butterworth order, padding, and padding type. Default is False.

    Returns
    -------
    None
        This function modifies the parser in place and does not return any value.

    Notes
    -----
    - The `--filterband` argument allows selection of predefined frequency bands.
    - The `--filterfreqs` and `--filterstopfreqs` arguments allow custom frequency
      filtering, with `--filterstopfreqs` requiring `--filterfreqs` to be specified.
    - When `details=True`, additional options such as `--filtertype`, `--butterorder`,
      `--padseconds`, and `--padtype` are added.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addfilteropts(parser, filtertarget="signal", defaultmethod="hrv_lf")
    >>> args = parser.parse_args()
    """
    filt_opts = parser.add_argument_group("Filtering options")
    filt_opts.add_argument(
        "--filterband",
        dest="filterband",
        action="store",
        type=str,
        choices=[
            "None",
            "vlf",
            "lfo",
            "resp",
            "cardiac",
            "hrv_ulf",
            "hrv_vlf",
            "hrv_lf",
            "hrv_hf",
            "hrv_vhf",
            "lfo_legacy",
            "lfo_tight",
        ],
        help=(
            f'Filter {filtertarget} to specific band. Use "None" to disable filtering.  '
            f'Default is "{defaultmethod}".  Ranges are: '
            f'vlf: {tide_filt.getfilterbandfreqs("vlf", asrange=True)}, '
            f'lfo: {tide_filt.getfilterbandfreqs("lfo", asrange=True)}, '
            f'cardiac: {tide_filt.getfilterbandfreqs("cardiac", asrange=True)}, '
            f'hrv_ulf: {tide_filt.getfilterbandfreqs("hrv_ulf", asrange=True)}, '
            f'hrv_vlf: {tide_filt.getfilterbandfreqs("hrv_vlf", asrange=True)}, '
            f'hrv_lf: {tide_filt.getfilterbandfreqs("hrv_lf", asrange=True)}, '
            f'hrv_hf: {tide_filt.getfilterbandfreqs("hrv_hf", asrange=True)}, '
            f'hrv_vhf: {tide_filt.getfilterbandfreqs("hrv_vhf", asrange=True)}, '
            f'lfo_legacy: {tide_filt.getfilterbandfreqs("lfo_legacy", asrange=True)}, '
            f'lfo_tight: {tide_filt.getfilterbandfreqs("lfo_tight", asrange=True)}'
        ),
        default=defaultmethod,
    )
    filt_opts.add_argument(
        "--filterfreqs",
        dest="passvec",
        action="store",
        nargs=2,
        type=float,
        metavar=("LOWERPASS", "UPPERPASS"),
        help=(
            "Filter " + filtertarget + " to retain LOWERPASS to "
            "UPPERPASS. If --filterstopfreqs is not also specified, "
            "LOWERSTOP and UPPERSTOP will be calculated "
            "automatically. "
        ),
        default=None,
    )
    filt_opts.add_argument(
        "--filterstopfreqs",
        dest="stopvec",
        action="store",
        nargs=2,
        type=float,
        metavar=("LOWERSTOP", "UPPERSTOP"),
        help=(
            "Filter " + filtertarget + " to with stop frequencies LOWERSTOP and UPPERSTOP. "
            "LOWERSTOP must be <= LOWERPASS, UPPERSTOP must be >= UPPERPASS. "
            "Using this argument requires the use of --filterfreqs."
        ),
        default=None,
    )
    if details:
        filt_opts.add_argument(
            "--filtertype",
            dest="filtertype",
            action="store",
            type=str,
            choices=["trapezoidal", "brickwall", "butterworth"],
            help=(
                f"Filter {filtertarget} "
                "using a trapezoidal FFT, brickwall FFT, or "
                "butterworth bandpass filter. "
                f'Default is "{DEFAULT_FILTERTYPE}".'
            ),
            default=DEFAULT_FILTERTYPE,
        )
        filt_opts.add_argument(
            "--butterorder",
            dest="filtorder",
            action="store",
            type=int,
            metavar="ORDER",
            help=(
                "Set order of butterworth filter (if used). " f"Default is {DEFAULT_FILTER_ORDER}."
            ),
            default=DEFAULT_FILTER_ORDER,
        )
        filt_opts.add_argument(
            "--padseconds",
            dest="padseconds",
            action="store",
            type=float,
            metavar="SECONDS",
            help=(
                "The number of seconds of padding to add to each end of a "
                "timecourse to be filtered "
                f"to reduce end effects.  Default is {DEFAULT_PAD_SECONDS}."
            ),
            default=DEFAULT_PAD_SECONDS,
        )
        filt_opts.add_argument(
            "--padtype",
            dest="ncfiltpadtype",
            action="store",
            type=str,
            choices=["reflect", "zero", "cyclic", "constant", "constant+"],
            help=(
                f"The type of padding at each end of a "
                "timecourse to be filtered "
                f'to reduce end effects.  Default is "{DEFAULT_PREFILTERPADTYPE}".'
            ),
            default=DEFAULT_PREFILTERPADTYPE,
        )


def postprocesssamplerateopts(args: Namespace, debug: bool = False) -> Namespace:
    """
    Process sample rate options for the application.

    This function handles the sample rate configuration, setting it to a default value
    when "auto" is specified, or using the provided value otherwise.

    Parameters
    ----------
    args : Namespace
        Command line arguments namespace containing the samplerate option
    debug : bool, optional
        Debug flag for additional logging (default is False)

    Returns
    -------
    Namespace
        Updated arguments namespace with processed sample rate value

    Notes
    -----
    When samplerate is set to "auto", it will be converted to 1.0. This is typically
    used as a default fallback value when automatic sample rate detection is not
    implemented or desired.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(samplerate="auto")
    >>> result = postprocesssamplerateopts(args)
    >>> print(result.samplerate)
    1.0

    >>> args = argparse.Namespace(samplerate=44100)
    >>> result = postprocesssamplerateopts(args)
    >>> print(result.samplerate)
    44100
    """
    # set the sample rate
    if args.samplerate == "auto":
        samplerate = 1.0
        args.samplerate = samplerate
    else:
        samplerate = args.samplerate

    return args


def postprocessfilteropts(args: Namespace, debug: bool = False) -> Tuple[Namespace, Any]:
    """
    Post-process filter options and configure a non-causal filter for tidal analysis.

    This function configures the filter parameters based on the input arguments,
    sets up an appropriate filter type (e.g., trapezoidal or arbitrary passband),
    and initializes a `NoncausalFilter` object for subsequent use in tidal data processing.

    Parameters
    ----------
    args : Namespace
        A namespace object containing filter configuration parameters such as
        `filtertype`, `filtorder`, `padseconds`, `prefilterpadtype`, `passvec`,
        `stopvec`, and `filterband`.
    debug : bool, optional
        If True, prints debug information about the filter configuration and
        internal state during processing. Default is False.

    Returns
    -------
    Tuple[Namespace, Any]
        A tuple containing:
        - `args`: The updated namespace with additional attributes like `arbvec`,
          `lowerstop`, `lowerpass`, `upperpass`, and `upperstop`.
        - `theprefilter`: The configured `NoncausalFilter` object used for filtering.

    Notes
    -----
    - If `stopvec` is provided without `passvec`, a `ValueError` is raised.
    - The function supports both predefined filter bands and arbitrary pass/stop
      frequency vectors.
    - The `arbvec` is constructed as `[lowerpass, upperpass, lowerstop, upperstop]`
      for internal use, but the filter expects frequencies in the order
      `[lowerstop, lowerpass, upperpass, upperstop]`.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace()
    >>> args.filtertype = "trapezoidal"
    >>> args.filtorder = 4
    >>> args.padseconds = 10
    >>> args.prefilterpadtype = "constant"
    >>> args.passvec = [0.1, 0.2]
    >>> args.stopvec = None
    >>> args.filterband = "M2"
    >>> args, filter_obj = postprocessfilteropts(args)
    >>> print(args.lowerpass)
    0.1
    """
    # configure the filter
    # set the trapezoidal flag, if using
    try:
        thetype = args.filtertype
    except AttributeError:
        args.filtertype = "trapezoidal"
    try:
        theorder = args.filtorder
    except AttributeError:
        args.filtorder = DEFAULT_FILTER_ORDER
    try:
        thepadseconds = args.padseconds
    except AttributeError:
        args.padseconds = DEFAULT_PAD_SECONDS
    try:
        prefilterpadtype = args.prefilterpadtype
    except AttributeError:
        args.prefilterpadtype = DEFAULT_PREFILTERPADTYPE

    # if passvec, or passvec and stopvec, are set, we are going set up an arbpass filter
    args.arbvec = None
    if debug:
        print("before preprocessing")
        print("\targs.arbvec:", args.arbvec)
        print("\targs.passvec:", args.passvec)
        print("\targs.stopvec:", args.stopvec)
        print("\targs.filterband:", args.filterband)
    if args.stopvec is not None:
        if args.passvec is not None:
            args.arbvec = [args.passvec[0], args.passvec[1], args.stopvec[0], args.stopvec[1]]
        else:
            raise ValueError("--filterfreqs must be used if --filterstopfreqs is specified")
    else:
        if args.passvec is not None:
            args.arbvec = [
                args.passvec[0],
                args.passvec[1],
                args.passvec[0] * 0.95,
                args.passvec[1] * 1.05,
            ]
    if args.arbvec is not None:
        # NOTE - this vector is LOWERPASS, UPPERPASS, LOWERSTOP, UPPERSTOP
        # setfreqs expects LOWERSTOP, LOWERPASS, UPPERPASS, UPPERSTOP
        theprefilter = tide_filt.NoncausalFilter(
            "arb",
            transferfunc=args.filtertype,
            padtime=args.padseconds,
            padtype=args.prefilterpadtype,
        )
        theprefilter.setfreqs(args.arbvec[2], args.arbvec[0], args.arbvec[1], args.arbvec[3])
    else:
        theprefilter = tide_filt.NoncausalFilter(
            args.filterband,
            transferfunc=args.filtertype,
            padtime=args.padseconds,
            padtype=args.prefilterpadtype,
        )

    # set the butterworth order
    theprefilter.setbutterorder(args.filtorder)

    if debug:
        print("before preprocessing")
        print("\targs.arbvec:", args.arbvec)
        print("\targs.passvec:", args.passvec)
        print("\targs.stopvec:", args.stopvec)
        print("\targs.filterband:", args.filterband)

    (
        args.lowerstop,
        args.lowerpass,
        args.upperpass,
        args.upperstop,
    ) = theprefilter.getfreqs()

    if debug:
        print("after getfreqs")
        print("\targs.arbvec:", args.arbvec)

    return args, theprefilter


def addwindowopts(parser: argparse.ArgumentParser, windowtype: str = DEFAULT_WINDOWFUNC) -> None:
    """
    Add windowing options to an argument parser for correlation analysis.

    This function adds command-line arguments related to windowing operations
    that can be applied prior to correlation calculations. These options include
    the choice of window function and zero padding strategy.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which windowing options will be added.
    windowtype : str, optional
        The default window function to use. Default is DEFAULT_WINDOWFUNC.

    Returns
    -------
    None
        This function modifies the parser in-place and returns nothing.

    Notes
    -----
    The windowing options are added to a dedicated argument group called "Windowing options".
    The zero padding parameter allows for control over edge artifacts in correlation calculations,
    with negative values triggering automatic padding selection.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addwindowopts(parser)
    >>> args = parser.parse_args(['--windowfunc', 'hann'])
    >>> args.windowfunc
    'hann'
    """
    wfunc = parser.add_argument_group("Windowing options")
    wfunc.add_argument(
        "--windowfunc",
        dest="windowfunc",
        action="store",
        type=str,
        choices=["hamming", "hann", "blackmanharris", "None"],
        help=(
            "Window function to use prior to correlation. "
            "Options are hamming, hann, "
            f"blackmanharris, and None. Default is {windowtype}"
        ),
        default=windowtype,
    )
    wfunc.add_argument(
        "--zeropadding",
        dest="zeropadding",
        action="store",
        type=int,
        metavar="PADVAL",
        help=(
            "Pad input functions to correlation with PADVAL zeros on each side. "
            "A PADVAL of 0 does circular correlations, positive values reduce edge artifacts. "
            f"Set PADVAL < 0 to set automatically. Default is {DEFAULT_PADVAL}."
        ),
        default=DEFAULT_PADVAL,
    )


def addplotopts(parser: argparse.ArgumentParser, multiline: bool = True) -> None:
    """
    Add general plot appearance options to an argument parser.

    This function adds a set of arguments to the provided `argparse.ArgumentParser`
    object that control the appearance of plots, including titles, axis labels,
    legends, colors, line widths, and output options.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the plot options will be added.
    multiline : bool, optional
        If True, allows multiple legends and colors to be specified as comma-separated
        lists. If False, expects single legend and color values. Default is True.

    Returns
    -------
    None
        This function modifies the parser in place and does not return any value.

    Notes
    -----
    The arguments added by this function are grouped under "General plot appearance options".
    The `multiline` parameter affects the behavior of the `--legends`, `--colors`, and
    `--linewidth` arguments:

    - When `multiline=True`, these arguments accept comma-separated lists.
    - When `multiline=False`, they expect single values.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addplotopts(parser, multiline=True)
    >>> args = parser.parse_args()
    """
    plotopts = parser.add_argument_group("General plot appearance options")
    plotopts.add_argument(
        "--title",
        dest="thetitle",
        metavar="TITLE",
        type=str,
        action="store",
        help="Use TITLE as the overall title of the graph.",
        default=None,
    )
    plotopts.add_argument(
        "--xlabel",
        dest="xlabel",
        metavar="LABEL",
        type=str,
        action="store",
        help="Label for the plot x axis.",
        default=None,
    )
    plotopts.add_argument(
        "--ylabel",
        dest="ylabel",
        metavar="LABEL",
        type=str,
        action="store",
        help="Label for the plot y axis.",
        default=None,
    )
    if multiline:
        plotopts.add_argument(
            "--legends",
            dest="legends",
            metavar="LEGEND[,LEGEND[,LEGEND...]]",
            type=str,
            action="store",
            help="Comma separated list of legends for each timecourse.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--legend",
            dest="legends",
            metavar="LEGEND",
            type=str,
            action="store",
            help="Legends for the timecourse.",
            default=None,
        )
    plotopts.add_argument(
        "--legendloc",
        dest="legendloc",
        metavar="LOC",
        type=int,
        action="store",
        help=(
            "Integer from 0 to 10 inclusive specifying legend location.  Legal values are: "
            "0: best, 1: upper right, 2: upper left, 3: lower left, 4: lower right, "
            "5: right, 6: center left, 7: center right, 8: lower center, 9: upper center, "
            "10: center.  Default is 2."
        ),
        default=2,
    )
    if multiline:
        plotopts.add_argument(
            "--colors",
            dest="colors",
            metavar="COLOR[,COLOR[,COLOR...]]",
            type=str,
            action="store",
            help="Comma separated list of colors for each timecourse.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--color",
            dest="colors",
            metavar="COLOR",
            type=str,
            action="store",
            help="Color of the timecourse plot.",
            default=None,
        )
    plotopts.add_argument(
        "--nolegend",
        dest="dolegend",
        action="store_false",
        help="Turn off legend label.",
        default=True,
    )
    plotopts.add_argument(
        "--noxax",
        dest="showxax",
        action="store_false",
        help="Do not show x axis.",
        default=True,
    )
    plotopts.add_argument(
        "--noyax",
        dest="showyax",
        action="store_false",
        help="Do not show y axis.",
        default=True,
    )
    if multiline:
        plotopts.add_argument(
            "--linewidth",
            dest="linewidths",
            metavar="LINEWIDTH[,LINEWIDTH[,LINEWIDTH...]]",
            type=str,
            help="A comma separated list of linewidths (in points) for plots.  Default is 1.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--linewidth",
            dest="linewidths",
            metavar="LINEWIDTH",
            type=str,
            help="Linewidth (in points) for plot.  Default is 1.",
            default=None,
        )
    plotopts.add_argument(
        "--tofile",
        dest="outputfile",
        metavar="FILENAME",
        type=str,
        action="store",
        help="Write figure to file FILENAME instead of displaying on the screen.",
        default=None,
    )
    plotopts.add_argument(
        "--fontscalefac",
        dest="fontscalefac",
        metavar="FAC",
        type=float,
        action="store",
        help="Scaling factor for annotation fonts (default is 1.0).",
        default=1.0,
    )
    plotopts.add_argument(
        "--saveres",
        dest="saveres",
        metavar="DPI",
        type=int,
        action="store",
        help="Write figure to file at DPI dots per inch (default is 1000).",
        default=1000,
    )


def addpermutationopts(parser: argparse.ArgumentParser, numreps: int = 10000) -> None:
    """
    Add permutation testing options to an argument parser.

    This function adds command-line arguments for configuring permutation-based
    significance testing, including the permutation method and number of null
    correlations to compute.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the permutation options will be added.
    numreps : int, optional
        Number of null correlations to run for significance estimation.
        Default is 10000. Set to 0 to disable significance testing.

    Returns
    -------
    None
        This function modifies the parser in-place and does not return any value.

    Notes
    -----
    The function adds two main arguments to the parser:
    1. ``--permutationmethod``: Specifies the permutation method ('shuffle' or 'phaserandom')
    2. ``--numnull``: Sets the number of null correlations for significance testing

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addpermutationopts(parser, numreps=5000)
    >>> args = parser.parse_args()
    """
    sigcalc_opts = parser.add_argument_group("Significance calculation options")
    permutationmethod = sigcalc_opts.add_mutually_exclusive_group()
    permutationmethod.add_argument(
        "--permutationmethod",
        dest="permutationmethod",
        action="store",
        type=str,
        choices=["shuffle", "phaserandom"],
        help=(
            "Permutation method for significance testing. "
            f'Default is "{DEFAULT_PERMUTATIONMETHOD}".'
        ),
        default=DEFAULT_PERMUTATIONMETHOD,
    )
    sigcalc_opts.add_argument(
        "--numnull",
        dest="numestreps",
        action="store",
        type=int,
        metavar="NREPS",
        help=(
            "Estimate significance threshold by running "
            f"NREPS null correlations (default is {numreps}, "
            "set to 0 to disable). "
        ),
        default=numreps,
    )


def addsearchrangeopts(
    parser: argparse.ArgumentParser,
    details: bool = False,
    defaultmin: float = -30.0,
    defaultmax: float = 30.0,
) -> None:
    """
    Add search range options to an argument parser for lag fitting.

    This function adds command-line arguments for specifying the range of lags
    to consider during fitting operations. It provides options for setting
    the minimum and maximum lag values, as well as an optional fixed delay
    parameter.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the search range options will be added.
    details : bool, optional
        If True, adds additional detailed options including --fixdelay.
        Default is False.
    defaultmin : float, optional
        The default minimum lag value in seconds. Default is -30.0.
    defaultmax : float, optional
        The default maximum lag value in seconds. Default is 30.0.

    Returns
    -------
    None
        This function modifies the parser in-place and returns None.

    Notes
    -----
    The function adds two main arguments:
    - --searchrange: Specifies the lag range from LAGMIN to LAGMAX
    - --fixdelay: When details=True, sets a fixed delay time for all voxels

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addsearchrangeopts(parser)
    >>> args = parser.parse_args(['--searchrange', '-10', '10'])

    >>> addsearchrangeopts(parser, details=True)
    >>> args = parser.parse_args(['--searchrange', '-5', '5', '--fixdelay', '2.0'])
    """
    parser.add_argument(
        "--searchrange",
        dest="lag_extrema",
        action=IndicateSpecifiedAction,
        nargs=2,
        type=float,
        metavar=("LAGMIN", "LAGMAX"),
        help=(
            "Limit fit to a range of lags from LAGMIN to "
            "LAGMAX.  Default is -30.0 to 30.0 seconds. "
        ),
        default=(defaultmin, defaultmax),
    )
    if details:
        parser.add_argument(
            "--fixdelay",
            dest="initialdelayvalue",
            action="store",
            type=float,
            metavar="DELAYTIME",
            help=("Don't fit the delay time - set it to " "DELAYTIME seconds for all voxels. "),
            default=None,
        )


def postprocesssearchrangeopts(args: Namespace) -> Namespace:
    """
    Post-process search range options for delay estimation.

    This function handles additional argument parsing for delay estimation parameters
    that cannot be handled by argparse alone. It processes initial delay values and
    sets appropriate lag extrema for search range optimization.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing command line arguments. Expected attributes include:
        - initialdelayvalue : float, optional
            Initial delay value to fix for delay estimation
        - lag_extrema : tuple of float
            Tuple containing (lag_min, lag_max) for search range
        - lag_extrema_nondefault : tuple of float, optional
            Non-default lag extrema values to override defaults

    Returns
    -------
    argparse.Namespace
        Updated args namespace with processed delay and search range parameters.
        Adds/updates the following attributes:
        - fixdelay : bool
            Flag indicating if delay was fixed
        - lagmin : float
            Minimum lag value for search range
        - lagmax : float
            Maximum lag value for search range
        - lagmin_nondefault : bool
            Flag indicating if lagmin was overridden from default
        - lagmax_nondefault : bool
            Flag indicating if lagmax was overridden from default

    Notes
    -----
    If initialdelayvalue is provided, the function sets fixdelay=True and creates
    a search range centered around the initial delay value with ±10.0 range.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace()
    >>> args.initialdelayvalue = 5.0
    >>> args.lag_extrema = (-10.0, 10.0)
    >>> result = postprocesssearchrangeopts(args)
    >>> print(result.fixdelay)
    True
    >>> print(result.lagmin)
    -5.0
    >>> print(result.lagmax)
    15.0
    """
    # Additional argument parsing not handled by argparse
    # first handle fixed delay
    try:
        test = args.initialdelayvalue
    except:
        args.initialdelayvalue = None
    if args.initialdelayvalue is not None:
        args.fixdelay = True
        args.lag_extrema = (args.initialdelayvalue - 10.0, args.initialdelayvalue + 10.0)
    else:
        args.fixdelay = False

    # now set the extrema
    try:
        test = args.lag_extrema_nondefault
        args.lagmin_nondefault = True
        args.lagmax_nondefault = True
    except AttributeError:
        pass
    args.lagmin = args.lag_extrema[0]
    args.lagmax = args.lag_extrema[1]
    return args


def addtimerangeopts(parser: argparse.ArgumentParser) -> None:
    """
    Add time range arguments to an argument parser for timepoint filtering.

    This function adds a command-line argument '--timerange' that allows users
    to specify a range of timepoints to include in the analysis. The argument
    accepts two integer values representing the start and end timepoints.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the time range arguments will be added.

    Returns
    -------
    None
        This function modifies the parser in-place and does not return any value.

    Notes
    -----
    - The time range is inclusive on both ends
    - If END is set to -1, analysis will continue to the last timepoint
    - Negative values for START will be automatically set to 0
    - Default behavior (no timerange specified) is to use all timepoints
    - Both START and END must be non-negative integers when not equal to -1

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addtimerangeopts(parser)
    >>> args = parser.parse_args(['--timerange', '10', '50'])
    >>> args.timerange
    (10, 50)

    >>> args = parser.parse_args(['--timerange', '0', '-1'])
    >>> args.timerange
    (0, -1)
    """
    parser.add_argument(
        "--timerange",
        dest="timerange",
        action="store",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "Limit analysis to data between timepoints "
            "START and END in the input file. If END is set to -1, "
            "analysis will go to the last timepoint.  Negative values "
            "of START will be set to 0. Default is to use all timepoints."
        ),
        default=(-1, -1),
    )


def postprocesstimerangeopts(args: Namespace) -> Namespace:
    """
    Process timerange options and set start and end points for time range filtering.

    This function takes a namespace object containing timerange information and
    converts it into startpoint and endpoint values that can be used for time
    range filtering operations. The timerange is expected to be a list or tuple
    with two elements: [start_time, end_time].

    Parameters
    ----------
    args : Namespace
        A namespace object containing a 'timerange' attribute. The timerange
        should be a list or tuple of two elements: [start_time, end_time].
        The end_time can be -1 to indicate no upper bound.

    Returns
    -------
    Namespace
        The same namespace object with added 'startpoint' and 'endpoint' attributes
        set based on the timerange values.

    Notes
    -----
    - If the end_time in timerange is -1, the endpoint is set to 10000000000
    - Both startpoint and endpoint are converted to integers
    - The function modifies the input namespace in-place and returns it

    Examples
    --------
    >>> args = Namespace()
    >>> args.timerange = [100, 200]
    >>> result = postprocesstimerangeopts(args)
    >>> print(result.startpoint)
    100
    >>> print(result.endpoint)
    200

    >>> args = Namespace()
    >>> args.timerange = [50, -1]
    >>> result = postprocesstimerangeopts(args)
    >>> print(result.startpoint)
    50
    >>> print(result.endpoint)
    10000000000
    """
    args.startpoint = int(args.timerange[0])
    if args.timerange[1] == -1:
        args.endpoint = 10000000000
    else:
        args.endpoint = int(args.timerange[1])
    return args


def parserange(
    timerange: Tuple[int, int], descriptor: str = "timerange", debug: bool = False
) -> Tuple[int, int]:
    """
    Parse and validate a time range tuple, ensuring valid start and end points.

    This function processes a time range tuple by handling negative values and
    validating the relationship between start and end points. Negative start values
    are set to 0, and negative end values are set to 100000000. The function ensures
    that the start point is strictly less than the end point.

    Parameters
    ----------
    timerange : tuple of int
        A tuple containing (start, end) time points. Negative values are handled
        according to the function's logic.
    descriptor : str, optional
        A descriptive name used in error messages, default is "timerange".
    debug : bool, optional
        If True, prints debug information including startpoint, endpoint, and
        original timerange values, default is False.

    Returns
    -------
    tuple of int
        A validated tuple containing (startpoint, endpoint) where both values
        are non-negative and startpoint < endpoint.

    Raises
    ------
    ValueError
        If the calculated startpoint is greater than or equal to the calculated
        endpoint.

    Examples
    --------
    >>> parserange((10, 20))
    (10, 20)

    >>> parserange((-5, 15))
    (0, 15)

    >>> parserange((10, -5))
    (10, 100000000)

    >>> parserange((20, 10))
    ValueError: timerange startpoint must be < endpoint
    """
    if timerange[0] < 0:
        startpoint = 0
    else:
        startpoint = timerange[0]
    if timerange[1] < 0:
        endpoint = 100000000
    else:
        endpoint = timerange[1]
    if debug:
        print("startpoint:", startpoint)
        print("endpoint:", endpoint)
        print("timerange:", timerange)
    if endpoint <= startpoint:
        raise ValueError(f"{descriptor} startpoint must be < endpoint")
    return startpoint, endpoint


def addsimilarityopts(parser: argparse.ArgumentParser) -> None:
    """
    Add mutual information similarity options to an argument parser.

    This function adds a command-line argument for specifying the time constant
    of temporal smoothing to apply to the mutual information function. The
    smoothing helps reduce noise in the similarity calculations.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the mutual information smoothing options will be added.

    Returns
    -------
    None
        This function modifies the parser in-place and does not return any value.

    Notes
    -----
    The smoothing time constant (TAU) controls the temporal smoothing of the
    mutual information function. When TAU <= 0.0, smoothing is disabled.
    Default value is 3.0 seconds.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> addsimilarityopts(parser)
    >>> args = parser.parse_args(['--mutualinfosmoothingtime', '2.5'])
    >>> args.smoothingtime
    2.5
    """
    parser.add_argument(
        "--mutualinfosmoothingtime",
        dest="smoothingtime",
        action="store",
        type=float,
        metavar="TAU",
        help=(
            "Time constant of a temporal smoothing function to apply to the "
            "mutual information function. "
            "Default is 3.0 seconds.  TAU <=0.0 disables smoothing."
        ),
        default=3.0,
    )


def setargs(
    thegetparserfunc: Callable[[], argparse.ArgumentParser],
    inputargs: Optional[List[str]] = None,
) -> Tuple[Namespace, List[str]]:
    """
    Compile arguments for rapidtide workflow.

    This function processes command line arguments or a provided argument list
    using an argument parser function. It handles both interactive command line
    argument parsing and programmatic argument parsing with error handling.

    Parameters
    ----------
    thegetparserfunc : callable
        A function that returns an argparse.ArgumentParser object. This function
        should not take any arguments and should return a configured parser.
    inputargs : list of str, optional
        List of arguments to parse. If None (default), arguments are parsed
        from sys.argv. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - args : argparse.Namespace
          Parsed arguments as a namespace object
        - argstowrite : list of str
          The argument list that was parsed (either sys.argv or inputargs)

    Raises
    ------
    SystemExit
        If the argument parser encounters an error or --help option is used.

    Notes
    -----
    This function provides a unified interface for argument parsing in the
    rapidtide workflow, supporting both command line and programmatic usage.
    When --help is invoked, a helpful message is displayed before raising
    SystemExit.

    Examples
    --------
    >>> from argparse import ArgumentParser
    >>> def get_parser():
    ...     parser = ArgumentParser()
    ...     parser.add_argument('--input', help='Input file')
    ...     return parser
    >>> args, arglist = setargs(get_parser)
    >>> args.input
    None

    >>> args, arglist = setargs(get_parser, ['--input', 'test.txt'])
    >>> args.input
    'test.txt'
    """
    if inputargs is None:
        # get arguments from the command line
        # LGR.info("processing command line arguments")
        try:
            args = thegetparserfunc().parse_args()
            argstowrite = sys.argv
        except SystemExit:
            print("Use --help option for detailed information on options.")
            raise
    else:
        # get arguments from the passed list
        # LGR.info("processing passed argument list:")
        # LGR.info(inputargs)
        try:
            args = thegetparserfunc().parse_args(inputargs)
            argstowrite = inputargs
        except SystemExit:
            print("Use --help option for detailed information on options.")
            raise

    return args, argstowrite


def generic_init(
    theparser: Callable[[], argparse.ArgumentParser],
    themain: Callable[[Namespace], None],
    inputargs: Optional[List[str]] = None,
) -> None:
    """
    Compile arguments either from the command line, or from an argument list.

    This function serves as a generic initialization wrapper that handles
    argument parsing for command-line applications. It can parse arguments
    from either sys.argv or a provided list of arguments, and then executes
    the main function with the parsed arguments.

    Parameters
    ----------
    theparser : Callable[[], argparse.ArgumentParser]
        A callable that returns an argparse.ArgumentParser instance
        configured with the desired command-line arguments.
    themain : Callable[[Namespace], None]
        A callable that takes a parsed arguments Namespace object
        and performs the main application logic.
    inputargs : List[str], optional
        List of argument strings to parse. If None, arguments are parsed
        from sys.argv. Default is None.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function automatically saves the raw command line as an attribute
    `commandline` on the parsed arguments object for later use.

    Examples
    --------
    >>> def create_parser():
    ...     parser = argparse.ArgumentParser()
    ...     parser.add_argument('--verbose', action='store_true')
    ...     return parser
    ...
    >>> def main_func(args):
    ...     print(f"Verbose mode: {args.verbose}")
    ...
    >>> generic_init(create_parser, main_func)
    """
    if inputargs is None:
        print("processing command line arguments")
        # write out the command used
        try:
            args = theparser().parse_args()
            argstowrite = sys.argv
        except SystemExit:
            print("Use --help option for detailed information on options.")
            raise
    else:
        print("processing passed argument list:")
        try:
            args = theparser().parse_args(inputargs)
            argstowrite = inputargs
        except SystemExit:
            print("Use --help option for detailed information on options.")
            raise

    # save the raw and formatted command lines
    args.commandline = " ".join(argstowrite)

    themain(args)
