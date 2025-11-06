#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2021-2025 Blaise Frederick
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
"""Utility functions for rapidtide workflows."""
import logging
import os
from typing import Any, Optional

LGR = logging.getLogger("GENERAL")
TimingLGR = logging.getLogger("TIMING")
MemoryLGR = logging.getLogger("MEMORY")
ErrorLGR = logging.getLogger("ERROR")

starttime = None


class ContextFilter(logging.Filter):
    """
    A filter to allow specific logging handlers to ignore specific loggers.

    We use this to prevent our secondary loggers from printing to the general log file or to
    stdout.
    """

    NAMES = {"TIMING", "MEMORY"}

    def filter(self, record: logging.LogRecord) -> bool:
        if not any([n in record.name for n in self.NAMES]):
            return True
        return False


class TimingFormatter(logging.Formatter):
    """
    Determine whether to filter a log record based on its name.

    This filter checks if any of the names in `self.NAMES` are contained
    within the log record's name. If none of the names are found, the record
    is allowed through (returns True). If any name is found, the record is
    filtered out (returns False).

    Parameters
    ----------
    record : logging.LogRecord
        The log record to be filtered.

    Returns
    -------
    bool
        True if the record should be allowed through (not filtered),
        False if the record should be filtered out.

    Notes
    -----
    The filtering is performed using substring matching. If `self.NAMES` is
    empty, all records will be allowed through.

    Examples
    --------
    >>> class MyFilter:
    ...     NAMES = ['DEBUG', 'INFO']
    ...     def filter(self, record):
    ...         if not any([n in record.name for n in self.NAMES]):
    ...             return True
    ...         return False
    ...
    >>> filter_instance = MyFilter()
    >>> record = logging.LogRecord('DEBUG', 10, 'file.py', 1, 'message', None, None)
    >>> filter_instance.filter(record)
    False
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record, extracting message2 and message3 from record.args.

        This method extends the standard logging formatter by checking if the log record's
        args attribute is a dictionary. If so, it extracts "message2" and "message3"
        values from the dictionary and assigns them as attributes to the record for
        use in log formatting.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted. The record's args attribute is checked
            to determine if message2 and message3 should be extracted.

        Returns
        -------
        str
            The formatted log message string.

        Notes
        -----
        This method modifies the record in-place by adding message2 and message3
        attributes when record.args is a dictionary. If record.args is not a
        dictionary, these attributes are set to None.

        Examples
        --------
        >>> import logging
        >>> formatter = MyFormatter()
        >>> record = logging.LogRecord('name', 1, 'file', 1, 'message', {'message2': 'extra', 'message3': 'info'}, None)
        >>> formatted = formatter.format(record)
        >>> print(formatted)
        'message'
        """
        if isinstance(record.args, dict):
            record.message2 = record.args.get("message2", None)
            record.message3 = record.args.get("message3", None)
        else:
            record.message2 = None
            record.message3 = None
        return super().format(record)


def setup_logger(
    logger_filename: str,
    timing_filename: str,
    memory_filename: Optional[str] = None,
    error_filename: Optional[str] = None,
    isverbose: bool = False,
    debug: bool = False,
) -> None:
    """
    Set up a set of loggers for handling generic, timing, memory, and error logging.

    This function configures multiple loggers with custom handlers and formatters,
    writing output to specified files. It also sets up a custom "VERBOSE" logging level
    and supports optional console output filtering.

    Parameters
    ----------
    logger_filename : str
        Output file for generic logging information.
    timing_filename : str
        Output file for timing-related information.
    memory_filename : str, optional
        Output file for memory usage-related information. If not provided, memory logging is disabled.
    error_filename : str, optional
        Output file for error-related information. If not provided, error logging is disabled.
    isverbose : bool, optional
        Sets the target logging level to VERBOSE (a custom level between INFO and DEBUG).
        Is overridden by ``debug``, if ``debug = True``. Default is False.
    debug : bool, optional
        Sets the target logging level to DEBUG. Default is False.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    - Existing log files from previous runs are automatically removed.
    - A custom logging level "VERBOSE" (level 15) is added, placed between INFO and DEBUG.
    - Console output is filtered using a `ContextFilter` to avoid duplicate messages.
    - Loggers for timing, memory, and error are configured only if their respective filenames are provided.

    Examples
    --------
    >>> setup_logger(
    ...     logger_filename="app.log",
    ...     timing_filename="timing.log",
    ...     memory_filename="memory.log",
    ...     error_filename="error.log",
    ...     isverbose=True,
    ...     debug=False
    ... )
    """
    # Clean up existing files from previous runs
    for fname in [logger_filename, timing_filename, memory_filename, error_filename]:
        if fname is not None:
            if os.path.isfile(fname):
                LGR.info(f"Removing existing file: {fname}")
                os.remove(fname)

    # Create a new "verbose" logging level
    VERBOSE_LEVEL = 15  # between info and debug
    logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

    def verbose(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a message with VERBOSE_LEVEL if it is enabled.

        This method provides a verbose logging capability that only executes
        when the logger's verbosity level is set to include VERBOSE_LEVEL.

        Parameters
        ----------
        self : logging.Logger
            The logger instance calling this method.
        message : str
            The message to be logged. This message may contain placeholders
            for additional arguments.
        *args : Any
            Additional arguments to be formatted into the message.
        **kwargs : Any
            Additional keyword arguments to be passed to the logging system.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        This method is a convenience wrapper around the standard logging system
        that only executes when VERBOSE_LEVEL is enabled. It allows for
        conditional verbose logging without requiring explicit checks in the
        calling code.

        Examples
        --------
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.setLevel(logging.DEBUG)
        >>> logger.verbose("Processing item %s", item_name)
        """
        if self.isEnabledFor(VERBOSE_LEVEL):
            self._log(VERBOSE_LEVEL, message, args, **kwargs)

    logging.Logger.verbose = verbose

    # Set logging level for main logger
    if debug:
        logging.root.setLevel(logging.DEBUG)
    elif isverbose:
        logging.root.setLevel(VERBOSE_LEVEL)
    else:
        logging.root.setLevel(logging.INFO)

    # Set up handler for main logger's output file
    log_formatter = logging.Formatter("%(message)s")
    log_handler = logging.FileHandler(logger_filename)
    log_handler.setFormatter(log_formatter)

    # A handler for the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)

    # Removing handlers after basicConfig doesn't work, so we use filters
    # for the relevant handlers themselves.
    log_handler.addFilter(ContextFilter())
    stream_handler.addFilter(ContextFilter())

    LGR.addHandler(log_handler)
    LGR.addHandler(stream_handler)
    LGR.propagate = False  # do not print to console, except for messages for StreamHandler

    # A timing logger
    timing_formatter = TimingFormatter(
        "%(asctime)s.%(msecs)03d\t%(message)s\t%(message2)s\t%(message3)s",
        datefmt="%Y%m%dT%H%M%S",
    )
    timing_handler = logging.FileHandler(timing_filename)
    timing_handler.setFormatter(timing_formatter)
    TimingLGR.setLevel(logging.INFO)
    TimingLGR.addHandler(timing_handler)
    TimingLGR.propagate = False  # do not print to console

    # A memory logger
    if memory_filename is not None:
        memory_formatter = logging.Formatter("%(message)s")
        memory_handler = logging.FileHandler(memory_filename)
        memory_handler.setFormatter(memory_formatter)
        MemoryLGR.setLevel(logging.INFO)
        MemoryLGR.addHandler(memory_handler)
        MemoryLGR.propagate = False  # do not print to console

    # An error logger
    if error_filename is not None:
        error_formatter = logging.Formatter("%(message)s")
        error_handler = logging.FileHandler(error_filename)
        error_handler.setFormatter(error_formatter)
        ErrorLGR.setLevel(logging.INFO)
        ErrorLGR.addHandler(error_handler)
        ErrorLGR.propagate = False  # do not print to console
