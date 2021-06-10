"""Utility functions for rapidtide workflows."""
import logging
import os

LGR = logging.getLogger("GENERAL")
TimingLGR = logging.getLogger("TIMING")
MemoryLGR = logging.getLogger("MEMORY")

starttime = None


class ContextFilter(logging.Filter):
    """A filter to allow specific logging handlers to ignore specific loggers.

    We use this to prevent our secondary loggers from printing to the general log file or to
    stdout.
    """

    NAMES = {"TIMING", "MEMORY"}

    def filter(self, record):
        if not any([n in record.name for n in self.NAMES]):
            return True


class TimingFormatter(logging.Formatter):
    """A formatter to allow optional extra fields (message2 and message3) in a logger.

    The fields must be passed as a dictionary, without a keyword.
    """

    def format(self, record):
        if isinstance(record.args, dict):
            record.message2 = record.args.get("message2", None)
            record.message3 = record.args.get("message3", None)
        else:
            record.message2 = None
            record.message3 = None
        return super().format(record)


def setup_logger(logger_filename, timing_filename, memory_filename, verbose=False, debug=False):
    """Set up a set of loggers.

    Parameters
    ----------
    logger_filename : str
        Output file for generic logging information.
    timing_filename : str
        Output file for timing-related information.
    memory_filename : str
        Output file for memory usage-related information.
    verbose : bool, optional
        Sets the target logging level to VERBOSE (a custom level between INFO and DEBUG).
        Is overridden by ``debug``, if ``debug = True``.
        Default is False.
    debug : bool, optional
        Sets the target logging level to DEBUG. Default is False.
    """
    # Clean up existing files from previous runs
    for fname in [logger_filename, timing_filename, memory_filename]:
        if os.path.isfile(fname):
            LGR.info(f"Removing existing file: {fname}")
            os.remove(fname)

    # Create a new "verbose" logging level
    VERBOSE_LEVEL = 15  # between info and debug
    logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

    def verbose(self, message, *args, **kwargs):
        if self.isEnabledFor(VERBOSE_LEVEL):
            self._log(VERBOSE_LEVEL, message, args, **kwargs)

    logging.Logger.verbose = verbose

    # Set logging level for main logger
    if debug:
        logging.root.setLevel(logging.DEBUG)
    elif verbose:
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
    memory_formatter = logging.Formatter("%(message)s")
    memory_handler = logging.FileHandler(memory_filename)
    memory_handler.setFormatter(memory_formatter)
    MemoryLGR.setLevel(logging.INFO)
    MemoryLGR.addHandler(memory_handler)
    MemoryLGR.propagate = False  # do not print to console
