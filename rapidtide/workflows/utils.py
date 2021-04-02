"""Utility functions for rapidtide workflows."""
import logging
import os

LGR = logging.getLogger(__name__)
TimingLGR = logging.getLogger("TIMING")
MemoryLGR = logging.getLogger("MEMORY")


class ContextFilter(logging.Filter):
    """A filter to allow specific logging handlers to ignore specific loggers.

    We use this to prevent our report-generation and reference-compiling
    loggers from printing to the general log file or to stdout.
    """

    NAMES = {"TIMING", "MEMORY"}

    def filter(self, record):
        if not any([n in record.name for n in self.NAMES]):
            return True


def setup_logger(logger_filename, timing_filename, memory_filename, verbose=False, debug=False):
    """Set up a logger."""
    for fname in [logger_filename, timing_filename, memory_filename]:
        if os.path.isfile(fname):
            LGR.info(f"Removing existing file: {fname}")
            os.remove(fname)

    # set logging format
    log_formatter = logging.Formatter(
        "%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    timing_formatter = logging.Formatter(
        "%(asctime)s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    memory_formatter = logging.Formatter("%(message)s")

    # Create a new "verbose" logging level
    VERBOSE_LEVEL = 15  # between info and debug
    logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

    def verbose(self, message, *args, **kwargs):
        if self.isEnabledFor(VERBOSE_LEVEL):
            self._log(VERBOSE_LEVEL, message, args, **kwargs)

    logging.Logger.verbose = verbose

    # set up logging file and open it for writing
    log_handler = logging.FileHandler(logger_filename)
    log_handler.setFormatter(log_formatter)

    # Removing handlers after basicConfig doesn't work, so we use filters
    # for the relevant handlers themselves.
    log_handler.addFilter(ContextFilter())
    logging.root.addHandler(log_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(ContextFilter())
    logging.root.addHandler(stream_handler)

    if debug:
        logging.root.setLevel(logging.DEBUG)
    elif verbose:
        logging.root.setLevel(VERBOSE_LEVEL)
    else:
        logging.root.setLevel(logging.INFO)

    # Loggers for timing and memory usage
    timing_handler = logging.FileHandler(timing_filename)
    timing_handler.setFormatter(timing_formatter)
    TimingLGR.setLevel(logging.INFO)
    TimingLGR.addHandler(timing_handler)
    TimingLGR.propagate = False  # do not print to console

    memory_handler = logging.FileHandler(memory_filename)
    memory_handler.setFormatter(memory_formatter)
    MemoryLGR.setLevel(logging.INFO)
    MemoryLGR.addHandler(memory_handler)
    MemoryLGR.propagate = False  # do not print to console
