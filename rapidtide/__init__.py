from .version import __version__  # noqa
from .tide_funcs import *  # noqa
import os

try:
    from _version import __version__ as v
    __version__ = v
    del v
except ImportError:
    __version__ = "UNKNOWN"

#from .tidepoolTemplate import *  # noqa
#from .OrthoImageItem import *  # noqa
