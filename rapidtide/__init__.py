# -*- coding: latin-1 -*-
#from .version import __version__  # noqa
#from ._gittag import __gittag__

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
