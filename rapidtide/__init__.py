from .version import __version__  # noqa
from ._gittag import __gittag__
from .tide_funcs import *  # noqa
try:
    from .tidepoolTemplate_qt5 import *  # noqa
except:
    from .tidepoolTemplate_qt4 import *  # noqa
from .OrthoImageItem import *  # noqa
