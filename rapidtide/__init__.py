from .version import __version__  # noqa
from ._gittag import __gittag__
# from .tide_funcs import *  # noqa
from .correlate import autocorrcheck
from .filter import padvec
from .fit import gaussresidualssk
from .io import readfromnifti
from .miscmath import phase
from .stats import printthresholds
from .util import checkimports

del (autocorrcheck, padvec, gaussresidualssk, readfromnifti, phase,
     printthresholds, checkimports)

try:
    from .tidepoolTemplate_qt5 import *  # noqa
except:
    from .tidepoolTemplate_qt4 import *  # noqa
from .OrthoImageItem import *  # noqa
