# -*- coding: utf-8 -*-
import os
import tempfile
from pathlib import Path

from . import _version


def _configure_matplotlib_cache():
    if "MPLCONFIGDIR" in os.environ:
        return

    default_mpl_dir = Path.home() / ".matplotlib"
    try:
        if default_mpl_dir.exists():
            if os.access(default_mpl_dir, os.W_OK | os.X_OK):
                return
        elif os.access(default_mpl_dir.parent, os.W_OK | os.X_OK):
            return
    except OSError:
        pass

    fallback_dir = Path(tempfile.gettempdir()) / "rapidtide-mplconfig"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback_dir)


_configure_matplotlib_cache()

__version__ = _version.get_versions()["version"]
