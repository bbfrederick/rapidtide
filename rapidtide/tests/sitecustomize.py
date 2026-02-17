#!/usr/bin/env python

import os
from pathlib import Path


def _configure_matplotlib_cache():
    # Keep matplotlib cache/config in a writable, persistent location.
    tests_dir = Path(__file__).resolve().parent
    tmp_dir = tests_dir / "tmp"
    mpl_config_dir = tmp_dir / "mplconfig"
    xdg_cache_home = tmp_dir / "xdg-cache"
    xdg_config_home = tmp_dir / "xdg-config"

    for dirname in (tmp_dir, mpl_config_dir, xdg_cache_home, xdg_config_home):
        dirname.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))
    os.environ.setdefault("XDG_CONFIG_HOME", str(xdg_config_home))


def _configure_coverage():
    try:
        import coverage
    except ImportError:
        return
    coverage.process_startup()


_configure_matplotlib_cache()
_configure_coverage()
