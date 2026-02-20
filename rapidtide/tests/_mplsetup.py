#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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
import os
import sys
import tempfile
import time
from pathlib import Path


def configure_matplotlib_env() -> None:
    # Keep matplotlib cache off the repository filesystem and isolate each process
    # to avoid lock contention during font cache initialization.
    base = Path(
        os.environ.get(
            "RAPIDTIDE_MPL_BASEDIR",
            str(Path(tempfile.gettempdir()) / "rapidtide-mpl"),
        )
    )
    pyver = f"py{sys.version_info.major}{sys.version_info.minor}"
    worker = os.environ.get("PYTEST_XDIST_WORKER", "shared")
    mpl_config_dir = base / pyver / worker / "mplconfig"
    xdg_cache_home = base / pyver / worker / "xdg-cache"
    xdg_config_home = base / pyver / worker / "xdg-config"

    for dirname in (mpl_config_dir, xdg_cache_home, xdg_config_home):
        dirname.mkdir(parents=True, exist_ok=True)

    # A crashed test process can leave stale matplotlib lock files behind and
    # block subsequent runs indefinitely while waiting on font-cache writes.
    for lockfile in mpl_config_dir.glob("*.matplotlib-lock"):
        try:
            if (time.time() - lockfile.stat().st_mtime) > 600:
                lockfile.unlink(missing_ok=True)
        except Exception:
            pass

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))
    os.environ.setdefault("XDG_CONFIG_HOME", str(xdg_config_home))


def warmup_matplotlib() -> None:
    # Trigger font manager initialization once at startup.
    try:
        import matplotlib.font_manager as font_manager

        font_manager._load_fontmanager()
    except Exception:
        pass
