#!/usr/bin/env python

try:
    from rapidtide.tests._mplsetup import configure_matplotlib_env
except Exception:
    from _mplsetup import configure_matplotlib_env


def _configure_coverage():
    try:
        import coverage
    except ImportError:
        return
    coverage.process_startup()


configure_matplotlib_env()
_configure_coverage()
