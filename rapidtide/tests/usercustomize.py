#!/usr/bin/env python

print("invoking coverage.process_startup()")
try:
    import coverage
except ImportError:
    pass
else:
    coverage.process_startup()
