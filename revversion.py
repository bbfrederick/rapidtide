#!/usr/bin/env python

from os import path

import versioneer

# Write version number out to VERSION file
version = versioneer.get_version()

here = path.abspath(path.dirname(__file__))

try:
    with open(path.join(here, "VERSION"), "w", encoding="utf-8") as f:
        f.write(version)
except PermissionError:
    print("can't write to VERSION file - moving on")
