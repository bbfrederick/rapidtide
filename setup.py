# Stub file to call setuptools.setup with versioneer hooks
from setuptools import setup

from versioneer import get_cmdclass, get_version

setup(
    version=get_version(),
    cmdclass=get_cmdclass(),
)
