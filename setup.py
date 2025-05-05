# Stub file to call setuptools.setup with versioneer hooks
from setuptools import setup
from versioneer import get_version, get_cmdclass

setup(
    version=get_version(),
    cmdclass=get_cmdclass(),
)
