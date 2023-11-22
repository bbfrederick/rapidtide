#!/bin/env python
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
from codecs import open
from os import path

# Always prefer setuptools over distutils
from setuptools import find_namespace_packages, find_packages, setup

import versioneer

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()


# Write version number out to VERSION file
version = versioneer.get_version()
try:
    with open(path.join(here, "VERSION"), "w", encoding="utf-8") as f:
        f.write(version)
except PermissionError:
    print("can't write to VERSION file - moving on")


modules_list = [
    "rapidtide/OrthoImageItem",
    "rapidtide/calccoherence",
    "rapidtide/calcnullsimfunc",
    "rapidtide/calcsimfunc",
    "rapidtide/correlate",
    "rapidtide/dlfilter",
    "rapidtide/externaltools",
    "rapidtide/filter",
    "rapidtide/fit",
    "rapidtide/glmpass",
    "rapidtide/happy_supportfuncs",
    "rapidtide/helper_classes",
    "rapidtide/io",
    "rapidtide/maskutil",
    "rapidtide/miscmath",
    "rapidtide/multiproc",
    "rapidtide/peakeval",
    "rapidtide/refine",
    "rapidtide/resample",
    "rapidtide/simfuncfit",
    "rapidtide/stats",
    "rapidtide/util",
    "rapidtide/wiener",
    "rapidtide/_version",
    "rapidtide/workflows/aligntcs",
    "rapidtide/workflows/fdica",
    "rapidtide/workflows/glmfilt",
    "rapidtide/workflows/gmscalc",
    "rapidtide/workflows/happy",
    "rapidtide/workflows/happy_parser",
    "rapidtide/workflows/histnifti",
    "rapidtide/workflows/niftidecomp",
    "rapidtide/workflows/niftistats",
    "rapidtide/workflows/parser_funcs",
    "rapidtide/workflows/polyfitim",
    "rapidtide/workflows/rapidtide",
    "rapidtide/workflows/rapidtide_parser",
    "rapidtide/workflows/roisummarize",
    "rapidtide/workflows/showtc",
    "rapidtide/workflows/tidepool",
    "rapidtide/workflows/utils",
    "rapidtide/tidepoolTemplate",
    "rapidtide/tidepoolTemplate_alt",
]


script_list = [
    "rapidtide/scripts/adjustoffset",
    "rapidtide/scripts/aligntcs",
    "rapidtide/scripts/applydlfilter",
    "rapidtide/scripts/atlasaverage",
    "rapidtide/scripts/atlastool",
    "rapidtide/scripts/calcicc",
    "rapidtide/scripts/calcttest",
    "rapidtide/scripts/calctexticc",
    "rapidtide/scripts/ccorrica",
    "rapidtide/scripts/diffrois",
    "rapidtide/scripts/endtidalproc",
    "rapidtide/scripts/fdica",
    "rapidtide/scripts/filtnifti",
    "rapidtide/scripts/filttc",
    "rapidtide/scripts/fingerprint",
    "rapidtide/scripts/glmfilt",
    "rapidtide/scripts/gmscalc",
    "rapidtide/scripts/happy",
    "rapidtide/scripts/happy2std",
    "rapidtide/scripts/happywarp",
    "rapidtide/scripts/histnifti",
    "rapidtide/scripts/histtc",
    "rapidtide/scripts/pairwisemergenifti",
    "rapidtide/scripts/pairproc",
    "rapidtide/scripts/physiofreq",
    "rapidtide/scripts/pixelcomp",
    "rapidtide/scripts/plethquality",
    "rapidtide/scripts/polyfitim",
    "rapidtide/scripts/proj2flow",
    "rapidtide/scripts/rapidtide",
    "rapidtide/scripts/rapidtide2std",
    "rapidtide/scripts/resamp1tc",
    "rapidtide/scripts/resamplenifti",
    "rapidtide/scripts/roisummarize",
    "rapidtide/scripts/showarbcorr",
    "rapidtide/scripts/showhist",
    "rapidtide/scripts/showtc",
    "rapidtide/scripts/showstxcorr",
    "rapidtide/scripts/showxcorr_legacy",
    "rapidtide/scripts/showxcorrx",
    "rapidtide/scripts/showxy",
    "rapidtide/scripts/simdata",
    "rapidtide/scripts/spatialdecomp",
    "rapidtide/scripts/spatialfit",
    "rapidtide/scripts/spatialmi",
    "rapidtide/scripts/spectrogram",
    "rapidtide/scripts/synthASL",
    "rapidtide/scripts/tcfrom2col",
    "rapidtide/scripts/tcfrom3col",
    "rapidtide/scripts/temporaldecomp",
    "rapidtide/scripts/threeD",
    "rapidtide/scripts/tidepool",
    "rapidtide/scripts/variabilityizer",
]


setup(
    name="rapidtide",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Tools for performing correlation analysis on fMRI data.",
    long_description=long_description,
    # The project's main homepage.
    url="https://github.com/bbfrederick/rapidtide",
    # Author details
    author="Blaise Frederick",
    author_email="bbfrederick@mclean.harvard.edu",
    # Choose your license
    license="Apache Software License",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: Apache Software License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # What does your project relate to?
    keywords=["fMRI", "correlation", "RIPTiDe", "noise"],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_namespace_packages(
        exclude=[
            "contrib",
            "docs",
            "rapidtide/data/examples/dst",
            "rapidtide/data/examples/extrasrc",
        ]
    ),
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    py_modules=modules_list,
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-image",
        "scikit-learn",
        "nibabel",
        "nilearn",
        "matplotlib",
        "pyfftw",
        "pyqtgraph",
        "pyqt",
        "pyqt5-sip",
        "requests",
        "statsmodels",
        "pywavelets",
        "tqdm",
    ],
    # "keras",
    # "tensorflow>=2.4.0",
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "doc": [
            "sphinx>=1.8.0",
            "sphinx_rtd_theme",
            "sphinx-argparse",
            "sphinx-gallery",
            "m2r",
            "numpydoc",
            "recommonmark",
            "mistune==0.8.4",
        ],
        "tests": [
            "codecov",
            "coverage",
            "coveralls",
            "flake8-black",
            "pytest",
            "pytest-cov",
        ],
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    include_package_data=True,
    zip_safe=False,
    package_data={
        "examples": [
            "data/examples/src/test*",
            "data/examples/src/installtestdata",
        ],
        "reference": [
            "data/reference/JHU-ArterialTerritoriesNoVent-LVL*.xml",
            "data/reference/JHU-ArterialTerritoriesNoVent-LVL*regions.txt",
            "data/reference/JHU-ArterialTerritoriesNoVent-LVL*2mm.nii.gz",
            "data/reference/ASPECTS*",
            "data/reference/ATT*",
            "data/reference/HCP1200*",
            "data/reference/MNI152*",
            "data/reference/slicetimes/*",
        ],
        "models": [
            "data/models/*",
        ],
        "testdata": [
            "tests/testdata/*.txt",
        ],
        "tmpdata": [
            "tests/tmp/.placeholder.txt",
        ],
    },
    # additional_packagedata={
    #    "testtargets": [
    #        "tests/testtargets/*.txt",
    #        "tests/testtargets/*.nii.gz",
    #    ],
    # },
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    scripts=script_list,
)
