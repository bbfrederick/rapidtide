# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rapidtide is a Python package for modeling, characterizing, visualizing, and removing time-varying physiological blood signals from fMRI and fNIRS datasets. The package has two primary workhorses:

- **rapidtide**: Characterizes bulk blood flow through time delay analysis on functional imaging data, finding time-lagged correlations between voxelwise time series in the low-frequency oscillation (LFO) band
- **happy**: Extracts and analyzes cardiac signals from fMRI data using hypersampling techniques, even when TR is too long to properly sample cardiac waveforms

## Architecture

### Code Organization

```
rapidtide/
├── scripts/          # Command-line entry points (~65 utilities)
├── workflows/        # Main processing pipelines (rapidtide, happy, and various utilities)
├── tests/            # Pytest test suite (~44 test files)
├── candidatetests/   # Work-in-progress tests (not run in CI)
├── data/             # Reference data, models, and examples
└── [modules]         # Core processing modules (see below)
```

### Key Modules

- **io.py**: NIFTI/text file I/O operations
- **filter.py**: Signal filtering and preprocessing
- **correlate.py**: Cross-correlation and time-lag analysis
- **fit.py**: Peak fitting and parameter estimation
- **resample.py**: Time series resampling utilities
- **stats.py**: Statistical analysis functions
- **multiproc.py**: Parallel processing infrastructure
- **happy_supportfuncs.py**: Cardiac signal processing for happy workflow
- **dlfilter.py / dlfiltertorch.py**: Deep learning filters for signal cleaning
- **RapidtideDataset.py**: Dataset class for tidepool GUI
- **OrthoImageItem.py**: Orthogonal image display for tidepool GUI

### Script/Workflow Architecture

All command-line tools follow a consistent pattern:
1. `rapidtide/scripts/<name>.py` - Minimal entry point that imports from workflows
2. `rapidtide/workflows/<name>.py` - Main processing logic
3. `rapidtide/workflows/<name>_parser.py` - Argument parsing (for complex tools)

Entry points are registered in `pyproject.toml` under `[project.scripts]`.

### Main Workflows

**rapidtide workflow** (`rapidtide/workflows/rapidtide.py`):
- Performs voxel-wise time delay analysis on fMRI data
- Generates multiple 3D NIFTI maps (lag time, correlation values, masks, etc.)
- Outputs text files with significance thresholds and processing parameters

**happy workflow** (`rapidtide/workflows/happy.py`):
- Extracts cardiac waveforms from fMRI using slice-selective averaging
- Cleans estimates using deep learning filters
- Constructs cardiac pulsation maps over a single cardiac cycle

## Development Commands

### Setup and Installation

```bash
# Install package in development mode with all dependencies
pip install -e .[tests,doc]

# Or for all optional dependencies
pip install -e .[all]
```

### Testing

```bash
# Run full test suite
pytest rapidtide/tests/

# Run specific test file
pytest rapidtide/tests/test_filter.py

# Run with coverage
pytest --cov=rapidtide rapidtide/tests/

# Run specific test function
pytest rapidtide/tests/test_filter.py::test_function_name -v
```

Note: Tests are run in CI via CircleCI for Python 3.9, 3.10, 3.11, and 3.12.

### Code Formatting

```bash
# Format code with black (line length: 99)
black rapidtide/

# Check specific file
black --check rapidtide/filter.py
```

Black configuration in `pyproject.toml`:
- Line length: 99
- Target: Python 3.9+
- Excludes: versioneer files, candidatetests, disabledtests, data/examples

### Building and Distribution

```bash
# Build package
python -m build

# Install locally
pip install .

# Build Docker container
./builddocker.sh

# Test Docker container
./testdocker.sh
```

### Running Main Tools

```bash
# Run rapidtide analysis
rapidtide <input_4d_nifti> <output_root> [options]

# Run happy analysis
happy <input_4d_nifti> <output_root> [options]

# View results in GUI
tidepool  # Then select a lag time map file

# Quick timecourse visualization
showtc <textfile>

# Cross-correlation between two timecourses
showxcorrx <file1> <file2>
```

## Important Constraints

### Python Version
- **Minimum**: Python 3.9
- **Maximum**: Python 3.12 (tensorflow limitation)
- Uses modern Python features (f-strings, type hints)

### Data Formats
- Input: 4D NIFTI files for fMRI data
- Output: 3D NIFTI maps, text files with timecourses/parameters
- Timecourses: Whitespace-separated text files

### FSL Dependency
Some tools (rapidtide2std, happy2std) require a working FSL installation for registration to MNI152 space.

## Key Design Patterns

### Versioning
Uses versioneer for automatic version management from git tags:
- Version set in `rapidtide/_version.py` (auto-generated)
- Tag prefix: `v` (e.g., v2.9.0)

### Testing Philosophy
- Main tests in `rapidtide/tests/` are run in CI
- Experimental/incomplete tests in `rapidtide/candidatetests/`
- Many tests use synthetic data and compare against reference outputs
- Full workflow tests: `test_fullrunrapidtide_v*.py`, `test_fullrunhappy_v*.py`

### Multiprocessing
Many operations support parallel processing:
- Uses `rapidtide/multiproc.py` and `rapidtide/genericmultiproc.py`
- Configurable number of worker processes
- Shared memory for efficiency

## Special Notes

- The codebase uses extensive command-line argument parsing with validation in `workflows/parser_funcs.py`
- Deep learning models for signal filtering are in `rapidtide/data/models/`
- The package includes a GUI tool (tidepool) built with PyQt6 for visualizing results
- Reference data and example datasets are in `rapidtide/data/`
- Documentation is built with Sphinx and hosted on ReadTheDocs

## Style Conventions

See the contributing guide at http://rapidtide.readthedocs.io/en/latest/contributing.html for full style guidelines.

Key points:
- Use Black formatter with 99-character line length
- Follow NumPy docstring format
- Keep changes focused on specific issues/features
- Work on feature branches, not main