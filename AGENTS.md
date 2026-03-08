# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

Rapidtide is a Python package for modeling, characterizing, visualizing, and removing
time-varying physiological blood signals from fMRI and fNIRS datasets.

Primary workhorses:
- `rapidtide`: Characterizes bulk blood flow through time delay analysis on functional
  imaging data, finding time-lagged correlations between voxelwise time series in the
  low-frequency oscillation (LFO) band.
- `happy`: Extracts and analyzes cardiac signals from fMRI data using hypersampling
  techniques, even when TR is too long to properly sample cardiac waveforms.

## Architecture

### Code Organization

```
rapidtide/
|- scripts/          # Command-line entry points
|- workflows/        # Main processing pipelines (rapidtide, happy, and utilities)
|- tests/            # Pytest test suite
|- candidatetests/   # Work-in-progress tests (not run in CI)
|- data/             # Reference data, models, and examples
`- [modules]         # Core processing modules
```

### Key Modules

- `io.py`: NIFTI/text file I/O
- `filter.py`: Signal filtering and preprocessing
- `correlate.py`: Cross-correlation and time-lag analysis
- `fit.py`: Peak fitting and parameter estimation
- `resample.py`: Time series resampling
- `stats.py`: Statistical analysis
- `multiproc.py`: Parallel processing infrastructure
- `happy_supportfuncs.py`: Cardiac processing for happy workflow
- `dlfilter.py` / `dlfiltertorch.py`: Deep learning filters
- `RapidtideDataset.py`: Dataset class for tidepool GUI
- `OrthoImageItem.py`: Orthogonal image display for tidepool GUI

### Script/Workflow Pattern

Most command-line tools follow:
1. `rapidtide/scripts/<name>.py` entry point
2. `rapidtide/workflows/<name>.py` processing logic
3. `rapidtide/workflows/<name>_parser.py` argument parser (for complex tools)

Entry points are registered in `pyproject.toml` under `[project.scripts]`.

## Development Commands

### Setup

```bash
pip install -e .[test,doc]
# or
pip install -e .[all]
```

### Testing

```bash
pytest rapidtide/tests/
pytest rapidtide/tests/test_filter.py
pytest --cov=rapidtide rapidtide/tests/
pytest rapidtide/tests/test_filter.py::test_function_name -v
```

### Formatting

```bash
black rapidtide/
black --check rapidtide/filter.py
```

Black settings are in `pyproject.toml`:
- Line length: 99
- Target: Python 3.10+

### Build/Distribution

```bash
python -m build
pip install .
./builddocker.sh
./testdocker.sh
```

### Main Tools

```bash
rapidtide <input_4d_nifti> <output_root> [options]
happy <input_4d_nifti> <output_root> [options]
tidepool
showtc <textfile>
showxcorrx <file1> <file2>
```

## Important Constraints

- Never change files in `rapidtide/notforprimetime` or `rapidtide/candidatetests`.
- Keep code clean and modular.
- Prefer shorter functions/methods when practical.

### Python Version

- Minimum: Python 3.10
- Maximum: Python 3.14

## Domain and Dependency Notes

- Inputs are primarily 4D NIFTI files.
- Outputs are primarily 3D NIFTI maps and text files.
- Some tools (`rapidtide2std`, `happy2std`) require a working FSL installation for
  registration to MNI152 space.

## Testing and Quality Expectations

- Tests in `rapidtide/tests/` are CI-relevant.
- `rapidtide/candidatetests/` contains experimental tests; do not modify these files.
- Prefer synthetic/reference-based validations where existing tests use them.

## Versioning

- Uses `versioneer` for version management from git tags.
- Version is auto-generated in `rapidtide/_version.py`.
- Tag prefix: `v` (example: `v2.9.0`).

## Documentation and Style

- Use Black formatting (99-char line length).
- Follow NumPy docstring style where applicable.
- Keep changes focused on the issue/feature requested.
- See contribution guidance:
  `http://rapidtide.readthedocs.io/en/latest/contributing.html`.

## Verification Protocol

Before returning control:
1. Re-read the task requirements.
2. Verify each requirement with concrete evidence (tests, checks, or outputs).
3. Address implicit quality bars (edge cases, error handling, formatting).
4. If something fails, fix and re-run verification.
5. After 3 unsuccessful fix/verify cycles on the same blocker, stop and report the
   blocker and diagnosis clearly.
