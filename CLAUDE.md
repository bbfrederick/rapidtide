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
├── scripts/          # Command-line entry points (~61 utilities)
├── workflows/        # Main processing pipelines (rapidtide, happy, and various utilities)
├── tests/            # Pytest test suite (~116 test files)
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
pip install -e .[test,doc]

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

Note: Tests are run in CI via CircleCI for Python 3.10, 3.11, 3.12, 3.13, and 3.14.

### Code Formatting

```bash
# Format code with black (line length: 99)
black rapidtide/

# Sort imports (configured to match black)
isort rapidtide/

# Check specific file
black --check rapidtide/filter.py
```

Black configuration in `pyproject.toml`:
- Line length: 99
- Target: Python 3.10+
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

- Never change files in rapidtide/candidatetests
- Always work on feature branches, not main

### Code Style (NON-NEGOTIABLE)
- Write code that is clean and modular
- Prefer shorter functions/methods over longer ones
- **Every routine must have a numpydoc-style docstring** (Parameters, Returns, and any other
  relevant sections). When you modify a routine, re-read its docstring and verify it still matches
  the code — parameter names, types, defaults, return values, and raised exceptions. Fix any
  drift as part of the same change; a stale docstring is a bug.
- **Every function must have type annotations** for all arguments and for the return value
  (use `-> None` when nothing is returned). Applies to new code and to any existing function
  you touch.

### Python Version
- **Minimum**: Python 3.10
- **Maximum**: Python 3.14
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
- Follow NumPy docstring format — see the mandatory docstring and type-annotation rules under
  [Code Style (NON-NEGOTIABLE)](#code-style-non-negotiable)
- Keep changes focused on specific issues/features

## Known Gotchas

- **`postprocessfilteropts` attribute mismatch**: The parser uses `dest="ncfiltpadtype"` but
  `postprocessfilteropts` reads `args.prefilterpadtype`. The `try/except` fallback hides this — it
  silently uses the default rather than the user-specified value.
- **`candidatetests/`**: These tests are not run in CI and may be incomplete or broken. Do not
  rely on them passing.

## VERIFICATION PROTOCOL (execute before returning control)
- Re-read the full original task specification.
- For each stated requirement: test it, confirm it works, and state the evidence.  Do not self-report "done" without executing the actual check.
- For each implicit quality bar (error handling, edge cases, formatting): apply the same standard.
- If something fails: fix and re-verify from scratch. Do not patch and assume.
- After 3 full fix-verify cycles with a persistent failure, stop and report the specific blocker with your diagnosis. Do not return broken work and do not loop silently.
- Only return control when every requirement has verified evidence of passing, or you've explicitly flagged what you couldn't solve and why.


## Model Delegation for Coding Tasks

For coding tasks, use your judgement to delegate work to a subagent running an appropriately lower-power model when the task doesn't need the full capability of the current model. For example: if you are Fable, delegate suitable tasks to Opus or Sonnet subagents; if you are Opus, delegate suitable tasks to Sonnet.
   
Model tiers for ANY delegated work - Agent-tool calls and Workflow-script `agent()` calls alike. Set the `model` parameter explicitly on every call; never omit it (omission silently inherits the session model):
- `haiku` - mechanical bulk work: renames, boilerplate, format conversion, log triage
- `sonnet` - default for well-specified implementation with clear acceptance criteria
- `opus` - genuinely tricky work: concurrency, subtle algorithms, adversarial verify/judge panels, gnarly debugging
- `fable` - rare; only when independence from your own context is the point (e.g. adversarial review of your own plan or a large diff). If you want to call a Fable sub-agent because the complexity of the task warrants it, ALWAYS check with me first - never spawn one unprompted.
   
When unsure between tiers, pick the cheaper and escalate on failure.
   
   
## Dynamic workflows (Workflow tool)

Applies to ALL sessions, any model. Dynamic workflows do not need to be avoided - reach for the Workflow tool when a task has 3+ independent parallelizable subtasks or would benefit from a pipeline/judge panel. Standing rule on opt-in: if ultracode is NOT on for the session (no "ultracode" keyword, toggle, or an orchestration request in my own words), check with me first - propose the workflow in one or two sentences with the rough shape and cost, and wait for my reply; my "yes" is the opt-in. If ultracode IS on, invoke directly.

**Agent models inside workflow scripts:** every `agent()` call MUST set the `model` parameter explicitly, chosen per "Delegating to sub-agents" above - with one tightening: NEVER use `fable` agents in a dynamic workflow, not even with approval. Only `haiku`, `sonnet`, or `opus`. If a Fable review is warranted, it happens AFTER the workflow completes, as a standalone Agent-tool call (ask first, per above) - never as a workflow stage.
