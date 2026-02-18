# Codebase Reorg Plan (Recovered from Chat Session)

This is a reconstructed copy of the earlier response to:
"Is there a better way to organize this codebase?"

It is not a verbatim transcript. It captures the plan that was actually executed during this session.

## Goal

Move from a flat, legacy module surface toward a layered structure with clearer ownership:
- `rapidtide.core.signal.*`
- `rapidtide.core.delay.*`
- `rapidtide.core.models.*`
- `rapidtide.core.masks.*`
- `rapidtide.core.io.*`

while keeping CLI/workflow behavior stable and using compatibility shims during transition.

## Guiding Principles

1. Preserve public API compatibility during migration.
2. Move one domain at a time with tests green at each step.
3. Normalize imports at callsites to core modules before deleting shims.
4. Keep workflow code thin; keep compute logic in `core`.
5. Treat tests/tooling as a parallel track to reduce CI noise while refactoring.

## Phased Plan

### Phase 0: Baseline and Packaging Hygiene

- Ensure package discovery includes new subpackages (`rapidtide*`, `cloud*`).
- Confirm branch workflow (`reorg`) and incremental PR-sized changes.

### Phase 1: Shim-First Extraction

- Create new `core` implementation modules.
- Convert legacy modules into thin wrappers/shims that re-export core symbols.
- Keep existing import paths operational.

### Phase 2: Domain Moves (Executed)

1. `voxelData` extraction into `rapidtide.core.models.voxel_data`.
2. `happy_supportfuncs` extraction into `rapidtide.core.signal.*` split modules.
3. Correlation/math/stats/delay move:
   - `rapidtide.core.signal.correlate`
   - `rapidtide.core.signal.miscmath`
   - `rapidtide.core.signal.stats`
   - `rapidtide.core.delay.refinedelay`

### Phase 3: Full Cutover (Executed)

- Normalize workflow imports to core modules directly.
- Normalize tests to core imports.
- Keep compatibility shims for transition; expose required legacy globals/helpers where needed.

### Phase 4: Mask/Region Signal Cutover (Executed)

- Replace `rapidtide.maskutil` callsites with direct imports:
  - `rapidtide.core.masks.mask_ops`
  - `rapidtide.core.masks.region_signal`
  - `rapidtide.core.io.mask_io`
- Update workflow and test patch targets accordingly.

### Phase 5: Test/Tooling Cleanup (Parallel Track, Executed)

- Add checked-in `pytest.ini`.
- Remove CI step that dynamically overwrote `pytest.ini`.
- Add stable test env in CI (`MPLBACKEND=Agg`, `MPLCONFIGDIR`, `PYTHONPATH`).
- Refactor CircleCI duplicated test commands into reusable command blocks.
- Test/Tooling Cleanup (Parallel Track)
- Central matplotlib cache/backend config
- Central repo-root import handling

#### PHase 5b: Standardize script entry:
- One helper (tests/_bootstrap.py) used by all direct-run files
- Keep:
  - unit for pure functions
  - integration for workflow-level
  - regression for historical bug reproductions

## Practical Rules Used During Migration

- Avoid destructive git operations.
- Keep each change compile-checkable.
- Update monkeypatch paths whenever imports are moved.
- For test files, keep direct executable entrypoint pattern (`if __name__ == "__main__": ...`).

## What Remains (if/when continuing)

1. Decide when to remove compatibility shims after downstream stabilization.
2. Final pass to remove any residual legacy import references in non-critical scripts/examples.
3. Optional docs update describing the new architectural layers and import conventions.
4. Optional CI runtime optimizations once refactor churn settles.

## Suggested Ongoing Convention

- New logic goes into `rapidtide.core.*`.
- Legacy top-level modules should be wrappers only.
- Workflows import from `core` directly.
- Tests target `core` modules unless explicitly testing compatibility wrappers.
