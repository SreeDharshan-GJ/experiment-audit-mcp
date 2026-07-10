# Test Fixtures

Per design-spec-v1.md §7, this project tests against **recorded real API
responses**, not hand-written mocks, wherever a fixture originates from a
real backend (W&B, MLflow). Hand-written data is reserved for the
`FakeBackend` (Milestone 2), which is a deliberate in-memory test double,
not a mock of a real API's response shape.

## Conventions

- **Directory layout:** `tests/fixtures/wandb/<scenario_name>/` and
  `tests/fixtures/mlflow/<scenario_name>/` (MLflow fixtures arrive in v2).
- **Format:** raw JSON as returned by the backend's client library,
  captured via a one-off recording script (added in Milestone 3), not
  transcribed by hand. This keeps fixtures honest to real API shape,
  including quirks a hand-written mock would accidentally omit.
- **Scenario naming:** descriptive, matching the adversarial cases named
  in design-spec-v1.md §7, e.g.:
  - `happy_path_run/`
  - `sweep_too_small/` (3 runs)
  - `ablation_seed_only_diff/`
  - `ablation_two_nonallowlisted_diffs/`
  - `crashed_run_with_nan/`
  - `partial_data_run/`
  - `sweep_correlated_hyperparams/`
- **Versioning:** each fixture directory includes a `recorded_at` date and
  the backend API/library version it was captured against, so drift
  between fixture shape and live API shape is detectable in CI
  (per design-spec-v1.md §7 point 4).
- **No secrets:** fixtures are sanitized before commit — API keys,
  usernames, and any personally identifying project names are replaced
  with placeholders. This is checked manually before each fixture is
  added; no automated scrubbing tool exists yet (candidate for a future
  milestone if fixture volume grows).

This directory is empty as of Milestone 0. Fixtures are added starting
Milestone 3 (W&B backend) and expanded through Milestone 9 (adversarial
consolidation).

## Milestone 3 status

`scripts/record_wandb_fixtures.py` (the recording script this README
promises) now exists, but has not been run — the build environment used
for Milestone 3 has no `WANDB_API_KEY` and no network path to the W&B
API. `tests/test_wandb_backend.py` currently tests `WandbBackend`
against an in-memory fake client built from W&B's documented public API
attribute shapes, not against files in this directory. This is a
deliberate, logged deviation from this README's own convention, not a
silent one — run the recording script locally against a real project
(MAMFAC/CARM++ recommended) and re-point the test suite at the resulting
fixtures before treating Milestone 3 as fully satisfying spec §7.

## Milestone 9 status

`adversarial_cases.py` (this directory) is new in Milestone 9: it is the
"Consolidated adversarial fixture set... exercised end-to-end through the
MCP layer... for every case in spec §7" deliverable named in the roadmap.
It is not a recorded-fixture directory in the sense the rest of this
README describes -- it is a Python module of `FakeBackend`-seeding
builders, one per spec §7 point-2 case, imported by
`tests/test_adversarial_mcp_layer.py`. This is a deliberate choice, not
an oversight: every one of these six cases is about a specific *shape*
of data (a 3-run sweep, a NaN mid-curve, two correlated hyperparameters)
that is easy and honest to construct by hand, per the same rationale
`FakeBackend` itself was built on in Milestone 2 -- real recorded API
responses add value for API-shape fidelity (Milestone 3's concern), not
for these already-well-specified adversarial shapes. The Milestone 3
status note above still applies unchanged: no fixtures in
`tests/fixtures/wandb/` have been recorded from a real project as of
this milestone either.
