# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-07-10

Initial public release. W&B backend only, per the v1 scope in
`docs/design-spec-v1.md` §10.

### Added

- Eight MCP tools: `test_connection`, `list_runs`, `get_run_summary`,
  `get_metric_history`, `compare_runs`, `audit_training_curve`,
  `audit_ablation`, `audit_sweep`.
- `ExperimentBackend` abstraction with a capability-declaration model
  (`BackendCapability`), a real `WandbBackend`, and an in-memory
  `FakeBackend` test double capable of injecting every adversarial state
  named in the design spec.
- Structured `ToolError` taxonomy (spec §5) so no bare exception crosses
  the MCP boundary.
- Full adversarial fixture suite exercised end-to-end through the MCP
  layer (Milestone 9), plus a fixed 15-prompt tool-selection evaluation
  harness (`scripts/tool_selection_eval.py`).
- Complete documentation set: `docs/design-spec-v1.md` (frozen v1
  design), `docs/implementation-roadmap-v1.md` (milestone-by-milestone
  build history with two logged design revisions), `docs/audit-methods.md`
  (full methodology for every `audit_*` tool), `docs/tool-selection-eval.md`.
- 233 tests, 100% passing; `ruff check .` clean.

### Known gaps at release

- Fixture recording against a real, live W&B project has not been
  performed (build sandbox has no `WANDB_API_KEY` / network access).
  `WandbBackend` is tested against an in-memory fake client built from
  W&B's documented API shapes instead. See `tests/fixtures/README.md`.
- The tool-selection eval has not been run against a live MCP client
  (build sandbox has no `ANTHROPIC_API_KEY` / network access to
  `api.anthropic.com`). See `docs/tool-selection-eval.md`.
- MCP Registry, Glama, and cursor.directory submission has not been
  performed from this environment (no outbound access to those services
  here) — the package is publish-ready; submission is left to whoever
  runs the release from a machine with that access.

### Design revisions along the way (see `docs/design-spec-v1.md` Revision Log)

- **Revision 1** — `RunRef` gained a required `entity` field after review
  found that `project` alone doesn't uniquely scope a run against the
  real W&B API (two entities can each own a same-named project).
- **Revision 2** — `ExperimentBackend.list_runs` gained an optional
  `page_size` parameter after a mismatch was found between the frozen
  ABC (no per-call page size) and the `list_runs` MCP tool's spec'd
  signature (`page_size=25`).

[1.0.0]: https://github.com/<your-username>/experiment-audit-mcp/releases/tag/v1.0.0
