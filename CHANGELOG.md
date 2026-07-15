# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] — 2026-07-15

The Scientific Reasoning Engine becomes the project's primary,
first-class capability, exposed through a public Python API and a new
CLI, alongside the unchanged v1.0.0 MCP/W&B audit tools.

### Added

- `experiment_audit.reasoning` public API: `ScientificReasoningPipeline`
  (the six-rule claim/evidence/contradiction/confidence/judgment/
  recommendation pipeline), `ScientificReport`, `Claim`, `ClaimSet`,
  `Evidence`, `EvidenceItem`, `Contradiction`, and the generic
  `ScientificReasoningEngine`.
- `experiment-audit` CLI (`experiment-audit reasoning run`,
  `experiment-audit reasoning schema`) — run the reasoning pipeline
  against a JSON file of claims/evidence and get a Markdown, JSON, or
  plain-text scientific report.
- 4 new regression tests covering bugs found during this release (see
  below), plus recovery of the reasoning engine's full test suite (32
  tests total for the reasoning package).

### Fixed

- **Broken CLI entry point.** `pyproject.toml`'s console script pointed
  at `experiment_audit_mcp.server:main`, a module that no longer existed
  after an earlier package rename to `experiment_audit`. Running the
  installed CLI raised `ModuleNotFoundError` unconditionally. Fixed to
  `experiment_audit.server:main`.
- **Contradiction/rule engine was an empty stub.** `reasoning/rules.py`
  contained a 5-line docstring and no logic; the six concrete rules
  (`scientific_rules/*.py`) didn't exist in this branch's history at
  all. Recovered from a separate development branch
  (`reasoning/core-engine`) where the complete implementation had been
  built and tested but never merged. No rule logic was written or
  modified as part of this recovery.
- **`ScientificReasoningEngine.review()` called `ConfidenceAssessor.assess()`
  and `JudgmentGenerator.generate()` with the wrong number of arguments**
  relative to their real implementations in `confidence.py`/`judgment.py`,
  raising `TypeError` on every call with a non-null assessor/generator.
  Fixed both call sites and their `Protocol` signatures.
- **`ScientificReasoningPipeline.build_initial_context()` never passed
  `observations`/`hypotheses` to `RuleContext`**, both required
  fields with no default, raising `TypeError` on every call regardless
  of arguments. Fixed by defaulting both to empty sets, since none of
  the six rules read them.

### Changed

- README, package description, and keywords rewritten to lead with the
  Scientific Reasoning Engine; the MCP server, W&B backend, and CLI are
  now presented as integrations around it rather than as the product
  itself. No tool schemas, model fields, or backend behavior changed —
  this is a documentation and packaging-metadata change plus the fixes
  and additions above.

### Known gaps introduced by this release

- No adapter yet converts a W&B run directly into `Claim`s/`EvidenceItem`s
  — see README's Known Gaps section.
- The generic `ScientificReasoningEngine` pipeline still defaults its
  rule-engine stage to a no-op; it does not share the six concrete rules
  the main pipeline uses.

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

[1.1.0]: https://github.com/SreeDharshan-GJ/experiment-audit/releases/tag/v1.1.0
[1.0.0]: https://github.com/SreeDharshan-GJ/experiment-audit/releases/tag/v1.0.0
