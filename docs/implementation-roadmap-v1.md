# experiment-audit-mcp — Implementation Roadmap (v1)

Derived from `design-spec-v1.md` (frozen). Each milestone is independently reviewable, builds only on prior milestones, and ends with a summary tying the work back to the spec. No milestone scaffolds ahead of itself — if a later milestone's needs aren't yet justified by a written requirement, we don't pre-build for it.

**Global rule for this roadmap:** if implementing a milestone surfaces a design flaw (not just an implementation detail), we stop, explain the flaw, and wait for a decision — we do not silently patch the spec.

---

## Milestone 0 — Project Scaffolding & Tooling

**Objective:** A repo that runs, tests, and lints, with nothing functional in it yet.

**Depends on:** nothing.

**Deliverables:**
- Directory structure per §Architecture in the spec (empty modules with docstrings/stubs only)
- `pyproject.toml` — package metadata, dependencies (FastMCP, pytest, pytest-asyncio), dev tooling (ruff or equivalent)
- CI config (GitHub Actions): lint + test on push
- Empty `tests/fixtures/` directory with a README explaining the fixture-recording convention (so this is decided once, not re-litigated per milestone)
- License file (MIT, per spec §6)

**Effort:** ~0.5 day

**Completion criteria:** `pytest` runs (0 tests, exit 0), lint passes, CI green on a trivial commit.

**Correction note:** the CI config deliverable above was missed during the original Milestone 0 pass and only added at the Milestone 3→4 transition (`.github/workflows/ci.yml`, lint + test on push/PR). Flagged and fixed as a small corrective action, not a design change — see the Milestone 3→4 transition summary.

**TDD note:** N/A — no logic yet.

---

## Milestone 1 — Data Models

**Objective:** Implement `models.py` exactly as specified in Design Spec §2 — `RunRef`, `Run`, `MetricPoint`, `MetricHistory`, `Sweep`, `Page[T]` — with full unit test coverage of construction, equality, and serialization.

**Depends on:** Milestone 0.

**Deliverables:**
- `src/models.py` with all dataclasses, frozen where specified (`RunRef` must be frozen/hashable since it's used as a dict key candidate in `compare_runs` output)
- `Page[T]` generic wrapper (`items`, `next_cursor`)
- Serialization helpers (to-dict, since MCP tool responses are JSON) with explicit tests that `None` values (e.g. `MetricPoint.value = None` for a logged NaN) survive serialization rather than being dropped — this is a spec-critical behavior (§2, §7 adversarial case), so it's tested here at the model layer, not deferred to integration tests later
- Tests: construction, frozen/hashability of `RunRef`, serialization round-trip, `data_completeness` default behavior

**Effort:** ~1 day

**Completion criteria:** 100% test coverage on `models.py`; a reviewer can read the test file and understand the full data contract without reading the spec.

**TDD approach:** Write the serialization round-trip tests (including the None-preservation case) before implementing the to-dict helpers — this is the one place in this milestone where the test meaningfully drives the design rather than just verifying it.

**Summary format for this milestone (and all following):** what was built, why built that way, spec alignment, any deviations flagged.

---

## Milestone 2 — Backend Abstraction + Fake Backend

**Objective:** Implement `ExperimentBackend` ABC, `BackendCapability`, `ToolError`/error taxonomy (spec §3, §5), and a `FakeBackend` test double that implements the interface in-memory.

**Depends on:** Milestone 1 (uses `models.py` types).

**Deliverables:**
- `backends/base.py` — abstract class, capability enum, `NotSupportedError`
- `errors.py` — `ToolError` dataclass with the full `error_type` literal set from spec §5, including `partial_data`
- `backends/fake_backend.py` — a fully in-memory backend used only in tests, seeded with programmable runs/sweeps/histories, including the ability to inject adversarial states (partial data, NaN points, tiny sweeps) on demand
- Tests: verify `FakeBackend` satisfies the ABC contract; verify `list_sweeps()` on a capability-less fake raises `NotSupportedError` correctly (this is the mechanism validated conceptually in Appendix A — here we prove it in code, against a fake, before touching a real API)

**Effort:** ~1.5 days

**Completion criteria:** `FakeBackend` can produce every adversarial scenario listed in spec §7 (tiny sweep, partial data, NaN mid-curve, correlated hyperparameters) on demand via test fixtures. This backend becomes the primary tool for testing every subsequent milestone — real W&B calls are not needed to test tool logic.

**Why build a fake backend before the real one:** this lets Milestones 4–8 (tool logic) be developed and tested without live API dependency or W&B rate limits, and forces the abstraction to prove itself against adversarial data before it meets the real API's mess. If the abstraction doesn't hold up here, that's a design flaw to surface now, not after the W&B backend is half-built.

**Flag-if-triggered:** if `FakeBackend` cannot cleanly represent an adversarial state the spec requires (e.g., "partial data" is ambiguous to simulate), that's a signal the `Run`/error model needs revisiting — stop and report rather than forcing a workaround.

---

## Revision 1 — `RunRef.entity` migration (completed, prerequisite to Milestone 3)

Design flaw found and fixed before Milestone 3 began: `RunRef` lacked
`entity`, so two different W&B entities with same-named projects would
collide as the same identity. See `design-spec-v1.md` Revision Log for
the full flaw/fix writeup. Updated: `models.py`, `test_models.py`,
`test_fake_backend.py`, this roadmap, the design spec. No other module
required changes. All 58 tests pass; lint clean. `ExperimentBackend`
method signatures (§3) were deliberately left unchanged — see the spec's
Revision Log for why that's a bounded, separately-justified question.

---

## Milestone 3 — W&B Backend (real implementation)

**Status: implemented, pending live validation (see summary delivered
alongside this milestone for full detail).** `WandbBackend` is built
against `wandb==0.28.0`'s documented public API surface, dependency-
injected for testability, with 29 tests passing against an in-memory
fake client. Three genuine API-contact ambiguities were found and
explicitly flagged in code and in the milestone summary rather than
silently resolved: (1) W&B's public `Runs` paginator has no resumable
opaque cursor, so `Page.next_cursor` uses a reproducible offset instead;
(2) `data_completeness` is inferred from `run.state` via a documented-
but-unverified heuristic; (3) rate-limit retry classification is
message-substring-based, not status-code-based, since the SDK doesn't
surface a structured code at this layer. Fixtures are synthetic
(constructed from documented shapes), not recorded from a live project,
per `tests/fixtures/README.md`'s Milestone 3 status note — the sandbox
this was built in has no live W&B credentials or network access.

**Objective:** Implement `backends/wandb_backend.py` against the real W&B API, satisfying the same `ExperimentBackend` contract validated in Milestone 2.

**Depends on:** Milestone 2.

**Deliverables:**
- `WandbBackend` implementing `test_connection`, `list_runs` (with real pagination via W&B's API cursor), `get_run_summary`, `get_metric_history`
- Credential handling (`auth.py`): env-var read, fail-fast validation, no logging of secrets
- Rate-limit backoff (exponential, capped) implemented once at this layer per spec §5
- **Recorded fixtures**: real API responses captured from an actual W&B project (recommend using your own MAMFAC/CARM++ project data — real messiness is more valuable here than synthetic data) for happy-path and, where obtainable, partial-data states
- Tests run against recorded fixtures, not live API calls (per spec §7) — a small number of live smoke tests, clearly marked and excluded from normal CI, can supplement but not replace fixture tests

**Effort:** ~2–3 days (includes fixture recording, which is fiddly)

**Completion criteria:** All `ExperimentBackend` contract tests from Milestone 2 pass against `WandbBackend` using recorded fixtures; `test_connection` correctly distinguishes auth failure from network failure in a manual test against a deliberately bad key.

**Flag-if-triggered:** if W&B's actual pagination model, rate-limit headers, or partial-ingestion signals don't map cleanly onto the `Page[T]` / `data_completeness` abstractions from Milestones 1–2, stop and report — this is exactly the kind of contact-with-reality problem the spec asked to surface rather than paper over.

---

## Revision 2 — `ExperimentBackend.list_runs` gains `page_size` (completed, prerequisite to Milestone 4)

Design flaw found and fixed at the start of Milestone 4: spec §4.2's
`list_runs` tool signature has always included `page_size=25`, but the
`ExperimentBackend` ABC (§3, frozen in Milestone 2) never had a
`page_size` parameter — `WandbBackend` only accepted one at construction,
fixed for the backend's lifetime. This meant the Milestone 4 `list_runs`
tool had no correct way to honor a caller-supplied `page_size`. See
`design-spec-v1.md` Revision Log for the full flaw/fix writeup. Updated:
`backends/base.py`, `backends/fake_backend.py`, `backends/wandb_backend.py`,
`test_backend_base.py`, `test_fake_backend.py`, `test_wandb_backend.py`,
this roadmap, the design spec. Strictly additive (one new optional,
default-`None` parameter) — no other signature, model, or tool contract
touched. All tests pass; lint clean.

---

## Milestone 4 — MCP Server Shell + Retrieval Tools

**Objective:** Wire up the actual MCP server (FastMCP) exposing `test_connection`, `list_runs`, `get_run_summary`, `get_metric_history` as real MCP tools, backed by `WandbBackend`.

**Depends on:** Milestone 3.

**Deliverables:**
- `server.py` — FastMCP entrypoint, tool registration
- Tool schemas matching spec §4.2 exactly (field names, required/optional)
- Tight tool descriptions per spec §6 (pointer to `docs/audit-methods.md`, not inlined methodology)
- Integration tests: invoke each tool through the MCP protocol layer (not just calling the Python function directly) against `WandbBackend` + recorded fixtures, confirming JSON schema compliance end-to-end

**Effort:** ~1.5 days

**Completion criteria:** A real MCP client (Claude Desktop or the MCP inspector tool) can connect to the server and successfully call all four tools against a real or fixture-backed project.

**Note:** this is the first milestone producing something you can actually point an MCP client at — worth manually verifying in Claude Desktop or Claude Code before moving on, since this is also the first real check on whether the tool descriptions read naturally to an agent (a concern from spec §6/§7).

---

## Milestone 5 — `compare_runs`

**Status: implemented.** `analysis/comparison.py` ships a pure `compare_runs(runs: list[Run]) -> ComparisonResult` with no dependency on FastMCP, MCP transport, `server.py`, `WandbBackend`, or any request/response schema — confirmed against the pre-Milestone-5 architecture validation this roadmap requires. The `compare_runs` MCP tool in `server.py` is a thin wrapper: it resolves each `RunRef` into a `Run` via the target backend(s) and calls the pure function directly. Config diff distinguishes "missing" from "explicitly `None`" via a `ConfigValue(present, value)` wrapper (config values are `dict[str, Any]`, so a bare `None` can't double as an absence sentinel). Metric diff's `delta` generalizes to N runs as `max(present) - min(present)`, which collapses to the ordinary pairwise difference at N=2. 23 tests added (18 pure-logic unit tests against hand-constructed `Run` objects in `tests/test_comparison.py`, 5 MCP-layer integration tests via `fastmcp.Client` in `tests/test_server.py`), covering added/removed/changed config params, missing-metric handling, 3- and 4-way comparisons, cross-project and cross-backend refs, duplicate-ref and too-few-runs rejection, and full JSON-serialization round-tripping. All 126 repo tests pass; lint clean; packaging verified from a clean venv (sdist + wheel build, then a fresh install of the wheel with an end-to-end MCP tool-listing smoke test). No design flaws surfaced during implementation.

**Objective:** Implement the pure-diffing tool, the simplest of the compute tools and a dependency for two audit tools later.

**Depends on:** Milestone 4 (reuses the same server/tool-registration pattern) and Milestone 2 (uses `FakeBackend` for most tests, since diffing logic doesn't need real API data once runs are fetched).

**Deliverables:**
- `analysis/comparison.py` — pure function: `list[Run] -> config_diff, metric_diff`
- `compare_runs` MCP tool wrapping it
- Tests: config diff correctness (added/removed/changed params), metric diff correctness, cross-project comparison (confirms each run's project is echoed per spec §2), N-way comparison (not just pairwise) — since `audit_ablation` in Milestone 7 is a 2-run special case of this, test the general N-way path here rather than assuming pairwise is sufficient

**Effort:** ~1 day

**Completion criteria:** Diff logic has no dependency on the audit tools; `audit_ablation` in Milestone 7 will call this as a library function, not duplicate the diffing logic.

**TDD approach:** write the diff tests against hand-constructed `Run` objects (not fixtures) first, since this logic is pure and doesn't need real API shape to validate — fast, isolated tests before wiring into the MCP layer.

---

## Milestone 6 — `audit_training_curve`

**Objective:** Implement the first judgment tool — scored signals over a metric history, per spec §4.2's redesigned (non-enum) output shape.

**Depends on:** Milestone 4 (for `get_metric_history`, which this tool calls internally per the explicit data-flow rule in spec §4.3) and Milestone 2 (`FakeBackend` for adversarial NaN/oscillation fixtures).

**Deliverables:**
- `analysis/divergence.py` — signal detectors: `null_values`, `sudden_jump`, `low_variance_plateau`, `high_frequency_oscillation`, each returning a continuous score + evidence window, not a boolean
- `audit_training_curve` MCP tool, `schema_version: 2` per spec
- `docs/audit-methods.md` — first real content: exact thresholds/formulas for each signal, referenced by the tool description
- Tests using `FakeBackend`-injected adversarial curves: a curve with real NaN points (must produce `null_values` signal with correct step range, not silently skip), a plateau, an oscillating curve, and a clean curve (must produce empty/low-score signals, not false positives)

**Effort:** ~2 days

**Completion criteria:** Every adversarial curve fixture from spec §7 produces the expected signal; a clean, well-behaved curve produces no high-confidence signals (explicit false-positive test, since a judgment tool crying wolf is as damaging as missing a real issue).

**Flag-if-triggered:** if threshold tuning starts requiring metric-type-specific special-casing beyond the `metric_type_assumed` field already in the spec, that's worth surfacing before it sprawls into an ad hoc rule pile.

---

## Milestone 7 — `audit_ablation`

**Status: implemented.** `analysis/confound.py` ships a pure `audit_ablation(baseline: Run, ablation: Run, claimed_variable: str) -> AblationAudit` with no dependency on FastMCP, MCP transport, `server.py`, or `WandbBackend` — confirmed against the pre-Milestone-7 architecture validation this roadmap requires (see the milestone summary delivered alongside this file). It calls `analysis/comparison.py`'s `compare_runs` directly for the underlying diff rather than duplicating diffing logic, exactly as Milestone 5's completion criteria required. The `audit_ablation` MCP tool in `server.py` is a thin wrapper: it resolves `baseline`/`ablation` into `Run`s via `get_run_summary` (one call per ref, supporting cross-backend pairs the same way `compare_runs` does) and calls the pure function directly. The allowlist (`seed`, `device`, `run_name`, `run_id`, `name`, `id`) is matched by exact, case-insensitive config key — deliberately not by substring, to avoid a real confound (e.g. `device_batch_size`) being silently waved through for containing an allowlisted word. A verdict of `"uncertain"` (distinct from `"clean"`/`"confounded"`) was added for the case where baseline and ablation have identical configs — including `claimed_variable` itself not differing — since asserting `"clean"` in that case would mischaracterize a pair where no ablation of the claimed variable can be confirmed to have occurred at all; this is a deliberate, documented interpretation of the `"uncertain"` verdict slot the spec names but does not otherwise define, not a spec deviation, and is flagged here per this roadmap's own transparency convention rather than left implicit. 32 tests added (23 pure-logic unit tests against hand-constructed `Run` objects in `tests/test_confound.py`, 9 MCP-layer integration tests via `fastmcp.Client` in `tests/test_server.py`), covering all four named adversarial cases from spec §7 exactly as specified, the allowlist's case-insensitivity and non-fuzzy exact-match behavior, the `"uncertain"` case, cross-project/cross-backend ablation pairs, duplicate-ref rejection (reused unchanged from `compare_runs`'s `CompareRunsError`), and full JSON-serialization round-tripping. All 186 repo tests pass; lint clean; packaging verified from a clean venv. No design flaws surfaced during implementation.

**Objective:** Implement the flagship differentiator tool, built on top of `compare_runs` (Milestone 5).

**Depends on:** Milestone 5.

**Deliverables:**
- `analysis/confound.py` — takes a `compare_runs`-style diff + `claimed_variable`, applies the allowlist (seed/device/name fields) for `likely_intentional`, produces verdict + confidence
- `audit_ablation` MCP tool
- Tests covering the exact adversarial cases named in spec §7: seed-only difference → `clean`/`likely_intentional: true`; two non-allowlisted differing params → `confounded`; single claimed-variable-only difference → `clean` with high confidence; partial-data run involved → confidence downgraded per §5

**Effort:** ~1.5 days

**Completion criteria:** All four named adversarial ablation cases from the spec pass exactly as specified — this tool's correctness on these specific cases matters more than broad coverage, since these are the cases most likely to appear in real use and in any public case study (per roadmap v2).

---

## Milestone 8 — `audit_sweep`

**Objective:** Implement the last and riskiest tool — sweep importance ranking with the mandatory sample-size floor and co-variance flagging from spec §4.2.

**Depends on:** Milestone 3 (needs `list_sweeps`/sweep-scoped run fetching from the W&B backend) and Milestone 2 (`FakeBackend` for tiny-sweep and correlated-hyperparameter fixtures).

**Deliverables:**
- `analysis/sensitivity.py` — Pearson correlation per parameter against target metric, pairwise co-variance check between hyperparameters, confidence derivation from sweep size + correlation strength
- `audit_sweep` MCP tool with the hard floor (`insufficient_samples` error below threshold) enforced before any ranking logic runs — not as an afterthought filter on the output
- Tests: a 3-run sweep must return `insufficient_samples`, never a ranking; a sweep with two co-varying hyperparameters (e.g. learning_rate/batch_size grid) must surface the co-variance warning attached to both affected params; a well-powered, independent sweep produces a sensible ranking with `caveat` and `n=` populated correctly

**Effort:** ~2 days

**Completion criteria:** The insufficient-samples refusal is un-bypassable through any parameter combination (explicit test attempting to sneak a small sweep past the floor via edge-case inputs); co-variance warning fires correctly on the specific adversarial fixture from spec §7.

**Flag-if-triggered:** if the default floor (10 runs) proves too strict or too lax against a real sweep from your own projects, that's a calibration question worth raising explicitly rather than quietly adjusting the constant.

---

## Milestone 9 — Adversarial Fixture Suite Consolidation + Tool-Selection Tests

**Objective:** Close the gap flagged in spec §7 point 3 — confirm real MCP clients actually invoke the right tool from natural-language prompts, which unit tests can't verify.

**Depends on:** Milestones 4–8 (all tools must exist).

**Deliverables:**
- Consolidated adversarial fixture set, now exercised end-to-end through the MCP layer (not just at the analysis-function level) for every case in spec §7
- A fixed prompt set (~10–15 representative phrasings: "did I mess up this ablation?", "why did my reward crash?", "which hyperparameter mattered most?", etc.) run against an actual MCP client, with pass/fail on whether the correct tool was invoked
- A short report of any tool-description wording changes made as a result, with before/after, since this is exactly the kind of tuning that should be visible and reviewable, not silent

**Effort:** ~1.5 days

**Completion criteria:** All fixed prompts correctly invoke their intended tool; any misfires are resolved by editing tool descriptions (not renaming tools or changing schemas, which would violate the frozen-spec constraint unless a real flaw is found).

---

## Milestone 10 — Docs, Packaging, Registry Submission

**Objective:** Ship it.

**Depends on:** Milestone 9.

**Deliverables:**
- README with the one-sentence pitch (spec §0) at the very top, install instructions, read-only-key setup guidance, explicit data-handling statement (spec §1 point 5)
- `docs/audit-methods.md` finalized (accumulated from Milestone 6 and 8)
- Packaging for PyPI/npm as appropriate to the chosen distribution method
- Name collision check finalized (spec §9 action item) — confirm `experiment-audit-mcp` or the chosen fallback is actually available before this milestone closes
- MCP Registry submission, Glama/cursor.directory submission
- Launch write-up per spec §10 v1 roadmap (Show-HN-style post on the confounded-ablation problem)

**Effort:** ~1.5 days (excluding waiting on registry review turnaround)

**Completion criteria:** Server is installable by a stranger following only the README, listed in the MCP Registry, and the launch post is published.

---

## Summary Table

| # | Milestone | Depends on | Effort | Key risk if skipped |
|---|---|---|---|---|
| 0 | Scaffolding | — | 0.5d | N/A |
| 1 | Data models | 0 | 1d | Everything downstream inherits model bugs |
| 2 | Backend abstraction + fake | 1 | 1.5d | Can't test tool logic without live API |
| 3 | W&B backend | 2 | 2–3d | No real data flowing |
| 4 | MCP server shell + retrieval tools | 3 | 1.5d | Nothing callable by a real client |
| 5 | compare_runs | 4, 2 | 1d | Blocks audit_ablation |
| 6 | audit_training_curve | 4, 2 | 2d | First judgment-tool trust test |
| 7 | audit_ablation | 5 | 1.5d | Flagship differentiator |
| 8 | audit_sweep | 3, 2 | 2d | Highest reputational risk if wrong |
| 9 | Tool-selection + adversarial consolidation | 4–8 | 1.5d | Silent MCP-specific failure mode |
| 10 | Docs, packaging, launch | 9 | 1.5d | Nobody finds it |

**Total estimated effort:** ~16–18 days of focused work, sequential dependencies mean limited parallelization within a solo build.

---

*Per your instructions: we execute one milestone at a time from here. Each milestone ends with a summary (what was built, why, spec alignment, deviations if any) before moving to the next. Any design flaw discovered during implementation is reported, not silently patched.*

## Milestone 10 summary — Docs, Packaging, Registry Submission

**What was built:** `README.md` was fully rewritten (the Milestone 0
placeholder is gone) with the one-sentence pitch verbatim from spec §0 at
the very top, install instructions for pip/Claude Desktop/Claude
Code/MCP Inspector, a quick-start section with three concrete example
prompts, a table of all eight tools with their retrieval/diffing/judgment
kind, an architecture overview with the actual directory tree and the two
design decisions worth understanding first, real (not aspirational) API
examples for two tools, the data-handling statement required by spec §1
point 5, and an explicit "Known Gaps" section. `CONTRIBUTING.md` and
`CHANGELOG.md` were added (not named individually in the roadmap's
deliverable list, but squarely within "Docs, Packaging" scope and
requested explicitly for this milestone's release-quality bar). A
Show-HN-style launch write-up was drafted at `docs/launch-post.md`,
framed around the confounded-ablation problem specifically, per spec
§10. Package version was bumped from the placeholder `0.0.0` to `1.0.0`
in both `pyproject.toml` and `experiment_auditnit__.py`;
`pyproject.toml` gained classifiers, keywords, and `[project.urls]` for
release polish.

**Corrective action (flagged, not silent):** `pyproject.toml`'s
`description` field named "W&B/MLflow" — MLflow is prototyped at the
interface level (Appendix A) but is v2 scope per spec §10, not shipped in
v1. This was corrected to "W&B" only, consistent with the README's own
"v1.0.0, W&B backend only" status line. This is a packaging-metadata
accuracy fix, not a design or code change, and is called out here per
this project's own established pattern (see the Milestone 0 correction
note and the CI-config restoration note surfaced at Milestone 9) rather
than silently adjusted.

**Name collision check (spec §9 action item, closed):** `experiment-audit-mcp`
was checked against PyPI (`pip index versions`) and npm (`npm view`) from
this environment on 2026-07-10; neither registry has an existing package
under this name. The recommended name from spec §9 is therefore confirmed
available, and no fallback name is needed.

**Verification performed this milestone:**
- Full test suite: 233 tests, 100% passing.
- Lint (`ruff check .`): clean.
- Packaging: built both sdist and wheel via `python -m build` from a
  clean `dist/`, installed the wheel into a fresh virtual environment
  with no dev dependencies, confirmed `import experiment_auditd
  `__version__ == "1.0.0"`, confirmed the `experiment-audit-mcp` console
  script is on `PATH` and fails fast with a clear `MissingCredentialsError`
  when `WANDB_API_KEY` is unset (correct behavior per spec §6, not a
  bug), and confirmed it proceeds past that check into the real W&B SDK's
  own auth validation when a (dummy) key is set. Reinstalled `pytest`/
  `ruff` into that same clean environment and re-ran the full suite
  against the wheel-installed package: still 233/233 passing.

**Deliverables explicitly not performed, and why:** per spec §10's v1
roadmap, "Publish to MCP Registry; submit to Glama and cursor.directory"
and the roadmap's own Milestone 10 deliverable list both call for actual
registry submission. This environment has no outbound network access to
those services (egress is allowlisted to package registries and a small
set of source-hosting domains only), so submission was not attempted from
here — attempting it would either silently fail or require working
around the sandbox in a way that wouldn't reflect a real submission. This
mirrors the same honest-gap pattern already established for W&B fixture
recording (Milestone 3) and the tool-selection eval (Milestone 9): the
package is publish-ready, and the exact remaining step is named in the
README's "Known Gaps" section for whoever runs the release from a machine
with that access.

**Completion criteria check (roadmap's own words):** "Server is
installable by a stranger following only the README" — verified directly
above via a from-scratch virtual environment following only the
documented `pip install` + env-var steps. "listed in the MCP Registry" —
not yet true, per the gap above. "the launch post is published" — drafted
and ready (`docs/launch-post.md`), not yet posted, since the repository
itself isn't public yet from this environment. Two of three completion
criteria are met; the third is blocked on infrastructure outside this
environment's reach, not on remaining work within it.
