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

**TDD note:** N/A — no logic yet.

---

## Milestone 1 — Data Models

**Objective:** Implement `models.py` exactly as specified in Design Spec §2 — `RunRef`, `Run`, `MetricPoint`, `MetricHistory`, `Sweep`, `Page[T]` — with full unit test coverage of construction, equality, and serialization.

**Depends on:** Milestone 0.

**Deliverables:**
- `src/experiment_audit_mcp/models.py` with all dataclasses, frozen where specified (`RunRef` must be frozen/hashable since it's used as a dict key candidate in `compare_runs` output)
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

## Milestone 3 — W&B Backend (real implementation)

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
