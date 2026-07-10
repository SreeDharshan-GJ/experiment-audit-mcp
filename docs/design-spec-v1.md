# ML Experiment Audit MCP Server — Canonical v1 Design Specification

**Status:** Frozen for implementation
**Working name:** `experiment-audit-mcp` (see §9 — Naming, for rationale and alternatives)

---

## Revision Log

**Revision 2 — `ExperimentBackend.list_runs` gains `page_size`.**

*Flaw:* §4.2 specified the MCP tool signature as
`list_runs(backend, project, filters?, cursor?, page_size=25)`, but the
`ExperimentBackend` ABC (§3, frozen in Milestone 2) never had a
`page_size` parameter at all — `WandbBackend` only accepted a page size
at construction time, fixed for the life of the backend instance. This
meant the `list_runs` MCP tool (Milestone 4) had no correct way to honor
a caller-supplied `page_size`: the two closest workarounds (silently
truncating results in the tool layer, or extending the frozen ABC without
flagging it) were both rejected — the former breaks `next_cursor`
continuity, the latter is exactly the kind of undocumented interface
drift the frozen-spec process exists to prevent. This was caught during
Milestone 4 implementation, before any tool code was written against the
mismatch, and reported rather than silently resolved.

*Fix:* `ExperimentBackend.list_runs` gains `page_size: int | None = None`
as its fourth parameter. `None` (the default) means "use whatever default
the backend instance was configured with," so every existing caller and
test that omits it is unaffected. `WandbBackend` uses it as a true
per-call override of `self._page_size`. `FakeBackend` accepts it (for ABC
signature parity) but does not enforce it, consistent with `FakeBackend`
already not implementing real cursor pagination — the same rationale
applies to page-size truncation, and nothing in spec §7's adversarial
cases or Milestone 2's completion criteria requires otherwise.

*Scope of this revision:* strictly additive — one new optional parameter
on one method. No other method signature, model, or tool contract
changes. `compare_runs`, `audit_ablation`, `audit_sweep` (§4.2) do not
call `list_runs` with an explicit `page_size` and are unaffected.

*Downstream updates made as part of this revision:* `backends/base.py`
(`ExperimentBackend.list_runs`), `backends/fake_backend.py`,
`backends/wandb_backend.py`, `tests/test_backend_base.py`,
`tests/test_fake_backend.py`, `tests/test_wandb_backend.py`, this design
specification (§3, §4.2), `implementation-roadmap-v1.md`.

---

**Revision 1 — `RunRef` gains `entity`.**

*Flaw:* `RunRef` was scoped as `{backend, project, run_id}`. W&B actually
scopes every run as `entity/project/run_id` — `entity` is the team or user
namespace a project lives under. Two different entities can each own a
project named e.g. `"mamfac"`, so `project` alone is not a sufficient
scope key: two runs from different entities with the same project name
and (theoretically) colliding run IDs would have been indistinguishable,
silently colliding as the same dict key and the same identity everywhere
a `RunRef` is used (including as a `compare_runs` dict key, per §2).

*Fix:* `RunRef` becomes `{backend, entity, project, run_id}`, `entity`
required (not optional), consistent with the frozen spec's own principle
that a `RunRef` must always be fully resolvable against the real API
without relying on an implicit default (§2, "Never pass a bare run_id
string between tools").

*Scope of this revision:* the `ExperimentBackend` abstract method
signatures (§3) are unchanged — `list_runs(project, filters, cursor)`
still takes `project` only, not `entity`. A backend instance is
configured with a single entity at construction time (e.g. via
`WANDB_ENTITY` or the authenticated user's default entity) and uses it
when building every `RunRef`/`Run` it returns. This was a deliberate
choice to keep this revision to the one genuine flaw found (`RunRef`
identity) rather than also renegotiating the backend interface itself,
which no validation step has flagged as a problem. If a future milestone
needs multi-entity access through a single backend instance, that is a
new, separately-justified design question, not an extension of this
revision.

*Downstream updates made as part of this revision:* `models.py`
(`RunRef`, `_runref_to_dict`), `tests/test_models.py`,
`tests/test_fake_backend.py`, this design specification, and
`implementation-roadmap-v1.md`. `backends/base.py` and
`backends/fake_backend.py` required no changes — both already treat
`RunRef` as an opaque, fully-scoped value and never reconstruct it
field-by-field.

---

## 0. One-Sentence Pitch

> Your agent can catch confounded ablations, training pathologies, and misleading sweep conclusions across dozens of runs that you'd otherwise have to notice by eye — this is leverage on researcher attention, not another dashboard.

This sentence goes verbatim at the top of the README, before any tool list or install instructions.

---

## 1. Design Principles (non-negotiable across all versions)

1. **Retrieval and judgment are structurally separate.** An agent or human must be able to tell, from the tool name and schema alone, whether a result is a deterministic fact or a heuristic judgment that could be wrong.
2. **Judgment tools show their work.** No tool returns a bare verdict. Every judgment ships with the evidence and method that produced it.
3. **Refuse rather than mislead.** When a computation would be statistically unreliable (e.g., importance ranking on 3 runs), the tool refuses with a clear message rather than returning a shaky number.
4. **Deterministic computation is the product.** The pitch is not "smarter than an LLM reasoning over JSON" — it's "reliable, cheap, and exact for a task LLMs are bad at (comparing dozens of floats across dozens of runs)."
5. **Data never leaves the user's machine** except calls to the user's own W&B/MLflow endpoint. Stateless, open source, auditable. State this explicitly in the README.

---

## 2. Data Model (`models.py`)

Establishing identity and scoping correctly here is the single most important fix from review — every tool signature depends on it.

```python
@dataclass(frozen=True)
class RunRef:
    """Fully-scoped run identity. Never pass a bare run_id string between tools."""
    backend: str          # "wandb" | "mlflow"
    entity: str            # required — team/user namespace; project names collide across entities (Revision 1)
    project: str          # required — run IDs are project-scoped in W&B
    run_id: str

@dataclass
class Run:
    ref: RunRef
    name: str
    tags: list[str]
    status: str            # "running" | "finished" | "crashed" | "failed"
    created_at: datetime
    config: dict[str, Any]         # hyperparameters
    summary_metrics: dict[str, float]  # final/latest values only — cheap
    data_completeness: Literal["complete", "partial", "unknown"]  # see §5

@dataclass
class MetricPoint:
    step: int
    value: float | None    # None represents a logged NaN/null — never silently dropped

@dataclass
class MetricHistory:
    ref: RunRef
    metric_name: str
    points: list[MetricPoint]
    schema_version: int = 1

@dataclass
class Sweep:
    ref: RunRef  # backend + project; sweep_id lives in a separate field
    sweep_id: str
    method: str  # "grid" | "random" | "bayes" | "unsupported"
    run_refs: list[RunRef]
    target_metric: str | None
```

**Cross-project behavior:** tools taking multiple runs (`compare_runs`, `check_ablation_validity`) accept a list of full `RunRef` objects, not bare strings. Cross-project comparisons are allowed and explicitly supported — the tool does not require matching projects, but the output always echoes back each run's project so nothing is ambiguous in a truncated context window.

---

## 3. Backend Abstraction (`backends/base.py`)

```python
class BackendCapability(Enum):
    SWEEPS = "sweeps"
    ARTIFACTS = "artifacts"

class ExperimentBackend(ABC):
    name: str
    capabilities: set[BackendCapability]  # declared per-backend; see below

    @abstractmethod
    async def test_connection(self) -> ConnectionStatus: ...

    @abstractmethod
    async def list_runs(
        self, project: str, filters: RunFilter, cursor: str | None = None,
        page_size: int | None = None,  # Revision 2
    ) -> Page[Run]: ...

    @abstractmethod
    async def get_run_summary(self, ref: RunRef) -> Run: ...

    @abstractmethod
    async def get_metric_history(
        self, ref: RunRef, metric: str, step_range: tuple[int, int] | None = None
    ) -> MetricHistory: ...

    def list_sweeps(self, project: str) -> list[Sweep]:
        raise NotSupportedError(self.name, BackendCapability.SWEEPS)
```

**Key decisions from review:**

- **Capability declaration, not blanket abstract methods.** `list_sweeps` has a default `NotSupportedError` implementation rather than being a required abstract method. This directly resolves the MLflow-sweep-shape problem: MLflow has no first-class native Sweep object the way W&B does. Rather than forcing a fictional mapping, the MLflow backend declares `capabilities = {ARTIFACTS}` (no `SWEEPS`), and `summarize_sweep` checks capability before calling, returning a clear "not supported by this backend" error instead of a bad shim.
- **Pagination is part of the interface from v1**, via a generic `Page[T]` cursor wrapper (`{items: [...], next_cursor: str | None}`). Every list-returning backend method returns `Page[T]`. This is a breaking change to retrofit later, so it ships now even though no v1 user has 500-run projects yet.
- **`page_size` is a per-call override (Revision 2), not a fixed backend setting.** `None` defers to the backend's configured default; a caller (ultimately the `list_runs` tool, §4.2) can request a smaller or larger page without reconstructing the backend. `FakeBackend` accepts but does not enforce it, matching its existing no-op stance on `cursor`.
- **MLflow sweep mapping was prototyped against this interface before freezing it** (see Appendix A) — confirms the capability-flag approach survives contact with a second backend without forcing a redesign.

---

## 4. Tool Design

### 4.1 Naming convention (structural trust signal)

Tools are namespaced by prefix so the trust level is visible in the tool name itself, not just the docs — this is how an agent (and a human skimming a tool list) can tell retrieval from judgment at a glance:

| Prefix | Meaning | Tools |
|---|---|---|
| `get_*` / `list_*` | Pure retrieval. Deterministic, no interpretation. | `test_connection`, `list_runs`, `get_run_summary`, `get_metric_history` |
| `compare_*` | Deterministic computation over multiple runs (diffing). No judgment calls. | `compare_runs` |
| `audit_*` | Heuristic judgment. Always returns `confidence` + `method` + raw evidence. | `audit_ablation`, `audit_training_curve`, `audit_sweep` |

Every `audit_*` tool's output schema **requires** (not optionally includes) three fields: `method` (string describing exactly what heuristic ran), `confidence` (`"high" | "medium" | "low"` — never a bare bool verdict alone), and `evidence` (the raw data that produced the judgment). This is enforced at the schema level, not by convention alone, so it can't be forgotten when a new audit tool is added in v3.

### 4.2 Tool specifications

**`test_connection()`**
Runs on install / first use. Validates credentials against the configured backend with a lightweight call. Returns `{backend, authenticated: bool, scopes_detected, error?}`. Failing fast here, rather than three tool calls deep into a task, is the single highest-leverage fix for first-impression developer experience.

**`list_runs(backend, project, filters?, cursor?, page_size=25)`**
→ `Page[RunSummary]` where `RunSummary` omits full config/metrics (id, name, tags, status, created_at only) to keep the default cheap. Filters: `tags`, `status`, `created_after`, `created_before`. `page_size` is passed through to `ExperimentBackend.list_runs`'s `page_size` parameter (Revision 2); the tool's own default of `25` is a tool-layer default, distinct from whatever default a backend instance was constructed with.

**`get_run_summary(ref: RunRef)`**
→ full `Run` object: config + summary_metrics + `data_completeness` flag. Does **not** include metric history — that's a separate call by design (see §4.3 data-flow note).

**`get_metric_history(ref: RunRef, metric: str, step_range?: [int, int])`**
→ `MetricHistory` for exactly one metric. Replaces the old bolted-on `full_history: bool` flag on `get_run_summary` — cleaner separation of "cheap default" vs. "explicit, scoped expensive call." This is the tool `audit_training_curve` calls internally; **the data flow is explicit**: `audit_training_curve` always fetches its own history via this same method rather than accepting a pre-fetched blob, so there is one code path for "how do I get a curve" and no ambiguity about staleness.

**`compare_runs(refs: list[RunRef])`**
→ `{config_diff: [{param, values: {ref: value}}], metric_diff: [{metric, values: {ref: value}, delta}]}`
Pure diffing. No verdict, no confidence field needed — this is why it keeps the `compare_` prefix rather than `audit_`.

**`audit_training_curve(ref: RunRef, metric: str)`**
Replaces `detect_divergence`. Returns **scored signals, not a fixed label taxonomy** — resolving the review concern that a locked enum (`nan_spike | plateau | reward_collapse | oscillation`) would be under-validated and metric-type-blind.

```
→ {
    schema_version: 2,
    metric_type_assumed: "loss" | "reward" | "unknown",  # inferred from metric name / declared by caller
    signals: [
      {
        signal: "null_values" | "sudden_jump" | "low_variance_plateau" | "high_frequency_oscillation",
        score: float,          # continuous, not a boolean flag
        step_range: [int, int],
        evidence: {raw values in that window},
        confidence: "high" | "medium" | "low"
      }
    ],
    method: "threshold-based, see docs/audit-methods.md#training-curve"
  }
```
Labeling ("this is reward collapse") is left to the calling agent/human by thresholding scores — this is more falsifiable and avoids shipping an unvalidated fixed taxonomy in v1. The taxonomy can grow in v2/v3 without breaking existing consumers, since `schema_version` is bumped and additive.

**`audit_ablation(baseline: RunRef, ablation: RunRef, claimed_variable: str)`**
Replaces `flag_confound` / `check_ablation_validity`. `claimed_variable` is **required, explicit** — v1 does not attempt to infer intent from run naming/tags (resolves the open question from the prior design pass; inference is deferred to v3 at earliest, as an opt-in, clearly-labeled "experimental" mode, never the default).

```
→ {
    verdict: "clean" | "confounded" | "uncertain",
    confidence: "high" | "medium" | "low",
    differing_params: [{param, baseline_value, ablation_value, likely_intentional: bool}],
    method: "full config diff against claimed_variable; params tagged intentional if name matches claimed_variable or is a known seed/infra field",
    evidence: <the full compare_runs-style diff>
  }
```
`likely_intentional` uses a conservative allowlist (seed, device, run name/id fields) rather than guessing — anything not on the allowlist and not the claimed variable counts against the verdict.

**`audit_sweep(sweep_ref, target_metric?)`**
Replaces `summarize_sweep`. **Hard sample-size floor**: refuses to return an importance ranking below a minimum run count (default floor: 10 runs; configurable but defaults conservative), returning `{error: "insufficient_samples", run_count, minimum_required}` instead of a misleading ranking. When it does return a ranking, **sweep size and method travel with the result every time**, not just in a header a caller might drop:

```
→ {
    sweep_size: int,
    parameter_importance: [{param, correlation, rank, warning?: "co-varies with X"}],
    caveat: "Correlation-based; unreliable with correlated hyperparameters or small sweeps. n=<sweep_size>.",
    confidence: "high" | "medium" | "low",  # derived from sweep_size and correlation strength together
    method: "pairwise Pearson correlation with target_metric; co-variance flagged where |corr(param_i, param_j)| > 0.6"
  }
```
Basic co-variance flagging between hyperparameter pairs (e.g., learning_rate vs. batch_size in a grid) is included in v1 specifically because review flagged this as the most likely source of a viral "your tool was wrong" screenshot — cheap to add, high reputational protection.

### 4.3 Explicit data flow (resolves the ambiguity flagged in review)

```
audit_training_curve(ref, metric)
    └── calls get_metric_history(ref, metric) internally — always live, never pre-fetched/passed in

audit_ablation(baseline, ablation, claimed_variable)
    └── calls compare_runs([baseline, ablation]) internally, then applies the intentional-param allowlist

audit_sweep(sweep_ref, target_metric)
    └── calls list_runs(...) scoped to the sweep's run_refs, then get_run_summary for each (config + summary_metrics only — never full history, keeps this cheap even for large sweeps)
```

No audit tool accepts raw data as a parameter. This keeps one code path per data type and eliminates the staleness ambiguity flagged in review.

---

## 5. Error Handling

```python
@dataclass
class ToolError:
    error_type: Literal[
        "auth_failed", "rate_limited", "run_not_found",
        "backend_unsupported_capability", "insufficient_samples",
        "partial_data", "unknown"
    ]
    message: str
    recoverable: bool
    retry_after_seconds: int | None = None
```

**`partial_data` is new in this revision** — resolves the review gap around W&B's real soft-fail states (run exists but is still ingesting, artifacts still processing). Rather than forcing this into `run_not_found` (wrong) or silently returning incomplete data as if complete (dangerous — this is exactly the "confidently wrong" failure mode), every `Run` object carries `data_completeness: "complete" | "partial" | "unknown"`, and any audit tool operating on a partial-data run downgrades its own `confidence` to `"low"` automatically and says why in `method`.

Rate limiting: backoff handled once in the backend layer (exponential, capped), never duplicated per-tool. Auth failures surface immediately and specifically, distinguished from generic network errors.

---

## 6. Developer Experience

- **`test_connection` runs automatically on server start** where the MCP transport allows it, and is additionally exposed as a callable tool for mid-session re-checks.
- **Tool descriptions kept minimal in the schema itself.** Full methodology detail (the threshold logic behind `audit_training_curve`, the correlation caveats behind `audit_sweep`) lives in a `docs/audit-methods.md` file referenced by a short pointer in each tool's description, not repeated in full in every schema — this was flagged in review because verbose tool descriptions cost context budget on *every* turn of a conversation, not just when the tool is actually invoked.
- **Setup:** env-var credentials (`WANDB_API_KEY`, or `MLFLOW_TRACKING_URI` + token), matching the convention already established by `GITHUB_PERSONAL_ACCESS_TOKEN`-style servers, minimizing the "new pattern to learn" cost for adoption.
- **Recommended default: read-only API keys**, documented as the first setup step, directly addressing the trust barrier around handing over research data credentials.

---

## 7. Testing Strategy

**Fixture categories (recorded real API responses, not hand-written mocks):**

1. **Happy path:** clean runs, clean sweeps, well-behaved metric curves.
2. **Adversarial / edge cases (new in this revision, directly from review):**
   - A sweep with 3 runs → must trigger `insufficient_samples`, not a ranking.
   - An ablation pair where exactly one parameter differs and it's a random seed → must be flagged `likely_intentional: true` via the allowlist, verdict should lean `clean`, not `confounded`.
   - An ablation pair with two differing params, neither on the allowlist → must return `confounded`.
   - A real crashed run with logged NaN values mid-curve → `audit_training_curve` must surface a `null_values` signal, not silently skip the points.
   - A run mid-ingestion (`data_completeness: "partial"`) → any audit tool touching it must downgrade confidence and explain why.
   - Two correlated hyperparameters in a grid sweep → `audit_sweep` must surface the co-variance warning, not present both as independently important.
3. **Tool-selection tests (MCP-specific, new in this revision):** a fixed set of representative natural-language prompts ("did I mess up this ablation?", "why did my reward crash?", "which hyperparameter actually mattered?") run against Claude/other MCP clients to confirm the *right* tool gets invoked from its name/description alone — a correct tool that never gets called due to a mismatched description is a silent failure mode specific to this kind of software and doesn't show up in unit tests.
4. **API-drift regression:** fixtures pinned to a documented W&B/MLflow API version range; CI flags when live smoke tests diverge from recorded fixture shapes.

---

## 8. Extensibility / Versioning

- `schema_version` field on every structured response (starts at relevant versions per tool, e.g. `audit_training_curve` ships at `schema_version: 2` reflecting the signals-not-labels redesign in this revision). Additive changes bump the version; breaking changes require a new tool name or a major server version bump, documented in the changelog.
- Backend interface additions use default `NotSupportedError` implementations (per §3), so adding a 5th backend method in v2 does not break v1-only backends.
- New `audit_*` tools in v2/v3 must implement the mandatory `method` / `confidence` / `evidence` schema from day one — this is now a documented contribution requirement, not just a convention, so external contributors (v3 roadmap includes opening to PRs) can't accidentally ship a bare-verdict tool.

---

## 9. Naming

`experiment-insight-mcp` (original) and `mlrun-analyst-mcp` (alternative) are both rejected — too generic, sound like BI/dashboard products, don't signal the actual differentiator (rigor/sanity-checking, not visualization).

**Recommended: `experiment-audit-mcp`.** "Audit" directly signals the rigor/verification framing that differentiates this from every existing W&B/MLflow integration, and matches the `audit_*` tool prefix chosen in §4.1 for internal consistency between the project name and its own API.

**Action before implementation:** check `experiment-audit-mcp` for collisions on PyPI, npm, and the MCP Registry before writing setup.py / package.json. If taken, fallback candidates in the same register: `ablation-audit-mcp` (narrower, very on-the-nose for the flagship tool), `sanity-check-mcp` (broader, less ML-specific — weaker).

---

## 10. Roadmap

**v1 — W&B only, ship something rock-solid**
- Tools: `test_connection`, `list_runs`, `get_run_summary`, `get_metric_history`, `compare_runs`, `audit_ablation`, `audit_training_curve`, `audit_sweep`
- W&B backend only; MLflow's sweep-mapping already prototyped against the abstract interface (Appendix A) to validate extensibility, but not shipped
- Full fixture suite including adversarial cases (§7)
- Read-only-key setup docs, MIT license, `docs/audit-methods.md`
- Publish to MCP Registry; submit to Glama and cursor.directory
- Launch wedge: a "Show HN"-style post framed around the confounded-ablation problem specifically, not a generic feature announcement

**v2 — MLflow backend + trust-building**
- `MLflowBackend` implementing the same interface; `capabilities` declared per §3 (no SWEEPS support initially, documented clearly rather than faked)
- Versioned compatibility matrix against W&B/MLflow API versions, tested in CI
- First public case study: a concrete confound or pathology caught in a real project (ideally from your own prior MAMFAC/CARM++ ablation work, where you already know exactly what a broken ablation looks like)

**v3 — deeper RL-specific differentiation**
- RL-specific pathology signals: reward-hacking heuristics, multi-seed variance analysis with proper statistical tests (paired t-test, Cohen's d)
- Optional experimental `claimed_variable` inference mode for `audit_ablation` (opt-in, clearly labeled, never default — per the deferred decision in §4.2)
- Optuna/Ray Tune sweep support alongside native backends
- Open to external contributions; new `audit_*` tools must conform to the mandatory method/confidence/evidence schema (§8)

---

## Appendix A: MLflow Sweep-Mapping Prototype (validation only, not shipped in v1)

MLflow has no first-class "Sweep" object equivalent to W&B's. The closest analog is a parent run with nested child runs (common pattern for `hyperopt`/`optuna`-driven MLflow experiments) or simply a set of runs sharing an experiment ID and a naming convention.

Validated approach for v2: `MLflowBackend.list_sweeps()` is **not implemented** — the backend declares `capabilities = {ARTIFACTS}` only (no `SWEEPS`), and `audit_sweep` returns a `backend_unsupported_capability` error when called against an MLflow ref, with a message pointing users to `list_runs` + `compare_runs` as the manual alternative. This confirms the capability-flag design in §3 survives contact with a structurally different second backend without forcing a fictional Sweep mapping — the exact test a reviewer would ask for before trusting the abstraction.

---

*This specification is frozen as of this design pass. Implementation should treat any deviation from this document as requiring an explicit, logged design decision, not a silent choice made in code.*
