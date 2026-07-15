# Reference: MCP Integration — Tool Schemas, Thresholds, and Formulas

This file documents the **`experiment-audit-mcp` integration** — one of
several ways the Experiment Audit reasoning engine pulls in evidence
(alongside the CLI and Python package, and alongside reasoning directly
over data the user provides). It has nothing to say about non-MCP
reasoning; for the discipline that applies everywhere, see `prompts.md`.

Everything in this file reflects the actual v1.0.0 implementation of
`experiment-audit-mcp` — no aspirational or planned (v2/v3) behavior is
included except where explicitly labeled "not yet implemented."

## Table of contents

- [Core identity type: RunRef](#core-identity-type-runref)
- [test_connection](#test_connection)
- [list_runs](#list_runs)
- [get_run_summary](#get_run_summary)
- [get_metric_history](#get_metric_history)
- [compare_runs](#compare_runs)
- [audit_training_curve](#audit_training_curve)
- [audit_ablation](#audit_ablation)
- [audit_sweep](#audit_sweep)
- [Error taxonomy](#error-taxonomy)

---

## Core identity type: RunRef

Every tool that identifies a run or sweep takes a fully-scoped reference,
never a bare ID string:

```json
{"backend": "wandb", "entity": "your-team", "project": "mamfac", "run_id": "xj29fk1a"}
```

`entity` (the W&B team/user namespace) is required because two different
entities can have projects with the same name — `project` alone is not a
safe scope key. `list_runs` results always return the full `ref` for
each run so you can chain into `get_run_summary` / `get_metric_history`
without asking the user to repeat identifying info.

A `Sweep` reference (`SweepRefInput`, used by `audit_sweep`) has the same
shape but with `sweep_id` instead of `run_id`.

---

## test_connection

**Signature:** `test_connection()` — no arguments. Assumes exactly one
configured backend (a multi-backend `test_connection` is an open v2
design question, not yet resolved).

**Returns:** a `ConnectionStatus` dict — `{backend, authenticated, scopes_detected, error?}`.

**When to call it:** first thing in a session, or immediately after any
tool call fails with `auth_failed`, to confirm whether the problem is
credentials versus something else (bad run ID, wrong project name, etc).
Only the *presence* of `WANDB_API_KEY` is checked automatically at server
startup (fail-fast) — whether the key actually authenticates is only
confirmed by calling this tool.

---

## list_runs

**Signature:** `list_runs(backend, project, filters?, cursor?, page_size=25)`

`filters` (all optional, unset = no filtering): `tags: list[str]`,
`status: str`, `created_after: datetime`, `created_before: datetime`.

**Returns:** `Page[RunSummary]` — `{items: [...], next_cursor}`. Each
item is a **lightweight** projection: `{ref, name, tags, status,
created_at, data_completeness}` — no `config`, no `summary_metrics`.
Call `get_run_summary` on a specific run's `ref` when you need those.

**Cost note:** this is the cheap way to enumerate/browse runs. Don't
loop `get_run_summary` over every run just to answer a listing question.

---

## get_run_summary

**Signature:** `get_run_summary(ref: RunRef)`

**Returns:** the full `Run` object:

```json
{
  "ref": {"backend": "...", "entity": "...", "project": "...", "run_id": "..."},
  "name": "...",
  "tags": ["..."],
  "status": "running | finished | crashed | failed",
  "created_at": "ISO-8601",
  "config": {"...": "arbitrary hyperparameters"},
  "summary_metrics": {"...": "final/latest float values"},
  "data_completeness": "complete | partial | unknown"
}
```

`data_completeness` matters: `"partial"` means the run is still
ingesting or the record is otherwise known-incomplete — every downstream
`audit_*` tool that touches a partial-data run downgrades its own
`confidence`. `"unknown"` is a distinct state from `"partial"` (backend
hasn't determined completeness) and does **not** trigger that downgrade.
Does **not** include metric history — that's `get_metric_history`, by
deliberate design (keeps "cheap default" and "explicit expensive call"
separate).

---

## get_metric_history

**Signature:** `get_metric_history(ref: RunRef, metric: str, step_range?: [int, int])`

**Returns:** `MetricHistory` — `{ref, metric_name, points: [{step, value}], schema_version}`.

`value` is `float | None`. `None` represents a **logged NaN**, preserved
exactly — never dropped, filled, or interpolated. If you're summarizing
this data yourself (rather than calling `audit_training_curve`), do not
silently skip `None` points; mention them.

This is also the internal data source `audit_training_curve` calls —
always live, never a stale/pre-fetched blob passed in.

---

## compare_runs

**Signature:** `compare_runs(refs: list[RunRef])` — 2 or more runs, may
span different projects or backends; every value is keyed by its run's
full ref so nothing is ambiguous.

**Returns:**

```json
{
  "config_diff": [
    {"param": "batch_size", "values": {"<ref-key>": {"present": true, "value": 64}, "<ref-key>": {"present": true, "value": 32}}}
  ],
  "metric_diff": [
    {"metric": "reward", "values": {"<ref-key>": 0.82, "<ref-key>": 0.61}, "delta": 0.21}
  ]
}
```

Only parameters/metrics that actually **differ** across the compared
runs appear — this is a diff, not a full dump of every config key.
`present: false` distinguishes "this run's config never had this key" from
an explicit `None`/null value — don't conflate the two. `delta` for N > 2
runs is `max(present values) - min(present values)` (the full spread),
not a pairwise subtraction — the natural N-way generalization. `delta`
is `None` when fewer than two runs actually logged that metric.

**No verdict, no confidence field** — this is intentional (hence the
`compare_` prefix, not `audit_`). Never phrase a `compare_runs` result as
"clean" or "confounded" yourself — that judgment is `audit_ablation`'s
job specifically because it requires knowing which variable was
*claimed* to be isolated.

---

## audit_training_curve

**Signature:** `audit_training_curve(ref: RunRef, metric: str)` — fetches
`get_metric_history` internally; always live.

**Returns:**

```json
{
  "schema_version": 2,
  "metric_type_assumed": "loss | reward | unknown",
  "signals": [
    {"signal": "sudden_jump", "score": 0.91, "step_range": [40120, 40140], "evidence": {...}, "confidence": "high"}
  ],
  "method": "threshold-based, see docs/audit-methods.md#training-curve"
}
```

`metric_type_assumed` is inferred from the metric **name only**
(case-insensitive substring: contains `"loss"` → `loss`; contains
`"reward"` → `reward`; else `"unknown"`) and is reported as context —
**none of the four detectors change behavior based on it.** Don't imply
otherwise to the user (e.g. don't say "because this is a reward metric,
the tool checked for reward hacking" — it didn't branch on that).

An empty `signals` list is a real, clean result — not a failure or
"nothing detected yet." State it as "no pathology signals fired" rather
than treating it as inconclusive.

Signals are **not mutually exclusive**; more than one can legitimately
fire on the same curve.

### The four detectors

**`null_values`** — fires if *any* point's `value` is `None`, regardless
of what fraction of the curve. `score` = null count / total count.
`confidence` is always `"high"` (presence of a logged NaN is a fact, not
a judgment call) — this is the one detector whose confidence does not
come from the score-bucketing rule below.

**`sudden_jump`** — a robust median/MAD z-score over the curve's *rate of
change* (`(value[i+1]-value[i]) / step_gap`, not raw diff — dividing by
the step gap avoids a NaN-adjacent point falsely widening the diff).
Requires ≥3 present points. `z = |rate - median(rates)| / (1.4826 × MAD)`,
where MAD is floored at `max(1e-9, 0.01 × scale)` (scale = curve's value
range) to avoid a near-constant series blowing up the z-score. Fires only
if the **max** z-score across the curve is `≥ 4.0`. `score = min(max_z / 8.0, 1.0)`.
**Known limitation:** uses one global median/MAD, not a windowed/local
baseline — so `sudden_jump` can co-fire at the *edges* of a genuine
`low_variance_plateau` (the tiny in-plateau fluctuation looks anomalous
against the whole curve's larger typical rate). This is a real precision
limitation, not a bug — mention it if both signals fire near the same
step range.

**`low_variance_plateau`** — longest contiguous run of a 5-point sliding
window whose coefficient of variation (population stdev / curve's value
range) is `< 0.02`. Only the single longest such run is reported, even if
multiple flat regions exist. Requires ≥5 present points. `score = min(run_length / 10, 1.0)`.

**`high_frequency_oscillation`** — fraction of adjacent non-zero-diff
pairs that flip sign. Requires ≥6 present points and ≥2 non-zero diffs
after excluding exact-zero diffs. Fires only if `sign_flip_ratio ≥ 0.7`.
`score = sign_flip_ratio` directly (already 0–1).

**Confidence bucketing** (the three score-derived detectors only —
`null_values` is always `"high"`): `score ≥ 0.85 → "high"`;
`score ≥ 0.6 → "medium"`; else `"low"`.

---

## audit_ablation

**Signature:** `audit_ablation(baseline: RunRef, ablation: RunRef, claimed_variable: str)`

`claimed_variable` is **required and explicit** — the tool never tries
to infer intent from run names/tags. If the user hasn't told you which
variable they meant to change, ask rather than guess.

**Returns:**

```json
{
  "schema_version": 1,
  "verdict": "clean | confounded | uncertain",
  "confidence": "high | medium | low",
  "differing_params": [
    {"param": "use_memory", "baseline_value": true, "ablation_value": false, "likely_intentional": true},
    {"param": "batch_size", "baseline_value": 64, "ablation_value": 32, "likely_intentional": false}
  ],
  "method": "full config diff against claimed_variable; params tagged intentional if name matches claimed_variable or is on the allowlist (seed, device, run name/id)",
  "evidence": { "config_diff": [...], "metric_diff": [...] }
}
```

`evidence` is the **full** `compare_runs`-style diff (config AND
metrics), not just the parameters that drove the verdict — always check
whether the target metric actually moved, not just the config diff, when
summarizing for the user.

### The allowlist

`{"seed", "device", "run_name", "run_id", "name", "id"}` — matched
**case-insensitively on the exact key only, never by substring**. A key
like `device_batch_size` is NOT matched by `"device"`. This is a
deliberate conservative choice: a false `confounded` (a benign,
differently-named field like `random_seed` gets flagged) is preferred
over a false `clean` from fuzzy matching hiding a real confound. If you
see an unaccounted param that's obviously seed/infra-flavored but not
exactly on the allowlist, say so explicitly rather than either silently
excusing it or treating the tool's flag as unquestionable.

### Verdict logic

- **No config parameters differ at all** → `"uncertain"` — deliberately
  *not* `"clean"`, since nothing changing (including `claimed_variable`
  itself) isn't evidence of a valid ablation; it may mean the wrong runs
  were compared.
- **Every differing param is `claimed_variable` or on the allowlist** →
  `"clean"`.
- **At least one differing param is neither** → `"confounded"`.

### Confidence

- **`"low"`**, unconditionally, if either run's `data_completeness ==
  "partial"` — appended as a reason string in `method`. This overrides
  everything below; a `"confounded"` verdict stays `"confounded"` but at
  low confidence.
- **`"low"`** for an `"uncertain"` verdict, independent of the above.
- **`"high"`** for `"clean"`/`"confounded"` otherwise — this verdict is a
  direct deterministic consequence of the diff and the fixed allowlist,
  not a continuous score, so there's no `"medium"` tier here.

---

## audit_sweep

**Signature:** `audit_sweep(sweep_ref: SweepRef, target_metric?: str)` —
`target_metric` defaults to the sweep's own recorded target metric if
omitted.

**Returns (success):**

```json
{
  "schema_version": 1,
  "sweep_size": 24,
  "usable_run_count": 22,
  "target_metric": "reward",
  "parameter_importance": [
    {"param": "learning_rate", "correlation": 0.71, "rank": 1, "p_value": 0.0003, "warning": "co-varies with batch_size (|r|=0.74)"}
  ],
  "excluded_parameters": [
    {"param": "optimizer", "reason": "non_numeric"}
  ],
  "caveat": "Correlation-based; unreliable with correlated hyperparameters or small sweeps. n=22.",
  "confidence": "high | medium | low",
  "method": "pairwise Pearson correlation with target_metric; co-variance flagged where |corr| >= 0.7"
}
```

**Returns (refusal, below the sample floor):**

```json
{"error": {"error_type": "insufficient_samples", "message": "...", "recoverable": false, ...}, "run_count": 3, "minimum_required": 10}
```

### The hard sample-size floor

`DEFAULT_MINIMUM_SAMPLES = 10`, enforced **before any ranking runs**, in
two places: (1) against the sweep's raw run count, and (2) against the
*usable* run count (runs that actually logged a numeric `target_metric`
value). Never route around a refusal by asking for a ranking a different
way — a sweep that refuses genuinely doesn't have enough data for a
statistically meaningful ranking. Tell the user what would unblock it:
more completed runs, or checking why runs aren't logging the target
metric.

### Which parameters get excluded (and why)

Checked in this order — a parameter is excluded from `parameter_importance`
into `excluded_parameters` with a `reason` if:

- **`non_numeric`** — never numeric (`int`/`float`/`bool`) on any usable
  run (e.g. an optimizer name string). `bool` counts as numeric (0/1) —
  correlating a binary hyperparameter is Pearson's standard point-biserial
  case, not an encoding artifact.
- **`insufficient_overlap`** — numeric, but present on fewer than
  `minimum_samples` usable runs (e.g. a conditional hyperparameter).
- **`constant`** — numeric and present on enough runs, but never varies.
- **`target_metric_constant`** — the target metric itself never varies
  across usable runs. In this case *every* parameter is excluded this
  way and the tool returns a normal (non-error) result: empty
  `parameter_importance`, `"low"` confidence, and a `caveat` explaining
  why — an honestly-empty ranking, not a fabricated one.

### Co-variance warning

Despite the name, this is **correlation**, not raw covariance (covariance
isn't scale-invariant). For every pair of *included, ranked* parameters,
restricted to runs where both have a numeric value: flagged if
`|corr(param_i, param_j)| ≥ 0.7`. Both parameters' entries name the other
and its `|r|`. **If two parameters both rank highly and warn about each
other, do not present both as independently important** — say the effect
may be attributable to either or both, and the ranking can't disentangle
them.

### Confidence: Fisher z-transformation significance test

Not a bare magnitude cutoff — a real two-tailed significance test:
`z = atanh(r)`, `SE = 1/sqrt(n-3)`, `z_statistic = z/SE`,
`p = 2 × (1 - Φ(|z_statistic|))`. The overall `confidence` is derived from
the **top-ranked** (largest `|r|`) parameter's p-value:
`"high"` → `p < 0.01`; `"medium"` → `0.01 ≤ p < 0.05`; `"low"` → `p ≥ 0.05`
or no p-value computable. Every `ParameterImportance` entry also carries
its own `p_value` — surface it for the top result rather than only the
correlation coefficient, since a moderate correlation from a small `n`
can still land at `"low"` confidence.

### Two things to always caveat, even if the tool's own text doesn't repeat them every time

1. **Linear-only.** Pearson correlation misses non-monotonic effects
   (e.g. a learning rate with an interior optimum). A near-zero-ranked
   parameter is not proven unimportant — it may have a non-linear effect
   this method can't see. This is a documented method limitation, not a
   bug.
2. **No multiple-comparisons correction.** Ranking many parameters from a
   modest sample means *some* parameter will show a moderately large
   correlation by chance alone, especially near the sample floor. Don't
   present a borderline-significant mid-pack parameter as confidently
   important without noting this.

---

## Error taxonomy

Every tool can return `{"error": {"error_type", "message", "recoverable", "retry_after_seconds"}}`
instead of its normal result. The seven `error_type` values, and what to
do about each:

| `error_type` | Meaning | What to do |
|---|---|---|
| `auth_failed` | Credentials missing or rejected by W&B. | Tell the user to check `WANDB_API_KEY`; suggest `test_connection` to confirm once fixed. Don't retry the same call. |
| `rate_limited` | Backend rate limit hit (backoff already attempted internally). | Respect `retry_after_seconds` if present; don't hammer retries yourself. |
| `run_not_found` | The `run_id`/`entity`/`project` combination doesn't resolve. | Double-check the `RunRef` fields with the user; consider `list_runs` to find the correct ID rather than guessing variations. |
| `backend_unsupported_capability` | The backend doesn't support this operation (e.g. sweeps on a backend without native sweep support). | Explain the capability gap plainly; don't attempt a manual workaround that fabricates the missing capability. |
| `insufficient_samples` | `audit_sweep` refused below the 10-run floor (see above). Includes `run_count` and `minimum_required`. | Explain the floor; suggest more runs or checking why runs lack the target metric. Never compute your own substitute correlation from raw data to route around this. |
| `partial_data` | A tool must refuse outright due to incomplete data (distinct from the `data_completeness: "partial"` field on `Run`, which downgrades confidence rather than refusing). | Explain that the run/data is still ingesting; suggest retrying later. |
| `unknown` | Anything not covered by the above (includes "sweep ID not found," which has no dedicated error type in the current taxonomy). | Relay the `message` field directly — it's written to be informative. |
