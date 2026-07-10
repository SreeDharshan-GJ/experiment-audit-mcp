# Audit Methods

Full methodology for every `audit_*` tool: exact thresholds, formulas, and
the reasoning behind each. Tool descriptions in the MCP schema itself stay
minimal and link here instead of repeating this content — verbose tool
descriptions cost context budget on every turn of a conversation, not just
when the tool is actually invoked (design-spec-v1.md §6).

This file grows one section per `audit_*` tool as each is implemented.
Currently covers: `audit_training_curve` (Milestone 6), `audit_ablation`
(Milestone 7).

## training-curve

`audit_training_curve(ref, metric)` fetches the full recorded history for
one metric on one run (via `get_metric_history` — always live, never a
pre-fetched blob, per design-spec-v1.md §4.3) and runs four independent
signal detectors over it. Every detector produces a **continuous score**,
not a boolean — the fixed `nan_spike | plateau | reward_collapse |
oscillation` enum from the original design pass was deliberately dropped
because it was under-validated and metric-type-blind (design-spec-v1.md
§4.2). Labeling a curve as e.g. "reward collapse" is left to the calling
agent or human by thresholding these scores.

A signal only appears in the result's `signals` list if it clears its own
reporting condition below. A clean, well-behaved curve produces an **empty**
list, not a list of near-zero-score entries. Signals are **not mutually
exclusive** — more than one can legitimately fire on the same curve (see
the co-firing note under `sudden_jump` below); that is intentional, not a
bug, and consistent with these tools reporting evidence rather than a
single verdict (design-spec-v1.md §4.1).

All four detectors operate only on the metric's own points; none inspect
`metric_type_assumed`, `Run.status`, or `Run.data_completeness` — a
partial-data run's incomplete curve is audited exactly like a complete
one's, since `MetricHistory` carries only points, not a completeness flag
(that flag lives on `Run`, per models.py). Confidence downgrading for
partial-data runs, where required, is the calling tool's responsibility,
not this module's.

### metric_type_assumed

Inferred from the metric name (case-insensitive substring match): contains
`"loss"` → `loss`; contains `"reward"` → `reward`; otherwise → `unknown`.
None of the four detectors below branch on this value — it is reported
alongside the signals as context for the reader, not used to change
detection behavior. A caller who already knows the metric's type can pass
`metric_type` explicitly to override the inference (not currently exposed
as an MCP tool parameter — Milestone 6 exposes only `ref` and `metric`, per
design-spec-v1.md §4.2's exact `audit_training_curve(ref, metric)`
signature; a future milestone could add an optional override if this
inference proves unreliable in practice).

### null_values

**Reported whenever at least one point's `value` is `None`** (a logged
NaN — see models.py's `MetricPoint.value` contract) — regardless of what
fraction of the curve is affected. Presence of a logged NaN is a
deterministic fact, not a judgment call, so `confidence` is always `"high"`
for this signal specifically (the other three detectors compute confidence
from their score; this one does not).

- `score` = (count of `None` points) / (total point count).
- `step_range` = `[min(null step), max(null step)]` across every `None`
  point found.
- `evidence` includes the null step count, total step count, and the
  affected step numbers (capped at the first 50, to keep the payload
  bounded on a very sparse or heavily-NaN curve).

### sudden_jump

Flags an abrupt level shift using a robust (median / MAD) z-score over the
curve's **rate of change** — `(value[i+1] - value[i]) / (step[i+1] -
step[i])` between consecutive *present* (non-`None`) points, not the raw
diff. Dividing by the step gap matters: a `None` point between two present
points widens the effective gap, and a perfectly normal per-step slope
measured across that wider gap produces a larger raw diff without the
curve having actually jumped. Normalizing by the gap removes this
NaN-adjacent artifact.

- Requires at least 3 present points; otherwise no signal (too little data
  for a MAD-based test to mean anything).
- `median_abs_rate` (the MAD) is floored at
  `max(1e-9, 0.01 × scale)`, where `scale` is the curve's overall value
  range (`max(values) - min(values)`, or the max absolute value if the
  range is exactly zero). This floor is scale-relative rather than a bare
  epsilon on purpose: on a near-constant series the raw MAD can be ~0, and
  an absolute-epsilon floor would make the z-score blow up for any
  negligible fluctuation however small relative to the metric's own range
  — exactly the "confidently wrong" false positive this floor exists to
  prevent, not cause.
- `z_score = |rate - median(rates)| / (1.4826 × median_abs_rate)`
  (1.4826 is the standard constant that makes MAD a consistent estimator
  of the standard deviation for a normal distribution). Reported only if
  the **maximum** z-score across all rate-of-change values is `≥ 4.0`.
- `score = min(max_z / 8.0, 1.0)` — so a z-score exactly at the threshold
  (4.0) scores 0.5, and a z-score of 8.0 or higher saturates at 1.0.
- `step_range` and `evidence` identify the single adjacent pair of present
  points where the maximum z-score occurred.

**Known limitation — co-firing with `low_variance_plateau`:** this detector
uses one global median/MAD over the *entire* rate-of-change series, not a
local/windowed baseline. On a curve with genuinely different regimes (a
steady decline, then a flat plateau, then another decline), the tiny
in-plateau fluctuations can look anomalous against the whole curve's much
larger typical rate, so `sudden_jump` can fire at a plateau's edges
alongside the correctly-detected `low_variance_plateau`. This isn't
strictly wrong (a regime change is a real property of the curve) but it is
a real precision limitation, flagged here rather than silently accepted. A
windowed/local baseline would be the natural v2 fix; not implemented in
v1 to avoid scope creep beyond this milestone's roadmap entry.

### low_variance_plateau

Flags the longest contiguous run of present points whose local variability
is small relative to the curve's overall scale — training has stalled,
not merely converged smoothly.

- Uses a sliding window of 5 present points. For each window, computes the
  population standard deviation of the values in that window, divided by
  `scale` (the curve's overall value range, defined identically to
  `sudden_jump`'s `scale` above). A window is "flat" if this coefficient
  of variation is `< 0.02`.
- Contiguous flat windows are merged into runs; the longest run is
  reported (only one plateau signal per curve, even if multiple separate
  flat regions exist — the longest one is the most defensible finding).
- No signal at all if no window is flat, or fewer than 5 present points
  exist.
- `score = min(run_length / 10, 1.0)` — a plateau exactly at the minimum
  window length (5 points) scores 0.5; a plateau of 10 or more points
  saturates at 1.0.
- `step_range` covers the full merged run, not just one window.
- `evidence` includes the window size, run length, the CV threshold used,
  and the mean/stdev of the values within the reported run.

### high_frequency_oscillation

Flags a jagged up/down pattern (as opposed to a directional trend) using
the fraction of adjacent diff-pairs that flip sign.

- Requires at least 6 present points (5 diffs). Diffs of exactly zero are
  excluded from the sign sequence entirely (an exact repeat carries no
  direction to compare against) rather than counted as neither a flip nor
  a non-flip. Requires at least 2 non-zero diffs remaining after that
  filter; otherwise no signal.
- `sign_flip_ratio` = (number of adjacent non-zero-diff pairs with opposite
  sign) / (non-zero diff count − 1). Reported only if this ratio is
  `≥ 0.7`.
- `score` = `sign_flip_ratio` directly — it is already a natural 0–1 ratio,
  so no further normalization is applied.
- `step_range` spans the full present-point range considered (not a
  sub-window), since oscillation is a property of the whole segment
  examined, not a single localized event the way `sudden_jump` is.

### Confidence bucketing (shared across all four detectors)

For the three score-derived detectors (`sudden_jump`, `low_variance_plateau`,
`high_frequency_oscillation`): `score ≥ 0.85` → `"high"`; `score ≥ 0.6` →
`"medium"`; otherwise `"low"`. (`null_values` is always `"high"`, per its
own section above — its detection is a deterministic fact, not a
score-thresholded judgment.)

## ablation

`audit_ablation(baseline, ablation, claimed_variable)` fetches both runs'
full config (via `get_run_summary`) and reuses `compare_runs`'s config
diff directly rather than re-deriving it (design-spec-v1.md §4.3). Every
config parameter that differs between the two runs is classified as
`likely_intentional` — either it *is* `claimed_variable`, or it's on a
fixed allowlist — and the verdict follows mechanically from that
classification. `evidence` in the result is the **full** `compare_runs`
diff (both `config_diff` and `metric_diff`), not just the config
parameters used for the verdict, so a reader can see whether the claimed
variable actually moved the target metric alongside the confound
judgment itself.

### The allowlist

`ALLOWLIST_PARAMS = {"seed", "device", "run_name", "run_id", "name",
"id"}` (`analysis/confound.py`), matched **case-insensitively on the
exact config key** — never by substring. A key like `device_batch_size`
is *not* matched by `"device"`: substring/fuzzy matching would risk
silently exempting a real confound just because its name happens to
contain an allowlisted word, which is exactly the "confidently wrong"
failure mode design principle #3 exists to prevent. The trade-off runs
the other way instead: a project using a differently-spelled field (e.g.
`random_seed` instead of `seed`) will have that field correctly counted
as an *unaccounted* difference rather than silently waved through — a
missed allowlist entry produces a false `confounded`, which a human can
investigate and correct by renaming or (in a future milestone) extending
the allowlist, whereas a false `clean` from fuzzy matching could hide a
real confound entirely. This is a known, deliberate limitation, not an
oversight.

### Verdict

Given the set of differing config parameters between `baseline` and
`ablation`:

- **No parameters differ at all** → `"uncertain"`. This is deliberately
  distinct from `"clean"`: `"clean"` asserts *this is a validated test of
  claimed_variable*, which cannot be asserted if nothing changed —
  including `claimed_variable` itself. An unchanged-config pair is not
  evidence of a clean ablation; it's evidence the two `RunRef`s may not
  represent the ablation the caller thinks they do.
- **Every differing parameter is `claimed_variable` itself or on the
  allowlist** → `"clean"`.
- **At least one differing parameter is neither** → `"confounded"`.

`differing_params` in the result always lists every config parameter
that differs (not just the unaccounted ones), each tagged with its own
`likely_intentional` — spec §4.1's requirement that judgment tools show
their work in full, not a summary.

### Confidence

- **`"low"`**, unconditionally, whenever `baseline.data_completeness ==
  "partial"` or `ablation.data_completeness == "partial"` — the
  automatic downgrade design-spec-v1.md §5 requires for any audit tool
  operating on a partial-data run. The reason is appended to `method`
  (e.g. `"... (confidence downgraded to low: partial data on at least
  one run, per design-spec-v1.md §5)"`) so it's visible without
  inspecting `evidence` separately. This check takes priority over
  everything below — a `"confounded"` verdict on a partial-data run is
  still reported as `"confounded"` (partial data does not change the
  verdict), just at `"low"` confidence, since we may be looking at an
  incomplete config snapshot rather than a real discrepancy.

  Note: `"unknown"` (the default `Run.data_completeness` when a backend
  hasn't determined completeness) does **not** trigger this downgrade —
  only a confirmed `"partial"` state does. Treating `"unknown"` the same
  as `"partial"` would conflate "we don't know" with "we know it's
  incomplete," which is a real distinction worth preserving rather than
  collapsing for a marginally more conservative default.

- **`"low"`** for an `"uncertain"` verdict (independent of the partial-
  data check above): the tool cannot even confirm an ablation of
  `claimed_variable` was performed, so a confident judgment is not
  possible either way.

- **`"high"`** for `"clean"`/`"confounded"` otherwise. Unlike
  `audit_training_curve`'s three heuristic signals, the verdict here is
  a direct, deterministic consequence of the config diff and the fixed
  allowlist — there is no continuous score to threshold into a `"medium"`
  tier.

## sweep

`audit_sweep(sweep_ref, target_metric?)` fetches every run in the sweep
(via `list_sweeps` + `get_run_summary` for each — see `analysis/
sensitivity.py`'s module docstring for why this departs slightly from the
roadmap's literal "calls list_runs(...) scoped to the sweep's run_refs"
data-flow wording, a consequence of the frozen `ExperimentBackend` ABC
only exposing `list_sweeps`, not any sweep-scoped `list_runs` variant),
then ranks each numeric config parameter by its Pearson correlation with
`target_metric` (defaulting to the sweep's own recorded `target_metric`
if the tool call omits one).

**Read this section before trusting a low-ranked (near-zero correlation)
parameter as "doesn't matter":** Pearson correlation only detects *linear*
relationships. A hyperparameter with a non-monotonic effect — e.g. a
learning rate with an interior optimum, where both too low and too high
hurt — can show a near-zero correlation despite being the most important
parameter in the sweep. This is a known, documented limitation of the
method the frozen spec names explicitly (design-spec-v1.md §4.2's `method`
string), not an implementation gap; a rank-based measure (Spearman) would
handle monotonic-but-nonlinear relationships better but was not
substituted in, since doing so silently would be exactly the kind of
undocumented spec deviation the frozen-spec process exists to prevent.
`caveat` (every result) restates this.

### The hard sample-size floor, applied twice

`DEFAULT_MINIMUM_SAMPLES = 10` (design-spec-v1.md §4.2: "default floor: 10
runs; configurable but defaults conservative"), enforced **before any
ranking logic runs**, in two places:

1. Against the sweep's raw run count (`len(sweep.run_refs)`) — spec §7's
   named case: a 3-run sweep must return `insufficient_samples`, never a
   ranking.
2. Against the *usable* run count — runs that actually logged a numeric
   value for `target_metric`. A nominally large sweep where most runs
   crashed before logging the target metric would otherwise recreate the
   exact shaky-number problem the floor exists to prevent, just hidden
   behind a large `sweep_size`.

Both refusals raise `InsufficientSamplesError` (`run_count`,
`minimum_required`, `reason`), translated at the MCP layer into spec
§4.2's literal `{error: "insufficient_samples", run_count,
minimum_required}` shape, layered onto this codebase's standard
`ToolError` dict rather than a one-off shape just for this tool.

### Which parameters get ranked

A config parameter is **excluded** from `parameter_importance` (and
reported instead, with a reason, in `excluded_parameters`) if any of the
following hold, checked in this order:

- **`non_numeric`** — the parameter's value is present but never numeric
  (`int`/`float`/`bool`) on any usable run, e.g. an optimizer name. Pearson
  correlation is undefined for non-numeric data; label-encoding a category
  would produce a correlation that depends on arbitrary encoding order, not
  on any real relationship — exactly the "confidently wrong" failure mode
  design principle #3 exists to prevent. `bool` **is** treated as numeric
  (0/1): correlating a binary hyperparameter against a continuous target
  is Pearson's standard point-biserial special case, not an encoding
  artifact.
- **`insufficient_overlap`** — the parameter is numeric but present
  (and numeric) on fewer than `minimum_samples` usable runs (e.g. a
  conditional hyperparameter only set for a few configurations). The same
  sample-size floor rationale applies per-parameter, not just at the
  sweep level.
- **`constant`** — the parameter is numeric and present on enough runs,
  but never varies among them (zero variance makes Pearson's denominator
  zero).
- **`target_metric_constant`** — `target_metric` itself never varies
  across usable runs. In this case *every* parameter is excluded with
  this reason, regardless of how the parameters themselves vary — no
  correlation is computable against a constant target, no matter how
  informative a parameter might otherwise be. This is **not** remapped
  onto `insufficient_samples`: that error is specifically about sample
  *count*, and conflating it with "no variance in the target metric"
  would blur what a caller should actually do about it (get more runs vs.
  check whether the sweep's target metric was configured correctly).
  `audit_sweep` returns a normal (non-error) result in this case — empty
  `parameter_importance`, `\"low\"` confidence, and a `caveat` explaining
  why — an honestly-empty ranking rather than a fabricated one (design
  principle #3).

### Co-variance warning

Despite design-spec-v1.md §4.2's "co-variance" wording, this is
implemented as **correlation**, not raw covariance: covariance is not
scale-invariant (its magnitude depends on the units of both variables, so
a literal covariance threshold would flag or miss pairs depending on
arbitrary unit choices like learning_rate in `1e-3` units vs. batch_size
in the hundreds). The spec's own worked formula — "co-variance flagged
where `|corr(param_i, param_j)| ...`" — already uses `corr`, confirming
this reading.

For every pair of *included* (ranked) numeric parameters, restricted to
the runs where both have a numeric value, `COVARIANCE_WARNING_THRESHOLD =
0.7` flags the pair if `|corr(param_i, param_j)| >= 0.7`. Both parameters'
`ParameterImportance.warning` name the other (and its `|r|`) — spec §7's
named case (two correlated hyperparameters in a grid, e.g. learning_rate
vs. batch_size) is exercised directly in `tests/test_sensitivity.py`.
0.7 is this module's own calibrated choice (the frozen spec names no exact
number), matching the same "strong, hard to ignore" register
`divergence.py`'s own sign-flip-ratio threshold uses.

### Confidence: a real significance test, not a bare magnitude cutoff

Design-spec-v1.md §4.2 requires confidence to come from "sweep_size and
correlation strength together." Rather than picking two independent
thresholds (one on `n`, one on `|r|`) and combining them ad hoc, this uses
the standard **Fisher z-transformation significance test** for a Pearson
correlation coefficient:

```
z = atanh(r)                     # Fisher z-transform
SE = 1 / sqrt(n - 3)              # standard error of z
z_statistic = z / SE
p = 2 * (1 - Φ(|z_statistic|))    # two-tailed p-value, Φ = standard normal CDF
```

computed with only the standard library (`math.atanh`, `math.erf` for
`Φ`) — no `scipy` dependency needed. `SE` shrinks as `n` grows, so the
same `r` yields a smaller `p` at a larger sweep: sample size and
correlation strength are already combined into one statistic, rather than
gated separately. The overall `confidence` bucket is derived from the
**top-ranked** (largest `|r|`) parameter's p-value:

- **`"high"`** — `p < 0.01`
- **`"medium"`** — `0.01 ≤ p < 0.05`
- **`"low"`** — `p ≥ 0.05`, or no p-value could be computed (`n < 4` for
  a given parameter's own overlap, or `parameter_importance` is empty)

Every `ParameterImportance` entry also carries its own `p_value` — an
additive field beyond spec §4.2's minimum output shape, in the same
"show your work" spirit as `audit_ablation` reporting the full
`compare_runs` diff rather than only the parameters that mattered for its
verdict.

### Multiple comparisons (no correction applied)

Ranking many parameters by raw correlation from a modest sample is
exploratory: with enough parameters and a sweep just above the sample
floor, *some* parameter will show a moderately large correlation by
chance alone. No Bonferroni/FDR correction is applied — the frozen spec's
`method` string names none, and picking a per-parameter significance
threshold that changes with the number of parameters tested would break
the independence of ranking one sweep at a time. `caveat` reports the
number of ranked parameters and the sample size so a reader has what they
need to judge this themselves.
