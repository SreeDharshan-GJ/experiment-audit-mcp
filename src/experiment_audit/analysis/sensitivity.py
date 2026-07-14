"""Sweep parameter-importance logic for audit_sweep (Milestone 8).

Implements design-spec-v1.md §4.2 (`audit_sweep`) and §4.3's data-flow rule
("calls list_runs(...) scoped to the sweep's run_refs, then get_run_summary
for each" -- see the "Flag" note in `server.py`'s `audit_sweep` tool for the
one place this milestone's actual implementation necessarily departs from
that literal wording, since the frozen `ExperimentBackend` ABC (§3) only
exposes `list_sweeps(project) -> list[Sweep]`, not any sweep-scoped variant
of `list_runs`).

**Architectural constraint, validated before this milestone began** (see the
Milestone 8 review delivered alongside this file): this module has no
dependency on FastMCP, MCP transport, `server.py`, or `WandbBackend`. It
operates only on the normalized `Sweep`/`Run` models (models.py, Milestone 1)
and returns plain frozen dataclasses with their own `to_dict()` -- exactly
the pattern `analysis/comparison.py`, `analysis/divergence.py`, and
`analysis/confound.py` already established. The MCP tool in `server.py` is a
thin wrapper: it resolves `sweep_ref` into a `Sweep` via `list_sweeps` and
each of that sweep's `run_refs` into `Run`s via `get_run_summary`, then
passes them here -- this module never fetches data itself.

**Statistical-validity review (performed before implementation, per explicit
instruction not to silently weaken rigor just to always return an answer):**

1. **Pearson correlation is linear-only.** The frozen spec names Pearson
   correlation explicitly (§4.2's `method` string is given verbatim:
   "pairwise Pearson correlation with target_metric..."), so this module
   implements Pearson exactly as specified rather than silently swapping in
   a rank-based measure (e.g. Spearman) that would handle monotonic-but-
   nonlinear relationships better. This is a real, known blind spot worth
   surfacing rather than quietly working around: a hyperparameter with a
   non-monotonic effect on the target metric (e.g. a learning rate with an
   interior optimum -- too low *and* too high both hurt) can produce a
   near-zero Pearson correlation despite being the most important parameter
   in the sweep, and a reader trusting a low-ranked `correlation` value
   here could wrongly conclude that parameter doesn't matter. This is
   flagged explicitly in `caveat` (every result) and in
   docs/audit-methods.md#sweep, not silently accepted -- it is a
   documented methodological limitation of the frozen spec's chosen method,
   not an implementation bug, and a candidate for a v2 look (the roadmap's
   own "Flag-if-triggered" pattern applies to exactly this kind of finding).
2. **Multiple comparisons.** Ranking many parameters by raw correlation
   from a modest sample is an exploratory procedure with no correction
   applied (no Bonferroni/FDR adjustment) -- with enough parameters and a
   sweep just above the sample floor, *some* parameter will show a
   moderately large correlation by chance alone. `caveat` reports the
   number of parameters ranked and the sample size so a reader has what
   they need to judge this themselves; this module does not attempt to
   auto-correct multiple comparisons, since doing so silently would itself
   be a design decision the frozen spec doesn't make (§4.2 lists no
   correction), and picking a per-parameter significance schema that
   changes with the number of parameters tested would break the
   independence of ranking one sweep at a time.
3. **Non-numeric (categorical) hyperparameters.** Not addressed by §4.2 at
   all -- real sweeps commonly vary a categorical parameter (e.g. optimizer
   name). Pearson correlation is undefined for non-numeric data, and
   silently label-encoding a category (e.g. "adam" -> 0, "sgd" -> 1) would
   produce a correlation that depends entirely on arbitrary encoding order,
   not on any real relationship -- exactly the "confidently wrong" failure
   mode design principle #3 exists to prevent. **Resolution:** non-numeric
   parameters (and parameters that never vary, or that vary but overlap
   with too few runs with a numeric target-metric value) are excluded from
   `parameter_importance` and reported instead, with reasons, in
   `excluded_parameters` -- an additive field beyond §4.2's minimum output
   shape, in the same spirit as `audit_ablation` reporting the *full*
   compare_runs diff rather than only the parameters that mattered for its
   verdict (design principle #2: judgment tools show their work).
   `bool` is treated as numeric (0/1): correlating a binary hyperparameter
   against a continuous target metric is exactly Pearson's well-established
   point-biserial special case, not an encoding artifact.
4. **"Co-variance" (§4.2's wording) is implemented as correlation.** Raw
   covariance is not scale-invariant (its magnitude depends on the units of
   both variables), so a literal covariance-based threshold would flag or
   miss pairs depending on arbitrary unit choices (e.g. learning_rate in
   1e-3 units vs. batch_size in the hundreds). The spec's own worked
   example formula -- "co-variance flagged where |corr(param_i,
   param_j)| ..." -- already uses `corr`, confirming this reading; this
   module computes pairwise Pearson correlation between included numeric
   parameters and flags pairs at or above `COVARIANCE_WARNING_THRESHOLD`.
5. **The hard sample-size floor is applied twice, not once**, per this
   review: once against the sweep's raw run count (spec §4.2's literal
   "run_count" case: `sweep_too_small` from spec §7), and again against the
   *usable* run count -- runs that actually logged a numeric value for
   `target_metric` -- since correlating against fewer effective data points
   than the floor recreates exactly the shaky-number problem the floor
   exists to prevent, even when the nominal sweep is large enough. Both
   refusals share the same `InsufficientSamplesError` shape (`run_count` /
   `minimum_required`), consistent with spec §4.2's literal
   `{error: "insufficient_samples", run_count, minimum_required}` shape,
   just with a different `run_count` and an explanatory `reason`.
6. **Confidence is derived from a real significance test, not a bare
   magnitude cutoff.** §4.2 requires confidence to come from "sweep_size
   and correlation strength together." Rather than picking two arbitrary
   thresholds and combining them ad hoc (the way `divergence.py`'s three
   heuristic detectors reasonably do, since those have no analogous closed-
   form test available), this module uses the standard Fisher z-transform
   significance test for a Pearson correlation coefficient
   (`z = atanh(r)`, `SE = 1/sqrt(n-3)`, two-tailed p-value from the
   standard normal CDF via `math.erf` -- no `scipy` dependency needed).
   This is a well-established, textbook technique that already combines
   sample size and correlation strength into a single, principled
   statistic (`SE` shrinks as `n` grows, so the same `r` yields a smaller
   p-value at a larger sweep), rather than gating on `n` and `|r|`
   separately with two independently-chosen cutoffs.
7. **What happens if nothing is rankable at all** (every parameter
   excluded -- e.g. the target metric itself never varies across usable
   runs, or every parameter is non-numeric/constant): this is not remapped
   onto `insufficient_samples` (that error is specifically about sample
   *count*, not about variance in the parameters or the target metric --
   conflating the two would blur what a caller should actually do about it:
   get more runs vs. check the sweep config). Instead, `audit_sweep`
   returns a normal (non-error) result with an empty `parameter_importance`,
   `low` confidence, and a `caveat` explaining why -- an honestly-empty
   ranking, not a fabricated one, per design principle #3 ("refuse rather
   than mislead").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import correlation
from typing import Any, Literal

from experiment_audit.models import Run, RunRef, Sweep

Confidence = Literal["high", "medium", "low"]
ExclusionReason = Literal[
    "non_numeric", "constant", "insufficient_overlap", "target_metric_constant"
]

SCHEMA_VERSION = 1

#: Spec §4.2: "default floor: 10 runs; configurable but defaults
#: conservative." Kept as a module constant + optional override parameter
#: (not exposed as an MCP tool parameter -- §4.2's exact tool signature is
#: `audit_sweep(sweep_ref, target_metric?)`, mirroring how
#: `audit_training_curve`'s `metric_type` override exists at the pure-
#: function layer but isn't an MCP-visible parameter either, per
#: docs/audit-methods.md#training-curve).
DEFAULT_MINIMUM_SAMPLES = 10

#: Threshold for flagging two hyperparameters as co-varying (§4.2's
#: "co-variance flagged where |corr(param_i, param_j)| ..."). 0.7 matches
#: the same "strong, hard to ignore" register `divergence.py` uses for its
#: own sign-flip-ratio threshold -- not derived from the frozen spec (which
#: names no exact number), so documented here and in
#: docs/audit-methods.md#sweep as this module's own calibrated choice,
#: same as every other detector threshold in this codebase.
COVARIANCE_WARNING_THRESHOLD = 0.7

#: Two-tailed significance thresholds for the Fisher z-transform test
#: (see point 6 in the module docstring's statistical-validity review).
_P_VALUE_HIGH_CONFIDENCE = 0.01
_P_VALUE_MEDIUM_CONFIDENCE = 0.05

_METHOD = (
    "pairwise Pearson correlation with target_metric; co-variance flagged "
    f"where |corr(param_i, param_j)| >= {COVARIANCE_WARNING_THRESHOLD}; "
    "confidence derived from a Fisher z-transform significance test on the "
    "top-ranked parameter's correlation; see docs/audit-methods.md#sweep"
)


class SweepAuditError(ValueError):
    """Structurally invalid input to `audit_sweep` -- mirrors
    `CompareRunsError`'s convention (analysis/comparison.py). Currently
    raised only when no `target_metric` is resolvable from either the
    tool call or the `Sweep`'s own `target_metric` field. Deliberately
    *not* the same exception as `InsufficientSamplesError` below: this is
    an input-contract violation (the caller didn't give this function
    enough information to even attempt an audit), not the sample-size
    refusal spec §4.2 explicitly designs for and expects as a normal
    outcome.
    """


class InsufficientSamplesError(Exception):
    """Raised when the sweep does not clear the hard sample-size floor --
    spec §4.2's `{error: "insufficient_samples", run_count,
    minimum_required}`. Raised (not returned as a `SweepAudit`) so this
    refusal is enforced structurally, before any ranking logic runs, and
    is un-bypassable through any parameter combination (roadmap's
    Milestone 8 completion criteria) -- there is no code path in
    `audit_sweep` below that computes a correlation before this check has
    passed.
    """

    def __init__(self, run_count: int, minimum_required: int, reason: str) -> None:
        self.run_count = run_count
        self.minimum_required = minimum_required
        self.reason = reason
        super().__init__(
            f"Sweep has only {run_count} usable run(s) ({reason}); "
            f"audit_sweep requires at least {minimum_required} to compute "
            f"a statistically meaningful importance ranking."
        )


@dataclass(frozen=True)
class ParameterImportance:
    """One ranked parameter -- spec §4.2's `{param, correlation, rank,
    warning?}` entry shape, plus `p_value` (additive: the significance
    test in point 6 above already computes it per-parameter to derive the
    overall `confidence`, and surfacing it here lets a reader see *why*
    a given rank got the confidence bucket it did, per design principle
    #2, "judgment tools show their work")."""

    param: str
    correlation: float
    rank: int
    p_value: float | None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "param": self.param,
            "correlation": self.correlation,
            "rank": self.rank,
            "p_value": self.p_value,
        }
        if self.warning is not None:
            result["warning"] = self.warning
        return result


@dataclass(frozen=True)
class ExcludedParameter:
    """A config parameter deliberately left out of `parameter_importance`,
    with why -- reported explicitly rather than silently dropped (point 3
    in the module docstring's statistical-validity review)."""

    param: str
    reason: ExclusionReason

    def to_dict(self) -> dict[str, Any]:
        return {"param": self.param, "reason": self.reason}


@dataclass(frozen=True)
class SweepAudit:
    """Full result of `audit_sweep` -- spec §4.2's `{sweep_size,
    parameter_importance, caveat, confidence, method}` shape, plus two
    additive fields (`usable_run_count`, `excluded_parameters`) that make
    the ranking's own limitations visible rather than requiring a reader
    to re-derive them (design principle #2)."""

    schema_version: int
    sweep_size: int
    usable_run_count: int
    target_metric: str
    parameter_importance: list[ParameterImportance]
    excluded_parameters: list[ExcludedParameter]
    caveat: str
    confidence: Confidence
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sweep_size": self.sweep_size,
            "usable_run_count": self.usable_run_count,
            "target_metric": self.target_metric,
            "parameter_importance": [p.to_dict() for p in self.parameter_importance],
            "excluded_parameters": [e.to_dict() for e in self.excluded_parameters],
            "caveat": self.caveat,
            "confidence": self.confidence,
            "method": self.method,
        }


def _is_numeric(value: Any) -> bool:
    """True for `int`, `float`, and `bool` (see point 3 above); false for
    everything else, including `None`. Checked against `bool` explicitly
    even though `bool` is already an `int` subclass, so the reasoning is
    legible at the call site rather than relying on that subtyping quirk
    implicitly."""
    return isinstance(value, bool) or isinstance(value, (int, float))


def _two_tailed_p_value(r: float, n: int) -> float | None:
    """Fisher z-transform significance test for a Pearson correlation
    coefficient `r` computed from `n` paired observations. Returns `None`
    if `n < 4` (the transform's standard error, `1/sqrt(n-3)`, is
    undefined/non-positive below that) -- callers treat `None` as "cannot
    establish significance," bucketed as low confidence, not as an error.

    `r` is clamped away from exactly +/-1 before the `atanh` transform,
    which is otherwise infinite at a perfect correlation -- a perfect
    correlation from a small, finite sample is often a sign of a
    degenerate fit (e.g. only two distinct x-values), not literally
    infinite certainty, so clamping avoids reporting a nonsensical p-value
    of exactly zero from floating-point overflow.
    """
    if n < 4:
        return None
    r_clamped = max(min(r, 1 - 1e-10), -1 + 1e-10)
    z = math.atanh(r_clamped)
    standard_error = 1 / math.sqrt(n - 3)
    z_statistic = z / standard_error
    p_value = 2 * (1 - _standard_normal_cdf(abs(z_statistic)))
    return max(0.0, min(1.0, p_value))


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _confidence_from_p_value(p_value: float | None) -> Confidence:
    if p_value is None:
        return "low"
    if p_value < _P_VALUE_HIGH_CONFIDENCE:
        return "high"
    if p_value < _P_VALUE_MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def audit_sweep(
    sweep: Sweep,
    runs: list[Run],
    target_metric: str | None = None,
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
) -> SweepAudit:
    """Rank hyperparameter importance across a sweep.

    Pure and deterministic: no I/O, no dependency on how `sweep`/`runs`
    were fetched. The MCP tool (`server.py`) is responsible for calling
    `ExperimentBackend.list_sweeps` and `get_run_summary` and passing the
    results here -- this function never fetches data itself, mirroring
    every other audit tool's data-flow discipline (spec §4.3).

    Args:
        sweep: The sweep being audited. Only `run_refs` and (as a
            fallback) `target_metric` are used from it.
        runs: `Run` objects already fetched for (at least) every ref in
            `sweep.run_refs`. Extra runs not in `sweep.run_refs` are
            ignored; a ref in `sweep.run_refs` missing from `runs` is
            silently excluded from `sweep_size` accounting here -- *not*
            raised on -- since deciding whether a missing run is itself
            an error (vs. an expected partial fetch) is the caller's
            concern, not this pure function's.
        target_metric: Overrides `sweep.target_metric` when the caller
            already knows which metric to rank against. `None` (the
            default) falls back to the sweep's own recorded
            `target_metric`.
        minimum_samples: The hard sample-size floor (spec §4.2: "default
            floor: 10 runs; configurable"). Applied to both the raw
            sweep run count and the usable (numeric-target-metric) run
            count -- see point 5 in the module docstring.

    Returns:
        A `SweepAudit` at `schema_version: 1`.

    Raises:
        SweepAuditError: if no `target_metric` is resolvable at all.
        InsufficientSamplesError: if either sample-size floor isn't
            cleared. Enforced before any correlation is computed.
    """
    resolved_metric = target_metric if target_metric is not None else sweep.target_metric
    if not resolved_metric:
        raise SweepAuditError(
            "audit_sweep requires a target_metric: none was passed to the "
            "tool call, and the Sweep record itself has no target_metric "
            "set."
        )

    sweep_size = len(sweep.run_refs)
    if sweep_size < minimum_samples:
        raise InsufficientSamplesError(sweep_size, minimum_samples, "raw sweep run count")

    runs_by_ref = {r.ref: r for r in runs}
    sweep_runs = [runs_by_ref[ref] for ref in sweep.run_refs if ref in runs_by_ref]

    usable_runs = [r for r in sweep_runs if _is_numeric(r.summary_metrics.get(resolved_metric))]
    usable_run_count = len(usable_runs)
    if usable_run_count < minimum_samples:
        raise InsufficientSamplesError(
            usable_run_count,
            minimum_samples,
            f"only {usable_run_count} of {sweep_size} sweep run(s) logged a "
            f"numeric {resolved_metric!r} summary metric",
        )

    # Keyed by RunRef (frozen/hashable, models.py), not by Run itself: Run
    # is a plain (non-frozen) dataclass and therefore not hashable, so it
    # can't be used as a dict key -- RunRef is exactly the identity every
    # other module in this codebase already uses for this purpose (see
    # analysis/comparison.py's `_runref_key`).
    y_by_ref: dict[RunRef, float] = {
        r.ref: float(r.summary_metrics[resolved_metric]) for r in usable_runs
    }
    target_metric_varies = len(set(y_by_ref.values())) > 1

    all_params = sorted({key for r in usable_runs for key in r.config})

    parameter_importance: list[ParameterImportance] = []
    excluded: list[ExcludedParameter] = []

    if not target_metric_varies:
        # Every usable run logged the exact same target_metric value: no
        # correlation is computable for *any* parameter (a zero-variance
        # y makes Pearson's denominator zero for every x), regardless of
        # how the parameters themselves vary. Point 7 above: this is not
        # remapped onto insufficient_samples.
        excluded = [ExcludedParameter(p, "target_metric_constant") for p in all_params]
    else:
        included_series: dict[str, dict[RunRef, float]] = {}
        for param in all_params:
            numeric_series = {
                r.ref: float(r.config[param])
                for r in usable_runs
                if param in r.config and _is_numeric(r.config[param])
            }
            if len(numeric_series) < minimum_samples:
                has_nonnumeric_occurrence = any(
                    param in r.config and not _is_numeric(r.config[param]) for r in usable_runs
                )
                reason: ExclusionReason = (
                    "non_numeric"
                    if has_nonnumeric_occurrence and not numeric_series
                    else "insufficient_overlap"
                )
                excluded.append(ExcludedParameter(param, reason))
                continue
            if len(set(numeric_series.values())) <= 1:
                excluded.append(ExcludedParameter(param, "constant"))
                continue
            included_series[param] = numeric_series

        scored: list[tuple[str, float, float | None]] = []
        for param, series in included_series.items():
            paired_refs = list(series.keys())
            xs = [series[ref] for ref in paired_refs]
            ys = [y_by_ref[ref] for ref in paired_refs]
            r_value = correlation(xs, ys)
            p_value = _two_tailed_p_value(r_value, len(paired_refs))
            scored.append((param, r_value, p_value))

        scored.sort(key=lambda item: (-abs(item[1]), item[0]))
        warnings = _covariance_warnings(included_series)

        rank = 0
        previous_abs_r: float | None = None
        for position, (param, r_value, p_value) in enumerate(scored, start=1):
            if previous_abs_r is None or abs(r_value) != previous_abs_r:
                rank = position
            previous_abs_r = abs(r_value)
            parameter_importance.append(
                ParameterImportance(
                    param=param,
                    correlation=r_value,
                    rank=rank,
                    p_value=p_value,
                    warning=warnings.get(param),
                )
            )

    if parameter_importance:
        top_p_value = min(
            (p.p_value for p in parameter_importance if p.p_value is not None), default=None
        )
        confidence = _confidence_from_p_value(top_p_value)
    else:
        confidence = "low"

    caveat = (
        "Correlation-based (Pearson): detects only linear relationships "
        "and can miss non-monotonic or threshold effects (e.g. an "
        "interior optimum for a learning rate). Unreliable with "
        "correlated hyperparameters or small sweeps; exploratory across "
        f"{len(parameter_importance)} ranked parameter(s) with no "
        f"multiple-comparison correction applied. "
        f"n={usable_run_count} of {sweep_size} sweep run(s) used "
        f"(target_metric={resolved_metric!r})."
    )
    if not target_metric_varies:
        caveat += (
            f" No parameters could be ranked: {resolved_metric!r} did not "
            "vary across the usable runs, so no correlation is computable."
        )
    elif not parameter_importance:
        caveat += (
            " No parameters could be ranked: every config parameter was "
            "either non-numeric, constant, or present on too few usable "
            "runs -- see excluded_parameters."
        )

    return SweepAudit(
        schema_version=SCHEMA_VERSION,
        sweep_size=sweep_size,
        usable_run_count=usable_run_count,
        target_metric=resolved_metric,
        parameter_importance=parameter_importance,
        excluded_parameters=excluded,
        caveat=caveat,
        confidence=confidence,
        method=_METHOD,
    )


def _covariance_warnings(included_series: dict[str, dict[RunRef, float]]) -> dict[str, str]:
    """Pairwise Pearson correlation between every pair of included numeric
    parameters, restricted (per pair) to the runs where *both* parameters
    have a numeric value -- these overlap sets can differ pair-to-pair
    when a parameter is conditionally present, so each pair's correlation
    is computed on its own aligned subset, not on the union of both
    parameters' individual `included_series` entries.

    Returns a `{param: warning_text}` map covering only parameters that
    co-vary with at least one other included parameter at or above
    `COVARIANCE_WARNING_THRESHOLD`; a parameter with no such pairing is
    simply absent from the returned dict (not present with an empty
    string), matching every other module's convention of a `None`/absent
    optional field over an empty placeholder value.
    """
    params = sorted(included_series)
    hits: dict[str, list[tuple[str, float]]] = {p: [] for p in params}
    for i, param_a in enumerate(params):
        for param_b in params[i + 1 :]:
            shared_runs = set(included_series[param_a]) & set(included_series[param_b])
            if len(shared_runs) < 3:
                # Fewer than 3 points can't produce a meaningful
                # correlation at all (variance-based statistics need at
                # least this many); silently skip this pair rather than
                # raise -- this is a secondary diagnostic, not the
                # headline ranking, so absence of a warning here just
                # means "not enough overlap to check," not "confirmed
                # independent."
                continue
            ordered_runs = list(shared_runs)
            xs = [included_series[param_a][r] for r in ordered_runs]
            ys = [included_series[param_b][r] for r in ordered_runs]
            if len(set(xs)) <= 1 or len(set(ys)) <= 1:
                continue
            pair_r = correlation(xs, ys)
            if abs(pair_r) >= COVARIANCE_WARNING_THRESHOLD:
                hits[param_a].append((param_b, pair_r))
                hits[param_b].append((param_a, pair_r))

    warnings: dict[str, str] = {}
    for param, co_varying in hits.items():
        if not co_varying:
            continue
        co_varying.sort(key=lambda item: (-abs(item[1]), item[0]))
        parts = ", ".join(f"{other} (|r|={abs(r):.2f})" for other, r in co_varying)
        warnings[param] = f"co-varies with {parts}"
    return warnings
