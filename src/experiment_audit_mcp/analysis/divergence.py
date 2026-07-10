"""Training-curve signal detectors for audit_training_curve (Milestone 6).

Implements design-spec-v1.md §4.2 (audit_training_curve) and §4.3's data-flow
rule. **Architectural constraint, validated before this milestone began**
(see the Milestone 6 review delivered alongside this file): this module has
no dependency on FastMCP, MCP transport, `server.py`, or `WandbBackend`. It
operates only on the normalized `MetricHistory`/`MetricPoint` models
(models.py, Milestone 1) and returns plain frozen dataclasses with their own
`to_dict()`, exactly the pattern `analysis/comparison.py` established in
Milestone 5. The MCP tool in `server.py` is a thin wrapper: it fetches a
`MetricHistory` via `ExperimentBackend.get_metric_history` and passes it to
`audit_training_curve` here — this module never fetches data itself and
never imports anything backend- or transport-shaped.

Signals are continuous scores, not a fixed label taxonomy (spec: "resolving
the review concern that a locked enum ... would be under-validated and
metric-type-blind"). Labeling ("this is reward collapse") is left to the
calling agent/human by thresholding scores; this module only detects and
scores four signal *shapes* (§4.2): `null_values`, `sudden_jump`,
`low_variance_plateau`, `high_frequency_oscillation`.

Exact thresholds and formulas are documented in
`docs/audit-methods.md#training-curve` and referenced (not repeated) in the
`method` field of every result and in the MCP tool's schema description,
per spec §6 ("verbose tool descriptions cost context budget on every turn").
The constants below are the single source of truth; docs/audit-methods.md
restates them for human readers and must be kept in sync by hand if they
change here — there is no generation step tying the two together.

A signal is only included in a result's `signals` list if it clears its own
reporting condition (see each detector). A clean, well-behaved curve should
produce an **empty** `signals` list — not a list of near-zero-score entries
— per the roadmap's Milestone 6 completion criteria ("a clean curve must
produce empty/low-score signals, not false positives").

**Known limitation, flagged rather than silently accepted:** `sudden_jump`
computes one global median/MAD over the whole rate-of-change series, not a
local/windowed baseline. On a curve with genuinely different regimes (e.g.
a steady decline, a flat plateau, then another decline), the transition
into/out of the flat region can register as a `sudden_jump` alongside the
correctly-detected `low_variance_plateau`, because the tiny in-plateau
fluctuations look anomalous against the whole curve's much larger typical
rate. This is not incorrect exactly (a regime change is a real property of
the curve) but it means `sudden_jump` and `low_variance_plateau` can and do
co-fire at a plateau's edges — this is intentional non-exclusivity (spec:
audit tools report continuous, possibly-overlapping signals, not a single
verdict), not a bug, but it is a real precision limitation worth a v2 look
(a windowed/local baseline would be the natural fix) rather than something
to quietly special-case here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any, Literal

from experiment_audit_mcp.models import MetricHistory, MetricPoint

SignalName = Literal[
    "null_values", "sudden_jump", "low_variance_plateau", "high_frequency_oscillation"
]
Confidence = Literal["high", "medium", "low"]
MetricType = Literal["loss", "reward", "unknown"]

SCHEMA_VERSION = 2
METHOD = "threshold-based, see docs/audit-methods.md#training-curve"

# -- Thresholds -- kept in sync by hand with docs/audit-methods.md ---------

_MAX_EVIDENCE_STEPS = 50

_SUDDEN_JUMP_MIN_PRESENT_POINTS = 3
_SUDDEN_JUMP_Z_THRESHOLD = 4.0
_SUDDEN_JUMP_MIN_MAD = 1e-9
_SUDDEN_JUMP_MAD_FLOOR_FRACTION = 0.01

_PLATEAU_WINDOW = 5
_PLATEAU_CV_THRESHOLD = 0.02

_OSCILLATION_MIN_PRESENT_POINTS = 6
_OSCILLATION_SIGN_FLIP_RATIO_THRESHOLD = 0.7

_CONFIDENCE_HIGH_MIN_SCORE = 0.85
_CONFIDENCE_MEDIUM_MIN_SCORE = 0.6


@dataclass(frozen=True)
class TrainingCurveSignal:
    """One detected signal — spec §4.2's per-signal shape.

    Every audit_* result requires (not optionally includes) confidence and
    evidence alongside the signal name and score (spec §4.1) — enforced
    here by these being non-optional fields, not by convention alone.
    """

    signal: SignalName
    score: float
    step_range: tuple[int, int]
    evidence: dict[str, Any]
    confidence: Confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "score": self.score,
            "step_range": list(self.step_range),
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TrainingCurveAudit:
    """Full result of `audit_training_curve` — spec §4.2's exact shape:
    `{schema_version, metric_type_assumed, signals, method}`."""

    schema_version: int
    metric_type_assumed: MetricType
    signals: list[TrainingCurveSignal]
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_type_assumed": self.metric_type_assumed,
            "signals": [s.to_dict() for s in self.signals],
            "method": self.method,
        }


def infer_metric_type(metric_name: str) -> MetricType:
    """Infer `metric_type_assumed` from the metric's name (spec §4.2:
    "inferred from metric name / declared by caller"). A simple substring
    match — "loss" or "reward" appearing anywhere in the (lowercased) name
    — since v1 has no metric-type registry and guessing wrong here only
    affects a label attached to the result, not the detectors themselves
    (none of the four signal detectors branch on metric_type_assumed)."""
    name = metric_name.lower()
    if "loss" in name:
        return "loss"
    if "reward" in name:
        return "reward"
    return "unknown"


def _confidence_from_score(score: float) -> Confidence:
    if score >= _CONFIDENCE_HIGH_MIN_SCORE:
        return "high"
    if score >= _CONFIDENCE_MEDIUM_MIN_SCORE:
        return "medium"
    return "low"


def _present(points: Sequence[MetricPoint]) -> list[MetricPoint]:
    return [p for p in points if p.value is not None]


def _detect_null_values(points: Sequence[MetricPoint]) -> TrainingCurveSignal | None:
    """A logged NaN/null must never be silently dropped (spec §2, §7's
    "a real crashed run with logged NaN values mid-curve" adversarial
    case) — reported whenever at least one `MetricPoint.value is None`
    exists, regardless of how small a fraction of the curve it is:
    presence is the deterministic fact here, so `confidence` is always
    `"high"`, unlike the other three (heuristic) detectors below."""
    null_steps = [p.step for p in points if p.value is None]
    if not points or not null_steps:
        return None
    score = len(null_steps) / len(points)
    return TrainingCurveSignal(
        signal="null_values",
        score=score,
        step_range=(min(null_steps), max(null_steps)),
        evidence={
            "null_step_count": len(null_steps),
            "total_step_count": len(points),
            "null_steps": null_steps[:_MAX_EVIDENCE_STEPS],
        },
        confidence="high",
    )


def _detect_sudden_jump(points: Sequence[MetricPoint]) -> TrainingCurveSignal | None:
    """Flags an abrupt level shift using a robust (median/MAD) z-score
    over the rate of change between consecutive *present* points.

    Rate-of-change (`diff / step_gap`), not raw diff: a NaN/missing point
    between two present points widens the effective step gap, and a
    normal per-step slope over a wider gap produces a larger raw diff
    without the curve having actually jumped. Normalizing by the step gap
    avoids treating that as a false `sudden_jump`.

    The MAD floor is scale-relative (`_SUDDEN_JUMP_MAD_FLOOR_FRACTION` of
    the curve's overall value range), not a bare epsilon: on a near-
    constant series the raw MAD can be ~0, and an absolute-epsilon floor
    would make the z-score blow up for any negligible fluctuation however
    small relative to the metric's own range — exactly the "confidently
    wrong" false-positive failure mode this floor exists to prevent, not
    cause. See the module docstring's "known limitation" note for the
    one case (a plateau's edges) where this can still co-fire correctly-
    but-aggressively alongside `low_variance_plateau`.
    """
    present = _present(points)
    if len(present) < _SUDDEN_JUMP_MIN_PRESENT_POINTS:
        return None

    rates: list[float] = []
    for i in range(len(present) - 1):
        gap = present[i + 1].step - present[i].step
        gap = gap if gap != 0 else 1
        rates.append((present[i + 1].value - present[i].value) / gap)

    med = median(rates)
    abs_devs = [abs(r - med) for r in rates]
    mad = median(abs_devs)
    values = [p.value for p in present]
    scale = max(values) - min(values) or (max(abs(v) for v in values) or 1.0)
    mad = max(mad, _SUDDEN_JUMP_MIN_MAD, _SUDDEN_JUMP_MAD_FLOOR_FRACTION * scale)

    z_scores = [abs(r - med) / (1.4826 * mad) for r in rates]
    max_z = max(z_scores)
    if max_z < _SUDDEN_JUMP_Z_THRESHOLD:
        return None

    idx = z_scores.index(max_z)
    score = min(max_z / (2 * _SUDDEN_JUMP_Z_THRESHOLD), 1.0)
    before, after = present[idx], present[idx + 1]
    return TrainingCurveSignal(
        signal="sudden_jump",
        score=score,
        step_range=(before.step, after.step),
        evidence={
            "step_before": before.step,
            "value_before": before.value,
            "step_after": after.step,
            "value_after": after.value,
            "diff": after.value - before.value,
            "rate": rates[idx],
            "z_score": max_z,
            "median_abs_rate": mad,
        },
        confidence=_confidence_from_score(score),
    )


def _detect_low_variance_plateau(points: Sequence[MetricPoint]) -> TrainingCurveSignal | None:
    """Flags the longest contiguous run of `_PLATEAU_WINDOW`-or-more
    present points whose coefficient of variation (population stdev over
    the window, divided by the curve's overall value range) stays below
    `_PLATEAU_CV_THRESHOLD` — i.e. training has stalled rather than
    merely converged smoothly. Reported only if such a run exists at all;
    `score` scales with how many multiples of the minimum window the
    plateau spans, capped at 1.0, so a plateau exactly at the minimum
    length reports a lower (but still present) score than a much longer
    one.
    """
    present = _present(points)
    if len(present) < _PLATEAU_WINDOW:
        return None

    values = [p.value for p in present]
    scale = max(values) - min(values)
    if scale == 0:
        scale = max(abs(v) for v in values) or 1.0

    flat_window_starts = [
        i
        for i in range(len(present) - _PLATEAU_WINDOW + 1)
        if pstdev(values[i : i + _PLATEAU_WINDOW]) / scale < _PLATEAU_CV_THRESHOLD
    ]
    if not flat_window_starts:
        return None

    best_start, best_end = flat_window_starts[0], flat_window_starts[0] + _PLATEAU_WINDOW - 1
    run_start, prev = flat_window_starts[0], flat_window_starts[0]
    for start in flat_window_starts[1:]:
        if start != prev + 1:
            run_start = start
        prev = start
        end = start + _PLATEAU_WINDOW - 1
        if end - run_start > best_end - best_start:
            best_start, best_end = run_start, end

    run_length = best_end - best_start + 1
    run_values = values[best_start : best_end + 1]
    score = min(run_length / (2 * _PLATEAU_WINDOW), 1.0)
    return TrainingCurveSignal(
        signal="low_variance_plateau",
        score=score,
        step_range=(present[best_start].step, present[best_end].step),
        evidence={
            "window_size": _PLATEAU_WINDOW,
            "run_length": run_length,
            "coefficient_of_variation_threshold": _PLATEAU_CV_THRESHOLD,
            "value_mean": mean(run_values),
            "value_pstdev": pstdev(run_values),
        },
        confidence=_confidence_from_score(score),
    )


def _detect_high_frequency_oscillation(
    points: Sequence[MetricPoint],
) -> TrainingCurveSignal | None:
    """Flags a metric whose consecutive (non-zero) diffs flip sign at a
    high rate — a jagged up/down pattern rather than a directional trend
    — using the fraction of adjacent diff-pairs with opposite sign as the
    score directly (already a natural 0-1 ratio). Zero diffs (an exact
    repeat) are excluded from the sign sequence entirely rather than
    counted as neither a flip nor a non-flip, since they carry no
    direction to compare against."""
    present = _present(points)
    if len(present) < _OSCILLATION_MIN_PRESENT_POINTS:
        return None

    diffs = [present[i + 1].value - present[i].value for i in range(len(present) - 1)]
    nonzero_diffs = [d for d in diffs if d != 0]
    if len(nonzero_diffs) < 2:
        return None

    flips = sum(
        1
        for i in range(len(nonzero_diffs) - 1)
        if (nonzero_diffs[i] > 0) != (nonzero_diffs[i + 1] > 0)
    )
    ratio = flips / (len(nonzero_diffs) - 1)
    if ratio < _OSCILLATION_SIGN_FLIP_RATIO_THRESHOLD:
        return None

    return TrainingCurveSignal(
        signal="high_frequency_oscillation",
        score=ratio,
        step_range=(present[0].step, present[-1].step),
        evidence={
            "sign_flip_ratio": ratio,
            "sign_flip_count": flips,
            "diff_count": len(nonzero_diffs),
        },
        confidence=_confidence_from_score(ratio),
    )


def audit_training_curve(
    history: MetricHistory, metric_type: MetricType | None = None
) -> TrainingCurveAudit:
    """Run all four signal detectors over one metric's full history.

    Pure and deterministic: no I/O, no dependency on how `history` was
    fetched. The MCP tool (`server.py`) is responsible for calling
    `ExperimentBackend.get_metric_history` and passing the result here —
    this function never fetches data itself, per spec §4.3's explicit
    data-flow rule ("audit_training_curve always fetches its own history
    via [get_metric_history] rather than accepting a pre-fetched blob").

    Args:
        history: The full metric history to audit.
        metric_type: Overrides `infer_metric_type(history.metric_name)`
            when the caller already knows the metric's type. `None` (the
            default) infers from the metric name, per spec §4.2.

    Returns:
        A `TrainingCurveAudit` at `schema_version: 2`. `signals` is empty
        for a curve with no detected pathology — not populated with
        near-zero-score entries.
    """
    assumed = metric_type if metric_type is not None else infer_metric_type(history.metric_name)
    detectors = (
        _detect_null_values,
        _detect_sudden_jump,
        _detect_low_variance_plateau,
        _detect_high_frequency_oscillation,
    )
    signals = [
        signal
        for signal in (detector(history.points) for detector in detectors)
        if signal is not None
    ]
    return TrainingCurveAudit(
        schema_version=SCHEMA_VERSION,
        metric_type_assumed=assumed,
        signals=signals,
        method=METHOD,
    )
