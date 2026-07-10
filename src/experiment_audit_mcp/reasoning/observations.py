"""
Experiment Audit Scientific Reasoning Engine

Module: observations

Defines the Observations stage of the reasoning pipeline (Evidence ->
**Observations** -> Hypotheses -> Rules -> Confidence -> Judgment ->
Recommendation, per research/07_reasoning_engine/reasoning-engine.md).
This corresponds to that document's "Evidence Validation" / "Pattern
Detection" steps: turning one or more `Evidence` bundles (evidence.py)
into a flat set of `Observation`s — plain, objectively measurable
statements about what evidence contains or omits.

**Scope, strictly bounded.** This module answers exactly one question
per observation: "what is measurably true about this evidence?" It
never answers "why," "how confident," or "what should be done." Per
the reasoning pipeline's staged design, those questions belong to
later modules:

- *why* -> `hypotheses.py` (Hypothesis Generation)
- *is this a problem, and under what rule* -> `rules.py`
- *how sure are we* -> `confidence.py` (per confidence-system.md:
  "Confidence is never guessed. Confidence is computed" — deliberately
  not computed here)
- *what do we conclude* -> `judgment.py`
- *what should change* -> `recommendation.py`

Concretely, this means every `Observation` produced here must be:

- **objectively measurable** — derived from a concrete comparison,
  threshold, or presence/absence check over `Evidence`, never from
  a subjective read of what a pattern "means";
- **reproducible** — re-running extraction over the same `Evidence`
  always yields the same `Observation`s (no randomness, no I/O, no
  wall-clock dependence);
- **backend independent** — derived only from the normalized
  `models.py` / `evidence.py` types, never from a specific backend's
  raw API shapes;
- **derived only from Evidence** — an `Observation` may cite the
  `Evidence`/`EvidenceItem`s that produced it, but introduces no
  outside fact.

A statement such as "probably overfitting" is a hypothesis, not an
observation, and does not belong here. The corresponding *observation*
this module does produce is the measurable pair of facts a later stage
would reason about: e.g. "Metric 'train_loss' decreased ..." and
"Metric 'val_loss' increased ..." as two separate, literal
`Observation`s. Synthesizing those two into "possible overfitting" is
`hypotheses.py`'s job, not this module's.

**Architectural constraint, mirrored from `evidence.py` and
`analysis/*.py`:** this module has no dependency on FastMCP, MCP
transport, `server.py`, or any backend implementation (`WandbBackend`,
`FakeBackend`, MLflow, ...). It operates only on `evidence.py`'s
`Evidence`/`EvidenceItem` and `models.py`'s `RunRef`, and reuses
`analysis/comparison.py`'s pure `compare_runs` for cross-run config
diffing rather than re-implementing that logic — the same "call the
existing pure function, don't duplicate the diff" pattern
`analysis/confound.py` already documents for itself. It returns plain
dataclasses with their own `to_dict()`, consistent with every other
module in this codebase.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from statistics import fmean, pstdev
from typing import Any

from experiment_audit_mcp.analysis.comparison import CompareRunsError, compare_runs
from experiment_audit_mcp.models import MetricPoint, RunRef
from experiment_audit_mcp.reasoning.evidence import Evidence

# -- Thresholds -- pure constants, kept in sync by hand if ever documented
# elsewhere (the same caveat analysis/divergence.py flags for its own
# thresholds). Each is referenced by name in the `detector` field of every
# `Observation` it produces, so a reader never has to guess which constant
# a given observation used. ---------------------------------------------

_PLATEAU_WINDOW = 5
"""Minimum number of consecutive present points considered for a plateau."""

_PLATEAU_CV_THRESHOLD = 0.02
"""Max coefficient of variation (population stdev / value range) within a
plateau window for it to still count as "flat"."""

_VARIANCE_CV_THRESHOLD = 0.15
"""Min coefficient of variation across runs' summary-metric values for
`LARGE_VARIANCE_BETWEEN_RUNS` to fire."""

_BASELINE_TAG_MARKERS = frozenset({"baseline", "control"})
"""Run tags (matched case-insensitively) that mark a run as a baseline."""

_BASELINE_CONFIG_KEYS = frozenset({"baseline", "is_baseline"})
"""Config keys whose truthy value marks a run as a baseline."""


class ObservationKind(StrEnum):
    """The closed set of observation categories this module can produce.

    A `str` subclass, matching `EvidenceKind`'s convention in
    `evidence.py`, so a kind serializes as its own value without a
    manual `.value` lookup at every call site.
    """

    NAN_DETECTED = "nan_detected"
    METRIC_MISSING = "metric_missing"
    MISSING_METRIC_HISTORY = "missing_metric_history"
    TRAINING_PLATEAU_DETECTED = "training_plateau_detected"
    METRIC_INCREASING = "metric_increasing"
    METRIC_DECREASING = "metric_decreasing"
    SINGLE_RANDOM_SEED = "single_random_seed"
    MULTIPLE_RANDOM_SEEDS = "multiple_random_seeds"
    MISSING_SEED_INFORMATION = "missing_seed_information"
    CONFIGURATION_CHANGED = "configuration_changed"
    MISSING_BASELINE = "missing_baseline"
    LARGE_VARIANCE_BETWEEN_RUNS = "large_variance_between_runs"
    EMPTY_LOGS = "empty_logs"
    MISSING_DATASET_INFORMATION = "missing_dataset_information"
    MISSING_CODE_VERSION = "missing_code_version"
    MISSING_HARDWARE_INFORMATION = "missing_hardware_information"


@dataclass(frozen=True, slots=True)
class Observation:
    """One objectively measurable statement about `Evidence`.

    Deliberately has no `confidence`, `severity`, or `verdict` field —
    those belong to `confidence.py` and `judgment.py`. What this class
    does carry, so nothing downstream has to re-derive it, is exactly
    which run(s) and which measured values produced the statement:
    `subjects` and `measurements` together make every `Observation`
    traceable back to the `Evidence` it came from, per
    evidence-model.md's "reasoning is always traceable back to
    evidence."

    Attributes:
        kind: Which category of observation this is.
        statement: A single, literal, human-readable sentence stating
            the measured fact — e.g. "Metric 'val_loss' increased from
            0.10 at step 0 to 0.35 at step 400." Written in the past
            tense, without hedging words ("probably", "likely",
            "seems") and without a causal claim, per this module's
            docstring.
        subjects: The run(s) this observation is about. A single-run
            observation (e.g. a NaN in one run's curve) carries one
            `RunRef`; a cross-run observation (e.g. a config change
            across a comparison group) carries every run involved.
        metric: The metric name this observation concerns, if any.
            `None` for observations that aren't metric-specific (e.g.
            `MISSING_BASELINE`, `CONFIGURATION_CHANGED`).
        measurements: The raw measured values supporting `statement`
            (e.g. `{"first_value": 0.10, "last_value": 0.35,
            "first_step": 0, "last_step": 400}`). Always JSON-safe.
            This is this module's analog of
            `analysis/divergence.py`'s `TrainingCurveSignal.evidence`
            field — the concrete numbers a reader can check `statement`
            against, not a re-statement of it.
        detector: A short, human-readable description of exactly which
            rule and threshold produced this observation (e.g.
            `"low_variance_plateau: window=5, cv_threshold=0.02"`).
            Present so re-running extraction is not the only way to
            audit how an observation was reached, and so the exact
            threshold used travels with the result even if this
            module's constants change later.
    """

    kind: ObservationKind
    statement: str
    subjects: tuple[RunRef, ...]
    metric: str | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    detector: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "statement": self.statement,
            "subjects": [_runref_to_dict(ref) for ref in self.subjects],
            "metric": self.metric,
            "measurements": dict(self.measurements),
            "detector": self.detector,
        }


@dataclass(slots=True)
class ObservationSet:
    """An ordered collection of `Observation`s, with lookup helpers.

    Mutable and append-only by convention (mirrors `Evidence.items` /
    `Evidence.add_item` in evidence.py): `ObservationExtractor` builds
    one of these incrementally, and a caller may append further
    observations of its own (e.g. from a custom detector) without
    losing anything already collected. There is intentionally no
    `remove`, for the same "never discard evidence-derived facts"
    reason `evidence.py` gives for `Evidence.add_item`.
    """

    observations: list[Observation] = field(default_factory=list)

    def add(self, observation: Observation) -> None:
        """Append a single observation."""
        self.observations.append(observation)

    def extend(self, observations: Iterable[Observation]) -> None:
        """Append every observation from `observations`, in order."""
        self.observations.extend(observations)

    def by_kind(self, kind: ObservationKind) -> list[Observation]:
        """Every observation of a given `kind`, in the order recorded."""
        return [obs for obs in self.observations if obs.kind is kind]

    def by_subject(self, ref: RunRef) -> list[Observation]:
        """Every observation whose `subjects` includes `ref`."""
        return [obs for obs in self.observations if ref in obs.subjects]

    def by_metric(self, metric: str) -> list[Observation]:
        """Every observation whose `metric` equals `metric`."""
        return [obs for obs in self.observations if obs.metric == metric]

    def kinds(self) -> set[ObservationKind]:
        """The distinct `ObservationKind`s present in this set."""
        return {obs.kind for obs in self.observations}

    def is_empty(self) -> bool:
        """Whether no observations were recorded at all."""
        return not self.observations

    def to_dict(self) -> dict[str, Any]:
        return {"observations": [obs.to_dict() for obs in self.observations]}

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.observations)

    def __bool__(self) -> bool:
        return bool(self.observations)


class ObservationExtractor:
    """Converts `Evidence` into `Observation`s. No inference, no scoring.

    Stateless and pure: every method here is a deterministic function
    of its `Evidence` argument(s) plus this instance's configured
    thresholds (constructor parameters, defaulted to the module-level
    constants above). Calling any method twice with the same inputs
    always returns equal results.

    Some observations are meaningful for a single run in isolation
    (e.g. a NaN in a curve, a missing dataset field); others are only
    meaningful across a *group* of runs being considered together
    (e.g. "configuration changed", "missing baseline", "large variance
    between runs" — none of these mean anything for a lone run). The
    per-run detectors (`detect_missing_information`,
    `detect_metric_patterns`, `detect_seed_information`) take one
    `Evidence`; the cross-run detectors
    (`detect_configuration_changes`, `detect_missing_baseline`,
    `detect_variance_between_runs`) take a `Sequence[Evidence]`.
    `extract` dispatches to both automatically.
    """

    def __init__(
        self,
        *,
        plateau_window: int = _PLATEAU_WINDOW,
        plateau_cv_threshold: float = _PLATEAU_CV_THRESHOLD,
        variance_cv_threshold: float = _VARIANCE_CV_THRESHOLD,
    ) -> None:
        """Configure detection thresholds.

        Args:
            plateau_window: Minimum number of consecutive present
                points examined for `TRAINING_PLATEAU_DETECTED`.
            plateau_cv_threshold: Maximum coefficient of variation
                within a window for it to count as a plateau.
            variance_cv_threshold: Minimum coefficient of variation
                across runs' summary-metric values for
                `LARGE_VARIANCE_BETWEEN_RUNS` to fire.
        """
        if plateau_window < 2:
            raise ValueError(f"plateau_window must be >= 2, got {plateau_window}.")
        if plateau_cv_threshold < 0:
            raise ValueError(
                f"plateau_cv_threshold must be >= 0, got {plateau_cv_threshold}."
            )
        if variance_cv_threshold < 0:
            raise ValueError(
                f"variance_cv_threshold must be >= 0, got {variance_cv_threshold}."
            )
        self._plateau_window = plateau_window
        self._plateau_cv_threshold = plateau_cv_threshold
        self._variance_cv_threshold = variance_cv_threshold

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def extract(self, evidence: Evidence | Sequence[Evidence]) -> ObservationSet:
        """Run every detector over `evidence` and collect the results.

        Args:
            evidence: A single `Evidence` bundle, or a group of them to
                be considered together (e.g. a baseline and its
                ablations, or several seeds of the same configuration).
                A bare `Evidence` is treated as a one-element group.

        Returns:
            An `ObservationSet` with every per-run observation for
            each bundle in the group, plus — when the group has two or
            more bundles — every cross-run observation for the group
            as a whole.

        Raises:
            ValueError: if `evidence` is an empty sequence.
        """
        group: tuple[Evidence, ...] = (
            (evidence,) if isinstance(evidence, Evidence) else tuple(evidence)
        )
        if not group:
            raise ValueError("extract() requires at least one Evidence bundle.")

        result = ObservationSet()
        for item in group:
            result.extend(self.detect_missing_information(item))
            result.extend(self.detect_metric_patterns(item))
            result.extend(self.detect_seed_information(item))

        if len(group) > 1:
            result.extend(self.detect_configuration_changes(group))
            result.extend(self.detect_missing_baseline(group))
            result.extend(self.detect_variance_between_runs(group))

        return result

    # ------------------------------------------------------------------
    # Per-run detectors
    # ------------------------------------------------------------------

    def detect_missing_information(self, evidence: Evidence) -> list[Observation]:
        """Flag evidence categories that are absent from `evidence`.

        Each check here is a plain presence/absence test against one
        `Evidence` field — no thresholding, no interpretation. Note
        the same caveat `evidence.py` documents for these fields: an
        empty `dict`/`list` here means "nothing was supplied when this
        bundle was assembled," which is indistinguishable from "known
        to genuinely have none" — this module reports the former fact
        as measured, without claiming to know which case it is.

        Args:
            evidence: The bundle to inspect.

        Returns:
            One `Observation` per missing category detected; an empty
            list if `evidence` has no gaps this detector checks for.
        """
        observations: list[Observation] = []
        subjects = (evidence.ref,)

        has_summary = evidence.run is not None and bool(evidence.run.summary_metrics)
        has_curves = bool(evidence.metric_histories)
        if not has_summary and not has_curves:
            observations.append(
                Observation(
                    kind=ObservationKind.METRIC_MISSING,
                    statement="No metrics (summary or curve) were recorded for this run.",
                    subjects=subjects,
                    detector="presence check: run.summary_metrics and metric_histories both empty",
                )
            )

        if has_summary and evidence.run is not None:
            missing_curve_metrics = sorted(
                set(evidence.run.summary_metrics) - set(evidence.metric_histories)
            )
            if missing_curve_metrics:
                observations.append(
                    Observation(
                        kind=ObservationKind.MISSING_METRIC_HISTORY,
                        statement=(
                            "Summary values are present for "
                            f"{len(missing_curve_metrics)} metric(s) with no corresponding "
                            f"training curve: {', '.join(missing_curve_metrics)}."
                        ),
                        subjects=subjects,
                        measurements={"metrics": missing_curve_metrics},
                        detector="presence check: summary_metrics key absent from metric_histories",
                    )
                )

        if not evidence.logs:
            observations.append(
                Observation(
                    kind=ObservationKind.EMPTY_LOGS,
                    statement="No log lines were recorded for this run.",
                    subjects=subjects,
                    detector="presence check: evidence.logs is empty",
                )
            )

        if not evidence.dataset:
            observations.append(
                Observation(
                    kind=ObservationKind.MISSING_DATASET_INFORMATION,
                    statement="No dataset information was recorded for this run.",
                    subjects=subjects,
                    detector="presence check: evidence.dataset is empty",
                )
            )

        if evidence.code_version is None:
            observations.append(
                Observation(
                    kind=ObservationKind.MISSING_CODE_VERSION,
                    statement="No code version (commit/tag) was recorded for this run.",
                    subjects=subjects,
                    detector="presence check: evidence.code_version is None",
                )
            )

        if not evidence.hardware:
            observations.append(
                Observation(
                    kind=ObservationKind.MISSING_HARDWARE_INFORMATION,
                    statement="No hardware information was recorded for this run.",
                    subjects=subjects,
                    detector="presence check: evidence.hardware is empty",
                )
            )

        return observations

    def detect_metric_patterns(self, evidence: Evidence) -> list[Observation]:
        """Detect NaNs, plateaus, and net increase/decrease in each curve.

        Operates only on `evidence.metric_histories` — metrics with a
        summary value but no curve are already covered by
        `detect_missing_information`'s `MISSING_METRIC_HISTORY` check,
        since a trend or plateau cannot be measured from a single
        summary value.

        Args:
            evidence: The bundle to inspect.

        Returns:
            Every metric-pattern observation found, across every curve
            in `evidence.metric_histories`. A metric with a clean,
            fully-present, flat-free, trend-free curve contributes no
            observations — this detector never manufactures a signal
            where none was measured.
        """
        observations: list[Observation] = []
        subjects = (evidence.ref,)

        for metric_name, history in evidence.metric_histories.items():
            points = history.points
            null_steps = [p.step for p in points if p.value is None]
            if null_steps:
                observations.append(
                    Observation(
                        kind=ObservationKind.NAN_DETECTED,
                        statement=(
                            f"Metric '{metric_name}' has {len(null_steps)} null/NaN "
                            f"value(s) among {len(points)} logged point(s), at step(s) "
                            f"{min(null_steps)}-{max(null_steps)}."
                        ),
                        subjects=subjects,
                        metric=metric_name,
                        measurements={
                            "null_step_count": len(null_steps),
                            "total_point_count": len(points),
                            "null_steps": null_steps,
                        },
                        detector="presence check: MetricPoint.value is None",
                    )
                )

            present = _present_points(points)

            plateau = self._detect_plateau(present)
            if plateau is not None:
                start, end, cv = plateau
                observations.append(
                    Observation(
                        kind=ObservationKind.TRAINING_PLATEAU_DETECTED,
                        statement=(
                            f"Metric '{metric_name}' was flat (coefficient of variation "
                            f"{cv:.4g}) from step {present[start].step} to "
                            f"{present[end].step}."
                        ),
                        subjects=subjects,
                        metric=metric_name,
                        measurements={
                            "step_range": [present[start].step, present[end].step],
                            "window_length": end - start + 1,
                            "coefficient_of_variation": cv,
                        },
                        detector=(
                            "low_variance_plateau: window="
                            f"{self._plateau_window}, cv_threshold={self._plateau_cv_threshold}"
                        ),
                    )
                )

            trend = _net_trend(present)
            if trend is not None:
                kind = (
                    ObservationKind.METRIC_INCREASING
                    if trend > 0
                    else ObservationKind.METRIC_DECREASING
                )
                direction = "increased" if trend > 0 else "decreased"
                first, last = present[0], present[-1]
                observations.append(
                    Observation(
                        kind=kind,
                        statement=(
                            f"Metric '{metric_name}' {direction} from {first.value:.6g} "
                            f"at step {first.step} to {last.value:.6g} at step {last.step}."
                        ),
                        subjects=subjects,
                        metric=metric_name,
                        measurements={
                            "first_step": first.step,
                            "first_value": first.value,
                            "last_step": last.step,
                            "last_value": last.value,
                            "net_change": trend,
                        },
                        detector="net trend: sign of last present value minus first present value",
                    )
                )

        return observations

    def detect_seed_information(self, evidence: Evidence) -> list[Observation]:
        """Report how many distinct random seeds this bundle carries.

        Reads `evidence.seeds` directly (per `evidence.py`, this field
        may already aggregate seed evidence across repeated runs of
        the same configuration, so no cross-bundle grouping is needed
        here to state this fact for one bundle).

        Args:
            evidence: The bundle to inspect.

        Returns:
            A single-element list with exactly one of
            `MISSING_SEED_INFORMATION`, `SINGLE_RANDOM_SEED`, or
            `MULTIPLE_RANDOM_SEEDS`.
        """
        subjects = (evidence.ref,)
        distinct_seeds = sorted(set(evidence.seeds))

        if not distinct_seeds:
            return [
                Observation(
                    kind=ObservationKind.MISSING_SEED_INFORMATION,
                    statement="No random seed was recorded for this run.",
                    subjects=subjects,
                    detector="presence check: evidence.seeds is empty",
                )
            ]

        if len(distinct_seeds) == 1:
            return [
                Observation(
                    kind=ObservationKind.SINGLE_RANDOM_SEED,
                    statement=f"Only one random seed ({distinct_seeds[0]}) was recorded.",
                    subjects=subjects,
                    measurements={"seeds": distinct_seeds},
                    detector="count check: len(set(evidence.seeds)) == 1",
                )
            ]

        return [
            Observation(
                kind=ObservationKind.MULTIPLE_RANDOM_SEEDS,
                statement=(
                    f"{len(distinct_seeds)} distinct random seeds were recorded: "
                    f"{distinct_seeds}."
                ),
                subjects=subjects,
                measurements={"seeds": distinct_seeds},
                detector="count check: len(set(evidence.seeds)) > 1",
            )
        ]

    # ------------------------------------------------------------------
    # Cross-run detectors
    # ------------------------------------------------------------------

    def detect_configuration_changes(self, group: Sequence[Evidence]) -> list[Observation]:
        """Report every config parameter that differs across `group`.

        Delegates the actual diffing to
        `analysis/comparison.py`'s `compare_runs` rather than
        re-implementing it — the same run-normalized `Run` model both
        modules already share makes this a direct reuse, not an
        adaptation. Only bundles with a `run` attached participate,
        since `compare_runs` operates on `Run`, not `Evidence`;
        bundles missing a `run` are silently excluded here (their
        absence is separately reported by `detect_missing_information`
        for that specific bundle, not duplicated here).

        Args:
            group: Two or more `Evidence` bundles to compare.

        Returns:
            One `CONFIGURATION_CHANGED` observation per differing
            config parameter (covering examples such as a changed
            learning rate or batch size, which are just config keys
            like any other — this detector does not hard-code
            particular hyperparameter names). Empty if fewer than two
            bundles carry a `run`, or if every run's config agrees.
        """
        runs = [item.run for item in group if item.run is not None]
        if len(runs) < 2:
            return []

        try:
            result = compare_runs(runs)
        except CompareRunsError:
            # compare_runs also rejects duplicate RunRefs within `runs`;
            # duplicate Evidence for the same run carries no config-diff
            # signal, so treat it the same as "nothing to compare."
            return []

        observations: list[Observation] = []
        for entry in result.config_diff:
            subjects = tuple(entry.values.keys())
            rendered = {
                _runref_key(ref): (value.value if value.present else None)
                for ref, value in entry.values.items()
            }
            observations.append(
                Observation(
                    kind=ObservationKind.CONFIGURATION_CHANGED,
                    statement=(
                        f"Configuration parameter '{entry.param}' differs across "
                        f"{len(subjects)} runs: {rendered}."
                    ),
                    subjects=subjects,
                    measurements={"parameter": entry.param, "values_by_run": rendered},
                    detector="analysis.comparison.compare_runs: config_diff",
                )
            )
        return observations

    def detect_missing_baseline(self, group: Sequence[Evidence]) -> list[Observation]:
        """Report when no run in `group` is identifiable as a baseline.

        A run counts as marked-baseline if it carries a tag matching
        `_BASELINE_TAG_MARKERS` (case-insensitively) or a truthy config
        value under one of `_BASELINE_CONFIG_KEYS`. This is a lexical
        check against explicit markers a run's own metadata provides —
        not an inference about which run *functions* as a baseline,
        which would require judging experimental intent rather than
        measuring evidence.

        Args:
            group: Two or more `Evidence` bundles being considered
                together (e.g. a candidate ablation set).

        Returns:
            A single-element list with a `MISSING_BASELINE` observation
            if no bundle in `group` carries a baseline marker; an
            empty list if at least one does, or if `group` has fewer
            than two bundles.
        """
        if len(group) < 2:
            return []

        for item in group:
            if item.run is None:
                continue
            tags = {tag.lower() for tag in item.run.tags}
            if tags & _BASELINE_TAG_MARKERS:
                return []
            for key in _BASELINE_CONFIG_KEYS:
                if bool(item.run.config.get(key, False)):
                    return []

        subjects = tuple(item.ref for item in group)
        return [
            Observation(
                kind=ObservationKind.MISSING_BASELINE,
                statement=(
                    f"None of the {len(group)} runs being compared carry a baseline "
                    "marker (a tag or config value identifying it as the baseline)."
                ),
                subjects=subjects,
                measurements={
                    "checked_tags": sorted(_BASELINE_TAG_MARKERS),
                    "checked_config_keys": sorted(_BASELINE_CONFIG_KEYS),
                },
                detector="marker check: run.tags / run.config against known baseline markers",
            )
        ]

    def detect_variance_between_runs(self, group: Sequence[Evidence]) -> list[Observation]:
        """Report summary metrics whose values vary widely across `group`.

        For each metric name with a present summary value on at least
        two bundles in `group`, computes the coefficient of variation
        (population standard deviation divided by the mean of the
        absolute values, to stay well-defined when the mean is near
        zero) and reports it when it clears `variance_cv_threshold`.

        Args:
            group: Two or more `Evidence` bundles being considered
                together.

        Returns:
            One `LARGE_VARIANCE_BETWEEN_RUNS` observation per metric
            whose spread clears the threshold; empty if no metric
            does, or if `group` has fewer than two bundles.
        """
        if len(group) < 2:
            return []

        by_metric: dict[str, dict[RunRef, float]] = {}
        for item in group:
            for name in item.metric_names():
                value = item.get_summary_metric(name)
                if value is None:
                    value = item.latest_metric_value(name)
                if value is not None:
                    by_metric.setdefault(name, {})[item.ref] = value

        observations: list[Observation] = []
        for metric_name, values_by_ref in sorted(by_metric.items()):
            if len(values_by_ref) < 2:
                continue
            values = list(values_by_ref.values())
            scale = fmean(abs(v) for v in values) or 1.0
            cv = pstdev(values) / scale
            if cv < self._variance_cv_threshold:
                continue
            observations.append(
                Observation(
                    kind=ObservationKind.LARGE_VARIANCE_BETWEEN_RUNS,
                    statement=(
                        f"Metric '{metric_name}' varies widely across "
                        f"{len(values_by_ref)} runs (coefficient of variation {cv:.4g}, "
                        f"threshold {self._variance_cv_threshold:.4g})."
                    ),
                    subjects=tuple(values_by_ref.keys()),
                    metric=metric_name,
                    measurements={
                        "values_by_run": {
                            _runref_key(ref): value for ref, value in values_by_ref.items()
                        },
                        "coefficient_of_variation": cv,
                    },
                    detector=(
                        "coefficient of variation across run summary metrics: "
                        f"threshold={self._variance_cv_threshold}"
                    ),
                )
            )
        return observations

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_plateau(
        self, present: Sequence[MetricPoint]
    ) -> tuple[int, int, float] | None:
        """Find the longest contiguous flat run in `present`, if any.

        Returns `(start_index, end_index, coefficient_of_variation)`
        for the longest window (indices into `present`) whose
        coefficient of variation stays below
        `self._plateau_cv_threshold`, or `None` if no window of at
        least `self._plateau_window` points is that flat.
        """
        window = self._plateau_window
        if len(present) < window:
            return None

        values = [p.value for p in present if p.value is not None]
        if len(values) < window:
            return None

        value_range = max(values) - min(values)
        scale = value_range if value_range > 0 else (max(abs(v) for v in values) or 1.0)

        flat_starts = [
            i
            for i in range(len(present) - window + 1)
            if all(p.value is not None for p in present[i : i + window])
            and pstdev([p.value for p in present[i : i + window]]) / scale
            < self._plateau_cv_threshold
        ]
        if not flat_starts:
            return None

        best_start, best_end = flat_starts[0], flat_starts[0] + window - 1
        run_start, prev = flat_starts[0], flat_starts[0]
        for start in flat_starts[1:]:
            if start != prev + 1:
                run_start = start
            prev = start
            end = start + window - 1
            if end - run_start > best_end - best_start:
                best_start, best_end = run_start, end

        window_values = [p.value for p in present[best_start : best_end + 1]]
        cv = pstdev(window_values) / scale
        return best_start, best_end, cv


def _present_points(points: Sequence[MetricPoint]) -> list[MetricPoint]:
    """Every point in `points` whose value is not a logged NaN/null."""
    return [p for p in points if p.value is not None]


def _net_trend(present: Sequence[MetricPoint]) -> float | None:
    """Difference between the last and first present values in `present`.

    Returns `None` if fewer than two present points exist (no trend is
    measurable), or if the first and last present values are exactly
    equal (no net direction to report).
    """
    if len(present) < 2:
        return None
    delta = present[-1].value - present[0].value
    return delta if delta != 0 else None


def _runref_key(ref: RunRef) -> str:
    """Stable, JSON-safe string key for a `RunRef`.

    Local copy of the same helper `analysis/comparison.py` defines for
    itself (`_runref_key`) and `evidence.py` defines for itself
    (`_runref_to_dict`, dict-shaped rather than string-shaped) — each
    module keeps its own private copy rather than importing another
    module's underscore-prefixed name, per the convention `evidence.py`
    documents for its own `_runref_to_dict`.
    """
    return f"{ref.backend}/{ref.entity}/{ref.project}/{ref.run_id}"


def _runref_to_dict(ref: RunRef) -> dict[str, str]:
    """Local copy of `models.py`'s private `_runref_to_dict`, used by
    `Observation.to_dict()` to serialize `subjects`."""
    return {
        "backend": ref.backend,
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
    }
