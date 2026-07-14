"""
Experiment Audit Scientific Reasoning Engine

Module: recommendation

Defines the Recommendation stage of the reasoning pipeline (Evidence ->
Observations -> Hypotheses -> Rules -> Confidence -> Judgment ->
**Recommendation**, per research/07_reasoning_engine/reasoning-engine.md).
This is the pipeline's terminal stage: it turns a `JudgmentSet` (the
engine's scientific conclusions) into a flat set of `Recommendation`s —
concrete, actionable next steps a researcher can take in response to
those conclusions ("Repeat with multiple random seeds.", "Add a
baseline.", "Reduce learning rate.", ...).

**Scope, strictly bounded.** This module answers exactly one question
per recommendation: "given what the engine has already concluded, what
should the researcher do next?" It never re-derives a verdict, never
re-weighs evidence, and never second-guesses a `Judgment`'s confidence
level — those questions belong to earlier stages. Concretely, every
`Recommendation` produced here must be:

- **traceable** — every `Recommendation` cites the `Judgment`(s) (via
  `related_judgments`) that motivated it, per evidence-model.md's
  "reasoning is always traceable back to evidence" carried through to
  the pipeline's last stage;
- **deterministic** — re-running generation over the same `JudgmentSet`
  always yields the same `RecommendationSet`, in the same order (no
  randomness, no I/O, no wall-clock dependence, no LLM calls);
- **actionable** — a recommendation names a concrete next step (repeat
  with more seeds, add a baseline, reduce the learning rate, verify
  logging, ...), never a vague restatement of the judgment itself.

**Architectural constraint, mirrored from `evidence.py` and
`observations.py`:** this module has no dependency on FastMCP, MCP
transport, `server.py`, or any backend implementation (`WandbBackend`,
`FakeBackend`, ...), and it does not call out to an LLM — recommendation
generation here is a pure, rule-based mapping from judgment verdicts to
recommendation templates, consistent with reasoning-engine.md's "The
LLM is only the interface. Experiment Audit performs the reasoning."
It returns plain dataclasses with their own `to_dict()`, consistent with
every other module in this codebase.

**A note on `Judgment` / `JudgmentSet`.** These types are, per the
pipeline diagram above, properly owned by `judgment.py` — the same way
`Evidence` is owned by `evidence.py` and `Observation` is owned by
`observations.py`. As of this stage's implementation, `judgment.py` is
not yet built out (it is a placeholder module), and this task is scoped
to `recommendation.py` only. Rather than importing a `Judgment` shape
that does not exist yet, this module defines the minimal `Judgment` /
`JudgmentSet` / `JudgmentVerdict` / `ConfidenceLevel` types it needs to
do its job, directly below, clearly marked as this stage's placeholder
for that upstream contract. When `judgment.py` is implemented, it is
expected to either match this shape or for this module to be updated to
import from it instead; nothing here otherwise depends on that migration.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from experiment_audit.models import RunRef

# ---------------------------------------------------------------------------
# Placeholder upstream contract (see module docstring's note on Judgment /
# JudgmentSet). Everything in this section is what recommendation.py needs
# from the Judgment stage; nothing here performs judgment-stage reasoning.
# ---------------------------------------------------------------------------


class ConfidenceLevel(StrEnum):
    """How sure the engine is in a `Judgment`'s verdict.

    Per confidence-system.md ("Confidence is never guessed. Confidence
    is computed."), this is expected to be *set* by `confidence.py`
    and merely *read* here to calibrate recommendation priority — a
    `LOW`-confidence judgment should not, on its own, drive a
    `CRITICAL` recommendation.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class JudgmentVerdict(StrEnum):
    """The closed set of scientific conclusions a `Judgment` can reach.

    Named after the examples in research/07_reasoning_engine's
    reasoning-rules.md (Rules 001-005) and the pattern categories
    `observations.py`'s `ObservationKind` already establishes, so a
    later `judgment.py` implementation has a ready-made target shape.
    """

    CONFOUNDED_EXPERIMENT = "confounded_experiment"
    LOW_STATISTICAL_CONFIDENCE = "low_statistical_confidence"
    POSSIBLE_OVERFITTING = "possible_overfitting"
    POSSIBLE_GRADIENT_INSTABILITY = "possible_gradient_instability"
    COMPARISON_INVALID = "comparison_invalid"
    MISSING_BASELINE = "missing_baseline"
    TRAINING_STALLED = "training_stalled"
    LARGE_RUN_VARIANCE = "large_run_variance"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    METRIC_LOGGING_ANOMALY = "metric_logging_anomaly"
    DATASET_INTEGRITY_CONCERN = "dataset_integrity_concern"
    SOUND = "sound"


@dataclass(frozen=True, slots=True)
class Judgment:
    """One scientific conclusion the engine has reached.

    Attributes:
        id: A stable, caller-assigned identifier for this judgment
            (e.g. `"j001"`). Used by `Recommendation.related_judgments`
            so a recommendation can cite exactly which judgment(s)
            motivated it, per this module's traceability requirement.
        verdict: The closed-set conclusion reached.
        statement: A human-readable explanation of the conclusion
            (e.g. "Three hyperparameters changed simultaneously between
            the baseline and treatment runs.").
        confidence: How sure the engine is in `verdict`.
        subjects: The run(s) this judgment is about.
        supporting_observations: Free-text references to the
            `Observation`s (e.g. their `statement`s) that support this
            judgment. Not interpreted by this module; carried through
            purely for traceability.
        contradictions: Free-text notes on any evidence that
            contradicts `verdict`, per reasoning-engine.md's "identify
            contradictions" principle.
        limitations: Free-text notes on what this judgment does *not*
            establish, per reasoning-engine.md's "state limitations"
            principle. Folded into a recommendation's `rationale` when
            present.
    """

    id: str
    verdict: JudgmentVerdict
    statement: str
    confidence: ConfidenceLevel
    subjects: tuple[RunRef, ...] = ()
    supporting_observations: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "verdict": self.verdict.value,
            "statement": self.statement,
            "confidence": self.confidence.value,
            "subjects": [_runref_to_dict(ref) for ref in self.subjects],
            "supporting_observations": list(self.supporting_observations),
            "contradictions": list(self.contradictions),
            "limitations": list(self.limitations),
        }


@dataclass(slots=True)
class JudgmentSet:
    """An ordered collection of `Judgment`s, with lookup helpers.

    Mirrors `ObservationSet` (observations.py) and `Evidence.items`
    (evidence.py): mutable and append-only by convention, so a caller
    building this incrementally never loses a judgment already added.
    """

    judgments: list[Judgment] = field(default_factory=list)

    def add(self, judgment: Judgment) -> None:
        """Append a single judgment."""
        self.judgments.append(judgment)

    def extend(self, judgments: Iterable[Judgment]) -> None:
        """Append every judgment from `judgments`, in order."""
        self.judgments.extend(judgments)

    def by_verdict(self, verdict: JudgmentVerdict) -> list[Judgment]:
        """Every judgment with a given `verdict`, in the order recorded."""
        return [j for j in self.judgments if j.verdict is verdict]

    def is_empty(self) -> bool:
        """Whether no judgments were recorded at all."""
        return not self.judgments

    def to_dict(self) -> dict[str, Any]:
        return {"judgments": [j.to_dict() for j in self.judgments]}

    def __len__(self) -> int:
        return len(self.judgments)

    def __iter__(self) -> Iterator[Judgment]:
        return iter(self.judgments)

    def __bool__(self) -> bool:
        return bool(self.judgments)


# ---------------------------------------------------------------------------
# Recommendation stage proper.
# ---------------------------------------------------------------------------


class RecommendationPriority(StrEnum):
    """How urgently a `Recommendation` should be acted on.

    A `str` subclass, matching `EvidenceKind` / `ObservationKind`'s
    convention, so a priority serializes as its own value without a
    manual `.value` lookup at every call site. Ordered LOW -> CRITICAL;
    see `_PRIORITY_RANK` for the module's ordering logic, kept outside
    the enum itself since `StrEnum` members do not compare by rank.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_PRIORITY_ORDER: tuple[RecommendationPriority, ...] = (
    RecommendationPriority.LOW,
    RecommendationPriority.MEDIUM,
    RecommendationPriority.HIGH,
    RecommendationPriority.CRITICAL,
)
_PRIORITY_RANK: dict[RecommendationPriority, int] = {
    priority: rank for rank, priority in enumerate(_PRIORITY_ORDER)
}


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One actionable scientific suggestion produced from a `JudgmentSet`.

    Frozen, matching `Observation` / `EvidenceItem`'s convention: a
    recommendation, once generated, is not mutated in place — a caller
    that wants a different recommendation re-runs generation rather
    than editing this record.

    Attributes:
        id: A stable, deterministic identifier for this recommendation
            (derived from its underlying template's key), so the same
            `JudgmentSet` always yields recommendations with the same
            ids across runs.
        title: A short, imperative, actionable suggestion (e.g.
            "Repeat with multiple random seeds."), suitable for display
            as a checklist item.
        explanation: One or two sentences on what the suggested action
            *is* and why it addresses the underlying issue in general
            (not specific to any one judgment).
        priority: How urgently this recommendation should be acted on.
        rationale: Why *this* recommendation was generated for *this*
            `JudgmentSet` specifically — built from the statement(s) and
            confidence of the judgment(s) that motivated it, per
            reasoning-engine.md's "explain why" principle.
        related_judgments: The `Judgment.id`s that motivated this
            recommendation, per this module's traceability requirement.
            May reference more than one judgment when several judgments
            independently point at the same corrective action.
        expected_impact: What acting on this recommendation should
            achieve (e.g. "Distinguishes a genuine effect from
            run-to-run noise.").
    """

    id: str
    title: str
    explanation: str
    priority: RecommendationPriority
    rationale: str
    related_judgments: tuple[str, ...] = ()
    expected_impact: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "explanation": self.explanation,
            "priority": self.priority.value,
            "rationale": self.rationale,
            "related_judgments": list(self.related_judgments),
            "expected_impact": self.expected_impact,
        }


@dataclass(slots=True)
class RecommendationSet:
    """An ordered collection of `Recommendation`s, with lookup helpers.

    Mirrors `ObservationSet` / `JudgmentSet`: mutable and append-only by
    convention. There is intentionally no `remove`, for the same "never
    silently drop a pipeline-derived conclusion" reason those classes
    give for their own append-only design.
    """

    recommendations: list[Recommendation] = field(default_factory=list)

    def add(self, recommendation: Recommendation) -> None:
        """Append a single recommendation."""
        self.recommendations.append(recommendation)

    def extend(self, recommendations: Iterable[Recommendation]) -> None:
        """Append every recommendation from `recommendations`, in order."""
        self.recommendations.extend(recommendations)

    def by_priority(self, priority: RecommendationPriority) -> list[Recommendation]:
        """Every recommendation at a given `priority`, in the order recorded."""
        return [r for r in self.recommendations if r.priority is priority]

    def sorted_by_priority(self) -> list[Recommendation]:
        """All recommendations, `CRITICAL` first and `LOW` last.

        Ties (equal priority) keep their original relative order
        (Python's sort is stable), which is generation order — itself
        deterministic, so this method's output is deterministic too.
        """
        return sorted(
            self.recommendations,
            key=lambda r: _PRIORITY_RANK[r.priority],
            reverse=True,
        )

    def highest_priority(self) -> RecommendationPriority | None:
        """The most urgent priority present, or `None` if this set is empty."""
        if not self.recommendations:
            return None
        return max(
            (r.priority for r in self.recommendations),
            key=lambda p: _PRIORITY_RANK[p],
        )

    def is_empty(self) -> bool:
        """Whether no recommendations were recorded at all."""
        return not self.recommendations

    def to_dict(self) -> dict[str, Any]:
        return {"recommendations": [r.to_dict() for r in self.recommendations]}

    def __len__(self) -> int:
        return len(self.recommendations)

    def __iter__(self) -> Iterator[Recommendation]:
        return iter(self.recommendations)

    def __bool__(self) -> bool:
        return bool(self.recommendations)


@dataclass(frozen=True, slots=True)
class _RecommendationTemplate:
    """A verdict-independent recommendation blueprint.

    Internal to this module. One `JudgmentVerdict` may map to more than
    one template (e.g. a confounded experiment should both be re-run
    with an isolated variable *and* followed up with an ablation), and
    several judgments that map to the *same* template are merged into
    one `Recommendation` by `RecommendationGenerator` rather than
    repeated verbatim.
    """

    key: str
    title: str
    explanation: str
    base_priority: RecommendationPriority
    expected_impact: str


# Verdict -> recommendation template(s). Order within a tuple is the order
# those recommendations appear in the output when a judgment reaches that
# verdict. Kept as a module-level constant (not computed) so the mapping
# itself is inspectable and stays deterministic across runs.
_RECOMMENDATION_RULES: dict[JudgmentVerdict, tuple[_RecommendationTemplate, ...]] = {
    JudgmentVerdict.CONFOUNDED_EXPERIMENT: (
        _RecommendationTemplate(
            key="run_ablation",
            title="Run an ablation.",
            explanation=(
                "Isolate the changed variables by re-running with only one "
                "hyperparameter or component altered at a time."
            ),
            base_priority=RecommendationPriority.CRITICAL,
            expected_impact=(
                "Isolates which changed factor is actually responsible for the "
                "observed effect, enabling a valid causal claim."
            ),
        ),
    ),
    JudgmentVerdict.LOW_STATISTICAL_CONFIDENCE: (
        _RecommendationTemplate(
            key="repeat_seeds",
            title="Repeat with multiple random seeds.",
            explanation=(
                "Re-run the same configuration under several random seeds and "
                "compare the resulting distribution of outcomes, not a single run."
            ),
            base_priority=RecommendationPriority.HIGH,
            expected_impact=(
                "Distinguishes a genuine effect from run-to-run noise, producing a "
                "statistically defensible comparison."
            ),
        ),
    ),
    JudgmentVerdict.POSSIBLE_OVERFITTING: (
        _RecommendationTemplate(
            key="add_regularization",
            title="Add regularization or early stopping.",
            explanation=(
                "Introduce weight decay, dropout, or an early-stopping criterion "
                "keyed to validation performance."
            ),
            base_priority=RecommendationPriority.HIGH,
            expected_impact=(
                "Reduces the generalization gap between training and validation performance."
            ),
        ),
    ),
    JudgmentVerdict.POSSIBLE_GRADIENT_INSTABILITY: (
        _RecommendationTemplate(
            key="reduce_learning_rate",
            title="Reduce learning rate.",
            explanation=(
                "Lower the learning rate (and/or add gradient clipping) and "
                "re-run to check whether the instability recurs."
            ),
            base_priority=RecommendationPriority.CRITICAL,
            expected_impact=(
                "Reduces the chance of gradient explosion or divergence, stabilizing training."
            ),
        ),
    ),
    JudgmentVerdict.COMPARISON_INVALID: (
        _RecommendationTemplate(
            key="rerun_matched_comparison",
            title="Re-run the comparison with matched configurations.",
            explanation=(
                "Hold every setting constant between the compared runs except the "
                "one variable under test before re-comparing."
            ),
            base_priority=RecommendationPriority.HIGH,
            expected_impact=(
                "Restores a like-for-like comparison so any observed difference "
                "can be attributed to the intended variable."
            ),
        ),
    ),
    JudgmentVerdict.MISSING_BASELINE: (
        _RecommendationTemplate(
            key="add_baseline",
            title="Add a baseline.",
            explanation=(
                "Run the unmodified/control configuration alongside the "
                "treatment so results have a reference point."
            ),
            base_priority=RecommendationPriority.HIGH,
            expected_impact=(
                "Establishes a reference point the treatment can be meaningfully compared against."
            ),
        ),
    ),
    JudgmentVerdict.TRAINING_STALLED: (
        _RecommendationTemplate(
            key="increase_training_duration",
            title="Increase training duration.",
            explanation=(
                "Extend the run past the point where the metric appeared to "
                "plateau to check whether it was a true convergence point."
            ),
            base_priority=RecommendationPriority.MEDIUM,
            expected_impact=(
                "Determines whether the plateau is genuine convergence or premature stopping."
            ),
        ),
    ),
    JudgmentVerdict.LARGE_RUN_VARIANCE: (
        _RecommendationTemplate(
            key="investigate_variance",
            title="Investigate the source of run-to-run variance.",
            explanation=(
                "Audit configuration, seeding, and hardware/environment details "
                "across the runs for an unintended difference."
            ),
            base_priority=RecommendationPriority.MEDIUM,
            expected_impact=(
                "Surfaces an unintended source of variation, or confirms the "
                "variance is intrinsic to the method."
            ),
        ),
    ),
    JudgmentVerdict.INCOMPLETE_EVIDENCE: (
        _RecommendationTemplate(
            key="collect_missing_metadata",
            title="Collect the missing experiment metadata.",
            explanation=(
                "Backfill the absent config, seed, hardware, or dataset details "
                "before drawing further conclusions from this run."
            ),
            base_priority=RecommendationPriority.MEDIUM,
            expected_impact=(
                "Fills evidentiary gaps so future judgments can be made with higher confidence."
            ),
        ),
    ),
    JudgmentVerdict.METRIC_LOGGING_ANOMALY: (
        _RecommendationTemplate(
            key="verify_metric_logging",
            title="Verify metric logging.",
            explanation=(
                "Check the logging code path and any metric-computation logic "
                "for the affected metric(s) for a measurement artifact."
            ),
            base_priority=RecommendationPriority.MEDIUM,
            expected_impact=(
                "Rules out a measurement or logging artifact as the cause of the "
                "anomalous readings before drawing conclusions."
            ),
        ),
    ),
    JudgmentVerdict.DATASET_INTEGRITY_CONCERN: (
        _RecommendationTemplate(
            key="check_dataset_integrity",
            title="Check dataset integrity.",
            explanation=(
                "Verify dataset version, split boundaries, and preprocessing for "
                "corruption, leakage, or an unintended change."
            ),
            base_priority=RecommendationPriority.HIGH,
            expected_impact=(
                "Rules out data corruption, leakage, or a preprocessing error as a confound."
            ),
        ),
    ),
    JudgmentVerdict.SOUND: (
        _RecommendationTemplate(
            key="proceed",
            title="Proceed to the next experiment stage.",
            explanation=(
                "No corrective action is indicated; the current setup can be built on directly."
            ),
            base_priority=RecommendationPriority.LOW,
            expected_impact=(
                "Confirms the current configuration is valid, so effort can move "
                "to the next experiment rather than re-validating this one."
            ),
        ),
    ),
}


def _fallback_template(verdict: JudgmentVerdict) -> _RecommendationTemplate:
    """A generic template for a verdict with no rule in `_RECOMMENDATION_RULES`.

    Deterministic in `verdict` alone, so an unmapped verdict still
    produces a stable, reproducible recommendation rather than being
    silently dropped — consistent with this module's traceability
    requirement (every judgment should be answered by *something*, even
    if that something is "review this by hand").
    """
    readable = verdict.value.replace("_", " ")
    return _RecommendationTemplate(
        key=f"review_{verdict.value}",
        title=f"Manually review the '{readable}' judgment.",
        explanation=(
            "This verdict has no codified recommendation rule yet; a researcher "
            "should review the supporting evidence and observations directly."
        ),
        base_priority=RecommendationPriority.LOW,
        expected_impact="Ensures this judgment is not silently dropped from the audit trail.",
    )


def _adjust_priority(
    base_priority: RecommendationPriority, confidence: ConfidenceLevel
) -> RecommendationPriority:
    """Calibrate a template's `base_priority` against a judgment's confidence.

    Per confidence-system.md ("Confidence is never guessed. Confidence
    is computed."), this module treats a `LOW`-confidence judgment as
    grounds to soften the urgency of the recommendation it produces —
    a judgment the engine is not sure about should not, on its own,
    drive a `CRITICAL` call to action. `MEDIUM`/`HIGH` confidence
    judgments use the template's priority unmodified.
    """
    if confidence is not ConfidenceLevel.LOW:
        return base_priority
    downgraded_rank = max(_PRIORITY_RANK[base_priority] - 1, 0)
    return _PRIORITY_ORDER[downgraded_rank]


@dataclass(slots=True)
class _Accumulator:
    """Mutable, internal merge-state for one recommendation template.

    `RecommendationGenerator.generate` merges every judgment that maps
    to the same template into a single `Recommendation`; this class
    holds that recommendation's state while it is being built up.
    """

    template: _RecommendationTemplate
    priority: RecommendationPriority
    related_judgments: list[str] = field(default_factory=list)
    rationale_lines: list[str] = field(default_factory=list)

    def absorb(self, judgment: Judgment, priority: RecommendationPriority) -> None:
        """Fold one more `Judgment` into this accumulator.

        `priority` only ever escalates (never downgrades) as more
        judgments are absorbed, since if *any* contributing judgment
        warrants a higher priority, the merged recommendation should
        carry that higher priority rather than average it away.
        """
        if _PRIORITY_RANK[priority] > _PRIORITY_RANK[self.priority]:
            self.priority = priority
        if judgment.id not in self.related_judgments:
            self.related_judgments.append(judgment.id)
        line = f"{judgment.statement} (confidence: {judgment.confidence.value})."
        if judgment.limitations:
            line += f" Noted limitations: {'; '.join(judgment.limitations)}."
        if line not in self.rationale_lines:
            self.rationale_lines.append(line)


class RecommendationGenerator:
    """Converts a `JudgmentSet` into a `RecommendationSet`.

    Stateless and pure: `generate` is a deterministic function of its
    `judgment_set` argument alone (plus this module's fixed
    `_RECOMMENDATION_RULES` table) — the same `judgment_set` always
    yields an equal `RecommendationSet`, in the same order. No random
    sampling, no I/O, no network or LLM calls, per this module's and
    reasoning-engine.md's "The LLM is only the interface. Experiment
    Audit performs the reasoning." constraint.
    """

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate(self, judgment_set: JudgmentSet) -> RecommendationSet:
        """Generate every recommendation warranted by `judgment_set`.

        Each `Judgment` is mapped to one or more recommendation
        templates via `_RECOMMENDATION_RULES` (or `_fallback_template`
        for a verdict with no codified rule). Judgments that map to the
        same template are merged into a single `Recommendation`: its
        `priority` is the highest priority any contributing judgment
        warrants, and its `rationale`/`related_judgments` accumulate
        across every contributing judgment.

        Args:
            judgment_set: The judgments to generate recommendations
                for. An empty `JudgmentSet` yields an empty
                `RecommendationSet` — this is a legitimate result, not
                an error, mirroring `ObservationSet.is_empty`.

        Returns:
            A `RecommendationSet` with one `Recommendation` per distinct
            template touched by `judgment_set`, in the order each
            template was first encountered.
        """
        accumulators: dict[str, _Accumulator] = {}
        order: list[str] = []

        for judgment in judgment_set:
            templates = _RECOMMENDATION_RULES.get(judgment.verdict)
            if templates is None:
                templates = (_fallback_template(judgment.verdict),)
            for template in templates:
                priority = _adjust_priority(template.base_priority, judgment.confidence)
                accumulator = accumulators.get(template.key)
                if accumulator is None:
                    accumulator = _Accumulator(template=template, priority=priority)
                    accumulators[template.key] = accumulator
                    order.append(template.key)
                accumulator.absorb(judgment, priority)

        result = RecommendationSet()
        for key in order:
            accumulator = accumulators[key]
            result.add(
                Recommendation(
                    id=f"rec_{key}",
                    title=accumulator.template.title,
                    explanation=accumulator.template.explanation,
                    priority=accumulator.priority,
                    rationale=" ".join(accumulator.rationale_lines),
                    related_judgments=tuple(accumulator.related_judgments),
                    expected_impact=accumulator.template.expected_impact,
                )
            )
        return result


def _runref_to_dict(ref: RunRef) -> dict[str, str]:
    """Local copy of `models.py`'s private `_runref_to_dict`.

    `models.py` deliberately doesn't export its underscore-prefixed
    helper for cross-module reuse; `evidence.py` and `observations.py`
    each keep their own local copy rather than reaching into another
    module's private name, and this module follows the same convention.
    """
    return {
        "backend": ref.backend,
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
    }
