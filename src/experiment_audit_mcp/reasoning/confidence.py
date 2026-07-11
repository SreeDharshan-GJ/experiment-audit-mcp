"""
Experiment Audit Scientific Reasoning Engine

Module: confidence

Defines the Confidence stage of the reasoning pipeline (Evidence ->
Observations -> Hypotheses -> Scientific Rules -> **Confidence** ->
Judgment -> Recommendation, per
research/07_reasoning_engine/reasoning-engine.md). This corresponds to
that document's "Confidence Estimation" step and implements
research/07_reasoning_engine/confidence-system.md verbatim:

    Confidence depends on
        Evidence Quality
        Evidence Quantity
        Contradictions
        Agreement between signals
        Known failure modes
        Missing information
    Confidence is never guessed.
    Confidence is computed.

**Scope, strictly bounded.** This module answers exactly one question
per hypothesis: "how strongly does the available evidence support (or
undermine) this hypothesis?" It never inspects raw `Evidence`, never
talks to a backend (W&B, MLflow, ...), never generates hypotheses, and
never decides what the hypothesis *means* or what should be done about
it. Per the reasoning pipeline's staged design, those questions belong
to other modules:

- *what evidence exists* -> `evidence.py`
- *what is measurably true about that evidence* -> `observations.py`
- *why, and what might explain it* -> `hypotheses.py` (Hypothesis
  Generation)
- *is this a problem, and under what rule* -> `rules.py`
- *what do we conclude, given confidence* -> `judgment.py`
- *what should change* -> `recommendation.py`

This module consumes only the `Observation`s (`observations.py`) that
a hypothesis cites as supporting or contradicting it, plus whatever
`assumptions` the hypothesis itself declares. It never re-derives
those from `Evidence`, and it has no dependency on FastMCP, MCP
transport, `server.py`, or any backend implementation — the same
architectural constraint `evidence.py`, `observations.py`, and
`analysis/*.py` each state for themselves.

**Confidence is not probability.** A `ConfidenceAssessment.score` is a
bounded evidence-strength score in `[0.0, 1.0]`: how much support the
currently available evidence provides for a hypothesis, weighed
against how much it contradicts it and how much relevant information
is simply missing. It is not a calibrated probability that the
hypothesis is *true*, and this module makes no claim that it is. A
hypothesis can score `VERY_LOW` confidence not because it is false,
but because too little evidence has been gathered to say either way —
that distinction is exactly what the `limitations` field on each
assessment exists to preserve.

**A note on `Hypothesis` / `HypothesisSet`.** `hypotheses.py` (the
pipeline stage immediately upstream of this one) is not yet
implemented in this codebase — it is currently an empty stub. Per this
module's task boundary ("do not modify any other file"), the minimal
`Hypothesis` / `HypothesisSet` shapes this module needs as its *input*
are defined locally, below, rather than left unresolved. They follow
the same conventions `observations.py` established for `Observation` /
`ObservationSet` (frozen fact-holding dataclass + mutable append-only
collection, both with `to_dict()`), so that once `hypotheses.py` is
implemented for real, adopting its types here should be a matter of
replacing an import, not reshaping this module's logic. Nothing in
this module *generates* a hypothesis; it only reads the fields a
hypothesis exposes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from experiment_audit_mcp.models import RunRef
from experiment_audit_mcp.reasoning.observations import Observation, ObservationKind

# -- Thresholds and weights -- pure constants, kept in sync by hand if
# ever documented elsewhere (the same caveat observations.py flags for
# its own thresholds). Each is a constructor default on
# `ConfidenceAssessor`, so a caller can override any of them without
# editing this module. ------------------------------------------------

_SUPPORT_SATURATION = 3.0
"""Saturation constant for the evidence-*quantity* term of
`score_support`: with `n` supporting observations, the quantity term is
`n / (n + _SUPPORT_SATURATION)`, so it approaches (but never reaches)
1.0 as `n` grows, rather than scaling unboundedly with observation
count."""

_CONTRADICTION_SATURATION = 2.0
"""Saturation constant for the absolute-count term of
`score_contradictions`, analogous to `_SUPPORT_SATURATION`."""

_MISSING_INFO_SATURATION = 2.0
"""Saturation constant for `score_missing_information`'s combined count
of missing-information observations and declared assumptions."""

_CONTRADICTION_WEIGHT = 0.7
"""How strongly `score_contradictions`'s penalty discounts the final
score in `combine_scores`. Weighted higher than `_MISSING_INFO_WEIGHT`
because confidence-system.md lists "Contradictions" ahead of "Missing
information" among the dimensions confidence depends on."""

_MISSING_INFO_WEIGHT = 0.4
"""How strongly `score_missing_information`'s penalty discounts the
final score in `combine_scores`."""

_MISSING_INFO_KINDS = frozenset(
    {
        ObservationKind.METRIC_MISSING,
        ObservationKind.MISSING_METRIC_HISTORY,
        ObservationKind.SINGLE_RANDOM_SEED,
        ObservationKind.MISSING_SEED_INFORMATION,
        ObservationKind.MISSING_BASELINE,
        ObservationKind.EMPTY_LOGS,
        ObservationKind.MISSING_DATASET_INFORMATION,
        ObservationKind.MISSING_CODE_VERSION,
        ObservationKind.MISSING_HARDWARE_INFORMATION,
    }
)
"""`ObservationKind`s that represent an information *gap* rather than a
measured fact — confidence-system.md's "Missing information" dimension.
`score_missing_information` counts how many of a hypothesis's cited
observations fall in this set, regardless of whether they were cited
as supporting or contradicting; a missing baseline or a single random
seed weakens the evidentiary basis for a hypothesis no matter which
side of it the caller filed the observation under. Also doubles, per
this module's docstring, as this implementation's expression of
confidence-system.md's "Known failure modes" dimension: a failure mode
that was actually detected (e.g. `NAN_DETECTED`) shows up as a
*contradicting* observation and is scored by `score_contradictions`
instead; a failure mode that could not even be checked for (e.g. no
seed recorded at all) shows up here.

`NAN_DETECTED`, `TRAINING_PLATEAU_DETECTED`, `METRIC_INCREASING`,
`METRIC_DECREASING`, `MULTIPLE_RANDOM_SEEDS`,
`CONFIGURATION_CHANGED`, and `LARGE_VARIANCE_BETWEEN_RUNS` are
deliberately excluded: each is a positive, measured fact about
evidence that *was* collected, not an absence, so it belongs to
`score_support` / `score_contradictions` rather than here.
"""


class ConfidenceLevel(StrEnum):
    """A human-readable band over `ConfidenceAssessment.score`.

    A `str` subclass, matching `EvidenceKind` / `ObservationKind`'s
    convention, so a level serializes as its own value without a
    manual `.value` lookup at every call site. Boundaries are fixed by
    `ConfidenceAssessor._LEVEL_THRESHOLDS` (below) rather than baked
    into this enum, so the numeric cutoffs live in exactly one place.
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A candidate explanation, as `confidence.py` needs to consume it.

    This is the minimal shape this module requires from the
    not-yet-implemented `hypotheses.py` stage — see this module's
    docstring, "A note on `Hypothesis` / `HypothesisSet`". It carries
    no scoring or verdict of its own; those are exactly what
    `ConfidenceAssessment` (below) adds on top of one of these.

    Attributes:
        id: A short, stable, caller-assigned identifier (e.g.
            `"overfitting-val-loss"`), unique within a `HypothesisSet`.
            Used to cross-reference a `ConfidenceAssessment` back to
            the hypothesis it scored without repeating the full
            `statement`.
        statement: The hypothesis itself, stated once, in prose (e.g.
            "The model is overfitting on the validation set after
            step 400.").
        supporting_observations: Every `Observation` the hypothesis's
            author cited as consistent with `statement`.
        contradicting_observations: Every `Observation` the
            hypothesis's author cited as inconsistent with
            `statement`. An observation may legitimately appear here
            for one hypothesis and in `supporting_observations` for a
            different, competing hypothesis over the same evidence —
            this module scores one `Hypothesis` at a time and does not
            attempt to reconcile that across a `HypothesisSet`.
        assumptions: Plain-language conditions `statement` relies on
            that are not themselves backed by an `Observation` (e.g.
            "assumes the validation split did not change between
            epochs"). More declared assumptions means more of the
            hypothesis rests on unverified ground, which
            `score_missing_information` accounts for.
    """

    id: str
    statement: str
    supporting_observations: tuple[Observation, ...] = ()
    contradicting_observations: tuple[Observation, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Hypothesis.id must be a non-empty string.")
        if not self.statement:
            raise ValueError("Hypothesis.statement must be a non-empty string.")

    def subjects(self) -> tuple[RunRef, ...]:
        """Every distinct `RunRef` cited by this hypothesis's observations,
        in first-seen order across `supporting_observations` then
        `contradicting_observations`.
        """
        seen: dict[RunRef, None] = {}
        for obs in (*self.supporting_observations, *self.contradicting_observations):
            for ref in obs.subjects:
                seen.setdefault(ref, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "supporting_observations": [
                obs.to_dict() for obs in self.supporting_observations
            ],
            "contradicting_observations": [
                obs.to_dict() for obs in self.contradicting_observations
            ],
            "assumptions": list(self.assumptions),
        }


@dataclass(slots=True)
class HypothesisSet:
    """An ordered collection of `Hypothesis` objects, with lookup helpers.

    Mirrors `ObservationSet` in `observations.py`: mutable and
    append-only by convention, with no `remove`, so a caller building
    one incrementally never loses a hypothesis already added.
    """

    hypotheses: list[Hypothesis] = field(default_factory=list)

    def add(self, hypothesis: Hypothesis) -> None:
        """Append a single hypothesis."""
        self.hypotheses.append(hypothesis)

    def extend(self, hypotheses: Iterable[Hypothesis]) -> None:
        """Append every hypothesis from `hypotheses`, in order."""
        self.hypotheses.extend(hypotheses)

    def by_id(self, hypothesis_id: str) -> Hypothesis | None:
        """The hypothesis whose `id` equals `hypothesis_id`, or `None`."""
        for hyp in self.hypotheses:
            if hyp.id == hypothesis_id:
                return hyp
        return None

    def is_empty(self) -> bool:
        """Whether no hypotheses were added at all."""
        return not self.hypotheses

    def to_dict(self) -> dict[str, Any]:
        return {"hypotheses": [hyp.to_dict() for hyp in self.hypotheses]}

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __iter__(self) -> Iterator[Hypothesis]:
        return iter(self.hypotheses)

    def __bool__(self) -> bool:
        return bool(self.hypotheses)


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """How strongly available evidence supports one `Hypothesis`.

    Deliberately carries no `verdict` or `recommendation` field — those
    belong to `judgment.py` and `recommendation.py`. What this class
    does carry, so nothing downstream has to re-derive it, is exactly
    which observations and assumptions the score is based on and what
    it does *not* account for (`limitations`), per
    reasoning-engine.md's principle that every conclusion must "include
    confidence", "cite supporting observations", "identify
    contradictions", and "state limitations".

    Attributes:
        hypothesis: The `Hypothesis` this assessment is about.
        score: The evidence-strength score, in `[0.0, 1.0]`. Not a
            probability that `hypothesis` is true — see this module's
            docstring.
        confidence_level: `score` translated into a human-readable
            `ConfidenceLevel` band.
        supporting_observations: Copied from `hypothesis` for
            convenience, so a caller reading a `ConfidenceAssessment`
            in isolation (e.g. after serializing to JSON) does not
            need to re-fetch the `Hypothesis` to see what supported
            it.
        contradicting_observations: Copied from `hypothesis`, same
            rationale as `supporting_observations`.
        assumptions: Copied from `hypothesis.assumptions`.
        limitations: Plain-language caveats about what this specific
            assessment does not, or cannot, account for (e.g. "no
            supporting observations were provided", "3 of 4 cited
            observations report missing information rather than a
            measured fact"). Always includes a fixed reminder that
            `score` is an evidence-strength score, not a probability.
        explanation: A single human-readable paragraph stating how
            `score` was reached — the count and kind-diversity of
            supporting observations, the count of contradicting
            observations, and the missing-information signal — so a
            reader never has to reconstruct the computation from
            `score` alone.
    """

    hypothesis: Hypothesis
    score: float
    confidence_level: ConfidenceLevel
    supporting_observations: tuple[Observation, ...]
    contradicting_observations: tuple[Observation, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"ConfidenceAssessment.score must be in [0.0, 1.0], got {self.score}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "score": self.score,
            "confidence_level": self.confidence_level.value,
            "supporting_observations": [
                obs.to_dict() for obs in self.supporting_observations
            ],
            "contradicting_observations": [
                obs.to_dict() for obs in self.contradicting_observations
            ],
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class ConfidenceSet:
    """An ordered collection of `ConfidenceAssessment`s, with lookup helpers.

    Mirrors `ObservationSet` in `observations.py`: mutable and
    append-only by convention, with no `remove`.
    """

    assessments: list[ConfidenceAssessment] = field(default_factory=list)

    def add(self, assessment: ConfidenceAssessment) -> None:
        """Append a single assessment."""
        self.assessments.append(assessment)

    def extend(self, assessments: Iterable[ConfidenceAssessment]) -> None:
        """Append every assessment from `assessments`, in order."""
        self.assessments.extend(assessments)

    def by_hypothesis_id(self, hypothesis_id: str) -> ConfidenceAssessment | None:
        """The assessment whose `hypothesis.id` equals `hypothesis_id`, or
        `None` if none was recorded.
        """
        for assessment in self.assessments:
            if assessment.hypothesis.id == hypothesis_id:
                return assessment
        return None

    def by_level(self, level: ConfidenceLevel) -> list[ConfidenceAssessment]:
        """Every assessment at a given `ConfidenceLevel`, in the order
        recorded.
        """
        return [a for a in self.assessments if a.confidence_level is level]

    def sorted_by_score(self, *, descending: bool = True) -> list[ConfidenceAssessment]:
        """Every assessment, ordered by `score`.

        Ties are broken by the order assessments were added
        (`sorted` is stable), so calling this twice on an unchanged
        `ConfidenceSet` always returns the same order.
        """
        return sorted(self.assessments, key=lambda a: a.score, reverse=descending)

    def is_empty(self) -> bool:
        """Whether no assessments were recorded at all."""
        return not self.assessments

    def to_dict(self) -> dict[str, Any]:
        return {"assessments": [a.to_dict() for a in self.assessments]}

    def __len__(self) -> int:
        return len(self.assessments)

    def __iter__(self) -> Iterator[ConfidenceAssessment]:
        return iter(self.assessments)

    def __bool__(self) -> bool:
        return bool(self.assessments)


class ConfidenceAssessor:
    """Converts a `HypothesisSet` into a `ConfidenceSet`. No inference.

    Stateless and pure: every method here is a deterministic function
    of its `Hypothesis` argument plus this instance's configured
    weights and thresholds (constructor parameters, defaulted to the
    module-level constants above). Calling any method twice with the
    same inputs always returns equal results — no randomness, no I/O,
    no wall-clock dependence, no LLM calls, matching
    confidence-system.md's "Confidence is never guessed. Confidence is
    computed."

    The dimensions confidence-system.md lists are covered as follows:

    - *Evidence Quantity* and *Evidence Quality* -> `score_support`'s
      saturating count term.
    - *Agreement between signals* -> `score_support`'s kind-diversity
      term (independent kinds of supporting evidence agreeing counts
      for more than repeated instances of the same kind).
    - *Contradictions* -> `score_contradictions`.
    - *Known failure modes* and *Missing information* ->
      `score_missing_information` (see `_MISSING_INFO_KINDS`'s
      docstring for the distinction between a detected failure mode,
      scored as a contradiction, and an uncheckable one, scored here).

    `combine_scores` folds all three into the final `[0.0, 1.0]`
    score, and `assess` is the only public entry point most callers
    need.
    """

    _LEVEL_THRESHOLDS: tuple[tuple[float, ConfidenceLevel], ...] = (
        (0.2, ConfidenceLevel.VERY_LOW),
        (0.4, ConfidenceLevel.LOW),
        (0.6, ConfidenceLevel.MODERATE),
        (0.8, ConfidenceLevel.HIGH),
    )
    """Ascending `(upper_bound_exclusive, level)` pairs. A score `>=` the
    last pair's bound is `VERY_HIGH`. Kept as a class attribute (rather
    than a constructor parameter) since a caller wanting different
    bands can subclass and override `_level_for_score` directly; the
    common case (tuning saturation/weight constants) is covered by the
    constructor instead.
    """

    def __init__(
        self,
        *,
        support_saturation: float = _SUPPORT_SATURATION,
        contradiction_saturation: float = _CONTRADICTION_SATURATION,
        missing_info_saturation: float = _MISSING_INFO_SATURATION,
        contradiction_weight: float = _CONTRADICTION_WEIGHT,
        missing_info_weight: float = _MISSING_INFO_WEIGHT,
    ) -> None:
        """Configure scoring weights and saturation constants.

        Args:
            support_saturation: Saturation constant for
                `score_support`'s evidence-quantity term. Must be
                positive.
            contradiction_saturation: Saturation constant for
                `score_contradictions`'s absolute-count term. Must be
                positive.
            missing_info_saturation: Saturation constant for
                `score_missing_information`. Must be positive.
            contradiction_weight: How strongly `score_contradictions`
                discounts the final score in `combine_scores`. Must be
                in `[0.0, 1.0]`.
            missing_info_weight: How strongly
                `score_missing_information` discounts the final score
                in `combine_scores`. Must be in `[0.0, 1.0]`.

        Raises:
            ValueError: If any saturation constant is not positive, or
                either weight is outside `[0.0, 1.0]`.
        """
        if support_saturation <= 0:
            raise ValueError(f"support_saturation must be > 0, got {support_saturation}.")
        if contradiction_saturation <= 0:
            raise ValueError(
                f"contradiction_saturation must be > 0, got {contradiction_saturation}."
            )
        if missing_info_saturation <= 0:
            raise ValueError(
                f"missing_info_saturation must be > 0, got {missing_info_saturation}."
            )
        if not 0.0 <= contradiction_weight <= 1.0:
            raise ValueError(
                f"contradiction_weight must be in [0.0, 1.0], got {contradiction_weight}."
            )
        if not 0.0 <= missing_info_weight <= 1.0:
            raise ValueError(
                f"missing_info_weight must be in [0.0, 1.0], got {missing_info_weight}."
            )
        self._support_saturation = support_saturation
        self._contradiction_saturation = contradiction_saturation
        self._missing_info_saturation = missing_info_saturation
        self._contradiction_weight = contradiction_weight
        self._missing_info_weight = missing_info_weight

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def assess(self, hypothesis_set: HypothesisSet) -> ConfidenceSet:
        """Score every hypothesis in `hypothesis_set`.

        Args:
            hypothesis_set: The hypotheses to assess. Never mutated.

        Returns:
            A `ConfidenceSet` with exactly one `ConfidenceAssessment`
            per hypothesis in `hypothesis_set`, in the same order.
        """
        result = ConfidenceSet()
        for hypothesis in hypothesis_set:
            result.add(self._assess_one(hypothesis))
        return result

    def _assess_one(self, hypothesis: Hypothesis) -> ConfidenceAssessment:
        support = self.score_support(hypothesis)
        contradiction = self.score_contradictions(hypothesis)
        missing = self.score_missing_information(hypothesis)
        score = self.combine_scores(support, contradiction, missing)
        level = self._level_for_score(score)

        n_support = len(hypothesis.supporting_observations)
        n_contra = len(hypothesis.contradicting_observations)
        distinct_kinds = len({obs.kind for obs in hypothesis.supporting_observations})
        missing_count = sum(
            1
            for obs in (*hypothesis.supporting_observations, *hypothesis.contradicting_observations)
            if obs.kind in _MISSING_INFO_KINDS
        )

        explanation = (
            f"{n_support} supporting observation(s) across {distinct_kinds} distinct "
            f"kind(s) (support={support:.2f}) versus {n_contra} contradicting "
            f"observation(s) (contradiction={contradiction:.2f}); "
            f"{missing_count} cited observation(s) and {len(hypothesis.assumptions)} "
            f"declared assumption(s) reflect missing information (missing={missing:.2f}). "
            f"Combined score={score:.2f} ({level.value})."
        )

        limitations: list[str] = [
            "This score is an evidence-strength measure, not a probability "
            "that the hypothesis is true."
        ]
        if n_support == 0:
            limitations.append("No supporting observations were provided for this hypothesis.")
        if n_contra > 0:
            limitations.append(
                f"{n_contra} contradicting observation(s) were cited against this hypothesis."
            )
        if missing_count > 0:
            limitations.append(
                f"{missing_count} of the cited observations report missing information "
                "rather than a measured fact."
            )
        if hypothesis.assumptions:
            limitations.append(
                f"This hypothesis relies on {len(hypothesis.assumptions)} unverified "
                "assumption(s) not backed by an observation."
            )

        return ConfidenceAssessment(
            hypothesis=hypothesis,
            score=score,
            confidence_level=level,
            supporting_observations=hypothesis.supporting_observations,
            contradicting_observations=hypothesis.contradicting_observations,
            assumptions=hypothesis.assumptions,
            limitations=tuple(limitations),
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------

    def score_support(self, hypothesis: Hypothesis) -> float:
        """Score how much `hypothesis.supporting_observations` supports it.

        Combines two of confidence-system.md's dimensions:

        - *Evidence Quantity* / *Evidence Quality*: a saturating
          function of the observation count, `n / (n +
          support_saturation)`. More supporting observations raise the
          score, with diminishing returns rather than unbounded
          growth.
        - *Agreement between signals*: a diversity term rewarding
          supporting observations that come from *distinct*
          `ObservationKind`s over the same number of observations of a
          single repeated kind, since independent kinds of evidence
          agreeing is stronger support than one kind repeated.

        Args:
            hypothesis: The hypothesis to score.

        Returns:
            A value in `[0.0, 1.0]`. `0.0` if `hypothesis` has no
            supporting observations.
        """
        observations = hypothesis.supporting_observations
        n = len(observations)
        if n == 0:
            return 0.0

        quantity_term = n / (n + self._support_saturation)
        distinct_kinds = len({obs.kind for obs in observations})
        diversity_fraction = distinct_kinds / n
        agreement_term = min(1.0, 0.7 + 0.3 * diversity_fraction)

        return _clamp(quantity_term * agreement_term)

    def score_contradictions(self, hypothesis: Hypothesis) -> float:
        """Score how much `hypothesis.contradicting_observations` undermines it.

        Confidence-system.md's *Contradictions* dimension, expressed as
        a penalty strength in `[0.0, 1.0]` (`0.0` = no contradictions
        at all; `1.0` = maximal contradiction). Combines two views of
        the same contradicting observations so that both a *lot* of
        contradicting evidence and a contradicting-heavy *mix* (relative
        to supporting evidence) are penalized:

        - the fraction of all cited observations (supporting plus
          contradicting) that contradict the hypothesis;
        - a saturating function of the raw contradicting count,
          `n_contra / (n_contra + contradiction_saturation)`, so even a
          single contradicting observation against many supporting
          ones still registers rather than being diluted to near
          zero.

        The two views are averaged.

        Args:
            hypothesis: The hypothesis to score.

        Returns:
            A value in `[0.0, 1.0]`. `0.0` if `hypothesis` has no
            contradicting observations.
        """
        n_contra = len(hypothesis.contradicting_observations)
        if n_contra == 0:
            return 0.0

        n_support = len(hypothesis.supporting_observations)
        total = n_support + n_contra
        ratio_term = n_contra / total
        absolute_term = n_contra / (n_contra + self._contradiction_saturation)

        return _clamp((ratio_term + absolute_term) / 2.0)

    def score_missing_information(self, hypothesis: Hypothesis) -> float:
        """Score how much relevant information is absent for `hypothesis`.

        Confidence-system.md's *Missing information* dimension (and,
        per `_MISSING_INFO_KINDS`'s docstring, its treatment of
        uncheckable *Known failure modes*), expressed as a penalty
        strength in `[0.0, 1.0]`. Counts two things, combined with a
        saturating function of their sum:

        - cited observations (supporting or contradicting) whose kind
          is in `_MISSING_INFO_KINDS` — a missing baseline, a single
          random seed, an absent dataset record, and so on;
        - `hypothesis.assumptions` — conditions the hypothesis relies
          on that no observation backs at all.

        Args:
            hypothesis: The hypothesis to score.

        Returns:
            A value in `[0.0, 1.0]`. `0.0` if no cited observation
            reports missing information and `hypothesis` declares no
            assumptions.
        """
        missing_observation_count = sum(
            1
            for obs in (
                *hypothesis.supporting_observations,
                *hypothesis.contradicting_observations,
            )
            if obs.kind in _MISSING_INFO_KINDS
        )
        gap_count = missing_observation_count + len(hypothesis.assumptions)
        if gap_count == 0:
            return 0.0

        return _clamp(gap_count / (gap_count + self._missing_info_saturation))

    def combine_scores(self, support: float, contradiction: float, missing: float) -> float:
        """Fold the three component scores into one final confidence score.

        `support` sets the ceiling; `contradiction` and `missing` each
        multiplicatively discount it, weighted by
        `contradiction_weight` and `missing_info_weight` respectively
        (constructor parameters, defaulted so contradictions discount
        more heavily than missing information, per
        confidence-system.md's ordering of its dimensions). Multiplying
        discounts (rather than subtracting them) keeps the result
        naturally bounded in `[0.0, 1.0]` without needing a separate
        clamp step for the common case, and means a hypothesis with
        zero support stays at exactly `0.0` regardless of
        `contradiction` or `missing` — there is nothing for a penalty
        to discount.

        Args:
            support: `score_support`'s output, expected in `[0.0,
                1.0]`.
            contradiction: `score_contradictions`'s output, expected
                in `[0.0, 1.0]`.
            missing: `score_missing_information`'s output, expected in
                `[0.0, 1.0]`.

        Returns:
            The final confidence score, clamped to `[0.0, 1.0]`.
        """
        support = _clamp(support)
        contradiction = _clamp(contradiction)
        missing = _clamp(missing)

        contradiction_discount = 1.0 - self._contradiction_weight * contradiction
        missing_discount = 1.0 - self._missing_info_weight * missing

        return _clamp(support * contradiction_discount * missing_discount)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _level_for_score(self, score: float) -> ConfidenceLevel:
        """Translate a `[0.0, 1.0]` score into a `ConfidenceLevel` band,
        per `_LEVEL_THRESHOLDS`.
        """
        for upper_bound, level in self._LEVEL_THRESHOLDS:
            if score < upper_bound:
                return level
        return ConfidenceLevel.VERY_HIGH


def _clamp(value: float) -> float:
    """Clamp `value` into `[0.0, 1.0]`.

    A tiny shared helper so every scoring method returns a value that
    is provably in range even if float arithmetic edges just past a
    boundary, without each method repeating `max(0.0, min(1.0, ...))`
    inline.
    """
    return max(0.0, min(1.0, value))
