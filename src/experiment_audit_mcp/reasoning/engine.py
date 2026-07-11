"""
Experiment Audit Scientific Reasoning Engine

Module: engine

Orchestrates the reasoning pipeline (Evidence Collection -> Evidence
Validation -> Pattern Detection -> Hypothesis Generation -> Contradiction
Search -> Confidence Estimation -> Scientific Judgment -> Recommendations,
per research/07_reasoning_engine/reasoning-engine.md), which this
codebase's module layout maps onto as:

    Evidence (evidence.py)
        -> ObservationExtractor (observations.py)
        -> HypothesisGenerator (hypotheses.py)
        -> ScientificRuleEngine (rules.py)
        -> ConfidenceAssessor (confidence.py)
        -> JudgmentGenerator (judgment.py)
        -> RecommendationGenerator (recommendation.py)

**Scope, strictly bounded.** This module's only job is to connect those
stages in order and carry each stage's output into the next. It contains
no scientific reasoning of its own:

- it does not detect patterns (`observations.py`'s job),
- it does not synthesize explanations (`hypotheses.py`'s job),
- it does not evaluate rules such as reasoning-rules.md's Rule 001-005
  (`rules.py`'s job),
- it does not compute confidence (`confidence.py`'s job, and per
  confidence-system.md: "Confidence is never guessed. Confidence is
  computed" -- computed *there*, not here),
- it does not render a verdict (`judgment.py`'s job),
- it does not decide what should change (`recommendation.py`'s job).

Every one of those modules is a stub at the time this file is written
(only `evidence.py` and `observations.py` currently have real
implementations). Rather than reach into those files -- which this task
explicitly forbids -- every stage this module cannot yet import a
concrete implementation for is expressed as a `Protocol` defined *here*,
describing only the shape `ScientificReasoningEngine` needs to call. This
keeps `engine.py` honest about the one thing a pure orchestrator is
allowed to know about a stage: its interface, not its behavior. Per the
task's explicit requirement, `ScientificRuleEngine` in particular is
represented as an interface only -- this module never evaluates a rule,
it only calls whatever `ScientificRuleEngine` implementation was injected
into it.

**Dependency injection, not construction.** `ScientificReasoningEngine`
never builds its own stage implementations (with the sole exception of
`ObservationExtractor`, which already exists as a concrete, pure class in
`observations.py` and is safe to default to). Every other stage is
supplied by the caller at construction time. This means:

1. `engine.py` stays importable and testable today, even though five of
   the six stages it orchestrates have no real implementation yet -- a
   caller injects fakes/stubs satisfying the `Protocol`s below.
2. Once `hypotheses.py`, `rules.py`, `confidence.py`, `judgment.py`, and
   `recommendation.py` are implemented, wiring the real
   `ScientificReasoningEngine` requires no change to this file -- a
   caller just injects the real classes, which will structurally satisfy
   these `Protocol`s (or can be adapted to).

**Architectural constraint, mirrored from every other module in this
package:** this module has no dependency on FastMCP, MCP transport,
`server.py`, or any backend implementation. It operates only on
`evidence.py`'s `Evidence` and `observations.py`'s `ObservationExtractor`
/ `ObservationSet`, plus the `Protocol`s defined below for the remaining
stages, and returns a plain dataclass (`ReasoningResult`) with its own
`to_dict()`, consistent with the pattern `models.py`, `evidence.py`, and
`observations.py` each already establish for themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from experiment_audit_mcp.reasoning.evidence import Evidence
from experiment_audit_mcp.reasoning.observations import ObservationExtractor, ObservationSet

# ----------------------------------------------------------------------
# Stage output type parameters.
#
# Each later pipeline stage (hypotheses.py, rules.py, confidence.py,
# judgment.py, recommendation.py) is a stub as of this writing, so this
# module has no concrete `Hypothesis`, `RuleFinding`, `Confidence`,
# `Judgment`, or `Recommendation` type to import. Rather than guess at
# those shapes -- which would mean this file silently encodes reasoning
# structure that belongs to those modules -- every stage output is a
# free type parameter. `ScientificReasoningEngine` is generic over all
# five, so once each real type exists a caller gets full static typing
# simply by instantiating `ScientificReasoningEngine[Hypothesis,
# RuleFinding, Confidence, Judgment, Recommendation]`; nothing in this
# file needs to change.
# ----------------------------------------------------------------------

HypothesisT = TypeVar("HypothesisT")
RuleFindingT = TypeVar("RuleFindingT")
ConfidenceT = TypeVar("ConfidenceT")
JudgmentT = TypeVar("JudgmentT")
RecommendationT = TypeVar("RecommendationT")


# ----------------------------------------------------------------------
# Stage interfaces.
#
# Structural (`Protocol`, `runtime_checkable`) rather than nominal (ABC)
# on purpose: whatever concrete classes eventually land in hypotheses.py,
# rules.py, confidence.py, judgment.py, and recommendation.py should not
# be required to import this module or subclass anything defined here
# just to be pipeline-compatible -- they only need to expose the method
# shape each `Protocol` names. This mirrors how `ObservationExtractor`
# itself needs no special base class to be usable below.
# ----------------------------------------------------------------------


@runtime_checkable
class HypothesisGenerator(Protocol[HypothesisT]):
    """Turns pattern-level `Observation`s into candidate explanations.

    Corresponds to reasoning-engine.md's "Hypothesis Generation" step.
    `hypotheses.py` owns what a hypothesis actually is and how it is
    derived (e.g. synthesizing a `METRIC_INCREASING` observation on
    `val_loss` together with a `METRIC_DECREASING` observation on
    `train_loss` into a "possible overfitting" hypothesis) -- this
    `Protocol` only names the call `ScientificReasoningEngine` needs to
    make.
    """

    def generate(self, observations: ObservationSet) -> Sequence[HypothesisT]:
        """Return every hypothesis `observations` supports."""
        ...


@runtime_checkable
class ScientificRuleEngine(Protocol[HypothesisT, RuleFindingT]):
    """Evaluates scientific rules (reasoning-rules.md) against evidence.

    Represented as an interface only, per this task's explicit
    requirement: this module declares the shape a rule engine must have
    to participate in the pipeline, but performs no rule evaluation of
    its own and makes no assumption about which rules exist. That is
    entirely `rules.py`'s responsibility -- e.g. reasoning-rules.md's
    Rule 001 ("multiple hyperparameters changed AND claim == ablation ->
    confounded experiment") or Rule 003 ("validation loss increases AND
    training loss decreases -> possible overfitting") are rule-engine
    concerns, never engine.py concerns.

    Corresponds to reasoning-engine.md's "Contradiction Search" step:
    a rule engine's findings are expected to include not just pattern
    matches but the contradictions those patterns imply (e.g. a claimed
    ablation whose evidence shows more than one changed hyperparameter).
    """

    def evaluate(
        self,
        observations: ObservationSet,
        hypotheses: Sequence[HypothesisT],
    ) -> Sequence[RuleFindingT]:
        """Return every rule finding `observations` and `hypotheses` support."""
        ...


@runtime_checkable
class ConfidenceAssessor(Protocol[RuleFindingT, ConfidenceT]):
    """Computes confidence for a set of rule findings.

    Corresponds to reasoning-engine.md's "Confidence Estimation" step.
    Per confidence-system.md ("Confidence is never guessed. Confidence
    is computed" -- as a function of evidence quality, evidence
    quantity, contradictions, signal agreement, known failure modes,
    and missing information), the *computation* lives entirely in
    `confidence.py`; this `Protocol` only names the call site.
    """

    def assess(
        self,
        observations: ObservationSet,
        rule_findings: Sequence[RuleFindingT],
    ) -> ConfidenceT:
        """Return the computed confidence for `rule_findings`."""
        ...


@runtime_checkable
class JudgmentGenerator(Protocol[RuleFindingT, ConfidenceT, JudgmentT]):
    """Renders scientific judgments from rule findings and confidence.

    Corresponds to reasoning-engine.md's "Scientific Judgment" step, and
    scientific-reviewer.md's framing ("Would I trust this conclusion?").
    Per reasoning-engine.md's principles, every judgment this produces
    is expected to be supported by evidence, carry confidence, explain
    why, cite supporting observations, identify contradictions, and
    state limitations -- all of which is `judgment.py`'s responsibility,
    not this module's.
    """

    def generate(
        self,
        rule_findings: Sequence[RuleFindingT],
        confidence: ConfidenceT,
    ) -> Sequence[JudgmentT]:
        """Return every judgment `rule_findings` and `confidence` support."""
        ...


@runtime_checkable
class RecommendationGenerator(Protocol[JudgmentT, RecommendationT]):
    """Turns scientific judgments into actionable recommendations.

    Corresponds to reasoning-engine.md's final "Recommendations" step.
    What a recommendation looks like, and how it follows from a
    judgment, is entirely `recommendation.py`'s concern.
    """

    def generate(self, judgments: Sequence[JudgmentT]) -> Sequence[RecommendationT]:
        """Return every recommendation `judgments` support."""
        ...


class NullRuleEngine:
    """A `ScientificRuleEngine` that evaluates no rules.

    The pipeline diagram this module implements names the rule-engine
    stage a "placeholder only" -- this class is that placeholder made
    concrete enough to be a usable default. It satisfies
    `ScientificRuleEngine` structurally (see `evaluate`) while producing
    no findings at all, so `ScientificReasoningEngine` is fully
    constructible and `review()`-able today, before `rules.py` has a
    real implementation, without silently fabricating rule findings.

    Not the default for any other stage: `HypothesisGenerator`,
    `ConfidenceAssessor`, `JudgmentGenerator`, and
    `RecommendationGenerator` have no evidence-free "do nothing" reading
    that wouldn't misrepresent a real result (an empty confidence value,
    for instance, is not a meaningful default the way "no rule findings"
    is), so those remain required constructor arguments with no default.
    """

    def evaluate(
        self,
        observations: ObservationSet,
        hypotheses: Sequence[Any],
    ) -> Sequence[Any]:
        """Always returns an empty sequence of findings."""
        del observations, hypotheses  # unused: this is an intentional no-op
        return ()


@dataclass(frozen=True, slots=True)
class ReasoningResult(Generic[HypothesisT, RuleFindingT, ConfidenceT, JudgmentT, RecommendationT]):
    """The full output of one `ScientificReasoningEngine.review()` call.

    Purely a carrier: one field per pipeline stage's output, in pipeline
    order, so a caller can inspect any intermediate stage's result
    (e.g. for debugging or for citing which observations a judgment
    drew on) rather than only the final recommendations --
    evidence-model.md's "reasoning is always traceable back to
    evidence" applies to this result as a whole, not just to
    `recommendations`.

    Frozen and `slots=True`: a result is a snapshot of one `review()`
    call and should not be mutated after the fact, matching the
    "evidence is never discarded, never edited in place" convention
    `evidence.py`'s `EvidenceItem` documents for itself.

    Attributes:
        evidence: The evidence this result was computed from. Kept as
            given to `review()` -- a single `Evidence` bundle, or the
            group considered together.
        observations: Every observation `ObservationExtractor` derived
            from `evidence`.
        hypotheses: Every hypothesis `HypothesisGenerator` derived from
            `observations`.
        rule_findings: Every finding `ScientificRuleEngine` derived from
            `observations` and `hypotheses`. Empty when the injected
            rule engine is a `NullRuleEngine` (the default).
        confidence: The confidence `ConfidenceAssessor` computed for
            `rule_findings`.
        judgments: Every judgment `JudgmentGenerator` derived from
            `rule_findings` and `confidence`.
        recommendations: Every recommendation `RecommendationGenerator`
            derived from `judgments`.
    """

    evidence: Evidence | Sequence[Evidence]
    observations: ObservationSet
    hypotheses: Sequence[HypothesisT]
    rule_findings: Sequence[RuleFindingT]
    confidence: ConfidenceT
    judgments: Sequence[JudgmentT]
    recommendations: Sequence[RecommendationT]

    def to_dict(self) -> dict[str, Any]:
        """Best-effort JSON-safe serialization.

        Each stage-output field is heterogeneous (`HypothesisT`,
        `RuleFindingT`, ...) since those types don't exist in this
        codebase yet; this delegates to each item's own `to_dict()`
        when present (the convention every concrete type in this
        package follows, per `evidence.py` and `observations.py`) and
        otherwise passes the value through unchanged, so this method
        never raises just because a stage's concrete type doesn't
        exist yet.
        """
        evidence_value = (
            self.evidence.to_dict()
            if isinstance(self.evidence, Evidence)
            else [item.to_dict() for item in self.evidence]
        )
        return {
            "evidence": evidence_value,
            "observations": self.observations.to_dict(),
            "hypotheses": [_to_jsonable(item) for item in self.hypotheses],
            "rule_findings": [_to_jsonable(item) for item in self.rule_findings],
            "confidence": _to_jsonable(self.confidence),
            "judgments": [_to_jsonable(item) for item in self.judgments],
            "recommendations": [_to_jsonable(item) for item in self.recommendations],
        }


class ScientificReasoningEngine(
    Generic[HypothesisT, RuleFindingT, ConfidenceT, JudgmentT, RecommendationT]
):
    """Connects the reasoning pipeline's stages. Implements none of them.

    Every stage after evidence collection is supplied via constructor
    injection (`HypothesisGenerator`, `ScientificRuleEngine`,
    `ConfidenceAssessor`, `JudgmentGenerator`, `RecommendationGenerator`)
    rather than constructed internally, so this class has no knowledge
    of *how* a hypothesis is generated, a rule is evaluated, a
    confidence is computed, a judgment is rendered, or a recommendation
    is derived -- only that each step, given the previous step's
    output, produces the next. `observation_extractor` is the sole
    exception, defaulting to `ObservationExtractor()` since that class
    is already a complete, pure implementation in this package.

    Public API is a single method, `review`, matching this task's
    requirement: one entry point that runs `evidence` through every
    stage and returns a `ReasoningResult` carrying each stage's output.
    """

    def __init__(
        self,
        *,
        hypothesis_generator: HypothesisGenerator[HypothesisT],
        confidence_assessor: ConfidenceAssessor[RuleFindingT, ConfidenceT],
        judgment_generator: JudgmentGenerator[RuleFindingT, ConfidenceT, JudgmentT],
        recommendation_generator: RecommendationGenerator[JudgmentT, RecommendationT],
        rule_engine: ScientificRuleEngine[HypothesisT, RuleFindingT] | None = None,
        observation_extractor: ObservationExtractor | None = None,
    ) -> None:
        """Wire the pipeline together.

        Args:
            hypothesis_generator: Turns `Observation`s into hypotheses.
                Required -- this pipeline has no meaningful default
                reading of "no hypothesis generator".
            confidence_assessor: Computes confidence for rule findings.
                Required, for the same reason.
            judgment_generator: Renders judgments from rule findings and
                confidence. Required, for the same reason.
            recommendation_generator: Derives recommendations from
                judgments. Required, for the same reason.
            rule_engine: Evaluates scientific rules against observations
                and hypotheses. Defaults to `NullRuleEngine()` -- the
                pipeline's explicit "placeholder only" stage -- so this
                engine is fully usable before `rules.py` has a real
                implementation; inject a real `ScientificRuleEngine`
                once one exists.
            observation_extractor: Extracts `Observation`s from
                `Evidence`. Defaults to `ObservationExtractor()`, since
                that class is already a complete, pure implementation;
                inject a differently-configured instance (e.g. custom
                thresholds) if needed.
        """
        self._observation_extractor = observation_extractor or ObservationExtractor()
        self._hypothesis_generator = hypothesis_generator
        self._rule_engine: ScientificRuleEngine[HypothesisT, RuleFindingT] = (
            rule_engine if rule_engine is not None else NullRuleEngine()
        )
        self._confidence_assessor = confidence_assessor
        self._judgment_generator = judgment_generator
        self._recommendation_generator = recommendation_generator

    def review(
        self, evidence: Evidence | Sequence[Evidence]
    ) -> ReasoningResult[HypothesisT, RuleFindingT, ConfidenceT, JudgmentT, RecommendationT]:
        """Run `evidence` through the full reasoning pipeline.

        Args:
            evidence: A single `Evidence` bundle, or a group of them to
                be considered together (e.g. a baseline and its
                ablations, or several seeds of the same configuration).
                Passed straight through to
                `ObservationExtractor.extract`, which accepts the same
                union and treats a bare `Evidence` as a one-element
                group.

        Returns:
            A `ReasoningResult` carrying `evidence` itself plus every
            intermediate and final stage output, in pipeline order:
            observations, hypotheses, rule findings, confidence,
            judgments, and recommendations.
        """
        observations = self._observation_extractor.extract(evidence)
        hypotheses = tuple(self._hypothesis_generator.generate(observations))
        rule_findings = tuple(self._rule_engine.evaluate(observations, hypotheses))
        confidence = self._confidence_assessor.assess(observations, rule_findings)
        judgments = tuple(self._judgment_generator.generate(rule_findings, confidence))
        recommendations = tuple(self._recommendation_generator.generate(judgments))

        return ReasoningResult(
            evidence=evidence,
            observations=observations,
            hypotheses=hypotheses,
            rule_findings=rule_findings,
            confidence=confidence,
            judgments=judgments,
            recommendations=recommendations,
        )


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of a stage output item to something
    JSON-serializable, for `ReasoningResult.to_dict()`.

    Local, minimal analog of `evidence.py`'s `_json_safe`: delegates to
    `value.to_dict()` when present (the convention every concrete type
    in this package follows) and otherwise passes `value` through
    unchanged, since stage-output types (`HypothesisT`, `RuleFindingT`,
    `ConfidenceT`, `JudgmentT`, `RecommendationT`) are not owned by this
    module and may not exist as concrete types yet.
    """
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value
