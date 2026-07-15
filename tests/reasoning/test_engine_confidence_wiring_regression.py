"""Regression test for a confirmed reasoning-engine bug found in
production audit:

`ScientificReasoningEngine.review` (engine.py) called
`self._confidence_assessor.assess(observations, rule_findings)` --
two positional arguments -- but `confidence.py`'s real
`ConfidenceAssessor.assess` takes exactly one (`hypothesis_set`).
Engine.py's own docstring promises that "wiring the real
`ScientificReasoningEngine` requires no change to this file -- a
caller just injects the real classes"; doing exactly that raised
`TypeError: ConfidenceAssessor.assess() takes 2 positional arguments
but 3 were given` on every call to `review()`, for every input,
including the trivial empty-evidence case. This is not a corner
case: it made the real `ConfidenceAssessor` unusable with
`ScientificReasoningEngine` under any input whatsoever.

The fix changes the `ConfidenceAssessor` `Protocol` and the
`review()` call site to pass `hypotheses` (matching confidence.py's
actual, documented contract: it scores each hypothesis from the
`Observation`s/`assumptions` that hypothesis itself carries, and
never depends on `ObservationSet` or rule findings at all).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from experiment_audit.models import MetricHistory, MetricPoint, Run, RunRef
from experiment_audit.reasoning.confidence import ConfidenceAssessor
from experiment_audit.reasoning.engine import ScientificReasoningEngine
from experiment_audit.reasoning.evidence import Evidence
from experiment_audit.reasoning.hypotheses import HypothesisGenerator
from experiment_audit.reasoning.judgment import JudgmentGenerator

_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


class _NullJudgmentGenerator:
    def generate(self, confidence: Any) -> Sequence[Any]:
        del confidence
        return ()


class _NullRecommendationGenerator:
    def generate(self, judgments: Sequence[Any]) -> Sequence[Any]:
        del judgments
        return ()


def _run_ref(run_id: str) -> RunRef:
    return RunRef(backend="wandb", entity="test-team", project="proj", run_id=run_id)


def _make_engine() -> ScientificReasoningEngine:
    return ScientificReasoningEngine(
        hypothesis_generator=HypothesisGenerator(),
        confidence_assessor=ConfidenceAssessor(),
        judgment_generator=_NullJudgmentGenerator(),
        recommendation_generator=_NullRecommendationGenerator(),
    )


def _make_engine_with_real_judgment_generator() -> ScientificReasoningEngine:
    return ScientificReasoningEngine(
        hypothesis_generator=HypothesisGenerator(),
        confidence_assessor=ConfidenceAssessor(),
        judgment_generator=JudgmentGenerator(),
        recommendation_generator=_NullRecommendationGenerator(),
    )


def test_review_does_not_crash_with_real_confidence_assessor_on_empty_evidence() -> None:
    """The minimal case: no run, no metrics, nothing. Must not raise."""
    engine = _make_engine()
    ev = Evidence(ref=_run_ref("empty"))

    result = engine.review(ev)

    # Every hypothesis generated (all missing-information ones, from an
    # empty bundle) must have received a real confidence assessment.
    assert len(result.confidence) == len(result.hypotheses)
    for assessment in result.confidence:
        assert 0.0 <= assessment.score <= 1.0
    # Must still be serializable end-to-end.
    assert result.to_dict()["confidence"]


def test_review_scores_an_overfitting_hypothesis_with_real_confidence_assessor() -> None:
    """A realistic case that actually produces a hypothesis with
    supporting observations, so `score_support` / `score_contradictions`
    / `score_missing_information` all run against real data, not just
    the degenerate all-missing-information path."""
    engine = _make_engine()
    ref = _run_ref("r1")
    run = Run(
        ref=ref,
        name="r1",
        tags=[],
        status="finished",
        created_at=_CREATED_AT,
        config={},
        summary_metrics={"train_loss": 0.1, "val_loss": 0.9},
    )
    train_hist = MetricHistory(
        ref=ref,
        metric_name="train_loss",
        points=tuple(MetricPoint(step=i, value=1.0 - i * 0.1) for i in range(10)),
    )
    val_hist = MetricHistory(
        ref=ref,
        metric_name="val_loss",
        points=tuple(MetricPoint(step=i, value=0.1 + i * 0.1) for i in range(10)),
    )
    ev = Evidence(
        ref=ref,
        run=run,
        metric_histories={"train_loss": train_hist, "val_loss": val_hist},
    )

    result = engine.review(ev)

    assert len(result.hypotheses) >= 1
    assert len(result.confidence) == len(result.hypotheses)
    kinds = {a.hypothesis.kind.value for a in result.confidence}
    assert "possible_overfitting" in kinds


def test_review_is_deterministic_across_repeated_calls() -> None:
    """Confidence is computed, never guessed: identical evidence must
    yield bit-identical scores across repeated `review()` calls."""
    engine = _make_engine()
    ev = Evidence(ref=_run_ref("repeat-me"))

    first = [round(a.score, 12) for a in engine.review(ev).confidence]
    second = [round(a.score, 12) for a in engine.review(ev).confidence]

    assert first == second


def test_review_does_not_crash_with_real_judgment_generator_on_empty_evidence() -> None:
    """Companion regression test for the same class of bug this file
    documents for `ConfidenceAssessor`: `engine.py`'s `review()` called
    `self._judgment_generator.generate(rule_findings, confidence)` --
    two positional arguments -- but `judgment.py`'s real
    `JudgmentGenerator.generate` takes exactly one (`confidence_set`).
    This was not caught by the original confidence-wiring fix because
    that fix's own test double (`_NullJudgmentGenerator`) still stubbed
    out the two-argument signature instead of the real class's actual
    one-argument contract. Wiring the real `JudgmentGenerator` in,
    exactly as this test does, is what exposes it."""
    engine = _make_engine_with_real_judgment_generator()
    ev = Evidence(ref=_run_ref("empty"))

    result = engine.review(ev)

    assert len(result.judgments) <= len(result.confidence)
    assert result.to_dict()["judgments"] is not None
