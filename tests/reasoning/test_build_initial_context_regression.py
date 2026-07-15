"""Regression test for a third confirmed reasoning-engine bug found in
production audit, alongside the two documented in
`test_engine_confidence_wiring_regression.py`:

`ScientificReasoningPipeline.build_initial_context` -- documented as
"a thin, literal pass-through to `RuleContext`'s own constructor...
so a caller does not need to import `RuleContext` directly to start a
pipeline run" -- never passed `observations` or `hypotheses` to
`RuleContext`. Both fields are required (no default) on `RuleContext`,
so every call to `build_initial_context`, with any arguments
whatsoever, raised:

    TypeError: RuleContext.__init__() missing 2 required positional
    arguments: 'observations' and 'hypotheses'

This was not caught by any existing test because every existing test
in this suite constructs `RuleContext` directly (passing
`observations=ObservationSet()` and `hypotheses=HypothesisSet()` by
hand) rather than calling `build_initial_context` -- the one method
whose entire documented purpose is to let a caller avoid doing that.

The fix adds `observations` and `hypotheses` as optional keyword
arguments to `build_initial_context`, defaulting to empty
`ObservationSet()` / `HypothesisSet()` when not supplied, since none
of this pipeline's six rules (R001-R006) read either field from the
context -- both exist on `RuleContext` for a rule that chooses to use
them, not because this concrete pipeline requires them populated.
"""

from __future__ import annotations

from experiment_audit.reasoning.claims import Claim, ClaimCategory, ClaimSet, Scope
from experiment_audit.reasoning.hypotheses import HypothesisSet
from experiment_audit.reasoning.observations import ObservationSet
from experiment_audit.reasoning.pipeline import ScientificReasoningPipeline
from experiment_audit.reasoning.scientific_report import ScientificReport


def test_build_initial_context_does_not_crash_with_no_arguments() -> None:
    """The minimal case this method is documented to support: a caller
    who supplies nothing. Must not raise."""
    context = ScientificReasoningPipeline.build_initial_context()

    assert context.observations == ObservationSet()
    assert context.hypotheses == HypothesisSet()


def test_build_initial_context_result_is_usable_by_execute() -> None:
    """The context this method returns must actually be a valid input
    to `execute()` -- the whole point of offering this convenience
    constructor."""
    claim = Claim(
        id="c1",
        subject="model-x",
        statement="model-x achieves 95% accuracy",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(dataset="cifar-10"),
    )
    context = ScientificReasoningPipeline.build_initial_context(
        claims=ClaimSet([claim]),
        evidence=[],
    )

    pipeline_report = ScientificReasoningPipeline().execute(context)
    report = ScientificReport.from_pipeline_report(pipeline_report)

    assert isinstance(report, ScientificReport)
    assert report.to_markdown()
    assert report.to_json()


def test_build_initial_context_accepts_explicit_observations_and_hypotheses() -> None:
    """A caller who already has observations/hypotheses from an earlier
    stage must still be able to pass them through explicitly."""
    context = ScientificReasoningPipeline.build_initial_context(
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
    )

    assert context.observations == ObservationSet()
    assert context.hypotheses == HypothesisSet()
