"""Regression tests for two confirmed reasoning-engine bugs found in
production audit:

1. `ScientificReport.to_dict()` (pipeline.py) crashed with
   `AttributeError: 'str' object has no attribute 'to_dict'` whenever
   `MissingEvidenceRule` (R001) or `ScopeRule` (R002) actually
   triggered, because `RuleResult.missing_evidence` is `tuple[str, ...]`
   while `RuleContext.missing_evidence` is typed/documented as
   `tuple[MissingEvidenceRecord, ...]`, and `ScientificReasoningPipeline
   ._advance_context` merges the former straight into the latter with
   no conversion. `RuleContext.to_dict()` unconditionally called
   `record.to_dict()` on every entry, which a plain `str` does not have.

2. `ConfidenceRule` (R004), `JudgmentRule` (R005), and
   `RecommendationRule` (R006) never see a contradiction that
   `ContradictionRule` (R003) detects *during the same pipeline run*:
   `ScientificReasoningPipeline._advance_context` has no branch for
   `rule_id == "R003"`, so `RuleContext.detected_contradictions` after
   the pipeline runs is identical to what it was before R003 ran. A
   claim can have a freshly-detected, structural, within-claim or
   cross-claim contradiction and still receive a confidence adjustment,
   judgment, and recommendation computed as if no contradiction existed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from experiment_audit_mcp.models import Run, RunRef
from experiment_audit_mcp.reasoning.claims import Claim, ClaimCategory, ClaimSet, Scope
from experiment_audit_mcp.reasoning.contradictions import (
    Contradiction,
    ContradictionCategory,
)
from experiment_audit_mcp.reasoning.evidence import Evidence
from experiment_audit_mcp.reasoning.hypotheses import HypothesisSet
from experiment_audit_mcp.reasoning.observations import ObservationSet
from experiment_audit_mcp.reasoning.pipeline import ScientificReasoningPipeline
from experiment_audit_mcp.reasoning.rules import RuleContext

_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _run_ref(run_id: str) -> RunRef:
    return RunRef(backend="wandb", entity="test-team", project="proj", run_id=run_id)


def _run(ref: RunRef, **summary_metrics: float) -> Run:
    return Run(
        ref=ref,
        name=ref.run_id,
        tags=[],
        status="finished",
        created_at=_CREATED_AT,
        config={},
        summary_metrics=summary_metrics,
    )


def test_report_to_dict_does_not_crash_when_missing_evidence_rule_triggers() -> None:
    """Regression test for bug #1.

    A `COMPARISON` claim with no `PREVIOUS_EXPERIMENT` evidence
    deterministically triggers `MissingEvidenceRule`, which is enough
    to reproduce the crash: before the fix, `report.to_dict()` raised
    `AttributeError` here every time.
    """
    ref = _run_ref("run-001")
    evidence = Evidence(ref=ref, run=_run(ref, accuracy=0.9))
    claim = Claim(
        id="C1",
        subject="run-001",
        statement="Model beats baseline.",
        category=ClaimCategory.COMPARISON,
        scope=Scope(),
        evidence_trace=(ref,),
    )
    context = RuleContext(
        evidence=[evidence],
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet([claim]),
    )

    report = ScientificReasoningPipeline().execute(context)

    # Precondition: the bug can only manifest if R001 actually reported a gap.
    assert report.context.missing_evidence, (
        "Test fixture must trigger MissingEvidenceRule for this regression test to be meaningful."
    )

    serialized = report.to_dict()  # must not raise AttributeError

    assert serialized["context"]["missing_evidence"] == [
        str(entry) for entry in report.context.missing_evidence
    ]


def test_contradiction_detected_this_pass_reaches_confidence_rule() -> None:
    """Regression test for bug #2 (fixed).

    Two claims each cite the same two runs, whose `hardware.latency_ms`
    disagrees (42 vs. 87 for the same normalized fact) -- a within-claim
    structural contradiction for both, per `ContradictionRule`. Before
    the fix, `ConfidenceRule`'s per-claim adjustment never reflected
    this, because `ContradictionRule`'s (R003) same-pass findings never
    reached `RuleContext.detected_contradictions` or anywhere else
    `ConfidenceRule` reads from -- `pipeline.py`'s `_advance_context` had
    no branch for `rule_id == "R003"` at all. This test locks in the
    fix: R003's same-pass findings now reach `ConfidenceRule` via
    `RuleContext.metadata["newly_detected_contradictions"]`.
    """
    ref_a = _run_ref("run-a")
    ref_b = _run_ref("run-b")
    evidence = [
        Evidence(ref=ref_a, run=_run(ref_a, accuracy=0.9), hardware={"latency_ms": 42}),
        Evidence(ref=ref_b, run=_run(ref_b, accuracy=0.9), hardware={"latency_ms": 87}),
    ]
    claim_a = Claim(
        id="A",
        subject="shared-subject",
        statement="Runtime is 42ms.",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(),
        evidence_trace=(ref_a, ref_b),
    )
    claim_b = Claim(
        id="B",
        subject="shared-subject",
        statement="Runtime is 87ms.",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(),
        evidence_trace=(ref_a, ref_b),
    )
    context = RuleContext(
        evidence=evidence,
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet([claim_a, claim_b]),
    )

    report = ScientificReasoningPipeline().execute(context)

    r003_results = [r for r in report.results if r.rule_id == "R003"]
    assert r003_results and r003_results[0].triggered, (
        "Test fixture must trigger ContradictionRule for this regression test to be meaningful."
    )

    # The fix: R003's same-pass findings are forwarded into metadata ...
    forwarded = report.context.metadata.get("newly_detected_contradictions")
    assert forwarded, "R003's same-pass findings must reach RuleContext.metadata."

    # ... and never double-count anything already carried forward via
    # RuleContext.detected_contradictions (there was nothing carried
    # forward in this fixture, but the exclusion itself is what this
    # asserts the shape of: no forwarded string should ever start with
    # the carried-forward marker prefix).
    assert not any(text.startswith("Contradiction ") for text in forwarded)

    # ... and ConfidenceRule's per-claim adjustment for BOTH claims now
    # reflects a contradiction penalty, not just the pre-existing
    # strengthening factors.
    r004_results = [r for r in report.results if r.rule_id == "R004"]
    assert r004_results and r004_results[0].triggered
    reasoning_by_claim = r004_results[0].metadata.get("confidence_reasoning", {})
    for claim_id in ("A", "B"):
        assert "1 unresolved and 0 resolved contradiction(s)" in reasoning_by_claim[claim_id], (
            f"Claim {claim_id}'s confidence reasoning should now count the "
            "same-pass contradiction R003 detected, instead of silently "
            "ignoring it."
        )


def test_cross_claim_contradiction_penalizes_both_claims_not_just_the_first() -> None:
    """Regression test for a bug introduced, then caught and fixed,
    while fixing bug #2.

    `ContradictionRule`'s actual cross-claim finding text is `"Claim
    {a} (...) and claim {b} (...) record conflicting values ..."` --
    which itself begins with the same `"Claim {id} ("` prefix the
    within-claim finding format uses. An initial fix that checked the
    within-claim pattern first silently captured only the first claim
    id (`claim_a`) and dropped the second (`claim_b`) entirely, so only
    one side of a two-claim contradiction ever received a confidence
    penalty. This test locks in that both claims in a genuine
    cross-claim contradiction are penalized.
    """
    ref_a = _run_ref("run-001")
    ref_b = _run_ref("run-002")
    evidence = [
        Evidence(ref=ref_a, run=_run(ref_a, latency_ms=42.0)),
        Evidence(ref=ref_b, run=_run(ref_b, latency_ms=87.0)),
    ]
    claim_a = Claim(
        id="C004",
        subject="SparseAttn inference latency on Benchmark-B",
        statement="Latency is 42ms.",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(dataset="benchmark-b", hardware="A100"),
        evidence_trace=(ref_a,),
    )
    claim_b = Claim(
        id="C005",
        subject="SparseAttn inference latency on Benchmark-B",
        statement="Latency is 87ms.",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(dataset="benchmark-b", hardware="A100"),
        evidence_trace=(ref_b,),
    )
    context = RuleContext(
        evidence=evidence,
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet([claim_a, claim_b]),
    )

    report = ScientificReasoningPipeline().execute(context)

    r003_results = [r for r in report.results if r.rule_id == "R003"]
    assert r003_results and r003_results[0].triggered

    r004_results = [r for r in report.results if r.rule_id == "R004"]
    reasoning_by_claim = r004_results[0].metadata.get("confidence_reasoning", {})
    for claim_id in ("C004", "C005"):
        assert "1 unresolved and 0 resolved contradiction(s)" in reasoning_by_claim[claim_id], (
            f"Claim {claim_id} must be penalized -- a cross-claim contradiction "
            "must count against both parties, never only the first-mentioned one."
        )


def test_carried_forward_contradiction_is_not_double_counted() -> None:
    """A contradiction already known before the pipeline runs (supplied
    via `RuleContext.detected_contradictions`) is surfaced again by
    `ContradictionRule` as a "carried forward" finding in its
    `RuleResult.contradictions` (see `contradiction_rule.py`'s "Known,
    carried-forward contradictions" check) -- but it must be counted by
    `ConfidenceRule` exactly once, via `RuleContext.detected_contradictions`
    itself, never a second time via the new
    `RuleContext.metadata["newly_detected_contradictions"]` channel this
    fix adds. `pipeline.py` guards against this by never forwarding a
    finding whose text starts with the literal `"Contradiction "` marker.
    """
    ref = _run_ref("run-001")
    evidence = [Evidence(ref=ref, run=_run(ref, accuracy=0.9))]
    claim = Claim(
        id="C1",
        subject="run-001",
        statement="Model achieves 0.9 accuracy.",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(),
        evidence_trace=(ref,),
    )
    # A Contradiction must name at least two parties (Section 2); the
    # second claim here need not be in this context's ClaimSet at all
    # -- only that the *known* contradiction names `claim` is relevant
    # to what this test is checking.
    other_claim = Claim(
        id="C2",
        subject="run-002",
        statement="A different, unrelated claim.",
        category=ClaimCategory.PERFORMANCE,
        scope=Scope(),
        evidence_trace=(),
    )
    known_contradiction = Contradiction(
        id="K1",
        categories=(ContradictionCategory.CLAIM,),
        claims=(claim, other_claim),
        evidence_items=(),
    )
    context = RuleContext(
        evidence=evidence,
        observations=ObservationSet(),
        hypotheses=HypothesisSet(),
        claims=ClaimSet([claim]),
        detected_contradictions=(known_contradiction,),
    )

    report = ScientificReasoningPipeline().execute(context)

    r003_results = [r for r in report.results if r.rule_id == "R003"]
    assert r003_results and r003_results[0].triggered
    assert any(text.startswith("Contradiction K1") for text in r003_results[0].contradictions), (
        "Test fixture must actually produce a carried-forward finding."
    )

    forwarded = report.context.metadata.get("newly_detected_contradictions", ())
    assert not any(text.startswith("Contradiction ") for text in forwarded), (
        "The carried-forward finding must never be forwarded into the "
        "same-pass metadata channel -- it is already counted via "
        "RuleContext.detected_contradictions."
    )

    r004_results = [r for r in report.results if r.rule_id == "R004"]
    reasoning = r004_results[0].metadata.get("confidence_reasoning", {})["C1"]
    assert "1 unresolved and 0 resolved contradiction(s)" in reasoning, (
        "The known contradiction must be counted exactly once, not twice."
    )
