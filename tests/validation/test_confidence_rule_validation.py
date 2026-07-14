"""Scientific validation of `ConfidenceRule` (R004).

R004 aggregates, independently per claim: four strengthening factors
computed directly from the `Evidence` bundles a claim's own
`evidence_trace` attributes to it (reproducibility, statistical
support, independent sources, traceability -- each worth +0.25), and
two reducing factors read from already-computed upstream findings
(missing-evidence/scope gaps at -0.20 each, capped at -0.60; unresolved
contradictions at -0.35 each and resolved ones at -0.05 each, capped at
-0.60). The result is reported per claim in
`RuleResult.metadata["confidence_adjustments"]`, and as their mean in
`RuleResult.confidence_adjustment`.
"""

from __future__ import annotations

import pytest

from experiment_audit_mcp.reasoning.claims import ClaimCategory
from experiment_audit_mcp.reasoning.contradictions import (
    Contradiction,
    ContradictionCategory,
    ContradictionStatus,
)
from experiment_audit_mcp.reasoning.scientific_rules.confidence_rule import ConfidenceRule

from .builders import claim, context, evidence, run_ref


@pytest.fixture
def rule() -> ConfidenceRule:
    return ConfidenceRule()


def test_strong_evidence_yields_maximum_adjustment(rule: ConfidenceRule) -> None:
    """A claim backed by two independent runs (>=2 distinct sources),
    each contributing a seed (>=2 seeds total, statistical support), a
    linked prior experiment (reproducibility), and a fully resolvable
    evidence trace (traceability) satisfies all four strengthening
    factors and none of the reducing ones -- the strongest evidentiary
    picture this rule recognizes, and it must report the maximum
    adjustment of 1.0.
    """
    prior_ref = run_ref("run-390")
    prior = evidence(prior_ref, accuracy=0.90)

    ref_a = run_ref("run-401")
    ref_b = run_ref("run-402")
    ev_a = evidence(ref_a, seeds=[1], previous_experiments=[prior], accuracy=0.93)
    ev_b = evidence(ref_b, seeds=[2], accuracy=0.94)

    c = claim(
        "C401",
        "SparseAttn's accuracy gain is real and reproducible",
        ClaimCategory.STATISTICAL,
        evidence_trace=(ref_a, ref_b),
    )

    result = rule.evaluate(context(evidence_bundles=[ev_a, ev_b], claims=[c]))

    assert result.triggered is True
    assert result.metadata["confidence_adjustments"]["C401"] == pytest.approx(1.0)
    assert result.confidence_adjustment == pytest.approx(1.0)


def test_weak_evidence_yields_low_or_negative_adjustment(rule: ConfidenceRule) -> None:
    """A claim with no declared evidence trace at all earns none of
    the four strengthening factors (no reproducibility, no statistical
    support, no independent sources, and -- critically -- no
    traceability, since traceability requires a non-empty trace that
    resolves). Combined with one already-identified missing-evidence
    gap attributed to it, the claim's confidence is driven negative:
    a scientifically correct outcome for a claim with essentially no
    evidentiary backing.
    """
    ref = run_ref("run-403")
    ev = evidence(ref, accuracy=0.5)
    c = claim(
        "C403",
        "SparseAttn works well in general",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(),  # deliberately untraceable
    )
    gap = "Claim C403 ('SparseAttn works well in general'): Evaluation metric evidence"

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c], missing_evidence=(gap,)))

    assert result.metadata["confidence_adjustments"]["C403"] == pytest.approx(-0.20)
    reasoning_403 = result.metadata["confidence_reasoning"]["C403"]
    assert "no reproducibility evidence is present" in reasoning_403
    assert (
        "its evidentiary chain is not fully traceable"
        in (result.metadata["confidence_reasoning"]["C403"])
    )


def test_contradictory_evidence_reduces_confidence(rule: ConfidenceRule) -> None:
    """A claim named by an unresolved `Contradiction` incurs R004's
    heavier "conflicting evidence" penalty (-0.35), reflecting
    07_confidence.md §4's statement that conflicting evidence reduces
    confidence "more sharply than an equivalent quantity of merely
    absent evidence." Isolated here with no other strengthening or
    reducing factor in play.
    """
    ref = run_ref("run-404")
    ev = evidence(ref, accuracy=0.9)
    c = claim(
        "C404",
        "Model X accuracy on Benchmark-B",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
    )
    counterpart = claim(
        "C404b",
        "Model X accuracy on Benchmark-B (independent replication)",
        ClaimCategory.PERFORMANCE,
    )
    unresolved = Contradiction(
        id="CD-010",
        categories=(ContradictionCategory.EVIDENCE,),
        status=ContradictionStatus.DETECTED,
        claims=(c, counterpart),
    )

    result = rule.evaluate(
        context(evidence_bundles=[ev], claims=[c], detected_contradictions=(unresolved,))
    )

    # +0.25 traceability (trace is non-empty and resolves) - 0.35 unresolved
    # contradiction penalty = -0.10.
    assert result.metadata["confidence_adjustments"]["C404"] == pytest.approx(-0.10)
    assert (
        "1 unresolved and 0 resolved contradiction(s)"
        in (result.metadata["confidence_reasoning"]["C404"])
    )


def test_missing_evidence_penalty_is_capped(rule: ConfidenceRule) -> None:
    """Four separately-identified missing-evidence gaps attributed to
    the same claim would, uncapped, cost 4 x 0.20 = 0.80 -- but
    07_confidence.md §5 is explicit that a claim should not be driven
    arbitrarily far below others merely for accumulating many small,
    already-known gaps, so this rule caps the missing-evidence penalty
    at 0.60.
    """
    ref = run_ref("run-405")
    ev = evidence(ref, accuracy=0.5)
    c = claim(
        "C405",
        "Model X generalizes across four benchmarks",
        ClaimCategory.GENERALIZATION,
        evidence_trace=(),
    )
    gaps = tuple(
        f"Claim C405 ('Model X generalizes across four benchmarks'): missing dimension {i}"
        for i in range(4)
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c], missing_evidence=gaps))

    # 0 strengthening factors (untraceable) - min(0.60, 0.20*4) = -0.60.
    assert result.metadata["confidence_adjustments"]["C405"] == pytest.approx(-0.60)


def test_multiple_independent_sources_strengthens_confidence(rule: ConfidenceRule) -> None:
    """A claim whose evidence spans two distinct runs (rather than
    many data points drawn from one run) satisfies "independent
    evidence" (07_confidence.md §3): the two sources did not share a
    single point of possible failure. Isolated here alongside the
    traceability factor that automatically accompanies any resolvable,
    non-empty trace.
    """
    ref_a = run_ref("run-406")
    ref_b = run_ref("run-407")
    ev_a = evidence(ref_a, accuracy=0.9)
    ev_b = evidence(ref_b, accuracy=0.91)
    c = claim(
        "C406",
        "SparseAttn's accuracy is consistent across independent runs",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref_a, ref_b),
    )

    result = rule.evaluate(context(evidence_bundles=[ev_a, ev_b], claims=[c]))

    # +0.25 independent sources + 0.25 traceability = 0.50.
    assert result.metadata["confidence_adjustments"]["C406"] == pytest.approx(0.50)
    assert (
        "evidence spans more than one independent source"
        in (result.metadata["confidence_reasoning"]["C406"])
    )


def test_reproducibility_evidence_strengthens_confidence(rule: ConfidenceRule) -> None:
    """A claim whose attributed evidence links to a prior, independent
    experiment satisfies 07_confidence.md §3's "reproducibility"
    factor -- a documented, independent attempt to regenerate the
    result -- isolated here alongside traceability.
    """
    prior_ref = run_ref("run-408-prior")
    prior = evidence(prior_ref, accuracy=0.90)
    ref = run_ref("run-408")
    ev = evidence(ref, previous_experiments=[prior], accuracy=0.91)
    c = claim(
        "C408",
        "SparseAttn's result has been independently regenerated",
        ClaimCategory.REPRODUCIBILITY,
        evidence_trace=(ref,),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    # +0.25 reproducibility + 0.25 traceability = 0.50.
    assert result.metadata["confidence_adjustments"]["C408"] == pytest.approx(0.50)
    assert (
        "reproducibility evidence is present" in (result.metadata["confidence_reasoning"]["C408"])
    )


def test_applies_false_with_no_claims(rule: ConfidenceRule) -> None:
    assert rule.applies(context(evidence_bundles=[], claims=[])) is False
