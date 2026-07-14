"""Scientific validation of `MissingEvidenceRule` (R001).

R001 checks each `Claim`'s declared `ClaimCategory` against the fixed,
per-category evidence expectations `_CATEGORY_EXPECTATIONS` declares
(missing_evidence_rule.py), and reports -- by name -- any expected
dimension the evidence in the `RuleContext` does not cover. Every check
below is a *global* check over `context.evidence_sequence()`; R001
never attributes evidence to one claim over another (that per-claim
attribution is `ContradictionRule`/`ConfidenceRule`'s concern, via
`Claim.evidence_trace`), so each scenario below supplies exactly the
evidence dimensions relevant to the one gap under test and nothing
that would incidentally satisfy it.
"""

from __future__ import annotations

import pytest

from experiment_audit_mcp.reasoning.claims import ClaimCategory
from experiment_audit_mcp.reasoning.scientific_rules.missing_evidence_rule import (
    MissingEvidenceRule,
)

from .builders import claim, context, evidence, run_ref


@pytest.fixture
def rule() -> MissingEvidenceRule:
    return MissingEvidenceRule()


def test_missing_statistical_evidence(rule: MissingEvidenceRule) -> None:
    """A STATISTICAL claim ("SparseAttn's accuracy gain is a real
    effect, not noise") needs at least two recorded seeds
    (01_evidence.md Rule 002: "only one random seed -> low statistical
    confidence"). One run, logged with a single seed, cannot
    distinguish a genuine effect from ordinary run-to-run variance --
    scientifically, this is exactly the gap R001 exists to name.
    """
    ref = run_ref("run-001")
    ev = evidence(ref, seeds=[42], accuracy=0.912)
    c = claim(
        "C001",
        "SparseAttn's accuracy gain over the dense baseline is a real effect",
        ClaimCategory.STATISTICAL,
        evidence_trace=(ref,),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 1
    assert "C001" in result.missing_evidence[0]
    assert "repeated measurement" in result.missing_evidence[0]
    assert any("at least two seeds" in r for r in result.recommendations)


def test_missing_hardware_evidence(rule: MissingEvidenceRule) -> None:
    """An EFFICIENCY claim ("SparseAttn is 2x faster at inference")
    needs both a resource-cost metric *and* the hardware it was
    measured on (01_evidence.md §3, "Contextual": a latency number is
    uninterpretable without knowing what it ran on). Here the resource
    cost metric is present (`latency_ms`) but no hardware fact was
    ever recorded, so only the hardware dimension should be reported
    missing -- the resource-cost dimension is already satisfied and
    must not be flagged twice.
    """
    ref = run_ref("run-002")
    ev = evidence(ref, latency_ms=8.4)  # no hardware= supplied
    c = claim(
        "C002",
        "SparseAttn halves inference latency versus dense attention",
        ClaimCategory.EFFICIENCY,
        evidence_trace=(ref,),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 1
    assert "Hardware specification evidence" in result.missing_evidence[0]
    # The resource-cost dimension is satisfied by latency_ms and must not
    # also be reported missing.
    assert "Resource-cost metric evidence" not in result.missing_evidence[0]


def test_missing_dataset_evidence(rule: MissingEvidenceRule) -> None:
    """A GENERALIZATION claim ("SparseAttn's gains generalize beyond
    CIFAR-10") requires evidence from more than one dataset
    (01_evidence.md's own worked example: evidence on one dataset does
    not warrant a claim the method generalizes). With evidence from
    only CIFAR-10, both of GENERALIZATION's registered expectations
    (cross-domain evaluation and additional datasets) are unmet, since
    both reduce to the same "more than one distinct dataset" check.
    """
    ref = run_ref("run-003")
    ev = evidence(ref, dataset={"name": "CIFAR-10"}, accuracy=0.89)
    c = claim(
        "C003",
        "SparseAttn's accuracy gains generalize across image datasets",
        ClaimCategory.GENERALIZATION,
        evidence_trace=(ref,),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 2
    assert all("C003" in m for m in result.missing_evidence)
    assert any("Cross-domain evaluation" in m for m in result.missing_evidence)
    assert any("Additional datasets" in m for m in result.missing_evidence)


def test_missing_reproducibility_evidence(rule: MissingEvidenceRule) -> None:
    """A REPRODUCIBILITY claim ("the published SparseAttn result can be
    independently regenerated") is never satisfied by the original run
    alone, however carefully recorded (01_evidence.md §4): it requires
    a linked, independent re-execution. With no `previous_experiments`
    on hand, this is exactly the gap R001 must report.
    """
    ref = run_ref("run-004")
    ev = evidence(ref, accuracy=0.912)  # no previous_experiments
    c = claim(
        "C004",
        "The published SparseAttn accuracy result is reproducible",
        ClaimCategory.REPRODUCIBILITY,
        evidence_trace=(ref,),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 1
    assert "independent re-execution" in result.missing_evidence[0]


def test_complete_evidence_reports_no_gap(rule: MissingEvidenceRule) -> None:
    """A PERFORMANCE claim ("SparseAttn attains 91% top-1 accuracy on
    CIFAR-10") requires only evaluation-metric evidence
    (01_evidence.md §4). A run with a recorded summary metric fully
    covers what this category expects, so R001 must not trigger and
    must report a clean bill of health for this claim.
    """
    ref = run_ref("run-005")
    ev = evidence(ref, accuracy=0.912)
    c = claim(
        "C005",
        "SparseAttn attains 91.2% top-1 accuracy on CIFAR-10",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is False
    assert result.missing_evidence == ()
    assert result.recommendations == ()
    assert "covering every dimension" in result.reasoning


def test_applies_false_with_no_claims(rule: MissingEvidenceRule) -> None:
    """With no claims to check evidence against, R001 must decline to
    run rather than reporting on evidence in the abstract -- the same
    precondition every rule in this pipeline shares.
    """
    assert rule.applies(context(evidence_bundles=[], claims=[])) is False
    assert rule.applies(context(evidence_bundles=[], claims=None)) is False
