"""Scientific validation of `ScopeRule` (R002).

R002 checks a claim's own *declared* `Scope` (claims.py) against the
values the evidence in the `RuleContext` actually records for each
declared dimension (dataset, hardware, model, evaluation_protocol,
software_environment). A **scope violation**, per 02_claims.md §5, is
evidence that is narrower than, or conflicts with, the scope a claim
declares -- e.g. a claim that says "on A100" when its evidence was
gathered on a V100. Like R001, every comparison here is global over
`context.evidence_sequence()`, not attributed via `Claim.evidence_trace`
(scope.py never reads that field).
"""

from __future__ import annotations

import pytest

from experiment_audit_mcp.reasoning.claims import ClaimCategory
from experiment_audit_mcp.reasoning.scientific_rules.scope_rule import ScopeRule

from .builders import claim, context, evidence, run_ref, scope


@pytest.fixture
def rule() -> ScopeRule:
    return ScopeRule()


def test_correct_scope_reports_no_violation(rule: ScopeRule) -> None:
    """A claim that declares exactly the conditions its evidence was
    gathered under (CIFAR-10, on an A100) is, by definition, not
    exceeding its own scope -- R002 must find nothing to report.
    """
    ref = run_ref("run-101")
    ev = evidence(ref, dataset={"name": "cifar-10"}, hardware={"name": "A100"}, accuracy=0.91)
    c = claim(
        "C101",
        "SparseAttn accuracy on CIFAR-10 using an A100",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
        claim_scope=scope(dataset="cifar-10", hardware="A100"),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is False
    assert result.missing_evidence == ()
    assert "fully supported at its declared scope" in result.reasoning


def test_hardware_mismatch_is_a_violation(rule: ScopeRule) -> None:
    """A claim declaring its latency was measured on an A100, when the
    evidence attributed to the run actually records a V100, is a
    scope violation: the claim asserts a condition the evidence does
    not attest to, regardless of how good the underlying measurement
    otherwise is.
    """
    ref = run_ref("run-102")
    ev = evidence(ref, hardware={"name": "V100"}, latency_ms=12.3)
    c = claim(
        "C102",
        "SparseAttn inference latency on an A100",
        ClaimCategory.EFFICIENCY,
        evidence_trace=(ref,),
        claim_scope=scope(hardware="A100"),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 1
    assert "hardware of 'A100'" in result.missing_evidence[0]
    assert "v100" in result.missing_evidence[0].lower()


def test_dataset_mismatch_is_a_violation(rule: ScopeRule) -> None:
    """A claim declaring CIFAR-10 as its scope, backed only by
    ImageNet evidence, is a dataset scope violation -- the claim's own
    declared condition and the evidence's recorded condition disagree.
    """
    ref = run_ref("run-103")
    ev = evidence(ref, dataset={"name": "imagenet"}, accuracy=0.77)
    c = claim(
        "C103",
        "SparseAttn accuracy on CIFAR-10",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
        claim_scope=scope(dataset="cifar-10"),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 1
    assert "dataset of 'cifar-10'" in result.missing_evidence[0]
    assert "imagenet" in result.missing_evidence[0].lower()


def test_model_mismatch_is_a_violation(rule: ScopeRule) -> None:
    """`model` is not a dedicated `Evidence` bundle attribute; it is
    resolved from a run's own recorded config (`run.config["model"]`).
    A claim that declares "resnet50" but whose evidence was recorded
    against a run configured with "resnet101" is exactly the kind of
    conflict R002's generic keyed-value lookup exists to catch.
    """
    ref = run_ref("run-104")
    ev = evidence(ref, config={"model": "resnet101"}, accuracy=0.88)
    c = claim(
        "C104",
        "resnet50's accuracy on CIFAR-10",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
        claim_scope=scope(model="resnet50"),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 1
    assert "model of 'resnet50'" in result.missing_evidence[0]
    assert "resnet101" in result.missing_evidence[0].lower()


def test_benchmark_evaluation_protocol_mismatch_is_a_violation(rule: ScopeRule) -> None:
    """A claim declaring a "zero-shot" evaluation protocol (the
    benchmark it was actually run under), whose evidence's recorded
    config shows "few-shot", cannot be evaluated as though it holds
    under the declared benchmark protocol.
    """
    ref = run_ref("run-105")
    ev = evidence(ref, config={"evaluation_protocol": "few-shot"}, accuracy=0.63)
    c = claim(
        "C105",
        "SparseAttn's zero-shot accuracy on the benchmark",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
        claim_scope=scope(evaluation_protocol="zero-shot"),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 1
    assert "evaluation_protocol of 'zero-shot'" in result.missing_evidence[0]
    assert "few-shot" in result.missing_evidence[0].lower()


def test_multiple_simultaneous_mismatches_are_all_reported(rule: ScopeRule) -> None:
    """A claim that overstates its scope on three independent
    dimensions at once (hardware, dataset, and model) must have all
    three violations reported -- R002 never stops at the first
    mismatch it finds for a claim.
    """
    ref = run_ref("run-106")
    ev = evidence(
        ref,
        dataset={"name": "imagenet"},
        hardware={"name": "V100"},
        config={"model": "resnet101"},
        accuracy=0.81,
    )
    c = claim(
        "C106",
        "resnet50's CIFAR-10 accuracy on an A100",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
        claim_scope=scope(dataset="cifar-10", hardware="A100", model="resnet50"),
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is True
    assert len(result.missing_evidence) == 3
    joined = " ".join(result.missing_evidence)
    assert "dataset of 'cifar-10'" in joined
    assert "hardware of 'A100'" in joined
    assert "model of 'resnet50'" in joined


def test_unspecified_scope_is_never_checked(rule: ScopeRule) -> None:
    """A claim that declares no scope at all cannot be checked for a
    violation -- per specification §8, treating an undeclared scope
    as though some default applied would itself violate Evidence
    First. R002 must report this as a claim it could not evaluate,
    never as either a pass or a violation.
    """
    ref = run_ref("run-107")
    ev = evidence(ref, accuracy=0.5)
    c = claim(
        "C107",
        "SparseAttn accuracy",
        ClaimCategory.PERFORMANCE,
        evidence_trace=(ref,),
        claim_scope=scope(),  # is_unspecified() is True
    )

    result = rule.evaluate(context(evidence_bundles=[ev], claims=[c]))

    assert result.triggered is False
    assert "scope ambiguity" in result.reasoning
