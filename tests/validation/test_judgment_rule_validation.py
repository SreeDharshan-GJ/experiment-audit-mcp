"""Scientific validation of `JudgmentRule` (R005).

R005 synthesizes a terminal `ClaimJudgment` for each claim from
already-computed upstream findings, read via the documented
`RuleContext` fields (`missing_evidence`, `detected_contradictions`)
and the `context.metadata["confidence_adjustments"]` /
`context.metadata["confidence_adjustment"]` contract R004 populates.
It never inspects raw evidence itself. The decision procedure:

- Undeclared/unspecified scope, or no resolvable confidence input ->
  ABSTAIN.
- An unresolved contradiction caps the claim below SUPPORTED,
  regardless of confidence.
- No missing-evidence/scope gap, no unresolved contradiction, and
  confidence >= 0.5 -> SUPPORTED.
- Confidence <= -0.25 (and no unresolved contradiction) -> UNSUPPORTED.
- Everything else -> PARTIALLY_SUPPORTED.
"""

from __future__ import annotations

import pytest

from experiment_audit_mcp.reasoning.claims import ClaimCategory
from experiment_audit_mcp.reasoning.contradictions import (
    Contradiction,
    ContradictionCategory,
    ContradictionStatus,
)
from experiment_audit_mcp.reasoning.scientific_rules.judgment_rule import (
    ClaimJudgment,
    JudgmentRule,
)

from .builders import claim, context, scope


@pytest.fixture
def rule() -> JudgmentRule:
    return JudgmentRule()


def test_supported_when_clean_and_confident(rule: JudgmentRule) -> None:
    """A claim with a declared scope, no missing-evidence gap, no
    contradiction, and a high (R004-computed) confidence adjustment
    (0.75, above the 0.5 threshold) has earned SUPPORTED -- the
    evidentiary record, as it currently stands, justifies the claim
    outright.
    """
    c = claim(
        "C501",
        "SparseAttn attains 91% top-1 accuracy on CIFAR-10",
        ClaimCategory.PERFORMANCE,
        claim_scope=scope(dataset="cifar-10"),
    )

    result = rule.evaluate(context(claims=[c], metadata={"confidence_adjustments": {"C501": 0.75}}))

    assert result.metadata["judgments"]["C501"] == ClaimJudgment.SUPPORTED.value
    assert result.triggered is False  # SUPPORTED is the only non-triggering judgment


def test_partially_supported_with_a_confidence_in_the_moderate_range(rule: JudgmentRule) -> None:
    """A claim with no missing-evidence gap and no contradiction, but
    a confidence adjustment (0.10) that falls in the moderate middle
    ground between the unsupported and supported thresholds, is
    neither clearly justified nor clearly unjustified --
    PARTIALLY_SUPPORTED is the scientifically honest middle verdict.
    """
    c = claim(
        "C502",
        "SparseAttn's gains are modest but present on ImageNet",
        ClaimCategory.PERFORMANCE,
        claim_scope=scope(dataset="imagenet"),
    )

    result = rule.evaluate(context(claims=[c], metadata={"confidence_adjustments": {"C502": 0.10}}))

    assert result.metadata["judgments"]["C502"] == ClaimJudgment.PARTIALLY_SUPPORTED.value
    assert result.triggered is True


def test_unsupported_with_strongly_negative_confidence(rule: JudgmentRule) -> None:
    """A claim whose confidence adjustment (-0.40) sits at or below
    the unsupported threshold, with no unresolved contradiction in
    play, is UNSUPPORTED: the currently attributed evidence does not
    justify believing it.
    """
    c = claim(
        "C503",
        "SparseAttn works well on out-of-distribution audio data",
        ClaimCategory.ROBUSTNESS,
        claim_scope=scope(dataset="audio-ood"),
    )

    result = rule.evaluate(
        context(claims=[c], metadata={"confidence_adjustments": {"C503": -0.40}})
    )

    assert result.metadata["judgments"]["C503"] == ClaimJudgment.UNSUPPORTED.value
    assert result.triggered is True


def test_abstain_when_scope_is_unspecified(rule: JudgmentRule) -> None:
    """A claim that never declared a scope at all cannot have its
    relationship to its evidence characterized -- per specification
    §8's "Unknown claim" condition, R005 abstains rather than guessing
    at an implicit default scope, regardless of how favorable the
    confidence figure might otherwise look.
    """
    c = claim(
        "C504",
        "SparseAttn is generally better",
        ClaimCategory.PERFORMANCE,
        claim_scope=scope(),  # is_unspecified() is True
    )

    result = rule.evaluate(context(claims=[c], metadata={"confidence_adjustments": {"C504": 0.9}}))

    assert result.metadata["judgments"]["C504"] == ClaimJudgment.ABSTAIN.value
    assert "scope is absent or unspecified" in result.reasoning


def test_abstain_when_no_confidence_input_is_resolvable(rule: JudgmentRule) -> None:
    """A claim with a properly declared scope, but for which R004 never
    reported a confidence figure (neither per-claim nor a context-wide
    fallback), cannot be judged -- per specification §8's "Incomplete
    metadata" condition, R005 abstains rather than assuming a neutral
    default confidence of 0.0.
    """
    c = claim(
        "C505",
        "SparseAttn accuracy on CIFAR-10",
        ClaimCategory.PERFORMANCE,
        claim_scope=scope(dataset="cifar-10"),
    )

    result = rule.evaluate(context(claims=[c], metadata={}))

    assert result.metadata["judgments"]["C505"] == ClaimJudgment.ABSTAIN.value
    assert "no" in result.reasoning and "confidence adjustment" in result.reasoning


def test_unresolved_contradiction_caps_support(rule: JudgmentRule) -> None:
    """A claim with an *unresolved* contradiction standing against it
    is never judged SUPPORTED, no matter how high its confidence
    adjustment is (0.90 here, well above the supported threshold) --
    an open, unsettled conflict in the record must not be reported
    with the same appearance as a fully settled one. Confidence still
    above the unsupported threshold, so the outcome is
    PARTIALLY_SUPPORTED, not UNSUPPORTED.
    """
    c = claim(
        "C506",
        "SparseAttn's throughput improvement on an A100",
        ClaimCategory.EFFICIENCY,
        claim_scope=scope(hardware="A100"),
    )
    counterpart = claim(
        "C506b",
        "SparseAttn's throughput improvement on an A100 (independent replication)",
        ClaimCategory.EFFICIENCY,
        claim_scope=scope(hardware="A100"),
    )
    unresolved = Contradiction(
        id="CD-020",
        categories=(ContradictionCategory.EVIDENCE,),
        status=ContradictionStatus.DETECTED,
        claims=(c, counterpart),
    )

    result = rule.evaluate(
        context(
            claims=[c],
            detected_contradictions=(unresolved,),
            metadata={"confidence_adjustments": {"C506": 0.90}},
        )
    )

    assert result.metadata["judgments"]["C506"] == ClaimJudgment.PARTIALLY_SUPPORTED.value

    # The same claim, but now with confidence collapsed to the
    # unsupported threshold as well -- the unresolved contradiction and
    # a rock-bottom confidence figure together warrant UNSUPPORTED, not
    # merely PARTIALLY_SUPPORTED.
    result_low_confidence = rule.evaluate(
        context(
            claims=[c],
            detected_contradictions=(unresolved,),
            metadata={"confidence_adjustments": {"C506": -0.30}},
        )
    )
    assert result_low_confidence.metadata["judgments"]["C506"] == ClaimJudgment.UNSUPPORTED.value


def test_applies_false_with_no_claims(rule: JudgmentRule) -> None:
    assert rule.applies(context(claims=[])) is False
