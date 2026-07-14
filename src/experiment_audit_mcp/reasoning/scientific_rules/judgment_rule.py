"""
Experiment Audit Scientific Reasoning Engine

Module: scientific_rules.judgment_rule

Defines `JudgmentRule`, the fifth concrete `ScientificRule` (rules.py)
in this pipeline. It implements the synthesis step described in the
specification's Section 4 ("Judgment") and Section 10 ("Judgment
Generation"): given everything the four prior rules --
`MissingEvidenceRule` (R001), `ScopeRule` (R002), `ContradictionRule`
(R003), and `ConfidenceRule` (R004) -- have already, and independently,
concluded about a claim, decide the claim's current scientific status
and report that decision as a traceable `OutputCategory.JUDGMENT`
finding.

**Why this rule exists.** Every rule that precedes it in this pipeline
answers one narrow, structural question and stops there: whether
evidence a claim's category expects is absent (R001), whether the
evidence actually attributed to a claim matches its own declared scope
(R002), whether two or more claims or evidence items genuinely conflict
(R003), and how strongly the combination of those findings, together
with the claim's own positive evidentiary factors, justifies believing
it (R004). None of those four rules ever states what the *claim*
itself currently stands as, scientifically -- each produces a
finding, not a verdict. `JudgmentRule` exists to close that gap: it is
the first, and only, rule in this pipeline whose entire job is to look
at everything already found and say, plainly, what a claim's current
standing is. Per the specification's own ordering of output categories
("ordered by consequence ... through Judgment and Recommendation at the
most consequential end"), a judgment is deliberately the single most
consequential thing this pipeline says about a claim short of a
recommendation for action -- which is exactly why it is reserved for
this one rule, run last, rather than folded into any of R001-R004.

**Why judgment is not confidence.** `ConfidenceRule` (R004) answers "how
strongly is this claim justified" with a signed, continuous number in
`[-1.0, 1.0]` -- a measurement, not a verdict. `JudgmentRule` answers a
different question: "given that measurement, together with the
*specific kind* of finding that produced it (an evidentiary gap, an
unresolved conflict, a fully clean record), what is the claim's current
standing." The distinction matters because two claims can share an
identical confidence adjustment for entirely different reasons -- one
because its evidence is thin but uncontested, another because its
evidence is rich but actively disputed by an unresolved contradiction --
and this rule treats those two cases differently (see "Unresolved
contradictions cap support" below) precisely because a bare number
cannot. Judgment is also not truth: a `ClaimJudgment.SUPPORTED` verdict
states that the evidentiary record, as currently assembled, justifies
the claim -- it does not assert the claim is correct, and a `SUPPORTED`
claim can be revised the moment new evidence, a new contradiction, or a
new scope check changes what R001-R004 report. Judgment is likewise not
a publication or acceptance decision -- what should happen as a
consequence of a judgment is `recommendation.py`'s job (specification,
Section 10: "Recommendation Engine ... expected to consume Judgment
outputs"), never this rule's.

**Why judgment is downstream of every other rule.** A judgment that
synthesizes R001-R004's findings can only be as sound as those findings
themselves; producing one before they exist, or by re-deriving what
they already computed, would silently reintroduce exactly the kind of
untraceable, unregenerable reasoning the specification's Section 1
identifies as unacceptable ("a conclusion that cannot be independently
regenerated from its stated inputs has not ... been shown to follow
from those inputs at all"). This rule therefore performs no scientific
detection of its own: it never re-counts which evidence dimensions a
claim's category expects (R001's job), never re-compares a claim's
declared scope against its attributed evidence (R002's job), never
re-detects a conflict between claims or evidence (R003's job), and
never re-weighs strengthening or reducing factors into a new confidence
number (R004's job). Every one of those questions is closed by the time
this rule runs; this rule only reads the closed answers.

**Upstream contract for confidence, documented because `rules.py` does
not yet field it.** `RuleContext` (rules.py) already carries dedicated
fields this rule reads directly for two of its three required inputs:
`RuleContext.missing_evidence` (populated by R001 and R002 alike --
`scope.py`'s own docstring is explicit that a scope violation "is
reported ... under `OutputCategory.MISSING_EVIDENCE`, the same category
`MissingEvidenceRule` uses," so this single field already carries both
rules' combined findings) and `RuleContext.detected_contradictions`
(populated by R003). Neither `RuleContext` nor any sibling module in
this package defines a field carrying R004's per-claim confidence
adjustment, however -- `RuleContext.confidence` is typed as an optional
`ConfidenceSet` from `confidence.py`'s older, `Hypothesis`-keyed design,
not `ConfidenceRule`'s per-`Claim` output. Rather than block on that
field being added (which would require modifying `rules.py`, outside
this task's remit) or silently reach into `confidence.py`'s mismatched
type, this module is written *against* the following documented
contract, using the one channel `rules.py` already reserves for exactly
this situation -- `RuleContext.metadata`, "free-form, caller-supplied
context that does not fit any of the above":

- `context.metadata["confidence_adjustments"]`, if present, is a
  mapping from `Claim.id` to that claim's own `float` confidence
  adjustment, as computed by `ConfidenceRule` for that specific claim.
  This is the preferred, most precise source.
- `context.metadata["confidence_adjustment"]`, if present, is a single
  `float` -- `ConfidenceRule`'s current `RuleResult.confidence_adjustment`,
  which that rule computes as one aggregate value averaged across every
  claim in its own `RuleContext` (see `confidence_rule.py`'s
  `ConfidenceRule.evaluate`). When no per-claim mapping is available,
  this rule falls back to applying that single aggregate value
  uniformly to every claim, and says so explicitly in each affected
  claim's reported reasoning, so a reader is never left thinking a
  claim-specific confidence figure was used when it was not.
- If neither key is present, this rule treats confidence as genuinely
  unavailable for every claim it cannot otherwise resolve, and abstains
  for each of them (see "Abstain conditions" below) rather than
  assuming a default confidence of any kind -- consistent with the
  specification's Section 2 restriction that a rule never invents a
  value not traceable to an existing input.

Only `_extract_confidence_inputs` and `_confidence_for_claim` below
depend on this contract; if `rules.py` is later revised to field R004's
per-claim output directly, only those two functions need to change.
Because `context.metadata` is untyped (`Mapping[str, Any]`,
free-form and caller-supplied), `_extract_confidence_inputs` also
guards both keys with `_is_usable_confidence_number` before trusting
them: a malformed entry (wrong type, a `bool`, or a non-finite
`float`) is treated as though that entry were absent, for that key or
claim alone, rather than raising or silently miscomparing it against
this rule's thresholds. This is defensive input handling at this
rule's own boundary, not a re-validation of `ConfidenceRule`'s output
contract, which this rule still trusts fully once a value has passed
that check.

**Abstain conditions.** Per the specification's Section 7 ("Abstain
When Necessary") and Section 8 ("Rule Failures"), this rule abstains --
reports `ClaimJudgment.ABSTAIN` -- for a claim, rather than guessing,
whenever:

- The claim's declared `Scope` is absent or `Scope.is_unspecified()` is
  `True` (Section 8's "Unknown claim": a claim whose scope was never
  declared cannot have any relationship to its evidence characterized,
  including by this rule, without first resolving that ambiguity --
  the same precondition `ScopeRule` itself already declines to check
  past).
- No confidence input can be resolved for the claim at all, per the
  upstream contract above (Section 8's "Incomplete metadata": the
  metadata this rule's procedure requires is not present among its
  inputs).

**Unresolved contradictions cap support.** A claim named by at least
one *unresolved* `Contradiction` (`RuleContext.detected_contradictions`)
is never judged `SUPPORTED` by this rule, regardless of how high its
confidence adjustment is. An unresolved contradiction is, by Chapter
8's own definition, a live, unsettled conflict in the evidentiary
record; a claim standing against that kind of open conflict has not yet
had the conflict resolved in its favor, and reporting it as fully
supported would present an unsettled state with the same appearance as
a settled one -- precisely the failure mode Section 8 warns against. A
*resolved* contradiction is not treated as an independent penalty here:
its cost is already reflected in the confidence adjustment R004
computed for the claim (confidence_rule.py's own `_RESOLVED_CONTRADICTION_PENALTY`),
and applying a second, separate penalty for the same finding here would
duplicate R004's own analysis rather than synthesize it.

**Determinism.** Every judgment this rule reaches is a fixed
comparison of already-computed inputs -- claim scope declaration,
missing-evidence gap counts, unresolved/resolved contradiction counts,
and a resolved confidence figure -- against two fixed numeric
thresholds (`_SUPPORTED_CONFIDENCE_THRESHOLD`,
`_UNSUPPORTED_CONFIDENCE_THRESHOLD`). None of it is probabilistic, none
of it calls out to a model, and re-running this rule against an
unchanged `RuleContext` (and unchanged `metadata` confidence inputs)
always yields the same `RuleResult`.

**Architectural constraint.** This module depends on `rules.py` (the
framework it plugs into), `claims.py`'s `Claim` (to read a claim's `id`,
`subject`, and declared `Scope` -- the same claims-stage data every
other rule in this package already depends on), and
`contradictions.py`'s `Contradiction` (to read `.claims` and
`.is_resolved` from `RuleContext.detected_contradictions`, the same
attributes `confidence_rule.py` itself already reads from that type).
It has no dependency on `evidence.py` -- this rule never inspects
`Evidence` or `EvidenceItem`, directly or via `RuleContext.evidence_items()`,
per this task's explicit "MUST NOT inspect raw evidence again" -- and no
dependency on `missing_evidence.py`, `scope.py`, `contradiction.py`, or
`confidence_rule.py` themselves: this rule consumes what those rules
already produced, through `RuleContext`'s own fields and the documented
`metadata` contract above, and never imports, calls, or re-executes any
of their logic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from experiment_audit_mcp.reasoning.claims import Claim
from experiment_audit_mcp.reasoning.contradictions import Contradiction
from experiment_audit_mcp.reasoning.rules import (
    OutputCategory,
    RuleContext,
    RuleResult,
    ScientificRule,
)

# ---------------------------------------------------------------------
# Judgment thresholds
#
# Fixed constants, not tuned or learned, mirroring `confidence_rule.py`'s
# own convention of naming every numeric constant it uses. Chosen so
# that a genuinely high confidence adjustment (R004's top half of its
# `[-1.0, 1.0]` range) is required for `SUPPORTED`, and a genuinely
# negative one is required for `UNSUPPORTED`, leaving the broad middle
# ground -- and every case where a missing-evidence gap or unresolved
# contradiction is present -- to `PARTIALLY_SUPPORTED`.
# ---------------------------------------------------------------------

_SUPPORTED_CONFIDENCE_THRESHOLD = 0.5
_UNSUPPORTED_CONFIDENCE_THRESHOLD = -0.25


class ClaimJudgment(StrEnum):
    """The closed, named vocabulary of scientific standings this rule
    can assign to a claim, per this task's specification.

    A `str` subclass, matching this package's `EvidenceKind` /
    `ObservationKind` / `JudgmentKind` convention, so a judgment
    serializes as its own value without a manual `.value` lookup at
    every call site. Deliberately closed and small: this rule never
    invents a fifth category, and a claim this rule cannot confidently
    place in one of the first three is placed in `ABSTAIN` instead of
    being forced into the nearest one.
    """

    #: Evidence sufficiently supports the claim: no missing-evidence or
    #: scope gap, no unresolved contradiction, and a confidence
    #: adjustment at or above `_SUPPORTED_CONFIDENCE_THRESHOLD`.
    SUPPORTED = "supported"

    #: Evidence supports part of the claim but important limitations
    #: remain -- a missing-evidence/scope gap, an unresolved
    #: contradiction, or a confidence adjustment in the moderate middle
    #: ground between the two thresholds.
    PARTIALLY_SUPPORTED = "partially_supported"

    #: Available evidence does not justify the claim: a confidence
    #: adjustment at or below `_UNSUPPORTED_CONFIDENCE_THRESHOLD`.
    UNSUPPORTED = "unsupported"

    #: The available information is insufficient to reach any reliable
    #: conclusion -- the claim's scope is undeclared/unspecified, or no
    #: confidence input could be resolved for it at all.
    ABSTAIN = "abstain"


# ---------------------------------------------------------------------
# Scope-ambiguity check
# ---------------------------------------------------------------------


def _is_scope_unspecified(claim: Claim) -> bool:
    """Whether `claim`'s declared `Scope` leaves this rule unable to
    proceed past scope ambiguity, per the specification's Section 8
    "Unknown claim" abstain condition.

    A `claim.scope` of `None` is treated identically to a `Scope` whose
    own `is_unspecified()` reports `True` -- both mean "this claim's
    scope was never meaningfully declared." Read via `getattr` rather
    than a direct `claims.py` import of `Scope`, so this rule's only
    runtime dependency on that module remains `Claim` itself.
    """
    scope = claim.scope
    if scope is None:
        return True
    is_unspecified = getattr(scope, "is_unspecified", None)
    if callable(is_unspecified):
        return bool(is_unspecified())
    return False


# ---------------------------------------------------------------------
# Reading R001/R002's and R003's already-computed findings
#
# Both functions mirror `confidence_rule.py`'s functions of the same
# name exactly -- same attribution rule, same rationale -- re-declared
# locally rather than imported from that sibling module, per this
# module's architectural constraint that it does not depend on
# `confidence_rule.py`.
# ---------------------------------------------------------------------


def _missing_evidence_count_for_claim(context: RuleContext, claim: Claim) -> int:
    """How many of `RuleContext.missing_evidence`'s already-identified
    gaps (from R001 and R002 combined) bear on `claim`.

    `MissingEvidenceRecord` (evidence.py) is consulted only through
    duck-typed attribute access, never through an import of that type
    or of the rules that produce it -- this rule does not depend on
    `evidence.py`, `missing_evidence.py`, or `scope.py` at all. A record
    naming `claim.id` directly (`claim_id`) or among several claims
    (`claim_ids`) is attributed only to the claim(s) it names; a record
    with neither attribute is a context-wide gap with no claim-specific
    attribution recorded, and is conservatively counted against every
    claim, since this rule has no basis to narrow it to only some.
    """
    count = 0
    for record in context.missing_evidence:
        claim_id = getattr(record, "claim_id", None)
        if claim_id is not None:
            if claim_id == claim.id:
                count += 1
            continue
        claim_ids = getattr(record, "claim_ids", None)
        if claim_ids is not None:
            if claim.id in claim_ids:
                count += 1
            continue
        count += 1
    return count


#: Matches `ContradictionRule`'s within-claim finding prefix, e.g.
#: `"Claim C1 ('run-001') rests on evidence that conflicts with
#: itself: ..."`. Re-declared locally rather than imported from
#: `confidence_rule.py`, per this module's architectural constraint
#: that it depends only on `rules.py` and the normalized domain types
#: -- never on a sibling rule module's internals.
_NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN = re.compile(r"^Claim\s+(\S+)\s*\(")

#: Matches `ContradictionRule`'s cross-claim finding prefix, e.g.
#: `"Claim C004 ('subject') and claim C005 ('subject') record
#: conflicting values ..."`. Checked *before*
#: `_NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN` in
#: `_new_contradiction_claim_ids`, since this text also begins with the
#: literal prefix the within-claim pattern matches -- checking that
#: pattern first would silently capture only the first claim id and
#: drop the second.
_NEW_CONTRADICTION_CROSS_CLAIM_PATTERN = re.compile(
    r"^Claim\s+(\S+)\s*\([^)]*\)\s+and\s+claim\s+(\S+)\s*\("
)


def _new_contradiction_claim_ids(finding: str) -> frozenset[str]:
    """Every `Claim.id` a single same-pass contradiction finding (one
    entry of `RuleContext.metadata["newly_detected_contradictions"]`,
    forwarded by `pipeline.py` from `ContradictionRule`'s (R003)
    `RuleResult.contradictions`) names, or an empty `frozenset` if this
    finding's format is not recognized.

    `pipeline.py` only forwards `R003` findings whose text does not
    begin with `"Contradiction "` (the carried-forward-known-
    contradiction marker) into this channel, so a contradiction already
    counted via `RuleContext.detected_contradictions` is never counted
    a second time here.
    """
    cross = _NEW_CONTRADICTION_CROSS_CLAIM_PATTERN.match(finding)
    if cross is not None:
        return frozenset({cross.group(1), cross.group(2)})
    within = _NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN.match(finding)
    if within is not None:
        return frozenset({within.group(1)})
    return frozenset()


def _contradiction_counts_for_claim(context: RuleContext, claim: Claim) -> tuple[int, int]:
    """`(unresolved_count, resolved_count)` among
    `RuleContext.detected_contradictions` (R003's output) that name
    `claim`, plus any contradiction `ContradictionRule` detected during
    this same pipeline pass and named `claim` in, via
    `RuleContext.metadata["newly_detected_contradictions"]` (see
    `_new_contradiction_claim_ids`). A same-pass finding is counted at
    the unresolved weight: a contradiction just detected, and not yet
    carried through Chapter 8's Investigated/Resolved lifecycle, is
    unresolved by definition. This mirrors `confidence_rule.py`'s
    identical fix, so a claim's judgment can never disagree with its
    own confidence adjustment about whether a same-pass contradiction
    exists.

    Reads only `Contradiction.claims` (matched by `Claim.id`, never by
    dataclass equality, so the same claim at a different lifecycle
    stage is still recognized) and `Contradiction.is_resolved`. This
    rule never recomputes a `Contradiction`'s status, sources, or
    quality, and never advances it through Chapter 8's lifecycle; it
    only tallies what has already been recorded.
    """
    unresolved = 0
    resolved = 0
    known: Contradiction
    for known in context.detected_contradictions:
        if not any(c.id == claim.id for c in known.claims):
            continue
        if known.is_resolved:
            resolved += 1
        else:
            unresolved += 1

    for finding in context.metadata.get("newly_detected_contradictions", ()):
        if claim.id in _new_contradiction_claim_ids(finding):
            unresolved += 1

    return unresolved, resolved


# ---------------------------------------------------------------------
# Reading R004's already-computed confidence adjustment
#
# See this module's docstring, "Upstream contract for confidence," for
# the full justification of reading these two `context.metadata` keys
# rather than a dedicated `RuleContext` field.
# ---------------------------------------------------------------------


def _is_usable_confidence_number(value: object) -> bool:
    """Whether `value` is a genuine, finite confidence figure this rule
    can safely compare against its fixed thresholds.

    `context.metadata` is typed as `Mapping[str, Any]` (rules.py) --
    free-form and caller-supplied -- so nothing upstream guarantees
    that either confidence key actually holds a well-formed number by
    the time this rule reads it. This check exists purely to keep this
    rule from crashing or silently misjudging a claim on malformed
    input; it is not a re-validation of `ConfidenceRule`'s own output
    contract (which this rule still trusts completely when the value
    *is* well-formed). Two cases are excluded, deliberately treated the
    same as "absent" rather than coerced:

    - `bool`: a `bool` is an `int` subclass in Python, so
      `isinstance(True, int)` is `True` and `float(True) == 1.0` --
      but a truth value was never a confidence adjustment, so it is
      rejected explicitly rather than silently accepted as `1.0`/`0.0`.
    - Non-finite `float`s (`nan`, `inf`, `-inf`): these cannot be
      meaningfully compared against `_SUPPORTED_CONFIDENCE_THRESHOLD`
      / `_UNSUPPORTED_CONFIDENCE_THRESHOLD` (a `nan` comparison is
      always `False`, which would silently steer every such claim into
      the `PARTIALLY_SUPPORTED` branch without that ever being a
      genuine finding about the claim).
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _extract_confidence_inputs(
    context: RuleContext,
) -> tuple[Mapping[str, float], float | None]:
    """`(per_claim_adjustments, fallback_adjustment)` read from
    `context.metadata`.

    `per_claim_adjustments` is `context.metadata["confidence_adjustments"]`
    if present and a `Mapping`, filtered to only those entries whose
    value is a usable confidence number (`_is_usable_confidence_number`)
    -- an entry that fails that check is treated identically to that
    claim having no per-claim figure at all, falling through to
    `fallback_adjustment` for that claim alone rather than discarding
    every other claim's valid entry or raising. `fallback_adjustment`
    is `context.metadata["confidence_adjustment"]` if present and a
    usable confidence number, else `None`. Beyond usability, neither
    value is validated further here (e.g. bounds-checked against
    `[-1.0, 1.0]`) -- this rule only reads what `ConfidenceRule` already
    computed and bounded; re-validating it further would be re-deriving
    R004's own contract, not consuming its output.
    """
    raw_per_claim = context.metadata.get("confidence_adjustments")
    per_claim: Mapping[str, float]
    if isinstance(raw_per_claim, Mapping):
        per_claim = {
            claim_id: float(value)
            for claim_id, value in raw_per_claim.items()
            if _is_usable_confidence_number(value)
        }
    else:
        per_claim = {}
    fallback = context.metadata.get("confidence_adjustment")
    fallback_adjustment = float(fallback) if _is_usable_confidence_number(fallback) else None
    return per_claim, fallback_adjustment


def _confidence_for_claim(
    claim: Claim,
    per_claim_adjustments: Mapping[str, float],
    fallback_adjustment: float | None,
) -> tuple[float, bool, bool]:
    """`(value, is_known, is_fallback)` for `claim`'s confidence
    adjustment.

    Prefers `per_claim_adjustments[claim.id]` (R004's precise,
    claim-specific figure) when present. Falls back to
    `fallback_adjustment` (R004's single, context-wide aggregate) only
    when no claim-specific figure is available, and reports that a
    fallback was used via `is_fallback` so the caller can say so
    explicitly in this claim's reasoning. `is_known` is `False`, with
    `value` meaningless, only when neither source has anything for this
    claim -- the case this rule must abstain for.
    """
    if claim.id in per_claim_adjustments:
        return float(per_claim_adjustments[claim.id]), True, False
    if fallback_adjustment is not None:
        return float(fallback_adjustment), True, True
    return 0.0, False, False


# ---------------------------------------------------------------------
# _ClaimJudgmentResult
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ClaimJudgmentResult:
    """One claim's synthesized judgment.

    A small, internal record bundling one claim's assigned
    `ClaimJudgment` together with the plain-language sentence
    explaining exactly which already-computed findings produced it, so
    `JudgmentRule.evaluate` can build the overall `RuleResult.reasoning`
    and `metadata` from the same underlying per-claim results without
    recomputing either.

    Attributes:
        claim: The `Claim` this judgment concerns.
        judgment: The `ClaimJudgment` assigned to `claim`.
        reasoning: The plain-language sentence explaining `judgment`,
            citing the specific missing-evidence/scope gap count,
            contradiction counts, and confidence figure that produced
            it.
    """

    claim: Claim
    judgment: ClaimJudgment
    reasoning: str


def _judge_claim(
    context: RuleContext,
    claim: Claim,
    per_claim_adjustments: Mapping[str, float],
    fallback_adjustment: float | None,
) -> _ClaimJudgmentResult:
    """Synthesize `claim`'s `ClaimJudgment` from R001-R004's
    already-computed findings for it, per this module's decision
    procedure (see module docstring, "Abstain conditions" and
    "Unresolved contradictions cap support").
    """
    if _is_scope_unspecified(claim):
        return _ClaimJudgmentResult(
            claim=claim,
            judgment=ClaimJudgment.ABSTAIN,
            reasoning=(
                f"Claim {claim.id} ({claim.subject!r}) is judged ABSTAIN: its "
                "declared scope is absent or unspecified, and per specification "
                'Section 8\'s "Unknown claim" condition, judgment cannot proceed '
                "past unresolved scope ambiguity."
            ),
        )

    missing_count = _missing_evidence_count_for_claim(context, claim)
    unresolved, resolved = _contradiction_counts_for_claim(context, claim)
    confidence_value, confidence_known, used_fallback = _confidence_for_claim(
        claim, per_claim_adjustments, fallback_adjustment
    )

    if not confidence_known:
        return _ClaimJudgmentResult(
            claim=claim,
            judgment=ClaimJudgment.ABSTAIN,
            reasoning=(
                f"Claim {claim.id} ({claim.subject!r}) is judged ABSTAIN: no "
                "confidence adjustment (ConfidenceRule / R004 output) could be "
                "resolved for this claim, and per specification Section 8's "
                '"Incomplete metadata" condition, judgment cannot proceed '
                "without it."
            ),
        )

    confidence_note = f"confidence_adjustment={confidence_value:.2f}" + (
        " (context-wide fallback, not claim-specific)" if used_fallback else ""
    )
    detail = (
        f"{missing_count} missing-evidence/scope gap(s); {unresolved} unresolved "
        f"and {resolved} resolved contradiction(s); {confidence_note}"
    )

    if unresolved > 0:
        if confidence_value <= _UNSUPPORTED_CONFIDENCE_THRESHOLD:
            judgment = ClaimJudgment.UNSUPPORTED
            rationale = (
                "an unresolved contradiction stands against this claim and its "
                f"confidence is at or below the unsupported threshold "
                f"({_UNSUPPORTED_CONFIDENCE_THRESHOLD:.2f})"
            )
        else:
            judgment = ClaimJudgment.PARTIALLY_SUPPORTED
            rationale = (
                "an unresolved contradiction stands against this claim; a claim "
                "with a live, unresolved contradiction is never judged fully "
                "supported regardless of its confidence"
            )
    elif missing_count == 0 and confidence_value >= _SUPPORTED_CONFIDENCE_THRESHOLD:
        judgment = ClaimJudgment.SUPPORTED
        rationale = (
            "no missing-evidence/scope gap and no unresolved contradiction "
            f"remain, and confidence meets the supported threshold "
            f"({_SUPPORTED_CONFIDENCE_THRESHOLD:.2f})"
        )
    elif confidence_value <= _UNSUPPORTED_CONFIDENCE_THRESHOLD:
        judgment = ClaimJudgment.UNSUPPORTED
        rationale = (
            "confidence is at or below the unsupported threshold "
            f"({_UNSUPPORTED_CONFIDENCE_THRESHOLD:.2f}), even with no unresolved "
            "contradiction recorded"
        )
    else:
        judgment = ClaimJudgment.PARTIALLY_SUPPORTED
        rationale = (
            "evidence partially supports this claim: a missing-evidence/scope "
            "gap remains, or confidence falls in the moderate range between the "
            "supported and unsupported thresholds"
        )

    reasoning = (
        f"Claim {claim.id} ({claim.subject!r}) is judged "
        f"{judgment.value.upper()}: {rationale} ({detail})."
    )
    return _ClaimJudgmentResult(claim=claim, judgment=judgment, reasoning=reasoning)


# ---------------------------------------------------------------------
# JudgmentRule
# ---------------------------------------------------------------------


class JudgmentRule(ScientificRule):
    """Synthesizes `MissingEvidenceRule` (R001), `ScopeRule` (R002),
    `ContradictionRule` (R003), and `ConfidenceRule` (R004)'s
    already-computed findings into a terminal `ClaimJudgment` for each
    claim in a `RuleContext`, reported as an `OutputCategory.JUDGMENT`
    finding.

    For every `Claim` in `RuleContext.claims`, this rule:

    1. Checks whether the claim's declared `Scope` is absent or
       unspecified, abstaining immediately if so (scope ambiguity is a
       precondition failure, not a finding this rule can synthesize
       past).
    2. Reads how many already-identified missing-evidence/scope gaps
       (`RuleContext.missing_evidence`, R001 and R002's combined
       output) name the claim.
    3. Reads how many unresolved and resolved contradictions
       (`RuleContext.detected_contradictions`, R003's output) name the
       claim.
    4. Resolves the claim's confidence adjustment via the documented
       `context.metadata` contract (R004's output; see this module's
       docstring), abstaining if none can be resolved.
    5. Combines these into exactly one of `ClaimJudgment.SUPPORTED`,
       `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, or `ABSTAIN`, per the fixed
       procedure in `_judge_claim`.

    This rule performs no scientific detection of its own -- see this
    module's docstring for the full boundary. It never touches
    `RuleResult.confidence_adjustment` (left at its `0.0` default: this
    rule reports a judgment, not a new confidence figure), never
    populates `RuleResult.contradictions` or `RuleResult.missing_evidence`
    (restating R001/R002/R003's own findings in this rule's own output
    fields would duplicate them rather than synthesize them), and never
    populates `RuleResult.recommendations` (proposing an action belongs
    to a future `RecommendationRule`, downstream of this one). Per-claim
    judgments are reported in `RuleResult.metadata["judgments"]`, a
    mapping from `Claim.id` to that claim's `ClaimJudgment` value,
    since `RuleResult` (rules.py) has no dedicated `judgment` field of
    its own; the full, itemized reasoning for every claim is
    nonetheless always present in `RuleResult.reasoning`, so the
    traceability this rule's finding requires never depends solely on
    that metadata entry.
    """

    @property
    def id(self) -> str:
        return "R005"

    @property
    def name(self) -> str:
        return "Scientific Judgment"

    @property
    def description(self) -> str:
        return (
            "Synthesizes each claim's already-computed missing-evidence, scope, "
            "contradiction, and confidence findings into a terminal judgment -- "
            "SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or ABSTAIN -- without "
            "re-deriving any of those upstream findings."
        )

    @property
    def version(self) -> str:
        return "1.0.0"

    def applies(self, context: RuleContext) -> bool:
        """Relevant whenever `context` has at least one claim to judge.

        With no claims present, there is nothing for this rule to
        synthesize a judgment for -- the same precondition
        `MissingEvidenceRule.applies`, `ScopeRule.applies`,
        `ContradictionRule.applies`, and `ConfidenceRule.applies` all
        use.
        """
        return context.claims is not None and len(context.claims) > 0

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Synthesize a `ClaimJudgment` for every claim in
        `context.claims`.

        Assumes `self.applies(context)` has already returned `True`
        (i.e. `context.claims` is non-empty), per `ScientificRule`'s
        two-phase evaluation contract.
        """
        assert context.claims is not None  # guaranteed by applies()

        claims = tuple(context.claims)
        per_claim_adjustments, fallback_adjustment = _extract_confidence_inputs(context)

        results = tuple(
            _judge_claim(context, claim, per_claim_adjustments, fallback_adjustment)
            for claim in claims
        )

        reasoning = (
            " ".join(result.reasoning for result in results)
            if results
            else "No claims were available to reach a judgment for."
        )

        triggered = any(result.judgment is not ClaimJudgment.SUPPORTED for result in results)

        return RuleResult(
            rule_id=self.id,
            rule_name=self.name,
            triggered=triggered,
            output_category=OutputCategory.JUDGMENT,
            reasoning=reasoning,
            affected_claims=claims,
            metadata={
                "judgments": {result.claim.id: result.judgment.value for result in results},
            },
        )


__all__ = ["JudgmentRule", "ClaimJudgment"]
