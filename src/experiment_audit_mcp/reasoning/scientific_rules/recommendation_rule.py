"""
Experiment Audit Scientific Reasoning Engine

Module: scientific_rules.recommendation_rule

Defines `RecommendationRule`, the sixth and final concrete
`ScientificRule` (rules.py) in this pipeline. It implements the
terminal synthesis step described in the specification's Section 4
("Recommendation") output category: given everything the five prior
rules -- `MissingEvidenceRule` (R001), `ScopeRule` (R002),
`ContradictionRule` (R003), `ConfidenceRule` (R004), and `JudgmentRule`
(R005) -- have already, independently, found and concluded, decide
what a researcher should concretely *do next* in response.

**Why this rule exists.** Every rule that precedes it in this pipeline
answers a question and stops there: whether evidence a claim's
category expects is absent (R001), whether a claim's own declared
scope exceeds its attributed evidence (R002), whether two or more
claims or evidence items genuinely conflict (R003), how strongly the
combination of those findings justifies believing a claim (R004), and
what the claim's current scientific standing is as a result (R005).
None of those five rules ever states what should change as a
consequence of what it found -- each stops at a finding or a verdict,
never an action. `RecommendationRule` exists to close that final gap:
it is the one rule in this pipeline whose entire job is to turn
already-closed findings into a concrete, actionable next step. Per the
specification's own consequence ordering ("ordered by consequence ...
through Judgment and Recommendation at the most consequential end"),
Recommendation is deliberately the single most consequential output
this pipeline produces, and deliberately the last rule to run.

**Why a recommendation is not a judgment.** `JudgmentRule` (R005)
answers "what does the evidentiary record, as it now stands, justify
concluding about this claim" -- a verdict about the claim's *current
standing*. `RecommendationRule` answers a different question: "given
that standing, together with the *specific kind* of finding that
produced it, what scientific action would most directly improve the
record." A judgment describes a state; a recommendation proposes a
transition out of an unsatisfactory one. The distinction matters
because a `SUPPORTED` judgment and an `ABSTAIN` judgment call for
opposite responses -- the first calls for no corrective action at all,
the second calls for collecting the evidence that would let a judgment
be reached in the first place -- and collapsing the two into "confident
vs. not confident" would lose exactly the information a researcher
needs to know what to do next. A recommendation also never restates a
judgment in different words ("this claim is not well supported" is not
a recommendation); it names a concrete next step, per this task's
"Specific, Actionable, Scientifically justified, Deterministic" design
requirement.

**Why recommendation is downstream of every other rule.** A
recommendation that is not traceable to an already-closed finding is
not a recommendation this methodology can stand behind -- it is a
guess. Per the specification's Section 1 ("a conclusion that cannot be
independently regenerated from its stated inputs has not ... been
shown to follow from those inputs at all"), this rule therefore
performs no scientific detection, confidence estimation, contradiction
detection, or judgment synthesis of its own. It never re-counts which
evidence dimensions a claim's category expects (R001's job), never
re-compares a claim's declared scope against its attributed evidence
(R002's job), never re-detects a conflict between claims or evidence
(R003's job), never re-weighs strengthening or reducing factors into a
new confidence figure (R004's job), and never re-synthesizes a claim's
scientific standing (R005's job). Every one of those questions is
closed by the time this rule runs; this rule only reads the closed
answers and asks what they imply should happen next.

**Upstream contract, documented because `rules.py` does not yet field
every prior rule's output as a dedicated `RuleContext` field.**
`RuleContext` (rules.py) already carries two fields this rule reads
directly:

- `RuleContext.missing_evidence` -- every already-identified
  evidentiary gap (R001's and R002's combined output; per `scope.py`'s
  own docstring, a scope violation "is reported ... under
  `OutputCategory.MISSING_EVIDENCE`, the same category
  `MissingEvidenceRule` uses," so this one field already carries both
  rules' findings). Each record is read only through duck-typed
  attribute access (`claim_id`, `claim_ids`, `description`,
  `recommendation`), never through an import of `evidence.py`'s
  `MissingEvidenceRecord` type -- this rule has no dependency on
  `evidence.py` at all, consistent with never inspecting raw evidence.
- `RuleContext.detected_contradictions` -- every already-recorded
  `Contradiction` (R003's output), read through its own public
  attributes (`id`, `claims`, `is_resolved`, `evidence_items`).

Neither `RuleContext` nor any sibling module in this package defines a
dedicated field carrying R004's per-claim confidence adjustment or
R005's per-claim judgment, however. Rather than block on `rules.py`
being revised to add one (outside this task's remit) or reach into
`confidence_rule.py` / `judgment_rule.py`'s internals directly (which
would make this rule depend on *how* those rules compute their
findings, not merely *what* they concluded), this module reads the
same channel `judgment_rule.py` itself already established and
documented for exactly this situation -- `RuleContext.metadata`,
"free-form, caller-supplied context that does not fit any of the
above":

- `context.metadata["confidence_adjustments"]`, if present, is a
  mapping from `Claim.id` to that claim's own `float` confidence
  adjustment, as computed by `ConfidenceRule` (R004) for that specific
  claim -- the preferred, most precise source.
- `context.metadata["confidence_adjustment"]`, if present, is a single
  `float` -- `ConfidenceRule`'s aggregate `RuleResult.confidence_adjustment`,
  averaged across every claim in its own `RuleContext`. Used only when
  no per-claim figure is available, applied uniformly, and always
  noted as a fallback in this rule's reasoning so a reader is never
  left thinking a claim-specific figure was used when it was not.
- `context.metadata["judgments"]`, if present, is a mapping from
  `Claim.id` to that claim's `str` judgment value, exactly as
  `JudgmentRule` (R005) itself already reports in its own
  `RuleResult.metadata["judgments"]` -- one of `"supported"`,
  `"partially_supported"`, `"unsupported"`, or `"abstain"`. This rule
  never imports `judgment_rule.py`'s `ClaimJudgment` enum; it matches
  against that closed vocabulary's known string values directly, so
  this rule's only dependency on the Judgment stage is the value it
  already serializes to, never the class that produced it.

A claim for which none of these three inputs, nor any relevant
missing-evidence record or unresolved contradiction, is available
contributes no recommendation at all -- this rule never invents a
recommendation for a claim it has nothing traceable to say about.

**Recommendation pathways, one per upstream source, never blended.**
Every recommendation this rule produces is traceable to exactly one of
four already-closed findings, mirroring this task's own worked
examples:

1. **Missing-evidence / scope gap present** (R001, R002 via
   `RuleContext.missing_evidence`) -> collect the specific evidence
   that gap names, preferring that record's own `recommendation` text
   when present, and otherwise a plain restatement of its
   `description`.
2. **Unresolved contradiction present** (R003 via
   `RuleContext.detected_contradictions`) -> reproduce or otherwise
   investigate the specific conflicting experiment(s) before drawing
   further conclusions. A *resolved* contradiction implies no
   recommendation here -- its resolution is already recorded, and
   recommending action against a settled matter would not be
   traceable to anything currently open.
3. **Confidence below this rule's own low-confidence threshold**
   (R004 via the `metadata` contract above) -> strengthen evidence
   quality before making broad claims from this record. This
   threshold (`_LOW_CONFIDENCE_RECOMMENDATION_THRESHOLD`) is this
   rule's own, independent constant for deciding when a recommendation
   of this specific kind is warranted; it is not R004's or R005's
   threshold re-derived, and this rule never uses it to re-synthesize
   a judgment of its own.
4. **A judgment other than `SUPPORTED` reached** (R005 via the
   `metadata` contract above) -> a fixed, judgment-specific
   recommendation (`_JUDGMENT_RECOMMENDATIONS`): collect sufficient
   evidence before concluding anything, for `ABSTAIN`; strengthen or
   replace the claim's evidence, for `UNSUPPORTED`; resolve the
   claim's outstanding gaps or conflicts, for `PARTIALLY_SUPPORTED`. A
   `SUPPORTED` judgment contributes no recommendation from this
   pathway -- no corrective action is implied by a judgment this rule
   did not itself reach and has no basis to second-guess.

These four pathways may all fire for the same claim, and often will --
a claim can simultaneously have an open evidentiary gap, an unresolved
contradiction, low confidence, and a non-`SUPPORTED` judgment, and each
of those facts warrants its own, separately traceable recommendation
rather than being collapsed into one. Recommendations are deduplicated
only when their exact text coincides (e.g. the same missing-evidence
record attributed to more than one claim by this rule's own
attribution logic); this rule never merges or rewords two distinct
findings into a single blended recommendation.

**Determinism.** Every recommendation this rule produces is a fixed,
literal transformation of already-computed inputs -- iterating
`RuleContext.missing_evidence` and `RuleContext.detected_contradictions`
in their existing order, reading `context.metadata`'s two documented
keys, and applying one fixed numeric threshold. None of it is
probabilistic, none of it calls out to a model, and re-running this
rule against an unchanged `RuleContext` always yields the same
`RuleResult`.

**Architectural constraint, mirrored from every other module in this
package, including `judgment_rule.py` and `confidence_rule.py`.** This
module depends only on `rules.py` (the framework it plugs into),
`claims.py`'s `Claim` (to read a claim's `id` and `subject`, the same
claims-stage data every other rule in this package already depends
on), and `contradictions.py`'s `Contradiction` (to read `.id`,
`.claims`, `.is_resolved`, and `.evidence_items` from
`RuleContext.detected_contradictions`, the same attributes
`confidence_rule.py` and `judgment_rule.py` themselves already read
from that type), and the Python standard library. It has no dependency
on FastMCP, MCP transport, `server.py`, `evidence.py`, `observations.py`,
or `hypotheses.py` -- this rule never inspects raw evidence,
observations, or hypotheses, per this task's explicit "MUST NOT inspect
raw Evidence / Observations / Hypotheses" requirement -- and no
dependency on `missing_evidence.py`, `scope.py`, `contradiction.py`,
`confidence_rule.py`, `judgment_rule.py`, `confidence.py`, `judgment.py`,
or `recommendation.py` themselves: this rule consumes what those
modules' rules already produced, through `RuleContext`'s own fields and
the documented `metadata` contract above, and never imports, calls, or
re-executes any of their logic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from experiment_audit_mcp.reasoning.claims import Claim
from experiment_audit_mcp.reasoning.contradictions import Contradiction
from experiment_audit_mcp.reasoning.rules import (
    OutputCategory,
    RuleContext,
    RuleResult,
    ScientificRule,
)

# ---------------------------------------------------------------------
# This rule's own low-confidence threshold
#
# Deliberately independent of `JudgmentRule`'s (R005)
# `_SUPPORTED_CONFIDENCE_THRESHOLD` / `_UNSUPPORTED_CONFIDENCE_THRESHOLD`
# and of `ConfidenceRule`'s (R004) own aggregation weights, even though
# the value happens to coincide with R005's supported-threshold: this
# constant answers a narrower question -- "is confidence low enough
# that a researcher should be told to strengthen the evidence before
# making broad claims" -- not "should this claim be judged supported."
# Re-declared here, independently, rather than imported from either
# sibling rule module, per this module's architectural constraint that
# it does not depend on `confidence_rule.py` or `judgment_rule.py`.
# ---------------------------------------------------------------------

_LOW_CONFIDENCE_RECOMMENDATION_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------
# Judgment-keyed recommendation templates
#
# Keyed on the closed, three-value vocabulary `JudgmentRule` (R005)
# itself already serializes to `context.metadata["judgments"]`
# (`"abstain"`, `"unsupported"`, `"partially_supported"`) -- matched
# here as plain strings, never via an import of `judgment_rule.py`'s
# `ClaimJudgment` enum, per this module's architectural constraint.
# `"supported"` is deliberately absent: a supported judgment implies no
# corrective action for this pathway to recommend.
# ---------------------------------------------------------------------

_JUDGMENT_RECOMMENDATIONS: dict[str, str] = {
    "abstain": (
        "Collect sufficient evidence for claim {claim_id} ({subject!r}) before "
        "reaching any scientific conclusion about it; its declared scope or "
        "confidence input is currently insufficient to judge."
    ),
    "unsupported": (
        "Strengthen or replace the evidence supporting claim {claim_id} "
        "({subject!r}) before continuing to assert it; the currently attributed "
        "evidence does not justify it."
    ),
    "partially_supported": (
        "Resolve the outstanding missing-evidence, scope, or contradiction gaps "
        "for claim {claim_id} ({subject!r}) before treating it as fully "
        "supported."
    ),
}


# ---------------------------------------------------------------------
# Reading R001/R002's already-computed missing-evidence findings
#
# Mirrors `judgment_rule.py`'s and `confidence_rule.py`'s own
# `_missing_evidence_count_for_claim` attribution rule exactly -- same
# rule, same rationale -- re-declared locally rather than imported from
# either sibling module, per this module's architectural constraint.
# Extended here, beyond a bare count, to return the records themselves,
# since a recommendation (unlike a count) needs each record's own
# content to be specific and actionable.
# ---------------------------------------------------------------------


#: Matches the `"Claim {id} ("` prefix `missing_evidence.py` and
#: `scope.py` both use, consistently, for every gap they report (see
#: e.g. `missing_evidence.py`'s `f"Claim {claim.id} ({claim.subject!r}): "`
#: and `scope.py`'s equivalent construction). Captures the claim id as
#: the text between "Claim " and the next whitespace, so it matches
#: regardless of what follows (a parenthesized subject, a colon, or
#: anything else either rule appends).
#:
#: Re-declared identically to `confidence_rule.py`'s own
#: `_GAP_CLAIM_ID_PATTERN` rather than imported from it, per this
#: module's architectural constraint that it does not depend on
#: `confidence_rule.py` -- but the parsing rule itself must stay in
#: lockstep with that module's, since both rules read the exact same
#: `RuleContext.missing_evidence` strings and must agree on which
#: claim each one names. A prefix-shaped string that does not resolve
#: to a known claim is not trusted, since this rule cannot verify it
#: names a real claim in this context.
_GAP_CLAIM_ID_PATTERN = re.compile(r"^Claim\s+(\S+)\s*\(")


def _gap_claim_ids(record: Any, known_claim_ids: frozenset[str]) -> frozenset[str] | None:
    """Every `Claim.id` a single already-identified evidentiary gap
    (one entry of `RuleContext.missing_evidence`) names, or `None` if
    this gap names no claim this rule can identify at all -- meaning
    it is a genuinely context-wide gap, to be attributed to every
    claim.

    Tries three sources, in order, preferring the most structured one
    available:

    1. A `claim_id` attribute (a future, structured
       `MissingEvidenceRecord` naming exactly one claim).
    2. A `claim_ids` attribute (a future, structured record naming
       several claims at once).
    3. For a plain `str` gap -- what `MissingEvidenceRule` and
       `ScopeRule` actually produce today -- the `Claim.id` named by
       its leading `"Claim {id} ("` prefix, but only when that id
       matches a claim actually present in `known_claim_ids`.

    Never re-derives *why* the gap exists (that remains R001/R002's
    own finding); only ever determines *which claim(s)* it was
    already reported against.
    """
    claim_id = getattr(record, "claim_id", None)
    if claim_id is not None:
        return frozenset({claim_id})

    claim_ids = getattr(record, "claim_ids", None)
    if claim_ids is not None:
        return frozenset(claim_ids)

    if isinstance(record, str):
        match = _GAP_CLAIM_ID_PATTERN.match(record)
        if match is not None and match.group(1) in known_claim_ids:
            return frozenset({match.group(1)})

    return None


def _missing_evidence_records_for_claim(
    context: RuleContext, claim: Claim, known_claim_ids: frozenset[str]
) -> list[Any]:
    """Every entry of `RuleContext.missing_evidence` (R001 and R002's
    combined output) attributable to `claim`.

    A record naming `claim.id` directly (`claim_id`), among several
    claims (`claim_ids`), or -- for the plain-string gaps
    `MissingEvidenceRule`/`ScopeRule` actually produce -- via its
    leading `"Claim {id} ("` prefix, is attributed only to the
    claim(s) it names, exactly as `confidence_rule.py`'s own
    `_missing_evidence_count_for_claim` already attributes these same
    records (see `_gap_claim_ids`, above, re-declared locally to match
    it). A gap this rule cannot attribute to any specific known claim
    at all is a genuinely context-wide gap with no claim-specific
    attribution recorded, and is conservatively attributed to every
    claim, since this rule has no basis to narrow it to only some. A
    gap attributed to one or more *other* known claims, and not to
    `claim`, is never attributed to `claim` -- a missing-evidence
    recommendation for Claim A must never be issued because Claim B
    has missing evidence.
    """
    matched: list[Any] = []
    for record in context.missing_evidence:
        claim_ids = _gap_claim_ids(record, known_claim_ids)
        if claim_ids is None or claim.id in claim_ids:
            matched.append(record)
    return matched


def _missing_evidence_recommendation(record: Any, claim: Claim) -> str:
    """The recommendation text implied by one already-identified
    missing-evidence/scope record, for `claim`.

    Prefers the record's own `recommendation` attribute, if it carries
    one -- the most direct, already-authored statement of the action
    R001 or R002 itself associated with this specific gap. Falls back
    to restating the record's own `description` (trying a small,
    documented set of plausible attribute names, since this rule does
    not import `evidence.py`'s `MissingEvidenceRecord` type and so
    cannot assume its exact shape) inside a fixed, generic action
    frame. Never fabricates a specific action this rule cannot trace to
    the record's own content.
    """
    explicit = getattr(record, "recommendation", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit

    description = (
        getattr(record, "description", None)
        or getattr(record, "missing_description", None)
        or getattr(record, "statement", None)
    )
    if isinstance(description, str) and description.strip():
        return (
            f"Claim {claim.id} ({claim.subject!r}): collect the missing "
            f"evidence already identified for this claim -- {description} -- "
            "before treating its evidentiary record as complete."
        )

    return (
        f"Claim {claim.id} ({claim.subject!r}): collect the missing evidence "
        "already identified for this claim (see this pass's missing-evidence "
        "findings) before treating its evidentiary record as complete."
    )


# ---------------------------------------------------------------------
# Reading R003's already-computed contradiction findings
# ---------------------------------------------------------------------


def _unresolved_contradictions_for_claim(context: RuleContext, claim: Claim) -> list[Contradiction]:
    """Every unresolved `Contradiction` in
    `RuleContext.detected_contradictions` (R003's output) that names
    `claim`.

    A *resolved* contradiction is deliberately excluded: its
    resolution is already recorded, and this rule has nothing open to
    recommend action against. This rule never re-derives `is_resolved`
    or any other attribute of a `Contradiction` -- it only reads what
    R003 already recorded.
    """
    return [
        contradiction
        for contradiction in context.detected_contradictions
        if not contradiction.is_resolved and any(c.id == claim.id for c in contradiction.claims)
    ]


def _contradiction_recommendation(contradiction: Contradiction, claim: Claim) -> str:
    """The recommendation text implied by one unresolved `Contradiction`
    bearing on `claim`: reproduce or otherwise investigate the specific
    conflicting experiment(s) before drawing further conclusions.
    """
    counterpart_ids = sorted(c.id for c in contradiction.claims if c.id != claim.id)
    counterpart_note = f" and claim(s) {', '.join(counterpart_ids)}" if counterpart_ids else ""
    return (
        f"Reproduce the conflicting experiment(s) behind unresolved "
        f"contradiction {contradiction.id} and investigate the discrepancy "
        f"affecting claim {claim.id}{counterpart_note} before drawing further "
        "conclusions from either."
    )


#: Matches `ContradictionRule`'s within-claim finding prefix. Re-declared
#: locally rather than imported from `confidence_rule.py` or
#: `judgment_rule.py`, per this module's architectural constraint that
#: it depends on neither sibling rule module's internals.
_NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN = re.compile(r"^Claim\s+(\S+)\s*\(")

#: Matches `ContradictionRule`'s cross-claim finding prefix. Checked
#: *before* `_NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN` in
#: `_new_contradiction_claim_ids` -- see that module-level function's
#: docstring in `confidence_rule.py` / `judgment_rule.py` for why order
#: matters here.
_NEW_CONTRADICTION_CROSS_CLAIM_PATTERN = re.compile(
    r"^Claim\s+(\S+)\s*\([^)]*\)\s+and\s+claim\s+(\S+)\s*\("
)


def _new_contradiction_claim_ids(finding: str) -> frozenset[str]:
    """Every `Claim.id` a single same-pass contradiction finding (one
    entry of `RuleContext.metadata["newly_detected_contradictions"]`,
    forwarded by `pipeline.py` from `ContradictionRule`'s (R003)
    `RuleResult.contradictions`) names, or an empty `frozenset` if this
    finding's format is not recognized. Identical in shape and
    rationale to `confidence_rule.py` / `judgment_rule.py`'s function
    of the same name.
    """
    cross = _NEW_CONTRADICTION_CROSS_CLAIM_PATTERN.match(finding)
    if cross is not None:
        return frozenset({cross.group(1), cross.group(2)})
    within = _NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN.match(finding)
    if within is not None:
        return frozenset({within.group(1)})
    return frozenset()


def _new_contradictions_for_claim(context: RuleContext, claim: Claim) -> tuple[str, ...]:
    """Every same-pass contradiction finding, in
    `RuleContext.metadata["newly_detected_contradictions"]`, that names
    `claim` -- the current-run counterpart to
    `_unresolved_contradictions_for_claim`'s already-known contradictions.
    Unlike that function, this one has only a plain-language finding to
    work with (no `Contradiction.id` or `.claims`), since `pipeline.py`
    forwards these as strings, not `Contradiction` instances -- see
    `pipeline.py`'s `_advance_context` for why.
    """
    return tuple(
        finding
        for finding in context.metadata.get("newly_detected_contradictions", ())
        if claim.id in _new_contradiction_claim_ids(finding)
    )


def _new_contradiction_recommendation(finding: str, claim: Claim) -> str:
    """The recommendation text implied by one same-pass contradiction
    finding bearing on `claim`: investigate the conflict this pass just
    detected before drawing further conclusions. Mirrors
    `_contradiction_recommendation`'s wording as closely as the absence
    of a `Contradiction.id` allows, since this finding has not yet been
    promoted into a first-class `Contradiction`.
    """
    return (
        f"Investigate the newly detected contradiction affecting claim "
        f"{claim.id} before drawing further conclusions from it: {finding}"
    )


# ---------------------------------------------------------------------
# Reading R004's already-computed confidence adjustment
#
# Identical in shape and rationale to `judgment_rule.py`'s functions of
# the same name -- see this module's docstring, "Upstream contract" --
# re-declared locally rather than imported, per this module's
# architectural constraint that it does not depend on
# `confidence_rule.py` or `judgment_rule.py`.
# ---------------------------------------------------------------------


def _extract_confidence_inputs(
    context: RuleContext,
) -> tuple[Mapping[str, float], float | None]:
    """`(per_claim_adjustments, fallback_adjustment)` read from
    `context.metadata`.

    `per_claim_adjustments` is `context.metadata["confidence_adjustments"]`
    if present and a `Mapping`, else an empty mapping. `fallback_adjustment`
    is `context.metadata["confidence_adjustment"]` if present and a real
    `int`/`float`, else `None`. Neither value is validated further here
    -- this rule only reads what `ConfidenceRule` already computed and
    bounded.
    """
    per_claim = context.metadata.get("confidence_adjustments")
    if not isinstance(per_claim, Mapping):
        per_claim = {}
    fallback = context.metadata.get("confidence_adjustment")
    if isinstance(fallback, bool) or not isinstance(fallback, (int, float)):
        fallback = None
    return per_claim, fallback


def _confidence_for_claim(
    claim: Claim,
    per_claim_adjustments: Mapping[str, float],
    fallback_adjustment: float | None,
) -> tuple[float, bool, bool]:
    """`(value, is_known, is_fallback)` for `claim`'s confidence
    adjustment, preferring a claim-specific figure over the
    context-wide aggregate. `is_known` is `False`, with `value`
    meaningless, only when neither source has anything for this claim.
    """
    if claim.id in per_claim_adjustments:
        return float(per_claim_adjustments[claim.id]), True, False
    if fallback_adjustment is not None:
        return float(fallback_adjustment), True, True
    return 0.0, False, False


def _confidence_recommendation(claim: Claim, confidence_value: float, used_fallback: bool) -> str:
    """The recommendation text implied by a resolved confidence
    adjustment below `_LOW_CONFIDENCE_RECOMMENDATION_THRESHOLD` for
    `claim`.
    """
    fallback_note = " (context-wide fallback, not claim-specific)" if used_fallback else ""
    return (
        f"Strengthen evidence quality for claim {claim.id} ({claim.subject!r}) "
        f"before making broad scientific claims from it: its confidence "
        f"adjustment ({confidence_value:.2f}{fallback_note}) is below this "
        f"rule's low-confidence threshold of "
        f"{_LOW_CONFIDENCE_RECOMMENDATION_THRESHOLD:.2f}."
    )


# ---------------------------------------------------------------------
# Reading R005's already-computed judgment
# ---------------------------------------------------------------------


def _extract_judgments(context: RuleContext) -> Mapping[str, str]:
    """`context.metadata["judgments"]` (R005's per-claim judgment
    mapping), if present and a `Mapping`, else an empty mapping.

    Values are read and matched as plain strings against
    `_JUDGMENT_RECOMMENDATIONS`'s known vocabulary -- this rule never
    imports `judgment_rule.py`'s `ClaimJudgment` enum, per this
    module's architectural constraint.
    """
    judgments = context.metadata.get("judgments")
    if not isinstance(judgments, Mapping):
        return {}
    return judgments


def _judgment_recommendation(claim: Claim, judgment_value: str) -> str | None:
    """The recommendation text implied by `claim`'s already-reached
    judgment, or `None` if that judgment (`"supported"`, or any value
    outside this rule's known vocabulary) implies no recommendation
    from this pathway.
    """
    template = _JUDGMENT_RECOMMENDATIONS.get(judgment_value)
    if template is None:
        return None
    return template.format(claim_id=claim.id, subject=claim.subject)


# ---------------------------------------------------------------------
# _ClaimRecommendations
# ---------------------------------------------------------------------


@dataclass(slots=True)
class _ClaimRecommendations:
    """One claim's accumulated recommendations and the reasoning lines
    explaining where each came from.

    A small, internal record used only so `RecommendationRule.evaluate`
    can build the overall `RuleResult.recommendations`, `reasoning`,
    `affected_claims`, and `evidence_used` from the same underlying
    per-claim results without recomputing any of them twice.
    """

    claim: Claim
    recommendations: list[str] = field(default_factory=list)
    reasoning_lines: list[str] = field(default_factory=list)
    evidence_used: list[Any] = field(default_factory=list)


def _recommendations_for_claim(
    context: RuleContext,
    claim: Claim,
    judgments: Mapping[str, str],
    per_claim_adjustments: Mapping[str, float],
    fallback_adjustment: float | None,
    known_claim_ids: frozenset[str],
) -> _ClaimRecommendations:
    """Synthesize every recommendation `claim` warrants from R001-R005's
    already-computed findings, per this module's four documented
    pathways (see module docstring, "Recommendation pathways").
    """
    result = _ClaimRecommendations(claim=claim)

    # Pathway 1: missing-evidence / scope gaps (R001, R002).
    for record in _missing_evidence_records_for_claim(context, claim, known_claim_ids):
        result.recommendations.append(_missing_evidence_recommendation(record, claim))
        result.reasoning_lines.append(
            f"claim {claim.id} has an already-identified missing-evidence/scope gap (R001/R002)"
        )

    # Pathway 2: unresolved contradictions (R003).
    for contradiction in _unresolved_contradictions_for_claim(context, claim):
        result.recommendations.append(_contradiction_recommendation(contradiction, claim))
        result.reasoning_lines.append(
            f"claim {claim.id} is named by unresolved contradiction {contradiction.id} (R003)"
        )
        for item in getattr(contradiction, "evidence_items", ()):
            result.evidence_used.append(item)

    # Pathway 2b: same-pass contradictions (R003, this run). A
    # contradiction ContradictionRule detects during the current pass
    # never becomes a `Contradiction` instance (see `pipeline.py`'s
    # `_advance_context` and `contradiction_rule.py`'s own docstring),
    # so it cannot go through Pathway 2 above -- it has no `.id` or
    # `.claims` to read. Without this pathway, a claim with a
    # freshly-detected contradiction would silently receive no
    # recommendation at all, even though R004's confidence adjustment
    # (and R005's judgment) for the same claim now reflect it.
    for finding in _new_contradictions_for_claim(context, claim):
        result.recommendations.append(_new_contradiction_recommendation(finding, claim))
        result.reasoning_lines.append(
            f"claim {claim.id} is named by a contradiction ContradictionRule "
            "detected during this pass (R003)"
        )

    # Pathway 3: low confidence (R004).
    confidence_value, confidence_known, used_fallback = _confidence_for_claim(
        claim, per_claim_adjustments, fallback_adjustment
    )
    if confidence_known and confidence_value < _LOW_CONFIDENCE_RECOMMENDATION_THRESHOLD:
        result.recommendations.append(
            _confidence_recommendation(claim, confidence_value, used_fallback)
        )
        result.reasoning_lines.append(
            f"claim {claim.id} has a confidence adjustment "
            f"({confidence_value:.2f}) below this rule's low-confidence "
            f"threshold (R004)"
        )

    # Pathway 4: non-SUPPORTED judgment (R005).
    judgment_value = judgments.get(claim.id)
    if judgment_value is not None:
        judgment_text = _judgment_recommendation(claim, judgment_value)
        if judgment_text is not None:
            result.recommendations.append(judgment_text)
            result.reasoning_lines.append(f"claim {claim.id} was judged {judgment_value!r} (R005)")

    if not result.recommendations:
        result.reasoning_lines.append(
            f"claim {claim.id} has no open missing-evidence gap, unresolved "
            "contradiction, low confidence, or non-supported judgment to act "
            "on"
        )

    return result


# ---------------------------------------------------------------------
# RecommendationRule
# ---------------------------------------------------------------------


class RecommendationRule(ScientificRule):
    """Synthesizes `MissingEvidenceRule` (R001), `ScopeRule` (R002),
    `ContradictionRule` (R003), `ConfidenceRule` (R004), and
    `JudgmentRule` (R005)'s already-computed findings into concrete,
    actionable scientific recommendations, reported as
    `OutputCategory.RECOMMENDATION` findings.

    For every `Claim` in `RuleContext.claims`, this rule reads:

    1. Every already-identified missing-evidence/scope gap
       (`RuleContext.missing_evidence`, R001 and R002's combined
       output) attributable to that claim.
    2. Every unresolved contradiction (`RuleContext.detected_contradictions`,
       R003's output) naming that claim.
    3. That claim's confidence adjustment, resolved via the documented
       `context.metadata` contract (R004's output; see this module's
       docstring).
    4. That claim's judgment, resolved via the same `context.metadata`
       contract (R005's output).

    and, for each of the four, produces a specific, actionable
    recommendation exactly where that already-closed finding warrants
    one (see module docstring, "Recommendation pathways"). This rule
    performs no scientific detection, confidence estimation,
    contradiction detection, or judgment synthesis of its own -- see
    this module's docstring for the full boundary. It never touches
    `RuleResult.confidence_adjustment` (left at its `0.0` default: this
    rule proposes actions, not a new confidence figure), and never
    populates `RuleResult.contradictions` or `RuleResult.missing_evidence`
    (restating R001/R002/R003's own findings in this rule's own output
    fields would duplicate them rather than act on them).
    """

    @property
    def id(self) -> str:
        return "R006"

    @property
    def name(self) -> str:
        return "Scientific Recommendation"

    @property
    def description(self) -> str:
        return (
            "Transforms each claim's already-computed missing-evidence, scope, "
            "contradiction, confidence, and judgment findings into concrete, "
            "traceable scientific recommendations, without re-deriving any of "
            "those upstream findings."
        )

    @property
    def version(self) -> str:
        return "1.0.0"

    def applies(self, context: RuleContext) -> bool:
        """Relevant whenever `context` has at least one claim to
        recommend action for.

        With no claims present, there is nothing this rule's four
        recommendation pathways could possibly attribute a
        recommendation to -- the same precondition
        `MissingEvidenceRule.applies`, `ScopeRule.applies`,
        `ContradictionRule.applies`, `ConfidenceRule.applies`, and
        `JudgmentRule.applies` all use.
        """
        return context.claims is not None and len(context.claims) > 0

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Synthesize every recommendation warranted for each claim in
        `context.claims`, from R001-R005's already-computed findings.

        Assumes `self.applies(context)` has already returned `True`
        (i.e. `context.claims` is non-empty), per `ScientificRule`'s
        two-phase evaluation contract.
        """
        assert context.claims is not None  # guaranteed by applies()

        claims = tuple(context.claims)
        known_claim_ids = frozenset(claim.id for claim in claims)
        judgments = _extract_judgments(context)
        per_claim_adjustments, fallback_adjustment = _extract_confidence_inputs(context)

        per_claim_results = tuple(
            _recommendations_for_claim(
                context,
                claim,
                judgments,
                per_claim_adjustments,
                fallback_adjustment,
                known_claim_ids,
            )
            for claim in claims
        )

        recommendations: list[str] = []
        seen_recommendations: set[str] = set()
        affected_claims: list[Claim] = []
        reasoning_lines: list[str] = []
        evidence_used: list[Any] = []
        seen_evidence_ids: set[int] = set()

        for result in per_claim_results:
            if result.recommendations:
                affected_claims.append(result.claim)
            for recommendation in result.recommendations:
                if recommendation not in seen_recommendations:
                    seen_recommendations.add(recommendation)
                    recommendations.append(recommendation)
            reasoning_lines.extend(result.reasoning_lines)
            for item in result.evidence_used:
                if id(item) not in seen_evidence_ids:
                    seen_evidence_ids.add(id(item))
                    evidence_used.append(item)

        reasoning = (
            " ".join(reasoning_lines)
            if reasoning_lines
            else "No claims were available to recommend action for."
        )

        return RuleResult(
            rule_id=self.id,
            rule_name=self.name,
            triggered=len(recommendations) > 0,
            output_category=OutputCategory.RECOMMENDATION,
            reasoning=reasoning,
            evidence_used=tuple(evidence_used),
            affected_claims=tuple(affected_claims),
            recommendations=tuple(recommendations),
        )


__all__ = ["RecommendationRule"]
