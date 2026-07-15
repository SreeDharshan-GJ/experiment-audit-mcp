"""
Experiment Audit Scientific Reasoning Engine

Module: scientific_rules.confidence_rule

Defines `ConfidenceRule`, the fourth concrete `ScientificRule` (rules.py)
in this pipeline. It implements the aggregation step described in
07_confidence.md ("Chapter 7"), Sections 2-4: given everything already
known about a claim -- its own attributed evidence, and the structured
findings `MissingEvidenceRule` (R001), `ScopeRule` (R002), and
`ContradictionRule` (R003) have already produced for it -- decide how
strongly that combined picture justifies believing the claim, and
report that decision as a traceable `OutputCategory.CONFIDENCE_ADJUSTMENT`
finding (specification, Section 4: "a change to a claim's existing
confidence value ... together with the specific reason for the
adjustment").

**How this differs from R001, R002, and R003.** Each of those three
rules asks a single, narrow structural question and reports it:
whether evidence a claim's *category* expects is absent (R001), whether
the evidence actually attributed to a claim matches its own *declared
scope* (R002), and whether two or more already scope-resolved claims or
evidence items genuinely *conflict* (R003). None of the three weighs
those findings against one another, or against the positive evidence a
claim also has, to say how strongly the claim is, on balance,
justified. That weighing -- 07_confidence.md Section 1's "assessment of
how strongly the available evidence ... justifies accepting a claim" --
is this rule's entire job, and its only job: it is an **aggregator**,
not a fourth independent structural check.

**Per-claim independence, mandatory.** Every signal this rule computes
-- both the strengthening factors read from a claim's own attributed
evidence and the reducing factors read from R001/R002/R003's
already-computed findings -- is scoped to exactly one claim at a time.
A finding that names a specific claim (or set of claims) is counted
only against the claim(s) it actually names, never against every claim
in the `RuleContext` merely because they were evaluated together. In
particular:

- `RuleContext.missing_evidence` entries (R001/R002's combined output)
  are, in this pipeline's normal operation, plain-language `str`
  descriptions rather than a structured record -- see `scope.py`'s and
  `missing_evidence.py`'s own `f"Claim {claim.id} (...)"`-prefixed
  message construction, which both rules use consistently for every
  gap they report. This rule reads that same, already-established
  prefix convention to attribute each gap to the one claim it names,
  rather than treating every gap as context-wide simply because the
  field's declared type (`MissingEvidenceRecord`, not yet a concrete
  class in `evidence.py`) is not what actually flows through it today.
  A future, genuinely structured `MissingEvidenceRecord` carrying its
  own `claim_id` / `claim_ids` attribute is checked for, and preferred
  over string parsing, first -- see `_gap_claim_ids` below. Only a gap
  that names no identifiable claim at all (neither a known `claim_id`
  attribute nor a recognizable `"Claim {id} ("` prefix matching a claim
  actually present in this context) is treated as context-wide and
  counted against every claim, since this rule then has no basis to
  narrow it further and silently dropping a real, already-identified
  gap would understate it.
- `RuleContext.detected_contradictions` entries (R003's output) are
  already a structured `Contradiction` type carrying its own `claims`
  tuple; this rule reads that tuple directly (`_contradiction_counts_for_claim`)
  and never lets a contradiction naming claims C004/C005 count against
  an unrelated claim C001.
- The strengthening factors (reproducibility, statistical support,
  independent sources, traceability) are computed only from the
  `Evidence` bundles a claim's own `evidence_trace` names
  (`_bundles_for_claim` / `_items_for_claim`); a claim with no evidence
  trace, or a trace naming different runs than another claim's, never
  shares another claim's strengthening signal.

**This module never repeats R001, R002, or R003's own analysis.**
Concretely:

- It never re-derives which evidence dimensions a claim's `ClaimCategory`
  expects (`missing_evidence.py`'s `_CATEGORY_EXPECTATIONS`) and never
  re-checks a claim's declared `Scope` against recorded dataset,
  hardware, or configuration facts (`scope.py`'s `_breadth_findings` /
  `_mismatch_findings`). Both of those findings arrive, already
  computed, as `RuleContext.missing_evidence` -- the same field
  `rules.py`'s own docstring describes as "every evidentiary gap
  already identified as relevant to this pass," which this rule
  consumes only to determine which claim(s) each already-identified gap
  names, per `_gap_claim_ids` below -- never to re-decide whether the
  gap itself is real.
- It never re-detects a structural conflict between two claims, or
  within one claim's own evidence (`contradiction.py`'s
  `_cross_claim_findings` / `_within_claim_findings`). Those findings
  arrive, already computed and already advanced through Chapter 8's
  Detected/Investigated/Resolved lifecycle, as
  `RuleContext.detected_contradictions` -- consumed here only as an
  opaque, already-resolved count of unresolved and resolved
  contradictions bearing on each claim, per
  `_contradiction_counts_for_claim` below.

What this rule *does* compute directly from `RuleContext.evidence` is a
second, disjoint set of signals -- 07_confidence.md Section 3's
strengthening factors that R001/R002/R003 have no reason to touch,
because none of those three rules' own questions depend on them:
reproducibility evidence, statistical (multi-seed) support, evidence
independence (more than one distinct source), and end-to-end
traceability of a claim's own `evidence_trace`. These are read the same
way `missing_evidence.py` reads comparable facts (a literal inspection
of `Evidence.seeds`, `Evidence.previous_experiments`, and `Evidence.ref`
against `Claim.evidence_trace`) -- never inferred from a claim's
`statement`, never probabilistic, never NLP.

**Confidence is not correctness, not truth, not a publication
decision.** Per 07_confidence.md Section 2, this rule never asserts
that a claim *is* true, never decides whether a claim should be
published, and never modifies a `Claim` or `Evidence` instance -- it
only reports a per-claim, signed, bounded confidence adjustment
together with the specific, itemized reason for it. Placing a claim at
a named `ConfidenceLevel` (confidence.py's `VERY_LOW` .. `VERY_HIGH`)
from that adjustment, or combining it with whatever confidence value
already existed for the claim, is `confidence.py`'s job --
specifically `ConfidenceAssessor`'s -- not this rule's; per
07_confidence.md Section 5, this methodology assigns no numeric
thresholds to those levels at all, and this rule does not invent any
on their behalf.

**How a per-claim result is reported through a single-valued
`RuleResult`.** `RuleResult.confidence_adjustment` (rules.py) is a
single `float` per evaluation, not one value per claim -- a constraint
of `rules.py` this task does not permit changing. Rather than collapse
every claim's independently-computed adjustment into that one scalar
and calling the result "per-claim" (the prior behavior this revision
replaces), this rule reports the *full* per-claim mapping through the
one channel `rules.py` already reserves for exactly this situation --
`RuleResult.metadata`, "free-form ... detail that does not fit any of
the above." Concretely, `RuleResult.metadata["confidence_adjustments"]`
is a mapping from `Claim.id` to that claim's own, independently
computed `float` adjustment, and `RuleResult.metadata["confidence_reasoning"]`
is a mapping from `Claim.id` to that claim's own itemized reasoning
string. This is not a new, invented contract: `JudgmentRule` (R005) and
`RecommendationRule` (R006) already document, in their own docstrings,
reading `context.metadata["confidence_adjustments"]` as their
*preferred* source for a claim's confidence, falling back to the
single, context-wide `context.metadata["confidence_adjustment"]` only
when no per-claim mapping is available. This rule now populates the
mapping those two rules were already written to prefer. The single
`RuleResult.confidence_adjustment` scalar is still reported, as the
mean of every claim's own adjustment, purely so a caller that only
wants one aggregate figure (e.g. `ScientificReport.summary()`'s
"Aggregate confidence adjustment" line) still receives one -- it is
never again the only place a claim's confidence is recorded, and no
downstream rule is required to read it.

**Determinism, mandatory.** Every signal this rule reads is a fixed,
literal inspection of `RuleContext`, a `Claim`'s `evidence_trace`, and
the already-computed `RuleContext.missing_evidence` /
`RuleContext.detected_contradictions` collections -- counting seeds,
counting distinct `RunRef`s, counting gaps and contradictions that name
a given claim. None of it is probabilistic, none of it calls out to a
model, and re-running this rule against an unchanged `RuleContext`
always yields the same `RuleResult`.

**Architectural constraint, mirrored from every other module in this
package, including `missing_evidence.py`, `scope.py`, and
`contradiction.py`.** This module depends only on `rules.py` (the
framework it plugs into), `claims.py`, `evidence.py`, and
`contradictions.py` (the normalized types it reasons over, reused
as-is, the same three modules `contradiction.py` itself depends on),
and the Python standard library. It has no dependency on FastMCP, MCP
transport, `server.py`, `confidence.py`, `judgment.py`,
`recommendation.py`, `missing_evidence.py`, `scope.py`, or
`contradiction.py` -- confidence aggregation is kept independent of how
any upstream rule computes its own findings, so a future upstream rule
(a `ClaimCategory` this rule has never heard of, a revised
`ScopeRule`) can change freely without this module changing at all, per
this task's explicit "should never know HOW ... work internally"
requirement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from experiment_audit.reasoning.claims import Claim
from experiment_audit.reasoning.contradictions import Contradiction
from experiment_audit.reasoning.evidence import Evidence, EvidenceItem
from experiment_audit.reasoning.rules import (
    OutputCategory,
    RuleContext,
    RuleResult,
    ScientificRule,
)

# ---------------------------------------------------------------------
# Aggregation weights
#
# Each weight below corresponds to exactly one named factor from
# 07_confidence.md Section 3 ("Sources of Confidence") or Section 4
# ("Factors that Reduce Confidence"). Weights are fixed constants, not
# tuned or learned, and are deliberately small in number and in
# magnitude so that no single factor can, by itself, push a claim's
# adjustment to either bound -- consistent with Section 5's own
# statement that "a single missing element ... can be sufficient to
# keep a claim from rising above Moderate regardless of how strong the
# rest of its evidence is," which this rule honors by making every
# reducing factor's penalty comparable in size to the full sum of
# every strengthening factor combined.
# ---------------------------------------------------------------------

#: Section 3, "Reproducibility": a documented, independent attempt to
#: regenerate a result. Proxied, like `missing_evidence.py`'s
#: `_has_reproducibility_evidence`, by at least one linked prior
#: experiment among the claim's own attributed evidence.
_REPRODUCIBILITY_WEIGHT = 0.25

#: Section 3, "Statistical support": an observed effect distinguishable
#: from the variation expected across repeated measurement. Proxied,
#: like `missing_evidence.py`'s `_has_statistical_evidence`, by at
#: least two recorded seeds among the claim's own attributed evidence.
_STATISTICAL_SUPPORT_WEIGHT = 0.25

#: Section 3, "Independent evidence": support drawn from sources that
#: did not influence one another's production. Proxied by the claim's
#: attributed evidence spanning more than one distinct `RunRef`,
#: mirroring `missing_evidence.py`'s `_has_baseline_evidence`
#: "more than one bundle" reading of independence.
_INDEPENDENT_SOURCES_WEIGHT = 0.25

#: Section 3, "Traceability": an evidentiary chain that can be
#: followed without gaps. Satisfied only when the claim declares at
#: least one evidence link and every declared link actually resolves
#: to evidence present in this `RuleContext` -- a broken or empty
#: `evidence_trace` earns nothing here, per Section 9's "a claim whose
#: supporting evidence cannot be traced to its origin cannot be
#: confidently assessed."
_TRACEABILITY_WEIGHT = 0.25

#: Section 4, "Missing evidence": penalty per evidentiary gap
#: (`RuleContext.missing_evidence`) already identified as bearing on a
#: claim -- whether that gap originated from R001's per-category check
#: or R002's scope check, this rule does not need to know which, only
#: that the gap exists and which claim it names. Capped (see
#: `_MISSING_EVIDENCE_PENALTY_CAP`) so a claim with many small,
#: already-known gaps is not driven arbitrarily far below claims with
#: only one or two.
_MISSING_EVIDENCE_PENALTY_PER_GAP = 0.20
_MISSING_EVIDENCE_PENALTY_CAP = 0.60

#: Section 4, "Conflicting evidence": conflicting evidence reduces
#: confidence "more sharply than an equivalent quantity of merely
#: absent evidence," so an unresolved, already-detected contradiction
#: (`RuleContext.detected_contradictions`) is weighted more heavily,
#: per claim, than a single missing-evidence gap.
_UNRESOLVED_CONTRADICTION_PENALTY = 0.35

#: A contradiction already marked resolved is no longer an active
#: disagreement in the record, but Section 7 ("Confidence and
#: Contradictions") is explicit that confidence "must reflect the
#: presence of an unresolved contradiction," implying a resolved one
#: still cost something on the way to resolution; this rule assigns it
#: a small, non-zero penalty rather than treating it as equivalent to
#: a claim that was never contradicted at all.
_RESOLVED_CONTRADICTION_PENALTY = 0.05
_CONTRADICTION_PENALTY_CAP = 0.60


# ---------------------------------------------------------------------
# Evidence attribution
#
# Mirrors `contradiction.py`'s `_items_for_claim` exactly (same
# attribution rule, same rationale), re-declared locally rather than
# imported from that sibling module, per this module's architectural
# constraint that it depends only on `rules.py`, `claims.py`,
# `evidence.py`, and `contradictions.py`.
# ---------------------------------------------------------------------


def _items_for_claim(context: RuleContext, claim: Claim) -> tuple[EvidenceItem, ...]:
    """Every `EvidenceItem` in `context` structurally attributed to
    `claim` -- i.e. whose `source` is one of the `RunRef`s named in
    `claim.evidence_trace`.

    A claim with an empty `evidence_trace` is attributed no items at
    all, the same "no guessing from `statement` or `subject`"
    discipline `contradiction.py`'s function of the same name follows.

    Delegates to `RuleContext.evidence_items_by_sources` (same
    per-context index `contradiction_rule.py`'s function of the same
    name now uses) rather than re-scanning `context.evidence_items()`
    on every call -- `_assess_claim` below calls this once per claim,
    so on a large audit that is the difference between one full
    evidence scan and one full evidence scan per claim.
    """
    if not claim.evidence_trace:
        return ()
    return context.evidence_items_by_sources(claim.evidence_trace)


def _bundles_for_claim(context: RuleContext, claim: Claim) -> tuple[Evidence, ...]:
    """Every `Evidence` bundle in `context` whose `ref` is one of the
    `RunRef`s named in `claim.evidence_trace`.

    The bundle-level counterpart to `_items_for_claim`: some
    confidence factors below (seed counts, `previous_experiments`) live
    on `Evidence` itself rather than on any single `EvidenceItem`, so
    this rule needs the attributed bundles, not just their items.

    Delegates to `RuleContext.evidence_bundles_by_refs` for the same
    once-per-context-index reason `_items_for_claim` now delegates to
    `evidence_items_by_sources`.
    """
    if not claim.evidence_trace:
        return ()
    return context.evidence_bundles_by_refs(claim.evidence_trace)


# ---------------------------------------------------------------------
# Strengthening factors (07_confidence.md Section 3)
#
# Each function below is a narrow, literal predicate over the
# `Evidence` bundles already attributed to one claim -- never over the
# claim's `statement`, never over evidence not attributed to it. None
# of these duplicate a check R001, R002, or R003 already performs: none
# of those three rules asks "how many seeds," "how many distinct
# sources," or "does this specific evidence_trace actually resolve."
# ---------------------------------------------------------------------


def _has_reproducibility_evidence(bundles: tuple[Evidence, ...]) -> bool:
    """Whether at least one bundle attributed to a claim links to a
    prior experiment -- the reproducibility evidence Section 3
    describes, proxied the same way `missing_evidence.py`'s
    `_has_reproducibility_evidence` proxies it for the Reproducibility
    `ClaimCategory` specifically. Here it is read generically, for
    every claim regardless of category, since Section 3 treats
    reproducibility as a universal strengthening factor, not one
    confined to claims that are themselves *about* reproducibility.
    """
    return any(bundle.previous_experiments for bundle in bundles)


def _has_statistical_support(bundles: tuple[Evidence, ...]) -> bool:
    """Whether the bundles attributed to a claim record at least two
    seeds in total -- the statistical support Section 3 describes,
    read the same way `missing_evidence.py`'s
    `_has_statistical_evidence` reads it for the Statistical
    `ClaimCategory`, but again applied generically here.
    """
    return sum(len(bundle.seeds) for bundle in bundles) >= 2


def _has_independent_sources(bundles: tuple[Evidence, ...]) -> bool:
    """Whether the bundles attributed to a claim come from more than
    one distinct run -- Section 3's "independent evidence," read as
    the number of distinct `Evidence.ref` values among the claim's own
    attributed bundles. Ten items drawn from one bundle share a single
    `ref` and therefore a single point of possible failure; items
    drawn from two or more distinct bundles do not.
    """
    return len({bundle.ref for bundle in bundles}) >= 2


def _is_fully_traceable(claim: Claim, context: RuleContext) -> bool:
    """Whether `claim.evidence_trace` is non-empty and every `RunRef`
    it names actually resolves to an `Evidence` bundle present in
    `context` -- Section 3's "traceability," read literally as
    end-to-end link resolution rather than mere non-emptiness. A claim
    that cites a run this context has no evidence for is exactly the
    "cannot be traced to its origin" case Section 9 describes, and
    earns nothing here even though `evidence_trace` is non-empty.

    Uses `RuleContext.resolvable_refs()`, which memoizes the same
    `{bundle.ref for bundle in context.evidence_sequence()}` set this
    function previously rebuilt from scratch on every call -- `_assess_claim`
    calls this once per claim, so that rebuild was previously an
    O(len(evidence_sequence())) cost paid once per claim rather than
    once per rule run.
    """
    if not claim.evidence_trace:
        return False
    resolvable_refs = context.resolvable_refs()
    return all(ref in resolvable_refs for ref in claim.evidence_trace)


# ---------------------------------------------------------------------
# Reducing factors (07_confidence.md Section 4), read from already-
# computed upstream findings, never re-derived
# ---------------------------------------------------------------------

#: Matches the `"Claim {id} ("` prefix `missing_evidence.py` and
#: `scope.py` both use, consistently, for every gap they report (see
#: e.g. `missing_evidence.py`'s `f"Claim {claim.id} ({claim.subject!r}): "`
#: and `scope.py`'s `f"Claim {claim.id} ({claim.subject!r}) declares no
#: scope..."` / `f"Claim {claim.id} ({claim.subject!r}): {finding.description}"`).
#: Captures the claim id as the text between "Claim " and the next
#: whitespace, so it matches regardless of what follows (a parenthesized
#: subject, a colon, or anything else either rule appends).
_GAP_CLAIM_ID_PATTERN = re.compile(r"^Claim\s+(\S+)\s*\(")


def _gap_claim_ids(record: object, known_claim_ids: frozenset[str]) -> frozenset[str] | None:
    """Every `Claim.id` a single already-identified evidentiary gap
    (one entry of `RuleContext.missing_evidence`) names, or `None` if
    this gap names no claim this rule can identify at all -- meaning it
    is a genuinely context-wide gap, to be counted against every claim,
    per this module's docstring.

    Tries three sources, in order, preferring the most structured one
    available:

    1. A `claim_id` attribute (a future, structured
       `MissingEvidenceRecord` naming exactly one claim).
    2. A `claim_ids` attribute (a future, structured record naming
       several claims at once).
    3. For a plain `str` gap -- what `MissingEvidenceRule` and
       `ScopeRule` actually produce today (see this module's docstring,
       "Per-claim independence, mandatory") -- the `Claim.id` named by
       its leading `"Claim {id} ("` prefix, but only when that id
       matches a claim actually present in `known_claim_ids`; a
       prefix-shaped string that does not resolve to a known claim is
       not trusted, since this rule cannot verify it names a real claim
       in this context.

    Never re-derives *why* the gap exists (that remains R001/R002's own
    finding); only ever determines *which claim(s)* it was already
    reported against.
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


def _missing_evidence_counts(
    context: RuleContext, known_claim_ids: frozenset[str]
) -> tuple[dict[str, int], int]:
    """`(per_claim_counts, unattributed_count)` summarizing every gap
    in `RuleContext.missing_evidence` exactly once.

    `per_claim_counts[claim_id]` is how many gaps `_gap_claim_ids`
    attributed directly to that claim; `unattributed_count` is how many
    gaps `_gap_claim_ids` could not attribute to any specific known
    claim at all (a context-wide gap, per this module's docstring,
    "conservatively counted against every claim in `context`").
    `_missing_evidence_count_for_claim` below combines the two for one
    claim in O(1); this function itself is the single O(len(
    context.missing_evidence)) pass every claim's count is drawn from,
    replacing what was previously a fresh, full O(len(
    context.missing_evidence)) scan *per claim* -- the dominant cost,
    on a large audit, of the previous implementation (`ConfidenceRule`
    evaluates every claim independently, so that scan previously ran
    once per claim).
    """
    per_claim: dict[str, int] = {}
    unattributed = 0
    for record in context.missing_evidence:
        claim_ids = _gap_claim_ids(record, known_claim_ids)
        if claim_ids is None:
            unattributed += 1
            continue
        for claim_id in claim_ids:
            per_claim[claim_id] = per_claim.get(claim_id, 0) + 1
    return per_claim, unattributed


def _missing_evidence_count_for_claim(
    claim: Claim, missing_evidence_counts: tuple[dict[str, int], int]
) -> int:
    """How many of `RuleContext.missing_evidence`'s already-identified
    gaps bear on `claim` specifically -- `missing_evidence_counts`'
    per-claim tally (attributed directly to `claim.id`) plus its
    context-wide, unattributed tally (conservatively counted against
    every claim; see `_missing_evidence_counts`' docstring). O(1) given
    the already-built `missing_evidence_counts`, which `evaluate` below
    builds exactly once per rule run rather than once per claim.
    """
    per_claim, unattributed = missing_evidence_counts
    return per_claim.get(claim.id, 0) + unattributed


#: Matches `ContradictionRule`'s (R003) within-claim finding prefix
#: exactly, e.g. `"Claim C1 ('run-001') rests on evidence that
#: conflicts with itself: ..."`. Same shape, same rationale, as
#: `_GAP_CLAIM_ID_PATTERN` above -- reused here rather than
#: re-declared, since both rules independently settled on the same
#: `"Claim {id} ("` convention for a single-claim finding.
_NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN = _GAP_CLAIM_ID_PATTERN

#: Matches `ContradictionRule`'s `_cross_claim_findings` description
#: format exactly, e.g. `"Claim C004 ('subject') and claim C005
#: ('subject') record conflicting values for 'metric.latency_ms'
#: under matched scope: 42.0 vs. 87.0"` (see `contradiction_rule.py`'s
#: `_cross_claim_findings`, the literal f-string `f"Claim {claim_a.id}
#: ({claim_a.subject!r}) and claim {claim_b.id} ({claim_b.subject!r})
#: record conflicting values for {label!r} under matched scope: ..."`).
#: Captures both claim ids so a cross-claim contradiction counts
#: against both parties, never only one. Deliberately checked *before*
#: `_NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN` in `_new_contradiction_claim_ids`:
#: this text also begins with the literal prefix `"Claim {id} ("` that
#: the within-claim pattern matches, so checking the within-claim
#: pattern first would silently capture only the first claim id and
#: drop the second -- exactly the kind of silent, partial attribution
#: this module's docstring says a rule must never produce.
_NEW_CONTRADICTION_CROSS_CLAIM_PATTERN = re.compile(
    r"^Claim\s+(\S+)\s*\([^)]*\)\s+and\s+claim\s+(\S+)\s*\("
)


def _new_contradiction_claim_ids(finding: str) -> frozenset[str]:
    """Every `Claim.id` a single not-yet-carried-forward contradiction
    finding (one entry of `RuleContext.metadata["newly_detected_
    contradictions"]`, forwarded by `pipeline.py` from `ContradictionRule`'s
    (R003) same-pass `RuleResult.contradictions`) names, or an empty
    `frozenset` if this finding's format is not recognized.

    `pipeline.py` only ever forwards `R003` findings whose text does
    not begin with `"Contradiction "` (the carried-forward-known-
    contradiction marker) into this channel, precisely so a
    contradiction already counted via `RuleContext.detected_
    contradictions` is never counted a second time here -- see this
    module's and `pipeline.py`'s docstrings.

    Checks the cross-claim pattern first, deliberately: see
    `_NEW_CONTRADICTION_CROSS_CLAIM_PATTERN`'s own docstring for why
    checking the within-claim pattern first would silently mis-attribute
    a cross-claim finding to only one of its two claims.
    """
    cross = _NEW_CONTRADICTION_CROSS_CLAIM_PATTERN.match(finding)
    if cross is not None:
        return frozenset({cross.group(1), cross.group(2)})
    within = _NEW_CONTRADICTION_WITHIN_CLAIM_PATTERN.match(finding)
    if within is not None:
        return frozenset({within.group(1)})
    return frozenset()


def _contradiction_counts(
    context: RuleContext,
) -> tuple[dict[str, int], dict[str, int]]:
    """`(unresolved_counts, resolved_counts)`, each a mapping from
    `Claim.id` to how many contradictions -- among
    `RuleContext.detected_contradictions` and `RuleContext.metadata
    ["newly_detected_contradictions"]` -- name that claim, tallied in a
    single pass over each collection.

    Mirrors `_contradiction_counts_for_claim`'s previous per-call
    semantics exactly, including de-duplication: a single
    `Contradiction` naming the same claim more than once in its own
    `claims` tuple still counts once against that claim (the same
    `any(c.id == claim.id for c in known.claims)` short-circuit
    behavior the direct-scan implementation had), achieved here by
    tallying the *set* of distinct claim ids each `Contradiction`
    names, once, rather than one increment per occurrence. Built once
    per rule run; `_contradiction_counts_for_claim` below turns this
    into an O(1) per-claim lookup, replacing what was previously a
    fresh, full scan of both collections *per claim*.
    """
    unresolved: dict[str, int] = {}
    resolved: dict[str, int] = {}
    known: Contradiction
    for known in context.detected_contradictions:
        target = resolved if known.is_resolved else unresolved
        for claim_id in {c.id for c in known.claims}:
            target[claim_id] = target.get(claim_id, 0) + 1

    for finding in context.metadata.get("newly_detected_contradictions", ()):
        for claim_id in _new_contradiction_claim_ids(finding):
            unresolved[claim_id] = unresolved.get(claim_id, 0) + 1

    return unresolved, resolved


def _contradiction_counts_for_claim(
    claim: Claim, contradiction_counts: tuple[dict[str, int], dict[str, int]]
) -> tuple[int, int]:
    """`(unresolved_count, resolved_count)` for `claim`, drawn from the
    already-built `contradiction_counts` (see `_contradiction_counts`)
    in O(1), rather than re-scanning `RuleContext.detected_contradictions`
    and `RuleContext.metadata["newly_detected_contradictions"]` for
    every claim.
    """
    unresolved, resolved = contradiction_counts
    return unresolved.get(claim.id, 0), resolved.get(claim.id, 0)


def _clamp(value: float) -> float:
    """`value`, bounded to `RuleResult.confidence_adjustment`'s own
    `[-1.0, 1.0]` contract.
    """
    return max(-1.0, min(1.0, value))


# ---------------------------------------------------------------------
# _ClaimConfidenceAssessment
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ClaimConfidenceAssessment:
    """One claim's confidence aggregation result.

    A small, internal record bundling everything one claim's assessment
    needs to contribute to the overall `RuleResult`: its own signed
    adjustment, the plain-language account of which factors produced
    it, and the `EvidenceItem`s actually inspected in reaching it -- so
    `ConfidenceRule.evaluate` can build the overall `reasoning`,
    per-claim `metadata`, and `evidence_used` from the same underlying
    per-claim assessments without recomputing any of them twice.

    Attributes:
        claim: The `Claim` this assessment concerns.
        adjustment: This claim's own signed adjustment, computed only
            from evidence and findings attributed to this claim,
            already bounded to `[-1.0, 1.0]` by `_clamp`.
        reasoning: The plain-language sentence explaining exactly which
            strengthening and reducing factors applied to this claim
            and how they combined into `adjustment`.
        evidence_used: Every `EvidenceItem` actually inspected while
            computing the strengthening factors for this claim (the
            reducing factors are read from already-computed upstream
            findings, not raw evidence, so they contribute no items
            here).
    """

    claim: Claim
    adjustment: float
    reasoning: str
    evidence_used: tuple[EvidenceItem, ...]


def _assess_claim(
    context: RuleContext,
    claim: Claim,
    missing_evidence_counts: tuple[dict[str, int], int],
    contradiction_counts: tuple[dict[str, int], dict[str, int]],
) -> _ClaimConfidenceAssessment:
    """Aggregate every confidence factor this rule recognizes for
    `claim`, and only `claim`, into one `_ClaimConfidenceAssessment`.

    Computes the four strengthening factors (07_confidence.md Section
    3) directly from `claim`'s own attributed evidence, and reads the
    two reducing factors (Section 4) from only the entries of
    `RuleContext.missing_evidence` and `RuleContext.detected_contradictions`
    that actually name `claim` -- the already-computed outputs of
    `MissingEvidenceRule`, `ScopeRule`, and `ContradictionRule` this
    rule aggregates rather than re-derives, filtered to this one
    claim's own attribution so that no other claim's findings can ever
    influence this claim's adjustment.

    `missing_evidence_counts` and `contradiction_counts` are built once
    by `evaluate` (see `_missing_evidence_counts` / `_contradiction_counts`)
    and passed in rather than recomputed here, so assessing N claims
    performs those two O(len(missing_evidence) + len(detected_contradictions))
    scans once in total, not once per claim.
    """
    bundles = _bundles_for_claim(context, claim)
    items = _items_for_claim(context, claim)

    has_reproducibility = _has_reproducibility_evidence(bundles)
    has_statistical_support = _has_statistical_support(bundles)
    has_independent_sources = _has_independent_sources(bundles)
    is_traceable = _is_fully_traceable(claim, context)

    strengths = 0.0
    strength_lines: list[str] = []
    if has_reproducibility:
        strengths += _REPRODUCIBILITY_WEIGHT
        strength_lines.append("reproducibility evidence is present")
    else:
        strength_lines.append("no reproducibility evidence is present")
    if has_statistical_support:
        strengths += _STATISTICAL_SUPPORT_WEIGHT
        strength_lines.append("statistical support (>=2 seeds) is present")
    else:
        strength_lines.append("statistical support (>=2 seeds) is absent")
    if has_independent_sources:
        strengths += _INDEPENDENT_SOURCES_WEIGHT
        strength_lines.append("evidence spans more than one independent source")
    else:
        strength_lines.append("evidence does not span more than one independent source")
    if is_traceable:
        strengths += _TRACEABILITY_WEIGHT
        strength_lines.append("its evidentiary chain is fully traceable")
    else:
        strength_lines.append("its evidentiary chain is not fully traceable")

    missing_count = _missing_evidence_count_for_claim(claim, missing_evidence_counts)
    unresolved_count, resolved_count = _contradiction_counts_for_claim(claim, contradiction_counts)

    missing_penalty = min(
        _MISSING_EVIDENCE_PENALTY_CAP, _MISSING_EVIDENCE_PENALTY_PER_GAP * missing_count
    )
    contradiction_penalty = min(
        _CONTRADICTION_PENALTY_CAP,
        _UNRESOLVED_CONTRADICTION_PENALTY * unresolved_count
        + _RESOLVED_CONTRADICTION_PENALTY * resolved_count,
    )

    reducing_lines: list[str] = [
        f"{missing_count} missing-evidence gap(s) attributed specifically to this "
        f"claim apply (penalty {missing_penalty:.2f})",
        f"{unresolved_count} unresolved and {resolved_count} resolved "
        f"contradiction(s) naming this claim apply (penalty {contradiction_penalty:.2f})",
    ]

    adjustment = _clamp(strengths - missing_penalty - contradiction_penalty)

    reasoning = (
        f"Claim {claim.id} ({claim.subject!r}): "
        + "; ".join(strength_lines + reducing_lines)
        + f"; net confidence_adjustment={adjustment:.2f}."
    )

    return _ClaimConfidenceAssessment(
        claim=claim,
        adjustment=adjustment,
        reasoning=reasoning,
        evidence_used=items,
    )


# ---------------------------------------------------------------------
# ConfidenceRule
# ---------------------------------------------------------------------


class ConfidenceRule(ScientificRule):
    """Aggregates `MissingEvidenceRule` (R001), `ScopeRule` (R002), and
    `ContradictionRule` (R003)'s already-computed findings, together
    with a small set of confidence factors this rule reads directly
    from a claim's own attributed evidence, into a transparent,
    traceable, **independently computed per claim**
    `OutputCategory.CONFIDENCE_ADJUSTMENT` finding per 07_confidence.md.

    For every `Claim` in `RuleContext.claims`, this rule computes, using
    only evidence and findings attributed to that one claim:

    1. **Strengthening factors** (Section 3), read directly from the
       `Evidence` bundles `claim.evidence_trace` attributes to it:
       reproducibility evidence, statistical (multi-seed) support,
       independent sources, and end-to-end traceability.
    2. **Reducing factors** (Section 4), read from the subset of
       already-computed upstream findings that actually name this
       claim: the count of `RuleContext.missing_evidence` gaps
       attributed to it (R001's and R002's combined contribution,
       attributed per claim via `_gap_claim_ids`), and the count of
       unresolved and resolved `RuleContext.detected_contradictions`
       naming it (R003's contribution, via `Contradiction.claims`).

    These combine, per claim, into one signed, bounded confidence
    adjustment (`_assess_claim`). Because `RuleResult.confidence_adjustment`
    (rules.py) is a single scalar per evaluation rather than one value
    per claim, this rule reports the full per-claim breakdown through
    `RuleResult.metadata["confidence_adjustments"]` (`Claim.id` ->
    adjustment) and `RuleResult.metadata["confidence_reasoning"]`
    (`Claim.id` -> itemized reasoning) -- the exact contract
    `JudgmentRule` (R005) and `RecommendationRule` (R006) already
    document reading as their preferred source for a claim's
    confidence. `RuleResult.confidence_adjustment` itself still carries
    the mean of every claim's own adjustment, purely as a single
    aggregate figure for a caller that wants one (e.g.
    `ScientificReport.summary()`); it is never the only place a claim's
    confidence is recorded. Every factor considered for every claim is
    nonetheless individually itemized in `reasoning`, so the
    traceability 07_confidence.md Section 9 requires -- "it must be
    possible to state, for any assigned level, which specific evidence
    and which specific factors ... justify that level" -- holds for
    each claim on its own, not merely for the aggregate.

    This rule never asserts that a claim is true, never recommends
    publication or any other action, and never modifies `Claim` or
    `Evidence` instances -- see this module's docstring for the full
    boundary. It populates `RuleResult.confidence_adjustment`,
    `reasoning`, `evidence_used`, `affected_claims`, and `metadata`
    only, leaving `contradictions`, `missing_evidence`, and
    `recommendations` at their empty defaults: reporting a
    contradiction or a missing-evidence finding is R001/R002/R003's
    job, already done by the time this rule runs, and this rule does
    not duplicate their output fields with its own restatement of the
    same finding.
    """

    @property
    def id(self) -> str:
        return "R004"

    @property
    def name(self) -> str:
        return "Confidence Aggregation"

    @property
    def description(self) -> str:
        return (
            "Aggregates each claim's own attributed missing-evidence, scope, "
            "and contradiction findings together with directly-observed "
            "reproducibility, statistical-support, independence, and "
            "traceability signals into an independently computed, itemized "
            "confidence adjustment per claim, without re-deriving any of "
            "those upstream findings and without letting one claim's "
            "findings affect another's adjustment."
        )

    @property
    def version(self) -> str:
        return "2.0.0"

    def applies(self, context: RuleContext) -> bool:
        """Relevant whenever `context` has at least one claim to assess.

        With no claims present, there is nothing for this rule to
        aggregate a confidence adjustment for -- the same precondition
        `MissingEvidenceRule.applies`, `ScopeRule.applies`, and
        `ContradictionRule.applies` all use.
        """
        return context.claims is not None and len(context.claims) > 0

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Independently aggregate a confidence adjustment for every
        claim in `context.claims`.

        Assumes `self.applies(context)` has already returned `True`
        (i.e. `context.claims` is non-empty), per `ScientificRule`'s
        two-phase evaluation contract.
        """
        assert context.claims is not None  # guaranteed by applies()

        claims = tuple(context.claims)
        known_claim_ids = frozenset(claim.id for claim in claims)
        missing_evidence_counts = _missing_evidence_counts(context, known_claim_ids)
        contradiction_counts = _contradiction_counts(context)
        assessments = tuple(
            _assess_claim(context, claim, missing_evidence_counts, contradiction_counts)
            for claim in claims
        )

        evidence_used: list[EvidenceItem] = []
        seen_evidence_ids: set[int] = set()
        for assessment in assessments:
            for item in assessment.evidence_used:
                if id(item) not in seen_evidence_ids:
                    seen_evidence_ids.add(id(item))
                    evidence_used.append(item)

        confidence_adjustments = {
            assessment.claim.id: assessment.adjustment for assessment in assessments
        }
        confidence_reasoning = {
            assessment.claim.id: assessment.reasoning for assessment in assessments
        }

        overall_adjustment = (
            _clamp(sum(a.adjustment for a in assessments) / len(assessments))
            if assessments
            else 0.0
        )

        reasoning = (
            " ".join(a.reasoning for a in assessments)
            if assessments
            else "No claims were available to assess for confidence."
        )

        triggered = any(a.adjustment != 0.0 for a in assessments)

        return RuleResult(
            rule_id=self.id,
            rule_name=self.name,
            triggered=triggered,
            output_category=OutputCategory.CONFIDENCE_ADJUSTMENT,
            reasoning=reasoning,
            evidence_used=tuple(evidence_used),
            affected_claims=claims,
            confidence_adjustment=overall_adjustment,
            metadata={
                "confidence_adjustments": confidence_adjustments,
                "confidence_reasoning": confidence_reasoning,
            },
        )


__all__ = ["ConfidenceRule"]