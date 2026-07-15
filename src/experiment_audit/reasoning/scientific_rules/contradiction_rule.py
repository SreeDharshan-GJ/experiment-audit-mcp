"""
Experiment Audit Scientific Reasoning Engine

Module: scientific_rules.contradiction

Defines `ContradictionRule`, the third concrete `ScientificRule`
(rules.py) in this pipeline. It implements the structural detection
step described in 08_contradictions.md ("Chapter 8"), Section 2's
definition of a scientific contradiction, and echoed in the
specification's Section 4 ("Contradiction") output category: whether
two or more evidence items, or two or more claims, are in a genuine,
structural conflict -- one cannot be true, valid, or fully supported
without the other being false, invalid, or unsupported -- *given that
both are already understood under the same scope, conditions, and
terms* (Chapter 8, Section 2's "load-bearing final clause").

**How this differs from `MissingEvidenceRule` (R001) and `ScopeRule`
(R002).** R001 asks whether evidence a claim's category expects is
absent. R002 asks whether the evidence actually attributed to a claim
supports that claim's own declared scope. Neither asks whether two
claims, or two evidence items, actively conflict with one another --
that question, per Chapter 8, Section 2, may only be asked *after*
scope has already been resolved: "[establishing genuine conflict vs.
scope difference] must be performed before any contradiction is
recorded as such." This rule performs exactly that resolved-scope
comparison and reports what it finds. Concretely:

- **Evidence Contradictions** (Chapter 8, Section 3): two
  `EvidenceItem`s, comparable because they record the same named fact
  (the same `EvidenceKind` *and* the same normalized `key` -- two
  facts recorded under different kinds are never the same fact, even
  if their key strings happen to coincide, e.g. a hardware bundle's
  `{"name": "A100"}` and a dataset bundle's `{"name":
  "benchmark-a"}`) and are attributed to claims that share a subject
  and a mutually compatible declared `Scope`, yet report different
  values. This is Example 1's pattern in this rule's task
  description: a "Runtime = 42ms" item and a "Runtime = 87ms" item,
  both attributed to claims about the same subject under the same
  declared conditions, cannot both be an accurate account of the same
  quantity.
- **Claim Contradictions** (Chapter 8, Section 3): the same evidence
  conflict, viewed at the claim level -- two claims, each individually
  traceable to its own evidence, whose attributed evidence
  nonetheless conflicts once their scopes are confirmed compatible.
  This rule reports both the evidence-level and claim-level framing of
  the same finding together, since one is simply the other's
  attribution.
- **Impossible simultaneous states**, within a single claim's own
  attributed evidence: two `EvidenceItem`s sharing the same
  `EvidenceKind` and normalized key but recording different values,
  both cited in support of the *same* claim (Example 3's pattern: a
  claim's own evidence records both a converged and a diverged
  outcome for what is supposed to be one and the same claim). A claim
  cannot rest on evidence that disagrees with itself.

**What this rule never does.** Per Chapter 8, Section 2's own
statement that "two statements that appear to conflict on the surface
but are actually scoped differently are not a contradiction under this
methodology," this rule never reports a conflict between two claims,
or two evidence items, whose declared `Scope`s explicitly diverge on a
named dimension (e.g. one claim's evidence is scoped to CIFAR-10, the
other's to ImageNet) -- Example 2 in this rule's task description.
Nor does this rule ever compare claims about different subjects, or
claims for which scope cannot be determined at all
(`Scope.is_unspecified()`): per the specification's Section 8, "Scope
ambiguity," a rule that proceeds to characterize a relationship as a
genuine contradiction without first resolving scope ambiguity has
violated the Evidence First principle. Such pairs are left
unclassified by this rule -- not silently treated as either
"contradiction" or "no contradiction" -- and are reported, plainly, as
pairs this rule could not evaluate.

Nor does this rule ever compare two `EvidenceItem`s of different
`EvidenceKind`s, even when their normalized `key` strings happen to
coincide -- `hardware.name` and `dataset.name` are two different
facts that never bear on one another, not two recorded values of one
fact, and comparing them would report a coincidence of naming as a
scientific conflict.

**Never NLP, never semantics from wording.** This rule never inspects
a `Claim.statement` and never infers that two claims disagree because
their prose sounds like it points in different directions (e.g. it
never reasons that "faster" and "requires longer inference" are
opposites). Every conflict this rule reports is a literal,
structural comparison of recorded `EvidenceItem` kinds, keys, and
values, nor does it use any probabilistic or model-based reasoning. A
claim's statement is read by this rule exactly once, and only for
identity (the plain-language description a finding is reported under,
so a reader knows which claim's evidence is in conflict) -- never as
an input to the conflict determination itself.

**Known contradictions, carried forward, never re-derived.** Per the
specification's Section 3, a rule reasoning about a claim must take
any already-recorded, relevant `Contradiction` (contradictions.py) as
input rather than rediscovering it or silently ignoring it. This rule
therefore also surfaces every unresolved `Contradiction` in
`RuleContext.detected_contradictions` that names at least one claim in
`RuleContext.claims`, reporting it alongside anything this rule newly
detects -- without recomputing its `status`, `sources`, `quality`, or
`resolution`, all of which remain that `Contradiction`'s own recorded
history, not this rule's to rewrite.

**What this module is not.** It is not a confidence estimator, a
scope verifier, or a judgment engine. It never touches
`RuleResult.confidence_adjustment` (left at its `0.0` default on every
`RuleResult` returned here). It never assigns a `ContradictionCategory`
or `ContradictionStatus`, never constructs a `contradictions.py`
`Contradiction` instance, and never advances one through Chapter 8,
Section 5's lifecycle (Detected -> Investigated -> Explained ->
Resolved/Persisting) -- this rule's entire contribution is the
"Detected" step for newly-found conflicts, reported as the
plain-language strings `RuleResult.contradictions` already provides
for exactly this purpose; investigating, explaining, and resolving a
contradiction are downstream work this rule does not perform.
Recommendations are added only where a detected conflict directly
implies one (verification or replication under matched conditions),
never a broader strategic suggestion.

**Determinism.** Every check this rule performs is a fixed, literal
inspection of `RuleContext`, a `Claim`'s `Scope`, and `EvidenceItem`
kinds/keys/values -- normalized-key grouping *within* a shared
`EvidenceKind`, and literal (case- and whitespace-insensitive, for
strings) value equality. None of it is probabilistic, none of it
calls out to a model, and re-running this rule against an unchanged
`RuleContext` always yields the same `RuleResult`.

**Architectural constraint, mirrored from every other module in this
package, including `missing_evidence.py` and `scope.py`.** This module
depends only on `rules.py` (the framework it plugs into), `claims.py`,
`evidence.py`, and `contradictions.py` (the normalized types it
reasons over, reused as-is), and the Python standard library. It has
no dependency on FastMCP, MCP transport, `server.py`, `confidence.py`,
`judgment.py`, `recommendation.py`, `missing_evidence.py`, `scope.py`,
or any backend implementation -- contradiction detection is kept
independent of confidence estimation, judgment generation, and
recommendation strategy, per this task's explicit architectural
requirement.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from experiment_audit.reasoning.claims import Claim, Scope
from experiment_audit.reasoning.contradictions import Contradiction
from experiment_audit.reasoning.evidence import EvidenceItem, EvidenceKind
from experiment_audit.reasoning.rules import (
    OutputCategory,
    RuleContext,
    RuleResult,
    ScientificRule,
)

# ---------------------------------------------------------------------
# Named scope dimensions
#
# The same fixed, ordered set of `Scope` (claims.py) named fields
# `scope.py`'s `ScopeRule` inspects, re-declared locally rather than
# imported from that sibling rule module, per this module's
# architectural constraint that it depends only on `rules.py`,
# `claims.py`, `evidence.py`, and `contradictions.py`.
# ---------------------------------------------------------------------

_NAMED_SCOPE_DIMENSIONS: tuple[str, ...] = (
    "dataset",
    "model",
    "hardware",
    "evaluation_protocol",
    "software_environment",
)

# A grouping key that identifies "the same recorded fact": an
# `EvidenceItem` is only ever comparable to another `EvidenceItem` that
# shares both its `EvidenceKind` and its normalized `key` -- two items
# recorded under different kinds are two different facts even when
# their key strings happen to coincide (e.g. `hardware.name` vs.
# `dataset.name`), so `EvidenceKind` must be part of this identity, not
# just the key string.
_FactKey = tuple[EvidenceKind, str]


# ---------------------------------------------------------------------
# Structural, literal comparison helpers
#
# Every function below is a narrow, deterministic predicate over
# already-structured values (a `Scope` field, an `EvidenceItem.kind`,
# `.key`, or `.value`). None of them read a `Claim.statement`, and
# none of them perform natural-language processing of any kind.
# ---------------------------------------------------------------------


def _normalize_key(key: str) -> str:
    """A key, normalized for structural comparison: lowercased,
    stripped of surrounding whitespace, and with spaces and hyphens
    folded to underscores.

    Used only to decide whether two `EvidenceItem`s record the *same*
    named fact (e.g. matching `"runtime_ms"` against a recorded key of
    `"Runtime-MS"`) -- a literal, deterministic string transform, not
    natural-language matching: it folds spelling variants of the
    *same* key, never guesses that two differently-named keys mean the
    same thing. This normalization alone is never sufficient to decide
    comparability -- see `_fact_key`, which also requires a matching
    `EvidenceKind`.
    """
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _fact_key(item: EvidenceItem) -> _FactKey:
    """The identity `EvidenceItem`s are grouped and compared by: their
    `EvidenceKind` together with their normalized `key`.

    Two `EvidenceItem`s are only ever the "same recorded fact" -- and
    therefore only ever comparable for a contradiction -- when both of
    these agree. A shared key string alone (e.g. `"name"` on both a
    `hardware` item and a `dataset` item) never establishes identity;
    those are two unrelated facts that merely reused a key name.
    """
    return (item.kind, _normalize_key(item.key))


def _fact_label(fact_key: _FactKey) -> str:
    """The plain-language label a finding is reported under for
    `fact_key`, e.g. `"metric.accuracy"` or `"hardware.gpu_type"` --
    always naming both the `EvidenceKind` and the key, since the kind
    is now part of what makes the fact identifiable.
    """
    kind, key = fact_key
    return f"{kind.value}.{key}"


def _normalize_scope_value(value: str) -> str:
    """A declared `Scope` field value, normalized for structural
    equality comparison: lowercased and stripped of surrounding
    whitespace -- the same normalization `scope.py`'s `ScopeRule` uses
    for its own `Scope`-vs-evidence comparisons, so the two rules treat
    incidental casing or padding consistently.
    """
    return value.strip().lower()


def _values_conflict(value_a: object, value_b: object) -> bool:
    """Whether two recorded `EvidenceItem` values are structurally
    incompatible -- i.e. not literally the same fact, once incidental
    string casing and whitespace are normalized away.

    A purely structural equality check: strings are compared
    case- and whitespace-insensitively; every other value (numbers,
    booleans, and anything else `EvidenceItem.value` may hold) is
    compared by plain `!=`. This never attempts a tolerance-based or
    "close enough" numeric comparison -- two distinct recorded numbers
    for the same fact are exactly the "incompatible recorded values"
    case this rule exists to surface (e.g. Runtime = 42ms vs.
    Runtime = 87ms), not a difference to be smoothed over here.
    """
    if isinstance(value_a, str) and isinstance(value_b, str):
        return value_a.strip().lower() != value_b.strip().lower()
    return value_a != value_b


def _scopes_conflict(scope_a: Scope, scope_b: Scope) -> bool:
    """Whether two claims' declared `Scope`s explicitly diverge on some
    dimension -- a genuine scope difference (Chapter 8, Section 2),
    which precludes treating any evidence disagreement between the two
    claims as a contradiction (this rule's task description, Example
    2: CIFAR-10 vs. ImageNet is a scope difference, not a
    contradiction).

    Compares every named `Scope` dimension, plus every
    `additional_constraints` key the two scopes share, and reports a
    conflict only where *both* scopes declare a value for the same
    dimension or key and those values differ. A dimension left
    undeclared (`None`) on either side is never treated as a conflict
    here -- an undeclared dimension is not a claim to a specific value,
    so it cannot disagree with one; see `_scope_conflict_pairs`
    below for how this function is used alongside the
    `is_unspecified()` precondition.
    """
    for dimension in _NAMED_SCOPE_DIMENSIONS:
        value_a = getattr(scope_a, dimension)
        value_b = getattr(scope_b, dimension)
        if value_a is None or value_b is None:
            continue
        if _normalize_scope_value(value_a) != _normalize_scope_value(value_b):
            return True

    shared_keys = set(scope_a.additional_constraints) & set(scope_b.additional_constraints)
    for key in shared_keys:
        if _normalize_scope_value(scope_a.additional_constraints[key]) != _normalize_scope_value(
            scope_b.additional_constraints[key]
        ):
            return True

    return False


def _items_for_claim(context: RuleContext, claim: Claim) -> tuple[EvidenceItem, ...]:
    """Every `EvidenceItem` in `context` structurally attributed to
    `claim` -- i.e. whose `source` is one of the `RunRef`s named in
    `claim.evidence_trace`.

    This is the same attribution mechanism Chapter 2, Section 9's
    traceability requirement establishes for a `Claim`: an item is
    "this claim's evidence" only because the claim's own
    `evidence_trace` names the run it came from, never because this
    rule guesses at a relationship from a claim's `statement` or
    `subject`. A claim with an empty `evidence_trace` is attributed no
    items at all.

    Delegates to `RuleContext.evidence_items_by_sources`, which
    returns exactly the same items in exactly the same order as
    filtering `context.evidence_items()` directly (this function's
    previous implementation), but via a per-context, once-built index
    rather than a fresh `O(len(evidence_items()))` scan on every call.
    `evaluate` below calls this once per claim, so on a large audit
    (many claims, much evidence) that difference is the difference
    between one full evidence scan and one full evidence scan *per
    claim*.
    """
    if not claim.evidence_trace:
        return ()
    return context.evidence_items_by_sources(claim.evidence_trace)


def _indices_by_subject(claims: tuple[Claim, ...]) -> dict[str, list[int]]:
    """`claims`' indices, grouped by `Claim.subject`, each group's
    index list kept in ascending (original `claims` order) order.

    A small helper for `ContradictionRule.evaluate`'s cross-claim
    conflict search: rather than comparing every pair of claims and
    discarding the ones whose `subject` differs, this lets that search
    jump directly to the (typically much smaller) set of claims that
    share a subject with a given claim -- see the "Performance note"
    in `evaluate` for the full rationale.
    """
    index: dict[str, list[int]] = {}
    for position, claim in enumerate(claims):
        index.setdefault(claim.subject, []).append(position)
    return index


def _grouped_by_fact(items: tuple[EvidenceItem, ...]) -> dict[_FactKey, list[EvidenceItem]]:
    """`items`, grouped by each item's `_fact_key` (its `EvidenceKind`
    together with its normalized `key`), preserving each group's
    recorded order.

    A small, shared building block for both the cross-claim and
    within-claim conflict searches below, so both use exactly the same
    grouping rule -- and therefore the same comparability boundary --
    and cannot drift out of sync with one another. Items of different
    `EvidenceKind`s are never placed in the same group, even when their
    `key` strings coincide, so they can never be compared against one
    another downstream.
    """
    grouped: dict[_FactKey, list[EvidenceItem]] = {}
    for item in items:
        grouped.setdefault(_fact_key(item), []).append(item)
    return grouped


# ---------------------------------------------------------------------
# _ContradictionFinding
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ContradictionFinding:
    """One concrete structural contradiction this rule has found.

    A small, internal record bundling everything one finding needs to
    contribute to a `RuleResult`: its plain-language description, the
    claim(s) it implicates, the evidence item(s) it rests on, and, when
    the finding directly implies one, a recommendation -- so
    `ContradictionRule.evaluate` can assemble `RuleResult.contradictions`,
    `affected_claims`, `evidence_used`, and `recommendations` from the
    same underlying finding without recomputing it four times.

    Attributes:
        description: The plain-language sentence describing this
            specific conflict, reported in `RuleResult.contradictions`.
        claims: The `Claim`(s) this finding implicates.
        evidence_items: The `EvidenceItem`(s) found to be in conflict.
        recommendation: A plain-language suggestion this finding
            directly implies (verification or replication), or `None`
            when no such suggestion is warranted (e.g. a known,
            already-investigated contradiction simply being carried
            forward).
    """

    description: str
    claims: tuple[Claim, ...]
    evidence_items: tuple[EvidenceItem, ...]
    recommendation: str | None = None


# ---------------------------------------------------------------------
# ContradictionRule
# ---------------------------------------------------------------------


class ContradictionRule(ScientificRule):
    """Detects structural contradictions among the claims and evidence
    in a `RuleContext`, per 08_contradictions.md Chapter 8, Section 2,
    and reports them as `OutputCategory.CONTRADICTION` findings.

    For every `Claim` in `RuleContext.claims`, this rule performs three
    independent, purely structural checks:

    1. **Within-claim conflicts.** A single claim's own attributed
       evidence (`Claim.evidence_trace`) is inspected for two
       `EvidenceItem`s that share the same `EvidenceKind` and
       normalized key but record different values -- an "impossible
       simultaneous state," since a claim cannot be supported by
       evidence that disagrees with itself (this rule's task
       description, Example 3).
    2. **Cross-claim conflicts.** For every pair of distinct claims
       that share a `subject` and whose declared `Scope`s do not
       explicitly diverge on any dimension (`_scopes_conflict`
       returns `False`), this rule compares their attributed evidence
       and reports any shared `EvidenceKind`-and-normalized-key
       recorded by both whose values differ -- Example 1's
       "Runtime = 42ms" vs. "Runtime = 87ms" pattern, reported both as
       the underlying evidence conflict and as the claim-level
       contradiction it produces.
    3. **Known, carried-forward contradictions.** Every unresolved
       `Contradiction` (contradictions.py) in
       `RuleContext.detected_contradictions` that names at least one
       claim present in `RuleContext.claims` is surfaced alongside
       this rule's own findings, per the specification's Section 3
       requirement that a rule take relevant, already-recorded
       contradictions as input rather than rediscovering or ignoring
       them. This rule never recomputes or overwrites that
       `Contradiction`'s own recorded `status`, `sources`, or
       `quality`.

    A pair of claims is never compared for check 2 when either claim's
    `Scope.is_unspecified()` is `True` -- scope ambiguity (Chapter 8,
    Section 2; specification, Section 8) precludes concluding either
    "contradiction" or "no contradiction," and this rule reports such
    pairs, plainly, as ones it could not evaluate rather than silently
    picking one answer. A pair whose scopes explicitly diverge on a
    named dimension is also never reported as a contradiction -- per
    Chapter 8, Section 2, that is a scope difference, a categorically
    different finding this rule does not conflate with a genuine
    conflict.

    Two `EvidenceItem`s are never compared, in either check, unless
    they share both their `EvidenceKind` and their normalized `key` --
    a `hardware.name` item and a `dataset.name` item are two different
    facts, not two recorded values of one fact, and reusing a key
    string across kinds never makes them comparable.

    This rule never infers a conflict from a claim's free-form
    `statement`, never uses natural-language processing, and never
    invents evidence that was not supplied -- see this module's
    docstring for the precise boundary. It never sets
    `RuleResult.confidence_adjustment` away from its `0.0` default and
    never constructs, categorizes, or advances a `contradictions.py`
    `Contradiction` through its lifecycle: detecting a structural
    conflict and reporting it as a plain-language finding is this
    rule's entire contribution; investigating, explaining, and
    resolving it belong to later stages.
    """

    @property
    def id(self) -> str:
        return "R003"

    @property
    def name(self) -> str:
        return "Structural Contradiction Detection"

    @property
    def description(self) -> str:
        return (
            "Detects structural contradictions -- incompatible recorded values, "
            "mutually exclusive observations, and conflicting experimental "
            "outcomes -- among the claims and evidence in scope, after "
            "confirming the parties being compared share a compatible, "
            "resolved scope, and surfaces any relevant already-known "
            "contradictions alongside anything newly detected."
        )

    @property
    def version(self) -> str:
        return "1.0.0"

    def applies(self, context: RuleContext) -> bool:
        """Relevant whenever `context` has at least one claim to check.

        With no claims present, there is no claim-attributed evidence
        for this rule to compare, and no claim `RuleContext.detected_
        contradictions` could possibly name -- the same precondition
        `MissingEvidenceRule.applies` and `ScopeRule.applies` use.
        """
        return context.claims is not None and len(context.claims) > 0

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Check every claim in `context.claims`, and every pair of
        them, for structural contradictions, and surface any relevant,
        already-known contradiction from `context.detected_contradictions`.

        Assumes `self.applies(context)` has already returned `True`
        (i.e. `context.claims` is non-empty), per `ScientificRule`'s
        two-phase evaluation contract.
        """
        assert context.claims is not None  # guaranteed by applies()

        claims = tuple(context.claims)
        items_by_claim: dict[str, tuple[EvidenceItem, ...]] = {
            claim.id: _items_for_claim(context, claim) for claim in claims
        }

        findings: list[_ContradictionFinding] = []
        reasoning_lines: list[str] = []

        # --- 1. Within-claim conflicts -----------------------------------
        for claim in claims:
            within_findings = _within_claim_findings(claim, items_by_claim[claim.id])
            findings.extend(within_findings)
            if within_findings:
                reasoning_lines.append(
                    f"Claim {claim.id} ({claim.subject!r}) rests on evidence that "
                    "conflicts with itself: "
                    + "; ".join(finding.description for finding in within_findings)
                    + "."
                )

        # --- 2. Cross-claim conflicts, scope-resolved first ---------------
        #
        # Performance note (audit #8): this rule's task description
        # requires checking every *same-subject* pair of claims, which
        # is genuinely O(k^2) within one subject's k claims -- that is
        # inherent to the check itself and not something this method
        # can avoid without changing what it verifies. What is *not*
        # inherent is re-visiting every (i, j) pair across the *entire*
        # claim set only to discard almost all of them on the
        # `claim_a.subject != claim_b.subject` check: with many
        # distinct subjects (the common case on a large audit), that
        # wastes O(n^2) work down to O(1) per pair.
        #
        # `_indices_by_subject` groups claim indices by `subject` in a
        # single O(n) pass, each group's index list kept in ascending
        # (original claims-order) order. For each `claim_a` at position
        # `index`, `bisect_right` finds where `index` itself sits in
        # its own subject's sorted index list, and everything after
        # that point is exactly the set of later, same-subject indices
        # -- the same `claim_b` candidates the original nested loop
        # would eventually reach, visited in the same ascending order,
        # just without stepping through every non-matching claim in
        # between. This changes no output: the exact same pairs are
        # compared, in the exact same order, so `reasoning_lines` and
        # `findings` end up identical to the direct O(n^2) nested loop.
        indices_by_subject = _indices_by_subject(claims)
        # Performance note (audit #8): `_grouped_by_fact(items_by_claim[
        # claim_id])` depends only on that one claim's own attributed
        # items -- never on which other claim it is being compared
        # against -- yet a claim sharing its subject with `k` other
        # claims previously had its evidence regrouped by fact `k`
        # separate times (once per pair), all producing an identical
        # result. `grouped_by_claim` computes that grouping at most once
        # per claim, lazily, the first time the claim participates in
        # any cross-claim comparison, and reuses it for every subsequent
        # pair -- collapsing what was an O(bucket_size) redundant
        # regrouping per claim down to O(1) amortized.
        grouped_by_claim: dict[str, dict[_FactKey, list[EvidenceItem]]] = {}
        for index, claim_a in enumerate(claims):
            same_subject_indices = indices_by_subject[claim_a.subject]
            start = bisect.bisect_right(same_subject_indices, index)
            for later_index in same_subject_indices[start:]:
                claim_b = claims[later_index]

                if claim_a.scope.is_unspecified() or claim_b.scope.is_unspecified():
                    reasoning_lines.append(
                        f"Claims {claim_a.id} and {claim_b.id} share subject "
                        f"{claim_a.subject!r} but at least one declares no scope "
                        "at all; scope ambiguity (specification, Section 8) "
                        "precludes checking this pair for a contradiction."
                    )
                    continue

                if _scopes_conflict(claim_a.scope, claim_b.scope):
                    reasoning_lines.append(
                        f"Claims {claim_a.id} and {claim_b.id} share subject "
                        f"{claim_a.subject!r} but declare explicitly different "
                        "scope; per Chapter 8, Section 2 this is a scope "
                        "difference, not a contradiction."
                    )
                    continue

                grouped_a = grouped_by_claim.get(claim_a.id)
                if grouped_a is None:
                    grouped_a = _grouped_by_fact(items_by_claim[claim_a.id])
                    grouped_by_claim[claim_a.id] = grouped_a
                grouped_b = grouped_by_claim.get(claim_b.id)
                if grouped_b is None:
                    grouped_b = _grouped_by_fact(items_by_claim[claim_b.id])
                    grouped_by_claim[claim_b.id] = grouped_b

                cross_findings = _cross_claim_findings(claim_a, claim_b, grouped_a, grouped_b)
                findings.extend(cross_findings)
                if cross_findings:
                    reasoning_lines.append(
                        f"Claims {claim_a.id} and {claim_b.id} share subject "
                        f"{claim_a.subject!r} and a compatible declared scope, "
                        "but their attributed evidence conflicts: "
                        + "; ".join(finding.description for finding in cross_findings)
                        + "."
                    )

        # --- 3. Known, carried-forward contradictions ----------------------
        claim_ids = {claim.id for claim in claims}
        known: Contradiction
        for known in context.detected_contradictions:
            if known.is_resolved:
                continue
            relevant_claims = tuple(c for c in known.claims if c.id in claim_ids)
            if not relevant_claims:
                continue
            findings.append(
                _ContradictionFinding(
                    description=(
                        f"Contradiction {known.id} ({known.status.value}) remains "
                        "unresolved and bears on "
                        + ", ".join(c.id for c in relevant_claims)
                        + " -- carried forward per the specification's Section 3 "
                        "input requirement, not re-derived by this rule."
                    ),
                    claims=relevant_claims,
                    evidence_items=known.evidence_items,
                )
            )
            reasoning_lines.append(
                f"Contradiction {known.id} was already recorded as "
                f"{known.status.value} and bears on this context's claims; it is "
                "carried forward, not re-evaluated."
            )

        contradictions: list[str] = []
        recommendations: list[str] = []
        evidence_used: list[EvidenceItem] = []
        seen_evidence_ids: set[int] = set()
        affected_claims: list[Claim] = []
        seen_claim_ids: set[str] = set()

        for finding in findings:
            contradictions.append(finding.description)
            if finding.recommendation is not None:
                recommendations.append(finding.recommendation)
            for item in finding.evidence_items:
                if id(item) not in seen_evidence_ids:
                    seen_evidence_ids.add(id(item))
                    evidence_used.append(item)
            for claim in finding.claims:
                if claim.id not in seen_claim_ids:
                    seen_claim_ids.add(claim.id)
                    affected_claims.append(claim)

        triggered = len(contradictions) > 0
        if not reasoning_lines:
            reasoning_lines.append(
                "No claims shared a subject and a resolved, compatible scope, "
                "so no contradiction could be evaluated."
            )
        reasoning = " ".join(reasoning_lines)

        return RuleResult(
            rule_id=self.id,
            rule_name=self.name,
            triggered=triggered,
            output_category=OutputCategory.CONTRADICTION,
            reasoning=reasoning,
            evidence_used=tuple(evidence_used),
            affected_claims=tuple(affected_claims),
            contradictions=tuple(contradictions),
            recommendations=tuple(recommendations),
        )


# ---------------------------------------------------------------------
# Per-claim / per-pair finding logic
#
# Kept as module-level functions, rather than methods, mirroring
# `scope.py`'s preference for small, independently testable functions
# over a monolithic `evaluate` body.
# ---------------------------------------------------------------------


def _within_claim_findings(
    claim: Claim, items: tuple[EvidenceItem, ...]
) -> list[_ContradictionFinding]:
    """Every "impossible simultaneous state" finding for `claim`: a
    pair of distinct `EvidenceItem`s, both attributed to `claim`, that
    share the same `EvidenceKind` and normalized key but record
    different values.

    A claim's own evidence disagreeing with itself is a structural
    defect regardless of any other claim -- this check never needs a
    second claim to compare against, and never depends on scope, since
    every item here is already attributed to one and the same claim.
    Items of different `EvidenceKind`s are never compared, even when
    their `key` strings coincide.
    """
    findings: list[_ContradictionFinding] = []
    grouped = _grouped_by_fact(items)

    for fact_key, group in grouped.items():
        label = _fact_label(fact_key)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item_a, item_b = group[i], group[j]
                if not _values_conflict(item_a.value, item_b.value):
                    continue
                findings.append(
                    _ContradictionFinding(
                        description=(
                            f"Claim {claim.id} ({claim.subject!r}) cites conflicting "
                            f"evidence for {label!r}: {item_a.value!r} and "
                            f"{item_b.value!r} cannot both hold"
                        ),
                        claims=(claim,),
                        evidence_items=(item_a, item_b),
                        recommendation=(
                            f"Re-inspect the evidence attributed to claim {claim.id} "
                            f"for {label!r}: it records two different values for what "
                            "should be a single fact, and the claim cannot rest on "
                            "both."
                        ),
                    )
                )

    return findings


def _cross_claim_findings(
    claim_a: Claim,
    claim_b: Claim,
    grouped_a: dict[_FactKey, list[EvidenceItem]],
    grouped_b: dict[_FactKey, list[EvidenceItem]],
) -> list[_ContradictionFinding]:
    """Every cross-claim contradiction between `claim_a` and `claim_b`:
    an `EvidenceKind`-and-normalized-key pair recorded by both claims'
    attributed evidence, with a different value on each side.

    Only called once the caller has already confirmed `claim_a` and
    `claim_b` share a `subject` and a mutually compatible, resolved
    `Scope` -- this function performs no scope reasoning of its own,
    only the literal kind/key/value comparison Chapter 8, Section 3's
    "Evidence Contradictions" and "Claim Contradictions" categories
    both reduce to once scope is settled. Items of different
    `EvidenceKind`s are never compared, even when their `key` strings
    coincide (e.g. `hardware.name` vs. `dataset.name`).

    Takes `claim_a`'s and `claim_b`'s evidence already grouped by fact
    (`_grouped_by_fact`), rather than their raw `EvidenceItem` tuples,
    so the caller can compute each claim's own grouping once and reuse
    it across every pair that claim appears in -- see `evaluate`'s
    "Performance note" on `grouped_by_claim` below. `_grouped_by_fact`
    depends only on one claim's own attributed items, never on which
    other claim it is being compared against, so regrouping it fresh
    for every pair (this function's previous signature took raw items
    and grouped them internally) was pure, avoidable duplicated work
    whenever a claim shares its subject with more than one other claim.
    """
    findings: list[_ContradictionFinding] = []

    shared_facts = sorted(
        set(grouped_a) & set(grouped_b), key=lambda fact_key: _fact_label(fact_key)
    )
    for fact_key in shared_facts:
        label = _fact_label(fact_key)
        for item_a in grouped_a[fact_key]:
            for item_b in grouped_b[fact_key]:
                if not _values_conflict(item_a.value, item_b.value):
                    continue
                findings.append(
                    _ContradictionFinding(
                        description=(
                            f"Claim {claim_a.id} ({claim_a.subject!r}) and claim "
                            f"{claim_b.id} ({claim_b.subject!r}) record conflicting "
                            f"values for {label!r} under matched scope: "
                            f"{item_a.value!r} vs. {item_b.value!r}"
                        ),
                        claims=(claim_a, claim_b),
                        evidence_items=(item_a, item_b),
                        recommendation=(
                            f"Verify or replicate {label!r} for claims {claim_a.id} "
                            f"and {claim_b.id} under identical conditions before "
                            "trusting either recorded value on its own."
                        ),
                    )
                )

    return findings


__all__ = ["ContradictionRule"]