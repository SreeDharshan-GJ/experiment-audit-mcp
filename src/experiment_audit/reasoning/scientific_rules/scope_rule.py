"""
Experiment Audit Scientific Reasoning Engine

Module: scientific_rules.scope

Defines `ScopeRule`, the second concrete `ScientificRule` (rules.py) in
this pipeline. It implements the check named in 02_claims.md, Chapter
2, Section 5 ("Claim Scope") and echoed in the specification's Section
3 ("Scope") and Section 8 ("Scope ambiguity"): whether the evidence
actually available for a claim supports the claim's own *declared*
scope (claims.py's `Scope`, attached to every `Claim`), as opposed to
some broader scope the claim's category or wording might otherwise be
read to assert.

**How this differs from `MissingEvidenceRule` (R001).** R001 asks "does
evidence exist for every dimension this claim's *category* expects,
full stop" -- a category-driven check that never looks at the claim's
own `Scope` object at all. This rule asks a narrower, more literal
question: "does the evidence actually attributed to this claim match,
value for value, the specific conditions this claim's own `Scope`
already declares." Concretely:

- Where `Scope` leaves a scope-defining dimension undeclared
  (`None`) for a category whose whole assertion depends on breadth
  along that dimension (per 02_claims.md, Section 5's own worked
  example: a generalization claim is, by definition, "beyond what was
  directly measured"), this rule checks whether the evidence actually
  spans more than one value along that dimension. Evidence pinned to a
  single value cannot warrant a claim that declines to say which value
  it is limited to.
- Where `Scope` *does* declare a specific value for a dimension (e.g.
  `hardware="A100"`), this rule checks whether that specific value is
  actually attested by the evidence the claim rests on -- not merely
  that *some* evidence of that kind exists (R001's concern), but that
  the evidence was gathered under the exact condition the claim says
  it was.

Both checks implement the same underlying principle, 02_claims.md
Section 5's own statement of it: "a claim may not be evaluated as
though it applies more broadly than its evidence was collected to
support." A **Scope Violation**, as this rule defines it, is exactly
that: the scope a claim's own `Scope` declares (or fails to narrow) is
broader than the scope the evidence actually attributed to it
demonstrates.

**What this module is not.** It is not a confidence estimator, a
contradiction detector, or a judgment engine, for the same reasons
`missing_evidence.py`'s module docstring gives for R001, and this rule
follows that precedent exactly: it never touches
`RuleResult.confidence_adjustment` (left at its `0.0` default on every
`RuleResult` returned here) and never populates
`RuleResult.contradictions`. The specification (Section 8, "Scope
ambiguity") is explicit that resolving whether a claim and its evidence
share the same scope is itself a *precondition* a rule must satisfy
before any relationship between them may be characterized as a genuine
contradiction -- this rule performs exactly that precondition check
and no more. Whether an unresolved scope violation should, in turn,
weaken confidence, register as a contradiction, or block publication is
downstream work for `confidence.py`, `contradictions.py`, and
`judgment.py`, never this rule's to decide. This rule reports its
finding under `OutputCategory.MISSING_EVIDENCE`, the same category
`MissingEvidenceRule` uses: a scope violation is, at bottom, evidence
that the *specific, declared* scope of a claim requires but that the
available evidence does not supply -- see `evaluate`'s docstring for
the full justification of that choice, including why the more
consequential categories (`CONTRADICTION`, `JUDGMENT`,
`RECOMMENDATION`) were deliberately not used.

**Scope ambiguity, handled explicitly, never assumed away.** Per the
specification's Section 8, a rule that cannot determine a claim's
declared scope must not proceed as though a default scope applied. A
claim whose `Scope.is_unspecified()` is `True` -- nothing declared on
any dimension -- is therefore never checked for a violation by this
rule; it is reported, plainly, as a claim this rule could not evaluate
for scope, per that same section. This rule never treats "nothing
declared" as either "unbounded, hence violated" or "unbounded, hence
fine" -- both would be silently assuming a default this rule has no
basis for.

**What "violation" means here, precisely.** A dimension counts as
violated only when the comparison is a *structural* one the framework
can verify directly from `RuleContext` and the claim's own `Scope` --
comparing recorded dataset, hardware, and configuration facts against
declared `Scope` fields and `Scope.additional_constraints` entries,
using literal (case- and whitespace-insensitive) value equality. This
rule never infers a mismatch from a claim's free-form `statement` text,
never uses natural-language processing, and never invents evidence
that was not supplied: a dimension with no evidence recorded for it at
all is left unchecked by this rule (that is a coverage gap, R001's
concern, not a scope mismatch -- this rule reports only on dimensions
where evidence for that dimension actually exists but does not match
what `Scope` declares).

**Determinism.** Every check this rule performs is a fixed, literal
inspection of `RuleContext` and a `Claim`'s `Scope` -- counting distinct
dataset and hardware identifiers, matching declared `Scope` field
values and `additional_constraints` entries against recorded
configuration facts by normalized key and value. None of it is
probabilistic, none of it calls out to a model, and re-running this
rule against an unchanged `RuleContext` always yields the same
`RuleResult`s.

**Architectural constraint, mirrored from every other module in this
package, including `missing_evidence.py`.** This module depends only on
`rules.py` (the framework it plugs into), `evidence.py` and `claims.py`
(the normalized types it reasons over, reused as-is), and the Python
standard library. It has no dependency on FastMCP, MCP transport,
`server.py`, `missing_evidence.py`, or any backend implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from experiment_audit.reasoning.claims import Claim, ClaimCategory, Scope
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
# `Scope` (claims.py, Chapter 2 Section 5) declares five named fields
# describing the conditions a claim's evidence was actually gathered
# under, plus an open-ended `additional_constraints` mapping for
# further conditions not covered by the named fields. This tuple fixes
# the named fields this rule inspects, in the same order `Scope`
# itself declares them, so the mapping between a dimension's name and
# its `Scope` attribute is a simple `getattr`, never a lookup table
# that could drift out of sync with `Scope`'s own declaration.
# ---------------------------------------------------------------------

_NAMED_SCOPE_DIMENSIONS: tuple[str, ...] = (
    "dataset",
    "model",
    "hardware",
    "evaluation_protocol",
    "software_environment",
)

# ---------------------------------------------------------------------
# Which EvidenceKind categories bear on which dimension
#
# Used only for `RuleResult.evidence_used` bookkeeping -- which
# `EvidenceItem`s this rule reports as "inspected" for a given
# dimension -- mirroring `missing_evidence.py`'s
# `_EvidenceExpectation.kinds` convention exactly. `dataset` and
# `hardware` map to their own dedicated `EvidenceKind`; the three
# dimensions this rule can only locate via a keyed configuration fact
# (`model`, `evaluation_protocol`, `software_environment`, and any
# `additional_constraints` entry) map to `EvidenceKind.CONFIG`, the
# kind `evidence.py` uses for exactly this sort of recorded setting.
# ---------------------------------------------------------------------

_DIMENSION_KINDS: dict[str, tuple[EvidenceKind, ...]] = {
    "dataset": (EvidenceKind.DATASET,),
    "hardware": (EvidenceKind.HARDWARE,),
    "model": (EvidenceKind.CONFIG,),
    "evaluation_protocol": (EvidenceKind.CONFIG,),
    "software_environment": (EvidenceKind.CONFIG,),
}

_ADDITIONAL_CONSTRAINT_KINDS: tuple[EvidenceKind, ...] = (
    EvidenceKind.DATASET,
    EvidenceKind.CONFIG,
)

# ---------------------------------------------------------------------
# Breadth-sensitive dimensions, by category
#
# Chapter 2, Section 4's own statement of the Generalization category
# ("assertions that a method's behavior ... extends to other
# conditions not directly tested") and Section 5's worked example ("a
# performance claim established on one dataset is a claim about that
# dataset; extending it ... is not a restatement of the same claim but
# the formulation of a new, broader claim that requires its own
# evidence") together fix exactly one structurally checkable case: a
# Generalization claim that leaves `Scope.dataset` undeclared is, by
# the category's own definition, asserting applicability beyond a
# single dataset -- and that assertion requires evidence spanning more
# than one dataset to be warranted at that breadth. No other category
# in claims.py's `ClaimCategory` makes an equally explicit, named
# textual commitment to "extends beyond what a single value of a
# scope dimension covers" tied to one of `Scope`'s own named fields,
# so this mapping is deliberately narrow rather than a guessed
# generalization of the pattern to every category. A claim that wants
# a similarly strict breadth obligation on a dimension this mapping
# does not cover (e.g. a Robustness claim's perturbation condition)
# expresses that instead through `Scope.additional_constraints`,
# checked by the separate, category-independent mismatch logic below.
# ---------------------------------------------------------------------

_BREADTH_SENSITIVE_DIMENSIONS: dict[ClaimCategory, tuple[str, ...]] = {
    ClaimCategory.GENERALIZATION: ("dataset",),
}


# ---------------------------------------------------------------------
# Structural value extraction
#
# Each function below is a narrow, literal predicate over a
# `RuleContext`: "which distinct values, if any, does the evidence
# actually record for this dimension." Every function reasons only
# over `RuleContext.evidence` (via `evidence_sequence()` /
# `evidence_items()`), never over `Claim.statement`, so a claim's own
# prose can never influence what this rule concludes the evidence
# shows.
# ---------------------------------------------------------------------


def _normalize_key(key: str) -> str:
    """A key, normalized for structural comparison: lowercased,
    stripped of surrounding whitespace, and with spaces and hyphens
    folded to underscores.

    Used only to compare a `Scope` dimension name (or an
    `additional_constraints` key) against a recorded configuration
    key's spelling -- e.g. matching `"evaluation_protocol"` against a
    recorded key of `"Evaluation-Protocol"`. This is a literal,
    deterministic string transform, not natural-language matching: it
    folds spelling variants of the *same* key, never guesses that two
    differently-named keys mean the same thing.
    """
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_value(value: str) -> str:
    """A value, normalized for structural equality comparison:
    lowercased and stripped of surrounding whitespace.

    Used to compare a `Scope`-declared value against a recorded
    evidence value without being tripped up by incidental casing or
    padding -- a literal equality check on normalized strings, never a
    fuzzy or semantic comparison.
    """
    return value.strip().lower()


def _distinct_bundle_dict_identifiers(context: RuleContext, attr_name: str) -> set[str]:
    """Every distinct identifier recorded under `attr_name` (`"dataset"`
    or `"hardware"`) across `context`'s evidence bundles.

    Mirrors `missing_evidence.py`'s `_distinct_dataset_identifiers`
    exactly, generalized to the one other bundle-level dict attribute
    this rule needs (`hardware`), since both attributes follow the
    same "prefer a `\"name\"` key, else fall back to the dict's full,
    sorted content" convention. Reads the attribute defensively via
    `getattr` so a bundle type that does not (yet) carry a given
    attribute contributes no identifiers rather than raising.
    """
    identifiers: set[str] = set()
    for bundle in context.evidence_sequence():
        value = getattr(bundle, attr_name, None)
        if not value:
            continue
        name = value.get("name") if isinstance(value, Mapping) else None
        if name is not None:
            identifiers.add(str(name))
        elif isinstance(value, Mapping):
            identifiers.add(repr(sorted(value.items(), key=lambda kv: kv[0])))
        else:
            identifiers.add(str(value))
    return identifiers


def _distinct_keyed_values(context: RuleContext, key: str) -> set[str]:
    """Every distinct value recorded, anywhere in `context`'s evidence,
    under a configuration or item key whose normalized spelling
    (`_normalize_key`) matches `key`.

    Searches two structural sources per evidence bundle: its run's
    recorded configuration (`bundle.run.config`, when a run is
    attached) and its individual `EvidenceItem`s' own `key`/`value`
    pairs -- the same two sources `missing_evidence.py`'s
    `_has_scalability_evidence` and `_has_robustness_evidence` already
    draw on for comparable keyed lookups. Used for every named `Scope`
    dimension this rule cannot resolve from a dedicated bundle
    attribute (`model`, `evaluation_protocol`, `software_environment`)
    and for every `Scope.additional_constraints` entry, so a single
    mechanism covers both the fixed and the open-ended parts of
    `Scope`.
    """
    normalized_target = _normalize_key(key)
    values: set[str] = set()
    for bundle in context.evidence_sequence():
        if bundle.run is not None:
            for config_key, config_value in bundle.run.config.items():
                if _normalize_key(config_key) == normalized_target:
                    values.add(str(config_value))
        for item in bundle.items:
            if _normalize_key(item.key) == normalized_target:
                values.add(str(item.value))
    return values


def _evidence_values_for_dimension(context: RuleContext, dimension: str) -> set[str]:
    """Every distinct value `context`'s evidence actually records for
    the named `Scope` dimension `dimension`.

    Dispatches to the bundle-level dict lookup (`_distinct_bundle_dict_identifiers`)
    for `"dataset"` and `"hardware"`, the two dimensions `evidence.py`
    represents as a dedicated bundle attribute, and to the generic
    keyed lookup (`_distinct_keyed_values`) for every other named
    dimension, which `evidence.py` represents only as a configuration
    or item-level fact rather than a bundle attribute of its own.
    """
    if dimension in ("dataset", "hardware"):
        return _distinct_bundle_dict_identifiers(context, dimension)
    return _distinct_keyed_values(context, dimension)


def _cached_evidence_values_for_dimension(
    context: RuleContext, dimension: str, cache: dict[str, set[str]]
) -> set[str]:
    """`_evidence_values_for_dimension(context, dimension)`, memoized
    in `cache` for the lifetime of one `ScopeRule.evaluate` call.

    Performance note (audit #8): `_evidence_values_for_dimension` is a
    pure function of `(context, dimension)` -- it never reads `claim`
    or `scope` -- yet `_breadth_findings` and `_mismatch_findings` both
    call it once per claim per dimension. Every claim that leaves the
    same dimension undeclared (breadth) or declares the same dimension
    (mismatch) triggers an identical, independently-recomputed
    O(len(evidence_sequence())) scan. Since `_NAMED_SCOPE_DIMENSIONS`
    and `_BREADTH_SENSITIVE_DIMENSIONS` name a small, fixed set of
    dimensions (not one per claim), caching by `dimension` collapses
    that to one scan per distinct dimension actually consulted, no
    matter how many claims share it -- on a large audit, the dominant
    remaining cost this rule had after evidence-item lookups were
    themselves indexed.
    """
    values = cache.get(dimension)
    if values is None:
        values = _evidence_values_for_dimension(context, dimension)
        cache[dimension] = values
    return values


def _cached_distinct_keyed_values(
    context: RuleContext, key: str, cache: dict[str, set[str]]
) -> set[str]:
    """`_distinct_keyed_values(context, key)`, memoized in `cache` for
    the lifetime of one `ScopeRule.evaluate` call.

    Same rationale as `_cached_evidence_values_for_dimension` above,
    for `scope.additional_constraints`' free-form keys: distinct
    claims frequently declare the same constraint key (e.g. every
    claim in a sweep might declare `batch_size`), and each occurrence
    previously re-scanned every evidence bundle's `run.config` and
    every `EvidenceItem` from scratch.
    """
    values = cache.get(key)
    if values is None:
        values = _distinct_keyed_values(context, key)
        cache[key] = values
    return values


def _items_of_kinds(context: RuleContext, kinds: tuple[EvidenceKind, ...]) -> list[EvidenceItem]:
    """Every `EvidenceItem` in `context` whose `kind` is in `kinds`.

    A local helper, identical in shape and purpose to
    `missing_evidence.py`'s function of the same name, so
    `ScopeRule.evaluate` can populate `RuleResult.evidence_used` with
    exactly the items each dimension check actually bears on, without
    importing a private helper from a sibling module.

    Delegates to `RuleContext.evidence_items_by_kinds`, which answers
    this same filter using a per-context, once-built index rather than
    a fresh O(len(evidence_items())) scan -- `ScopeRule.evaluate` calls
    this once per claim per scope dimension checked, so on a large
    audit that indexing is the difference between one full evidence
    scan and one full evidence scan per claim per dimension.
    """
    return list(context.evidence_items_by_kinds(kinds))


# ---------------------------------------------------------------------
# _ScopeFinding
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ScopeFinding:
    """One concrete scope violation this rule has found for one claim.

    A small, internal record used only to accumulate a violation's
    plain-language description and its matching recommendation
    together, so `ScopeRule.evaluate` can build `RuleResult.missing_evidence`
    and `RuleResult.recommendations` from the same underlying finding
    without recomputing it twice.

    Attributes:
        description: The plain-language sentence describing this
            specific violation, reported (per claim) in
            `RuleResult.missing_evidence`.
        recommendation: The plain-language suggestion this violation
            implies, reported in `RuleResult.recommendations`.
    """

    description: str
    recommendation: str


# ---------------------------------------------------------------------
# ScopeRule
# ---------------------------------------------------------------------


class ScopeRule(ScientificRule):
    """Determines whether the evidence actually attributed to a claim
    supports that claim's own *declared* scope (claims.py's `Scope`),
    per 02_claims.md Chapter 2, Section 5 and the specification's
    Section 3 ("Scope") and Section 8 ("Scope ambiguity").

    For every `Claim` in `RuleContext.claims` whose `Scope` declares at
    least one dimension (i.e. `Scope.is_unspecified()` is `False`),
    this rule performs two independent, purely structural comparisons
    between that `Scope` and the evidence available in `RuleContext`:

    1. **Breadth.** For the one category/dimension pairing where a
       category's own definition commits it to asserting applicability
       beyond a single value (`_BREADTH_SENSITIVE_DIMENSIONS`;
       currently Generalization claims and the `dataset` dimension), a
       `Scope` that leaves that dimension undeclared is checked
       against how many distinct values the evidence actually spans.
       Evidence confined to at most one value cannot warrant the
       breadth the claim's category, left unnarrowed, asserts.
    2. **Mismatch.** For every named `Scope` dimension that *is*
       declared, and for every entry in `Scope.additional_constraints`,
       this rule checks whether the declared value is actually among
       the values the evidence records for that same dimension or
       constraint key. A declared value with no matching evidence at
       all is left unchecked (a coverage gap, not a scope mismatch --
       `MissingEvidenceRule`'s concern); a declared value that
       *conflicts* with what evidence for that dimension does exist is
       reported as a scope violation.

    A claim whose `Scope.is_unspecified()` is `True` is never checked
    by either comparison and is reported, plainly, as a claim this rule
    could not evaluate -- per the specification's Section 8, treating
    an undeclared scope as though some default applied would itself be
    a violation of the Evidence First principle this rule is built to
    enforce.

    This rule never invents evidence that was not supplied, and never
    infers a mismatch from a claim's free-form `statement` -- see this
    module's docstring for the precise boundary. It never sets
    `RuleResult.confidence_adjustment` away from its `0.0` default and
    never populates `RuleResult.contradictions`: characterizing a
    scope violation as a genuine contradiction, or scoring its
    consequence for confidence, belongs to later stages, per
    `rules.py`'s own stated scope boundary and the specification's own
    insistence (Section 8) that scope resolution precedes, and is
    distinct from, contradiction classification.
    """

    @property
    def id(self) -> str:
        return "R002"

    @property
    def name(self) -> str:
        return "Claim Scope Violation"

    @property
    def description(self) -> str:
        return (
            "Checks each claim's declared Scope against the evidence actually "
            "attributed to it, and reports any dimension where that evidence "
            "is narrower than, or conflicts with, the scope the claim declares."
        )

    @property
    def version(self) -> str:
        return "1.0.0"

    def applies(self, context: RuleContext) -> bool:
        """Relevant whenever `context` has at least one claim to check.

        With no claims present, there is no declared `Scope` for this
        rule to compare evidence against, so it declines to run rather
        than reasoning about scope in the abstract -- the same
        precondition `MissingEvidenceRule.applies` uses.
        """
        return context.claims is not None and len(context.claims) > 0

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Check every claim in `context.claims` with a declared `Scope`
        against the evidence attributed to it, and report any
        dimension where that evidence does not support the declared
        scope.

        Assumes `self.applies(context)` has already returned `True`
        (i.e. `context.claims` is non-empty), per `ScientificRule`'s
        two-phase evaluation contract.

        **Why `OutputCategory.MISSING_EVIDENCE`.** A scope violation,
        as this rule defines it, is precisely a case where the
        evidence a claim's own declared scope requires -- either
        evidence spanning more than the single value `Scope` leaves
        undeclared, or evidence actually gathered under the specific
        value `Scope` does declare -- is not present among the
        evidence attributed to the claim. That is the same kind of
        finding `MissingEvidenceRule` reports under this category, only
        keyed to the claim's own `Scope` rather than to its category's
        fixed expectations. The more consequential categories were
        deliberately not used: `CONTRADICTION` is reserved for a
        relationship between two *already scope-resolved* claims or
        evidence items (specification, Section 8, "Scope ambiguity" --
        resolving scope is a precondition to, and therefore distinct
        from, classifying a contradiction); `JUDGMENT` and
        `RECOMMENDATION` both require a basis this single, standalone
        structural check does not by itself establish (specification,
        Section 4 and Section 7's "Abstain When Necessary").
        """
        assert context.claims is not None  # guaranteed by applies()

        missing_evidence: list[str] = []
        recommendations: list[str] = []
        evidence_used: list[EvidenceItem] = []
        seen_evidence_ids: set[int] = set()
        affected_claims: list[Claim] = []
        reasoning_lines: list[str] = []

        # Performance note (audit #8): these three caches are built
        # once for this whole `evaluate()` call and threaded through
        # `_breadth_findings` / `_mismatch_findings` / `_record_evidence_used`
        # below, rather than each claim recomputing the same
        # context-global (never claim-specific) evidence-value and
        # evidence-item lookups from scratch. See
        # `_cached_evidence_values_for_dimension`,
        # `_cached_distinct_keyed_values`, and `_record_evidence_used`'s
        # own docstrings for the full rationale.
        recorded_kinds: set[int] = set()
        dimension_values_cache: dict[str, set[str]] = {}
        keyed_values_cache: dict[str, set[str]] = {}

        for claim in context.claims:
            affected_claims.append(claim)
            scope = claim.scope

            if scope.is_unspecified():
                reasoning_lines.append(
                    f"Claim {claim.id} ({claim.subject!r}) declares no scope at all; "
                    "scope ambiguity (specification, Section 8) precludes checking "
                    "it for a scope violation."
                )
                continue
            findings = list(
                _breadth_findings(
                    context,
                    claim,
                    scope,
                    evidence_used,
                    seen_evidence_ids,
                    recorded_kinds,
                    dimension_values_cache,
                )
            )
            findings.extend(
                _mismatch_findings(
                    context,
                    claim,
                    scope,
                    evidence_used,
                    seen_evidence_ids,
                    recorded_kinds,
                    dimension_values_cache,
                    keyed_values_cache,
                )
            )

            if findings:
                for finding in findings:
                    missing_evidence.append(
                        f"Claim {claim.id} ({claim.subject!r}): {finding.description}"
                    )
                    recommendations.append(f"Claim {claim.id}: {finding.recommendation}")
                reasoning_lines.append(
                    f"Claim {claim.id} ({claim.category.value}) exceeds the scope its "
                    "evidence actually supports: "
                    + "; ".join(finding.description for finding in findings)
                    + "."
                )
            else:
                reasoning_lines.append(
                    f"Claim {claim.id} ({claim.category.value}) is fully supported at "
                    "its declared scope."
                )

        triggered = len(missing_evidence) > 0
        reasoning = (
            " ".join(reasoning_lines)
            if reasoning_lines
            else ("No claims were available to check for scope violations.")
        )

        return RuleResult(
            rule_id=self.id,
            rule_name=self.name,
            triggered=triggered,
            output_category=OutputCategory.MISSING_EVIDENCE,
            reasoning=reasoning,
            evidence_used=tuple(evidence_used),
            affected_claims=tuple(affected_claims),
            missing_evidence=tuple(missing_evidence),
            recommendations=tuple(recommendations),
        )


# ---------------------------------------------------------------------
# Per-claim finding logic
#
# Kept as module-level functions, rather than methods, mirroring
# `missing_evidence.py`'s preference for small, independently testable
# functions over a monolithic `evaluate` body. Each function takes the
# `evidence_used` accumulator and its `seen_evidence_ids` deduplication
# set as explicit arguments and mutates them in place -- the same
# pattern `MissingEvidenceRule.evaluate` uses inline, factored out here
# purely to keep `evaluate` itself short and legible.
# ---------------------------------------------------------------------


def _record_evidence_used(
    context: RuleContext,
    kinds: tuple[EvidenceKind, ...],
    evidence_used: list[EvidenceItem],
    seen_evidence_ids: set[int],
    recorded_kinds: set[int],
) -> None:
    """Append every not-yet-seen `EvidenceItem` of kind in `kinds` to
    `evidence_used`, deduplicated by object identity via
    `seen_evidence_ids` -- the same bookkeeping
    `MissingEvidenceRule.evaluate` performs inline for its own
    `expectation.kinds`.

    Performance note (audit #8): `kinds` is one of a small, fixed set
    of `_DIMENSION_KINDS` tuples (the same tuple object every time,
    for a given dimension), and once every item of a given `kinds` has
    been folded into `evidence_used` for one `ScopeRule.evaluate` call,
    every later claim sharing that dimension would otherwise re-walk
    the same (now O(1)-indexed, but still non-zero) item list only to
    find every item already in `seen_evidence_ids`. `recorded_kinds`,
    keyed by `id(kinds)` (stable, since `_DIMENSION_KINDS` hands out
    the same tuple instance every time), records which `kinds` this
    call has already fully processed and skips the repeat walk
    entirely -- with no change in `evidence_used`'s resulting content
    or order, since a repeat walk was always a no-op past the first.
    """
    key = id(kinds)
    if key in recorded_kinds:
        return
    recorded_kinds.add(key)
    for item in _items_of_kinds(context, kinds):
        if id(item) not in seen_evidence_ids:
            seen_evidence_ids.add(id(item))
            evidence_used.append(item)


def _breadth_findings(
    context: RuleContext,
    claim: Claim,
    scope: Scope,
    evidence_used: list[EvidenceItem],
    seen_evidence_ids: set[int],
    recorded_kinds: set[int],
    dimension_values_cache: dict[str, set[str]],
) -> list[_ScopeFinding]:
    """Every breadth-based scope violation for `claim`: a dimension
    `_BREADTH_SENSITIVE_DIMENSIONS` ties to `claim.category`, left
    undeclared in `scope`, for which the evidence spans at most one
    distinct value.

    A dimension explicitly narrowed in `scope` (a non-`None` value) is
    never flagged here, however few values the evidence spans --
    `scope` has already committed to a single value, so the claim
    is not asserting breadth on that dimension in the first place; see
    `_mismatch_findings` for the separate check that applies when a
    dimension *is* declared.
    """
    findings: list[_ScopeFinding] = []
    for dimension in _BREADTH_SENSITIVE_DIMENSIONS.get(claim.category, ()):
        if getattr(scope, dimension) is not None:
            continue

        _record_evidence_used(
            context,
            _DIMENSION_KINDS[dimension],
            evidence_used,
            seen_evidence_ids,
            recorded_kinds,
        )

        evidence_values = _cached_evidence_values_for_dimension(
            context, dimension, dimension_values_cache
        )
        if len(evidence_values) >= 2:
            continue

        findings.append(
            _ScopeFinding(
                description=(
                    f"declares no {dimension} restriction, which -- for a "
                    f"{claim.category.value} claim -- asserts applicability beyond a "
                    f"single {dimension}, but the evidence covers only "
                    f"{len(evidence_values)} distinct {dimension} value(s)"
                ),
                recommendation=(
                    f"Narrow this claim's declared scope to the specific "
                    f"{dimension}(s) the evidence actually covers, or gather "
                    f"evidence across additional distinct {dimension} values before "
                    "asserting this breadth."
                ),
            )
        )
    return findings


def _mismatch_findings(
    context: RuleContext,
    claim: Claim,
    scope: Scope,
    evidence_used: list[EvidenceItem],
    seen_evidence_ids: set[int],
    recorded_kinds: set[int],
    dimension_values_cache: dict[str, set[str]],
    keyed_values_cache: dict[str, set[str]],
) -> list[_ScopeFinding]:
    """Every mismatch-based scope violation for `claim`: a named `Scope`
    dimension, or a `Scope.additional_constraints` entry, that *is*
    declared but whose declared value is not among the values the
    evidence records for that same dimension or constraint key.

    A dimension or constraint key with no recorded evidence value at
    all is silently skipped -- that absence is a coverage gap
    (`MissingEvidenceRule`'s concern), not a value conflict, and this
    rule reports only genuine conflicts between a declared value and
    an actually-recorded, different one.
    """
    findings: list[_ScopeFinding] = []

    for dimension in _NAMED_SCOPE_DIMENSIONS:
        declared_value = getattr(scope, dimension)
        if declared_value is None:
            continue

        _record_evidence_used(
            context,
            _DIMENSION_KINDS[dimension],
            evidence_used,
            seen_evidence_ids,
            recorded_kinds,
        )
        evidence_values = _cached_evidence_values_for_dimension(
            context, dimension, dimension_values_cache
        )
        if not evidence_values:
            continue
        if _normalize_value(declared_value) in {_normalize_value(v) for v in evidence_values}:
            continue

        findings.append(
            _ScopeFinding(
                description=(
                    f"declares a {dimension} of {declared_value!r}, but the evidence "
                    f"attributed to it records a different {dimension} "
                    f"({sorted(evidence_values)!r})"
                ),
                recommendation=(
                    f"Verify that this claim's declared {dimension} ({declared_value!r}) "
                    "matches where its supporting evidence was actually gathered, or "
                    "correct the declared scope to match the evidence."
                ),
            )
        )

    for key, declared_value in scope.additional_constraints.items():
        _record_evidence_used(
            context,
            _DIMENSION_KINDS[dimension],
            evidence_used,
            seen_evidence_ids,
            recorded_kinds,
        )

        recorded_values = _cached_distinct_keyed_values(context, key, keyed_values_cache)
        if not recorded_values:
            continue
        if _normalize_value(declared_value) in {_normalize_value(v) for v in recorded_values}:
            continue

        findings.append(
            _ScopeFinding(
                description=(
                    f"declares a scope constraint {key!r}={declared_value!r}, but the "
                    f"evidence attributed to it records a different value for that "
                    f"condition ({sorted(recorded_values)!r})"
                ),
                recommendation=(
                    f"Verify that this claim's declared {key!r} constraint "
                    f"({declared_value!r}) matches the condition its supporting "
                    "evidence was actually gathered under, or correct the declared "
                    "scope to match the evidence."
                ),
            )
        )

    return findings


__all__ = ["ScopeRule"]