"""
Experiment Audit Scientific Reasoning Engine

Module: claims

Defines the Claims stage of the reasoning pipeline, per Chapter 2 of the
Experiment Audit methodology ("a claim is not a hypothesis"). A
scientific claim is an explicit, assertive, subject-scoped statement
about a method, model, or system -- the unit that evidence (Chapter 1),
observations, hypotheses, scientific rules, confidence assessment, and
judgment ultimately exist to support or refute. This module gives that
unit a fixed, unambiguous representation: `Claim`, the collection type
`ClaimSet` that holds many of them, and `Scope`, the specific conditions
under which a claim's evidence was gathered and under which the claim is
therefore actually warranted (Chapter 2, Section 5).

**Scope, strictly bounded -- this is a data model, not a reasoning
engine.** This module declares the domain objects Chapter 2 defines and
nothing else. It does not decide whether a claim is warranted, does not
check a claim's stated scope against the evidence attributed to it, does
not compute where a claim sits on the strength spectrum (Section 6),
does not detect contradictions between claims (Section 7), and does not
enforce that a claim's lifecycle stage (Section 3) has actually been
earned. Those responsibilities belong to Claim Verification, Confidence
Assessment, the Contradiction Engine, and Judgment Generation
respectively (Section 10) -- concrete components, living in other
modules, that consume the types defined here. A reviewer should be able
to read this file end to end and learn exactly how a claim, a claim
set, and a scope are represented -- and learn nothing about how any of
them is evaluated.

**Categories, lifecycle, strength, and unsupported reasons are closed
vocabularies, not algorithms.** `ClaimCategory`, `ClaimLifecycleStage`,
`ClaimStrength`, `ClaimRelationshipType`, and `UnsupportedReason` give
Chapter 2's Sections 3, 4, 6, 7, and 8 a fixed set of named values to
place a claim into. This module does not validate that a claim's chosen
category actually matches its statement's content, does not enforce that
a claim's `lifecycle_stage` has progressed through prior stages in
order, and does not compute a claim's `strength` from its evidence --
each of those is exactly the kind of scientific judgment Section 10
assigns elsewhere. This module only guarantees that, whatever a caller
decides, it is drawn from a well-defined, closed set and is always
present and inspectable.

**Architectural constraint, mirrored from every other module in this
package.** This module has no dependency on FastMCP, MCP transport,
`server.py`, or any backend implementation, and no dependency outside
the Python standard library other than `experiment_audit_mcp.models`'s
`RunRef` -- reused as-is, per this task's explicit instruction to reuse
rather than re-declare existing structures, exactly as `evidence.py` and
`hypotheses.py` already do for the same type.

**Determinism and immutability, mandatory.** Every type this module
defines is a frozen dataclass: a `Claim`, once constructed, is never
mutated in place -- a later re-evaluation or re-scoping produces a new
`Claim`, never an edited one. `ClaimSet` preserves the order its claims
were given in exactly, mirroring `rules.py`'s `RuleRegistry` preserving
registration order; iterating the same `ClaimSet` twice always yields
the same sequence.

**Traceability, mandatory.** Per Chapter 2, Section 9, a claim must be
traceable back through the full reasoning chain that led to it. This
module's contribution to that chain is `Claim.evidence_trace`: the
`RunRef`s identifying the evidence a claim's support rests on, mirroring
the same field and the same traceability discipline `hypotheses.py`
already establishes for `Hypothesis.evidence_trace`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from experiment_audit_mcp.models import RunRef


def _runref_to_dict(ref: RunRef) -> dict[str, str]:
    """Local copy of `models.py`'s private `_runref_to_dict`.

    `models.py` deliberately doesn't export its underscore-prefixed
    helper for cross-module reuse; `evidence.py` follows the same
    convention with its own local copy, and this module does the same
    rather than reaching into another module's private name. Without
    this, `Claim.to_dict()`'s `evidence_trace` field would contain raw
    `RunRef` instances instead of the JSON-safe structure every other
    `to_dict()` in this package produces -- `RunRef` itself has no
    `to_dict()` method (see `models.py`).
    """
    return {
        "backend": ref.backend,
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
    }


# ---------------------------------------------------------------------
# ClaimCategory
# ---------------------------------------------------------------------


class ClaimCategory(StrEnum):
    """Chapter 2, Section 4's closed taxonomy of what kind of assertion
    a claim makes.

    As with `evidence.py`'s evidence categories, these describe what
    *kind* of claim is being made, not how strong or well-supported it
    is -- strength is `ClaimStrength`'s job, below. A single piece of
    research frequently advances claims from several categories at
    once, and this module does not force a `Claim` into exactly one:
    a caller who genuinely needs to represent a claim spanning several
    categories may do so via `Claim.metadata`, or by formulating
    separate, related `Claim`s and recording the relationship between
    them with `ClaimRelationshipType`.

    Values are lowercase strings, matching `rules.py`'s
    `OutputCategory` convention, so a serialized `Claim.to_dict()`
    remains self-describing without a separate lookup table.
    """

    PERFORMANCE = "performance"
    COMPARISON = "comparison"
    GENERALIZATION = "generalization"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"
    STATISTICAL = "statistical"
    REPRODUCIBILITY = "reproducibility"


# ---------------------------------------------------------------------
# ClaimLifecycleStage
# ---------------------------------------------------------------------


class ClaimLifecycleStage(StrEnum):
    """Chapter 2, Section 3's ordered sequence of stages a claim passes
    through.

    Members are declared in the methodology's own order -- formulation
    through re-evaluation -- so that order is a structural fact of this
    enum's declaration, available to any caller that wants to reason
    about "earlier" or "later" stages (e.g. via `list(ClaimLifecycleStage)`
    and `.index(...)`). This module does not itself enforce that a
    `Claim`'s stage has been earned by actually completing the prior
    stages, nor does it detect a claim that has skipped a stage --
    Section 3 identifies that as a defect in a claim's standing, but
    detecting it is a verification concern, not a data-model concern.
    """

    FORMULATED = "formulated"
    SUPPORTED = "supported"
    EVALUATED = "evaluated"
    VERIFIED = "verified"
    PUBLISHED = "published"
    RE_EVALUATED = "re_evaluated"


# ---------------------------------------------------------------------
# ClaimStrength
# ---------------------------------------------------------------------


class ClaimStrength(StrEnum):
    """Chapter 2, Section 6's spectrum of claim strength.

    Members are ordered weakest to strongest, exactly as Section 6
    orders them, mirroring `rules.py`'s `OutputCategory` docstring
    convention of documenting an enum's members as meaningfully
    ordered rather than an arbitrary set. Per Section 6, this
    methodology "does not attach numerical thresholds to these
    positions and does not define an algorithm for computing which
    position a given claim occupies" -- this module fixes only the
    named positions themselves; computing which one applies to a given
    `Claim` is Confidence Assessment's job (Section 10), not this
    module's.
    """

    OBSERVATION = "observation"
    SUGGESTION = "suggestion"
    SUPPORTED_CLAIM = "supported_claim"
    STRONGLY_SUPPORTED_CLAIM = "strongly_supported_claim"
    ESTABLISHED_CLAIM = "established_claim"


# ---------------------------------------------------------------------
# UnsupportedReason
# ---------------------------------------------------------------------


class UnsupportedReason(StrEnum):
    """Chapter 2, Section 8's closed set of conditions under which a
    formulated claim lacks adequate support.

    A `Claim` records zero or more of these via
    `Claim.unsupported_reasons` (see below); recording one is not, by
    itself, grounds for discarding the claim -- Section 8 is explicit
    that an unsupported claim can still be a legitimate suggestion or a
    motivation for further work. What Section 8 forbids is *silent*
    acceptance: this field exists so that condition is always
    structurally recorded rather than left to convention, matching this
    package's general traceability discipline.
    """

    NO_EVIDENCE = "no_evidence"
    WEAK_EVIDENCE = "weak_evidence"
    MISSING_EVIDENCE = "missing_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ---------------------------------------------------------------------
# ClaimRelationshipType
# ---------------------------------------------------------------------


class ClaimRelationshipType(StrEnum):
    """Chapter 2, Section 7's closed set of relationships that can hold
    between two claims.

    Provided here as shared vocabulary for any component that records a
    relationship between two `Claim`s (e.g. a future
    `contradictions.py`'s handling of `CONTRADICTORY`, or a confidence
    component consulting `DEPENDENT` / `DERIVED` to avoid overweighting
    claims that are not genuinely independent support). This module
    defines only the vocabulary; *detecting* which relationship holds
    between two given claims is, per Section 7 and Section 10, that
    other component's job, not this one's.
    """

    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    SUPPORTING = "supporting"
    CONTRADICTORY = "contradictory"
    DERIVED = "derived"
    EQUIVALENT = "equivalent"


# ---------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scope:
    """The specific conditions under which a claim's supporting evidence
    was actually gathered (Chapter 2, Section 5).

    Per Section 5, a claim's scope is "not an optional qualifier that
    can be stripped away for convenience -- it is part of what the
    claim asserts." `Scope` gives that binding condition a concrete,
    inspectable shape rather than leaving it as free text embedded in a
    claim's statement, so a later component (Claim Verification, the
    Contradiction Engine) can compare two scopes structurally instead
    of re-parsing prose.

    Every field is optional: a caller may declare only the dimensions
    that are actually relevant to a given claim (Section 5's list is
    itself "among other factors," not exhaustive), and `None` means
    "not declared for this claim" -- distinct from an empty string,
    which would assert that the dimension was considered and found to
    have no value. `is_unspecified` reports whether *no* dimension has
    been declared at all, the condition `rules.py`'s
    `RuleContext.scope` docstring calls "scope ambiguity" when a rule
    depends on scope to proceed.

    Attributes:
        dataset: The specific dataset or task the supporting evidence
            was collected on.
        model: The specific model or architecture evaluated.
        hardware: The specific hardware the experiment was executed on,
            where relevant to the claim.
        evaluation_protocol: The specific evaluation protocol used to
            produce the supporting evidence.
        software_environment: The specific software and data
            environment in effect at the time the evidence was
            gathered.
        additional_constraints: Further scope dimensions not covered by
            the named fields above, as plain `name -> description`
            pairs (e.g. `{"random_seed_policy": "fixed at 0"}`).
            Defaults to an empty mapping, not `None`, so a caller can
            always safely call `.get(...)` on it without a `None`
            check.
    """

    dataset: str | None = None
    model: str | None = None
    hardware: str | None = None
    evaluation_protocol: str | None = None
    software_environment: str | None = None
    additional_constraints: Mapping[str, str] = field(default_factory=dict)

    def is_unspecified(self) -> bool:
        """Whether no scope dimension at all has been declared.

        `True` only when every named field is `None` and
        `additional_constraints` is empty -- the "scope has not been
        declared" case `rules.py`'s `RuleContext.scope` docstring
        distinguishes from an actually-declared (even if narrow) scope.
        """
        return (
            self.dataset is None
            and self.model is None
            and self.hardware is None
            and self.evaluation_protocol is None
            and self.software_environment is None
            and not self.additional_constraints
        )

    def to_dict(self) -> dict[str, Any]:
        """Best-effort JSON-safe serialization, matching the `to_dict()`
        convention every concrete type in this package follows.
        """
        return {
            "dataset": self.dataset,
            "model": self.model,
            "hardware": self.hardware,
            "evaluation_protocol": self.evaluation_protocol,
            "software_environment": self.software_environment,
            "additional_constraints": dict(self.additional_constraints),
        }


# ---------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Claim:
    """A single scientific claim (Chapter 2, Section 2): an explicit,
    assertive, subject-scoped statement about a method, model, or
    system.

    Per Section 2, a `Claim` is distinct from three concepts already
    established elsewhere in this package, and this module deliberately
    does not blur any of those distinctions:

    - **Not evidence.** A `Claim` never carries raw evidence content
      itself; it carries only `evidence_trace`, the `RunRef`s
      identifying the evidence its support rests on, exactly as
      `hypotheses.py`'s `Hypothesis.evidence_trace` does for a
      hypothesis. The evidence itself lives in `evidence.py`'s
      `Evidence` / `EvidenceItem`.
    - **Not an observation.** An observation is scoped tightly to a
      single piece of evidence; a `Claim` is broader, asserting
      something about the subject in general, within `scope`, not
      merely true of one recorded run.
    - **Not a hypothesis.** A hypothesis is offered as provisional from
      the outset; a `Claim` is put forward as an assertion its author
      is prepared to defend with evidence. This module never accepts a
      `Hypothesis` in place of a `Claim` -- there is no constructor or
      conversion here that blurs the two, matching `rules.py`'s
      `RuleResult.affected_claims` / `affected_hypotheses` keeping the
      two fields, and thus the two concepts, structurally distinct.

    Frozen, matching `Observation` / `Hypothesis` / `Judgment`'s
    convention elsewhere in this package: a claim, once constructed, is
    not mutated in place. Re-scoping a claim, advancing it to a later
    lifecycle stage, or revising its statement each produce a new
    `Claim`, never an edited one -- consistent with Section 3's
    "Re-evaluated" stage describing a *return* to the lifecycle, not an
    in-place edit of the original claim.

    Attributes:
        id: A short, stable, unique identifier for this claim (e.g.
            `"C001"`), analogous to `rules.py`'s `ScientificRule.id`.
            Used to reference this claim from elsewhere (e.g. a future
            `ClaimRelationshipRecord` or `Contradiction`) without
            copying its full content. Must be non-empty.
        subject: The specific method, model, or system this claim is
            about (Section 2's "scoped to a subject" requirement --
            never a free-floating generalization about the field).
            Must be non-empty.
        statement: The claim's own explicit, assertive text -- what is
            actually being asserted (e.g. "Model X attains at least 90%
            accuracy on Benchmark Y"). Must be non-empty; per Section
            2, a claim is stated, not implied, so an empty statement
            cannot itself constitute a claim.
        category: Which of Section 4's categories this claim belongs
            to. See `ClaimCategory`.
        scope: The specific conditions under which this claim's
            supporting evidence was gathered, and therefore the
            conditions under which the claim is actually warranted
            (Section 5). See `Scope`.
        lifecycle_stage: Where this claim currently sits in Section 3's
            sequence. Defaults to `ClaimLifecycleStage.FORMULATED`,
            the stage every claim starts at once it has been stated
            explicitly enough to evaluate.
        strength: Where this claim currently sits on Section 6's
            strength spectrum, if a confidence assessment has already
            placed it there. `None` when no such assessment has been
            made yet -- computing this value is Confidence Assessment's
            job (Section 10), never this module's, so `Claim` only
            stores the result once available.
        evidence_trace: Every `RunRef` identifying evidence this
            claim's support rests on, in caller-supplied order --
            the traceability link Section 9 requires, mirroring
            `Hypothesis.evidence_trace`'s role for hypotheses. Defaults
            to an empty tuple, not `None`, matching this package's
            "absence is itself a meaningful, iterable-safe value"
            convention (e.g. `rules.py`'s `RuleContext.detected_contradictions`).
        unsupported_reasons: Every `UnsupportedReason` (Section 8)
            currently recognized as applying to this claim, in
            caller-supplied order. Empty does not itself assert that
            the claim is fully warranted -- only that no unsupported
            condition has been recorded for it. Determining which
            reasons apply is Claim Verification's job (Section 10);
            this field only gives that determination somewhere to be
            recorded once made.
        metadata: Free-form, caller-supplied context that does not fit
            any of the above (e.g. `{"author": "team-a"}`). Never
            interpreted by this module. Defaults to an empty mapping
            rather than `None` so a caller can always safely call
            `.get(...)` on it without a `None` check.
    """

    id: str
    subject: str
    statement: str
    category: ClaimCategory
    scope: Scope
    lifecycle_stage: ClaimLifecycleStage = ClaimLifecycleStage.FORMULATED
    strength: ClaimStrength | None = None
    evidence_trace: tuple[RunRef, ...] = ()
    unsupported_reasons: tuple[UnsupportedReason, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Claim.id must be a non-empty string.")
        if not self.subject:
            raise ValueError("Claim.subject must be a non-empty string.")
        if not self.statement:
            raise ValueError(
                "Claim.statement must be a non-empty string -- a claim is stated, not implied."
            )
        if not isinstance(self.category, ClaimCategory):
            raise ValueError(
                f"Claim.category must be a ClaimCategory member, got {self.category!r}."
            )
        if not isinstance(self.scope, Scope):
            raise ValueError(f"Claim.scope must be a Scope instance, got {self.scope!r}.")
        if not isinstance(self.lifecycle_stage, ClaimLifecycleStage):
            raise ValueError(
                "Claim.lifecycle_stage must be a ClaimLifecycleStage member, "
                f"got {self.lifecycle_stage!r}."
            )
        if self.strength is not None and not isinstance(self.strength, ClaimStrength):
            raise ValueError(
                f"Claim.strength must be a ClaimStrength member or None, got {self.strength!r}."
            )
        for reason in self.unsupported_reasons:
            if not isinstance(reason, UnsupportedReason):
                raise ValueError(
                    "Claim.unsupported_reasons must contain only UnsupportedReason "
                    f"members, got {reason!r}."
                )

    def is_unsupported(self) -> bool:
        """Whether at least one `UnsupportedReason` has been recorded
        for this claim.

        A convenience predicate over `unsupported_reasons`, not a
        determination this module makes itself -- it reports only what
        has already been recorded, per Section 8's requirement that an
        unsupported claim never appear "indistinguishable from a claim
        that has actually earned a stronger position."
        """
        return len(self.unsupported_reasons) > 0

    def to_dict(self) -> dict[str, Any]:
        """Best-effort JSON-safe serialization, matching the `to_dict()`
        convention every concrete type in this package follows.
        """
        return {
            "id": self.id,
            "subject": self.subject,
            "statement": self.statement,
            "category": self.category.value,
            "scope": self.scope.to_dict(),
            "lifecycle_stage": self.lifecycle_stage.value,
            "strength": self.strength.value if self.strength is not None else None,
            "evidence_trace": [_runref_to_dict(ref) for ref in self.evidence_trace],
            "unsupported_reasons": [reason.value for reason in self.unsupported_reasons],
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


class ClaimSetError(ValueError):
    """Raised by `ClaimSet` for a construction-time misuse -- currently,
    only two or more `Claim`s sharing an `id`.

    Kept as a narrow, local exception, matching this package's
    convention of one small, purpose-specific error type per module
    (e.g. `rules.py`'s `RuleRegistrationError`) rather than raising a
    bare `ValueError` indistinguishable from any other.
    """


# ---------------------------------------------------------------------
# ClaimSet
# ---------------------------------------------------------------------


class ClaimSet:
    """An ordered, deduplicated-by-id collection of `Claim`s.

    `ClaimSet` mirrors `hypotheses.py`'s `HypothesisSet`: it owns no
    reasoning of its own and exists only to hold many `Claim`s together
    with the lookup helpers (`by_subject`, `by_category`,
    `by_lifecycle_stage`, `by_strength`, `unsupported`) a caller
    otherwise has to reimplement ad hoc every time a `RuleContext.claims`
    or similar collection needs to be queried.

    **Determinism.** Claims are kept in an insertion-ordered mapping
    keyed by `id` (a plain `dict`, whose insertion-order guarantee is
    part of the language since Python 3.7). Iteration, `to_dict()`, and
    every lookup helper below always visit claims in that same order.
    """

    __slots__ = ("_claims",)

    def __init__(self, claims: Iterable[Claim] | None = None) -> None:
        """Create a claim set, optionally pre-populated with `claims`.

        Args:
            claims: Claims to include, in the order given. Defaults to
                an empty set.

        Raises:
            ClaimSetError: If two or more given claims share an `id`.
        """
        self._claims: dict[str, Claim] = {}
        if claims is not None:
            for claim in claims:
                if claim.id in self._claims:
                    raise ClaimSetError(
                        f"Cannot add claim {claim.id!r}: a claim with that id is "
                        "already present in this ClaimSet."
                    )
                self._claims[claim.id] = claim

    def by_id(self, claim_id: str) -> Claim | None:
        """The claim registered under `claim_id`, or `None` if none is."""
        return self._claims.get(claim_id)

    def by_subject(self, subject: str) -> tuple[Claim, ...]:
        """Every claim whose `subject` equals `subject`, in this set's
        order.
        """
        return tuple(claim for claim in self._claims.values() if claim.subject == subject)

    def by_category(self, category: ClaimCategory) -> tuple[Claim, ...]:
        """Every claim whose `category` equals `category`, in this
        set's order.
        """
        return tuple(claim for claim in self._claims.values() if claim.category == category)

    def by_lifecycle_stage(self, stage: ClaimLifecycleStage) -> tuple[Claim, ...]:
        """Every claim whose `lifecycle_stage` equals `stage`, in this
        set's order.
        """
        return tuple(claim for claim in self._claims.values() if claim.lifecycle_stage == stage)

    def by_strength(self, strength: ClaimStrength) -> tuple[Claim, ...]:
        """Every claim whose `strength` equals `strength`, in this
        set's order. A claim whose `strength` is `None` (not yet
        assessed) never matches any `strength` argument here.
        """
        return tuple(claim for claim in self._claims.values() if claim.strength == strength)

    def unsupported(self) -> tuple[Claim, ...]:
        """Every claim with at least one recorded `UnsupportedReason`
        (Section 8), in this set's order.
        """
        return tuple(claim for claim in self._claims.values() if claim.is_unsupported())

    def to_dict(self) -> dict[str, Any]:
        """Best-effort JSON-safe serialization, matching the `to_dict()`
        convention every concrete type in this package follows.
        """
        return {"claims": [claim.to_dict() for claim in self._claims.values()]}

    def __len__(self) -> int:
        return len(self._claims)

    def __iter__(self) -> Iterator[Claim]:
        return iter(self._claims.values())

    def __contains__(self, claim_id: object) -> bool:
        return claim_id in self._claims

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self._claims)} claim(s): {list(self._claims)!r})"