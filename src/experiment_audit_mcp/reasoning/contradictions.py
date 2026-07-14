"""
Experiment Audit Scientific Reasoning Engine

Module: contradictions

Defines the foundational domain model for **scientific contradictions**,
per research/08_contradictions.md ("Chapter 8"). A contradiction, under
that chapter's Section 2 definition, is a documented relationship
between two or more evidence items, or between two or more claims, in
which one cannot be true, valid, or fully supported without the other
being false, invalid, or unsupported, given that both are understood
under the same scope, conditions, and terms.

**Scope, strictly bounded -- this is a data model, not a reasoning
engine.** This module defines the shapes a contradiction, once
detected, is recorded in: the categories a contradiction may belong to
(Chapter 8, Section 3), the sources a contradiction may be traced to
(Section 4), the lifecycle stage it currently occupies (Section 5), how
its quality is assessed once investigated (Section 8), and how its
resolution is recorded when the lifecycle reaches that stage. It
deliberately contains **no contradiction-detection logic of any kind**
-- nothing here inspects two `EvidenceItem`s or two `Claim`s and decides
whether they actually conflict, resolves an apparent conflict into a
scope difference, or assigns a category or source to a specific
detected conflict. That reasoning belongs to a future, concrete
`ScientificRule` subclass living in `rules.py`'s framework (e.g. a
`ContradictionRule`, per that module's own docstring), never in this
file. This module also contains **no rule execution machinery** --
registering, ordering, or running anything is `rules.py`'s
`RuleRegistry` / `RuleEngine` job, not this module's.

**Architectural constraint, mirrored from every other module in this
package.** This module has no dependency on FastMCP, MCP transport,
`server.py`, or any backend implementation, and no dependency outside
the Python standard library plus this package's own normalized types.
It reuses `evidence.py`'s `EvidenceItem` and `models.py`'s `RunRef`
directly, at runtime, rather than re-declaring them, per this
package's convention of building each pipeline stage's types on top of
the ones already established upstream. `claims.py`'s `Claim` is
referenced only under `TYPE_CHECKING`, mirroring the same discipline
`rules.py` itself already applies to this module's own `Contradiction`
type: `claims.py` is not guaranteed to exist yet, so this module's
runtime import graph does not grow ahead of what actually exists. A
`Contradiction` that references a claim does so structurally (a
`tuple["Claim", ...]` field), without this module ever needing to
import, construct, or inspect a `Claim` at runtime.

**Traceability, mandatory (Section 9).** Every `Contradiction` this
module can construct must, structurally, be capable of naming the
evidence items and/or claims it relates (`evidence_items`, `claims`)
and the scientific-reasoning principle it implicates or tests
(`scientific_rule`). This module enforces the structural minimum --
that a contradiction actually names at least two evidence items and/or
claims in conflict, per Section 2's "two or more" -- but does not
require `scientific_rule` to be populated at construction time, since
Section 9 traceability is completed during the investigated/explained
stages of the lifecycle (Section 5), which this module also does not
implement the logic for.

**Non-exclusivity of categories and sources (Sections 3 and 4).** Per
Chapter 8, "[categories] are not mutually exclusive... this methodology
does not require forcing a contradiction into exactly one category; it
requires that every category applicable to a given contradiction be
recorded." `Contradiction.categories` and `Contradiction.sources` are
therefore both non-empty tuples of their respective enums, never a
single value, so a caller can record every category or source that
applies rather than being forced to pick one.

**Contradictions are retained, never discarded (Section 7).** Nothing
in this module provides a way to delete or mutate a `Contradiction` in
place -- every dataclass here is frozen. A contradiction's lifecycle
advances by constructing a new `Contradiction` (or a new
`ContradictionSet`) from the old one plus whatever new information was
learned, never by editing history, mirroring `Observation` /
`Hypothesis` / `Judgment`'s convention elsewhere in this package.

**Determinism.** `ContradictionSet` preserves insertion order exactly
and never reorders, deduplicates by anything other than `id`, or
introduces randomness or wall-clock dependence of its own; the optional
`detected_at` field on `Contradiction` is caller-supplied, not
self-assigned.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from experiment_audit_mcp.models import RunRef
from experiment_audit_mcp.reasoning.evidence import EvidenceItem

if TYPE_CHECKING:
    # Imported only for static typing. `claims.py` is not guaranteed to
    # exist yet (see module docstring); this module never constructs,
    # inspects, or depends on `Claim` at runtime, only references it
    # structurally in a `Contradiction`'s type signature.
    from experiment_audit_mcp.reasoning.claims import Claim


# ---------------------------------------------------------------------
# Errors
#
# Local, narrow exception type, matching this codebase's convention of
# one small hierarchy per module (e.g. `rules.py`'s
# `RuleRegistrationError` / `RuleEvaluationError`) rather than raising
# bare `Exception` or reusing a stdlib exception for a domain-specific
# failure.
# ---------------------------------------------------------------------


class ContradictionError(ValueError):
    """Raised for a structurally invalid contradiction-model value.

    Covers a `Contradiction`, `ContradictionQuality`, or
    `ContradictionResolution` constructed with a shape this module's
    structural invariants forbid (e.g. a `Contradiction` naming fewer
    than two evidence items and/or claims in conflict, or a
    `ContradictionQuality` dimension outside its defined range) --
    never a judgment about whether a *detected* conflict is genuinely
    a contradiction, which this module does not decide.
    """


# ---------------------------------------------------------------------
# ContradictionCategory (Chapter 8, Section 3)
# ---------------------------------------------------------------------


class ContradictionCategory(Enum):
    """The categories of scientific contradiction Chapter 8, Section 3
    distinguishes.

    Categories are "not mutually exclusive" (Section 3): a single
    detected contradiction is frequently found, on investigation, to
    belong to more than one of these at once. `Contradiction.categories`
    is therefore always a tuple, never a bare `ContradictionCategory`,
    so every applicable category can be recorded rather than forcing a
    choice among them.

    Values are lowercase strings, matching `rules.py`'s
    `OutputCategory` convention, so a serialized `Contradiction` remains
    self-describing without a separate lookup table.
    """

    EVIDENCE = "evidence"
    CLAIM = "claim"
    EXPERIMENTAL = "experimental"
    STATISTICAL = "statistical"
    REPRODUCIBILITY = "reproducibility"
    SCOPE = "scope"
    METHODOLOGICAL = "methodological"


# ---------------------------------------------------------------------
# ContradictionSource (Chapter 8, Section 4)
# ---------------------------------------------------------------------


class ContradictionSource(Enum):
    """The recurring sources of contradiction Chapter 8, Section 4
    recognizes.

    Like `ContradictionCategory`, a single contradiction may trace to
    more than one source (e.g. a reproducibility contradiction that is
    also, on inspection, sourced to an evaluation mismatch), so
    `Contradiction.sources` is a tuple. Identifying a candidate source
    is Section 4's "investigated" work; this enum only names the
    possibilities that work can conclude among -- it does not perform
    that identification itself.
    """

    CONFLICTING_EXPERIMENTS = "conflicting_experiments"
    DIFFERENT_DATASETS = "different_datasets"
    DIFFERENT_ENVIRONMENTS = "different_environments"
    DIFFERENT_IMPLEMENTATIONS = "different_implementations"
    POOR_BASELINES = "poor_baselines"
    MISSING_CONTROLS = "missing_controls"
    DATA_LEAKAGE = "data_leakage"
    EVALUATION_MISMATCH = "evaluation_mismatch"


# ---------------------------------------------------------------------
# ContradictionStatus (Chapter 8, Section 5 -- lifecycle)
# ---------------------------------------------------------------------


class ContradictionStatus(Enum):
    """A contradiction's position in the Section 5 lifecycle:

        Detected -> Investigated -> Explained -> Resolved or Persisting

    "A contradiction's position in this sequence determines what may
    legitimately be said about it and what work remains outstanding
    before it can be treated as closed" (Section 5). This module does
    not enforce forward-only progression through these stages -- doing
    so would require this module to judge *when* an explanation is
    sufficient, or *when* investigation has been genuinely attempted,
    which is the investigation logic Section 5 describes and this
    module deliberately does not implement. It only names the stages a
    caller's own investigation, explanation, and resolution logic
    advances a `Contradiction` through.
    """

    DETECTED = "detected"
    INVESTIGATED = "investigated"
    EXPLAINED = "explained"
    RESOLVED = "resolved"
    PERSISTING = "persisting"


# ---------------------------------------------------------------------
# ScopeDetermination (Chapter 8, Sections 2 and 8)
# ---------------------------------------------------------------------


class ScopeDetermination(Enum):
    """Whether a contradiction, once investigated, remains a genuine
    conflict under matched scope or is better characterized as a scope
    difference (Section 2's load-bearing final clause; Section 8's
    "Scope" quality dimension).

    `UNDETERMINED` is the only valid value before scope has actually
    been checked (Section 2: "[establishing genuine conflict vs. scope
    difference] must be performed before any contradiction is recorded
    as such" -- i.e. before a caller is entitled to assert either of
    the other two values). This module does not perform that check; it
    only gives a caller's check somewhere to record its outcome.
    """

    UNDETERMINED = "undetermined"
    GENUINE_CONFLICT = "genuine_conflict"
    SCOPE_DIFFERENCE = "scope_difference"


# ---------------------------------------------------------------------
# ContradictionQuality (Chapter 8, Section 8)
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContradictionQuality:
    """A contradiction's quality, assessed along Section 8's five
    scored dimensions plus its scope determination.

    Section 8 requires that "contradiction quality be assessed along
    each of these dimensions before a contradiction is weighed into any
    downstream confidence assessment or judgment," rather than treating
    a contradiction "as a single flag that is either present or
    absent." This class is that structured assessment's shape; it does
    not compute one -- a caller (eventually, a concrete `ScientificRule`
    such as a future `ContradictionRule`) supplies each score after
    having actually investigated the contradiction.

    Attributes:
        severity: How much the contradiction, left unresolved,
            undermines the claim(s) it touches, in `[0.0, 1.0]`, where
            `0.0` is a disagreement well within ordinary measurement
            noise and `1.0` is a disagreement that directly reverses a
            comparative claim's conclusion.
        breadth: How much of the methodology's reasoning the
            contradiction affects, in `[0.0, 1.0]`, where `0.0` is
            confined to a single, low-stakes claim and `1.0` propagates
            into multiple claims, confidence assessments, or
            recommendations.
        repeatability: How often the contradiction has recurred across
            independent evidence-gathering attempts, in `[0.0, 1.0]`,
            where `0.0` is observed once and `1.0` is consistently
            reproduced across repeated, independent attempts.
        reliability: The quality of the evidence on each side of the
            contradiction (Chapter 1, Section 7), in `[0.0, 1.0]`,
            where `0.0` reflects poorly attributed, low-provenance
            evidence on at least one side and `1.0` reflects
            high-quality, well-attributed evidence on both.
        independence: How independent the conflicting evidence items or
            claims are from one another, in `[0.0, 1.0]`, where `0.0`
            reflects a shared, unacknowledged common origin that could
            itself explain the apparent disagreement and `1.0`
            reflects fully independent sources.
        scope: The `ScopeDetermination` reached for this contradiction.
            Defaults to `UNDETERMINED`, matching that enum's own
            default, since a `ContradictionQuality` may be constructed
            before scope has actually been checked.
    """

    severity: float
    breadth: float
    repeatability: float
    reliability: float
    independence: float
    scope: ScopeDetermination = ScopeDetermination.UNDETERMINED

    def __post_init__(self) -> None:
        for dimension in (
            "severity",
            "breadth",
            "repeatability",
            "reliability",
            "independence",
        ):
            value = getattr(self, dimension)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ContradictionError(
                    f"ContradictionQuality.{dimension} must be a float, "
                    f"got {type(value).__name__}."
                )
            if not 0.0 <= float(value) <= 1.0:
                raise ContradictionError(
                    f"ContradictionQuality.{dimension} must be within [0.0, 1.0], "
                    f"got {value!r}."
                )
        if not isinstance(self.scope, ScopeDetermination):
            raise ContradictionError(
                "ContradictionQuality.scope must be a ScopeDetermination member, "
                f"got {self.scope!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization, matching the `to_dict()` convention
        every concrete type in this package follows.
        """
        return {
            "severity": self.severity,
            "breadth": self.breadth,
            "repeatability": self.repeatability,
            "reliability": self.reliability,
            "independence": self.independence,
            "scope": self.scope.value,
        }


# ---------------------------------------------------------------------
# ContradictionResolution (Chapter 8, Section 5 -- "Resolved")
# ---------------------------------------------------------------------


class ContradictionResolutionKind(Enum):
    """The determinations Section 5's "Resolved" stage recognizes:

    "the conflict was a scope difference now made explicit, one side
    was found to rest on confounded or invalid evidence and is
    discounted accordingly, or both sides are found to hold under
    correctly stated, non-overlapping scopes."
    """

    SCOPE_DIFFERENCE_CLARIFIED = "scope_difference_clarified"
    EVIDENCE_DISCOUNTED = "evidence_discounted"
    NON_OVERLAPPING_SCOPES = "non_overlapping_scopes"


@dataclass(frozen=True, slots=True)
class ContradictionResolution:
    """The recorded outcome of a contradiction reaching Section 5's
    "Resolved" stage.

    Per Section 5, "[a] resolved contradiction is retained in the
    record together with its explanation; resolution updates how the
    contradiction is reported, but it does not delete the contradiction
    from the evidentiary history" -- and per Section 7's "Preserve
    conflicting evidence," any discounted evidence remains part of the
    record too. This class carries exactly that: which determination
    was reached, why, and -- when the determination is
    `EVIDENCE_DISCOUNTED` -- which evidence was discounted, still
    present rather than removed.

    Attributes:
        kind: Which of Section 5's three resolution determinations was
            reached.
        explanation: The evidence-backed account of why this
            determination was reached (Section 5's "Explained" stage,
            carried into the resolution). Never empty: a resolution
            without a stated reason is not yet a resolution under
            Section 5.
        discounted_evidence: The `EvidenceItem`s found to rest on
            confounded or invalid evidence and discounted accordingly.
            Populated only when `kind` is `EVIDENCE_DISCOUNTED`; empty
            for the other two kinds, since neither "scope difference
            clarified" nor "non-overlapping scopes" discounts any
            evidence -- both sides remain valid, just distinguished by
            scope.
    """

    kind: ContradictionResolutionKind
    explanation: str
    discounted_evidence: tuple[EvidenceItem, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ContradictionResolutionKind):
            raise ContradictionError(
                "ContradictionResolution.kind must be a ContradictionResolutionKind "
                f"member, got {self.kind!r}."
            )
        if not self.explanation:
            raise ContradictionError(
                "ContradictionResolution.explanation must be a non-empty string."
            )
        if (
            self.kind is ContradictionResolutionKind.EVIDENCE_DISCOUNTED
            and not self.discounted_evidence
        ):
            raise ContradictionError(
                "ContradictionResolution.discounted_evidence must be non-empty when "
                "kind is EVIDENCE_DISCOUNTED."
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization, matching the `to_dict()` convention
        every concrete type in this package follows.
        """
        return {
            "kind": self.kind.value,
            "explanation": self.explanation,
            "discounted_evidence": [item.to_dict() for item in self.discounted_evidence],
        }


# ---------------------------------------------------------------------
# Contradiction (Chapter 8, Sections 2, 6, and 9)
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Contradiction:
    """A documented scientific contradiction, per Chapter 8, Section 2:
    a relationship between two or more evidence items, or between two
    or more claims, in which one cannot be true, valid, or fully
    supported without the other being false, invalid, or unsupported,
    given that both are understood under the same scope, conditions,
    and terms.

    Frozen, matching `Evidence` / `Observation` / `Hypothesis` /
    `Judgment`'s convention elsewhere in this package: a contradiction,
    once recorded, is not mutated in place. Advancing it through
    Section 5's lifecycle means constructing a new `Contradiction`
    (typically by replacing `status` and whichever of `sources`,
    `explanation`, `quality`, or `resolution` the new stage supplies),
    never editing the old one -- consistent with Section 7's "Preserve
    conflicting evidence" and Section 5's "[a resolved contradiction]
    does not delete the contradiction from the evidentiary history."

    Attributes:
        id: A stable, caller-assigned identifier for this contradiction,
            unique within whatever `ContradictionSet` or broader record
            it is part of. Never empty.
        categories: Every `ContradictionCategory` (Section 3) that
            applies to this contradiction. Non-empty, and may contain
            more than one member, per Section 3's explicit
            non-exclusivity.
        status: This contradiction's current position in the Section 5
            lifecycle. Defaults to `DETECTED`, the lifecycle's starting
            stage -- a `Contradiction` is not constructible in some
            state prior to having been detected.
        evidence_items: The `EvidenceItem`s (evidence.py) in conflict,
            if this contradiction is (wholly or partly) an evidence-level
            conflict (Section 3's "Evidence Contradictions"; Section 6's
            "[a] contradiction is, at minimum, a relationship between
            evidence items"). May be empty only if `claims` alone
            already supplies at least two parties in conflict (a pure
            claim-claim contradiction, Section 3's "Claim
            Contradictions").
        claims: The `Claim`s (claims.py) in conflict, or the claim(s)
            an evidence-vs-claim contradiction bears on (Section 2's
            "When evidence contradicts a claim"; Section 3's "Claim
            Contradictions"). May be empty only if `evidence_items`
            alone already supplies at least two parties in conflict (a
            pure evidence-evidence contradiction). Referenced under
            `TYPE_CHECKING` only -- see module docstring.
        sources: Every `ContradictionSource` (Section 4) identified as
            a candidate explanation, in caller-supplied order. Empty
            until Section 5's "Investigated" stage has actually
            identified candidates; a `Contradiction` still at `DETECTED`
            typically has no sources yet.
        scientific_rule: The specific principle of correct scientific
            reasoning this contradiction implicates or tests (Section
            9's third traceability requirement, e.g. "fairness of
            comparison," "statistical rigor," "reproducibility
            requirements"). `None` until that principle has been
            identified -- Section 9 traceability is completed over the
            course of the lifecycle, not required at the `DETECTED`
            stage this module's default constructs.
        explanation: The evidence-backed account of why this
            contradiction exists (Section 5's "Explained" stage):
            which source(s) from `sources` produced it, and through
            what mechanism. `None` before that stage is reached.
        resolution: The `ContradictionResolution` reached, if `status`
            is `RESOLVED`. Must be `None` for every other `status`,
            and non-`None` when `status` is `RESOLVED` -- resolution and
            status are kept consistent structurally rather than left to
            convention.
        quality: This contradiction's `ContradictionQuality` assessment
            (Section 8), if it has been performed. `None` before
            assessment, consistent with `explanation` and `resolution`'s
            "not yet reached" pattern.
        run_refs: The specific experiment run(s) (`models.py`'s
            `RunRef`) this contradiction concerns, if applicable to a
            given `EvidenceItem`'s provenance. Optional context, not a
            substitute for `evidence_items`.
        detected_at: When this contradiction was first detected, if
            known. Caller-supplied, never self-assigned by this module
            -- consistent with this module's determinism requirement
            (module docstring) that it never reads the wall clock
            itself.
        metadata: Free-form, caller-supplied detail that does not fit
            any of the above (e.g. an internal investigation ticket
            reference). Never interpreted by this module.
    """

    id: str
    categories: tuple[ContradictionCategory, ...]
    status: ContradictionStatus = ContradictionStatus.DETECTED
    evidence_items: tuple[EvidenceItem, ...] = ()
    claims: tuple[Claim, ...] = ()
    sources: tuple[ContradictionSource, ...] = ()
    scientific_rule: str | None = None
    explanation: str | None = None
    resolution: ContradictionResolution | None = None
    quality: ContradictionQuality | None = None
    run_refs: tuple[RunRef, ...] = ()
    detected_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ContradictionError("Contradiction.id must be a non-empty string.")

        if not self.categories:
            raise ContradictionError(
                "Contradiction.categories must be non-empty (Section 3: every "
                "applicable category must be recorded)."
            )
        for category in self.categories:
            if not isinstance(category, ContradictionCategory):
                raise ContradictionError(
                    "Every member of Contradiction.categories must be a "
                    f"ContradictionCategory, got {category!r}."
                )

        if not isinstance(self.status, ContradictionStatus):
            raise ContradictionError(
                "Contradiction.status must be a ContradictionStatus member, "
                f"got {self.status!r}."
            )

        for source in self.sources:
            if not isinstance(source, ContradictionSource):
                raise ContradictionError(
                    "Every member of Contradiction.sources must be a "
                    f"ContradictionSource, got {source!r}."
                )

        # Section 2: "two or more evidence items, or between two or
        # more claims" -- a contradiction must name at least two
        # parties in conflict, whether both from evidence_items, both
        # from claims, or split across the two (an evidence-vs-claim
        # contradiction, Section 2's "When evidence contradicts a
        # claim").
        party_count = len(self.evidence_items) + len(self.claims)
        if party_count < 2:
            raise ContradictionError(
                "Contradiction must relate at least two evidence items and/or "
                f"claims in conflict (Section 2), got {party_count}."
            )

        if self.resolution is not None and self.status is not ContradictionStatus.RESOLVED:
            raise ContradictionError(
                "Contradiction.resolution may only be set when status is RESOLVED, "
                f"got status={self.status!r}."
            )
        if self.status is ContradictionStatus.RESOLVED and self.resolution is None:
            raise ContradictionError(
                "Contradiction.resolution is required when status is RESOLVED "
                "(Section 5: a resolved contradiction is retained together with "
                "its explanation)."
            )
        if (
            self.resolution is not None
            and not isinstance(self.resolution, ContradictionResolution)
        ):
            raise ContradictionError(
                "Contradiction.resolution must be a ContradictionResolution, "
                f"got {type(self.resolution).__name__}."
            )

        if self.quality is not None and not isinstance(self.quality, ContradictionQuality):
            raise ContradictionError(
                "Contradiction.quality must be a ContradictionQuality, "
                f"got {type(self.quality).__name__}."
            )

    @property
    def is_resolved(self) -> bool:
        """Whether this contradiction has reached Section 5's
        `RESOLVED` stage. Kept as a named predicate, rather than
        requiring every caller to compare `status` directly, since
        "resolved" is the single lifecycle distinction Section 6's
        downstream relationships (Confidence, Judgment,
        Recommendations) most frequently need to check.
        """
        return self.status is ContradictionStatus.RESOLVED

    @property
    def is_persisting(self) -> bool:
        """Whether this contradiction is recorded as `PERSISTING`
        (Section 5) -- investigated, per the standard Section 5 sets
        out, but not explained sufficiently to reach resolution. Per
        Section 6, a persisting contradiction "must be carried forward
        into any claim, confidence assessment, or judgment that depends
        on the evidence in conflict."
        """
        return self.status is ContradictionStatus.PERSISTING

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization, matching the `to_dict()` convention
        every concrete type in this package follows (see, e.g.,
        `rules.py`'s `RuleContext.to_dict`, which calls this method on
        every member of `detected_contradictions`).

        `evidence_items` and `run_refs` are serialized via their own
        `to_dict()`, per that same convention. `claims` is serialized
        the same way -- `Claim` is referenced only under
        `TYPE_CHECKING` in this module (see module docstring), but any
        concrete `Claim` instance actually stored here is still
        expected, per the package-wide convention, to expose its own
        `to_dict()` at runtime.
        """
        return {
            "id": self.id,
            "categories": [category.value for category in self.categories],
            "status": self.status.value,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "claims": [claim.to_dict() for claim in self.claims],
            "sources": [source.value for source in self.sources],
            "scientific_rule": self.scientific_rule,
            "explanation": self.explanation,
            "resolution": self.resolution.to_dict() if self.resolution is not None else None,
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "run_refs": [
    {
        "backend": run_ref.backend,
        "entity": run_ref.entity,
        "project": run_ref.project,
        "run_id": run_ref.run_id,
    }
    for run_ref in self.run_refs
],
            "detected_at": self.detected_at.isoformat() if self.detected_at is not None else None,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------
# ContradictionSet
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContradictionSet:
    """An ordered, deterministic collection of `Contradiction`s, unique
    by `id`.

    Mirrors this package's convention, established by `ObservationSet`
    and `HypothesisSet`, of wrapping a bare sequence of a stage's
    primary type in a small collection class with lookup helpers, so a
    consumer (e.g. a future `ContradictionRule`, or `confidence.py` /
    `judgment.py` per Chapter 8, Section 10's "Future Integration")
    gets those helpers for free rather than re-implementing filtering
    over a plain tuple. Registration order is preserved exactly,
    matching this module's determinism requirement.
    """

    contradictions: tuple[Contradiction, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for contradiction in self.contradictions:
            if not isinstance(contradiction, Contradiction):
                raise ContradictionError(
                    "Every member of ContradictionSet.contradictions must be a "
                    f"Contradiction, got {type(contradiction).__name__}."
                )
            if contradiction.id in seen:
                raise ContradictionError(
                    f"Duplicate Contradiction.id {contradiction.id!r} in ContradictionSet."
                )
            seen.add(contradiction.id)

    def by_id(self, contradiction_id: str) -> Contradiction | None:
        """The `Contradiction` with `id == contradiction_id`, or `None`
        if no member of this set has that `id`.
        """
        for contradiction in self.contradictions:
            if contradiction.id == contradiction_id:
                return contradiction
        return None

    def by_status(self, status: ContradictionStatus) -> tuple[Contradiction, ...]:
        """Every `Contradiction` whose `status` is exactly `status`, in
        this set's order.
        """
        return tuple(c for c in self.contradictions if c.status is status)

    def by_category(self, category: ContradictionCategory) -> tuple[Contradiction, ...]:
        """Every `Contradiction` for which `category` is one of
        `categories`, in this set's order.
        """
        return tuple(c for c in self.contradictions if category in c.categories)

    def by_source(self, source: ContradictionSource) -> tuple[Contradiction, ...]:
        """Every `Contradiction` for which `source` is one of
        `sources`, in this set's order.
        """
        return tuple(c for c in self.contradictions if source in c.sources)

    def unresolved(self) -> tuple[Contradiction, ...]:
        """Every `Contradiction` whose `status` is not `RESOLVED`, in
        this set's order -- the set Section 6 requires be carried
        forward into any dependent claim, confidence assessment, or
        judgment.
        """
        return tuple(c for c in self.contradictions if not c.is_resolved)

    def persisting(self) -> tuple[Contradiction, ...]:
        """Every `Contradiction` whose `status` is `PERSISTING`, in
        this set's order.
        """
        return tuple(c for c in self.contradictions if c.is_persisting)

    def __len__(self) -> int:
        return len(self.contradictions)

    def __iter__(self) -> Iterator[Contradiction]:
        return iter(self.contradictions)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Contradiction):
            return item in self.contradictions
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.contradictions)} contradiction(s))"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization, matching the `to_dict()` convention
        every concrete type in this package follows.
        """
        return {"contradictions": [c.to_dict() for c in self.contradictions]}


def contradiction_set_from(contradictions: Iterable[Contradiction]) -> ContradictionSet:
    """Build a `ContradictionSet` from any iterable of `Contradiction`s,
    preserving iteration order.

    A small convenience so a caller holding a generator, list, or other
    non-tuple iterable does not need to spell `ContradictionSet(tuple(...))`
    itself.
    """
    return ContradictionSet(tuple(contradictions))
