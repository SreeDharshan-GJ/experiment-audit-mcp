"""
Experiment Audit Scientific Reasoning Engine

Module: judgment

Defines the Judgment stage of the reasoning pipeline (Evidence ->
Observations -> Hypotheses -> Rules -> Confidence -> **Judgment** ->
Recommendation, per research/07_reasoning_engine/reasoning-engine.md's
"Evidence Collection -> Evidence Validation -> Pattern Detection ->
Hypothesis Generation -> Contradiction Search -> Confidence Estimation
-> Scientific Judgment -> Recommendations" pipeline). This module turns
a `ConfidenceSet` (confidence.py) into a `JudgmentSet` — the engine's
named, hedged scientific conclusions, e.g. "Likely overfitting" or
"Possible metric logging issue".

**Scope, strictly bounded.** This module answers exactly one question:
"given how confident we are in each hypothesis, what scientific
conclusion, if any, does that support?" It does not decide *what to do
about it* (that is `recommendation.py`'s job) and it does not compute
*how confident* to be in anything (that is `confidence.py`'s job,
already done by the time a `ConfidenceAssessment` reaches this module —
per confidence-system.md: "Confidence is never guessed. Confidence is
computed," and computed upstream of here). Concretely:

- *what is measurably true* -> `observations.py`
- *why might that be happening* -> `hypotheses.py`
- *is this a known failure pattern, and does anything contradict it* ->
  `rules.py`
- *how sure are we* -> `confidence.py`
- *what do we conclude, and how sure are we of the conclusion itself* ->
  this module
- *what should change* -> `recommendation.py`

Every `Judgment` this module produces must, per reasoning-engine.md's
"Principles" ("Every conclusion must be supported by evidence, include
confidence, explain why, cite supporting observations, identify
contradictions, state limitations. No unsupported claims."):

- **be traceable** — every `Judgment` names the exact `Hypothesis`
  object(s) (`supporting_hypotheses`) it was derived from, and carries
  the exact `ConfidenceAssessment` (`confidence_assessment`) that scored
  those hypotheses, so a reader can walk the citation chain back through
  `hypotheses.py` and `observations.py` without this module having to
  re-state anything it did not itself compute.
- **carry its own confidence** — a `Judgment` never asserts a bare
  verdict; `confidence_assessment` travels with it always, per
  design-spec-v1.md's "Judgment tools show their work" principle
  applied one layer up the pipeline.
- **explain itself** — `scientific_rationale` states the mechanistic
  "why" (e.g. why a train/val divergence pattern is read as overfitting)
  in prose a reviewer can evaluate, not just a label.
- **state its limitations** — `limitations` is never empty; every
  judgment names at least the generic limitation that it is a pattern
  match over prior pipeline stages, not a certainty, plus any
  kind-specific caveats.
- **make no unsupported claims** — a `ConfidenceAssessment` whose
  hypothesis kind this module does not recognize produces no `Judgment`
  at all. This module has a closed, named vocabulary of conclusions
  (`JudgmentKind`); it never invents a new category to force a match,
  and silence (no judgment) is the correct output when nothing in that
  vocabulary is supported by the evidence trail.

**Architectural constraint, mirrored from `evidence.py` and
`observations.py`.** Per this file's own instructions ("Do NOT inspect
Evidence. Do NOT inspect W&B. Do NOT inspect MLflow."), this module has
no dependency on `evidence.py`, `models.py`'s backend-facing types
beyond `RunRef` (used only for the read-only `Judgment.subjects()`
convenience helper, never for lookups against a backend), FastMCP, MCP
transport, `server.py`, or any `ExperimentBackend` implementation
(`WandbBackend`, `FakeBackend`, ...). It operates only on whatever a
`ConfidenceSet` (confidence.py) exposes and returns plain dataclasses
with their own `to_dict()`, consistent with every other module in this
pipeline.

**Upstream contract, documented because it is not yet enforceable.**
At the time this module was written, `confidence.py` and `hypotheses.py`
are themselves unimplemented stubs — there is nothing yet to import
`ConfidenceSet`, `ConfidenceAssessment`, or `Hypothesis` from. Rather
than block on that or invent placeholder classes here (which this
file's remit does not extend to — it may not modify `confidence.py` or
`hypotheses.py`), this module is written *against* the following
documented contract, and depends on it structurally rather than by
concrete import (see "Typing note" below):

- `ConfidenceSet` is iterable, yielding `ConfidenceAssessment` objects,
  in the same append-only, lookup-helper style `ObservationSet`
  (observations.py) already establishes for this codebase's Set types.
- Each `ConfidenceAssessment` exposes:
  - `hypotheses: tuple[Hypothesis, ...]` (preferred) or, for a design
    that scores one hypothesis at a time, a singular `hypothesis:
    Hypothesis` — this module accepts either (see `_hypotheses_of`).
  - `level` — a confidence level whose value compares equal to one of
    `"high"`, `"medium"`, `"low"` (a `StrEnum` member or a plain `str`,
    either works since this module only ever compares by value, never
    by `is`/`isinstance` against a concrete upstream class).
  - `rationale: str` — confidence.py's own explanation of *why* it
    scored the hypothesis the way it did (evidence quality/quantity,
    contradictions, agreement, missing information, per
    confidence-system.md). This module quotes it inside
    `scientific_rationale` rather than re-deriving it.
  - `to_dict() -> dict[str, Any]`, per this codebase's universal
    serialization convention.
- Each `Hypothesis` exposes `kind` (a value comparable against this
  module's expected string vocabulary — see
  `_HYPOTHESIS_KIND_TO_JUDGMENT_KIND`), `statement: str`, and
  `to_dict() -> dict[str, Any]`.

If `confidence.py`/`hypotheses.py` land with a different shape, only
`_hypotheses_of` and `_HYPOTHESIS_KIND_TO_JUDGMENT_KIND` below should
need updating — the rest of this module's logic (grouping, statement
construction, traceability) does not depend on the exact upstream
class, only on the attribute contract above.

**Typing note:** `ConfidenceSet`, `ConfidenceAssessment`, and
`Hypothesis` are imported only under `TYPE_CHECKING` (combined with
`from __future__ import annotations`, already project convention), so
this module can be imported and exercised today without a hard
dependency on `confidence.py`/`hypotheses.py` existing yet, while still
giving a type checker the real signatures once they do.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from experiment_audit.models import RunRef

if TYPE_CHECKING:
    from experiment_audit.reasoning.confidence import ConfidenceAssessment, ConfidenceSet
    from experiment_audit.reasoning.hypotheses import Hypothesis


class JudgmentKind(StrEnum):
    """The closed, named vocabulary of scientific conclusions this
    module can produce.

    A `str` subclass, matching `EvidenceKind`/`ObservationKind`'s
    convention (evidence.py, observations.py), so a kind serializes as
    its own value without a manual `.value` lookup at every call site.

    Deliberately closed: per this module's docstring ("make no
    unsupported claims"), a `ConfidenceAssessment` whose hypothesis
    kind does not map to one of these is not forced into the nearest
    category — it simply produces no `Judgment`. Each member's value is
    also the canonical label `_label_for` prefixes onto that judgment's
    `statement`.
    """

    LIKELY_OVERFITTING = "likely_overfitting"
    LIKELY_OPTIMIZATION_INSTABILITY = "likely_optimization_instability"
    LIKELY_INSUFFICIENT_REPRODUCIBILITY = "likely_insufficient_reproducibility"
    LIKELY_INCOMPLETE_EXPERIMENT = "likely_incomplete_experiment"
    LIKELY_CONFIGURATION_CONFOUND = "likely_configuration_confound"
    POSSIBLE_METRIC_LOGGING_ISSUE = "possible_metric_logging_issue"


@dataclass(frozen=True, slots=True)
class Judgment:
    """One named, hedged scientific conclusion, fully traceable to the
    hypotheses and confidence assessment that produced it.

    Frozen, matching `EvidenceItem`/`Observation`'s convention
    (evidence.py, observations.py): a judgment, once reached, is not
    mutated in place. A later pipeline run that reaches a different
    conclusion from the same or updated inputs produces a new
    `Judgment`, never an edited one — the same "never discard, always
    supersede" discipline this codebase applies throughout.

    Attributes:
        kind: Which named conclusion this is.
        statement: A literal, human-readable sentence stating the
            conclusion, prefixed with `kind`'s canonical label (e.g.
            "Likely overfitting: ..."), followed by the specifics drawn
            from `supporting_hypotheses`' own statements. Written to be
            read on its own, without requiring a reader to first open
            `supporting_hypotheses`.
        supporting_hypotheses: Every `Hypothesis` this judgment was
            derived from, in the order `confidence.py` supplied them
            via the source `ConfidenceAssessment`. Never empty — a
            judgment with no supporting hypothesis cannot be
            traceable, and `JudgmentGenerator` never constructs one
            without at least one.
        confidence_assessment: The exact `ConfidenceAssessment` (from
            `confidence.py`) this judgment's confidence rests on,
            carried through unchanged rather than re-derived — this
            module scores nothing itself, per confidence-system.md's
            "Confidence is never guessed. Confidence is computed"
            (computed upstream, quoted here).
        scientific_rationale: Prose explaining *why* this hypothesis
            pattern supports this conclusion (a mechanistic account,
            e.g. why a train/val divergence reads as overfitting),
            combined with `confidence_assessment.rationale` (the *how
            sure*, as opposed to this field's *why this label*) so a
            reader gets both halves of the explanation
            reasoning-engine.md's principles require ("explain why").
        limitations: Every caveat a reader should weigh before treating
            this judgment as decisive. Never empty (see class
            docstring); always includes at least this module's generic
            "pattern match over prior stages, not a certainty" caveat,
            plus any kind-specific and confidence-level-specific
            caveats `JudgmentGenerator` attaches.
    """

    kind: JudgmentKind
    statement: str
    supporting_hypotheses: tuple[Hypothesis, ...]
    confidence_assessment: ConfidenceAssessment
    scientific_rationale: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.supporting_hypotheses:
            raise ValueError(
                "Judgment.supporting_hypotheses must be non-empty — a judgment "
                "with no cited hypothesis is not traceable, per this module's "
                "traceability requirement."
            )
        if not self.limitations:
            raise ValueError(
                "Judgment.limitations must be non-empty — every judgment must "
                "state at least one limitation, per reasoning-engine.md's "
                "'state limitations' principle."
            )

    def subjects(self) -> tuple[RunRef, ...]:
        """Every run this judgment concerns, deduplicated and in first-seen
        order, unioned across `supporting_hypotheses`.

        A convenience derived from `supporting_hypotheses` (each of
        which is expected to expose its own `subjects`, mirroring
        `Observation.subjects` in observations.py) rather than a stored
        field — computed on demand so `Judgment` never needs to be
        constructed with a redundant, independently-settable copy of
        information its hypotheses already carry.
        """
        seen: dict[RunRef, None] = {}
        for hypothesis in self.supporting_hypotheses:
            for ref in getattr(hypothesis, "subjects", ()):
                seen.setdefault(ref, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "statement": self.statement,
            "supporting_hypotheses": [h.to_dict() for h in self.supporting_hypotheses],
            "confidence_assessment": self.confidence_assessment.to_dict(),
            "scientific_rationale": self.scientific_rationale,
            "limitations": list(self.limitations),
        }


@dataclass(slots=True)
class JudgmentSet:
    """An ordered collection of `Judgment`s, with lookup helpers.

    Mutable and append-only by convention, mirroring
    `ObservationSet`/`Evidence.items` (observations.py, evidence.py):
    `JudgmentGenerator` builds one of these incrementally, and a caller
    may append further judgments of its own (e.g. from a human review
    step) without losing anything already collected. There is
    intentionally no `remove`, for the same "never discard a reached
    conclusion" reason those modules give for their own append-only
    collections — a judgment later found wrong should be superseded by
    recording a new one (or handled at a layer above this module), not
    erased.
    """

    judgments: list[Judgment] = field(default_factory=list)

    def add(self, judgment: Judgment) -> None:
        """Append a single judgment."""
        self.judgments.append(judgment)

    def extend(self, judgments: Iterable[Judgment]) -> None:
        """Append every judgment from `judgments`, in order."""
        self.judgments.extend(judgments)

    def by_kind(self, kind: JudgmentKind) -> list[Judgment]:
        """Every judgment of a given `kind`, in the order recorded."""
        return [j for j in self.judgments if j.kind is kind]

    def by_subject(self, ref: RunRef) -> list[Judgment]:
        """Every judgment whose `subjects()` includes `ref`."""
        return [j for j in self.judgments if ref in j.subjects()]

    def by_hypothesis(self, hypothesis: Hypothesis) -> list[Judgment]:
        """Every judgment whose `supporting_hypotheses` includes `hypothesis`."""
        return [j for j in self.judgments if hypothesis in j.supporting_hypotheses]

    def kinds(self) -> set[JudgmentKind]:
        """The distinct `JudgmentKind`s present in this set."""
        return {j.kind for j in self.judgments}

    def is_empty(self) -> bool:
        """Whether no judgments were recorded at all."""
        return not self.judgments

    def to_dict(self) -> dict[str, Any]:
        return {"judgments": [j.to_dict() for j in self.judgments]}

    def __len__(self) -> int:
        return len(self.judgments)

    def __iter__(self) -> Iterator[Judgment]:
        return iter(self.judgments)

    def __bool__(self) -> bool:
        return bool(self.judgments)


# ---------------------------------------------------------------------
# Upstream hypothesis-kind -> judgment-kind mapping.
#
# The single source of truth for how a `Hypothesis.kind` value (a
# string, or a `StrEnum` member compared by value — see this module's
# docstring "Upstream contract") is classified into one of this
# module's closed `JudgmentKind`s. Deliberately generous with synonyms
# on the left-hand side: since `hypotheses.py` does not exist yet, the
# exact spelling its `HypothesisKind` will use is not yet knowable, so
# several plausible spellings per concept are mapped to the same
# `JudgmentKind` rather than guessing a single one and silently
# dropping every judgment if that guess is wrong. Kept in sync by hand
# with `hypotheses.py` once it lands — the same caveat
# `analysis/divergence.py` and `analysis/confound.py` already document
# for their own constants.
# ---------------------------------------------------------------------
_HYPOTHESIS_KIND_TO_JUDGMENT_KIND: dict[str, JudgmentKind] = {
    "overfitting": JudgmentKind.LIKELY_OVERFITTING,
    "possible_overfitting": JudgmentKind.LIKELY_OVERFITTING,
    "train_val_divergence": JudgmentKind.LIKELY_OVERFITTING,
    "optimization_instability": JudgmentKind.LIKELY_OPTIMIZATION_INSTABILITY,
    "gradient_instability": JudgmentKind.LIKELY_OPTIMIZATION_INSTABILITY,
    "training_instability": JudgmentKind.LIKELY_OPTIMIZATION_INSTABILITY,
    "loss_divergence": JudgmentKind.LIKELY_OPTIMIZATION_INSTABILITY,
    "insufficient_reproducibility": JudgmentKind.LIKELY_INSUFFICIENT_REPRODUCIBILITY,
    "low_statistical_confidence": JudgmentKind.LIKELY_INSUFFICIENT_REPRODUCIBILITY,
    "single_random_seed": JudgmentKind.LIKELY_INSUFFICIENT_REPRODUCIBILITY,
    "single_seed": JudgmentKind.LIKELY_INSUFFICIENT_REPRODUCIBILITY,
    "incomplete_experiment": JudgmentKind.LIKELY_INCOMPLETE_EXPERIMENT,
    "missing_evidence": JudgmentKind.LIKELY_INCOMPLETE_EXPERIMENT,
    "missing_baseline": JudgmentKind.LIKELY_INCOMPLETE_EXPERIMENT,
    "configuration_confound": JudgmentKind.LIKELY_CONFIGURATION_CONFOUND,
    "confounded_experiment": JudgmentKind.LIKELY_CONFIGURATION_CONFOUND,
    "confounded_comparison": JudgmentKind.LIKELY_CONFIGURATION_CONFOUND,
    "comparison_invalid": JudgmentKind.LIKELY_CONFIGURATION_CONFOUND,
    "metric_logging_issue": JudgmentKind.POSSIBLE_METRIC_LOGGING_ISSUE,
    "logging_anomaly": JudgmentKind.POSSIBLE_METRIC_LOGGING_ISSUE,
    "nan_metric_values": JudgmentKind.POSSIBLE_METRIC_LOGGING_ISSUE,
}

# Canonical human-readable label each `JudgmentKind` prefixes onto its
# `statement` — verbatim the phrasing named in this module's spec.
_JUDGMENT_LABELS: dict[JudgmentKind, str] = {
    JudgmentKind.LIKELY_OVERFITTING: "Likely overfitting",
    JudgmentKind.LIKELY_OPTIMIZATION_INSTABILITY: "Likely optimization instability",
    JudgmentKind.LIKELY_INSUFFICIENT_REPRODUCIBILITY: "Likely insufficient reproducibility",
    JudgmentKind.LIKELY_INCOMPLETE_EXPERIMENT: "Likely incomplete experiment",
    JudgmentKind.LIKELY_CONFIGURATION_CONFOUND: "Likely configuration confound",
    JudgmentKind.POSSIBLE_METRIC_LOGGING_ISSUE: "Possible metric logging issue",
}

# One-sentence mechanistic accounts of *why* each pattern supports its
# conclusion — the "explain why" half of `scientific_rationale`
# (the other half, "how sure", comes from `ConfidenceAssessment.rationale`).
_JUDGMENT_MECHANISMS: dict[JudgmentKind, str] = {
    JudgmentKind.LIKELY_OVERFITTING: (
        "Divergence between training and validation performance is a "
        "canonical signature of overfitting: the model is increasingly "
        "fitting patterns specific to the training set that do not "
        "generalize to held-out data."
    ),
    JudgmentKind.LIKELY_OPTIMIZATION_INSTABILITY: (
        "Diverging loss and/or exploding gradient norms indicate the "
        "optimizer has left a stable descent regime, typically from a "
        "learning rate, initialization, or numerical-precision issue "
        "rather than a property of the data or model capacity."
    ),
    JudgmentKind.LIKELY_INSUFFICIENT_REPRODUCIBILITY: (
        "A conclusion drawn from a single random seed (or otherwise "
        "under-replicated run) cannot be distinguished from a "
        "seed-specific artifact; repeated runs across seeds are needed "
        "before the result can be treated as a property of the method "
        "rather than of one stochastic realization of it."
    ),
    JudgmentKind.LIKELY_INCOMPLETE_EXPERIMENT: (
        "Evidence categories needed to evaluate the experiment on its "
        "own claims (e.g. a baseline to compare against, or metrics/"
        "dataset/hardware context) are absent, so the experiment cannot "
        "yet be fully assessed on the terms it appears to claim."
    ),
    JudgmentKind.LIKELY_CONFIGURATION_CONFOUND: (
        "More than one configuration parameter differs between the runs "
        "being compared, so any observed metric difference cannot be "
        "attributed to a single claimed variable — the comparison does "
        "not isolate its intended cause."
    ),
    JudgmentKind.POSSIBLE_METRIC_LOGGING_ISSUE: (
        "Null/NaN values or other irregularities in a metric's logged "
        "history are as consistent with an instrumentation or logging "
        "defect as with a genuine property of training, and cannot be "
        "distinguished from the metric values alone."
    ),
}

# Kind-specific limitations appended alongside the generic caveat every
# judgment carries (see `_GENERIC_LIMITATION`).
_JUDGMENT_KIND_LIMITATIONS: dict[JudgmentKind, tuple[str, ...]] = {
    JudgmentKind.LIKELY_OVERFITTING: (
        "Does not rule out a labeling error, a distribution shift "
        "between training and validation splits, or a validation-set "
        "logging issue as alternative explanations for the divergence.",
    ),
    JudgmentKind.LIKELY_OPTIMIZATION_INSTABILITY: (
        "Does not distinguish a genuine optimization failure from a "
        "one-off hardware/numerical fault (e.g. a transient NaN from "
        "mixed-precision under/overflow) without a repeat run.",
    ),
    JudgmentKind.LIKELY_INSUFFICIENT_REPRODUCIBILITY: (
        "Does not itself estimate how large the seed-to-seed variance "
        "is likely to be — only that it has not been measured.",
    ),
    JudgmentKind.LIKELY_INCOMPLETE_EXPERIMENT: (
        "Absence of recorded evidence is not proof the underlying fact "
        "does not exist elsewhere (e.g. in an untracked log); this "
        "judgment reflects what was supplied to the pipeline, not a "
        "verified absence.",
    ),
    JudgmentKind.LIKELY_CONFIGURATION_CONFOUND: (
        "Does not itself judge whether the differing parameter(s) are "
        "innocuous (e.g. an allowlisted bookkeeping field); that "
        "classification belongs to the rule that produced the "
        "underlying hypothesis.",
    ),
    JudgmentKind.POSSIBLE_METRIC_LOGGING_ISSUE: (
        "Does not distinguish a logging/instrumentation defect from a "
        "genuine numerical failure in training (e.g. an actual NaN loss) "
        "without further investigation.",
    ),
}

_GENERIC_LIMITATION = (
    "This judgment is a pattern match over the Hypotheses, Rules, and "
    "Confidence stages that precede it in the reasoning pipeline, not an "
    "independent re-examination of the underlying Evidence, run data, or "
    "backend; it is only as sound as those upstream stages."
)


class JudgmentGenerator:
    """Converts a `ConfidenceSet` into a `JudgmentSet`. No new scoring,
    no new evidence inspection — only classification and traceable
    composition of what upstream stages already computed.

    Stateless and pure: `generate` is a deterministic function of its
    `confidence_set` argument. Calling it twice with equal input always
    returns equal (by value) output.
    """

    def generate(self, confidence_set: ConfidenceSet) -> JudgmentSet:
        """Produce one `Judgment` per recognized `ConfidenceAssessment`.

        Args:
            confidence_set: The `ConfidenceSet` (confidence.py) to
                convert. Iterated once, in order; this method never
                reaches back into `evidence.py`, `observations.py`, a
                backend, or any other source — everything a `Judgment`
                needs is read off `confidence_set`'s own assessments
                and the `Hypothesis` objects they cite (per this
                module's "Do NOT inspect Evidence" constraint).

        Returns:
            A `JudgmentSet` with one `Judgment` per assessment whose
            hypothesis kind is recognized (see
            `_HYPOTHESIS_KIND_TO_JUDGMENT_KIND`). An assessment citing
            an unrecognized hypothesis kind, or citing no hypothesis at
            all, contributes no `Judgment` — silence, not a forced or
            invented category, per this module's "make no unsupported
            claims" principle.
        """
        result = JudgmentSet()
        for assessment in confidence_set:
            judgment = self._judge(assessment)
            if judgment is not None:
                result.add(judgment)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _judge(self, assessment: ConfidenceAssessment) -> Judgment | None:
        """Classify and build one `Judgment` from `assessment`, or return
        `None` if `assessment` cites no hypothesis this module recognizes.
        """
        hypotheses = _hypotheses_of(assessment)
        if not hypotheses:
            return None

        kind = self._classify(hypotheses[0])
        if kind is None:
            return None

        return Judgment(
            kind=kind,
            statement=self._build_statement(kind, hypotheses),
            supporting_hypotheses=hypotheses,
            confidence_assessment=assessment,
            scientific_rationale=self._build_rationale(kind, assessment),
            limitations=self._build_limitations(kind),
        )

    def _classify(self, hypothesis: Hypothesis) -> JudgmentKind | None:
        """Map `hypothesis.kind` to a `JudgmentKind`, or `None` if
        unrecognized.

        Compared by string value (`str(hypothesis.kind)`), not by
        `isinstance`/identity against a concrete `HypothesisKind` class
        — per this module's "Upstream contract" docstring section, a
        plain `str` or any `StrEnum`-like value works identically here.
        """
        key = str(getattr(hypothesis, "kind", "")).lower()
        return _HYPOTHESIS_KIND_TO_JUDGMENT_KIND.get(key)

    def _build_statement(
        self, kind: JudgmentKind, hypotheses: tuple[Hypothesis, ...]
    ) -> str:
        """Compose the literal, human-readable `Judgment.statement`.

        Prefixes `kind`'s canonical label, then appends the specifics
        already captured in each hypothesis's own `statement` — this
        module never re-derives those specifics itself, only cites
        them, per the traceability requirement in this module's
        docstring.
        """
        label = _JUDGMENT_LABELS[kind]
        details = "; ".join(
            str(getattr(h, "statement", "")).strip()
            for h in hypotheses
            if str(getattr(h, "statement", "")).strip()
        )
        if not details:
            return f"{label}."
        return f"{label}: {details}"

    def _build_rationale(self, kind: JudgmentKind, assessment: ConfidenceAssessment) -> str:
        """Compose `Judgment.scientific_rationale` from this module's
        boilerplate mechanistic account plus `confidence.py`'s own
        computed rationale for the assessment (the "why this label"
        and "how sure" halves, respectively — see `Judgment`'s
        docstring for `scientific_rationale`).
        """
        mechanism = _JUDGMENT_MECHANISMS[kind]
        confidence_rationale = str(getattr(assessment, "rationale", "")).strip()
        level = getattr(assessment, "level", None)
        level_clause = f" Assessed confidence: {level}." if level is not None else ""
        if confidence_rationale:
            return f"{mechanism} {confidence_rationale}{level_clause}"
        return f"{mechanism}{level_clause}"

    def _build_limitations(self, kind: JudgmentKind) -> tuple[str, ...]:
        """Compose `Judgment.limitations`: the generic caveat every
        judgment carries, followed by `kind`'s specific caveats.
        """
        return (_GENERIC_LIMITATION, *_JUDGMENT_KIND_LIMITATIONS.get(kind, ()))


def _hypotheses_of(assessment: ConfidenceAssessment) -> tuple[Hypothesis, ...]:
    """Normalize a `ConfidenceAssessment`'s cited hypothesis/hypotheses
    into a tuple, regardless of whether `confidence.py` exposes a
    plural `hypotheses` collection or a singular `hypothesis` field.

    Prefers a plural `hypotheses` attribute (symmetric with
    `Judgment.supporting_hypotheses`, and the more general of the two
    possible upstream shapes); falls back to wrapping a singular
    `hypothesis` attribute in a one-element tuple; returns an empty
    tuple if neither is present (or if `hypotheses` is itself empty),
    which `JudgmentGenerator._judge` treats as "no judgment possible"
    rather than an error, per this module's "make no unsupported
    claims" principle — an assessment this module cannot trace back to
    at least one hypothesis is not something this module can turn into
    a traceable `Judgment`.
    """
    plural = getattr(assessment, "hypotheses", None)
    if plural:
        return tuple(plural)
    singular = getattr(assessment, "hypothesis", None)
    if singular is not None:
        return (singular,)
    return ()
