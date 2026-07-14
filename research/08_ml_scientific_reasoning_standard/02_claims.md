# 1. Purpose

Machine learning research does not exist to produce experiments. It exists
to produce knowledge, and knowledge in a scientific field takes the form of
claims — explicit statements about how a method behaves, how it compares to
alternatives, and under what conditions those statements hold. An
experiment, however carefully designed and executed, is not itself a
contribution; it is the evidence (Chapter 1) that a contribution is built
from. A body of runs, logs, and metrics that is never distilled into a
claim has produced data, not research.

This distinction has a direct consequence for how Experiment Audit is
structured. Experiment Audit does not evaluate experiments in isolation —
it does not simply describe what a run did. It evaluates claims through the
evidence assembled to support them. This ordering is deliberate: an
experiment is judged well or poorly not by its own internal properties, but
by whether it succeeds in warranting the claim it is offered in support of.
A flawless run that supports no clearly stated claim is, for the purposes
of this methodology, scientifically inert. A modest run that is explicit
about the narrow claim it supports is not.

This chapter defines what a claim is, so that every later stage of this
methodology — evidence-to-claim traceability, confidence assessment,
contradiction detection, and judgment — has a fixed, unambiguous target to
operate on. A methodology that can rigorously evaluate evidence but leaves
"claim" informally defined has not solved the harder problem; claims are
the unit that all of this machinery ultimately exists to assess.

--------------------------------------------------

# 2. Definition of a Scientific Claim

A **scientific claim** is an explicit statement asserting something about
a method, model, or system, with respect to one or more of the following:
performance, behavior, robustness, efficiency, generalization,
reproducibility, or comparison to an alternative.

Three features of this definition are load-bearing:

- **Explicit.** A claim is stated, not implied. A collection of favorable
  metrics does not itself constitute a claim; a claim exists only once
  something has been asserted about what those metrics mean.
- **Assertive.** A claim commits to a position that could, in principle,
  be wrong. A statement phrased so cautiously that no evidence could ever
  contradict it does not function as a scientific claim within this
  methodology, whatever its wording.
- **Scoped to a subject.** A claim is always about something — a specific
  method, model, or system — not a free-floating generalization about the
  field. Section 5 addresses the scope of a claim in more detail.

A claim must be distinguished from three concepts already introduced or
implied in this methodology, each of which occupies a different position
in the reasoning chain:

**A claim is not evidence.** Evidence (Chapter 1) is the verifiable
artifact a claim is answerable to. A claim can be stated before, during,
or after evidence exists for it, but the claim itself is never the
artifact — it is the assertion made about what the artifact shows or will
show.

**A claim is not an observation.** An observation is a statement about
what a specific piece of evidence shows, scoped tightly to that evidence —
for example, that a particular run's loss decreased monotonically. A claim
is broader and more consequential: it is the assertion built from one or
more observations that says something is true of the method in general,
within a stated scope, not merely true of one recorded run.

**A claim is not a hypothesis.** A hypothesis is a proposed explanation
for a pattern, offered before or independent of full evidentiary support,
and is understood from the outset as provisional. A claim, by contrast, is
put forward as an assertion the author is prepared to defend with
evidence, not merely propose as a candidate explanation. A hypothesis can
motivate the work that eventually produces a claim, but the two remain
distinct: a hypothesis is where inquiry starts, a claim is what inquiry is
offered to have established.

Collapsing a claim into evidence, an observation, or a hypothesis is a
methodological error with the same consequence identified in Chapter 1 for
collapsing evidence with the concepts adjacent to it: it destroys the
distinctions that allow later reasoning — support, verification,
confidence, contradiction — to operate on well-defined inputs rather than
on an undifferentiated mass of assertion.

--------------------------------------------------

# 3. Claim Lifecycle

A claim, like an evidence item, passes through a defined sequence of
stages. Its position in that sequence determines what may legitimately be
said about it and what obligations remain outstanding before it can be
relied upon.

**Formulated.** The stage at which a claim is first stated explicitly, in
terms precise enough to be evaluated. A formulated claim need not yet have
any evidence attached to it; formulation is the act of committing to a
specific, checkable assertion rather than a vague impression.

**Supported.** The stage at which evidence has been identified and
associated with the claim, per the categories and relationships defined in
Chapter 1. A supported claim is not yet judged to be correct — support
establishes only that evidence relevant to the claim exists and has been
linked to it.

**Evaluated.** The stage at which the associated evidence has been
assessed against the properties and quality dimensions defined in Chapter
1 — completeness, consistency, coverage, and the rest — to determine how
well it actually bears on the claim as scoped. Evaluation can reveal that
apparently supporting evidence is weaker, narrower, or more equivocal than
it first appeared.

**Verified.** The stage at which the evaluated evidence has been checked
against the specific assertion the claim makes, confirming that the
evidence, taken as a whole, is consistent with the claim as stated rather
than with some related but different claim. Verification is where a claim
that has quietly drifted from what its evidence actually supports — a
common failure mode — is expected to be caught.

**Published.** The stage at which a verified claim is recorded as a
standing assertion available for others to build on, cite, or challenge.
Publication does not mean the claim is beyond question; it means the claim
has completed the stages above and is now part of the record other claims
may relate to.

**Re-evaluated.** The stage a published claim returns to whenever new
evidence becomes available that bears on it — a failed reproduction
attempt, a new baseline, a later result that contradicts it. A claim does
not exit the lifecycle upon publication; publication marks entry into an
ongoing obligation to remain consistent with the evidentiary record as
that record grows.

A claim that has not progressed through these stages in order — for
example, one treated as verified without ever having been evaluated —
carries an implicit gap that this methodology treats as a defect in the
claim's standing, not as a shortcut.

--------------------------------------------------

# 4. Categories of Claims

Experiment Audit recognizes the following categories of claims. As with
the evidence categories in Chapter 1, these describe what kind of
assertion a claim makes, not how strong or well-supported it is.

**Performance Claims.** Assertions about how well a method achieves a
stated objective, measured against a defined metric — for example, that a
model attains a certain accuracy or reward level. Performance claims are
the most common category and are frequently the claim category other
categories are ultimately in service of.

**Comparison Claims.** Assertions about how one method's performance,
behavior, or efficiency relates to another's, under matched conditions. A
comparison claim depends on comparison and baseline evidence specifically
(Chapter 1) and is invalid if the methods being compared were not
evaluated under genuinely matched conditions.

**Generalization Claims.** Assertions that a method's behavior, observed
under one set of conditions, extends to other conditions not directly
tested — a different dataset, a different distribution, a different task.
Generalization claims carry a scope obligation (Section 5) that is
stricter than most other categories, because they assert something beyond
what was directly measured.

**Robustness Claims.** Assertions about how a method's behavior holds up
under perturbation, noise, adversarial input, or conditions that deviate
from the conditions it was primarily developed under. A robustness claim
requires evidence collected specifically under the perturbed conditions
being claimed about, not evidence collected only under nominal conditions.

**Efficiency Claims.** Assertions about the resource cost of achieving a
result — compute, memory, wall-clock time, sample count — independent of
the quality of the result itself. Efficiency claims must specify what
resource is being claimed about, since a method can be efficient along one
axis and inefficient along another.

**Scalability Claims.** Assertions about how a method's behavior, cost, or
performance changes as some factor increases — model size, data volume,
number of agents, compute budget. A scalability claim requires evidence
gathered across a range of that factor, not evidence from a single point
along it.

**Statistical Claims.** Assertions about the properties of a result across
repeated measurement — that an observed difference is or is not
distinguishable from the variation expected by chance. Statistical claims
depend specifically on statistical evidence (Chapter 1) and are among the
categories most frequently made without adequate support (Section 8).

**Reproducibility Claims.** Assertions that a result can be, or has been,
regenerated by an independent execution of the same nominal experiment.
A reproducibility claim requires reproducibility evidence specifically —
an actual attempt at independent regeneration — and is not satisfied by
the original result alone, however carefully that original result was
recorded.

A single piece of research frequently advances claims from several of
these categories at once, and a claim in one category often depends on a
claim or evidence from another — a comparison claim, for instance,
ordinarily depends on performance claims for each method being compared.
Categorization exists to make these dependencies visible, not to force
each claim into exactly one box.

--------------------------------------------------

# 5. Claim Scope

Every claim has a scope: the specific conditions under which the evidence
supporting it was actually gathered, and therefore the specific conditions
under which the claim is actually warranted. Scope ordinarily includes,
among other factors:

- the specific dataset or task the evidence was collected on;
- the specific model or architecture evaluated;
- the specific hardware the experiment was executed on, where relevant to
  the claim;
- the specific evaluation protocol used to produce the supporting
  evidence;
- the specific software and data environment in effect at the time.

A claim's scope is not an optional qualifier that can be stripped away for
convenience — it is part of what the claim asserts. A performance claim
established on one dataset is a claim about that dataset; extending it, as
stated, to other datasets is not a restatement of the same claim but the
formulation of a new, broader claim that requires its own evidence.

This matters because the most common way a claim becomes unwarranted is
not through fabricated or absent evidence, but through scope drift: a
claim that begins narrowly and accurately scoped is progressively restated
in broader terms — "improves performance on the benchmark" becomes
"improves performance" without qualification — until the claim as
currently stated exceeds what any evidence was ever gathered to support.
Experiment Audit treats a claim's stated scope as binding: a claim may not
be evaluated as though it applies more broadly than its evidence was
collected to support, and any broadening of scope requires either new
evidence at the broader scope or an explicit, separate generalization
claim (Section 4) whose own support can be assessed on its own terms.

--------------------------------------------------

# 6. Claim Strength

Claims exist on a spectrum of strength, reflecting how much the available
evidence warrants the assertion being made. This methodology recognizes
the following positions along that spectrum, ordered from weakest to
strongest:

**Observation.** A statement tied tightly to a single piece of evidence,
not yet generalized into an assertion about the method itself. An
observation is the raw material a claim may eventually be built from, not
yet a claim in its own right.

**Suggestion.** An assertion offered on the basis of limited or
preliminary evidence — a single run, an early result — put forward
explicitly as provisional and inviting further investigation rather than
as an established position.

**Supported Claim.** An assertion backed by evidence that has passed
through evaluation (Section 3) and is judged relevant, but that has not
yet been checked against the range of conditions, repetitions, or
counter-evidence that would justify greater confidence.

**Strongly Supported Claim.** An assertion backed by evidence that spans
multiple independent sources, has been checked for consistency, and has
withstood attempts to find contradicting evidence, without yet having
accumulated the breadth of independent confirmation associated with an
established claim.

**Established Claim.** An assertion backed by evidence broad and
consistent enough, across independent sources and conditions matching the
claim's stated scope, that it functions as a stable basis for further
work rather than as a position still under active test.

Movement along this spectrum is governed by a single principle: a stronger
position on the spectrum requires stronger, not merely more voluminous,
evidence, per the quality dimensions defined in Chapter 1. This
methodology does not attach numerical thresholds to these positions and
does not define an algorithm for computing which position a given claim
occupies; that determination belongs to the confidence assessment
component described in Section 10 and detailed in a later chapter. What
this section fixes is only the ordered set of positions themselves, so
that later components have a shared, unambiguous vocabulary to place
claims into.

--------------------------------------------------

# 7. Claim Relationships

Claims, like evidence, do not exist in isolation from one another.
Experiment Audit recognizes the following relationships between claims:

**Independent.** Two claims are independent when neither's truth bears on
the other's — they concern different subjects, or the same subject along
dimensions that do not interact. Independence between claims, as with
independence between evidence, must be established rather than assumed by
default.

**Dependent.** One claim depends on another when its own validity
presupposes that the other claim holds — for example, a comparison claim
depending on the performance claims for each method being compared. A
dependent claim inherits the weaknesses of the claims it depends on; it
cannot be stronger than its weakest dependency.

**Supporting.** One claim supports another when it provides grounds that
increase the warrant for accepting the other, distinct from evidentiary
support (Chapter 1) in that this relationship holds between two claims
rather than between an evidence item and a claim — for example, an
efficiency claim supporting a broader claim of practical viability.

**Contradictory.** Two claims are contradictory when they cannot both be
true as stated, given their respective scopes. Contradiction between
claims is addressed at greater length in a later chapter of this
methodology; this document establishes only that the relationship must be
recognized and recorded, never silently resolved by favoring one claim
without justification.

**Derived.** One claim is derived from another when it is produced by
extending, restricting, or recombining an existing claim rather than by
direct evaluation of new evidence — for example, a generalization claim
derived from a narrower performance claim. A derived claim carries the
scope limitations of the claim it is derived from as an explicit
obligation, not as a detail that disappears in restatement.

**Equivalent.** Two claims are equivalent when they assert the same thing
within the same scope, differing only in phrasing or in the specific
evidence cited in their support. Recognizing equivalence prevents the same
underlying assertion from being counted, or contradicted, as though it
were two separate claims.

These relationships matter because scientific reasoning over a body of
claims — rather than over any single claim in isolation — depends on
knowing which claims stand or fall together, which are genuinely
independent sources of support, and which merely restate one another in
different words. A judgment formed without this relational structure risks
treating a set of dependent or equivalent claims as though they
constituted broad, independent confirmation, when in fact they trace back
to a single evidentiary source.

--------------------------------------------------

# 8. Unsupported Claims

A claim that has been formulated (Section 3) does not thereby become
warranted. Experiment Audit distinguishes several conditions under which a
claim lacks adequate support:

- **No evidence.** The claim has been stated but no evidence has been
  linked to it at all.
- **Weak evidence.** Evidence exists and is linked to the claim, but it is
  deficient along one or more of the quality dimensions defined in
  Chapter 1 — narrow coverage, low reliability, poor consistency.
- **Missing evidence.** A specific category of evidence the claim requires
  — a baseline for a comparison claim, statistical evidence for a
  statistical claim — is absent, per Chapter 1's treatment of missing
  evidence as itself methodologically significant.
- **Conflicting evidence.** Evidence linked to the claim includes items
  that contradict one another, leaving the claim's actual evidentiary
  position unresolved rather than simply weak.
- **Insufficient evidence.** The evidence present is relevant and
  consistent but does not reach the coverage or volume the claim's scope
  and strength position would require.

None of these conditions is grounds for discarding a claim outright — an
unsupported claim can be a legitimate suggestion (Section 6) or a
motivation for further work. What this methodology does not permit is
silent acceptance: a claim in any of the conditions above must be marked
as such, and must not be allowed to appear, in any later stage of
reasoning, indistinguishable from a claim that has actually earned a
stronger position on the strength spectrum. An unsupported claim that is
never labeled as unsupported is, functionally, a false claim of support in
its own right.

--------------------------------------------------

# 9. Traceability

Every claim admitted under this methodology must be traceable back
through the full reasoning chain that led to it:

Judgment
↓
Confidence
↓
Scientific Rules
↓
Hypotheses
↓
Observations
↓
Evidence

This chain is cumulative: a judgment about a claim rests on a confidence
assessment; that assessment rests on the application of scientific rules;
those rules operate over hypotheses; hypotheses are formed from
observations; and observations are drawn from evidence. A claim cannot be
considered properly established if any link in this chain is missing,
substituted with an unstated assumption, or replaced by a shortcut from a
higher stage directly to the claim without passing through the
intermediate stages.

Traceability is not a record-keeping convenience layered on top of claims
after the fact. It is what allows a claim to be re-examined — by a
reviewer, by a later researcher, or by the re-evaluation stage of the
claim lifecycle (Section 3) — without having to take the claim's standing
on faith. A claim that cannot be walked back through this chain, however
plausible it sounds, has not met the requirements this methodology sets
for what counts as a scientific claim in the first place, regardless of
how it is labeled.

--------------------------------------------------

# 10. Future Integration

This document defines claims independently of the components that will
later consume them. Those components are described elsewhere in this
methodology; this section notes only their conceptual dependency on the
definitions above.

**Scientific Rule Engine.** Applies the rules of correct scientific
reasoning to claims and the evidence, observations, and hypotheses beneath
them, depending on the claim lifecycle (Section 3) and category structure
(Section 4) established here to know what stage and kind of claim it is
operating on.

**Claim Verification.** Checks a claim's stated scope (Section 5) and
category (Section 4) against the evidence attributed to it, depending
directly on this chapter for the distinctions — claim versus evidence,
versus observation, versus hypothesis — that make verification a
well-defined operation rather than an impressionistic judgment.

**Confidence Assessment.** Places a claim on the strength spectrum defined
in Section 6, depending on the evidence quality dimensions of Chapter 1
and the relational structure of Section 7 to avoid overweighting claims
whose apparent support is largely dependent or derived rather than
independent.

**Contradiction Engine.** Detects and records contradictory relationships
between claims (Section 7), depending on claim scope (Section 5) to
correctly distinguish genuine contradiction from claims that merely
appear to conflict because their scopes were never made explicit.

**Judgment Generation.** Produces a terminal scientific judgment about a
claim, remaining traceable through the full chain defined in Section 9
back to the underlying evidence.

**Recommendation Engine.** Consumes judgments about claims to propose
actions, and depends on every prior stage — claim definition, scope,
strength, relationships, and traceability — having been preserved
faithfully, since a recommendation is only as defensible as the claim it
acts on.

No component in this list is permitted to treat a claim as self-
justifying. Every claim that reaches any of these components must arrive
with its scope, category, strength position, and evidentiary chain intact,
consistent with the definitions fixed in this chapter.
