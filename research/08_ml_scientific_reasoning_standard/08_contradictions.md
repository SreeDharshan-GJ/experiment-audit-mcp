# 1. Purpose

A methodology that only ever produces agreement has not been tested. If
every piece of evidence gathered in support of a claim happens to point
the same way, one of two things is true: either the claim is genuinely
well established, or the evidence has been selected, interpreted, or
gathered in a way that suppressed whatever would have disagreed with it.
From the outside, these two situations can look identical. The only way
to tell them apart is to have a methodology that actively looks for
disagreement and reports it when found, rather than one that simply
stops looking once a satisfying answer appears.

This is the role contradictions play in Experiment Audit. A contradiction
is not a malfunction of the reasoning process; it is frequently the most
scientifically informative output the process can produce. An experiment
that improves one metric while degrading a related one is not a broken
experiment — it is an experiment that has revealed a tradeoff. A claim
that holds under one evaluation protocol and fails under another has not
been falsified; it has had its true scope discovered. Machine learning
research is full of results that seem to disagree with each other only
because the disagreement was never surfaced clearly enough to be
understood. Contradictions, properly handled, are how that understanding
is built.

The alternative — a methodology that quietly discards or averages away
conflicting evidence — produces conclusions that look more confident than
they are entitled to be. This is a common failure mode in informal
research practice: a result that partially fails is reported as a
success with the failing part omitted, or two experiments that disagree
are resolved by simply trusting whichever one supports the preferred
narrative. Confidence built this way is not scientific confidence; it is
the appearance of confidence purchased by hiding the evidence that would
have undermined it.

This document defines what Experiment Audit treats as a contradiction,
distinguishes genuine contradictions from evidence that merely appears to
disagree, and specifies how a detected contradiction must be handled so
that it strengthens the methodology's output rather than being treated as
an inconvenience to be resolved by whichever route is fastest. It builds
directly on the definitions of evidence (Chapter 1) and claims (Chapter
2): a contradiction, as defined here, is always a relationship between
two or more instances of evidence, or between two or more claims, and
cannot be understood independently of those prior definitions.

--------------------------------------------------

# 2. Definition of a Scientific Contradiction

A **scientific contradiction** is a documented relationship between two
or more evidence items, or between two or more claims, in which one
cannot be true, valid, or fully supported without the other being false,
invalid, or unsupported, given that both are understood under the same
scope, conditions, and terms.

The final clause of this definition — *given that both are understood
under the same scope, conditions, and terms* — is the load-bearing part
of the definition and is the subject of most of the reasoning in this
chapter. Two statements that appear to conflict on the surface but are
actually scoped differently are not a contradiction under this
methodology. They are two claims about two different things that happen
to share similar wording. Establishing whether an apparent conflict is a
genuine contradiction or a difference in scope is the first analytical
step this methodology requires whenever disagreement is observed, and it
must be performed before any contradiction is recorded as such.

**When evidence contradicts a claim.** Evidence contradicts a claim when
the evidence, taken on its own terms, is inconsistent with what the claim
asserts, within the scope the claim declares for itself. If a claim
asserts that a method improves accuracy under a stated evaluation
protocol, and an evidence item produced under that same protocol shows no
improvement or a regression, the evidence contradicts the claim. If the
evidence was produced under a materially different protocol than the one
the claim specifies, the relationship is not automatically a
contradiction; it may instead indicate that the claim's scope was stated
too broadly, which is itself a finding worth recording (Section 3,
Scope Contradictions), but it is a different finding from a direct
evidentiary contradiction.

**When two claims contradict each other.** Two claims contradict each
other when both are stated with the same subject and the same scope, and
the assertions they make cannot both be true. A claim that method A
outperforms method B on a given benchmark, and a separately recorded
claim that method B outperforms method A on that same benchmark under
the same conditions, are in direct contradiction. This is distinct from
two claims that merely differ — for example, a claim about performance
on one benchmark and a claim about performance on a different benchmark
do not contradict each other even if their conclusions about which
method is "better" point in different directions, because their scopes
are not the same.

**When apparent contradictions are actually differences in scope.** The
majority of disagreements encountered in practice are not contradictions
under this definition; they are unstated scope differences that produce
the appearance of contradiction. Common instances include: a result that
holds on one dataset but not another, presented as a general claim about
a method rather than a dataset-specific one; a result obtained under one
set of hyperparameters, compared against a result obtained under a
different set, without the difference in configuration being
acknowledged; or a result that holds at one scale of model or data,
compared against a result at a different scale. This methodology
requires that scope be checked and, where necessary, made explicit before
any conflict is classified as a contradiction. A conflict that resolves
once scope is stated correctly is recorded as a resolved scope
difference (Section 3), not as a persisting contradiction, and the two
must never be conflated in downstream reporting — a resolved scope
difference communicates something categorically different from a
contradiction that remains open.

A contradiction, once established under this definition, is not itself
evidence of error on either side. It is a signal that further
investigation is required before either side can be trusted, and the
remainder of this document specifies how that investigation is expected
to proceed.

--------------------------------------------------

# 3. Categories of Contradictions

Contradictions are not a single undifferentiated phenomenon. This
methodology distinguishes the following categories, each of which
implicates different evidence, different investigative steps, and
different downstream handling.

**Evidence Contradictions.** A direct conflict between two evidence
items that are, on inspection, comparable — produced under matched
conditions, measuring the same quantity, drawn from the same category of
artifact (Chapter 1, Section 4) — yet reporting inconsistent values. An
evidence contradiction is the most granular category: it exists prior to
and independent of whatever claims the conflicting evidence items are
attached to.

**Claim Contradictions.** A direct conflict between two claims, as
defined in Section 2, where each claim may individually be well
supported by its own evidence but the two assertions cannot both hold
under the same scope. Claim contradictions are frequently the visible
symptom of an underlying evidence contradiction, but they can also arise
from two claims that are each supported by entirely consistent evidence
internally, while still asserting incompatible things about the same
subject.

**Experimental Contradictions.** A conflict that arises when two
experiments, each individually well-formed and internally consistent,
produce incompatible outcomes when they were intended to be comparable —
for example, two runs of what was believed to be the same method,
executed under what was believed to be the same protocol, producing
results that diverge beyond what the experiments' own reported variance
would predict. Experimental contradictions frequently indicate that the
two experiments were less comparable than assumed.

**Statistical Contradictions.** A conflict between what raw evidence
values appear to show and what a statistically rigorous treatment of
that evidence supports — for example, a reported difference between two
methods that appears consistent across evidence items but does not
exceed the variation expected from noise given the available sample, or
conversely, a claim of "no difference" that is not actually supported
because the evidence available lacks the statistical power to detect a
difference of the claimed magnitude. Statistical contradictions are
distinguished from evidence contradictions in that the individual
evidence items may all be accurate; the contradiction lies in the
inferential step from evidence to claim.

**Reproducibility Contradictions.** A conflict between a claim's
implicit or explicit assertion of reproducibility and evidence showing
that repeated, faithful execution under matched conditions fails to
regenerate the claimed result. A reproducibility contradiction is
distinct from ordinary run-to-run variance: it is present when the
divergence between attempts exceeds what any documented source of
legitimate non-determinism can account for.

**Scope Contradictions.** A conflict that, on investigation, is found to
originate in an unstated or mismatched scope rather than in a genuine
incompatibility, as discussed in Section 2. Scope contradictions are
retained as a named category, rather than being dismissed once resolved,
because the finding that a claim's true scope is narrower than its
stated scope is itself scientifically significant and must be recorded,
not discarded once the apparent conflict is explained away.

**Methodological Contradictions.** A conflict between the conclusions
supported by two evidence-gathering procedures that differ in rigor or
correctness — for example, one experiment that used a held-out test set
and one that did not, or one comparison that controlled for compute
budget and one that did not. A methodological contradiction is resolved,
in most cases, not by determining which outcome is "true" in some
absolute sense but by determining which procedure was methodologically
sound and discounting the evidence produced by the procedure that was
not, per the fairness-of-comparison principle established in the
philosophy chapter (Chapter 0).

These categories are not mutually exclusive. A single detected
contradiction is frequently, on investigation, found to belong to more
than one category simultaneously — a reproducibility contradiction that
turns out, on inspection, to also be a methodological contradiction
because the two attempted reproductions used different evaluation
procedures. This methodology does not require forcing a contradiction
into exactly one category; it requires that every category applicable to
a given contradiction be recorded.

--------------------------------------------------

# 4. Sources of Contradictions

Contradictions do not arise without cause. Identifying the source of a
contradiction is a distinct step from classifying its category (Section
3), and is a necessary input to the investigation stage of the
contradiction lifecycle (Section 5). This methodology recognizes the
following recurring sources.

**Conflicting experiments.** Two experiments, each executed and recorded
correctly on their own terms, produce results that disagree because they
were never as comparable as their surface similarity suggested. This is
among the most common sources of apparent contradiction in machine
learning research, precisely because two experiments can share a method
name, a dataset name, and a reported metric while differing in every
other respect that matters.

**Different datasets.** Evidence gathered from nominally the same
dataset can differ due to different versions, different splits,
different preprocessing, or different filtering applied before the data
reached the experiment. A contradiction sourced here is frequently
mistaken for a contradiction about the method under study, when it is in
fact a contradiction about which data the method was actually evaluated
on.

**Different environments.** Hardware, software library versions,
numerical precision settings, and driver versions can each independently
alter results, sometimes by margins large enough to reverse a reported
comparison. Evidence produced in different environments is a frequent,
and frequently unacknowledged, source of contradictions that appear to
be about a method when they are in fact about the infrastructure the
method was run on.

**Different implementations.** Two implementations that are believed to
realize the same method can differ in ways that are not evident from
their shared name — differences in initialization, in numerical
stability handling, in the precise definition of a hyperparameter, or in
undocumented default settings. A contradiction sourced to implementation
differences is a contradiction about two distinct artifacts that share a
label, not about a single method evaluated twice.

**Poor baselines.** A comparative claim's apparent contradiction with
another comparative claim can originate entirely in the choice of
baseline rather than in the method under study. A method that appears to
outperform a weak baseline and appears to underperform a strong baseline
is not contradicting itself; the baseline, not the method, is the
variable that changed between the two claims.

**Missing controls.** When an experiment does not hold constant the
factors that a fair comparison requires it to hold constant, any
resulting evidence is confounded, and contradictions with evidence from
better-controlled experiments are expected rather than surprising. A
missing control is frequently discovered only during the investigation
stage of a contradiction (Section 5), after the contradiction has
already been detected.

**Data leakage.** Evidence produced under conditions where information
from an evaluation set has influenced training, directly or indirectly,
will systematically disagree with evidence produced under conditions
free of that leakage. Contradictions sourced to data leakage are
particularly important to detect, since the leaked evidence is typically
the more favorable-looking of the two and is therefore the one most
likely to be preferred in the absence of a disciplined contradiction
process.

**Evaluation mismatch.** Two claims about "the same" metric can
disagree because the metric was computed differently — a different
averaging method, a different definition of a positive class, a
different treatment of edge cases, or a different evaluation harness
entirely. An evaluation mismatch produces evidence that is individually
accurate on its own terms while being fundamentally not comparable to
the evidence it is being placed in contradiction with.

Identifying which of these sources underlies a given contradiction is
what allows the contradiction to move from the detected stage to the
investigated stage of the lifecycle defined in the next section. A
contradiction whose source has not been identified cannot yet be
considered explained, regardless of how confidently a resolution is
proposed for it.

--------------------------------------------------

# 5. Contradiction Lifecycle

Like evidence (Chapter 1, Section 5) and claims (Chapter 2, Section 3),
a contradiction passes through a defined sequence of stages. A
contradiction's position in this sequence determines what may
legitimately be said about it and what work remains outstanding before
it can be treated as closed.

Detected
↓
Investigated
↓
Explained
↓
Resolved
or
Persisting

**Detected.** The contradiction has been identified as meeting the
definition in Section 2 — two or more evidence items or claims that
cannot both hold under the same scope, conditions, and terms. At this
stage, the contradiction is recorded but not yet understood. No
conclusion about which side, if either, is correct may be drawn from a
contradiction that has only reached the detected stage.

**Investigated.** The sources described in Section 4 have been checked
systematically against the specific evidence or claims in conflict, to
determine whether the conflict is genuine or is instead an unstated
scope difference (Section 2). Investigation is where the category or
categories from Section 3 are assigned, and where the specific source or
sources from Section 4 are identified as candidates for explaining the
conflict.

**Explained.** A specific, evidence-backed account has been established
for why the contradiction exists — which source or sources from Section
4 produced it, and through what mechanism. An explanation must itself be
traceable to evidence in the same way any other claim in this
methodology must be (Chapter 2, Section 9); an explanation that is merely
plausible but not verified against the actual conditions of the
conflicting evidence has not reached this stage.

**Resolved.** Following explanation, the contradiction is resolved when
the explanation allows a definitive determination: the conflict was a
scope difference now made explicit, one side was found to rest on
confounded or invalid evidence and is discounted accordingly, or both
sides are found to hold under correctly stated, non-overlapping scopes.
A resolved contradiction is retained in the record together with its
explanation; resolution updates how the contradiction is reported, but
it does not delete the contradiction from the evidentiary history.

**Persisting.** Not every contradiction can be explained with the
evidence available. A contradiction is recorded as persisting when
investigation has been genuinely attempted, per the standard set out
above, but no explanation sufficient to reach resolution has been
established. A persisting contradiction is not a failure of the
methodology; it is an accurate report of the current state of
knowledge, and it must be carried forward into any claim, confidence
assessment, or judgment that depends on the evidence in conflict, rather
than being quietly dropped once investigation has been attempted and
found inconclusive.

A contradiction may re-enter this lifecycle if new evidence becomes
available after it has reached resolved or persisting status. Resolution
and persistence are both provisional with respect to future evidence,
in the same way any conclusion in this methodology is provisional with
respect to evidence that does not yet exist.

--------------------------------------------------

# 6. Contradiction Relationships

A contradiction does not exist in isolation. It stands in a defined
relationship to several other constructs in this methodology, and each
of those relationships constrains what may be inferred once a
contradiction is present.

**Evidence.** A contradiction is, at minimum, a relationship between
evidence items (Section 2). Every contradiction must identify the
specific evidence items in conflict, per the properties of good evidence
established in Chapter 1 — traceable, source-attributed, time-aware.
A contradiction that cannot name its constituent evidence items has not
met the definition given in Section 2.

**Claims.** A contradiction between evidence items propagates to any
claim that evidence supports (Chapter 2, Section 7). A claim whose
supporting evidence includes an unresolved contradiction cannot be
treated, for purposes of confidence or judgment, as though its
evidentiary basis were clean; Chapter 2, Section 8 records this
explicitly as one of the conditions under which a claim lacks adequate
support — conflicting evidence.

**Confidence.** The presence of an unresolved contradiction is a direct
input to confidence assessment (to be defined in a later chapter of this
methodology). A body of evidence containing a persisting contradiction
cannot be assigned the same confidence as an equivalent body of evidence
free of contradiction, regardless of how much of the remaining evidence
agrees. Confidence reflects the true state of the evidence, per the
philosophy established in Chapter 0, and an unresolved contradiction is
part of that true state.

**Judgment.** A judgment formed over a claim whose evidence contains a
persisting contradiction must state that contradiction as part of the
judgment, not merely factor it into a numeric confidence figure. A
judgment that summarizes evidence favorably while omitting a known,
unresolved contradiction has violated the requirement, established in
Chapter 0, that disagreement be surfaced rather than smoothed over.

**Recommendations.** A recommendation that depends on a claim whose
underlying evidence contains a persisting contradiction inherits the
same obligation: the recommendation must be qualified by the
contradiction's existence, and the degree to which the recommendation
should be trusted must reflect that qualification. A recommendation is
the furthest point in the reasoning chain from raw evidence, and is
therefore the point at which an unacknowledged contradiction does the
most damage if it has been lost along the way.

These relationships establish that a contradiction, once detected, is
not a local fact confined to the two evidence items or claims in direct
conflict. It is a property that must be carried forward through every
later stage of reasoning that depends, even indirectly, on the evidence
or claims involved.

--------------------------------------------------

# 7. Handling Contradictions

The following principles govern how a detected contradiction must be
treated throughout its lifecycle. They are binding requirements of this
methodology, not optional best practices.

**Never ignore contradictory evidence.** A contradiction, once detected
under the definition in Section 2, must be recorded and carried forward.
It may not be treated as a data artifact to be cleaned away, nor may it
be silently excluded from a summary because it complicates an otherwise
clean-looking result.

**Prefer investigation over dismissal.** A contradiction may not be
dismissed as noise, as an outlier, or as "probably a mistake" without
the investigation described in Section 5 actually being carried out.
Dismissal without investigation is indistinguishable, from the outside,
from suppression of inconvenient evidence, and this methodology does not
permit the two to be conflated by treating an uninvestigated dismissal
as though it were an investigated resolution.

**Document uncertainty.** Where investigation does not reach an
explanation sufficient for resolution, the resulting uncertainty must be
stated explicitly, using the persisting status defined in Section 5,
rather than being rounded down to a false certainty in either direction.

**Preserve conflicting evidence.** Per the immutability property
established for evidence generally (Chapter 1, Section 3), evidence
found to be in contradiction with other evidence is not deleted,
suppressed, or excluded from the record once a resolution is reached.
Even where one side of a contradiction is ultimately discounted, the
discounted evidence remains part of the historical record, together with
the explanation for why it was discounted, so that the resolution itself
remains checkable.

**Avoid premature conclusions.** No claim, confidence assessment, or
judgment may treat a contradiction as resolved before it has completed
the investigated and explained stages defined in Section 5. A
contradiction that has only reached the detected stage must not be
quietly assumed to favor one side merely because that side seems, on
first inspection, more plausible or more consistent with other available
evidence.

Together, these principles express a single underlying commitment: a
contradiction is worth more to this methodology when it is visible and
partially understood than when it has been made to disappear, however
inconvenient its visibility may be to the claim it complicates.

--------------------------------------------------

# 8. Contradiction Quality

Not every contradiction carries equal scientific weight. Just as
evidence quality is assessed along defined dimensions (Chapter 1,
Section 7) rather than treated as a binary present-or-absent property,
this methodology assesses contradiction quality along the following
dimensions.

**Severity.** The degree to which the contradiction, if left
unresolved, undermines the specific claim or claims it touches. A
contradiction between two evidence items that differ by a margin well
within ordinary measurement noise is lower severity than a contradiction
between evidence items whose disagreement directly reverses a
comparative claim's conclusion.

**Breadth.** The extent of the methodology's reasoning that the
contradiction affects — whether it touches a single, narrowly scoped
claim or propagates, through the relationships described in Section 6,
into multiple claims, confidence assessments, or recommendations. A
high-breadth contradiction warrants more prominent surfacing than one
confined to a single, low-stakes claim.

**Repeatability.** Whether the contradiction has been observed once or
has recurred across independent instances of evidence gathering. A
contradiction observed a single time is weaker grounds for concern than
the same apparent conflict recurring across repeated, independent
attempts to gather comparable evidence.

**Reliability.** The quality, per Chapter 1 Section 7, of the evidence
items on each side of the contradiction. A contradiction between one
high-quality, well-attributed evidence item and one poorly attributed,
low-provenance item is not symmetric, and this methodology does not
treat it as though both sides carried equal evidentiary weight merely
because a conflict exists between them.

**Independence.** Whether the evidence items or claims in conflict were
produced by genuinely independent processes — different runs, different
environments, different evaluators — or share a common origin that could
itself explain apparent disagreement as a shared artifact rather than a
true conflict. A contradiction between two evidence items sharing an
unacknowledged common source is weaker evidence of genuine disagreement
than a contradiction between fully independent sources.

**Scope.** Whether the contradiction, once investigated, remains a
genuine conflict under matched scope (Section 2) or is better
characterized as a scope difference. A contradiction's scope
determination is not merely a category label (Section 3); it is itself a
quality dimension, since a contradiction that collapses into a scope
difference upon investigation carries less unresolved evidentiary weight
than one that survives scope scrutiny intact.

This methodology requires that contradiction quality be assessed along
each of these dimensions before a contradiction is weighed into any
downstream confidence assessment or judgment. Treating all contradictions
as interchangeable — a single flag that is either present or absent —
would discard information that is directly relevant to how much a given
contradiction should affect the conclusions built on top of it.

--------------------------------------------------

# 9. Traceability

Every contradiction admitted under this methodology must trace back to
three things:

**Evidence.** The specific evidence items in conflict, identified with
the same provenance requirements that apply to evidence generally
(Chapter 1, Section 9) — source, timestamp, origin, and the other fields
that make an evidence item independently checkable.

**Claim.** The specific claim or claims the contradiction bears on, per
the relationships established in Section 6. A contradiction that cannot
be connected to at least one claim it affects is incomplete under this
methodology, even if it is fully documented at the evidence level, since
the ultimate purpose of tracking contradictions is to protect the
integrity of claims built on evidence.

**Scientific Rule.** The specific principle of correct scientific
reasoning — fairness of comparison, statistical rigor, reproducibility
requirements, or another principle established elsewhere in this
methodology — that the contradiction implicates or tests. A
contradiction sourced to a poor baseline, for example, traces to the
fairness-of-comparison principle; a contradiction sourced to
insufficient sample size traces to the statistical rigor principle
governing confidence in comparative claims.

This traceability requirement exists for the same reason traceability is
required of evidence and claims generally: it is what allows a
contradiction to be independently re-examined, rather than accepted or
dismissed on the reputation of whoever reported it. A contradiction that
cannot be traced to specific evidence, a specific claim, and a specific
scientific rule is not yet a contradiction this methodology can act on;
it is, at best, an unverified suspicion of one.

--------------------------------------------------

# 10. Future Integration

This document defines contradictions independently of the components of
Experiment Audit that will detect, investigate, and act on them. Those
components are described elsewhere in this methodology or remain future
work; this section notes only their conceptual dependency on the
definitions above, without describing how any of them will be built.

**Scientific Rule Engine.** A future component responsible for applying
the principles of correct scientific reasoning that Section 9 requires
every contradiction to trace back to. The Scientific Rule Engine is the
component expected to determine, in the investigated stage of the
lifecycle (Section 5), which principle a given contradiction implicates
and whether that principle has in fact been violated.

**Claim Verification.** A future component responsible for connecting
claims to their supporting evidence (Chapter 2). Claim Verification
depends on contradiction detection to correctly identify claims whose
evidentiary basis includes conflicting evidence (Chapter 2, Section 8),
since a claim cannot be accurately verified without knowing whether its
evidence disagrees with itself.

**Confidence Assessment.** A future component responsible for producing
a calibrated estimate of how strongly evidence supports a claim.
Confidence Assessment depends on the contradiction quality dimensions
defined in Section 8 to determine how much a given unresolved
contradiction should reduce confidence in the claims it touches, and
depends on the lifecycle status defined in Section 5 to distinguish
claims affected by persisting contradictions from claims where an
apparent contradiction has been fully resolved.

**Judgment Generation.** A future component responsible for producing a
scientific judgment about a claim from its verified evidentiary status
and assessed confidence. Judgment Generation depends on the relationship
between contradictions and judgments established in Section 6: a
judgment may not present a claim as more settled than its contradiction
record supports.

**Recommendation Engine.** A future component responsible for proposing
actions based on judgments. The Recommendation Engine depends on
contradictions being carried forward, per Section 6, through every
intermediate stage between raw evidence and a proposed action, so that a
recommendation never rests on a claim whose unresolved contradictions
have been lost along the way.

No component in this list is permitted to treat a contradiction as
resolved without the investigation and explanation stages defined in
Section 5 having actually been completed, and none is permitted to
discard a persisting contradiction from its output. That constraint is
the reason this document treats contradictions as a first-class subject
of the methodology, consistent with the philosophy established in
Chapter 0: contradictions are a first-class output, not an error state.
