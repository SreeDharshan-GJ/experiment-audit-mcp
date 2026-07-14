# 1. Purpose

Scientific reasoning about a machine learning experiment is only as sound as
the material it is reasoning over. An observation about a training curve, a
hypothesis about why one configuration outperforms another, a confidence
estimate attached to a comparison, a judgment that a result is or is not
reproducible, a recommendation to adopt or discard a method — every one of
these outputs is a claim, and every claim in Experiment Audit must trace back
to something that was actually produced during the life of an experiment,
not to an impression, a memory, or an assumption.

This document defines that foundational material. It calls it evidence, and
it treats evidence as the base layer on which all later layers of the
methodology are built. If the evidence layer is ambiguous — if it is unclear
what counts as evidence, what evidence is made of, or how strong a given
piece of evidence is — then everything built above it inherits that
ambiguity. A hypothesis derived from poorly specified evidence is itself
poorly specified. A confidence score computed over undifferentiated evidence
cannot be defended. A recommendation issued without traceable evidentiary
support is not a scientific conclusion; it is an opinion wearing the shape
of one.

Experiment Audit exists to prevent that substitution. Its central premise is
that reasoning about experiments should never be permitted to produce a
conclusion that cannot be walked back, step by step, to the concrete
artifacts that justify it. Evidence is where that walk-back terminates. It
is therefore the first subject this methodology must define, and every
later chapter — observation, claims, reproducibility, baselines,
statistics, confidence, contradictions, and decisions — is written on the
assumption that evidence has already been defined rigorously and will not
be redefined implicitly by any downstream component.

--------------------------------------------------

# 2. Definition of Evidence

**Evidence** is any verifiable artifact produced during the lifecycle of an
experiment that can support, weaken, or contradict a scientific claim about
that experiment or about a comparison between experiments.

Three properties are load-bearing in this definition and must all hold for
something to qualify as evidence:

- **Verifiable.** The artifact must be checkable against its source. A
  number that cannot be traced back to where it was produced is not
  evidence; it is an assertion.
- **Produced during the experiment lifecycle.** Evidence originates from
  configuration, execution, logging, evaluation, or archival activity that
  actually occurred. It is not synthesized after the fact to fit a
  narrative, and it is not inferred from what an experiment plausibly would
  have produced.
- **Bearing on a claim.** An artifact only functions as evidence in
  relation to some claim it supports, weakens, or contradicts. The same
  underlying record can be evidence for one claim and irrelevant to
  another; evidentiary status is relational, not an intrinsic property of
  the artifact alone.

Evidence must be distinguished from three neighboring concepts with which
it is frequently and consequentially confused:

**Evidence vs. Observation.** An observation is a statement made by
inspecting evidence — for example, noting that a loss curve trends upward
after step 40,000. The observation is downstream of the evidence: it is an
interpretation of what the evidence shows, expressed in terms a reasoning
process can act on. The evidence is the recorded loss values themselves;
the observation is the claim that they trend upward. Two independent
observers can produce different observations from identical evidence,
which is precisely why the evidence, not the observation, must be the
retained, citable artifact.

**Evidence vs. Hypothesis.** A hypothesis is a proposed explanation for a
pattern in the evidence — for example, that the upward trend in loss is
caused by a learning rate that is too high for the batch size in use. A
hypothesis is answerable to evidence but is not itself evidence. It can be
supported or weakened by evidence that already exists, and it can motivate
the collection of further evidence, but it cannot substitute for the
evidence it seeks to explain.

**Evidence vs. Conclusion.** A conclusion is a terminal scientific judgment
— that method A outperforms method B under specified conditions, that a
result is reproducible, that a claimed effect is not statistically
distinguishable from noise. A conclusion is what evidence, observations,
and hypotheses are eventually assembled into; it sits at the far end of the
reasoning chain, not at its origin. A conclusion that cannot be decomposed
back into the evidence supporting it has left the boundaries of this
methodology, regardless of how confidently it is stated.

Evidence, observation, hypothesis, and conclusion form an ordered chain.
Experiment Audit treats collapsing any two of these into one as a
methodological error, whether the collapse happens by carelessness or by
convenience, because it destroys the traceability that is the entire
purpose of maintaining the chain.

--------------------------------------------------

# 3. Properties of Good Evidence

Not all verifiable artifacts are equally useful as evidence. The following
properties describe what distinguishes evidence that can bear real
scientific weight from evidence that is technically admissible but
practically weak.

**Verifiable.** Good evidence can be checked by someone other than the
party that produced it, against the original source rather than a
paraphrase of it. Verifiability is what separates evidence from testimony:
testimony asks to be believed, evidence asks to be checked.

**Traceable.** Good evidence carries an unbroken path back to its point of
origin — which run produced it, at what step, under what configuration.
Evidence that has been separated from its origin, even if the value itself
is correct, has lost the property that makes it usable in an audit: the
ability to confirm where it came from.

**Immutable once recorded.** Once an evidence item has been recorded, it
must not be altered in place. Experiments are frequently re-run, corrected,
or extended, and the temptation to update an existing record rather than
create a new one is strong — but doing so destroys the historical record
that later comparisons depend on. Correction is handled by recording new
evidence and explicitly relating it to the old, never by silent
overwriting.

**Reproducible.** Where the nature of the artifact allows it, good evidence
is evidence that a second, faithful execution of the same experiment would
be expected to regenerate, within the bounds of any legitimate
non-determinism the experiment is exposed to. Evidence that is a one-time,
unrepeatable accident of a specific execution environment is weaker than
evidence that reflects the underlying method.

**Contextual.** Good evidence is recorded together with the conditions
that produced it — configuration, environment, data version — rather than
as a bare number stripped of the setting that gives it meaning. A metric
value without its context can be misapplied to a comparison it was never
valid for.

**Time-aware.** Good evidence records when it was produced, both because
experiments and codebases change over time and because the same nominal
metric can mean different things at different points in a project's
history. Evidence without a timestamp cannot be correctly ordered relative
to other evidence, which makes lifecycle and provenance reasoning
(Sections 5 and 9) impossible.

**Source-attributed.** Good evidence identifies the specific system,
process, or instrument that produced it — a particular logging backend, a
particular evaluation run, a particular environment snapshot. Attribution
is what allows evidence from different sources to be weighed differently
when sources are known to differ in reliability.

**Comparable.** Good evidence is recorded in a form that permits like-for-
like comparison with other evidence of the same category — the same
metric name and units, the same evaluation protocol, the same measurement
window. Evidence that cannot be aligned with comparable evidence from
another run cannot support comparative claims, which are among the most
common claims this methodology exists to evaluate.

**Independent when possible.** Good evidence, where obtainable, comes from
sources that did not influence one another's production — for example, an
evaluation metric computed by a process separate from the training loop
that optimized against a different objective. Independence matters because
evidence that shares a common origin with the claim it is meant to test can
create the appearance of confirmation without providing it.

--------------------------------------------------

# 4. Categories of Evidence

Experiment Audit organizes evidence into the following categories. These
categories describe what kind of thing an evidence item is, not how strong
it is or what conclusions may be drawn from it — strength and inference are
addressed elsewhere in this methodology.

**Configuration Evidence.** The declared settings under which an
experiment was run: hyperparameters, architecture choices, random seeds,
data splits, and any other parameter fixed before execution began.
Configuration evidence establishes what was intended to happen and is the
baseline against which everything the experiment actually produced is
checked for consistency.

**Metric Evidence.** Quantitative measurements recorded during or after
execution — loss values, reward values, accuracy, or any other tracked
quantity, together with the step or time index at which each value was
recorded. Metric evidence is typically the highest-volume category and the
one most directly consumed by observation-level reasoning.

**Training Evidence.** Records of what occurred over the course of
optimization itself: the trajectory of tracked quantities across training,
checkpoint history, and any recorded interruptions, resumptions, or
anomalies in the training process. Training evidence differs from metric
evidence in scope — it concerns the process across its full duration
rather than any single measurement.

**Evaluation Evidence.** Records produced by assessing a trained model
against a held-out protocol distinct from the training process — the
dataset, split, and procedure used, and the results that procedure
produced. Evaluation evidence is the primary basis for claims about a
model's performance rather than its optimization behavior.

**Comparison Evidence.** Records that directly juxtapose two or more runs,
configurations, or methods under conditions intended to be held equal
except for the factor under study. Comparison evidence is what makes
claims of the form "A outperforms B" answerable at all, and its validity
depends heavily on how faithfully the non-varying conditions were actually
held equal.

**Baseline Evidence.** Records of the reference points against which a
new method is being evaluated — prior methods, ablated variants, or naive
controls, run under conditions matched to the method being studied.
Baseline evidence exists specifically to prevent claims of improvement
from being evaluated in isolation, against no defined reference.

**Statistical Evidence.** Records that describe the distribution of a
result across repeated measurement — multiple seeds, multiple runs, or
multiple evaluation passes — rather than a single point estimate.
Statistical evidence is what allows a distinction to be drawn between an
effect and noise, a distinction that no single run, however carefully
executed, can support on its own.

**Reproducibility Evidence.** Records of whether, and how closely, a
result was regenerated by an independent execution of the same nominal
experiment — by the same party at a different time, or by a different
party altogether. Reproducibility evidence speaks to the durability of a
claim rather than its initial measurement.

**Environmental Evidence.** Records of the execution context outside the
experiment's own configuration: software versions, hardware, dependency
versions, and any other environmental factor capable of affecting results
without appearing in configuration evidence. Environmental evidence exists
because a meaningful fraction of irreproducibility traces to environment
drift rather than configuration drift.

**Operational Evidence.** Records of what actually happened to the
experiment as a running process — start and end times, failures,
restarts, resource allocation, and manual interventions. Operational
evidence is often the difference between a result that looks anomalous
because of the underlying method and one that looks anomalous because its
execution was disrupted.

These categories are not mutually exclusive in the sense of being sealed
off from one another; a single recorded run typically contributes evidence
to several categories at once. What the categorization provides is a
consistent vocabulary for referring to what kind of evidentiary support a
given claim rests on, and for noticing when a claim rests on a category of
evidence it should not — for example, a reproducibility claim resting only
on metric evidence, with no reproducibility evidence of its own.

--------------------------------------------------

# 5. Evidence Lifecycle

An evidence item does not appear fully formed; it passes through a defined
sequence of stages, each of which changes what can legitimately be done
with it.

**Produced.** The stage at which the underlying fact comes into existence
— a metric is computed, a configuration is fixed, an environment is
instantiated. At this stage the evidence exists in the world but has not
yet been captured by the methodology.

**Recorded.** The stage at which the produced fact is captured in a
durable, retrievable form together with its immediate context. An
unrecorded fact, however real, cannot function as evidence within this
methodology; recording is the boundary between something that happened and
something that can be reasoned about.

**Validated.** The stage at which a recorded item is checked for internal
consistency and plausibility — that required fields are present, that
values fall within physically or logically sensible ranges, that the
record is not self-contradictory. Validation does not assess whether the
evidence supports any particular claim; it assesses only whether the
evidence is well-formed enough to be used at all.

**Linked.** The stage at which a validated item is connected to the other
evidence it relates to — the run it belongs to, the comparison it is part
of, the baseline it is measured against. An evidence item that has not
been linked is usable in isolation but cannot yet participate in the
relational reasoning described in Section 6.

**Referenced.** The stage at which linked evidence is actually drawn upon
by an observation, hypothesis, claim, or later evidence item. Reference is
what activates evidence — it is the point at which the evidence stops
being a passive record and starts doing evidentiary work.

**Archived.** The stage at which evidence is retained for long-term
availability after its immediate use has concluded. Archived evidence
remains fully valid and citable; archival changes where and how the
evidence is stored, not its evidentiary status. Evidence is not discarded
when a project moves on from it, because later claims — including claims
made much further downstream, such as reproducibility claims made against
a much earlier baseline — depend on archived evidence remaining available
and unaltered.

An evidence item is expected to reach every one of these stages in order.
A gap at any stage — evidence that was produced but never recorded,
recorded but never validated, validated but never linked — is itself
methodologically significant and is addressed directly in Section 8.

--------------------------------------------------

# 6. Evidence Relationships

Evidence items rarely stand alone. Experiment Audit recognizes the
following relationships between evidence items, and treats the relational
structure among evidence as being as important as the evidence itself.

**Support.** One evidence item supports a claim when it is consistent with
that claim and increases the warrant for accepting it. Support is the most
common relationship and the one most often assumed by default — which is
precisely why it must be stated explicitly rather than left implicit.

**Contradiction.** One evidence item contradicts a claim when it is
inconsistent with that claim and decreases the warrant for accepting it.
Contradiction is not a defect to be resolved by discarding the
inconvenient evidence; a documented contradiction is itself a piece of
methodological information, and its handling is addressed in a later
chapter of this methodology.

**Strengthening.** One evidence item strengthens another when its
presence increases the weight the other evidence item can bear — for
example, reproducibility evidence strengthening the metric evidence it
was generated to confirm. Strengthening relationships are what allow a
body of evidence to carry more weight collectively than any single item
within it carries alone.

**Dependency.** One evidence item depends on another when the second
evidence item is a precondition for interpreting or trusting the first —
for example, metric evidence depending on the configuration evidence that
establishes what was measured and how. A dependent evidence item cannot be
fully evaluated in isolation from what it depends on.

**Derivation.** One evidence item is derived from another when it is
produced by a defined transformation applied to existing evidence rather
than by direct observation of the experiment itself — for example, a
comparison record derived from two independent runs' evaluation evidence.
Derived evidence inherits limitations present in the evidence it is
derived from, and this inheritance must remain visible rather than being
absorbed into the derived item as if it were independently established.

**Independence.** Two evidence items are independent when neither their
production nor their content influenced the other. Independence is a
relationship in the sense that it must be actively established, not
assumed as a default; two evidence items with no documented dependency are
not automatically independent, they are simply undocumented.

These relationships matter because the strength of a scientific claim is
rarely determined by any one piece of evidence but by the structure
connecting many. A claim resting on evidence that is heavily
interdependent — several items all deriving from the same underlying
measurement — carries less real support than the raw count of supporting
items would suggest. Making relationships explicit is what allows later
stages of this methodology to distinguish genuinely broad evidentiary
support from evidence that merely appears broad because it has not been
traced back to a shared origin.

--------------------------------------------------

# 7. Evidence Quality

Evidence quality is evaluated along the following dimensions. These
dimensions are independent of one another and of evidence quantity, and a
body of evidence should be assessed on all of them rather than on any
single one in isolation.

**Completeness.** The degree to which the evidence available for a claim
covers everything the claim requires — for example, whether a comparison
claim has evidence for every condition being compared, not just the
conditions that happened to succeed. Incomplete evidence is not
disqualifying, but it must be recognized as incomplete rather than treated
as if it were sufficient.

**Consistency.** The degree to which evidence items bearing on the same
claim agree with one another. Low consistency does not by itself indicate
that the underlying claim is false; it indicates that the evidence base
requires closer examination before the claim can be assessed with
confidence.

**Freshness.** The degree to which evidence reflects the current state of
the code, data, and environment it purports to describe, rather than a
state that has since changed. Evidence can be perfectly well-formed and
still be stale, and stale evidence applied to a present-tense claim is a
common source of unwarranted confidence.

**Reliability.** The degree to which the source that produced an evidence
item is known, from its track record, to produce accurate records.
Reliability is a property of the source, assessed independently of any
particular claim the evidence is being used to support.

**Coverage.** The degree to which evidence spans the conditions relevant
to a claim — different seeds, different data subsets, different
environments — rather than concentrating on a narrow slice of the
conditions under which the claim is meant to hold. Coverage is related to
completeness but concerns breadth across conditions specifically, rather
than presence or absence of required fields.

**Repeatability.** The degree to which the process that produced the
evidence could, in principle, be executed again to produce comparable
evidence. Repeatability is a property of the evidence-producing process,
distinct from reproducibility evidence itself, which records whether that
repetition was actually attempted and what it found.

**Integrity.** The degree to which an evidence item can be confirmed to be
unaltered since it was recorded. Integrity is what makes the immutability
property described in Section 3 verifiable after the fact, rather than
merely asserted at the time of recording.

Evidence quality and evidence quantity are distinct axes, and this
methodology treats them as such. A larger volume of evidence does not
automatically constitute stronger evidence: ten metric readings drawn from
a single run under a single configuration are not stronger support for a
general claim than two readings drawn from independently configured runs,
regardless of which set is larger. Any procedure that aggregates evidence
into a strength or confidence estimate must account for quality along the
dimensions above, not merely for count.

--------------------------------------------------

# 8. Missing Evidence

The absence of evidence is not the same as the absence of information.
When evidence that a claim would require is not present, that absence is
itself a fact about the claim's evidentiary status, and this methodology
requires it to be represented explicitly rather than passed over in
silence.

Common forms of missing evidence include:

- **Missing baselines** — a comparative or improvement claim made without
  baseline evidence to compare against.
- **Missing seeds** — a claim about a method's typical behavior made from
  a single run, without statistical evidence across repeated
  initialization.
- **Missing evaluation metrics** — a performance claim made without
  evaluation evidence covering the dimension the claim concerns.
- **Missing statistical analysis** — a claim of difference between two
  results made without evidence addressing whether that difference
  exceeds the variation expected by chance.
- **Missing environment details** — a reproducibility-relevant claim made
  without environmental evidence sufficient to know whether a repeat
  execution occurred under comparable conditions.

Explicit representation of missing evidence matters for a specific reason:
silence about an absence is easily, and often unintentionally, read as
confirmation that no gap exists. A claim accompanied by evidence for nine
of its ten required dimensions, with the tenth simply unaddressed, can be
mistaken for a fully supported claim unless the missing dimension is
actively flagged. Experiment Audit treats an unaddressed evidentiary gap
as a defect in the claim's support, not a neutral state, and requires that
gap to be surfaced with the same explicitness as the evidence that is
present.

--------------------------------------------------

# 9. Evidence Provenance

Every evidence item admitted under this methodology must carry the
following provenance fields:

**Source.** The specific system, backend, or process that produced the
evidence item, sufficient to distinguish it from other sources capable of
producing superficially similar records.

**Timestamp.** The point in time at which the underlying fact was
produced, distinct from any later time at which the evidence item was
recorded, validated, or referenced.

**Origin.** The specific experiment, run, or process instance the
evidence item originates from, sufficient to trace the item back to a
single point of production rather than a class of similar productions.

**Experiment.** The identified experiment the evidence item is associated
with, connecting the item to the broader scientific question it was
produced to inform, not merely to the mechanical process that generated
it.

**Version.** The version of the code, configuration schema, or protocol
in effect at the time the evidence was produced, since the meaning of an
otherwise identical record can shift as versions change.

**Collection Method.** The means by which the evidence item was captured
— which mechanism, at what point in the pipeline, under what triggering
condition — distinguished from the evidence's content.

Provenance is essential to auditing for a reason distinct from the
properties discussed in Section 3: it is what allows an evidence item to
be independently re-examined by a party other than the one that produced
it. An auditor without provenance information can inspect the value an
evidence item carries but cannot check where that value came from, when
it was produced, under what version of the system, or by what process —
which means the auditor cannot distinguish a sound piece of evidence from
one that merely resembles it. Provenance is therefore not supplementary
metadata attached to evidence; it is part of what makes something count
as evidence under the definition in Section 2, since verifiability itself
depends on provenance being present and checkable.

--------------------------------------------------

# 10. Future Integration

This document defines evidence in isolation from the components of
Experiment Audit that will consume it. Those components are described
elsewhere in this methodology; this section notes only their conceptual
dependency on the definitions above, without describing how any of them
will be built.

**Observation Layer.** Consumes evidence to produce statements about what
the evidence shows, as distinguished in Section 2. The observation layer
depends on evidence being categorized (Section 4) and quality-assessed
(Section 7) before observations are drawn from it.

**Hypothesis Layer.** Consumes observations, together with the evidence
underlying them, to propose explanations. The hypothesis layer depends on
evidence relationships (Section 6) to determine which evidence a given
hypothesis is answerable to.

**Scientific Rule Engine.** Consumes evidence, observations, and
hypotheses together to apply the rules of correct scientific reasoning
defined elsewhere in this methodology. This layer depends on the
evidence/observation/hypothesis/conclusion distinction in Section 2 being
strictly maintained, since its rules are defined in terms of that
distinction.

**Claim Verification.** Consumes evidence to check whether a stated claim
is actually supported by the evidence attributed to it, including whether
that evidence is complete (Section 7) and whether relevant evidence is
missing (Section 8).

**Confidence Assessment.** Consumes evidence quality (Section 7) and
evidence relationships (Section 6) to produce a calibrated estimate of how
strongly a body of evidence supports a given claim, distinct from a raw
count of supporting items.

**Judgment Generation.** Consumes verified claims and their associated
confidence to produce a scientific judgment, remaining traceable at all
times back through claim verification to the underlying evidence.

**Recommendation Engine.** Consumes judgments to propose actions, and
depends on every layer beneath it having preserved the traceability this
document establishes as the requirement for evidence itself — a
recommendation is only as defensible as the evidentiary chain beneath the
judgment it acts on.

No component in this list is permitted to treat evidence as fungible with
observation, hypothesis, or conclusion, and none is permitted to originate
a claim that does not terminate, when traced backward, in evidence meeting
the definition given in Section 2. That constraint is the reason this
document exists as the first chapter of the methodology rather than as
supporting material for a later one.
