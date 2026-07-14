# ML Scientific Reasoning Standard: Philosophy and Design Principles

Status: Draft
Applies to: Experiment Audit

---

## 1. Purpose

Machine learning experimentation produces large volumes of artifacts: metrics,
logs, configurations, checkpoints, training curves, and ad-hoc notes. In
practice, the conclusions drawn from these artifacts are often informal.
A researcher looks at a results table, forms an impression, and writes a
sentence in a report or a paper. The reasoning that connects the evidence to
the conclusion frequently exists only in the researcher's head, and is lost
once the experiment is archived.

The ML Scientific Reasoning Standard exists to make that reasoning explicit,
inspectable, and repeatable. It defines how evidence from an ML experiment
should be collected, how that evidence supports or fails to support a claim,
and how the resulting conclusion should be presented so that another person
can verify it without repeating the entire investigation from scratch.

Experiment Audit is the project that implements this standard. Its purpose is
to provide transparent, repeatable, evidence-based evaluation of machine
learning experiments.

Experiment Audit is not intended to replace peer review or scientific
judgment. It does not decide whether a paper should be accepted, whether a
result is novel, or whether a research direction is worth pursuing. Those are
judgments that require human expertise, context, and accountability. What
Experiment Audit provides is a disciplined, traceable layer of evidence
evaluation that a human reviewer, author, or collaborator can use as an input
to their own judgment. It is a tool for organizing and checking evidence, not
a substitute for the people who are responsible for interpreting it.

---

## 2. Core Philosophy

The following principles describe the intellectual stance of the project.
They apply across every part of the standard and take precedence over any
individual feature or output format.

**Evidence before conclusions.** A conclusion is only as good as the evidence
behind it. The standard requires that evidence be identified and evaluated
before any conclusion is formed, not after a conclusion is chosen and then
justified retroactively.

**Reproducibility before novelty.** An experiment that cannot be reproduced
provides weak evidence, regardless of how interesting its result appears.
The standard treats reproducibility as a precondition for trusting a result,
not as a secondary concern to be addressed once a result looks promising.

**Transparency over opaque scoring.** A single number that summarizes an
experiment's quality is not useful on its own if the reasoning behind it is
hidden. The standard favors explanations that show what was checked, what was
found, and why it matters, over scores that cannot be traced back to their
source.

**Scientific claims must be traceable.** Every claim made about an
experiment must be traceable to specific, identifiable evidence: a metric
value, a log entry, a configuration field, a comparison against a baseline.
A claim that cannot be traced to evidence is not a scientific claim within
this standard; it is an assertion.

**Confidence reflects evidence quality, not model confidence.** When the
standard reports a level of confidence in a conclusion, that confidence
describes the strength, completeness, and consistency of the underlying
evidence. It is not a measure of how certain a language model or any other
component "feels" about its output.

**Every conclusion must be explainable.** A conclusion that cannot be
explained in terms a domain expert would accept is not acceptable output,
regardless of how it was produced. Explainability is a requirement for every
stage of the process, not an optional add-on.

**Absence of evidence is reported, not hidden.** When information needed to
evaluate a claim is missing, the standard requires that this gap be stated
explicitly rather than silently filled in or ignored.

**Disagreement is preserved, not smoothed over.** When evidence points in
different directions, the standard requires that the contradiction be
surfaced, not averaged away into a single misleadingly confident answer.

**Consistency across evaluations.** The same evidence, evaluated under the
same standard, should lead to the same conclusion. Reasoning that depends on
arbitrary or unrepeatable factors undermines the purpose of the standard.

**Humility about scope.** The standard is explicit about what it can and
cannot assess. Overstating what a conclusion supports is treated as a defect,
even when the overstated version would be more impressive.

These principles are stated here as philosophy. They do not define
algorithms, scoring formulas, or implementation mechanics; those belong in
other documents within this standard.

---

## 3. Scope

The ML Scientific Reasoning Standard defines how to evaluate the evidentiary
basis of a machine learning experiment. It applies to the following areas:

- **Experiments.** The configuration, execution, and recorded outcome of a
  single training or evaluation run.
- **Ablations.** Comparisons that isolate the effect of a specific change by
  holding other factors constant.
- **Baselines.** The reference points against which a result is compared,
  and whether those reference points are fair and appropriate.
- **Statistical evidence.** Whether reported differences are supported by
  the data in a rigorous way, including consideration of variance, sample
  size, and repeated runs where available.
- **Reproducibility.** Whether an experiment's setup, environment, and
  procedure are documented well enough that the result could plausibly be
  reproduced.
- **Claim support.** Whether a stated conclusion about an experiment is
  actually supported by the evidence attached to it, and to what degree.

The standard explicitly does not evaluate:

- **Novelty.** Whether an idea, method, or result is new relative to prior
  work is a judgment that requires broad awareness of a field and is outside
  this standard's scope.
- **Research impact.** Whether a result matters to a field, a community, or
  an application is a judgment for humans with domain context.
- **Publication decisions.** Whether a paper should be accepted, rejected,
  or revised is a decision that belongs to editors, program committees, and
  reviewers.
- **Scientific creativity.** Whether a research direction or hypothesis is
  interesting or worth pursuing is not something this standard assesses.

The standard evaluates whether the evidence for a claim is sound. It does not
evaluate whether the claim, if true, would matter.

---

## 4. Guiding Principles

The following principles elaborate on the core philosophy with more concrete
guidance for anyone applying or extending the standard. Each principle is
described in terms of what it means and why it matters; none of them
prescribe a specific algorithm or formula.

**1. Every evidence item has a source.**
*Explanation:* Any piece of evidence used in reasoning — a metric, a log
line, a configuration value — must be attributable to a specific origin
(a file, a run, a timestamp, a system).
*Rationale:* Without a known source, evidence cannot be checked, disputed,
or reproduced by another party.

**2. Evidence is not discarded.**
*Explanation:* Evidence that does not support the eventual conclusion is
still recorded and made available, rather than being filtered out because it
was inconvenient.
*Rationale:* Selectively reporting only confirming evidence is a well known
source of bias in scientific reasoning; retaining all evidence is a direct
defense against it.

**3. Comparisons must be fair before they are informative.**
*Explanation:* Before a comparison between two experiments is used as
evidence, the standard requires checking that the comparison is
apples-to-apples: consistent data splits, comparable compute budgets,
matched evaluation protocols, and so on.
*Rationale:* An unfair comparison can produce a confident-looking but
meaningless result.

**4. A single run is weaker evidence than multiple runs.**
*Explanation:* The standard distinguishes between claims backed by a single
execution and claims backed by repeated executions under controlled
variation (for example, different seeds).
*Rationale:* Single runs are subject to noise; treating them as equivalent
to repeated, controlled evidence overstates certainty.

**5. Missing information reduces confidence, it does not block evaluation.**
*Explanation:* When some evidence is unavailable, the standard still
produces an evaluation, but explicitly reflects the gap in how confidence is
communicated.
*Rationale:* Refusing to evaluate incomplete experiments would make the
standard impractical, since most real experiments have some gaps; silently
ignoring the gaps would make the standard misleading.

**6. Contradictions are a first-class output, not an error state.**
*Explanation:* When evidence conflicts — for example, a metric improves
while a related metric degrades — the standard treats identifying and
reporting that conflict as a valuable and expected outcome, not a failure of
the process.
*Rationale:* Contradictions often carry more scientific information than
clean agreement; suppressing them removes signal.

**7. Conclusions are written for a skeptical reader.**
*Explanation:* Output is structured as if it will be read by someone
actively looking for reasons to doubt it, with the evidence and reasoning
available for that scrutiny.
*Rationale:* This discipline keeps the standard honest about what it can
actually support.

**8. The standard evaluates the experiment, not the experimenter.**
*Explanation:* Reasoning is scoped to the evidence produced by an
experiment and its documented context, not to assumptions about the skill,
intent, or reputation of the person who ran it.
*Rationale:* Evaluating people rather than evidence introduces bias and is
outside what evidence-based reasoning can support.

**9. Backend and tooling choices do not change the standard.**
*Explanation:* The definitions of evidence, claim, confidence, and
reproducibility in this standard are independent of any particular storage
system, model provider, or orchestration tool used to apply them.
*Rationale:* Coupling the standard to a specific implementation would make
it fragile and would conflate philosophy with engineering choices.

**10. The standard is versioned and open to revision.**
*Explanation:* As gaps or weaknesses in the standard are found, it is
expected to be revised, with changes documented rather than applied
silently.
*Rationale:* A reasoning standard that cannot be examined or corrected is
not consistent with the transparency it asks of the experiments it
evaluates.

---

## 5. Design Goals

These are the properties the standard is designed to have. They describe
intent for anyone building against the standard, not a specific
implementation.

- **Deterministic reasoning.** Given the same evidence, the same evaluation
  process should be applied, so that outcomes are consistent rather than
  arbitrary.
- **Explainability.** Every output should be accompanied by a rationale that
  a domain expert can follow and, if necessary, dispute.
- **Modularity.** The stages of evaluation (evidence collection, validation,
  comparison, confidence assessment, and so on) should be separable, so that
  each can be reviewed, tested, and improved independently.
- **Extensibility.** New categories of evidence or new types of experiments
  should be addable without requiring the core philosophy to change.
- **Backend independence.** The standard should not depend on any specific
  storage system, experiment tracker, or model provider.
- **Reproducibility.** The standard's own evaluations should be
  reproducible: re-running an evaluation on unchanged evidence should
  produce an unchanged result.
- **Auditability.** Every conclusion should be traceable back through the
  reasoning steps and evidence that produced it, so that it can be reviewed
  after the fact by someone who was not present when it was generated.

---

## 6. Non-Goals

The following are explicitly outside the intent of this project. Listing
them is meant to prevent scope creep and to set correct expectations for
users and contributors.

- **Replacing human reviewers.** The standard supports human review; it does
  not replace the judgment, accountability, or expertise of a human
  reviewer.
- **Predicting conference or journal acceptance.** The standard does not
  attempt to model editorial or reviewer decision processes.
- **Automatically generating papers or claims.** The standard evaluates
  evidence for claims that a human has made or is considering; it does not
  generate scientific claims on its own initiative.
- **Acting as an oracle.** The standard does not claim to produce ground
  truth about whether a result is "correct." It produces an assessment of
  how well available evidence supports a stated claim.
- **Ranking researchers or teams.** The standard evaluates experiments, not
  the people who produced them, and is not designed to produce comparative
  judgments of individuals or groups.
- **Optimizing for impressive output.** The standard does not aim to produce
  confident or polished-sounding conclusions; it aims to produce accurate
  ones, including conclusions that are uncertain, incomplete, or negative.

---

## 7. Future Vision

The philosophy described in this document is intended to guide the design of
several future components of Experiment Audit. This section describes the
architectural role each component is expected to play, without specifying
how it will be implemented.

**Scientific Rule Engine.** A future component responsible for applying the
evaluative principles of this standard (fairness of comparisons, statistical
rigor, reproducibility checks) to a given body of evidence in a consistent,
inspectable way. Its role is to be the place where the standard's principles
are actually applied, rather than left implicit.

**Claim Verification.** A future component responsible for connecting a
stated claim about an experiment to the evidence that does or does not
support it, and reporting the degree and nature of that support. Its role is
to make the link between "what was said" and "what the evidence shows"
explicit and checkable.

**Evidence Graph.** A future structure for representing evidence and the
relationships between evidence items (which evidence supports which claim,
which items are drawn from the same source, which items are in tension with
each other). Its role is to make the traceability required by this standard
a structural property of the system, rather than something reconstructed by
hand for each evaluation.

**Contradiction Engine.** A future component responsible for detecting when
evidence disagrees, whether that is metrics that move in inconsistent
directions, claims that conflict with prior recorded results, or comparisons
that do not hold up under scrutiny. Its role is to ensure that
contradictions, per the core philosophy, are surfaced rather than
suppressed.

**Reasoning Engine.** A future component responsible for coordinating
evidence collection, rule application, claim verification, and
contradiction detection into a coherent evaluation of an experiment. Its
role is to be the orchestrator that ties the other components together while
respecting the boundaries set out in this document: the reasoning it
produces must remain traceable, explainable, and scoped to what the evidence
actually supports.

Each of these components is expected to be built, tested, and revised over
time. This document does not commit to a specific architecture for any of
them. What it commits to is that, whatever form they take, they must remain
consistent with the philosophy described here: evidence before conclusions,
transparency over opaque scoring, and conclusions that are traceable and
explainable.
