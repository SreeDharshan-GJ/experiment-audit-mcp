# Researcher Workflows — How ML Researchers Actually Analyze Experiments

**Status:** First real pass. `research/03_workflows/` was empty before this
document, same starting condition `02_literature/` was in before its first
pass. Written from independent search (practitioner blogs, official docs,
GitHub issues/discussions, arXiv methodology sections, open-source RL
tooling, one industry survey), not invented from first principles.

**Scope note on method:** This is a synthesis of *documented* workflows —
things practitioners wrote down because they taught them, complained about
them, or built tooling around them — not a first-hand user study. No
interviews were conducted for this pass (that is explicitly still open, see
`research-progress.md` §11 item 2). Where a claim below rests on a single
source, that is stated. Where a pattern shows up independently across
unrelated sources (academic methodology guidance, an open-source RL
project's design choices, and a product's own documentation, for instance),
that convergence is noted because it is stronger evidence than any one
source alone.

---

## 1. Why this document exists

Experiment Audit's differentiation claim (`research-progress.md` §8) is
"post-hoc, backend-agnostic judgment over data that already exists,
unprompted." That claim implicitly asserts something about *how people
currently look at their own experiment data* — that they mostly read
dashboards by eye, mostly compare fewer things than they think they're
comparing, and mostly don't have a check in the loop that would catch a
confound or a threshold violation before they act on a false conclusion.
This document tests that assumption against what practitioners have
actually written about their own process, rather than continuing to assert
it.

---

## 2. Workflow archetype 1 — The dashboard-scan loop (most common, weakest evidence of rigor)

**What it is:** Launch runs -> open the tracking-tool dashboard -> look at
loss/reward/accuracy curves side by side -> eyeball which run looks best ->
occasionally cross-reference a hyperparameter column -> decide, informally,
what to try next.

**Evidence this is the default mode:**
- W&B's own documentation frames the core workflow as: create a run, log a
  config dict, log metrics in a loop, then <cite index="7-1">review the results in an interactive dashboard or export data to Python for programmatic access via the Public API</cite>.
  The dashboard-first framing is the *documented default path*, with
  programmatic access positioned as the secondary option.
- A widely-cited practitioner writeup on cross-experiment comparison
  describes exactly this loop and names its own limitation directly: <cite index="1-1">looking across lots of experiments at once gets messy quickly, because there are lots of inputs changing and lots of different possible outputs</cite>.
  The same piece recommends a parallel-coordinates chart specifically
  because scanning dashboards run-by-run doesn't scale past a handful of
  runs.
- Andrej Karpathy's "A Recipe for Training Neural Networks" -- arguably the
  single most-referenced practitioner methodology document in deep
  learning, still cited and reproduced across course materials, debugging
  tool READMEs, and Medium explainers years after publication -- frames
  correctness verification as fundamentally visual and manual: <cite index="34-1">it can be mitigated by being thorough, defensive, paranoid, and obsessed with visualizations of basically every possible thing</cite>,
  and that <cite index="34-1">the qualities that correlate most strongly with success in deep learning are patience and attention to detail</cite>
  rather than any tool doing the checking. The recipe is explicitly a
  human-in-the-loop, eyes-on-every-plot discipline, not an automated one.

**What this implies:** The default unit of analysis is a human looking at
a chart and forming a qualitative impression. Nothing in this loop
structurally catches a confound, a mid-curve NaN silently dropped by a
plotting library, or a plateau that looks like convergence -- catching those
requires exactly the kind of paranoid, exhaustive attention Karpathy
describes, which is precisely the attention that degrades first as the
number of runs grows. This is consistent with, but does not by itself
prove, Experiment Audit's differentiation thesis -- it establishes that the
default workflow has no structural check, but doesn't establish that
researchers *want* one inserted unprompted (that remains untested; see §7).

---

## 3. Workflow archetype 2 -- RL-specific debugging: reward curve plus a stack of secondary signals

**What it is:** Distinct from the general dashboard-scan loop because RL
practitioners have converged on a *specific, named* secondary-signal
checklist beyond the headline reward curve, precisely because reward alone
is known to be misleading.

**Evidence:**
- A practitioner-oriented RL debugging guide lays out the checklist
  explicitly: <cite index="23-1">track the total reward per episode; plot a moving average to smooth out noise and reveal trends; a rising smoothed reward curve is generally a positive sign, but watch for plateaus or collapses</cite>,
  and beyond reward: <cite index="23-1">track the policy loss (actor loss), value function loss (critic loss), and entropy term; unusually high or low losses, or losses that stop decreasing, are red flags</cite>.
  It further recommends watching <cite index="23-1">gradient norm statistics -- very large norms indicate potential explosion, very small norms suggest vanishing gradients</cite>
  and, for policy collapse specifically, <cite index="23-1">monitoring entropy of the policy's output distribution -- a steady decrease is expected, but a rapid collapse to zero might signal insufficient exploration or premature convergence</cite>.
- The same source is explicit that this is *not* a substitute for direct
  behavioral inspection: <cite index="23-1">the most direct debugging tool is to render the environment and watch what the agent does</cite>.
  Metrics guide suspicion; watching behavior confirms it.
- The source explicitly warns against the debugger-first instinct: <cite index="23-1">standard debuggers are useful for finding obvious coding errors but are less effective for diagnosing issues related to the agent's emergent behavior over thousands of interactions; debugging RL often relies more on analyzing logged metrics and visualizations over time</cite>.
- Reward hacking specifically is treated as a distinct, named failure class
  requiring its own detection discipline, not just "reward went up." A
  widely-read synthesis on the topic notes reward hacking is <cite index="18-1">a critical practical challenge as RLHF becomes a de facto method for alignment training</cite>,
  and separately that <cite index="18-1">there is no reliable way to detect or prevent it yet, since improving prompt specification alone is insufficient</cite> --
  i.e., practitioners themselves report this as an open, unsolved detection
  problem, not a solved one with existing tooling.

**What this implies:** RL practitioners already have a *documented,
consensus checklist* of what to look at beyond the headline metric --
reward, policy/value loss, entropy, gradient norms -- which is a stronger
starting point than a from-scratch design would need. It also means
`audit_training_curve`'s scope (currently limited to a single metric's
curve: nulls, jumps, plateaus, oscillation) covers only the reward-curve
slice of this checklist, not the entropy-collapse or gradient-norm slice
that RL practitioners treat as equally load-bearing for diagnosing *why*
a curve looks the way it does.

---

## 4. Workflow archetype 3 -- Sweep-then-eyeball-the-importance-panel

**What it is:** Run a hyperparameter sweep (grid, random, or Bayesian) ->
open the sweep's built-in visualization panel -> read off which
hyperparameter is "most important" -> narrow the next sweep around it.

**Evidence this is a standard, product-supported workflow:**
- W&B ships this as a first-class panel. Its own documentation describes
  the mechanism directly: <cite index="44-1">W&B calculates importances using a tree-based model rather than a linear model, since tree-based models are more tolerant of both categorical data and data that isn't normalized</cite>,
  distinct from the correlation column shown alongside it.
- W&B's documentation is explicit about the panel's own limitations,
  unprompted, in its own help text: <cite index="44-1">correlations show evidence of association, not necessarily causation; correlations are sensitive to outliers, especially with a small sample size of hyperparameters tried; and correlations only capture linear relationships</cite>.
  This is a significant finding for Experiment Audit specifically: **W&B's
  own importance panel already uses a random-forest-style method that
  captures non-linear and interaction effects, explicitly to avoid the
  linear-only blind spot** -- the same blind spot `audit_sweep`'s Pearson-only
  approach has (per `research-progress.md` §3). W&B ships the caveat in
  its docs but the underlying mechanism is materially stronger than
  Pearson correlation, not just differently caveated.
- Real user confusion about this exact distinction shows up in W&B's own
  GitHub issue tracker and community forum, evidence the caveat doesn't
  fully land in practice: one user asks about <cite index="45-1">the difference between importance and correlation in the parameter importance panel</cite> for a categorical CNN architecture search;
  another explicitly asks how the *number of runs* in a Bayesian sweep
  affects the importance panel's reliability, i.e., whether a
  small-sample-size sweep is even trustworthy input to the panel at all
  (a low-run-count concern `audit_sweep`'s 10-run floor already encodes
  deterministically); a third user reports getting a <cite index="48-1">high importance and high positive correlation for a True/False layer-freezing variable but no way to tell from the panel which value (True or False) is the "higher" one</cite> --
  a readability gap, not a statistical one, but one that shows the panel is
  being read by users who don't have the statistical background to
  interpret it correctly unaided.

**What this implies:** The "sweep -> importance panel -> narrow the search"
loop is real, widely used, and partially instrumented already by the
market leader -- but with a materially stronger statistical method
(random-forest importance) than `audit_sweep` currently implements
(Pearson correlation + Fisher z-transform). This is a genuine
differentiation risk that should be weighed against `audit_sweep`'s actual
justification (backend-agnostic, works without going through W&B's sweep
infrastructure specifically) rather than assumed away.

---

## 5. Workflow archetype 4 -- Ablation studies: designed correctly on paper, executed with known confound risk in practice

**What it is:** Define a baseline configuration, define a component to
remove or vary, hold everything else constant, run both, compare.

**Evidence the "hold everything else constant" step is where things break:**
- A methodology synthesis on controlled ablation studies names the failure
  mode directly, independent of Experiment Audit's own framing: <cite index="24-1">recurring pitfalls in controlled ablation studies include misalignment between ablation design and the original system or context, leading to confounded outcomes; ambiguous ablation boundaries; and incomplete procedural specification</cite>.
  The same source's stated best practice is almost word-for-word what
  `audit_ablation` checks for: <cite index="24-1">cross-check every step against the full experimental or computational methodology; document and pre-register all constants (seeds, splits, hyperparameters)</cite>.
- Venue-level policy now treats this as enforceable, not optional: <cite index="33-1">NeurIPS requires error bars, confidence intervals, or statistical significance tests for main claims in submitted papers, and ICML 2026 encourages code submission and weighs reproducibility in decisions</cite>,
  and the expectation is stated plainly: <cite index="33-1">the expectation across top venues is that architecture claims come with ablation evidence, not anecdote</cite>.
  This is evidence of institutional pressure toward rigor that individual
  researcher workflows have not yet structurally caught up to -- the
  requirement exists at review time, well after the confound would have
  needed to be caught to avoid wasted compute or a retraction-worthy claim.
- The confound risk is severe enough that even benchmark-diagnostic papers
  build explicit self-checks for it: one recent benchmark paper runs a
  dedicated sub-study specifically to verify its own ablation metrics
  aren't <cite index="31-1">driven by a specific implementation choice such as crop-margin boundaries</cite>,
  i.e., researchers who are sophisticated enough to be building benchmarks
  still treat "is my ablation actually isolating what I think it's
  isolating" as a question requiring its own validation pass, not
  something they trust by default.

**What this implies:** The confound-checking discipline `audit_ablation`
automates (diffing all fields between a claimed-ablation pair, not just the
claimed variable) is not a hypothetical problem -- it is a named, recurring,
citable failure mode with institutional consequences (rejected or
weakened claims) attached. The gap is that none of these sources describe
*who currently catches this before submission* -- whether it's the
researcher re-reading their own config diffs, a labmate's review, or
nothing until a reviewer asks. That "who catches it, and when" step is
exactly the workflow-insertion-point question this document was supposed
to answer and could not fully resolve from documentation alone (see §7).

---

## 6. Workflow archetype 5 -- Reproducibility-first tracked-experiment culture (open-source RL specifically)

**What it is:** A minority but well-documented workflow where every run's
full environment -- code version, dependencies, exact CLI command, seed --
is captured at run time specifically so any run can be exactly reproduced
later by a third party, not just compared informally.

**Evidence:**
- CleanRL, a widely-used reference RL implementation library, treats this
  as a first-class design feature rather than an afterthought: <cite index="84-1">when running an experiment, W&B automatically tracks the source code, dependencies, hyperparameters, the exact terminal command used, training metrics, videos of the agent playing, system metrics, and logs</cite>.
- The associated Open RL Benchmark project built tooling specifically to
  close the loop from "tracked" to "actually reproducible": <cite index="76-1">CleanRL's reproduce module allows a user to generate, from a benchmark run reference, the exact command suite for an identical reproduction of that run</cite>,
  with the explicit motivation that <cite index="76-1">reproducing experimental results is often challenging due to evolving codebases, incomplete hyperparameter listings, version discrepancies, and compatibility issues</cite>.
- This workflow also normalizes a specific artifact this project does not
  yet touch: shared, narrative comparison documents. Open RL Benchmark
  explicitly uses <cite index="78-1">reports -- interactive documents designed to enhance the visualization of selected representations -- to provide a more user-friendly format for practitioners to share, discuss, and analyze experimental results, even across different projects</cite>.

**What this implies:** This is the workflow archetype where Experiment
Audit's backend-agnostic, evidence-attached judgment would be *easiest* to
insert, because these practitioners already treat full-fidelity run
capture as normal and already build tooling to close gaps between
"tracked" and "actually verified." It is also a narrower audience than
"ML researchers" broadly -- this rigor is disproportionately present in
open-source RL infrastructure projects and benchmark-building efforts, not
necessarily in the median single-author academic lab. Whether it
generalizes beyond RL/benchmark contexts to, e.g., NLP or CV labs running
supervised fine-tuning sweeps is not evidenced here and should not be
assumed.

---

## 7. What this pass could NOT establish (explicit gaps, not silently dropped)

- **No first-hand interview or survey evidence exists yet in this project**
  specific to the actual target user (a solo or small-team ML researcher
  deciding whether to trust an agent's unprompted judgment about their own
  runs). Everything above is inferred from what practitioners chose to
  write publicly, which skews toward people organized enough to blog,
  document, or build tooling -- likely *more* rigorous than the median
  researcher, not less. This means the pain points in `pain-points.md` are
  probably a floor, not a ceiling, on how common the underlying failure
  modes are.
- **No evidence was found of researchers requesting unprompted,
  agent-initiated auditing of their own runs** -- every documented workflow
  above is a human-initiated check, even in the most rigorous archetype
  (§6). This is the single most load-bearing unknown for Experiment
  Audit's core UX bet, restated from `research-progress.md` §7, and this
  pass did not close it. Reddit-specific search (an originally planned
  source per the task brief) returned no usable results in this pass --
  worth a follow-up with a different search strategy or a Pushshift-style
  archive query, not concluded to be absent.
- **No direct evidence was found of a "review meeting" or "standup"
  practice specific to experiment analysis** in ML research labs
  specifically (as opposed to general product/growth experimentation
  meetings, which is a different discipline entirely and was excluded from
  this synthesis after initial searches returned mostly non-ML-research
  material -- e.g., A/B-testing program management, clinical trial
  review boards). This is a genuine gap, not a null result -- it means the
  team/collaborative-review workflow layer remains unevidenced for ML
  research specifically.

---

## Decision Impact

**What assumptions were confirmed?**
- The dashboard-scan-by-eye loop is the documented default across general
  ML and RL tooling alike, with no structural check against confounds or
  curve pathologies built into the loop itself (§2). This supports
  Experiment Audit's premise that the *opportunity* for missed errors is
  real and structural, not anecdotal.
- Ablation confounds are a real, recurring, citable, and now
  institutionally-enforced (NeurIPS/ICML checklist-level) failure mode,
  independently corroborating `audit_ablation`'s reason for existing (§5).
- RL practitioners already converge on a specific secondary-signal
  checklist (entropy, gradient norms, value/policy loss) beyond the
  headline metric -- this is free, evidence-backed scope guidance for where
  `audit_training_curve` is currently narrower than practitioner practice
  (§3).
- The most rigorous, reproducibility-first workflow archetype (tracked
  open-source RL experiments) already treats full-fidelity run capture and
  shared comparison documents as normal -- this is the friendliest entry
  point for backend-agnostic post-hoc judgment (§6).

**What assumptions were disproven?**
- The implicit assumption that `audit_sweep`'s Pearson-based approach is
  merely "simpler than but comparable to" what practitioners already have
  access to was too generous: W&B's own shipped importance panel already
  uses a random-forest method that handles non-linearity and categorical
  variables, which Pearson does not (§4). "Works on any backend, no
  re-run required" remains `audit_sweep`'s real differentiator against
  this -- but "as statistically capable as what's already in the market
  leader's UI" is not, and should not be implied in the tool's own docs.

**What should we build because of this?**
- Nothing yet -- this document is evidence, not a design decision. But it
  identifies two concrete, literature-and-practice-grounded scope
  questions worth carrying into `04_benchmarks/` or a future design pass:
  (a) whether `audit_training_curve` should eventually cover
  entropy-collapse and gradient-norm signals, not just the headline
  metric, given §3's evidence that RL practitioners treat these as
  equally diagnostic; and (b) whether `audit_sweep`'s docs should state
  its Pearson-vs-tree-based tradeoff explicitly rather than leaving a
  reader to assume parity with W&B's panel, given §4's finding.

**What should we stop building because of this?**
- Nothing in this pass identifies a currently-planned feature to stop.
  It does raise a caution: do not build toward "replace the sweep
  importance panel" as a value proposition -- W&B's version is already
  statistically stronger for in-W&B sweeps, and `audit_sweep`'s honest
  differentiator (backend-agnostic, no re-run) is a narrower and more
  defensible claim than "better importance ranking."
