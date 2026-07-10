# Benchmark Plan — How Should Experiment Audit Itself Be Evaluated?

**Status:** First real pass. `research/04_benchmarks/benchmark-plan.md` was
0 bytes going into this pass, same starting condition every other
`research/` file was in before its own first pass (per
`research-progress.md` §12's established provenance discipline). Written
from independent search into benchmark design literature, not from any
prior framing of what this project's evaluation "should" look like.

**Scope note, stated up front:** this document does not propose building
anything. Per the task brief, it does not propose a v2, invent new tools,
or write production code. It answers one question — *how would a skeptical
ML researcher come to trust Experiment Audit's judgment tools?* — and is
explicit throughout about which parts of the answer are engineering work
already done (§Part 4), which parts are a real, currently-unfilled research
gap (§Part 3), and which existing project assumptions this pass challenges
(closing Decision Impact).

**One correction to the task brief's own framing, in the same spirit as
`research-progress.md` §12:** the task brief lists "competitor landscape"
as an already-completed phase. `research/01_landscape/` is, as of this
pass, three 0-byte files. The substance of a competitor landscape review
does exist — it lives in `research-progress.md` §5 and
`research/02_literature/related-work.md` §6 (W&B's MCP server, Optuna's
fANOVA) — so the *research* was not skipped, but it was never written to
the location the project's own directory structure promises it lives in.
This is not corrected in this pass (out of scope: this pass is
`04_benchmarks/` only), but it is named rather than silently treated as
"already done and therefore not worth checking" — exactly the failure mode
`research-progress.md` §7 already flagged once for this project and warned
could recur.

---

## Part 1 — What existing benchmarks get right, and what they cost

Each entry below states what the benchmark actually established
methodologically, not what its marketing copy claims. Entries for
Henderson et al., Dror et al., Hutter et al./fANOVA, and AblationBench are
**not re-derived here** — they were already synthesized in
`research/02_literature/related-work.md` §1–4, and this document only pulls
forward the specific *evaluation-methodology* lessons from them in §1.7.
Everything else below is new to this pass.

### 1.1 MLE-bench (OpenAI, 2024)

- **What it evaluates:** whether an AI agent can perform end-to-end ML
  engineering — training a model, preparing data, submitting a result —
  against 75 real Kaggle competitions.
- **Why researchers trust it:** the tasks are not authored by the
  benchmark's own creators; they are pre-existing, independently judged
  competitions with an external, adversarial-in-the-good-sense scoring
  mechanism (Kaggle's own leaderboard) that predates the benchmark and
  cannot be gamed by the benchmark designers.
- **Dataset design:** 75 curated competitions spanning tabular, vision, NLP,
  and signal-processing domains — breadth chosen specifically so no single
  task family dominates the score.
- **Ground-truth methodology:** OpenAI established human baselines for
  each competition using Kaggle's own publicly available leaderboards —
  ground truth is not "the benchmark authors' opinion of a good score," it
  is an independently-collected human performance distribution the agent's
  score is placed against (bronze/silver/gold medal thresholds Kaggle
  itself defines).
- **Metrics:** medal-rate against the human leaderboard distribution, not a
  single scalar accuracy — the best-performing setup achieved at least the
  level of a Kaggle bronze medal in only 16.9% of competitions, a number
  that is meaningless without the human-baseline context it's reported
  against.
- **Strengths:** external, pre-existing ground truth immune to
  benchmark-author bias; a real human performance distribution to compare
  against rather than an arbitrary pass bar; open-sourced grading code.
- **Weaknesses:** the benchmark's own maintainers note known issues with
  certain competitions that they are deliberately not fixing mid-stream to
  avoid invalidating an already-published leaderboard, addressing this
  instead in a versioned v2 release with a version column distinguishing
  v1 and v2 results — an admission that ground truth itself can have
  defects, and the fix is *versioning*, not silent correction.
- **Lesson for Experiment Audit:** ground truth quality problems are normal
  and expected, not a sign of a failed benchmark — what matters is whether
  defects are disclosed and versioned rather than silently patched. This
  bears directly on how Experiment Audit's own ground-truth set should be
  maintained (§3.7).

### 1.2 SWE-bench / SWE-bench Verified (Princeton/OpenAI collaboration)

- **What it evaluates:** whether a coding agent can resolve a real GitHub
  issue by producing a patch that makes previously-failing tests pass
  without breaking previously-passing ones.
- **Why researchers trust it:** it measures the entire agent system
  against real GitHub issues from popular open-source repositories, with
  the agent's submitted patch graded against the real unit tests from the
  pull request that actually closed the original issue — ground truth is
  not authored for the benchmark, it is recovered from history that
  already happened.
- **Dataset design:** 2,294 real issues in the full set; SWE-bench Verified
  is a curated, human-verified subset of 500 tasks from 12 repositories,
  deliberately smaller and higher-quality rather than larger and noisier.
- **Ground-truth methodology — the single most load-bearing lesson here:**
  human annotators reviewed each of the 500 Verified instances and
  confirmed three properties: the issue description is unambiguous and
  actionable, the designated tests actually verify the fix described, and
  a competent engineer could resolve it without external context.
  Critically, the original, unverified SWE-bench contained instances where
  even expert human annotators could not determine whether a model's
  solution was correct, because the ground-truth tests themselves were
  flawed or the problem statement was ambiguous — the *fix* wasn't better
  models, it was removing indeterminate cases from the ground-truth set
  entirely rather than forcing a label onto them.
- **Metrics:** binary resolved/unresolved per instance, aggregated to a
  pass rate, but explicitly reported alongside a difficulty tiering across
  four bands so a single aggregate score doesn't hide that a model might
  resolve nearly all easy instances and far fewer hard ones.
- **Strengths:** test-based grading removes human graders from the
  per-instance loop entirely (cheap, consistent, reproducible); the
  Verified subset's provenance (who checked what, against what criteria)
  is published, not just the pass rate.
- **Weaknesses — and this is the most important thing to import into
  Experiment Audit's own design:** even after verification, a follow-up
  differential-testing study found that when all developer-written tests
  are considered rather than only the tests the original PR happened to
  modify, 7.8% of "passed" patches actually fail additional tests, and
  manual inspection of behavioral discrepancies found that as many as
  11.0% of all reported successes were invalid. In other words: **a
  benchmark that looked airtight — test-based, human-verified, widely
  adopted — still had a measurable false-positive rate in its own grading
  mechanism**, discovered more than a year after Verified shipped, by
  researchers outside the benchmark's own team.
- **Lesson for Experiment Audit:** (a) test-based/deterministic grading is
  strictly preferable to human grading wherever it's possible, because it's
  cheap and reproducible — but (b) even deterministic grading can encode a
  false ground truth if the *grading criterion itself* is under-specified,
  and (c) the way this was caught was independent, outside re-verification
  using a different method than the original grading — which is a direct
  argument for §3.9's third-party-reproduction requirement, not an optional
  nicety.

### 1.3 HELM — Holistic Evaluation of Language Models (Stanford CRFM)

- **What it evaluates:** foundation models across many scenarios and many
  metric dimensions simultaneously, rather than a single leaderboard
  number.
- **Why researchers trust it:** HELM provides broad coverage while
  explicitly recognizing its own incompleteness, uses multi-metric
  measurement rather than a single score, and standardizes evaluation
  procedure across models so scores are actually comparable, with all raw
  model outputs and complete results released publicly for inspection, not
  just aggregate scores.
- **Dataset design:** dozens of existing scenarios/datasets aggregated
  under one standardized harness rather than one bespoke dataset —
  HELM's contribution is largely *procedural standardization* (same
  prompting method, same inference parameters, same metric definitions
  applied across every model), not novel data collection.
- **Ground-truth methodology:** inherited per-scenario from each underlying
  dataset (e.g. MMLU's answer keys); HELM's own contribution is not
  new ground truth but a documented, versioned procedure for how every
  model is scored against it identically.
- **Metrics:** deliberately plural — seven or more metric categories
  including accuracy, fairness, and toxicity are reported per scenario
  rather than collapsed into one leaderboard number, on the explicit
  premise that a single scalar hides which dimension a model is actually
  weak on.
- **Strengths:** versioned releases (e.g. HELM Lite, HELM Classic, named
  version numbers like v1.15.0) so a cited HELM score is always
  attributable to a specific, frozen procedure, not a moving target;
  transparency of raw outputs enables independent re-scoring.
- **Weaknesses:** metrics like fairness and toxicity are highly contextual
  and often hard to quantify reliably, and closed models can't be
  inspected for training data or fine-tuning details the way open models
  can, which complicates apples-to-apples comparison — breadth of metric
  coverage does not fully solve the problem of metric validity for the
  harder-to-define dimensions.
- **Lesson for Experiment Audit:** a single "accuracy" number for a
  judgment tool (e.g. "audit_ablation is 91% accurate") would replicate
  exactly the failure mode HELM was built to correct. Experiment Audit's
  benchmark should report per-tool, per-verdict-type metrics (§3.5) rather
  than one aggregate score, and should version each benchmark release the
  way HELM does, not silently update numbers in place.

### 1.4 OpenAI Evals

- **What it evaluates:** whether a model or prompt produces outputs meeting
  a specified criterion, via either deterministic code-based checks or a
  second model acting as grader.
- **Why researchers (partially) trust it:** deterministic/code-based
  checks are trusted for the same reason SWE-bench's test-based grading is
  — they're reproducible and have no grader-variance. Model-graded checks
  are trusted *less*, and the framework's own documentation says so
  explicitly.
- **Ground-truth methodology:** either an exact/fuzzy string match against
  a known-correct answer (deterministic), or a second LLM's judgment
  (model-graded) — the framework's own guidance is unusually candid about
  the second category's limits.
- **Metrics:** pass/fail per test case, aggregated to a pass rate; for
  model-graded evals, sometimes a 1–5 rating scale.
- **Strengths:** low cost of entry for deterministic checks; wide adoption
  and an open registry of reusable eval definitions.
- **Weaknesses — directly relevant to Experiment Audit:** model grading has
  a real error rate, so official guidance is to validate model-grader
  performance against human evaluation before trusting it at scale, and
  best practice is to use a different model to grade than the one that
  produced the answer being graded. More recent practitioner guidance is
  blunter still: graders that are not calibrated against human judgments
  risk measuring noise rather than quality, and a single grader can miss
  failure modes a composed, multi-grader approach would catch.
- **Lesson for Experiment Audit:** this is a direct, negative precedent
  against one tempting shortcut — using an LLM to grade whether
  `audit_ablation`'s verdict was "reasonable" instead of building a real
  human-labeled ground-truth set. Experiment Audit's own three judgment
  tools are *already* deterministic (no LLM in the loop at inference time,
  per `research-progress.md` §2's verified architecture claim), which is a
  genuine structural advantage over anything OpenAI Evals' own limitations
  describe — but this advantage would be **thrown away** if the benchmark
  built to validate those tools used an LLM-as-judge instead of human
  ground truth to decide whether a verdict was correct. Human-labeled
  ground truth is not optional for this project; it is the one part of
  "evidence-backed" that a deterministic tool cannot substitute LLM
  judgment for without contradicting its own design principle #1
  (`research-progress.md` §2).

### 1.5 Open RL Benchmark / CleanRL (Huang et al., 2024)

- **What it evaluates:** not an agent's *capability* the way the other
  entries here do — it is a **data source**: a large, public,
  fully-tracked corpus of real RL training runs, not a scored leaderboard.
  Listed here because it is the single most directly reusable asset found
  in this pass for building Experiment Audit's own ground-truth sets (see
  §3.2).
- **Why researchers trust it:** it is community-driven — anyone can
  download, use, and contribute to the data — and at time of writing more
  than 25,000 runs have been tracked, for a cumulative duration of more
  than 8 years, stored on W&B, the exact backend Experiment Audit's v1
  already targets.
- **Dataset design:** organized by library (CleanRL, Stable Baselines3,
  reference implementations) and environment family (Atari, MuJoCo,
  Procgen, and others), with runs spanning multiple algorithms, seeds, and
  hyperparameter settings per environment — real sweep-shaped and
  ablation-shaped data already exists inside this corpus, not synthetic
  approximations of it.
- **Ground-truth methodology:** N/A in the labeling sense — this is raw
  experimental data, not a labeled benchmark. Ground-truth labels
  (confounded/clean, pathological/normal) would have to be added by
  Experiment Audit's own annotation process (§3.4) on top of this corpus;
  the corpus itself only guarantees the *data* is real and reproducible.
- **Metrics:** N/A — not a scored benchmark.
- **Strengths — this is the most load-bearing finding of this section:**
  every experiment includes a complete configuration with all
  hyperparameters, frozen versions of dependencies, and the exact command
  including the random seed needed to reproduce it, specifically so
  results can be exactly replicated rather than merely approximated. This
  is a real corpus of real training curves, real sweeps, and real config
  diffs across paired runs (e.g. same algorithm, different seeds; same
  algorithm, different single hyperparameter) — precisely the shape of
  data `audit_ablation`, `audit_training_curve`, and `audit_sweep` are
  built to judge, already public, already versioned, and already on the
  exact backend (W&B) this project's `WandbBackend` already speaks to.
- **Weaknesses:** skews RL-heavy — the same skew `research-progress.md` §7
  already names as a property of what's publicly documented, not
  necessarily of the target user base. Contains no pathology labels
  (crashed/confounded/clean) out of the box; every hyperparameter sweep in
  it was run for its own algorithmic purpose, not designed as a labeled
  test case, so mining it for ground truth requires the same expert
  annotation effort a from-scratch dataset would (§3.4) — it saves the
  *data collection* cost, not the *labeling* cost.
- **Lesson for Experiment Audit:** this is the strongest concrete answer to
  "what datasets can be reused" (task Q2) found in this pass. It should be
  the primary source for the real-data half of the ground-truth benchmark
  in §3, specifically because it removes the single biggest confound a
  synthetic dataset would carry — "is this training curve/sweep/ablation
  pair actually shaped like something a real researcher produced, or does
  it only look realistic because the benchmark's own authors made it up."

### 1.6 ML Reproducibility Challenge / Papers with Code

- **What it evaluates:** whether a *published paper's own claimed results*
  can be independently reproduced by a third party using only the paper
  and, where available, its released code.
- **Why researchers trust it:** it is peer-adjacent (OpenReview-hosted
  contributed reviews), open, and — most relevantly — explicitly refuses to
  reduce its own findings to pass/fail: the result of a reproducibility
  study is not meant to be a simple pass/fail outcome; the stated goal is
  to identify which parts of a contribution can be reproduced, and at what
  cost in terms of compute, time, people, development effort, and
  communication with the original authors.
- **Dataset design:** self-selected — participants choose a paper from a
  pool of recent top-venue accepted papers, not a fixed curated set.
- **Ground-truth methodology:** the original paper's own reported numbers
  serve as the target; "reproduced" is graded on a graded scale (within
  what tolerance, under what compute budget, with what deviations
  documented) rather than binary match.
- **Metrics:** narrative + graded closeness to the original claim, not a
  single score across participants (the challenge doesn't produce a
  cross-paper leaderboard).
- **Strengths:** the graded, narrative reporting format resists the
  temptation to over-simplify a genuinely nuanced outcome ("half
  reproduced, at 3x the compute, with one hyperparameter undocumented")
  into a misleading single number.
- **Weaknesses:** not automatable at scale — every instance requires a
  human reproducer's real effort, which is why it operates as a periodic
  community challenge rather than a continuously-running benchmark.
- **Lesson for Experiment Audit:** the refusal to force a binary outcome
  onto a genuinely graded or ambiguous case is directly applicable to
  ground-truth construction in §3.4 — cases where Experiment Audit's own
  human annotator panel disagrees should be reported as a distinct,
  visible "contested" category, not silently resolved by majority vote and
  hidden inside an aggregate accuracy number.

### 1.7 Pulled forward from the existing literature review (not re-derived)

From `research/02_literature/related-work.md`, the specific
*evaluation-methodology* lessons (not the substantive findings, already
recorded there) that apply to benchmark design specifically:

- **AblationBench** (`related-work.md` §4) is the closest existing
  precedent for *how to score a judgment task against a human baseline*:
  it reports both a frontier-model score and a human baseline score on the
  same task, on the same metric (F1), so the model's number has an
  interpretable reference point rather than existing in isolation. This
  exact pattern — tool score alongside human-panel score, same metric,
  same cases — is the structure §3.5 and §3.10 recommend for Experiment
  Audit's own benchmark.
- **Dror et al.** (`related-work.md` §2) establishes that reporting a
  result without a significance test, and reporting multiple comparisons
  without correction, are both common and both wrong — directly relevant
  to how Experiment Audit's own benchmark results should be reported
  (§3.5's confidence-interval requirement) and a reminder that this
  project's benchmark work is subject to the same statistical discipline
  it's trying to hold its *product* to.
- **Hutter et al. / fANOVA** (`related-work.md` §3) is not a benchmark of
  Experiment Audit but a stronger *reference method* `audit_sweep` can be
  compared against on Optuna-run sweeps specifically — not as a
  ground-truth oracle (fANOVA and Pearson are different methods measuring
  related but non-identical things), but as a documented divergence
  characterization (§3.2, §3.5).

---

## Part 2 — What none of these benchmarks solve for Experiment Audit specifically

Every benchmark in Part 1 evaluates either (a) an agent's ability to
*produce* an artifact (a Kaggle submission, a code patch) graded against
an objective external outcome, or (b) a language model's output quality
against a labeled answer key. **None of them evaluate a deterministic,
non-LLM judgment function's calibration against expert human judgment on
real experimental data** — which is precisely what `audit_ablation`,
`audit_training_curve`, and `audit_sweep` are. This is a real, structural
gap, not a reason to treat the above as irrelevant: the *methodology*
lessons (external ground truth, versioned releases, human/baseline
comparison, disclosed ambiguity, third-party reproducibility) transfer
directly even though no existing benchmark's *data* or *task shape*
transfers directly. Part 3 designs the benchmark this gap requires.

---

## Part 3 — Designing Experiment Audit's own evaluation methodology

### 3.1 What capabilities should be benchmarked? (Task Q1)

Three genuinely different things need three genuinely different
evaluations. Conflating them — which nothing in this repository currently
does, but which would be an easy mistake for a future pass to make — would
produce a misleading trust signal:

1. **Implementation correctness** — does the code do what its own
   docstring says it does, on cases with one unambiguous right answer?
   *Already substantially covered* (233 passing tests, `analysis/*.py`
   coverage, `adversarial_cases.py`'s six MCP-layer cases). This is a
   regression suite, not a benchmark, and should keep being called that.
2. **Calibration against expert human judgment on real, ambiguous data** —
   when `audit_ablation` says `confounded`, would an independent panel of
   ML researchers, looking at the same real run pair, agree? This is the
   **actual gap** (§3.4–§3.6) — nothing in the repository currently
   measures this, and no amount of additional synthetic
   correctness-testing can substitute for it, because synthetic cases are
   constructed to have one clean answer by design, which is exactly the
   property real, contested cases don't have.
3. **Tool-selection accuracy** — does an MCP client (Claude or another
   agent) invoke the right tool given a natural-language prompt? *Already
   built* (`scripts/tool_selection_eval.py`,
   `scripts/tool_selection_prompts.py`, 15 prompts with 4 distractor
   pairs) but **not yet run** (`docs/tool-selection-eval.md`, blocked on
   `ANTHROPIC_API_KEY`/network access in this environment, per that
   document's own status note — unchanged by this pass).

Retrieval tools (`test_connection`, `list_runs`, `get_run_summary`,
`get_metric_history`, `compare_runs`) do **not** need a benchmark of this
kind at all — they need only correctness-against-backend-API-shape tests
(category 1), consistent with the project's own "don't compete on
retrieval" positioning (`research-progress.md` §8). Treating retrieval
tools as needing calibration-against-human-judgment benchmarking would
misapply category-2 rigor to a category-1 problem and would be a genuine
design mistake, not just wasted effort — retrieval tools have no
"judgment" to calibrate.

### 3.2 What datasets can be reused? (Task Q2)

- **Open RL Benchmark / CleanRL's tracked-experiment corpus** (§1.5) —
  primary reuse candidate. Real W&B-shaped runs, real seeds/hyperparameter
  variations, real training curves, already public and versioned. This
  supplies raw *cases*; it does not supply *labels* (see §3.4).
- **`tests/fixtures/adversarial_cases.py`'s six existing cases** — reusable
  as-is, but only for category 1 (implementation correctness), not
  category 2 (calibration). They were built to have one unambiguous
  correct answer and should stay in that role, not be repurposed as
  calibration evidence.
- **Optuna-run sweeps with `FanovaImportanceEvaluator` output** — reusable
  as a *reference-method comparison* dataset for `audit_sweep`
  specifically (any public Optuna study with logged trials), not as a
  ground-truth oracle (fANOVA and Pearson-with-Fisher-z are different,
  non-interchangeable methods; see §1.7).
- **NeurIPS/ICML supplementary materials with published ablation tables**
  (referenced generally in `research/03_workflows/researcher-workflows.md`
  §5 for the venue-enforcement finding) — a candidate source of real
  claimed-ablation pairs, but each one would need the underlying run
  configs recovered (not just the reported numbers), which is not
  guaranteed to be public even when the paper is. This is a *possible*
  reuse source, not a confirmed one — flagged as unconfirmed rather than
  assumed available.

### 3.3 What datasets must be created? (Task Q3)

Three ground-truth sets, one per judgment tool, are the real deliverable
this research phase should produce (in a future execution pass, not this
document):

1. **Ablation ground-truth set** — real baseline/ablation run pairs (from
   Open RL Benchmark where possible; hand-constructed only where a
   real-data example of a specific known failure mode can't be found)
   labeled `clean` / `confounded` / `contested` by an independent expert
   panel (§3.4), stratified to include both easy, obvious cases and the
   adversarial "trap" cases in §3.8.
2. **Curve pathology ground-truth set** — real training curves labeled for
   each of the four signals `audit_training_curve` currently claims to
   detect (`null_values`, sudden jump, plateau, oscillation), *and*
   labeled for cases that resemble but are not those signals (a scheduled
   LR-warmup jump, a genuinely converged plateau) — the negative cases
   matter as much as the positive ones and are currently absent from
   `adversarial_cases.py` entirely.
3. **Sweep importance divergence set** — sweeps with both a Pearson
   ranking (what `audit_sweep` produces) and a fANOVA ranking (from an
   Optuna-run version of the same or a matched sweep) computed, to
   characterize *how often and how much* the two methods disagree, framed
   as a documented tradeoff characterization, not a pass/fail grade (per
   §1.7's point that fANOVA is not a ground-truth oracle for this tool).

None of these three exist yet. This is the single largest concrete
deliverable this research phase identifies and does not itself produce.

### 3.4 How should ground truth be established? (Task Q4)

Adapted directly from SWE-bench Verified's three-property check (§1.2),
because it is the most directly transferable precedent found in this pass:

- **Independent panel, not the tool's own designers.** At minimum 3
  ML researchers per case, blind to `audit_*`'s own verdict when labeling
  (shown only the raw run data/curve/sweep, not the tool's output) —
  avoids anchoring, the same reason SWE-bench Verified's annotators
  reviewed problem statements independently of any candidate solution.
- **Three properties checked per case**, adapted from SWE-bench Verified's
  clarity/correctness/solvability triad: (1) **realism** — is this case
  actually representative of real research data, not an artificially
  clean or artificially confusing construction; (2) **determinacy** — does
  the panel actually agree, or is this a genuinely contested case; (3)
  **label confidence** — would the panel's judgment change under
  additional context a real researcher would also lack (e.g. they don't
  know the researcher's intent either).
- **Disagreement is a labeled outcome, not noise to average away.** Cases
  without panel majority go into a separate `contested` bucket, reported
  and scored separately (§3.5), not folded into the main accuracy number
  — directly following the ML Reproducibility Challenge's refusal to force
  binary outcomes (§1.6) and SWE-bench Verified's practice of removing
  indeterminate instances rather than keeping them in the graded set
  (§1.2).
- **Report inter-annotator agreement** (e.g. Fleiss' kappa across the
  panel) alongside every ground-truth set release — a number the current
  project has never had to report before (unit tests don't have
  "annotator agreement"; this is a genuinely new category of evidence for
  this project) and one of the clearest, cheapest signals of ground-truth
  quality a skeptical reader can check without re-doing the work
  themselves.
- **No LLM-as-judge for ground truth**, per §1.4's OpenAI Evals lesson —
  human panel labels are the ground truth; an LLM may assist a human
  annotator's workflow (e.g. surfacing candidate cases) but never
  substitutes for the human judgment that becomes the recorded label.

### 3.5 What metrics should be reported? (Task Q5)

Per tool, per verdict category, not one aggregate score (per HELM's §1.3
lesson):

- **Precision, recall, F1, and full confusion matrix** per verdict type
  (e.g. for `audit_ablation`: `clean` vs `confounded` vs
  `insufficient_samples`/refuse) — not accuracy alone, which is misleading
  under class imbalance (if most real ablation pairs in the wild are
  actually clean, a tool that always says `clean` would score deceptively
  well on accuracy while being useless).
- **Confidence calibration**, a genuinely new evaluation category for this
  project: does the tool's own `confidence: low/high` field actually track
  real accuracy — i.e., are `high`-confidence verdicts more often correct
  than `low`-confidence ones, against the ground-truth panel? Currently,
  nothing in the test suite checks this; existing tests confirm confidence
  is downgraded to `low` under specific documented conditions (e.g. partial
  data — see `adversarial_cases.py` case 5), which is a *correctness*
  check (category 1, §3.1), not a *calibration* check (category 2). These
  are different claims and the current suite only supports the former.
- **Abstention rate and abstention correctness, reported separately from
  accuracy-when-answered** — following SWE-bench Verified's separation of
  "resolved" from "removed as unsolvable" (§1.2): a tool could otherwise
  inflate its own apparent accuracy simply by refusing more often on hard
  cases, which would look like improvement but isn't. `audit_sweep`'s
  existing `insufficient_samples` refusal and any future refusal behavior
  in the other two tools should both be scored on whether the refusal
  itself was *warranted* (would the human panel also say "can't tell
  here"), not just counted.
- **Human-panel agreement rate as an explicit ceiling**, reported
  alongside the tool's own score on the same cases — directly following
  AblationBench's structure (§1.7): report the tool's F1 next to the
  panel's own inter-annotator F1/agreement, so a reader can see how close
  to the achievable ceiling the tool gets, not just an isolated number.
- **An unaided-Claude baseline on the same cases** — Claude given the same
  raw run data with no `audit_*` tool call, asked to reason about it
  directly. This is the metric that would actually validate or falsify the
  project's own central differentiation claim (`research-progress.md` §8):
  if unaided Claude matches the tool's score, the tool's value proposition
  weakens considerably regardless of how statistically sound its internals
  are; if it doesn't (which AblationBench's ~38–45% figure suggests is
  likely for the ablation case specifically, per `related-work.md` §4),
  that gap *is* the evidence the mission statement asks for.
- **Confidence intervals or an explicit small-sample caveat** on every
  reported metric, per Dror et al.'s core lesson (§1.7) — this project
  should not report a bare "F1 = 0.83" on a ground-truth set of, say, 40
  cases without saying so.

### 3.6 How should false positives and false negatives be measured? (Task Q6)

- **Define per tool, not generically**, because the two error directions
  are not symmetric in cost, and the project's own design principle #3
  ("refuse rather than mislead," `research-progress.md` §2) already states
  a directional preference that the benchmark should be built to check,
  not just assume:
  - `audit_ablation`: **false positive** = tool says `confounded` when the
    panel says `clean` (costs researcher time/trust, but errs toward
    caution); **false negative** = tool says `clean` when the panel says
    `confounded` (lets a bad comparison stand unflagged — the more
    dangerous direction for a tool whose whole premise is trust).
  - `audit_training_curve`: **false positive** = flags a pathology signal
    on a curve the panel considers normal (e.g. a scheduled LR-warmup
    jump misread as `sudden_jump`); **false negative** = misses a
    pathology the panel would flag (e.g. a slow-onset issue below every
    fixed threshold).
  - `audit_sweep`: **false positive** = a covariance warning on
    hyperparameters the panel considers independently meaningful; **false
    negative** = failing to warn on a real confound the panel would catch.
- **Report the two directions separately, never merged into one F1
  without also stating the split**, so a reader can check whether the
  tool's actual error pattern matches its own stated design philosophy —
  a tool whose errors lean toward false positives (over-caution) is
  arguably *more* aligned with "refuse rather than mislead" than one whose
  errors lean toward false negatives, even at equal aggregate F1, and
  reporting only the aggregate would hide exactly the distinction that
  matters for this project's specific trust claim.
- **Track FP/FN rates as a regression gate for any future threshold or
  allowlist change**, extending the discipline `docs/tool-selection-eval.md`
  already established for wording changes (dated before/after entries,
  never silent) to numeric threshold changes: any change to
  `_SUDDEN_JUMP_Z_THRESHOLD` (currently 4.0), `_PLATEAU_CV_THRESHOLD`
  (currently 0.02), `_OSCILLATION_SIGN_FLIP_RATIO_THRESHOLD` (currently
  0.7), `ALLOWLIST_PARAMS` (currently `seed`, `device`, `run_name`,
  `run_id`, `name`, `id`), or the 10-run floor should report before/after
  FP/FN rates on the frozen ground-truth set (§3.7), not just a
  qualitative description of the change.

### 3.7 How should reproducibility be ensured? (Task Q7)

- **Version and freeze each benchmark release**, per HELM's and
  MLE-bench's precedent (§1.3, §1.1) — e.g. `experiment-audit-bench-v1`,
  with any change to cases, labels, or scoring logic requiring a new
  version number. A cited number should always be attributable to an
  exact, frozen artifact, never a moving target.
- **Publish the raw annotator labels, not just the aggregated majority
  vote**, so a third party can recompute inter-annotator agreement and
  challenge the aggregation method itself — the single most direct lesson
  from SWE-bench Verified's own PatchDiff experience (§1.2), where an
  outside team re-checking the *grading mechanism itself*, not just
  re-running the benchmark, found a real defect over a year after initial
  release.
- **Determinism is free here in a way it isn't for the benchmarks
  surveyed in Part 1.** MLE-bench and SWE-bench must engineer around
  agent non-determinism (different runs of the same agent can produce
  different patches); Experiment Audit's three judgment tools are already
  verified deterministic (`research-progress.md` §2 — "zero dependency on
  `backends/` or `server.py` anywhere in the analysis layer," pure
  functions of their inputs). Bit-identical output on repeated runs
  against the same fixture is a cheap, free-standing check this project
  can add and should not skip simply because it's easy — SWE-Bench++'s
  own methodology (§1.2, three independent-container reruns of the golden
  solution) treats this as a first-class validation step precisely
  because it can't be assumed.
- **A single documented command per metric**, following the pattern
  `scripts/record_wandb_fixtures.py` and `scripts/tool_selection_eval.py`
  already establish for this project (a runnable script producing a
  reportable artifact, its non-execution status disclosed rather than
  hidden) — no benchmark number should ever be reported without the exact
  command that reproduces it.

### 3.8 What adversarial cases must always be included? (Task Q8)

The existing six cases in `adversarial_cases.py` are necessary but
insufficient for this purpose (§3.1, category 1 vs 2) precisely because
they were engineered to have one unambiguous correct answer — they test
whether the implementation matches its own spec, not where the spec itself
might be wrong on real data. A genuine adversarial set for the calibration
benchmark needs cases engineered to find where each tool's *documented,
self-acknowledged* limitation actually bites, using the specific blind
spots this project's own code comments and literature review already name
— not novel failure modes invented for this document, but the ones the
codebase itself already flags as risks and has never tested against real
data:

- **`audit_ablation` — allowlist too narrow, real case:** a config diff
  outside `ALLOWLIST_PARAMS` that is genuinely reproducibility-irrelevant
  in practice (e.g. a logging directory path, a wandb run tag, a timestamp
  string) — tests whether the exact-match allowlist's conservative-by-
  design failure mode (documented in `confound.py`'s own comments, cited
  in `research-progress.md` §3) actually produces false `confounded`
  verdicts at a rate a skeptical researcher would notice.
- **`audit_ablation` — allowlist too permissive, real case:** a field named
  `seed` that is semantically *not* a model-init seed (e.g. a
  data-shuffling seed under active study) — tests whether the allowlist's
  implicit assumption ("a field literally named `seed` is always
  reproducibility-noise, never the thing being ablated") holds on real
  projects.
- **`audit_training_curve` — scheduled event misread as pathology:** a
  curve with a large but *intended* jump (an LR-warmup restart, a
  curriculum-stage transition) — tests whether `_SUDDEN_JUMP_Z_THRESHOLD =
  4.0` produces a false positive on "the code did exactly what it was
  supposed to."
- **`audit_training_curve` — real pathology below every threshold:** a
  genuine slow-onset issue (per the reward-hacking/specification-gaming
  literature already reviewed in `related-work.md` §5) that crosses none
  of the four fixed thresholds — tests whether the tool's own documented
  scope limit ("generic curve-shape heuristics, not a reward-hacking
  detector," per `related-work.md` §5) is something the benchmark can show
  concretely rather than only assert.
- **`audit_sweep` — structural vs. accidental covariance:** two
  hyperparameters correlated by mathematical construction rather than
  coincidence (e.g. `total_steps = batch_size × num_epochs`, deliberately
  covarying by design) — tests whether the covariance warning correctly
  distinguishes "this should arguably be modeled as one parameter" from
  "these two happened to move together in this particular sweep by
  accident," which are different findings a researcher would want to know
  apart.
- **Cross-tool — the panel itself disagrees:** at least one case per tool
  drawn directly from the `contested` bucket (§3.4) included in every
  benchmark run, scored not on whether the tool "got it right" (there is
  no single right answer by construction) but on whether the tool's
  `confidence` field appropriately reflects the genuine ambiguity — a
  direct, testable operationalization of design principle #3 ("refuse
  rather than mislead") that nothing in the current test suite checks,
  because the current suite's cases were all built to be unambiguous.

### 3.9 How should other researchers independently reproduce benchmark results? (Task Q9)

- **All three ground-truth sets (§3.3), the annotation protocol (§3.4),
  raw de-identified annotator labels, and the benchmark-runner code**
  published under `research/04_benchmarks/` and `tests/fixtures/` in this
  same repository — no separate, harder-to-find location.
- **Source real cases from a public, citable corpus (Open RL Benchmark,
  §1.5, §3.2) wherever possible**, specifically because a third party can
  pull the exact same W&B project and get bit-identical raw data — a
  benchmark built entirely on privately-held or unpublished research data
  cannot be independently checked by anyone outside the project, no matter
  how careful the internal process was.
- **One command reproduces one number**, per §3.7 — a reviewer should
  never have to reconstruct an undocumented multi-step process to verify a
  reported metric.
- **A documented ground-truth defect-reporting channel**, following
  MLE-bench's own precedent (§1.1) of disclosing known issues and
  addressing them via a versioned release rather than silent in-place
  correction — if an outside researcher finds a mislabeled case in the
  ground-truth set (the SWE-bench Verified PatchDiff scenario, §1.2, is
  the cautionary example of what happens when this channel doesn't exist
  or isn't used), the fix should be a new dated version, with the defect
  and its correction both stated, not a quiet edit.

### 3.10 What evidence would convince a skeptical researcher? (Task Q10)

Synthesizing everything above into the minimum bar, not an exhaustive
wish-list:

1. Ground truth from an **independent expert panel**, not the tool's own
   designers, with **published inter-annotator agreement**.
2. **Per-verdict-type precision/recall/F1 and a confidence-calibration
   check**, not one aggregate accuracy number.
3. A **human-baseline and unaided-Claude-baseline comparison on the same
   cases**, so the tool's number has an interpretable reference point —
   this is the single most persuasive structural element every credible
   benchmark surveyed in Part 1 shares in some form (Kaggle leaderboards
   for MLE-bench, real developer tests for SWE-bench, AblationBench's
   human-vs-model F1 comparison) and the one thing the current project's
   evidence base (`research-progress.md` §5, citing AblationBench's
   ~38–45% figure) already gestures at without yet measuring directly for
   its *own* tools.
4. **Explicitly disclosed contested/ambiguous cases**, not silently
   resolved by majority vote — visible evidence the benchmark isn't hiding
   its hardest cases to inflate a headline number.
5. **A versioned, frozen benchmark release** with raw data, annotation
   protocol, and runner code all public in this repository, so a skeptical
   reader doesn't have to trust the number — they can recompute it.
6. **Explicit adversarial "trap" cases (§3.8) reported alongside
   successes**, showing where the tool is known to fail, not only where it
   succeeds — the same standard `docs/audit-methods.md`'s existing honest
   "known limitations" framing already sets for the *code*, extended here
   to the *evaluation* of that code.

None of this exists yet. All of it is buildable using assets already in
this repository (`FakeBackend`'s pattern for case construction, the
`AdversarialCase` dataclass's structure, `scripts/record_wandb_fixtures.py`
for pulling real Open RL Benchmark data once network access exists) plus
one genuinely new piece of process this project has not needed before: a
human expert annotation protocol and panel.

---

## Part 4 — Is the existing testing strategy sufficient? Where exactly it is, and isn't

Directly answering the task brief's instruction to say so plainly where
existing designs already suffice, rather than manufacturing a gap:

**Sufficient, as-is, for what they claim to be:**
- `adversarial_cases.py` + `test_adversarial_mcp_layer.py` fully satisfy
  design-spec-v1.md §7 point 2's original list, for the purpose that list
  was written for — confirming the MCP layer round-trips the six named
  edge cases correctly. Nothing about this pass's findings suggests
  expanding this specific set or changing its role.
- `scripts/tool_selection_eval.py` + `scripts/tool_selection_prompts.py`
  are a methodologically sound design for category 3 (§3.1) once actually
  run — the distractor-pairing design (`docs/tool-selection-eval.md`)
  already reflects real thought about what a naive prompt set would miss.
  No redesign is recommended here; execution is the only blocker, and it's
  already named as the top priority in `research-progress.md` §11 item 1.

**Not sufficient, and cannot be made sufficient by more of the same kind
of work — this is the actual, specific gap this pass identifies:**
- No existing test, fixture, or script in this repository measures
  calibration against expert human judgment on real, ambiguous data
  (category 2, §3.1). This is not a matter of writing more unit tests —
  synthetic cases with an unambiguous correct answer by construction
  structurally cannot answer "would a real researcher agree with this
  verdict on real, messy data," no matter how many more of them are added.
  This requires the three ground-truth sets in §3.3, the annotation
  protocol in §3.4, and the human/baseline comparison in §3.5 — none of
  which currently exist in any form, planned or partial.

---

## Decision Impact

**What assumptions were confirmed?**
- The project's existing engineering-test discipline (recorded fixtures
  over hand-mocks, `AdversarialCase`'s spec-numbered structure,
  `docs/tool-selection-eval.md`'s "report the constraint, don't route
  around it" norm) is methodologically sound and directly compatible with
  the additional evaluation layer this document proposes — nothing here
  requires undoing prior work, only adding a category of evidence
  (human-labeled calibration data) the prior work never claimed to
  provide.
- `audit_ablation`'s priority ranking in
  `research/03_workflows/workflow-ranking.md` (validate first, strongest
  combined evidence base) is reinforced independently by this pass: it is
  also the tool with the clearest, most directly transferable
  ground-truth-construction precedent (SWE-bench Verified's
  clarity/correctness/solvability triad maps cleanly onto a binary
  clean/confounded verdict) of the three judgment tools.
- Open RL Benchmark, already named in `research-progress.md` §10 and
  `research/03_workflows/researcher-workflows.md` §6 as a friendly
  community entry point, is independently confirmed by this pass to also
  be the strongest available *data source* for the benchmark itself — the
  same corpus serves both the "who to talk to first" question (§11 item 2,
  unresolved) and the "what data grounds the ground-truth set" question
  this document answers.

**What assumptions were disproven?**
- The implicit assumption that the existing adversarial fixture set
  (`adversarial_cases.py`) plus a live run of the tool-selection eval
  would constitute "the benchmark" once both are executed does not
  survive this pass. Both are real, necessary, and already well-designed
  — but they answer "does the code do what it says" and "does the right
  tool get called," not "should a skeptical researcher trust the verdict
  itself on real data." That third question has no existing coverage at
  all, planned or partial, and cannot be answered by executing the
  existing scripts, no matter how successfully.
- The assumption (never stated outright in this project's own documents,
  but implicit in "run the two blocked scripts" being framed as the
  primary remaining evidence gap in `research-progress.md` §10–§11) that
  the two blocked-on-credentials scripts represent the *last* major
  evidence gap is disproven — they close two real gaps (fixture-shape
  validation, tool-selection accuracy) but leave the calibration gap this
  document identifies completely untouched, and closing that gap requires
  work (human annotation) neither script performs.

**What should we build because of this?**
- Nothing in this document should be built yet — per the task brief, this
  is a plan, not an implementation. If and when execution resumes, the
  concrete, prioritized order this document implies: (1) run the two
  already-written, credential-blocked scripts first (cheapest, per
  `research-progress.md` §11, unchanged); (2) pull a sample of real
  ablation-shaped and curve-shaped data from Open RL Benchmark (§1.5,
  §3.2) once network/credential access exists; (3) design and pilot the
  human annotation protocol (§3.4) on `audit_ablation` first, matching
  `workflow-ranking.md`'s own priority order; (4) only then build out the
  `audit_training_curve` and `audit_sweep` ground-truth sets.
- A recommendation to add exactly one new *type* of automated check that
  is cheap and requires no human labeling: a determinism check (§3.7,
  three independent reruns of each judgment tool against the same fixture
  confirming bit-identical output) — free to add given the tools' already-
  verified pure-function architecture, and closes a category of doubt
  (non-determinism) a skeptical reader might otherwise reasonably raise
  even though this project's architecture makes it a non-issue in
  practice.

**What should we stop building because of this?**
- Any temptation to treat "the tool-selection eval passed" or "all
  adversarial cases pass" as evidence that the judgment tools' verdicts
  are *correct on real data* should stop now, explicitly — this document
  exists specifically because that conflation was a live risk (the task
  brief's own framing, "this is NOT simply find benchmark papers,"
  presupposes exactly this risk) and nothing currently in the repository
  corrects it on its own.
- No new `audit_*` tool, threshold, or scoring mechanism should be added
  in the name of "improving the benchmark" — per the task brief's explicit
  constraint and this project's own established discipline
  (`research-progress.md` §7, §9), tuning without evidence is exactly the
  failure mode this whole research phase exists to prevent. Any future
  threshold change should be a response to a specific, cited FP/FN finding
  from the ground-truth set once it exists (§3.6), never a
  speculative improvement made in anticipation of one.
