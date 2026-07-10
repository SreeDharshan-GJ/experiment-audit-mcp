# Related Work — Experiment Validation, Reproducibility, and Judgment Tools

**Status:** First real pass. Supersedes nothing (this file, and every other file
in `research/02_literature/`, was empty before this document). Written from
independent search, not from claims in any prior briefing document.

**Scope note on method:** This synthesizes rather than abstracts. Each entry
below states what the work actually established, where it breaks, and what it
implies for Experiment Audit specifically — not what the paper's abstract
says about itself.

---

## 1. Reproducibility and variance in deep RL

**Henderson, Islam, Bachman, Pineau, Precup, Meger — "Deep Reinforcement
Learning that Matters" (AAAI 2018)**, with the companion **Islam, Henderson,
Gomrokchi, Precup (ICML RML workshop 2017)**.

- *Problem:* DRL results were (and often still are) reported as single
  numbers from single seeds, on codebases with undocumented hyperparameter
  differences, making "improvement" claims uninterpretable.
- *Approach:* Large controlled comparison across policy-gradient methods,
  varying only seeds, network width, reward scale, and codebase, measuring
  how much reported performance moves from each source alone.
- *Strengths:* Empirically demonstrated that seed variance alone can exceed
  the reported gap between a paper's proposed method and its baseline —
  a genuinely load-bearing result the field still cites.
- *Weaknesses / limitations:* Diagnostic, not prescriptive at the tooling
  level — the paper's recommendations ("report more seeds," "use
  significance tests") are process guidance for humans, not something a
  tool can enforce. It also predates any notion of an agent doing this
  auditing on a researcher's behalf.
- *Remaining gap:* Nothing in this line of work operationalizes "was this
  specific pair of runs a confounded comparison?" as a callable, structured
  judgment. It diagnosed the disease at the field level; it didn't build
  the field a stethoscope for one instance.
- *Relevance to Experiment Audit:* This is the closest thing to a literature
  justification for `audit_ablation` and `audit_training_curve` existing at
  all — it establishes that the failure modes are real and common. It does
  not validate that Experiment Audit's specific heuristics (exact-allowlist
  matching, threshold-based curve signals) are the right operationalization.
  That validation still doesn't exist anywhere, in this repo or elsewhere.

---

## 2. Statistical significance and multiple comparisons

**Dror, Baumer, Shlomov, Reichart — "The Hitchhiker's Guide to Testing
Statistical Significance in NLP" (ACL 2018)**; **Dror, Baumer, Bogomolov,
Reichart — "Replicability Analysis... Testing Significance with Multiple
Datasets" (2017)**; **Dror et al. — *Statistical Significance Testing for
NLP* (Synthesis Lectures, 2020)**.

- *Problem:* Deep learning comparisons routinely report "improvement"
  without any significance test, or use tests invalid for the non-convex,
  high-variance setting; multi-dataset/multi-run comparisons compound this
  into a multiple-comparisons problem almost no paper corrects for.
- *Approach:* Surveyed actual ACL/TACL papers (2017) and found significance
  testing "often ignored or misused"; proposed a decision protocol for
  which test fits which comparison shape, plus a replicability-analysis
  framework for multiple datasets (a form of Bonferroni/FDR-style
  correction specific to NLP evaluation).
- *Strengths:* This is a mature, field-adopted framework (also the basis of
  the `deep-significance` package) with a genuine "correct answer" for a
  known, common mistake.
- *Weaknesses:* NLP-benchmark-shaped — built around comparing test-set
  scores across datasets, not around auditing a sweep's hyperparameter
  correlations or a single ablation pair's config diff. The multiple-testing
  correction machinery here doesn't drop in directly to `audit_sweep`'s
  situation (ranking many hyperparameters' correlations from one sweep).
- *Remaining gap:* No literature specifically addresses correcting a
  hyperparameter-importance ranking for multiple comparisons in the way
  this body of work corrects cross-dataset benchmark comparisons.
- *Relevance:* `audit_sweep`'s docstring **already flags** that it applies
  no multiple-comparisons correction and explicitly declines to invent one
  silently — which is the right call per this literature's core lesson
  (don't report significance without correction), but it also means the
  tool is currently *aware of* the problem without *solving* it. This is a
  legitimate, literature-grounded v2 candidate: a defensible per-sweep FDR
  correction, not an ad hoc one.

---

## 3. Hyperparameter importance

**Hutter, Hoos, Leyton-Brown — "An Efficient Approach for Assessing
Hyperparameter Importance" (ICML 2014)** — the fANOVA paper — and its
production implementation, **`optuna.importance.FanovaImportanceEvaluator`**.

- *Problem:* Which hyperparameters actually drove an outcome, accounting for
  interactions between them, not just marginal correlation.
- *Approach:* Fits a random forest surrogate over the full trial history,
  then uses functional ANOVA to decompose variance attribution across
  individual parameters and pairwise interactions.
- *Strengths:* Captures non-linear and interaction effects Pearson
  correlation structurally cannot. Production-grade, actively maintained,
  free (ships in Optuna). Confirmed still current as of Optuna 4.9.
- *Weaknesses:* Requires enough trials to fit a reasonable forest (Optuna's
  own docs note performance/runtime tradeoffs at 1000+ trials); requires a
  `Study` object — i.e., it only exists for hyperparameter search actually
  conducted *through* Optuna. It has no notion of "audit a sweep I ran
  through some other system after the fact."
- *Remaining gap:* No general-purpose, system-agnostic fANOVA-quality
  importance tool exists for sweeps run outside Optuna (e.g., a manual grid
  logged to W&B, or a sweep run through Ray Tune / a custom script).
- *Relevance:* This is the most direct, most damaging comparison for
  `audit_sweep` to sit next to. Pearson-with-Fisher-z is real statistics,
  correctly implemented, but it is a strictly weaker tool than fANOVA on
  every axis except one: it works on *any* backend's sweep data, post hoc,
  with no re-run required. That one axis is the entire justification for
  `audit_sweep` existing — worth stating plainly in the tool's own
  documentation instead of only in this research file, since right now a
  user who knows Optuna could reasonably ask "why wouldn't I just use
  fANOVA," and the honest answer is "you should, if your sweep ran inside
  Optuna."

---

## 4. Ablation methodology — critique

**Lipton & Steinhardt — "Troubling Trends in Machine Learning Scholarship"
(2018)**; **Biderman & Scheirer — "Pitfalls in Machine Learning Research:
Reexamining the Development Cycle" (2020)**.

- *Problem:* Both are field-level critiques (not tools) documenting that
  papers routinely propose multiple simultaneous changes without ablating
  each one, obscuring which change actually produced the reported gain.
- *Approach:* Case-study-driven argument, not a method or benchmark.
- *Strengths:* Widely cited, credible naming of the exact failure
  `audit_ablation` targets — an ablation pair that differs in more than the
  claimed variable, silently misattributing the effect.
- *Weaknesses:* Purely diagnostic essays. Neither proposes any automated
  or semi-automated way to catch this at the point of comparison; both
  assume a human reviewer or author will simply do better.
- *Remaining gap:* Between "this is a known problem" (2018-2020 essays) and
  "here is a tool that catches it" there is essentially nothing published
  until very recently (see AblationBench, below) — and even that addresses
  a different half of the problem.
- *Relevance:* These are the correct citations for *why* `audit_ablation`'s
  problem is real, but they offer zero methodological guidance on *how* to
  detect it — Experiment Audit's allowlist approach is not derived from or
  validated by this literature; it's an independent design choice.

**Fostiropoulos et al. — "ABLATOR: Robust Horizontal-Scaling of ML Ablation
Experiments" (2023)**.

- *Problem:* Running large-scale ablation experiments (thousands of runs)
  is operationally painful — orchestration, not judgment.
- *Approach:* A framework for horizontally scaling ablation experiment
  *execution* across a single unified codebase per method.
- *Relevance:* Solves a different problem entirely — running ablations, not
  auditing whether an already-run pair was clean. Useful to name precisely
  so nobody conflates the two: Experiment Audit is downstream of ABLATOR's
  problem, not competing with it.

**Abramovich & Chechik — "AblationBench: Evaluating Automated Planning of
Ablations in Empirical AI Research" (2025, revised through mid-2026)**.

- *Problem:* Can an LM agent look at a paper's method section (or a
  reviewer's comments) and correctly identify *which* ablations should be
  run — i.e., ablation planning, not ablation judgment.
- *Approach:* Two benchmark tasks (`AuthorAblation`, `ReviewerAblation`)
  built from real ICLR submissions 2023–2025, with an LM-judge evaluation
  framework.
- *Strengths:* This is the most directly comparable published work to
  Experiment Audit's mission that exists, and it is very recent.
- *Weaknesses / result:* Best frontier-model performance identifies only
  ~38–45% of the ablations a human would flag, below human baseline
  (human F1 ~0.65 vs. best model ~0.42 on the subset measured). Chain-of-
  thought prompting outperformed an agent-based approach — a relevant,
  slightly uncomfortable data point for anyone assuming "just let the agent
  reason about it" is sufficient without structured tool support.
- *Remaining gap:* AblationBench evaluates *proposing* ablations from a
  paper description. It does not touch the problem `audit_ablation` solves
  — judging whether an ablation pair *that was actually run* is a clean
  test. These are adjacent but non-overlapping problems; conflating them
  would be a real mistake in Experiment Audit's own positioning.
- *Relevance:* This is direct, current evidence that "AI co-scientist"
  agents are *not yet good* at ablation reasoning unassisted — which is the
  strongest available argument for why a deterministic, non-LLM audit tool
  (that an agent calls rather than reasons about from scratch) has genuine
  value right now. This deserves to be cited in the project's own
  positioning material, not just this research file.

---

## 5. Training-curve pathology / reward hacking detection

**Multiple 2025-2026 papers** on reward hacking / specification-gaming
detection in RL (e.g. empirical detection-framework studies reporting
category-specific ROC-AUC in the 0.76-0.85 range using learned classifiers
over episode-level features, and adversarial-reward-auditing approaches
studying the temporal onset dynamics of hacking behaviors).

- *Problem:* Detecting when an RL policy is exploiting a misspecified
  reward (reward hacking / specification gaming) rather than solving the
  intended task — a training-curve pathology, but a much narrower and
  higher-stakes one than the four generic signals `audit_training_curve`
  detects.
- *Approach:* Learned detectors (classifiers trained on expert-labeled
  episodes) over behavioral and reward-trajectory features, evaluated by
  ROC-AUC against human-annotated ground truth — a fundamentally different
  methodological tier than fixed-threshold heuristics.
- *Strengths:* Demonstrates that reward hacking has recognizable temporal
  signatures (sudden vs. gradual onset), and that learned detection
  meaningfully outperforms naive baselines.
- *Weaknesses:* Requires labeled training data specific to the environment
  and hacking mode; not a general-purpose, backend-agnostic, zero-shot
  tool — the opposite tradeoff from `audit_training_curve`'s threshold
  approach.
- *Remaining gap:* No published work bridges "cheap, zero-training,
  general-purpose curve heuristics" (what Experiment Audit does) and
  "expensive, labeled, environment-specific learned hacking detectors"
  (what this literature does). That gap is real and currently unaddressed
  by either side.
- *Relevance:* `audit_training_curve`'s four signals (null values, sudden
  jump, plateau, oscillation) are generic curve-shape heuristics, not
  reward-hacking detectors, and the project's own roadmap correctly defers
  "RL-specific pathology signals (reward-hacking heuristics)" to v3. This
  literature confirms that's a real, hard, and separately-resourced
  problem — not a small extension of the current threshold logic.

---

## 6. Retrieval-layer competition (product, not literature, but load-bearing)

**W&B's official MCP server** (`wandb/wandb-mcp-server`, hosted at
`mcp.withwandb.com`) is real, actively maintained, and provides natural-
language query over runs, sweeps, Weave traces, and report generation.

- Confirmed independently (not merely asserted): it is retrieval- and
  reporting-oriented (`query_wandb_tool`, `create_wandb_report_tool`,
  trace queries) with no equivalent to `audit_ablation`, `audit_sweep`, or
  `audit_training_curve`'s judgment layer.
- *Relevance:* This validates Experiment Audit's stated non-goal ("don't
  compete on retrieval") independent of the claim having been asserted
  without evidence — it happens to be true. Worth re-checking periodically
  since W&B's own MCP server is actively developed and could add judgment-
  layer tools at any time; this is not a permanent moat.

---

## What assumption do all existing approaches make that Experiment Audit refuses to make?

Every body of work above — reproducibility studies, significance-testing
frameworks, fANOVA, ablation-planning benchmarks, learned reward-hacking
detectors — shares one implicit assumption:

**that the researcher (or the paper, or the reviewer) is the one doing the
noticing, and tooling exists to help them decide correctly once they've
already noticed something is worth checking.**

Henderson et al. tell you seeds matter; they don't tell you *this specific
pair of runs* has a seed confound. Dror et al. tell you which significance
test to use *once you've decided to test something*. fANOVA tells you
importance *once you've already decided to rank your hyperparameters*.
AblationBench evaluates whether an agent can *propose* the right ablation
from a paper's prose — closer, but still upstream of "was the ablation that
was actually run clean."

Experiment Audit's stated bet — implemented, if not yet validated — is the
opposite direction: **surface the check unprompted, from the data that
already exists**, rather than waiting to be asked the right question. That
is a real, currently-underserved position in the literature. It is also,
notably, an unproven bet: nothing in this review or in the repository
demonstrates that researchers *want* unprompted judgment over a dashboard
they already trust their own eyes on. That is the load-bearing open
question the next research phase needs to resolve with actual users, not
more literature.

---

## Prioritized list of unanswered questions

1. **(Highest priority, blocks everything else)** Do researchers actually
   want unprompted post-hoc judgment, or do they only want it once they've
   already suspected something is wrong? This determines whether
   Experiment Audit's core UX bet is right at all. Answerable only with
   real users, not more reading.
2. Do real W&B projects' config keys collide with `audit_ablation`'s exact
   allowlist often enough to matter (e.g., `random_seed` vs `seed`)? This
   is a five-minute check once `scripts/record_wandb_fixtures.py` is run
   against a real project — no new research needed, just execution of work
   already written.
3. Do the fixed thresholds in `audit_training_curve` (z=4.0, CV=0.02,
   sign-flip ratio=0.7) produce sane results on real reward curves and real
   loss curves, or were they tuned to nothing? Same category as #2 —
   blocked on live data, not on literature.
4. Is there a principled, defensible per-sweep multiple-comparisons
   correction for `audit_sweep` that doesn't require the NLP-benchmark
   machinery in §2 to be forced into a shape it wasn't designed for? This
   is a genuine open methodological question, not just an execution gap.
5. Given AblationBench's result that current LM agents are weak at ablation
   reasoning unassisted, is there a specific, valuable role for Experiment
   Audit as the tool an agent calls *before* attempting that reasoning
   itself, rather than only after a human asks it to check a specific pair?
   This is a product-design question raised directly by the literature,
   not present in the current design spec at all.
6. Is "RL-specific pathology detection" (v3, per the roadmap) actually
   reachable without the labeled-data cost the reward-hacking literature
   implies, or does the roadmap's own framing understate the resourcing
   this would require?
