# Experiment Audit — Research Progress (Living Memory)

**Purpose of this file:** a new person joining this project should be able
to read only this file and understand where things actually stand — not
where a plan once said they'd be. Where a claim below is unverified, it says
so explicitly. Last substantive update: this pass (validation-protocol
design — see §11b and the newly-populated `validation/` directory),
building on the prior pass (independent benchmark-design research, §11a
and `research/04_benchmarks/benchmark-plan.md`), which built on the pass
before that (independent workflow research, §6a and
`research/03_workflows/`), which itself built on the pass before that
(independent repo audit + first real literature review, §1–§5, §12).

---

## 1. What this project is

An MCP server (`experiment-audit-mcp`, v1.0.0) that gives an agent 8 tools
against a W&B project, split into two trust tiers: cheap deterministic
retrieval (`test_connection`, `list_runs`, `get_run_summary`,
`get_metric_history`, `compare_runs`) and heuristic judgment that always
carries `method` + `confidence` + `evidence` (`audit_training_curve`,
`audit_ablation`, `audit_sweep`). The mission, as stated by the project
itself: become "the world's most trusted experiment-analysis companion for
Claude and ML researchers" — explicitly not a W&B/MLflow replacement, not a
dashboard.

## 2. Engineering milestones (verified, not just claimed)

I re-ran the test suite and coverage myself rather than trusting the
README's numbers:

- **233 tests, all passing.** Confirmed by direct execution.
- **96% overall line coverage.** Confirmed. Weakest modules:
  `wandb_backend.py` (95%, mostly network-error edge cases),
  `server.py` (89%, mostly defensive branches).
- **Architecture claim ("retrieval and judgment are structurally
  separate") is real in the code**, not just asserted in docs — verified by
  reading `analysis/*.py` import statements directly: zero dependency on
  `backends/` or `server.py` anywhere in the analysis layer.
- Ten roadmap milestones, two logged spec revisions (`RunRef` gaining
  `entity`; `list_runs` gaining `page_size`) — both revisions have a
  documented flaw/fix/scope trail in `docs/design-spec-v1.md`, which is a
  good practice worth preserving as the project grows.
- Four non-negotiable design principles (`docs/design-spec-v1.md` §1):
  retrieval/judgment separation, judgment tools show their work, refuse
  rather than mislead, deterministic computation is the product. All four
  are actually enforced in the code, not just stated.

## 3. What each judgment tool actually does (verified by reading the algorithms, not the marketing copy)

- **`audit_ablation`** — exact-match allowlist (`seed`, `device`, `run_name`,
  `run_id`, `name`, `id`, case-insensitive) diffed against a claimed
  variable. Deterministic, not statistical. Self-documented limitation: a
  differently-named seed/device field (e.g. `random_seed`) will not be
  recognized and will count against the verdict — a documented, deliberate
  conservative-direction failure mode, not an oversight.
- **`audit_training_curve`** — four threshold-based signal detectors
  (null values, sudden jump via global-MAD z-score, low-variance plateau
  via coefficient of variation, oscillation via sign-flip ratio). All
  thresholds are fixed constants chosen at design time, not fit to any real
  curve (no live data has been available to fit them against).
- **`audit_sweep`** — Pearson correlation per hyperparameter, ranked, with
  Fisher z-transform significance testing (no scipy dependency) and a hard
  10-run floor applied twice (raw run count and usable-run count). Real,
  correctly implemented statistics — but Pearson is linear-only, and the
  module's own docstring flags that a non-monotonic hyperparameter (e.g. an
  interior-optimum learning rate) can rank falsely low. No
  multiple-comparisons correction is applied, and the module explicitly
  declines to add one silently rather than inventing an ad hoc one.

## 4. Product philosophy (as stated, and how much of it is actually followed)

Stated philosophy: software first, evidence second, research grows from
real user problems, no feature without evidence. **The engineering side
genuinely follows this** — every analysis module documents its own
limitations rather than hiding them, and the README's "Known Gaps" section
is honest about what hasn't been verified (no live W&B fixture recording,
no live Claude tool-selection eval, both blocked by sandbox network
constraints, not silently skipped).

**The research side has not yet followed this philosophy at all.** See §7.

## 5. Competitor findings (now independently verified, not merely asserted)

- **W&B's official MCP server is real**, actively maintained, hosted at
  `mcp.withwandb.com` and on GitHub (`wandb/wandb-mcp-server`). It is
  retrieval/reporting-oriented (query runs and Weave traces, generate
  reports/charts) with no equivalent to a judgment layer. This validates
  Experiment Audit's "don't compete on retrieval" positioning — correctly,
  as it turns out, but that correctness was previously asserted, not
  checked. Not a permanent moat: W&B's MCP server is actively developed and
  could add judgment tools at any time.
- **Optuna's `FanovaImportanceEvaluator` is real** and uses a random-forest
  fANOVA decomposition — genuinely stronger than `audit_sweep`'s Pearson
  ranking on every axis except one: it only works for sweeps run *through*
  Optuna. `audit_sweep`'s entire justification is "works on any backend's
  sweep, post hoc, no re-run required" — this should be stated explicitly
  in the tool's own docs, not just here.
- **New finding this pass, not in any prior briefing:** AblationBench
  (Abramovich & Chechik, 2025-2026) is the most directly comparable
  published work to this project's mission. It benchmarks LM agents
  *planning* ablations from a paper's method section — a different problem
  from `audit_ablation`'s *judging an already-run pair* — but it found
  current frontier models identify only ~38-45% of ablations a human would
  flag, below human baseline. This is strong evidence that "just let the
  agent reason about it" is not currently sufficient, which is the
  strongest available argument for a deterministic tool like this one
  existing — and it should be cited in the project's own positioning.

## 6. Literature findings (first real pass — see `research/02_literature/related-work.md` for full synthesis)

Covered: DRL reproducibility (Henderson et al.), statistical significance
and multiple-comparisons frameworks for NLP/deep learning (Dror et al.),
fANOVA hyperparameter importance (Hutter et al. / Optuna), ablation
methodology critiques (Lipton & Steinhardt; Biderman & Scheirer), ablation
infrastructure and planning (ABLATOR; AblationBench), and reward-hacking /
training-curve pathology detection (multiple 2025-2026 preprints).

**Core finding:** every existing body of work assumes the researcher (or
paper, or reviewer) is the one doing the noticing, and helps them decide
correctly *once they've already decided to check something*. Experiment
Audit's actual, implemented (if unvalidated) bet is the reverse: surface the
check unprompted, from data that already exists. That's a real gap in the
literature. It is also an unproven product bet — nothing in the literature
or in this repo demonstrates researchers want this.

## 6a. Workflow findings (this pass — see `research/03_workflows/` for full synthesis)

This pass populated `research/03_workflows/` (`researcher-workflows.md`,
`workflow-ranking.md`, `pain-points.md`), which was empty going into it,
same starting condition `02_literature/` was in before its own first pass.
Method: independent search across practitioner blogs, official W&B/CleanRL
documentation, GitHub issues and community-forum threads, arXiv methodology
sections, and one industry survey. No user interviews were conducted —
that remains the single biggest open item (see §7, unchanged by this pass).

**Core findings:**
- Five documented workflow archetypes were identified: the general
  dashboard-scan-by-eye loop (weakest evidence of rigor, but the most
  universal); RL-specific debugging against a consensus secondary-signal
  checklist (reward, policy/value loss, entropy, gradient norms — wider
  than `audit_training_curve`'s current scope); sweep-then-read-the-
  importance-panel; ablation studies (confound risk is real, named, and
  now venue-enforced); and reproducibility-first tracked-experiment
  culture in open-source RL (CleanRL / Open RL Benchmark), which is the
  friendliest entry point for backend-agnostic post-hoc judgment but a
  narrower audience than "ML researchers" broadly.
- **New, unflattering finding:** W&B's own sweep "parameter importance"
  panel already uses a random-forest-based method, not a linear one,
  specifically to capture non-linear and interaction effects — the exact
  blind spot `audit_sweep`'s Pearson-only approach has. This does not
  invalidate `audit_sweep`'s actual justification (backend-agnostic,
  works without going through W&B's sweep infrastructure, no re-run
  required) but it does mean `audit_sweep` should not be described or
  implied anywhere as statistically comparable to what W&B already ships
  for in-W&B sweeps.
- Of the three judgment tools, `audit_ablation` has the strongest combined
  evidence base: independent methodology literature, venue-level
  enforcement (NeurIPS/ICML checklists), *and* the AblationBench benchmark
  result (§5) directly measuring that LLMs alone are insufficient at the
  task. Neither `audit_training_curve` nor `audit_sweep` has an equivalent
  direct benchmark showing LLM-alone reasoning is insufficient at their
  specific task — this is a real asymmetry in the evidence base across the
  three tools, not just a difference in polish, and
  `research/03_workflows/workflow-ranking.md` ranks the three accordingly.
- Reddit-specific search, called for explicitly in the task brief, returned
  no usable results in this pass. Not concluded to be absent — flagged as
  a method limitation (worth a different search strategy or an archive
  query) rather than silently dropped.

## 7. Assumptions validated vs. still lacking evidence

**Validated this pass (independently, not merely asserted):**
- W&B has an official MCP server, and it's retrieval-only. (Confirmed via
  search.)
- Optuna's fANOVA is real and statistically stronger than Pearson-based
  ranking for in-Optuna sweeps. (Confirmed via search.)
- The engineering claims (test count, coverage, architecture separation)
  are true. (Confirmed by direct execution and code reading.)
- The dashboard-scan-by-eye loop, with no structural check against
  confounds or curve pathologies, is the documented default workflow
  across general ML and RL tooling alike. (Confirmed via search — see
  `research/03_workflows/researcher-workflows.md` §2.)
- Confounded ablations are a real, recurring, and now venue-enforced
  (NeurIPS/ICML checklist-level) failure mode, independent of this
  project's own framing. (Confirmed via search — §5 of the same document.)
- W&B's own sweep parameter-importance panel uses tree-based (random
  forest) importance, not linear correlation — materially stronger than
  `audit_sweep`'s Pearson-based approach for the in-W&B case. (Confirmed
  via W&B's own documentation — §4 of the same document.)

**Still lacking evidence — this is the important list:**
- **The entire research/ directory was empty before this pass** — 17
  files across 6 planned phases (vision, landscape, literature, workflows,
  benchmarks, moonshots), all 0 bytes. A prior briefing document described
  a "landscape study already completed" with specific competitor findings.
  Those specific findings happen to check out (see §5), but they were
  asserted, not evidenced, in this repository. This is the single most
  important finding of this pass: the project's stated discipline
  ("evidence second," "no feature without evidence") was not, until now,
  being applied to its own research claims.
- Whether `audit_ablation`'s exact-match allowlist covers real projects'
  naming conventions (`random_seed` vs `seed`, etc.) — untested; blocked
  only on running `scripts/record_wandb_fixtures.py` against a real
  project, which is already written and ready.
- Whether `audit_training_curve`'s fixed thresholds (z=4.0, CV=0.02,
  sign-flip ratio=0.7) produce sane verdicts on real curves — untested,
  same blocker.
- Whether researchers actually want unprompted post-hoc judgment (the
  project's core UX bet) rather than only on-demand checks — completely
  untested, and not answerable by more reading; needs real users.
- Whether a principled multiple-comparisons correction for `audit_sweep`
  is worth building — open methodological question, not just an execution
  gap (see literature review §2, §4).
- Whether researchers who work outside RL and outside the
  reproducibility-first open-source culture documented in
  `research/03_workflows/researcher-workflows.md` §6 (e.g. NLP or CV labs
  running supervised fine-tuning sweeps) show the same workflow patterns —
  this pass's evidence skews RL-heavy because RL tooling and practitioner
  writing happened to be the most documented and searchable; that skew is
  a property of what's publicly written, not necessarily of the target
  user base, and should not be read as "this project is really an RL
  tool."
- Whether `audit_training_curve` should eventually cover entropy-collapse
  and gradient-norm signals, given RL practitioners treat these as equally
  load-bearing as the headline metric for diagnosing policy collapse — new
  scope question from this pass, not yet a decision either way (see
  `research/03_workflows/pain-points.md` §3).

## 8. Current differentiation (restated precisely, post-review)

Not "better statistics than Optuna" (it isn't, for in-Optuna sweeps). Not
"better retrieval than W&B's own MCP server" (not the goal, and W&B's is
good). The actual, defensible differentiation: **post-hoc, backend-agnostic
judgment over data that already exists, unprompted, with method/confidence/
evidence always attached** — a niche current literature confirms is real
(AblationBench's numbers) and currently unserved by both the academic tools
(fANOVA, significance-testing frameworks — all require the researcher to
already be running the check) and the product competitors (W&B MCP —
retrieval only).

## 9. Current risks

- **The core UX bet is unvalidated.** If researchers don't want unprompted
  judgment over their own dashboard-reading, the whole differentiation
  collapses regardless of how correct the statistics are.
- **Fixed thresholds were chosen, not fit.** Every threshold in
  `audit_training_curve` is a guess until tested against live data.
- **W&B's own MCP server is not static.** If it grows a judgment layer,
  the "don't compete on retrieval" positioning needs revisiting.
- **The research-discipline gap (§7) could recur.** Nothing structurally
  prevents future "research already done" claims from being asserted again
  without evidence — worth a norm, not just this one correction.
- **`audit_sweep` risks overclaiming relative to what W&B already ships.**
  Now that this pass has confirmed W&B's importance panel uses a
  statistically stronger (tree-based) method than `audit_sweep`'s Pearson
  approach, any future documentation, marketing, or positioning that
  implies `audit_sweep` is "better" sweep analysis rather than
  "differently-scoped, backend-agnostic" sweep analysis would be a
  regression from the honesty this project has otherwise maintained.

## 10. Current opportunities

- `scripts/record_wandb_fixtures.py` and the tool-selection eval script are
  already written and ready — the two biggest evidence gaps in the whole
  project are an execution problem, not a research problem, and could close
  fast once run against a real project (e.g. the user's own MAMFAC/CARM++
  work, per the design spec's own suggestion).
- AblationBench's result (agents are weak at ablation reasoning unassisted)
  is a genuinely strong, current, citable argument for this project's value
  proposition that isn't being used anywhere in the project's own
  documentation yet.
- A principled multiple-comparisons correction for `audit_sweep` is a real,
  literature-grounded v2 candidate — not just an ad hoc nice-to-have.
- `research/03_workflows/workflow-ranking.md` now gives a defensible
  priority order (`audit_ablation` > `audit_training_curve` > `audit_sweep`)
  for which tool to validate against live data first if time is
  constrained, based on strength of problem evidence and durability of
  differentiation — useful input whenever `04_benchmarks/` work resumes.
- CleanRL / Open RL Benchmark's tracked-experiment culture
  (`research/03_workflows/researcher-workflows.md` §6) is a concretely
  identified, well-documented community where backend-agnostic post-hoc
  judgment would be easiest to pilot first, if and when real-user
  validation (§11 item 2, still open) moves forward.

## 11. Next recommended research steps (in priority order)

1. Run `scripts/record_wandb_fixtures.py` against a real W&B project. This
   closes the biggest, cheapest evidence gap and would let §7's two
   "still lacking evidence" allowlist/threshold questions get answered with
   real data instead of guesses.
2. Talk to a small number of real researchers (even 3-5) about whether they
   would want unprompted post-hoc judgment vs. on-demand checks. This is
   the single highest-leverage unanswered question in the whole project and
   cannot be resolved by more reading. If time is constrained, prioritize
   recruiting from the CleanRL / Open RL Benchmark tracked-experiment
   community first (`research/03_workflows/researcher-workflows.md` §6) —
   the friendliest-evidenced entry point — rather than a general ML
   researcher population, to avoid conflating "nobody wants this" with
   "this pass under-sampled the right audience."
3. ~~Populate `research/01_landscape/`, `research/03_workflows/`, and
   `research/04_benchmarks/`~~ — **`research/03_workflows/` and
   `research/04_benchmarks/` are now done** (`researcher-workflows.md`,
   `workflow-ranking.md`, `pain-points.md` from the prior pass;
   `benchmark-plan.md` from this pass). **`research/01_landscape/` remains
   the one directory still empty** and still needs the same evidence
   discipline applied — see §11a's third finding for why this specific
   gap is now named explicitly rather than left implicit.
4. Decide whether a literature-grounded multiple-comparisons correction for
   `audit_sweep` is worth designing now or genuinely belongs in v2/v3 —
   this is a real methodological decision, not busywork.
5. Re-check W&B's MCP server periodically for judgment-layer features —
   the current differentiation argument depends on it staying
   retrieval-only.
6. Update `audit_sweep`'s own docstring/README language to state the
   Pearson-vs-tree-based-importance tradeoff explicitly, now that a
   concrete comparison point (W&B's shipped panel) exists — a small,
   low-cost honesty fix identified by this pass
   (`research/03_workflows/pain-points.md` §4), not blocked on anything.
7. ~~When `04_benchmarks/` work resumes, validate `audit_ablation`
   first~~ — **superseded by a more specific plan.** §11a and
   `research/04_benchmarks/benchmark-plan.md` §Part 3 now specify exactly
   what "validate `audit_ablation` first" requires: (a) real ablation-
   shaped run pairs pulled from Open RL Benchmark (blocked on the same
   network/credential constraint as item 1); (b) a human expert annotation
   protocol producing `clean`/`confounded`/`contested` labels from an
   independent panel (net-new process, not yet designed in operational
   detail beyond the plan document); (c) an unaided-Claude baseline run on
   the same cases. None of these three has been executed.
8. Design and pilot the human annotation protocol from
   `benchmark-plan.md` §3.4 on a small sample (e.g. 10-15 cases) before
   committing to the full three-tool ground-truth build — cheapest way to
   find protocol problems (unclear instructions, poor inter-annotator
   agreement) before investing in the full annotation effort per tool.
9. Add the bit-identical-output determinism check named in
   `benchmark-plan.md` §3.7 — free given the judgment tools' already-
   verified pure-function architecture (§2), blocked on nothing, and
   closes a category of doubt (non-determinism) independent of the larger
   annotation effort.

## 11a. Benchmark-design findings (this pass — see `research/04_benchmarks/benchmark-plan.md` for full synthesis)

This pass populated `research/04_benchmarks/benchmark-plan.md`, which was
empty going into it — the same starting condition every other
`research/` file was in before its own first pass. Method: independent
search across ML-engineering agent benchmarks (MLE-bench), software-agent
benchmarks (SWE-bench/SWE-bench Verified), holistic LLM evaluation
frameworks (HELM), LLM-application eval tooling (OpenAI Evals), an RL
tracked-experiment corpus (Open RL Benchmark/CleanRL), and community
reproducibility initiatives (ML Reproducibility Challenge/Papers with
Code) — plus a fresh read of this repository's own existing test
infrastructure (`tests/fixtures/adversarial_cases.py`,
`scripts/tool_selection_eval.py`, `docs/tool-selection-eval.md`,
`docs/design-spec-v1.md` §7) against that literature.

**Core finding, stated plainly:** this project's existing testing
strategy answers two questions well — "does the code do what its own spec
says" (233 passing tests, six spec-numbered adversarial MCP-layer cases)
and, once actually run, "does an MCP client invoke the right tool from a
prompt" (`scripts/tool_selection_eval.py`, not yet executed, same
credential/network blocker as `scripts/record_wandb_fixtures.py`). It
answers a third, more important question — **"would an independent panel
of ML researchers agree with this tool's verdict on real, ambiguous
data?"** — **not at all.** No fixture, script, or test in this repository
currently measures calibration against human judgment on real data, and
this cannot be fixed by writing more cases in the same style as the
existing six: those six were deliberately constructed to have one
unambiguous correct answer (that's what makes them good regression
tests), which is exactly the property real, contested research data
doesn't reliably have. This is a different category of evidence gap than
anything named in §7 or §9 so far, and closing it requires a genuinely new
kind of work this project has not needed before: an independent human
expert annotation protocol producing a ground-truth set, not more
engineering.

**New, concrete, actionable finding not previously in this project's
evidence base:** Open RL Benchmark (Huang et al., 2024) — already named in
§10 and `research/03_workflows/researcher-workflows.md` §6 as a friendly
*community* for eventual user outreach — is independently confirmed by
this pass to also be the strongest available *data source* for building
the ground-truth benchmark itself. It is public, community-maintained,
stores more than 25,000 tracked runs with full configs, frozen
dependency versions, and exact reproduction commands, and is hosted on
W&B — the exact backend `WandbBackend` already speaks to. It supplies real
ablation-shaped and sweep-shaped cases at no data-collection cost; it does
not supply labels (confounded/clean, pathological/normal), which still
require the human annotation process named above.

**Second finding, a direct methodological warning against a specific
future mistake:** OpenAI Evals' own documentation is explicit that
model-graded evaluation (using an LLM to judge whether another LLM's
output was "reasonable") has a real, disclosed error rate and needs
human-validation before being trusted. Experiment Audit's three judgment
tools are already deterministic — no LLM in the loop at inference time,
per §2's verified architecture claim — which is a genuine structural
advantage most of the benchmarks surveyed don't have. That advantage would
be **thrown away** if a future benchmarking pass used an LLM-as-judge
shortcut instead of real human-labeled ground truth to validate these
specific tools. This is now an explicit, named risk this project should
watch for, not just a hypothetical.

**Third finding, an honest correction to this project's own task framing,
not a substantive research result:** the task brief that initiated this
pass listed "competitor landscape" as an already-completed research phase.
`research/01_landscape/`'s three files are, as of this pass, still 0
bytes — unchanged since before even the first `research-progress.md` pass.
The *substance* of a competitor-landscape review does exist (§5 above, and
`research/02_literature/related-work.md` §6 — W&B's MCP server, Optuna's
fANOVA), so no research was actually skipped, but it was never written to
the location this project's own directory structure promises it lives in.
This is named here rather than silently accepted, in the same spirit as
§12's provenance discipline, and is **not** corrected in this pass (out of
scope — this pass's mandate was `04_benchmarks/` only) — but it should not
be treated as resolved just because it's now been said once.

**What this pass explicitly did not do, so it isn't mistaken for having
been done:** it did not build any ground-truth dataset, did not recruit or
run a human annotation panel, did not pull real data from Open RL
Benchmark (blocked on the same network/credential constraint as
`scripts/record_wandb_fixtures.py`), and did not write any new code,
fixture, or tool. Per the task brief's own explicit constraint, this pass
produced a *plan* for how evaluation should work, not the evaluation
itself. `research/04_benchmarks/benchmark-plan.md` §Part 3 lays out what
that plan requires in enough operational detail (annotation protocol,
metric definitions, adversarial trap cases, versioning scheme) that a
future execution pass could act on it directly, but "actionable plan"
and "completed benchmark" are not the same claim, and this document does
not conflate them.

## 11b. Validation-protocol design findings (this pass — see `validation/` for full package)

This pass populated all five files scaffolded (0 bytes each) by commit
`ff494be`, "Create validation framework": `validation/validation-plan.md`,
`validation/capability-matrix.md`, `validation/datasets.md`,
`validation/annotation-guidelines.md`. (`validation/README.md` remains
0 bytes — out of scope for this pass's explicit four-file mandate, named
here rather than silently left, in the same spirit §11a already
established for `research/01_landscape/`.) Method: this pass did **not**
re-run the literature/methodology research `research/04_benchmarks/
benchmark-plan.md` already completed. It read that document and
`research/research-progress.md` in full first, per the task brief's
explicit instruction not to repeat previous research, and then did the
work the benchmark plan's own Part 4 identified as still missing:
turning a designed methodology into an executable protocol — phases,
gates, roles, decision criteria, a concrete per-tool FP/FN specification,
dataset sourcing/sizing/versioning, and an annotator-facing labeling
rubric a person who has never read the benchmark plan could pick up and
follow.

**Why the protocol was designed this way — the decisions this pass had to
make that the benchmark plan left open:**

- **Calibration and utility were split into two explicitly separate
  questions (`validation-plan.md` §1), not scored together.** The
  benchmark plan's own scope was calibration only (its Part 2 explicitly
  frames the gap as "calibration against expert human judgment on real
  data"). This pass judged that calibration evidence alone cannot answer
  the task brief's actual mission question — "does Version 1 genuinely
  help ML researchers" — because a tool can be well-calibrated and still
  not be something a researcher wants surfaced unprompted (the core UX
  bet `research-progress.md` §7/§9 already named as the single largest
  unvalidated assumption in the whole project, independent of anything
  about calibration). Folding a small, necessarily qualitative utility
  track (`validation-plan.md` §5 Phase 5) into the same statistical
  report as the calibration numbers would have either diluted the
  calibration rigor or manufactured false quantitative precision from
  3-5 interviews. Keeping them structurally separate, with both required
  before a tool counts as "validated" (§7's decision criteria), was the
  most defensible way found to honor both halves of the mission
  statement without letting either one substitute for the other.
- **Per-tool, asymmetric FP/FN targets, not one shared bound.** The
  benchmark plan named the *direction* of asymmetry per tool (§3.6) but
  did not commit to numbers — deliberately, since it predates any real
  data. This pass assigned concrete recommended-default percentages
  (`capability-matrix.md`) so the protocol is actually executable and a
  future report has something to check its results against, while
  stating explicitly (`validation-plan.md` §10) that these are
  recommended defaults to be confirmed or revised at the Phase 2 pilot
  gate, not numbers this pass has evidence to fix precisely. The
  alternative — leaving the acceptable-rate fields blank until real data
  exists — would have technically satisfied the task brief's checklist
  without producing a usable protocol; a number stated as provisional and
  revisable is more useful than no number, provided the provisional
  status is never dropped silently.
- **A hard gate between piloting the annotation protocol and scaling it**
  (`validation-plan.md` §5 Phase 2→3), not just a recommendation. This
  was judged necessary, not optional, because the single biggest risk
  this whole package introduces beyond what the benchmark plan already
  covers is a genuinely new failure mode for this project: a
  ground-truth set built on a flawed protocol would *look* authoritative
  (published kappa, published raw labels, versioned) while actually being
  wrong, and nothing about versioning or publishing raw labels catches a
  bad protocol on its own — only a pilot-then-gate structure does. This
  is why `annotation-guidelines.md` §6 states explicit kappa bands with a
  "do not proceed" tier, not just a target to aim for.
- **Retrieval tools and tool-selection accuracy were kept in the
  capability matrix as explicit category-1/category-3 rows with `N/A`
  FP/FN cells, rather than omitted.** The task brief asked for every
  existing capability to have every field defined. Forcing invented FP/FN
  numbers onto a deterministic schema-translation function would have
  technically filled the cell while manufacturing false rigor — the exact
  failure mode `validation-plan.md` §4 principle 5 (inherited from
  benchmark plan §3.1) warns against. Marking the field `N/A` with a
  stated reason was judged more honest than a number with no meaning
  behind it, and still satisfies "every capability is addressed" — just
  not uniformly.

**What assumptions remain untested after this pass:**

- **Every numeric target in `capability-matrix.md` (FP/FN ceilings) and
  `validation-plan.md` §7 (the 15-percentage-point calibration margin,
  the ≥90%/≥75% tool-selection bars) is a design-time judgment call, not
  yet checked against any real data** — exactly the same status
  `research-progress.md` §3 already describes for `audit_training_curve`'s
  own thresholds. This pass did not, and could not, resolve that; it
  explicitly names each number as provisional (`validation-plan.md` §10)
  rather than presenting a false sense of precision.
- **Whether 3 annotators per case is sufficient**, versus needing more for
  a stable kappa estimate, especially on the smaller `audit_sweep` dataset
  (`datasets.md` §4's 15-25 sweeps, the smallest of the three sets) —
  untested; the Phase 2 pilot is specifically designed to surface this
  before the full run commits resources, but the pilot itself hasn't run.
- **Whether the stratification targets in `datasets.md` §2-§4 (e.g. "~40%
  obvious / ~40% moderate / ~20% adversarial" for `audit_ablation`) are
  achievable from Open RL Benchmark's actual real-world case mix** —
  this pass set stratification targets based on what a well-designed
  ground-truth set should look like (per the benchmark plan's synthesis
  of SWE-bench Verified and MLE-bench's own dataset-design lessons), not
  from having already inventoried what Open RL Benchmark's corpus
  actually contains case-by-case. If the real corpus doesn't naturally
  split this way, Phase 1 will need to either accept a different mix
  (disclosed, not silently substituted) or supplement with more
  hand-constructed trap cases than currently anticipated.
- **Whether the recruited annotation panel will actually be independent
  of this project** in practice (§1's requirement) is a staffing
  question this pass cannot verify in advance — it can only state the
  requirement and the reason for it.
- **The entire utility track (Phase 5) is unvalidated as a *method*, not
  just unexecuted** — this pass designed a lightweight qualitative
  interview structure because `research-progress.md` §11 item 2 already
  named "talk to 3-5 researchers" as the highest-leverage next step, but
  whether unstructured qualitative notes from 3-5 people can actually
  produce a defensible signal about a product-adoption question is itself
  an open methodological question this pass did not attempt to resolve
  beyond stating the limitation (`validation-plan.md` §10).

**What evidence would be required before Version 2 begins** (directly
answering the task brief's closing instruction, stated here rather than
left implicit):

1. **At minimum, one judgment tool** (`audit_ablation`, per the existing
   priority ranking) **must complete Phases 0 through 4** of
   `validation-plan.md` §5 with a published report following §8's
   template, including a pilot that passed Phase 2's kappa gate — not a
   partial run, not a report that skips the unaided-Claude baseline or
   the adversarial trap cases.
2. **The calibration bar and differentiation bar in `validation-plan.md`
   §7 must both be evaluated and reported, whether met or not.** A
   result showing the tool fails one or both bars is sufficient evidence
   to *inform* a Version 2 decision (e.g. "the allowlist needs revision"
   or "this tool doesn't clear the bar unaided-Claude already clears") —
   it is not evidence that no decision can be made. What would actually
   block a responsible Version 2 decision is the *absence* of this
   evidence, not a negative result within it.
3. **At least a preliminary utility signal from Phase 5**, even a small
   one, showing whether researchers want unprompted judgment versus
   on-demand checks — because a Version 2 built on calibration evidence
   alone could optimize a tool nobody wants used the way it currently
   works. This is the same unresolved core-UX-bet risk `research-progress.md`
   §9 has flagged since before this pass and remains flagged after it.
4. **No threshold, allowlist, or scoring change should be made in
   anticipation of a benchmark result** — any change made *in response* to
   a specific, cited FP/FN finding from an executed (not merely designed)
   ground-truth set is legitimate Version 2 input; a change made because
   this pass's protocol exists, before it has actually been run against
   real data, would violate the evidence discipline this entire project
   has tried to hold itself to since §7 first named its own gap.
5. **This document's own claims about the protocol's design should be
   re-checked once Phase 2 actually runs** — specifically, whether the
   kappa bands in `annotation-guidelines.md` §6 and the FP/FN targets in
   `capability-matrix.md` turn out to be achievable or need revision. A
   protocol that looks sound on paper and a protocol that survives
   contact with real annotators and real data are different claims, and
   only the second is sufficient grounding for Version 2 work.

**What this pass explicitly did not do, so it isn't mistaken for having
been done:** it did not source or freeze any dataset, did not recruit or
run any annotation panel (pilot or full), did not compute any real
inter-annotator agreement number, did not run the two credential-blocked
scripts (`record_wandb_fixtures.py`, `tool_selection_eval.py`), and did
not analyze any experiment. Per the task brief's explicit instruction,
this pass produced the *protocol*, not the validation itself — the same
distinction `research/04_benchmarks/benchmark-plan.md` §Part 3's closing
paragraph already drew between "actionable plan" and "completed
benchmark," now one level more operational but still on the plan side of
that line.

## 12. Explicit note on this document's own provenance

This file, and `research/02_literature/related-work.md` and
`bibliography.md`, were written after independently verifying the repo's
code and test claims and independently researching the literature — not by
trusting a prior briefing document's narrative about what had already been
established. That briefing document's specific factual claims turned out to
be correct on inspection (see §5), but the discipline of *not* trusting
them until checked is itself the most important thing to preserve going
forward.

**The prior pass (§6a and `research/03_workflows/`)** continued that same
discipline: every claim in `researcher-workflows.md`, `workflow-ranking.md`,
and `pain-points.md` is sourced from independent search conducted during
that pass, not carried over from assumption or from this project's own
prior framing of its differentiation. Where evidence was genuinely not
found (e.g. Reddit-specific discussion, ML-research-specific team review
practices), that is stated as a method limitation in
`researcher-workflows.md` §7, not silently omitted. The evidence base
still skews toward RL and toward practitioners organized enough to write
publicly — that skew is named explicitly in §7 above and should be
corrected by real user interviews (§11 item 2), not by more reading alone.

**The prior pass (§11a and `research/04_benchmarks/benchmark-plan.md`)**
applied the same discipline to benchmark-design literature: every
methodology claim about MLE-bench, SWE-bench/SWE-bench Verified, HELM,
OpenAI Evals, Open RL Benchmark, and the ML Reproducibility Challenge is
sourced from independent search conducted during that pass, not asserted
from prior familiarity with these names. That pass also explicitly named
one place where the task brief's own framing didn't match the repository's
actual state (the "competitor landscape... already completed" claim vs.
`research/01_landscape/`'s still-empty files, §11a's third finding) —
continuing, rather than repeating for the first time, the norm §7
established of checking claims about this project's own status before
building on them, including claims made by the task brief that initiates
a given pass.

**This pass (§11b and `validation/`)** did not do independent literature
research at all — deliberately, per the task brief's explicit "do not
repeat previous research" instruction, and because the benchmark plan had
already done that work. What this pass verified independently instead was
the *state of the repository itself*: that all five `validation/` files
were genuinely 0 bytes before this pass (confirmed by direct inspection
and by reading commit `ff494be`'s diff, which shows five empty-file
additions), rather than trusting the task brief's framing that validation
work was starting from scratch. It also re-read every claim in this file
and in `benchmark-plan.md` that this pass's protocol design decisions
depend on (the three judgment tools' documented failure modes, the exact
threshold constants, the existing adversarial case set's scope) directly
from the source code (`analysis/confound.py`, `analysis/divergence.py`,
`analysis/sensitivity.py`, `tests/fixtures/adversarial_cases.py`) rather
than from this document's own prior summaries of them, so that
`capability-matrix.md`'s per-tool failure-mode lists are traceable to the
actual code comments and docstrings that name them, not to a paraphrase
of a paraphrase. Continuing the same norm §7 and §11a established: nothing
in this pass's design decisions should be taken as validated just because
it is now written down in a structured document — §11b's own "what
remains untested" list exists specifically so this pass's own output
doesn't quietly become the next thing future work trusts without
checking.
