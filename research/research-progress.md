# Experiment Audit — Research Progress (Living Memory)

**Purpose of this file:** a new person joining this project should be able
to read only this file and understand where things actually stand — not
where a plan once said they'd be. Where a claim below is unverified, it says
so explicitly. Last substantive update: this pass (independent workflow
research — see §6a and `research/03_workflows/`), building on the prior
pass (independent repo audit + first real literature review, §1–§5, §12).

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
   `research/04_benchmarks/`~~ — **`research/03_workflows/` is now done**
   (`researcher-workflows.md`, `workflow-ranking.md`, `pain-points.md`,
   this pass). `research/01_landscape/` and `research/04_benchmarks/`
   remain empty and still need the same evidence discipline applied.
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
7. When `04_benchmarks/` work resumes, validate `audit_ablation` first —
   it has the strongest combined evidence base of the three judgment
   tools per `research/03_workflows/workflow-ranking.md`.

## 12. Explicit note on this document's own provenance

This file, and `research/02_literature/related-work.md` and
`bibliography.md`, were written after independently verifying the repo's
code and test claims and independently researching the literature — not by
trusting a prior briefing document's narrative about what had already been
established. That briefing document's specific factual claims turned out to
be correct on inspection (see §5), but the discipline of *not* trusting
them until checked is itself the most important thing to preserve going
forward.

**This pass (§6a and `research/03_workflows/`)** continued that same
discipline: every claim in `researcher-workflows.md`, `workflow-ranking.md`,
and `pain-points.md` is sourced from independent search conducted during
this pass, not carried over from assumption or from this project's own
prior framing of its differentiation. Where evidence was genuinely not
found (e.g. Reddit-specific discussion, ML-research-specific team review
practices), that is stated as a method limitation in
`researcher-workflows.md` §7, not silently omitted. The evidence base
still skews toward RL and toward practitioners organized enough to write
publicly — that skew is named explicitly in §7 above and should be
corrected by real user interviews (§11 item 2), not by more reading alone.
