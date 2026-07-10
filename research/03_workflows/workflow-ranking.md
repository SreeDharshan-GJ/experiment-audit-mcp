# Workflow Ranking — Where Should Experiment Audit's Judgment Insert First?

**Status:** First real pass, companion to `researcher-workflows.md` and
`pain-points.md`. This document ranks the five workflow archetypes
identified in `researcher-workflows.md` against the project's own stated
evaluation criteria (mission statement in the task brief and
`research-progress.md` §4): does the workflow solve a real problem,
increase trust, provide reasoning Claude alone cannot reliably perform,
and have long-term differentiation. Ranking is qualitative, not a scored
model -- there is not enough evidence yet (no user interviews, no live
usage data) to justify false numerical precision. Where confidence is low,
that is stated rather than papered over with a score.

---

## Ranking criteria, restated concretely

For each workflow archetype, this document asks four questions drawn
directly from the mission statement:

1. **Real problem?** Is there independent evidence (not just this
   project's own framing) that this workflow produces mistakes researchers
   care about?
2. **Trust increase?** Would inserting evidence-backed judgment here make
   a researcher trust their own conclusion more, specifically because the
   judgment shows its work (method + confidence + evidence)?
3. **Reasoning Claude can't reliably do alone?** Is the check something
   that benefits from deterministic computation over an LLM's own
   estimate -- per `research-progress.md` §5's AblationBench finding that
   frontier models catch only ~38-45% of ablations a human would flag?
4. **Long-term differentiation?** Does this stay defensible as
   competitors (chiefly W&B's own MCP server and dashboard) evolve, or
   does it get commoditized quickly?

---

## Ranking

### 1. Ablation confound checking (workflow archetype 4) -- highest priority, strongest evidence

- **Real problem?** Yes, and the strongest-evidenced of all five
  archetypes: named methodological pitfall in independent literature
  (`researcher-workflows.md` §5), *and* now institutionally enforced at
  the venue level (NeurIPS/ICML checklist requirements), *and* directly
  validated by the AblationBench finding that current frontier models
  catch fewer than half the ablations a human reviewer would flag
  (`research-progress.md` §5). This is the only one of the five workflows
  with evidence spanning practitioner methodology, venue policy, *and* a
  direct benchmark of LLM performance at the task.
- **Trust increase?** High -- a deterministic allowlist diff is easy to
  audit and explain, which matters more here than statistical
  sophistication would, because the claim being checked ("is this pair
  actually isolating one variable") is binary and verifiable, not a
  matter of degree.
- **Reasoning Claude can't do alone?** Yes, directly evidenced --
  AblationBench's ~38-45% figure is not this project's own framing, it is
  an external benchmark result.
- **Long-term differentiation?** Moderate-to-high. W&B's MCP server is
  retrieval-only today (`research-progress.md` §5); nothing in this pass's
  research suggests confound-checking is a roadmap item for W&B or any
  competitor. Risk: if W&B's MCP server adds a judgment layer, this
  specific check is conceptually simple enough to be replicated quickly.

### 2. Training curve pathology detection (workflow archetype 2, RL-specific slice) -- second priority, evidence is real but scope is narrower than the workflow

- **Real problem?** Yes -- multiple independently-named pathology classes
  (stagnant performance, divergence/NaN, policy collapse) in a widely-used
  practitioner reference (`researcher-workflows.md` §3), and these are
  explicitly described as something practitioners must catch by eye today.
- **Trust increase?** Moderate-to-high for the pathologies
  `audit_training_curve` already covers (nulls, jumps, plateaus,
  oscillation); this pass found no evidence bearing on trust specifically,
  since fixed thresholds remain unvalidated against live data
  (`research-progress.md` §7, unchanged by this pass).
- **Reasoning Claude can't do alone?** Plausible but *not* directly
  evidenced the way ablation-checking is -- no benchmark equivalent to
  AblationBench was found in this pass measuring LLM accuracy at spotting
  curve pathologies unaided. This is a real gap in the evidence base for
  this specific workflow, worth naming rather than assuming away.
- **Long-term differentiation?** Moderate. Deterministic curve heuristics
  are conceptually simple and could be replicated by a competitor;
  differentiation would come more from breadth (covering entropy collapse
  and gradient norms, per `researcher-workflows.md` §3's finding that RL
  practitioners treat these as equally load-bearing) than from the
  existing four signal detectors alone.

### 3. Sweep importance / significance judgment (workflow archetype 3) -- third priority, real but partially commoditized

- **Real problem?** Yes -- correlation-read-as-causation confusion is
  real and documented (`pain-points.md` §4), and the low-run-count
  reliability question is something real users ask about explicitly on
  W&B's own community forum.
- **Trust increase?** Moderate. The Fisher z-transform significance
  testing and 10-run floor add real rigor over "eyeball the panel," but
  the underlying ranking method (Pearson) is a downgrade from what W&B
  already ships for in-W&B sweeps, which caps how much *additional* trust
  this specific tool can add for W&B users specifically (it adds more
  trust for non-W&B or cross-backend sweep contexts, where no equivalent
  panel exists at all).
- **Reasoning Claude can't do alone?** Plausible (an LLM eyeballing a
  Pearson table is unlikely to correctly apply Fisher z-transform
  significance testing or enforce a 10-run floor consistently), but not
  independently benchmarked in this pass the way ablation-checking is.
- **Long-term differentiation?** Lower than the other two. W&B has
  already shipped a statistically stronger method for the in-W&B case
  (`researcher-workflows.md` §4); this tool's honest differentiator is
  narrower (backend-agnostic, no re-run required) and doesn't compound
  over time the way ablation-checking's AblationBench-validated gap does.

### 4. Reproducibility/tracked-experiment workflows (workflow archetype 5) -- out of current scope, correctly so

- **Real problem?** Yes, well-evidenced (`researcher-workflows.md` §6),
  but it is a retrieval/infrastructure problem (getting a run to actually
  re-run identically), not a judgment problem.
- **Trust increase / reasoning Claude can't do alone?** Not applicable in
  the same sense -- this is closer to "does the tooling correctly capture
  and replay an environment," which is deterministic engineering, not
  judgment-under-uncertainty.
- **Long-term differentiation?** Not relevant to rank here -- this
  workflow is correctly out of scope per the project's own "not another
  tracker" positioning, and should stay there. Ranked last not because
  it's unimportant to researchers, but because it's a different product
  category.

### 5. General dashboard-scan loop (workflow archetype 1) -- foundational context, not independently actionable

- **Real problem?** Yes, and it's the most universally-evidenced workflow
  of the five (`researcher-workflows.md` §2) -- but it is the *background
  condition* every other workflow happens inside, not a distinct
  insertion point of its own. "People read dashboards by eye" is the
  reason judgment tools matter generally; it does not by itself point to
  a specific tool to build.
- **Ranking note:** Not ranked against the others on the same axis for
  this reason -- it is context, not a competing candidate for "where to
  build next."

---

## What this ranking does NOT resolve

This ranking is built entirely from documentation, methodology papers, and
one external benchmark (AblationBench). It says which workflows have the
strongest *evidence of problem existence* and the clearest *evidence that
LLM-alone reasoning is insufficient* (ablation-checking, uniquely, has
both). It does **not** resolve, and should not be read as resolving:

- Whether researchers actually want any of these checks run *unprompted*
  (the core UX bet, still open per `research-progress.md` §7 and
  `researcher-workflows.md` §7).
- Whether `audit_training_curve`'s specific thresholds are well-calibrated
  (still blocked on live fixture data).
- Relative engineering cost of closing the entropy/gradient-norm gap in
  `audit_training_curve` versus deciding the multiple-comparisons question
  in `audit_sweep` -- this document ranks by evidence strength and mission
  fit, not by implementation cost, which is a separate analysis this pass
  did not attempt.

---

## Decision Impact

**What assumptions were confirmed?**
- Ablation-checking is not just *a* defensible priority but the
  *best-evidenced* one of the three judgment tools, uniquely supported by
  a direct external benchmark (AblationBench) showing LLMs alone are
  insufficient at the task -- no equivalent benchmark-level evidence
  exists yet for training-curve or sweep judgment.
- The project's existing "not another tracker" scope boundary is
  correctly drawn: reproducibility/tracking workflows are real researcher
  pain points but belong to a different product category, and this
  ranking process, applied honestly, independently arrives at the same
  boundary the project already holds.

**What assumptions were disproven?**
- The implicit even-handedness across the three judgment tools (treating
  `audit_ablation`, `audit_training_curve`, and `audit_sweep` as
  roughly-equal-priority siblings) does not survive this ranking exercise.
  `audit_sweep` in particular ranks lowest of the three on long-term
  differentiation specifically because a competitor (W&B) already ships a
  statistically stronger method for its own sweeps.

**What should we build because of this?**
- No new tool is recommended. If a future evidence-gathering pass (e.g.
  `04_benchmarks/`) has to prioritize which of the three judgment tools to
  validate against live data first, given constrained time, this ranking
  suggests `audit_ablation` first (strongest problem evidence, strongest
  differentiation, direct external benchmark support), then
  `audit_training_curve` (strong problem evidence, differentiation depends
  on closing the entropy/gradient-norm scope gap), then `audit_sweep`
  (real but narrower differentiation given W&B's existing panel).

**What should we stop building because of this?**
- Nothing should stop. This ranking is a prioritization aid for
  *validation and documentation* effort under the existing three-tool
  scope, not a recommendation to cut any of the three tools -- each has
  independent evidence of solving a real problem; they differ in strength
  of evidence and durability of differentiation, not in whether they
  belong in the product at all.
