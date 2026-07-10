# Tool-Selection Eval

Milestone 9 deliverable, per design-spec-v1.md §7 point 3 and the
roadmap: "a fixed prompt set... run against an actual MCP client, with
pass/fail on whether the correct tool was invoked" plus "a short report
of any tool-description wording changes made as a result... since this
is exactly the kind of tuning that should be visible and reviewable, not
silent."

## Status: written, not yet run against a live client

The eval harness (`scripts/tool_selection_eval.py`) and the fixed prompt
set (`scripts/tool_selection_prompts.py`, the single source of truth
this document is generated from) both exist and are exercised for
correctness up to the point of the actual API call. What has **not**
happened: an actual run against a live model. The build environment used
for this milestone has no `ANTHROPIC_API_KEY` and no network path to
`api.anthropic.com` — egress here is allowlisted to package registries
only.

This mirrors, deliberately, the exact situation Milestone 3's summary and
`scripts/record_wandb_fixtures.py` already document for W&B fixture
recording: a script that is real, runnable, and ready, blocked only by a
sandbox credential/network constraint, flagged explicitly rather than
worked around with a synthetic stand-in that would quietly satisfy the
letter of the completion criteria without the substance. Milestone 9's
own completion criteria — "all fixed prompts correctly invoke their
intended tool" — is therefore **not yet verified** and should not be
treated as met until someone runs:

```bash
export ANTHROPIC_API_KEY=...
python scripts/tool_selection_eval.py --json docs/tool-selection-eval-results.json
```

and this document is updated with the actual pass/fail table and any
wording changes that resulted from real misfires.

## The fixed prompt set

15 prompts covering all eight tools. Four of the `audit_*` prompts are
paired with a **distractor** prompt worded to tempt a nearby retrieval or
diffing tool instead — spec §4.1's whole premise is that the `get_*` /
`list_*` / `compare_*` / `audit_*` naming convention should let a client
tell retrieval from judgment apart by name alone, so the eval is designed
to actually stress that boundary rather than only confirm each tool
answers to its own easiest phrasing. See
`scripts/tool_selection_prompts.py` for the canonical list (this table is
a rendering of it, not an independent copy — if the two ever disagree,
the `.py` file is correct):

| # | Prompt | Expected tool |
|---|---|---|
| 1 | "Can you check that my W&B credentials are working before we start?" | `test_connection` |
| 2 | "What runs are in the mamfac project from the last week?" | `list_runs` |
| 3 | "Give me the full config and final metrics for run xj29fk1a." | `get_run_summary` |
| 4 | "Pull the loss curve for run xj29fk1a across the whole run." | `get_metric_history` |
| 5 | "Why did my reward crash partway through training?" | `audit_training_curve` |
| 6 | "What do the reward values actually look like over time for this run?" (distractor for #5) | `get_metric_history` |
| 7 | "What's different between run A and run B — config and metrics?" | `compare_runs` |
| 8 | "Did I mess up this ablation? I only meant to change the learning rate." | `audit_ablation` |
| 9 | "I changed the learning rate for my ablation run — what else differs from baseline?" (distractor for #8) | `compare_runs` |
| 10 | "Was my seed-only rerun actually a clean ablation, or did something else change too?" | `audit_ablation` |
| 11 | "Which hyperparameter mattered most in my sweep?" | `audit_sweep` |
| 12 | "List every run in my learning-rate sweep so I can eyeball them myself." (distractor for #11) | `list_runs` |
| 13 | "Is learning_rate and batch_size confounded in my grid sweep, or independently important?" | `audit_sweep` |
| 14 | "My run's data looks incomplete — can you tell me if it finished ingesting?" | `get_run_summary` |
| 15 | "Has my training run flatlined? The loss hasn't moved in a while." | `audit_training_curve` |

Prompt 9 is intentionally the softest case in the set (see its
`rationale` field in the `.py` source): "ablation" plus "what else
differs" is genuinely answerable by either `compare_runs` or
`audit_ablation`, and the eval treats only `compare_runs` as the strict
pass so the harness can *detect* over-triggering of the heavier judgment
tool, not because a client choosing `audit_ablation` there would be
obviously wrong.

## Supplementary offline review (not a substitute for the live run)

As a best-effort check given the credential constraint above, each of
the eight tool descriptions in `server.py` was read against all 15
prompts by hand. Every description already names its own trigger
vocabulary fairly distinctly (`audit_training_curve`: "pathologies...
NaNs... plateaus... oscillation"; `audit_sweep`: "importance... refuses
with insufficient_samples... co-varying"; `audit_ablation`: "isolates
claimed_variable... confounded"), and none of the eight descriptions
currently overlaps another's vocabulary closely enough to raise an
obvious concern on inspection. This offline read is not a substitute for
running real prompts through a real model — tool selection is a property
of the model's behavior, not of how carefully a human can parse the
schema — so it is recorded here only as a sanity check, not as evidence
the completion criteria is met.

## Wording changes made this milestone

**None.** Editing tool descriptions without an observed misfire would be
speculative tuning, which runs against this project's own established
discipline (see e.g. `analysis/confound.py`'s allowlist rationale, or any
of the roadmap's "Flag-if-triggered" notes): report the constraint,
don't quietly route around it with a change that looks like progress but
isn't backed by evidence. If a live run later surfaces a misfire, record
it here as a dated before/after entry, e.g.:

```
### 2026-XX-XX — audit_sweep vs list_runs misfire on prompt #12
Before: "Rank hyperparameter importance across a sweep via pairwise
         Pearson correlation with target_metric..."
After:  <the actual wording change, plus the specific prompt(s) it fixed>
```

per the roadmap's own constraint that any resulting change must be a
description edit, never a tool rename or schema change (that would
violate the frozen-spec process unless a real design flaw, not a wording
issue, is found).
