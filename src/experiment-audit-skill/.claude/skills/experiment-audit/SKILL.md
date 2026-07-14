---
name: experiment-audit
description: Use this skill whenever the user is working with experiment-audit-mcp (the eight MCP tools test_connection, list_runs, get_run_summary, get_metric_history, compare_runs, audit_training_curve, audit_ablation, audit_sweep) to inspect Weights & Biases runs, sweeps, and training curves. Trigger this for ANY request about comparing ML runs, diagnosing a training curve (crashes, plateaus, NaNs, oscillation), checking whether an ablation is confounded, ranking hyperparameter importance in a sweep, or generally asking "did I mess up this experiment" / "why did my loss/reward do X" / "which run is better" / "what changed between these runs" — even if the user doesn't name a tool directly. Also use it to sanity-check W&B credentials, list or summarize runs, or pull raw metric history. This skill teaches correct tool selection, output interpretation, scientific reasoning discipline, and how to avoid overclaiming from heuristic results.
---

# Experiment Audit MCP — Usage Skill

`experiment-audit-mcp` gives you eight tools over a user's W&B project.
Four are **retrieval** (`test_connection`, `list_runs`, `get_run_summary`,
`get_metric_history`), one is **deterministic diffing** (`compare_runs`),
and three are **heuristic judgment** (`audit_training_curve`,
`audit_ablation`, `audit_sweep`). The naming is a deliberate trust signal:
`get_*`/`list_*` results are facts, `compare_*` is exact computation, and
`audit_*` results are statistical judgments that can be wrong and always
carry `method` + `confidence` + `evidence` to prove it.

Your job is not just to call the right tool — it's to **reason like a
careful scientist over the output**, never state an audit result as if it
were a fact, and always surface the tool's own caveats to the user.

Read `reference.md` before interpreting any `audit_*` output in detail —
it has the exact thresholds, formulas, and schemas. Read `examples.md`
for worked end-to-end conversations. Read `prompts.md` for how to phrase
findings, hedge appropriately, and avoid the specific reasoning mistakes
this domain invites.

## The eight tools at a glance

| Tool | Kind | Use it when the user wants... |
|---|---|---|
| `test_connection` | retrieval | To verify W&B credentials work, or as your first call in a fresh session before anything else. |
| `list_runs` | retrieval | A list of runs in a project (cheap: id/name/tags/status only, no config/metrics). |
| `get_run_summary` | retrieval | Full config + summary metrics + data-completeness for **one** run. |
| `get_metric_history` | retrieval | The raw point-by-point series for **one metric on one run** — e.g. "show me the reward curve," not "what's wrong with it." |
| `compare_runs` | diffing | A factual config/metric diff across 2+ runs — "what's different between A and B" with no verdict attached. |
| `audit_training_curve` | judgment | To diagnose *why* a curve looks off — crashes, plateaus, NaNs, oscillation — not just to see the curve. |
| `audit_ablation` | judgment | To check whether an ablation pair (baseline vs. ablation run) actually isolates one claimed variable, or is confounded by other config drift. |
| `audit_sweep` | judgment | To rank which hyperparameters actually mattered in a sweep, with a hard 10-run floor and co-variance warnings. |

## Tool selection

Selection mistakes in this domain mostly go one of two ways: reaching for
an `audit_*` tool when the user just wants raw data (over-triggering
judgment), or reaching for `get_metric_history`/`compare_runs` when the
user is actually asking "why," which needs a heuristic diagnosis. Use
this decision order:

1. **Does the user want to know if their setup/credentials work, or is
   this the first tool call of the session?** → `test_connection`.
   Calling it first is cheap insurance — failing here beats failing three
   tool calls deep into a task.

2. **Does the user want to browse/enumerate runs** ("what runs do I
   have," "list my sweep runs," "find the run from last Tuesday")?
   → `list_runs`. Don't reach for `get_run_summary` in a loop to answer
   a listing question — `list_runs` is the cheap path.

3. **Does the user want everything about ONE specific run** (config,
   final metrics, whether it's still ingesting)? → `get_run_summary`.

4. **Does the user want to SEE a metric's values over time, with no
   diagnostic judgment implied** ("what does the reward curve look
   like," "pull the loss history")? → `get_metric_history`. This is also
   the correct choice when a request that sounds like `audit_training_curve`
   is actually just asking to see the numbers/plot the data — the
   distinguishing signal is whether the user wants a *diagnosis* or just
   the *data*.

5. **Does the user want a factual diff between named runs** ("what's
   different between A and B," "I changed the LR — what else moved?")
   with no request for a verdict on whether that's a problem?
   → `compare_runs`. This is also the right call even inside an
   ablation-flavored question if the user only wants the diff itself,
   not a judgment about confounding — `audit_ablation` is for when they
   want the verdict.

6. **Does the user want to know WHY a curve looks wrong** — a crash, a
   stall, jaggedness, or NaNs — or ask something like "did my training
   flatline" / "why did reward crash" / "is this loss curve healthy"?
   → `audit_training_curve`.

7. **Does the user claim an ablation isolates one variable and want that
   claim checked** — "did I mess up this ablation," "was my seed-only
   rerun actually clean," "is this a fair comparison"? → `audit_ablation`.
   You need `baseline`, `ablation`, and the `claimed_variable` name; ask
   for whichever of these three isn't already clear from context rather
   than guessing.

8. **Does the user want to know which hyperparameter mattered, or
   whether two hyperparameters are confounded in a sweep/grid**?
   → `audit_sweep`.

When a request is genuinely ambiguous between a `compare_*`/`get_*` tool
and its `audit_*` neighbor (e.g. "what else differs in my ablation run"
could mean either), default to the lighter retrieval/diffing tool unless
the user explicitly asks for a judgment, a verdict, or "is this okay" —
escalating to judgment uninvited risks manufacturing confidence the user
didn't ask for. See `examples.md` for the specific distractor pairs this
server was tested against.

## Core reasoning discipline

These rules apply to every `audit_*` result you relay to the user:

- **Never state a judgment tool's output as a bare fact.** Say "the
  ablation audit flags this as confounded because..." not "your ablation
  is broken." The tool's own `verdict`/`signal` is itself a hypothesis
  backed by `method` and `evidence`, not ground truth.
- **Always surface `confidence` and `method`, not just the headline
  result.** A `"low"` confidence `audit_sweep` ranking or an
  `audit_ablation` verdict downgraded because of `partial_data` changes
  what you should tell the user to do next (e.g. "treat this as
  provisional," "get more runs," "re-check this run's completeness").
- **Read `evidence` before summarizing, don't just parrot the top-line
  field.** `evidence` exists specifically so a `"confounded"` verdict or
  a `sudden_jump` signal can be checked, not taken on faith — glance at
  the actual differing params or the before/after values and mention the
  concrete numbers, not just the label.
- **Empty results are informative, not failures.** An empty `signals`
  list from `audit_training_curve` means the curve is clean by these
  four detectors — say that plainly, don't treat it as "nothing to
  report."
- **Respect refusals; don't work around them.** `audit_sweep` refusing
  with `insufficient_samples` below 10 usable runs is the tool doing its
  job, not an error to route around by asking for a ranking anyway or
  computing your own correlation from the raw data. Tell the user why it
  refused and what would unblock it (more runs, or runs that actually
  logged the target metric).
- **Know each method's blind spot and mention it when relevant.**
  `audit_sweep`'s Pearson correlation only catches linear relationships —
  a near-zero-correlation parameter with a non-monotonic (e.g. interior-
  optimum) effect can still be the most important one; the tool's own
  `caveat` field says this, repeat it rather than dropping it. Pearson
  results also do not correct for multiple comparisons, so with several
  parameters near the sample floor, some correlation will look
  moderately large by chance. `audit_ablation`'s allowlist is exact-match
  only (not fuzzy) — a differently-named seed field (e.g. `random_seed`
  instead of `seed`) will be flagged as an unaccounted difference even
  though it's probably benign; say so if you notice a plausibly-benign
  param flagged as non-intentional.
- **Don't chain audit tools into a stronger claim than either supports
  alone.** If `audit_training_curve` flags a `sudden_jump` and separately
  you see a confounded `audit_ablation`, don't declare the jump was
  *caused by* the confound unless the evidence actually connects them —
  say what each tool found and let the user draw the causal link, or
  flag it as a hypothesis worth investigating rather than a conclusion.
- **When confidence is genuinely insufficient to answer the user's real
  question, say so directly** rather than stretching a low-confidence
  result to sound more decisive. "This ranking is low-confidence with
  only 11 usable runs and a top p-value of 0.08 — I'd treat this as a
  weak signal, not an answer" is more useful than restating the ranking
  without context.

## Standard workflow patterns

Most real requests are actually small multi-tool workflows. A few
recurring ones — full detail and sample tool-call sequences in
`examples.md`:

- **"Did I mess up this ablation?"** → `audit_ablation(baseline, ablation, claimed_variable)`
  directly (it internally reuses `compare_runs`'s diff, so you don't need
  to call `compare_runs` separately first). If the user hasn't given a
  `run_id` for one side, use `list_runs` to find candidates before
  asking them to disambiguate.
- **"Why did my reward crash / is my loss healthy?"** → `audit_training_curve(ref, metric)`
  directly — it fetches history internally.
- **"Which hyperparameter mattered in my sweep?"** → `audit_sweep(sweep_ref, target_metric?)`.
  If it refuses with `insufficient_samples`, don't retry with a smaller
  ask — explain the floor and, if useful, offer `list_runs` to show what
  is actually in the sweep.
- **New session, first W&B-touching request** → consider `test_connection`
  first, especially if this is the first tool call, or if a prior call
  failed with `auth_failed`.
- **Exploring an unfamiliar project** → `list_runs` to find run IDs before
  calling anything that needs a `RunRef`.

## Error handling

Every tool can return a structured error instead of a result:
`{"error": {"error_type", "message", "recoverable", "retry_after_seconds"}}`.
`error_type` is one of `auth_failed`, `rate_limited`, `run_not_found`,
`backend_unsupported_capability`, `insufficient_samples`, `partial_data`,
`unknown`. Never silently retry a non-recoverable error with different
arguments hoping it works — read the message and either fix the input
(e.g. a wrong `run_id`) or tell the user what's blocking the request (a
missing API key, a project that doesn't support sweeps, etc.). Full
per-error guidance is in `reference.md`.

## Reference files

- **`reference.md`** — exact tool signatures, output schemas, thresholds,
  formulas, the error taxonomy, and every field's meaning. Read this
  before interpreting `audit_*` output in detail or before constructing
  a tool call whose exact argument shape you're unsure of.
- **`examples.md`** — worked multi-turn conversations for each tool,
  including the tool-selection distractor cases this server was actually
  tested against, and what a good final answer to the user looks like.
- **`prompts.md`** — scientific-reasoning and communication guidance:
  how to phrase findings, how to hedge without being wishy-washy, common
  mistakes to avoid, and how to combine multiple tool calls into a
  coherent investigation without overclaiming.
