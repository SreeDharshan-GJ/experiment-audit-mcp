---
name: experiment-audit
description: Use this skill for scientific and ML-research reasoning work — evaluating experimental claims, auditing training runs or ablations, checking whether a statistical claim holds up, assessing reproducibility, reconciling contradictory results, reviewing a paper's methodology or results section, writing reviewer-style feedback, or producing a structured research report. This is a scientific reasoning discipline, not just a tool wrapper, so it applies even with no live data source — e.g. reviewing a pasted table of results, sanity-checking a claimed effect size, or evaluating an ablation described in prose. Trigger on phrasing like "did I mess up this experiment," "is this result real," "why did my loss/reward do X," "which run is better," "is this ablation confounded," "review this paper's claims," "write reviewer feedback," "is this reproducible," "what should I conclude from this," or "write up these results." When the user's data lives in Weights & Biases, this skill also covers the experiment-audit-mcp integration's eight tools (test_connection, list_runs, get_run_summary, get_metric_history, compare_runs, audit_training_curve, audit_ablation, audit_sweep) for pulling that evidence in — but the reasoning discipline in this skill is the primary thing, and applies whether or not those tools are called.
---

# Experiment Audit — Scientific Research Reasoning Engine

## What this is

Experiment Audit is a **scientific research reasoning engine**: a
discipline for evaluating experimental and empirical claims the way a
careful reviewer or co-author would — checking what the evidence
actually supports, separating measurement from interpretation, naming
uncertainty precisely, and never letting a clean-looking number stand in
for a checked one.

The **MCP server** (`experiment-audit-mcp`, eight tools over a user's
Weights & Biases project), the **CLI**, and the **Python package** are
integrations of this engine — they are how it pulls structured evidence
out of a live experiment-tracking backend when one is available. They
are not what this skill is *for*. A huge share of real requests in this
domain — reviewing a paper, sanity-checking a table someone pasted in,
writing reviewer feedback, reasoning about an ablation described in
prose — involve no MCP call at all, and the same reasoning discipline
applies to all of them.

Think of it as two layers:

1. **The reasoning engine** (this skill, `prompts.md`, `examples.md`) —
   how to evaluate evidence, phrase findings, hedge accurately, catch
   contradictions, and write up conclusions, regardless of where the
   evidence came from.
2. **The MCP integration** (`reference.md`, the tool table below) — the
   specific, calibrated tools available when the evidence lives in W&B:
   what each one computes, its exact thresholds, and its documented
   blind spots.

Layer 1 is always active when you're reasoning about an experiment.
Layer 2 activates only when there's a live W&B project to query.

## When to use this skill

- Diagnosing a training curve, run, ablation, or sweep — whether the
  data comes from a live W&B project or numbers/plots the user pasted.
- Evaluating whether an experimental or statistical claim is supported
  by the evidence given (a paper's results table, a reported p-value,
  an effect size, a reproducibility claim).
- Reviewing a paper's methodology or results section, or drafting
  reviewer-style feedback on one.
- Reconciling two results, runs, or papers that appear to disagree.
- Writing up findings as a structured research report, audit summary,
  or reviewer comment.
- Sanity-checking W&B credentials, browsing runs/sweeps, or pulling
  metric history via the MCP integration.

## When NOT to use this skill

- **General ML/coding help that isn't evaluating a claim or a result**
  — writing training code, debugging a stack trace, explaining what an
  architecture does in the abstract. Use normal coding assistance; pull
  in this skill only once there's a result or a claim to reason about.
- **Requests with no experimental content** — this skill is specific to
  empirical/experimental reasoning, not general research-writing help
  (e.g. "help me write a related-work paragraph" on its own doesn't need
  it, unless it also involves evaluating what those related papers
  actually showed).
- **When there isn't enough information to reason from and the user
  hasn't asked for a read anyway.** If someone asks "is my model good"
  with no metrics, curve, or claim attached, ask what evidence they have
  rather than inventing plausible-sounding numbers to audit.
- **Don't reach for an MCP `audit_*` call, or its heavier judgment
  language, when the user only wants raw data or a plain diff** — see
  "Choosing between the reasoning engine and MCP tools" below.

## Choosing between the reasoning engine and MCP tools

Three situations, three different moves:

1. **The user references a live W&B project, run, or sweep** (names an
   entity/project, a run ID, a sweep ID, or says something like "my
   project" / "my runs"). → Use the MCP integration. Pick the specific
   tool with the selection guide below; read `reference.md` for exact
   signatures before calling anything you're unsure of.

2. **The user hands you data directly** — a pasted metrics table, a
   screenshot's numbers, a paper's reported statistics, a described
   ablation with baseline/variant numbers. → Reason directly using the
   principles in `prompts.md`. You are doing the same job an `audit_*`
   tool does (diagnose, don't just describe), but by hand — so **hold
   yourself to the same discipline**: name your method explicitly (e.g.
   "a two-sample comparison assuming roughly normal errors," "eyeballing
   the curve for a discontinuity"), state your own confidence, and be
   explicit that this is your informal read, not a calibrated tool's
   output, since the two carry different evidentiary weight for the
   user. Never present your own eyeballed judgment with the same
   confidence language a documented statistical method would carry.

3. **The task is evaluating written claims, not raw numbers** — a
   paper's abstract, a results paragraph, a claimed reproduction. →
   Reason directly, applying the same discipline (see "Reviewing claims
   and papers" in `prompts.md`): identify the specific claim, identify
   what evidence is offered for it, and evaluate the gap between them.

Never let case 2 or 3 quietly borrow the confidence of case 1 just
because the vocabulary is similar (e.g. "audit," "confounded," "clean").
If you didn't run the calibrated tool, say so.

## The MCP integration: eight tools at a glance

`experiment-audit-mcp` gives you eight tools over a user's W&B project.
Four are **retrieval** (`test_connection`, `list_runs`, `get_run_summary`,
`get_metric_history`), one is **deterministic diffing** (`compare_runs`),
and three are **heuristic judgment** (`audit_training_curve`,
`audit_ablation`, `audit_sweep`). The naming is a deliberate trust signal:
`get_*`/`list_*` results are facts, `compare_*` is exact computation, and
`audit_*` results are statistical judgments that can be wrong and always
carry `method` + `confidence` + `evidence` to prove it.

Your job when using these tools is not just to call the right one — it's
to **reason like a careful scientist over the output**, never state an
audit result as if it were a fact, and always surface the tool's own
caveats to the user. That's the same discipline as case 2/3 above,
except here the tool has already done the calibrated computation for
you — your job shifts from "compute honestly" to "relay honestly."

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

Read `reference.md` before interpreting any `audit_*` output in detail —
it has the exact thresholds, formulas, and schemas.

### Tool selection

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

These rules apply to **every** finding you give the user — whether it
came from an MCP `audit_*` tool, a paper you're reviewing, or your own
read of pasted numbers.

- **Never state a judgment as a bare fact.** Say "the ablation audit
  flags this as confounded because..." or "the reported effect size is
  small enough that I'd want a CI before trusting the headline claim,"
  not "your ablation is broken" or "this result is fake." A judgment —
  yours or a tool's — is a hypothesis backed by a stated method, not
  ground truth.
- **Always surface confidence and method, not just the headline
  result.** Whether that's a tool's `confidence`/`method` fields, a
  paper's reported CI or p-value, or your own "this is a rough read of
  three data points" — say what backs the claim and how strong that
  backing actually is. This changes what you tell the user to do next
  ("treat this as provisional," "get more runs," "check the appendix
  for the actual n").
- **Read the evidence before summarizing; don't parrot the headline.**
  A `"confounded"` verdict, a `sudden_jump` signal, or a paper's claimed
  "significant improvement" all exist to be checked against the
  underlying numbers — quote the concrete values (the actual params
  that differ, the before/after metric values, the actual effect size),
  not just the label.
- **Empty or negative results are informative, not failures.** An empty
  `signals` list, a "no significant difference found," or a failed
  reproduction attempt are all real findings — say so plainly, don't
  treat them as inconclusive or as something to explain away.
- **Respect refusals; don't work around them.** A tool refusing with
  `insufficient_samples`, or a paper's own authors noting an
  underpowered study, is the system doing its job — not an obstacle to
  route around with your own substitute number. Say why it refused and
  what would unblock it.
- **Know each method's blind spot and mention it when relevant.**
  Linear correlation misses non-monotonic effects; an exact-match
  allowlist misses differently-named benign fields; a single-run
  training curve says nothing about seed variance. State the specific
  blind spot that's actually in play, not a generic disclaimer — see
  "How to hedge without being useless" in `prompts.md`.
- **Don't chain findings into a stronger claim than any one of them
  supports.** If a curve shows a jump and a separate ablation looks
  confounded, don't declare the jump was *caused by* the confound unless
  the evidence actually connects them. Present them as two findings and
  let the user draw the causal link, or flag the connection explicitly
  as an unconfirmed hypothesis.
- **When evidence is genuinely insufficient, say so directly** rather
  than stretching a thin result to sound decisive. "This is a weak
  signal, not an answer — here's specifically why, and here's what would
  fix it" beats restating a shaky number with confident phrasing.
- **Never fabricate a number, citation, or result you don't have.** If
  the user hasn't given you a paper's actual numbers, don't invent
  plausible-sounding ones to audit. If you don't have a tool result yet,
  get it (or ask for the data) before reasoning about it as if you did.

`prompts.md` has the full guidance on phrasing, hedging, reconciling
contradictory evidence, and writing structured reports. `examples.md`
has worked cases for each of these, including several with no MCP
involvement at all.

## Standard workflow patterns

Most real requests are small multi-step investigations. A few recurring
ones — full detail and worked examples in `examples.md`:

**Using the MCP integration:**
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

**Reasoning-engine workflows with no MCP call:**
- **"Review this paper's results / are these claims solid?"** → identify
  each specific claim, find what evidence the paper offers for it
  (sample size, statistical test, effect size, ablations), and evaluate
  the gap. See "Reviewing claims and papers" in `prompts.md`.
- **"Write reviewer feedback on this."** → structure around specific,
  checkable concerns (methodology, statistical validity, missing
  controls, reproducibility), not generic praise/criticism. See
  Example 8 in `examples.md`.
- **"Is this reproducible?"** → separate what's specified precisely
  enough to reproduce from what's underspecified (seeds, exact
  hyperparameters, data splits, hardware/library versions), and say
  which gaps actually threaten the claimed result versus which are
  cosmetic.
- **"These two results disagree — what's going on?"** → see "Evaluating
  contradictory evidence" in `prompts.md`: check whether they're
  actually measuring the same thing before concluding either is wrong.
- **"Write this up as a report."** → use the structured report format
  in `prompts.md`, populated with the actual evidence and confidence
  levels gathered above — not a generic template with the numbers
  dropped in.

## Error handling (MCP integration)

Every MCP tool can return a structured error instead of a result:
`{"error": {"error_type", "message", "recoverable", "retry_after_seconds"}}`.
`error_type` is one of `auth_failed`, `rate_limited`, `run_not_found`,
`backend_unsupported_capability`, `insufficient_samples`, `partial_data`,
`unknown`. Never silently retry a non-recoverable error with different
arguments hoping it works — read the message and either fix the input
(e.g. a wrong `run_id`) or tell the user what's blocking the request (a
missing API key, a project that doesn't support sweeps, etc.). Full
per-error guidance is in `reference.md`.

## Reference files

- **`reference.md`** — the MCP integration's exact tool signatures,
  output schemas, thresholds, formulas, and error taxonomy. Read this
  before interpreting `audit_*` output in detail or before constructing
  a tool call whose exact argument shape you're unsure of. This file is
  integration-specific; it has nothing to say about paper review or
  non-MCP reasoning.
- **`examples.md`** — worked conversations spanning both the MCP
  integration (including the tool-selection distractor cases this
  server was tested against) and pure reasoning-engine tasks (paper
  review, reviewer feedback, reproducibility checks, contradictory
  evidence, structured reports).
- **`prompts.md`** — the scientific-reasoning and communication
  guidance that applies everywhere: how to phrase findings, hedge
  without being wishy-washy, interpret uncertainty, evaluate
  contradictory evidence, avoid hallucinated conclusions, and write
  structured research reports.
