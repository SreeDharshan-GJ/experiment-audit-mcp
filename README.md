# experiment-audit-mcp

**Your agent can catch confounded ablations, training pathologies, and
misleading sweep conclusions across dozens of runs that you'd otherwise
have to notice by eye — this is leverage on researcher attention, not
another dashboard.**

`experiment-audit-mcp` is an [MCP](https://modelcontextprotocol.io) server
that sits on top of your W&B project and gives an agent (or you, directly)
eight tools split cleanly into two kinds: cheap deterministic retrieval,
and heuristic judgment that always shows its work. It does not visualize
anything and does not replace your dashboard — it exists for the one thing
dashboards are bad at and LLMs are worse at: reliably comparing dozens of
floats across dozens of runs without hand-waving.

Status: **v1.0.0**, W&B backend only. All ten roadmap milestones are
complete and reviewed. Two verification steps remain genuinely blocked by
this environment's sandbox constraints (no live W&B credentials, no live
Anthropic API access) rather than skipped — see
[Known Gaps](#known-gaps-honest-status) below before you rely on this in
production.

---

## Contents

- [Why this exists](#why-this-exists)
- [Install](#install)
- [Quick start](#quick-start)
- [The eight tools](#the-eight-tools)
- [Architecture](#architecture)
- [API examples](#api-examples)
- [Data handling](#data-handling)
- [Known gaps (honest status)](#known-gaps-honest-status)
- [Development](#development)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

## Why this exists

Three specific, recurring failure modes motivated this project:

1. **Confounded ablations.** You change `use_memory: false` to test an
   ablation, but `batch_size` also silently changed between runs. The
   metric delta you're about to write up isn't measuring what you think
   it's measuring.
2. **Training pathologies that are easy to miss by eye** across dozens of
   runs and metrics — a NaN mid-curve that got silently dropped by a
   plotting library, a plateau that looks like convergence, a jagged
   oscillation.
3. **Misleading sweep conclusions** — a "most important hyperparameter"
   claim from a 3-run sweep, or two hyperparameters that move together so
   an importance ranking attributes one's effect to the other.

`experiment-audit-mcp` gives an agent tools that refuse to be wrong quietly
about any of these. See the design spec's non-negotiable design
principles (`docs/design-spec-v1.md` §1) for the four rules every tool
follows.

## Install

Requires Python 3.11+.

```bash
pip install experiment-audit-mcp
```

Set your credentials (a **read-only** W&B API key is recommended — this
server never writes to your project):

```bash
export WANDB_API_KEY="your-read-only-key"
# Optional: only needed if your key's default entity isn't the one you
# want to query against.
export WANDB_ENTITY="your-team-or-username"
```

Add it to your MCP client config. For Claude Desktop
(`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "experiment-audit": {
      "command": "experiment-audit-mcp",
      "env": {
        "WANDB_API_KEY": "your-read-only-key"
      }
    }
  }
}
```

For Claude Code:

```bash
claude mcp add experiment-audit -e WANDB_API_KEY=your-read-only-key -- experiment-audit-mcp
```

Or run it directly (useful for testing with the
[MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector)):

```bash
npx @modelcontextprotocol/inspector experiment-audit-mcp
```

## Quick start

Once connected, ask your agent something like:

> "Did I mess up my memory-ablation run? Compare `mamfac-baseline` and
> `mamfac-no-memory` in the `mamfac` project and check whether the only
> real difference is `use_memory`."

The agent will call `audit_ablation`, which returns a verdict
(`clean` / `confounded` / `uncertain`), a confidence level, and the full
config diff it based that verdict on — not just an assertion.

> "Why did the reward on run `xj29fk1a` crash around step 40,000?"

The agent will call `audit_training_curve`, which fetches the metric's
full history and returns scored signals (null values, sudden jumps, flat
plateaus, oscillation) with the exact step range and evidence for each —
never a bare label.

> "Which hyperparameter actually mattered in my `lr-sweep` sweep?"

The agent will call `audit_sweep`, which refuses to rank anything below
10 usable runs, and flags any hyperparameter pairs that moved together
so you don't mistake one's effect for the other's.

## The eight tools

Tools are named so the trust level is visible in the name itself, not
buried in the docs — a `get_*`/`list_*` result is a deterministic fact; a
`compare_*` result is deterministic computation over multiple runs; an
`audit_*` result is a heuristic judgment that **always** carries `method`,
`confidence`, and `evidence` fields, enforced at the schema level (see
`docs/design-spec-v1.md` §4.1).

| Tool | Kind | What it does |
|---|---|---|
| `test_connection` | retrieval | Validates W&B credentials; runs automatically on server start, also callable mid-session. |
| `list_runs` | retrieval | Cheap, paginated run listing (id/name/tags/status only — no config or metrics). |
| `get_run_summary` | retrieval | Full config + summary metrics + `data_completeness` for one run. |
| `get_metric_history` | retrieval | Full point-by-point history for one metric on one run. |
| `compare_runs` | diffing | Config + metric diff across N runs (not just pairwise). No verdict — just the facts. |
| `audit_training_curve` | judgment | Scored signals over a metric history: `null_values`, `sudden_jump`, `low_variance_plateau`, `high_frequency_oscillation`. |
| `audit_ablation` | judgment | Verdict (`clean`/`confounded`/`uncertain`) on whether an ablation pair actually isolates `claimed_variable`. |
| `audit_sweep` | judgment | Hyperparameter importance ranking with a hard 10-run floor and co-varying-parameter warnings. |

Full methodology — exact thresholds, formulas, and the reasoning behind
each detector — lives in `docs/audit-methods.md`. Tool descriptions in
the MCP schema itself stay short and point here, so they don't cost
context budget on every conversational turn.

## Architecture

```
experiment_audit_mcp/
├── models.py            # RunRef, Run, MetricPoint, MetricHistory, Sweep, Page[T]
├── errors.py             # ToolError + the frozen error_type taxonomy
├── auth.py               # env-var credential handling, fail-fast
├── server.py             # FastMCP entrypoint; registers all 8 tools
├── backends/
│   ├── base.py           # ExperimentBackend ABC, BackendCapability
│   ├── fake_backend.py   # in-memory test double (adversarial-state injectable)
│   └── wandb_backend.py  # real W&B implementation
└── analysis/
    ├── comparison.py      # compare_runs (pure diffing)
    ├── divergence.py       # audit_training_curve's 4 signal detectors
    ├── confound.py         # audit_ablation's allowlist + verdict logic
    └── sensitivity.py       # audit_sweep's correlation + significance testing
```

Two decisions worth understanding before you read the code:

- **Retrieval and judgment are structurally separate**, all the way down.
  Every `audit_*` tool at the `server.py` layer does nothing but fetch
  data and translate backend errors into structured `ToolError` dicts;
  the actual heuristics live entirely in `analysis/`, which has no
  knowledge an MCP call is even involved. You can unit-test
  `analysis/divergence.py`'s detectors against a hand-built curve with
  zero backend, zero MCP, zero network.
- **Backends are capability-declared, not blanket-abstract.** `list_sweeps`
  has a default `NotSupportedError` implementation rather than being a
  required abstract method, so a future backend without a native sweep
  concept (MLflow, planned for v2) can declare `capabilities = {ARTIFACTS}`
  and get a clear refusal instead of a fake mapping. See Appendix A of the
  design spec for how this was validated against MLflow's actual shape
  before being frozen.

For the full frozen contract (every field, every tool signature, every
adversarial case it's tested against) see `docs/design-spec-v1.md`. For
how it was built, milestone by milestone, including every design flaw
caught and fixed along the way, see
`docs/implementation-roadmap-v1.md`.

## API examples

Every audit tool call looks like this over MCP (shown here as the raw
tool-call JSON an MCP client sends/receives — you won't normally write
this by hand, your agent does):

**Checking an ablation:**

```json
{
  "tool": "audit_ablation",
  "arguments": {
    "baseline": {"backend": "wandb", "entity": "your-team", "project": "mamfac", "run_id": "baseline-run-id"},
    "ablation": {"backend": "wandb", "entity": "your-team", "project": "mamfac", "run_id": "ablation-run-id"},
    "claimed_variable": "use_memory"
  }
}
```

```json
{
  "verdict": "confounded",
  "confidence": "high",
  "differing_params": [
    {"param": "use_memory", "baseline_value": true, "ablation_value": false, "likely_intentional": true},
    {"param": "batch_size", "baseline_value": 64, "ablation_value": 32, "likely_intentional": false}
  ],
  "method": "full config diff against claimed_variable; params tagged intentional if name matches claimed_variable or is on the allowlist (seed, device, run name/id)",
  "evidence": { "...": "full compare_runs-style diff, config and metrics" }
}
```

**Auditing a training curve:**

```json
{
  "tool": "audit_training_curve",
  "arguments": {
    "ref": {"backend": "wandb", "entity": "your-team", "project": "mamfac", "run_id": "xj29fk1a"},
    "metric": "reward"
  }
}
```

```json
{
  "schema_version": 2,
  "metric_type_assumed": "reward",
  "signals": [
    {
      "signal": "sudden_jump",
      "score": 0.91,
      "step_range": [40120, 40140],
      "evidence": { "...": "the adjacent point pair and rate-of-change values" },
      "confidence": "high"
    }
  ],
  "method": "threshold-based, see docs/audit-methods.md#training-curve"
}
```

Every field in these responses is real output shape from the current
implementation, not aspirational. See `docs/design-spec-v1.md` §4.2 for
the complete, frozen schema of every tool.

## Data handling

- Data never leaves your machine except calls to your own W&B endpoint.
  This server is stateless and open source — read the code, there's
  nowhere for your data to go.
- Credentials are read once from environment variables (`WANDB_API_KEY`,
  optionally `WANDB_ENTITY`), validated fail-fast on server start, and
  never logged or echoed back in any error message.
- Use a **read-only** API key. This server has no write path to W&B —
  a read-only key is strictly sufficient and reduces what a
  misconfigured or compromised client could ever do.

## Known gaps (honest status)

Two verification steps that this project's own completion criteria call
for are **written and ready to run, but have not actually been run**,
because this build environment has no live network path to the relevant
services. Documented here rather than silently marked done:

1. **Fixture recording against a real W&B project** — `tests/` currently
   test `WandbBackend` against an in-memory fake client built from W&B's
   *documented* API shapes, not against fixtures recorded from a live
   project. `scripts/record_wandb_fixtures.py` exists and is ready to run
   against your own project (e.g. a MAMFAC/CARM++ project) to close this
   gap. See `tests/fixtures/README.md`.
2. **Tool-selection eval against a live MCP client** —
   `scripts/tool_selection_eval.py` and its 15-prompt fixed set
   (`scripts/tool_selection_prompts.py`) exist and are exercised for
   correctness up to the actual API call, but have not been run against
   a live model (this environment has no `ANTHROPIC_API_KEY` / network
   path to `api.anthropic.com`). See `docs/tool-selection-eval.md` for
   the exact command to run this yourself.

Additionally, as of this release:

- **Package name availability**: `experiment-audit-mcp` was confirmed
  unclaimed on both PyPI and npm as of 2026-07-10. This is a point-in-time
  check, not a lock — verify again immediately before publishing if time
  has passed.
- **MCP Registry / Glama / cursor.directory submission** has not been
  performed from this environment (no outbound network access to those
  services from here). The package is publish-ready; registry submission
  is a step for whoever runs the actual release from a machine with that
  access.
- This is a **v1, W&B-only** release. MLflow support is prototyped at the
  interface level (see Appendix A of the design spec) but not implemented.
- `audit_sweep`'s correlation-based ranking only detects *linear*
  relationships — a hyperparameter with a non-monotonic effect (e.g. an
  interior-optimum learning rate) can rank near the bottom despite
  mattering most. This is a documented method limitation, not a bug; see
  `docs/audit-methods.md` (sweep section).

None of these gaps are architectural. They are either genuine sandbox
network limitations or explicitly deferred v2/v3 scope per the frozen
roadmap — see Roadmap below.

## Development

```bash
git clone https://github.com/<your-username>/experiment-audit-mcp.git
cd experiment-audit-mcp
pip install -e ".[dev]"
pytest                # 233 tests
ruff check .          # lint
```

Everything is developed against `FakeBackend` (`backends/fake_backend.py`),
an in-memory test double that can inject every adversarial state named in
the design spec (tiny sweeps, NaN mid-curve, correlated hyperparameters,
partial data) on demand — no live W&B credentials or network access needed
to run the full suite.

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` first — in
short: the v1 design (`docs/design-spec-v1.md`) is frozen, so changes to
existing tool schemas, model fields, or the backend interface need an
explicit, logged design decision (a "Revision" entry in the spec,
following the pattern of Revision 1 and Revision 2 already in the
document), not a silent PR. New `audit_*` tools in future versions must
implement the mandatory `method` / `confidence` / `evidence` schema from
day one (spec §8).

## Roadmap

See `docs/implementation-roadmap-v1.md` for the full v1 build history (10
milestones, 2 logged spec revisions, all approved). Looking ahead, per
`docs/design-spec-v1.md` §10:

- **v2** — MLflow backend, versioned API compatibility matrix, first
  public case study from a real project.
- **v3** — RL-specific pathology signals (reward-hacking heuristics,
  proper multi-seed statistical tests), optional experimental
  `claimed_variable` inference for `audit_ablation`, Optuna/Ray Tune
  sweep support, open to external `audit_*` contributions.

## License

MIT — see `LICENSE`.
