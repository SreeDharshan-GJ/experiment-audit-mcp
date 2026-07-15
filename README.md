<div align="center">

# experiment-audit

**A scientific reasoning engine for ML experiments.**
Feed it claims and evidence — it checks for missing support, scopes evidence to claims, catches contradictions, scores confidence, and renders a structured scientific report. The kind of review a careful advisor would give your results before you write them up.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-v1.1.0-brightgreen.svg)](CHANGELOG.md)
[![Built with FastMCP](https://img.shields.io/badge/MCP-FastMCP-purple.svg)](https://github.com/jlowin/fastmcp)

Created and maintained by **[Sree Dharshan G J](https://github.com/SreeDharshan-GJ)**

</div>

---

## Why this exists

Most experiment-tracking tools show you numbers. None of them tell you
whether your *claim* about those numbers is actually supported — whether
your ablation is confounded, your comparison is in scope, your evidence
contradicts something you said three runs ago, or your confidence is
justified rather than assumed. `experiment-audit` is the reviewer that
checks that, before a reader has to.

The reasoning engine is the core of this project. Everything else — the
MCP server, the W&B backend, the CLI, the Python package — is an
interface built around it.

## Contents

- [What the reasoning engine does](#what-the-reasoning-engine-does)
- [Install](#install)
- [Quick start: the reasoning engine](#quick-start-the-reasoning-engine)
- [Quick start: the MCP server (W&B audit tools)](#quick-start-the-mcp-server-wb-audit-tools)
- [Quick start: as a Claude Code plugin](#quick-start-as-a-claude-code-plugin)
- [Architecture](#architecture)
- [Data handling](#data-handling)
- [Known gaps (honest status)](#known-gaps-honest-status)
- [Development](#development)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Citing this project](#citing-this-project)
- [Author](#author)
- [License](#license)

## What the reasoning engine does

Given a set of **claims** (*"model-x achieves 95% accuracy on
CIFAR-10"*) and the **evidence** backing them (metrics, configs, logs,
prior runs), the engine runs six rules in sequence and produces a
`ScientificReport`:

| # | Rule | What it checks |
|---|------|-----------------|
| 1 | **Missing evidence** | Does this claim have any supporting evidence trace at all? |
| 2 | **Scope** | Does the evidence actually match the claim's stated scope (same dataset, same hardware, same evaluation protocol)? |
| 3 | **Contradiction** | Does any other claim or evidence item conflict with this one? |
| 4 | **Confidence** | A computed score, not a guess — based on evidence quality, quantity, contradictions found, and what's missing. |
| 5 | **Judgment** | A verdict (supported / partially supported / unsupported) with the reasoning behind it. |
| 6 | **Recommendation** | What to do about it — gather more evidence, narrow the claim's scope, retract it. |

Every finding traces back to specific evidence. Nothing in the report is
an unsupported assertion — that would rather defeat the point.

> This is one of two reasoning pipelines in the package. The second,
> lower-level pipeline (`ScientificReasoningEngine` — Evidence →
> Observations → Hypotheses → Confidence → Judgment → Recommendation) is
> a more generic, extensible framework for injecting custom hypothesis
> and confidence logic. Most people should start with the six-rule
> pipeline above. See `src/experiment_audit/reasoning/__init__.py` for
> both.

## Install

Requires Python 3.11+.

```bash
pip install experiment-audit
```

Or from source:

```bash
git clone https://github.com/SreeDharshan-GJ/experiment-audit-mcp.git
cd experiment-audit-mcp
pip install -e .
```

## Quick start: the reasoning engine

**As a CLI:**

```bash
# See the expected input shape
experiment-audit reasoning schema > claims.json

# Edit claims.json with your own claims/evidence, then:
experiment-audit reasoning run --input claims.json --format markdown
```

Also supports `--format json` and `--format text`, and `--output <path>`
to write the report to a file instead of stdout.

**As a Python library:**

```python
from experiment_audit.reasoning import (
    ScientificReasoningPipeline,
    ScientificReport,
    Claim, ClaimCategory, Scope,
    EvidenceItem, EvidenceKind,
)

claim = Claim(
    id="c1",
    subject="model-x",
    statement="model-x achieves 95% accuracy on CIFAR-10",
    category=ClaimCategory.PERFORMANCE,
    scope=Scope(dataset="cifar-10"),
)

pipeline = ScientificReasoningPipeline()
context = pipeline.build_initial_context(claims=[claim], evidence=[])
pipeline_report = pipeline.execute(context)
report = ScientificReport.from_pipeline_report(pipeline_report)

print(report.to_markdown())
```

## Quick start: the MCP server (W&B audit tools)

The original W&B experiment-audit tools are still here, unchanged, as an
MCP integration. Set a **read-only** W&B API key:

```bash
export WANDB_API_KEY="your-read-only-key"
export WANDB_ENTITY="your-team-or-username"   # optional
```

**Claude Desktop** (`claude_desktop_config.json`):

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

**Claude Code:**

```bash
claude mcp add -e WANDB_API_KEY=your-read-only-key experiment-audit -- experiment-audit-mcp
```

> `-e` must come before the server name, not after — putting it after
> the name has been a source of "Invalid environment variable format"
> errors in some Claude Code versions.

Then ask your agent something like:

> "Did I mess up my memory-ablation run? Compare `mamfac-baseline` and
> `mamfac-no-memory` in the `mamfac` project and check whether the only
> real difference is `use_memory`."

The agent calls `audit_ablation`, which returns a verdict
(`clean` / `confounded` / `uncertain`), a confidence level, and the full
config diff it based that verdict on.

Full tool reference (all eight tools, exact schemas, methodology) is in
[`docs/design-spec-v1.md`](docs/design-spec-v1.md) and
[`docs/audit-methods.md`](docs/audit-methods.md) — unchanged from the
v1.0.0 release.

## Quick start: as a Claude Code plugin

The reasoning-engine discipline above — how to phrase findings, interpret
uncertainty, evaluate contradictory evidence, review papers, and produce
structured reports — is also packaged as a [Claude Code](https://docs.claude.com/claude-code)
**skill**, distributed as a plugin under [`dev/experiment-audit-plugin/`](dev/experiment-audit-plugin/).
It works standalone (reasoning directly over data you give Claude) and,
when `WANDB_API_KEY` is set, also gets the eight MCP audit tools above.

From inside a Claude Code session, in this repo:

```
/plugin marketplace add ./dev/experiment-audit-plugin
/plugin install experiment-audit@experiment-audit
```

Pick **user scope** at the install prompt to make it available across
all your projects, not just this repo. Once installed, it triggers
automatically on prompts like *"is this ablation confounded,"* *"why
did my loss crash,"* *"review this paper's results claim,"* or *"write
reviewer feedback on this."*

The plugin's `.mcp.json` points at the local build's console-script
entry point (`experiment-audit-mcp`, from `[project.scripts]` in
`pyproject.toml`) — if you install this package elsewhere, update the
`command` path there to match. Full skill documentation (triggers, tool
selection, reasoning discipline, worked examples) is in
[`dev/experiment-audit-plugin/skills/experiment-audit/SKILL.md`](dev/experiment-audit-plugin/skills/experiment-audit/SKILL.md).

## Architecture

```
experiment_audit/
├── reasoning/                 # the Scientific Reasoning Engine
│   ├── claims.py               # Claim, ClaimSet, Scope
│   ├── evidence.py              # Evidence, EvidenceItem (shared by both pipelines)
│   ├── contradictions.py        # Contradiction, ContradictionSet
│   ├── scientific_rules/        # the six concrete rules
│   │   ├── missing_evidence_rule.py
│   │   ├── scope_rule.py
│   │   ├── contradiction_rule.py
│   │   ├── confidence_rule.py
│   │   ├── judgment_rule.py
│   │   └── recommendation_rule.py
│   ├── rules.py                 # RuleContext, ScientificRule base
│   ├── pipeline.py              # ScientificReasoningPipeline: runs the six rules in order
│   ├── scientific_report.py     # ScientificReport: to_markdown/to_json/to_text
│   ├── observations.py          # generic pipeline: pattern detection over Evidence
│   ├── hypotheses.py             # generic pipeline: candidate explanations
│   ├── confidence.py             # generic pipeline: confidence scoring
│   ├── judgment.py                # generic pipeline: verdict rendering
│   ├── recommendation.py          # generic pipeline: recommendations
│   └── engine.py                  # ScientificReasoningEngine: the generic pipeline's orchestrator
├── cli.py                     # `experiment-audit reasoning run|schema`
├── models.py                  # RunRef, Run, MetricPoint, MetricHistory, Sweep, Page[T]
├── errors.py                  # ToolError + the frozen error_type taxonomy
├── server.py                  # FastMCP entrypoint; registers the 8 W&B audit tools
├── backends/
│   ├── base.py                # ExperimentBackend ABC, BackendCapability
│   ├── fake_backend.py        # in-memory test double
│   └── wandb_backend.py       # real W&B implementation
└── analysis/                  # the W&B audit tools' pure heuristics
    ├── comparison.py
    ├── divergence.py
    ├── confound.py
    └── sensitivity.py
```

The reasoning engine and the MCP/W&B layer are independent — the
reasoning engine takes `Claim`s and `EvidenceItem`s directly and has no
dependency on W&B, FastMCP, or any backend. Feeding W&B run data into the
reasoning engine as claims/evidence (rather than hand-constructing them,
as the quick-start example above does) is on the [roadmap](#roadmap).

For the reasoning engine's design rationale, see
[`research/07_reasoning_engine/`](research/07_reasoning_engine/)
(`reasoning-engine.md`, `reasoning-rules.md`, `confidence-system.md`,
`evidence-model.md`, `scientific-reviewer.md`). For the MCP/W&B layer's
frozen contract, see [`docs/design-spec-v1.md`](docs/design-spec-v1.md).

## Data handling

- Data never leaves your machine except calls to your own W&B endpoint
  (MCP layer only — the reasoning engine itself makes no network calls
  at all).
- Credentials are read once from environment variables, validated
  fail-fast on server start, and never logged.
- Use a **read-only** W&B API key — this server has no write path.

## Known gaps (honest status)

**Reasoning engine:**

- There is currently no built-in adapter that converts a W&B run
  directly into `Claim`s/`EvidenceItem`s — you construct them yourself
  (via the CLI's JSON schema or directly in Python), or write your own
  extraction step. Building this adapter is the natural next step to
  connect the MCP/W&B layer to the reasoning engine directly.
- The generic pipeline (`ScientificReasoningEngine`, `engine.py`)
  defaults its rule-engine stage to a no-op (`NullRuleEngine`) unless you
  inject one — it does not currently share the six concrete rules the
  main pipeline uses. Treat it as an extensibility point, not a second
  complete pipeline.
- 274 tests pass (`pytest tests/ -q`), 32 of them exercising the
  reasoning engine directly, including three regression tests for bugs
  found and fixed during this release (see `CHANGELOG.md`). This is real
  coverage of the pipeline's mechanics; it is not a substitute for
  domain review of the six rules' actual thresholds and heuristics by
  someone in your specific research area.

**MCP/W&B layer** (carried over from v1.0.0, unchanged):

- Fixture recording against a real, live W&B project has not been
  performed in this build environment (no live `WANDB_API_KEY` /
  network access) — `WandbBackend` is tested against an in-memory fake
  client built from W&B's documented API shapes.
  `scripts/record_wandb_fixtures.py` is ready to run against your own
  project. See `tests/fixtures/README.md`.
- Tool-selection eval against a live MCP client has not been run in this
  environment (no `ANTHROPIC_API_KEY` / network access). See
  `docs/tool-selection-eval.md`.
- This is a W&B-only release for the MCP layer; MLflow support is
  prototyped at the interface level but not implemented.
- `audit_sweep`'s correlation-based ranking only detects linear
  relationships — see the sweep section of `docs/audit-methods.md`.

None of these are architectural gaps. They're either genuinely deferred
scope or blocked by this build environment's lack of live credentials —
see [Roadmap](#roadmap).

## Development

```bash
git clone https://github.com/SreeDharshan-GJ/experiment-audit-mcp.git
cd experiment-audit-mcp
pip install -e ".[dev]"
pytest tests/ -q        # 274 tests
ruff check src/ tests/  # lint
```

The reasoning engine's tests need no network access or credentials at
all — they run entirely on in-memory `Claim`/`Evidence` fixtures. The
MCP/W&B layer's tests run against `FakeBackend`, an in-memory test double
that can inject every adversarial state named in the design spec.

## Contributing

Contributions are welcome — please read [`CONTRIBUTING.md`](CONTRIBUTING.md)
first.

The MCP/W&B layer's v1 design (`docs/design-spec-v1.md`) is **frozen**:
changes to its tool schemas, model fields, or backend interface need an
explicit, logged design decision, not a silent PR. The reasoning
engine's six rules and their thresholds are newer and more open to
discussion — if you're proposing a change to rule logic (as opposed to
wiring), explain the reasoning-quality tradeoff you're making, not just
the code change.

## Roadmap

- **Near-term** — a W&B-run-to-claims/evidence adapter, so the MCP audit
  tools can hand their findings directly to the reasoning engine instead
  of requiring hand-built `Claim`/`EvidenceItem` objects.
- **v2** — MLflow backend for the MCP layer, a versioned API
  compatibility matrix, first public case study from a real project.
- **v3** — RL-specific pathology signals, proper multi-seed statistical
  tests, Optuna/Ray Tune sweep support, open to external `audit_*` and
  reasoning-rule contributions.

## Citing this project

If `experiment-audit` was useful in your research or workflow, a citation
or a link back is genuinely appreciated:

```bibtex
@software{experiment_audit,
  author  = {Sree Dharshan G J},
  title   = {experiment-audit: A Scientific Research Reasoning Engine for ML Experiments},
  year    = {2026},
  url     = {https://github.com/SreeDharshan-GJ/experiment-audit-mcp}
}
```

## Author

Built and maintained by **Sree Dharshan G J**.

[![GitHub](https://img.shields.io/badge/GitHub-SreeDharshan--GJ-181717?logo=github)](https://github.com/SreeDharshan-GJ)

If this project is useful to you, a ⭐ on the repo is the easiest way to
support it and helps others find it.

## License

MIT — see [`LICENSE`](LICENSE).
