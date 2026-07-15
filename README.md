<div align="center">

<img src="docs/assets/banner.svg" alt="experiment-audit — a scientific reasoning engine for ML experiments" width="100%">

# experiment-audit

**A scientific reasoning engine for ML experiments.**

Feed it claims and evidence — it checks for missing support, scopes evidence to claims,
catches contradictions, scores confidence, and renders a structured scientific report.
The kind of review a careful advisor would give your results before you write them up.

[![PyPI](https://img.shields.io/pypi/v/experiment-audit)](https://pypi.org/project/experiment-audit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/SreeDharshan-GJ/experiment-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/SreeDharshan-GJ/experiment-audit/actions)
[![Built with FastMCP](https://img.shields.io/badge/MCP-FastMCP-purple.svg)](https://github.com/jlowin/fastmcp)
[![Status](https://img.shields.io/badge/status-v1.1.0-brightgreen.svg)](CHANGELOG.md)

Created and maintained by **[Sree Dharshan G J](https://github.com/SreeDharshan-GJ)**

</div>

<br>

> Most experiment-tracking tools show you numbers. `experiment-audit` checks whether your
> *claim* about those numbers actually holds up — missing evidence, out-of-scope
> comparisons, contradictions with earlier results, and confidence that isn't just assumed.

<br>

## Install

```bash
pip install experiment-audit
```

Requires Python 3.11+. See [Quick start](#quick-start-30-seconds) below, or jump straight
to [Claude Code integration](#claude-code-compatible).

<br>

## Quick start (30 seconds)

```python
from experiment_audit.reasoning import (
    ScientificReasoningPipeline, ScientificReport,
    Claim, ClaimCategory, Scope,
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
report = ScientificReport.from_pipeline_report(pipeline.execute(context))

print(report.to_markdown())
```

**Claim → Evidence → Reasoning → Scientific Report.** Every finding in the output traces
back to specific evidence — the engine doesn't assert anything it can't point to.

Prefer the command line?

```bash
experiment-audit reasoning schema > claims.json   # see the expected input shape
experiment-audit reasoning run --input claims.json --format markdown
```

<br>

## Claude Code Compatible

<img src="docs/assets/integration-badges.svg" alt="Integrations: Claude Code, MCP, Weights and Biases, Python, CLI">

The reasoning discipline behind this project — how to phrase findings, weigh
contradictory evidence, and write structured reviewer-style feedback — ships as a
[Claude Code](https://docs.claude.com/claude-code) skill, with the eight MCP audit tools
available automatically wherever `WANDB_API_KEY` is set.

<sub>**◆ Claude Code** — run from inside this repo</sub>

```
/plugin marketplace add ./dev/experiment-audit-plugin
/plugin install experiment-audit@experiment-audit
```

It triggers automatically on prompts like:

- *"Is this ablation confounded?"*
- *"Why did my loss crash?"*
- *"Review this paper's results claim."*
- *"Compare these ablation studies."*
- *"Write reviewer feedback on this."*

See [Quick start: the MCP server](#quick-start-the-mcp-server-wb-audit-tools) below for
manual MCP setup, or the
[plugin's own README](dev/experiment-audit-plugin/README.md) for full details.

<br>

## Why experiment-audit?

Traditional experiment trackers are good at one thing: displaying metrics. They will
happily tell you a run's final accuracy, loss curve, or config diff. What none of them do
is check whether the *sentence you're about to write about those numbers* is actually
supported by them.

`experiment-audit` treats a result the way a careful reviewer would before publication:

- Is there evidence behind this claim at all, or is it an assumption that snuck in?
- Does the evidence actually match what's being claimed — same dataset, same protocol,
  same scope — or is it being stretched to cover more than it proves?
- Does anything else you've measured contradict it?
- Is the confidence in the write-up proportional to the evidence, or borrowed from how
  confident the result *felt*?

This matters most for reproducibility, ablations, and the kind of paper-writing claims
that are easy to overstate under deadline pressure — the exact places research claim
verification tends to break down silently.

<br>

## Who is this for

| | |
|---|---|
| **ML engineers** | sanity-check a result before it ships in a report or a PR description |
| **AI researchers** | catch confounded ablations and out-of-scope comparisons before submission |
| **Graduate students** | get reviewer-style feedback on a results section before your advisor does |
| **Research labs** | a shared, deterministic check for scientific claims across a team's experiments |
| **Academic / open-source projects** | structured, evidence-traced scientific reports instead of ad hoc write-ups |

<br>

## What the reasoning engine does

Given a set of **claims** (*"model-x achieves 95% accuracy on CIFAR-10"*) and the
**evidence** backing them (metrics, configs, logs, prior runs), the engine runs six rules
in sequence and produces a `ScientificReport`:

| # | Rule | What it checks |
|---|------|-----------------|
| 1 | **Missing evidence** | Does this claim have any supporting evidence trace at all? |
| 2 | **Scope** | Does the evidence actually match the claim's stated scope (same dataset, same hardware, same evaluation protocol)? |
| 3 | **Contradiction** | Does any other claim or evidence item conflict with this one? |
| 4 | **Confidence** | A computed score, not a guess — based on evidence quality, quantity, contradictions found, and what's missing. |
| 5 | **Judgment** | A verdict (supported / partially supported / unsupported) with the reasoning behind it. |
| 6 | **Recommendation** | What to do about it — gather more evidence, narrow the claim's scope, retract it. |

Every finding traces back to specific evidence. Nothing in the report is an unsupported
assertion — that would rather defeat the point.

> This is one of two reasoning pipelines in the package. The second, lower-level pipeline
> (`ScientificReasoningEngine` — Evidence → Observations → Hypotheses → Confidence →
> Judgment → Recommendation) is a more generic, extensible framework for injecting custom
> hypothesis and confidence logic. Most people should start with the six-rule pipeline
> above. See `src/experiment_audit/reasoning/__init__.py` for both.

<br>

## Features

**Reasoning engine (the core)**

- Claim and evidence modeling (`Claim`, `EvidenceItem`, `Scope`) with structured categories
- Six-rule scientific reasoning pipeline, run end-to-end or rule-by-rule
- Contradiction detection across claims and evidence
- Confidence scoring driven by evidence quality/quantity, not a fixed heuristic
- Structured `ScientificReport` — Markdown, JSON, or plain text
- Zero network calls; runs entirely on data you provide

**Interfaces around the engine**

- **Python API** — `experiment_audit.reasoning`, for embedding the pipeline in your own tooling
- **CLI** — `experiment-audit reasoning run|schema`
- **Claude Code skill** — the reasoning discipline as an installable skill, with worked examples
- **MCP server** — eight tools for auditing Weights & Biases runs directly from an agent
- **Weights & Biases backend** — read-only run/sweep/metric access behind the MCP tools

<br>

## Quick start: the MCP server (W&B audit tools)

The original W&B `experiment-audit` tools are still here, unchanged, as an MCP
integration. Set a **read-only** W&B API key:

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

<sub>**◆ Claude Code**</sub>

```bash
claude mcp add -e WANDB_API_KEY=your-read-only-key experiment-audit -- experiment-audit-mcp
```

> `-e` must come before the server name, not after — putting it after the name has been a
> source of "Invalid environment variable format" errors in some Claude Code versions.

Then ask your agent something like:

> "Did I mess up my memory-ablation run? Compare `mamfac-baseline` and
> `mamfac-no-memory` in the `mamfac` project and check whether the only real difference
> is `use_memory`."

The agent calls `audit_ablation`, which returns a verdict
(`clean` / `confounded` / `uncertain`), a confidence level, and the full config diff it
based that verdict on.

Full tool reference (all eight tools, exact schemas, methodology) is in
[`docs/design-spec-v1.md`](docs/design-spec-v1.md) and
[`docs/audit-methods.md`](docs/audit-methods.md) — unchanged from the v1.0.0 release.

<br>

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

The reasoning engine and the MCP/W&B layer are independent — the reasoning engine takes
`Claim`s and `EvidenceItem`s directly and has no dependency on W&B, FastMCP, or any
backend. Feeding W&B run data into the reasoning engine as claims/evidence (rather than
hand-constructing them, as the quick-start example above does) is on the
[roadmap](#roadmap).

For the reasoning engine's design rationale, see
[`research/07_reasoning_engine/`](research/07_reasoning_engine/)
(`reasoning-engine.md`, `reasoning-rules.md`, `confidence-system.md`, `evidence-model.md`,
`scientific-reviewer.md`). For the MCP/W&B layer's frozen contract, see
[`docs/design-spec-v1.md`](docs/design-spec-v1.md).

<br>

## Data handling

- Data never leaves your machine except calls to your own W&B endpoint (MCP layer only —
  the reasoning engine itself makes no network calls at all).
- Credentials are read once from environment variables, validated fail-fast on server
  start, and never logged.
- Use a **read-only** W&B API key — this server has no write path.

<br>

## Known gaps (honest status)

- No built-in adapter converts a W&B run directly into `Claim`s/`EvidenceItem`s yet — you
  construct them yourself (CLI schema or Python), or write your own extraction step. This
  is the top [roadmap](#roadmap) item.
- The generic pipeline (`ScientificReasoningEngine`) defaults its rule-engine stage to a
  no-op unless you inject one — it's an extensibility point, not a second complete
  pipeline.
- 274 tests pass (`pytest tests/ -q`); this is real coverage of the pipeline's mechanics,
  not a substitute for domain review of the six rules' thresholds by someone in your
  research area.
- The MCP/W&B layer is W&B-only for now (MLflow is prototyped at the interface level, not
  implemented), and `audit_sweep`'s correlation ranking only detects linear relationships.

Full detail, including what's blocked purely by this build environment's lack of live
credentials, is in [`docs/design-spec-v1.md`](docs/design-spec-v1.md) and the
[CHANGELOG](CHANGELOG.md).

<br>

## Development

```bash
git clone https://github.com/SreeDharshan-GJ/experiment-audit.git
cd experiment-audit
pip install -e ".[dev]"
pytest tests/ -q        # 274 tests
ruff check src/ tests/  # lint
```

The reasoning engine's tests need no network access or credentials at all — they run
entirely on in-memory `Claim`/`Evidence` fixtures. The MCP/W&B layer's tests run against
`FakeBackend`, an in-memory test double that can inject every adversarial state named in
the design spec.

<br>

## Contributing

Contributions are welcome — please read [`CONTRIBUTING.md`](CONTRIBUTING.md) first.

The MCP/W&B layer's v1 design (`docs/design-spec-v1.md`) is **frozen**: changes to its
tool schemas, model fields, or backend interface need an explicit, logged design decision,
not a silent PR. The reasoning engine's six rules and their thresholds are newer and more
open to discussion — if you're proposing a change to rule logic (as opposed to wiring),
explain the reasoning-quality tradeoff you're making, not just the code change.

<br>

## Roadmap

- **Near-term** — a W&B-run-to-claims/evidence adapter, so the MCP audit tools can hand
  their findings directly to the reasoning engine instead of requiring hand-built
  `Claim`/`EvidenceItem` objects.
- **v2** — MLflow backend for the MCP layer, a versioned API compatibility matrix, first
  public case study from a real project.
- **v3** — RL-specific pathology signals, proper multi-seed statistical tests,
  Optuna/Ray Tune sweep support, open to external `audit_*` and reasoning-rule
  contributions.

<br>

## Citing this project

If `experiment-audit` was useful in your research or workflow, a citation or a link back
is genuinely appreciated:

```bibtex
@software{experiment_audit,
  author  = {Sree Dharshan G J},
  title   = {experiment-audit: A Scientific Research Reasoning Engine for ML Experiments},
  year    = {2026},
  url     = {https://github.com/SreeDharshan-GJ/experiment-audit}
}
```

<br>

## Author

Built and maintained by **Sree Dharshan G J**.

[![GitHub](https://img.shields.io/badge/GitHub-SreeDharshan--GJ-181717?logo=github)](https://github.com/SreeDharshan-GJ)

If this project is useful to you, a star on the repo is the easiest way to support it and
helps others find it.

<br>

## License

MIT — see [`LICENSE`](LICENSE).
<svg width="720" height="56" viewBox="0 0 720 56" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc">
  <title id="title">Integrations: Claude Code, MCP, Weights and Biases, Python, CLI</title>
  <desc id="desc">Pill badges listing experiment-audit's supported integrations</desc>

  <rect x="0" y="0" width="720" height="56" fill="none"/>

  <g font-family="'Segoe UI', Helvetica, Arial, sans-serif" font-size="13" font-weight="500">

    <!-- Claude Code -->
    <rect x="0" y="12" width="132" height="32" rx="16" fill="#151922" stroke="#2A303C" stroke-width="1"/>
    <path d="M14 28 l2.4 -5.4 L18.8 28 l-2.4 5.4 Z M11.6 28 h5.6 M14.4 22.6 v10.8" stroke="#F0B37E" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    <text x="27" y="32.5" fill="#E5E7EB">Claude Code</text>

    <!-- MCP -->
    <rect x="140" y="12" width="70" height="32" rx="16" fill="#151922" stroke="#2A303C" stroke-width="1"/>
    <circle cx="154" cy="28" r="3.4" fill="none" stroke="#5EEAD4" stroke-width="1.4"/>
    <text x="163" y="32.5" fill="#E5E7EB">MCP</text>

    <!-- Weights and Biases -->
    <rect x="218" y="12" width="130" height="32" rx="16" fill="#151922" stroke="#2A303C" stroke-width="1"/>
    <path d="M231 32 l2.2 -8 l2.2 5.5 l2.2 -5.5 l2.2 8" fill="none" stroke="#FBBF24" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="244" y="32.5" fill="#E5E7EB">Weights &amp; Biases</text>

    <!-- Python -->
    <rect x="356" y="12" width="88" height="32" rx="16" fill="#151922" stroke="#2A303C" stroke-width="1"/>
    <circle cx="370" cy="25" r="3" fill="#60A5FA"/>
    <circle cx="370" cy="31" r="3" fill="#FDE68A"/>
    <text x="380" y="32.5" fill="#E5E7EB">Python</text>

    <!-- CLI -->
    <rect x="452" y="12" width="66" height="32" rx="16" fill="#151922" stroke="#2A303C" stroke-width="1"/>
    <path d="M463 24 l4 4 l-4 4 M469 32 h6" fill="none" stroke="#9CA3AF" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="480" y="32.5" fill="#E5E7EB">CLI</text>

  </g>
</svg>

