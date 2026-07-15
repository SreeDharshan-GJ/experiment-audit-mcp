# Experiment Audit — Scientific Research Reasoning Engine

A Claude Code skill (packaged as a plugin) that teaches Claude to reason
like a careful scientist about experimental and empirical claims:
auditing ML training runs, ablations, and sweeps; reviewing papers and
results sections; checking reproducibility; reconciling contradictory
evidence; and producing structured, checkable research reports.

The **reasoning engine is the product**. The `experiment-audit-mcp`
server (an optional integration for Weights & Biases projects), the CLI,
and the Python package are ways to feed it live evidence — the same
discipline applies whether or not any of them are used.

## What's in this repo

```
.claude-plugin/
  plugin.json         Plugin manifest
  marketplace.json     Self-referencing marketplace manifest (installs this repo directly)
skills/
  experiment-audit/
    SKILL.md            Entry point: identity, triggers, tool selection, core discipline
    prompts.md          Scientific reasoning + communication guidance (general, then MCP-specific)
    examples.md          Worked conversations: MCP tool use, then pure reasoning-engine cases
    reference.md         Exact experiment-audit-mcp tool schemas, thresholds, formulas
.mcp.json               Optional MCP server registration for the W&B integration
README.md
CHANGELOG.md
```

## Installing as a plugin

From within a Claude Code session:

```
/plugin marketplace add /path/to/this/repo      # or a github org/repo reference once hosted
/plugin install experiment-audit@experiment-audit
```

This installs the skill and registers the optional `experiment-audit`
MCP server declared in `.mcp.json`. The skill works fully without W&B
credentials configured — the MCP tools simply won't be available, and
Claude will reason directly over whatever data you provide instead (see
Part 2 of `examples.md`).

### Using the W&B integration

Set `WANDB_API_KEY` in your environment before starting Claude Code, or
configure it through your plugin's user config once installed. Update
the `command`/`args` in `.mcp.json` to match how you actually distribute
`experiment-audit-mcp` (a published npm package, a local build, a Docker
image, etc.) — the values here are a template, not a verified install
command.

## Using as a plain project skill (no plugin)

If you don't want the plugin/marketplace machinery, copy the skill
directory directly into a project:

```
cp -r skills/experiment-audit /your-project/.claude/skills/experiment-audit
```

Claude Code will pick it up automatically; you lose the bundled MCP
server registration and would need to configure `experiment-audit-mcp`
separately if you want the W&B integration.

## Before publishing

- Fill in `author`, `repository`, and `license` in `.claude-plugin/plugin.json`
  with real values.
- Verify the `command`/`args`/`env` in `.mcp.json` against your actual
  `experiment-audit-mcp` distribution method.
- If hosting this as a GitHub-based marketplace, update the `source`
  fields once the plugin lives in its own subdirectory versus repo root.

## Versioning

This is v1.1.0 — the skill's identity was reframed from "an MCP server's
usage guide" to "a scientific research reasoning engine with an optional
MCP integration." See `CHANGELOG.md` for details. `reference.md` still
documents `experiment-audit-mcp` v1.0.0 unchanged; this release did not
touch the MCP server, CLI, or Python package implementations.
