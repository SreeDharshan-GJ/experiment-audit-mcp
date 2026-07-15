# Changelog

## 1.1.0

**Identity reframing: MCP usage guide → Scientific Research Reasoning
Engine.** No changes to the reasoning engine's underlying logic, the
MCP server, or any tool schema/threshold/formula — this release is a
skill-layer rewrite only.

- `SKILL.md`: rewritten around the "reasoning engine with optional
  integrations" identity. Added explicit "when to use" / "when NOT to
  use" sections, a decision guide for choosing between direct reasoning
  and the MCP integration, and a general (not MCP-only) core reasoning
  discipline section. Tightened the frontmatter `description` for
  triggering on research-reasoning requests generally, not just W&B/MCP
  ones.
- `prompts.md`: added general sections — interpreting uncertainty,
  evaluating contradictory evidence, reviewing claims and papers,
  writing reviewer feedback, assessing reproducibility, and generating
  structured research reports. Original MCP-specific hedging and
  common-mistakes guidance preserved verbatim under a clearly-scoped
  "MCP integration" section.
- `examples.md`: original six MCP worked examples and three
  tool-selection distractor pairs preserved verbatim (Part 1). Added
  eight new reasoning-engine examples with no MCP call (Part 2): paper
  review, reviewer feedback, a bare statistical claim, reproducibility
  assessment, contradictory evidence across two sources, confidence/
  p-value interpretation, a prose-described ablation, and a
  publication-quality structured report.
- `reference.md`: unchanged except for a reframed introduction
  clarifying it documents the MCP integration specifically, not the
  reasoning engine as a whole. All schemas, thresholds, and formulas
  preserved exactly as v1.0.0 documented them.
- Added plugin packaging: `.claude-plugin/plugin.json`,
  `.claude-plugin/marketplace.json`, `.mcp.json`, this file, and
  `README.md`. The skill directory moved from `.claude/skills/` (a
  project-local install) to `skills/` at the plugin root (the layout
  Claude Code plugins expect), with a README note on how to install it
  either way.

## 1.0.0

Initial skill release, covering the eight `experiment-audit-mcp` tools:
tool selection, output interpretation, and scientific reasoning
discipline for relaying `audit_*` results without overclaiming.
