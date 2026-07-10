#!/usr/bin/env python3
"""Tool-selection eval: run the fixed prompt set against a real MCP
client, per design-spec-v1.md §7 point 3 and the roadmap's Milestone 9
entry ("confirm real MCP clients actually invoke the right tool from
natural-language prompts, which unit tests can't verify").

**Written but not run as part of this milestone** — the build
environment this was implemented in has no `ANTHROPIC_API_KEY` and no
network access to `api.anthropic.com` (egress is allowlisted to package
registries only), mirroring exactly the situation `record_wandb_fixtures.py`
already documents for Milestone 3's fixture recording. Run it yourself,
locally, with real credentials, before treating Milestone 9's
tool-selection completion criteria ("All fixed prompts correctly invoke
their intended tool") as verified. `docs/tool-selection-eval.md` records
this status explicitly rather than silently asserting the criterion is
met.

What it does when run with credentials:
    1. Builds the MCP server (`build_server`) against a `FakeBackend`
       seeded with a few representative runs/sweeps, so a tool call that
       *does* get invoked resolves to real (fake) data rather than
       erroring out — a selection eval shouldn't be confounded by data
       the model can plausibly infer is missing.
    2. Lists the server's tool schemas via `fastmcp.Client` and converts
       them to the Anthropic Messages API's `tools` shape.
    3. Sends each prompt from `tool_selection_prompts.TOOL_SELECTION_PROMPTS`
       as a single user turn with `tool_choice: {"type": "auto"}`, and
       records the name of the first tool_use block the model produces
       (or None, if it answered in prose without calling anything).
    4. Writes a pass/fail report to stdout and, if requested, appends the
       machine-readable result to a JSON file for docs/tool-selection-eval.md
       to be regenerated from.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/tool_selection_eval.py [--model claude-sonnet-4-6] [--json out.json]

This script deliberately does not execute the tool the model selects —
Milestone 9's concern is *selection*, not round-trip correctness (that's
already covered by test_server.py and test_adversarial_mcp_layer.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tool_selection_prompts import TOOL_SELECTION_PROMPTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment_audit_mcp.backends.fake_backend import FakeBackend  # noqa: E402
from experiment_audit_mcp.models import MetricHistory, MetricPoint, Run, RunRef, Sweep  # noqa: E402
from experiment_audit_mcp.server import build_server  # noqa: E402

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-sonnet-4-6"

_ENTITY = "test-entity"
_PROJECT = "mamfac"


def _seeded_backend() -> FakeBackend:
    """A representative backend so a selected tool resolves to plausible
    data instead of erroring — see module docstring point 1."""
    backend = FakeBackend()
    ref_a = RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="xj29fk1a")
    ref_b = RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="run-b")
    backend.seed_run(
        Run(
            ref=ref_a,
            name="baseline",
            tags=["baseline"],
            status="finished",
            created_at=datetime(2026, 6, 1, tzinfo=UTC),
            config={"learning_rate": 0.001, "seed": 42},
            summary_metrics={"final_reward": 12.5},
        )
    )
    backend.seed_run(
        Run(
            ref=ref_b,
            name="ablation",
            tags=["ablation"],
            status="finished",
            created_at=datetime(2026, 6, 1, tzinfo=UTC),
            config={"learning_rate": 0.01, "seed": 42},
            summary_metrics={"final_reward": 11.0},
        )
    )
    backend.seed_metric_history(
        MetricHistory(
            ref=ref_a,
            metric_name="reward",
            points=[MetricPoint(step=i, value=1.0 + 0.01 * i) for i in range(20)],
        )
    )
    runs = [
        Run(
            ref=RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id=f"sweep-r{i}"),
            name=f"sweep-run-{i}",
            tags=[],
            status="finished",
            created_at=datetime(2026, 6, 1, tzinfo=UTC),
            config={"learning_rate": float(i), "batch_size": float(i) * 10, "seed": 42},
            summary_metrics={"reward": float(i)},
        )
        for i in range(1, 13)
    ]
    for run in runs:
        backend.seed_run(run)
    backend.seed_sweep(
        Sweep(
            ref=RunRef(backend="fake", entity=_ENTITY, project=_PROJECT, run_id="sweep-ref"),
            sweep_id="sweep-1",
            method="grid",
            run_refs=[r.ref for r in runs],
            target_metric="reward",
        )
    )
    return backend


async def _mcp_tools_as_anthropic_tools() -> list[dict]:
    from fastmcp import Client

    mcp = build_server(backends={"fake": _seeded_backend()})
    async with Client(mcp) as client:
        tools = await client.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in tools
    ]


def _call_claude(model: str, tools: list[dict], prompt: str) -> str | None:
    """Returns the name of the first tool_use block, or None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    body = json.dumps(
        {
            "model": model,
            "max_tokens": 512,
            "system": (
                "You are an assistant with access to experiment-audit-mcp, "
                "a set of tools for inspecting and auditing ML experiments "
                "tracked in Weights & Biases. Use whichever tool best "
                "answers the user's request."
            ),
            "tools": tools,
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        _ANTHROPIC_MESSAGES_URL,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())

    for block in payload.get("content", []):
        if block.get("type") == "tool_use":
            return block.get("name")
    return None


@dataclass
class EvalResult:
    prompt: str
    expected_tool: str
    invoked_tool: str | None
    passed: bool
    rationale: str


def run_eval(model: str) -> list[EvalResult]:
    tools = asyncio.run(_mcp_tools_as_anthropic_tools())
    results: list[EvalResult] = []
    for case in TOOL_SELECTION_PROMPTS:
        invoked = _call_claude(model, tools, case.prompt)
        results.append(
            EvalResult(
                prompt=case.prompt,
                expected_tool=case.expected_tool,
                invoked_tool=invoked,
                passed=(invoked == case.expected_tool),
                rationale=case.rationale,
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--json", type=Path, default=None, help="write machine-readable results here"
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. This script was written but not "
            "run as part of Milestone 9 for exactly this reason — see the "
            "module docstring and docs/tool-selection-eval.md. Set the key "
            "and re-run to actually validate tool selection.",
            file=sys.stderr,
        )
        return 1

    try:
        results = run_eval(args.model)
    except urllib.error.HTTPError as exc:
        print(f"Anthropic API request failed: {exc.code} {exc.reason}", file=sys.stderr)
        return 1

    n_passed = sum(r.passed for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] expected={r.expected_tool!r} invoked={r.invoked_tool!r}  {r.prompt}")
    print(f"\n{n_passed}/{len(results)} passed.")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "run_at": datetime.now(UTC).isoformat(),
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        print(f"Wrote machine-readable results to {args.json}")

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
