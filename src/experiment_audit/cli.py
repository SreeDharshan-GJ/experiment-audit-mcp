"""Command-line entry point for Experiment Audit's Scientific Reasoning
Engine (the concrete, six-rule `ScientificReasoningPipeline` --
see `reasoning/pipeline.py`).

This module performs no reasoning of its own. It only:

1. Reads a JSON file describing `Claim`s and `EvidenceItem`s, in the
   schema documented in `_load_claim` / `_load_evidence_item` below,
   and constructs the real dataclasses those functions are named
   after -- a mechanical, one-field-at-a-time conversion, not a new
   claim-extraction or evidence-interpretation algorithm.
2. Hands the result to `ScientificReasoningPipeline`, exactly as any
   other caller of the public API would.
3. Prints the resulting `ScientificReport` in the caller's requested
   format, using that class's own `to_markdown()` / `to_json()` /
   `to_text()` methods.

Usage:

    experiment-audit reasoning run --input claims.json
    experiment-audit reasoning run --input claims.json --format json --output report.json

See `experiment-audit reasoning schema` for the input JSON's exact shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiment_audit import __version__
from experiment_audit.models import RunRef
from experiment_audit.reasoning.claims import (
    Claim,
    ClaimCategory,
    ClaimLifecycleStage,
    ClaimSet,
    ClaimStrength,
    Scope,
    UnsupportedReason,
)
from experiment_audit.reasoning.evidence import Evidence, EvidenceItem, EvidenceKind
from experiment_audit.reasoning.pipeline import ScientificReasoningPipeline
from experiment_audit.reasoning.scientific_report import ScientificReport

_UNSCOPED_REF = RunRef(backend="cli", entity="unscoped", project="unscoped", run_id="unscoped")

_SCHEMA_EXAMPLE: dict[str, Any] = {
    "claims": [
        {
            "id": "c1",
            "subject": "model-x",
            "statement": "model-x achieves 95% accuracy on CIFAR-10",
            "category": "performance",
            "scope": {"dataset": "cifar-10"},
            "lifecycle_stage": "formulated",
            "strength": None,
            "evidence_trace": [
                {"backend": "wandb", "entity": "team", "project": "proj", "run_id": "run-1"}
            ],
            "unsupported_reasons": [],
            "metadata": {},
        }
    ],
    "evidence": [
        {
            "kind": "metric",
            "key": "accuracy",
            "value": 0.95,
            "source": {
                "backend": "wandb",
                "entity": "team",
                "project": "proj",
                "run_id": "run-1",
            },
            "note": None,
        }
    ],
    "metadata": {},
}


def _load_run_ref(data: dict[str, Any] | None) -> RunRef | None:
    if data is None:
        return None
    return RunRef(
        backend=data["backend"],
        entity=data["entity"],
        project=data["project"],
        run_id=data["run_id"],
    )


def _load_scope(data: dict[str, Any] | None) -> Scope:
    if data is None:
        return Scope()
    return Scope(
        dataset=data.get("dataset"),
        model=data.get("model"),
        hardware=data.get("hardware"),
        evaluation_protocol=data.get("evaluation_protocol"),
        software_environment=data.get("software_environment"),
        additional_constraints=data.get("additional_constraints", {}),
    )


def _load_claim(data: dict[str, Any]) -> Claim:
    """Construct a `Claim` from one entry of the input JSON's `claims`
    array. A mechanical field-by-field conversion -- see this module's
    docstring."""
    return Claim(
        id=data["id"],
        subject=data["subject"],
        statement=data["statement"],
        category=ClaimCategory(data["category"]),
        scope=_load_scope(data.get("scope")),
        lifecycle_stage=ClaimLifecycleStage(
            data.get("lifecycle_stage", ClaimLifecycleStage.FORMULATED.value)
        ),
        strength=ClaimStrength(data["strength"]) if data.get("strength") else None,
        evidence_trace=tuple(
            ref for ref in (_load_run_ref(r) for r in data.get("evidence_trace", [])) if ref
        ),
        unsupported_reasons=tuple(
            UnsupportedReason(r) for r in data.get("unsupported_reasons", [])
        ),
        metadata=data.get("metadata", {}),
    )


def _load_evidence_item(data: dict[str, Any]) -> EvidenceItem:
    """Construct an `EvidenceItem` from one entry of the input JSON's
    `evidence` array. A mechanical field-by-field conversion -- see
    this module's docstring."""
    return EvidenceItem(
        kind=EvidenceKind(data["kind"]),
        key=data["key"],
        value=data["value"],
        source=_load_run_ref(data.get("source")),
        note=data.get("note"),
    )


def _load_evidence_bundles(raw_items: list[dict[str, Any]]) -> list[Evidence]:
    """Group a flat list of evidence-item dicts into `Evidence` bundles.

    `RuleContext.evidence` expects `Evidence` bundles (each exposing an
    `.items` list), not bare `EvidenceItem`s -- `Evidence.items` has
    `init=False` and is populated only via `Evidence.add_item()`. Items
    with the same `source` `RunRef` are grouped into one bundle; items
    with no declared `source` are grouped into a single bundle keyed
    by a placeholder "unscoped" ref.
    """
    bundles: dict[RunRef, Evidence] = {}
    for item_data in raw_items:
        item = _load_evidence_item(item_data)
        ref = item.source or _UNSCOPED_REF
        if ref not in bundles:
            bundles[ref] = Evidence(ref=ref)
        bundles[ref].add_item(item)
    return list(bundles.values())


def _run_reasoning(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    try:
        raw = json.loads(input_path.read_text())
    except FileNotFoundError:
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: input file is not valid JSON: {exc}", file=sys.stderr)
        return 1

    try:
        claims = ClaimSet(_load_claim(c) for c in raw.get("claims", []))
        evidence = _load_evidence_bundles(raw.get("evidence", []))
    except (KeyError, ValueError) as exc:
        print(
            f"error: malformed input (see 'experiment-audit reasoning schema'): {exc}",
            file=sys.stderr,
        )
        return 1

    pipeline = ScientificReasoningPipeline()
    context = pipeline.build_initial_context(
        claims=claims,
        evidence=evidence,
        metadata=raw.get("metadata", {}),
    )
    pipeline_report = pipeline.execute(context)
    report = ScientificReport.from_pipeline_report(pipeline_report)

    rendered = {
        "markdown": report.to_markdown,
        "json": report.to_json,
        "text": report.to_text,
    }[args.format]()

    if args.output:
        Path(args.output).write_text(rendered)
        print(f"Report written to {args.output}")
    else:
        print(rendered)
    return 0


def _print_schema(_: argparse.Namespace) -> int:
    print(json.dumps(_SCHEMA_EXAMPLE, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment-audit",
        description="Experiment Audit's Scientific Reasoning Engine CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    reasoning_parser = subparsers.add_parser(
        "reasoning", help="Run the Scientific Reasoning Engine."
    )
    reasoning_subparsers = reasoning_parser.add_subparsers(
        dest="reasoning_command", required=True
    )

    run_parser = reasoning_subparsers.add_parser(
        "run", help="Run claims/evidence through the six-rule reasoning pipeline."
    )
    run_parser.add_argument(
        "--input", required=True, help="Path to a JSON file of claims/evidence."
    )
    run_parser.add_argument(
        "--format",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Output format for the resulting ScientificReport (default: markdown).",
    )
    run_parser.add_argument(
        "--output", default=None, help="Write the report to this path instead of stdout."
    )
    run_parser.set_defaults(func=_run_reasoning)

    schema_parser = reasoning_subparsers.add_parser(
        "schema", help="Print an example of the input JSON schema `reasoning run` expects."
    )
    schema_parser.set_defaults(func=_print_schema)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
