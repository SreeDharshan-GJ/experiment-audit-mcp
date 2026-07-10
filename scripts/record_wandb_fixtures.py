#!/usr/bin/env python3
"""One-off script to record real W&B API fixtures, per the convention in
tests/fixtures/README.md.

This is the recording script that README promises arrives in Milestone 3.
It was written but **not run** as part of this milestone — the build
environment has no WANDB_API_KEY and no network access to the W&B API
(egress is allowlisted to package registries only). Run it yourself,
locally, against a real project (MAMFAC/CARM++ recommended per the
roadmap's Milestone 3 deliverables) before treating WandbBackend's test
suite as satisfying spec §7's "recorded real API responses" requirement.

Usage:
    export WANDB_API_KEY=...          # a read-only key, per spec §6
    export WANDB_ENTITY=...           # optional; defaults to your default entity
    python scripts/record_wandb_fixtures.py --project mamfac --scenario happy_path_run

Writes to tests/fixtures/wandb/<scenario>/ as raw JSON, per the format
and `recorded_at`/library-version metadata tests/fixtures/README.md
specifies. Sanitize before committing: this script does not scrub
usernames or project names for you (README's "No secrets" convention is
a manual review step, not automated here).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    import wandb
    import wandb.apis.public as wandb_public
except ImportError:
    print("Install the project first: pip install -e '.[dev]'", file=sys.stderr)
    raise

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "wandb"


def record_run_scenario(
    api: wandb_public.Api, entity: str, project: str, run_id: str, scenario: str
) -> None:
    run = api.run(f"{entity}/{project}/{run_id}")
    out_dir = FIXTURES_ROOT / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    run_data = {
        "id": run.id,
        "name": run.name,
        "project": run.project,
        "entity": run.entity,
        "tags": list(run.tags or []),
        "state": run.state,
        "created_at": run.created_at,
        "config": dict(run.config or {}),
        "summary_metrics": dict(run.summary_metrics or {}),
    }
    (out_dir / "run.json").write_text(json.dumps(run_data, indent=2))

    history = list(run.scan_history())
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    metadata = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "wandb_sdk_version": wandb.__version__,
        "scenario": scenario,
        "source_path": f"{entity}/{project}/{run_id}",
    }
    (out_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {out_dir}/ — remember to review for secrets/PII before committing.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True, help="A specific run to record.")
    parser.add_argument(
        "--scenario", required=True, help="Fixture directory name, e.g. happy_path_run"
    )
    parser.add_argument("--entity", default=None, help="Defaults to your W&B default entity.")
    args = parser.parse_args()

    api = wandb_public.Api()
    entity = args.entity or api.default_entity
    record_run_scenario(api, entity, args.project, args.run_id, args.scenario)


if __name__ == "__main__":
    main()
