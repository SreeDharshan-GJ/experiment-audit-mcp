#!/usr/bin/env python3
"""Record real W&B API fixtures, per the convention in
tests/fixtures/README.md and the scenario plan in docs/wandb_fixture_plan.md.

This is a genuine recording script: every scenario it writes is captured
from a live call against your real W&B account (via WANDB_API_KEY). It
never fabricates a fixture — "error" scenarios record the actual
exception W&B raised, not a guessed one; if the call you asked it to
record doesn't fail, it says so and writes nothing.

Usage (see docs/wandb_fixture_plan.md for the full scenario list and the
regression each one guards against):

    export WANDB_API_KEY=...          # a read-only key, per spec §6
    export WANDB_ENTITY=...           # optional; defaults to your default entity

    # A single run, mapped fields + full metric history
    python scripts/record_wandb_fixtures.py run \\
        --project mamfac --run-id abc123 --scenario happy_path_run \\
        --metric reward

    # A project's run listing (pagination scenarios: use --per-page smaller
    # than the project's real run count to force paginator.more=True)
    python scripts/record_wandb_fixtures.py list-runs \\
        --project mamfac --scenario multiple_runs_paginated_project \\
        --per-page 2

    # An expected failure: run/project doesn't exist, or key lacks access
    python scripts/record_wandb_fixtures.py run-error \\
        --project mamfac --run-id does-not-exist --scenario run_not_found

    python scripts/record_wandb_fixtures.py list-runs-error \\
        --project some-private-project --scenario private_project_list_runs

    # Sweeps
    python scripts/record_wandb_fixtures.py sweeps \\
        --project mamfac --scenario sweep_with_runs

    # Auth failure (pass a deliberately bad key via --bad-api-key so your
    # real WANDB_API_KEY env var is untouched)
    python scripts/record_wandb_fixtures.py auth-error \\
        --bad-api-key not-a-real-key --scenario auth_failure_invalid_key

Every fixture is written to tests/fixtures/wandb/<scenario>/fixture.json
plus a sibling _metadata.json (recorded_at, wandb SDK version, scenario,
source path). Sanitization (entity name, project name if you pass
--sanitize-project, and secret-shaped strings) is applied automatically
before anything is written — see `_sanitize` below — but review the
written fixture yourself before committing; automated sanitization is a
safety net, not a substitute for a human pass over real data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

try:
    import wandb.apis.public as wandb_public
except ImportError:
    print("Install the project first: pip install -e '.[dev]'", file=sys.stderr)
    raise

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "wandb"

# ---------------------------------------------------------------------------
# Sanitization — applied automatically to every fixture before it's written.
# This is a safety net (secret-shaped strings, the entity name, and any
# --sanitize-project value), not a substitute for a human review pass, per
# tests/fixtures/README.md's "No secrets" convention.
# ---------------------------------------------------------------------------

# Matches strings that look like API keys / tokens: long runs of hex or
# base62-ish characters, the shape W&B and most SaaS tokens use.
_SECRET_LIKE_RE = re.compile(r"\b[a-fA-F0-9]{32,}\b|\b[A-Za-z0-9_-]{24,}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _sanitize_string(value: str, replacements: dict[str, str]) -> str:
    for real, placeholder in replacements.items():
        if real:
            value = value.replace(real, placeholder)
    value = _EMAIL_RE.sub("<email-redacted>", value)
    value = _SECRET_LIKE_RE.sub(
        lambda m: "<redacted-token>" if _looks_like_secret(m.group(0)) else m.group(0),
        value,
    )
    return value


def _looks_like_secret(token: str) -> bool:
    """Heuristic: only flag tokens that are plausibly a key/id, not e.g. a
    normal English word or a short config value that happens to be 24+
    chars. Requires a mix of letters and digits (pure words rarely have
    digits; pure hashes/tokens almost always do)."""
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    return has_digit and has_alpha


def _sanitize(obj: Any, replacements: dict[str, str]) -> Any:
    if isinstance(obj, str):
        return _sanitize_string(obj, replacements)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v, replacements) for v in obj]
    return obj


def _write_fixture(scenario: str, kind: str, data: dict[str, Any], source_path: str) -> Path:
    out_dir = FIXTURES_ROOT / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = {"kind": kind, **data}
    (out_dir / "fixture.json").write_text(json.dumps(fixture, indent=2))
    metadata = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "wandb_sdk_version": version("wandb"),
        "scenario": scenario,
        "kind": kind,
        "source_path": source_path,
    }
    (out_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {out_dir}/ — review for secrets/PII before committing, then run:")
    print(f"  git diff --stat {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Scenario recorders
# ---------------------------------------------------------------------------


def record_run(
    api: wandb_public.Api,
    entity: str,
    project: str,
    run_id: str,
    scenario: str,
    metric: str | None,
    replacements: dict[str, str],
) -> None:
    run = api.run(f"{entity}/{project}/{run_id}")
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
    history: list[dict[str, Any]] = []
    if metric is not None:
        history = list(run.scan_history(keys=[metric]))
    data = {"run": run_data, "history": history, "history_metric": metric}
    data = _sanitize(data, replacements)
    _write_fixture(scenario, "run", data, f"{entity}/{project}/{run_id}")


def record_run_error(
    api: wandb_public.Api,
    entity: str,
    project: str,
    run_id: str,
    scenario: str,
    replacements: dict[str, str],
) -> None:
    try:
        api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:  # noqa: BLE001 - the failure itself is the fixture
        data = _sanitize(
            {"exception_type": type(exc).__name__, "exception_message": str(exc)},
            replacements,
        )
        _write_fixture(scenario, "run_error", data, f"{entity}/{project}/{run_id}")
        return
    print(
        f"No exception was raised fetching {entity}/{project}/{run_id} — "
        "this scenario expects a failure. Nothing written. Pick a run_id "
        "that actually triggers the error you're recording.",
        file=sys.stderr,
    )
    sys.exit(1)


def record_list_runs(
    api: wandb_public.Api,
    entity: str,
    project: str,
    scenario: str,
    per_page: int,
    replacements: dict[str, str],
) -> None:
    paginator = api.runs(path=f"{entity}/{project}", per_page=per_page)
    page_items = list(paginator[0:per_page])
    runs_data = [
        {
            "id": r.id,
            "name": r.name,
            "project": r.project,
            "entity": r.entity,
            "tags": list(r.tags or []),
            "state": r.state,
            "created_at": r.created_at,
            "config": dict(r.config or {}),
            "summary_metrics": dict(r.summary_metrics or {}),
        }
        for r in page_items
    ]
    data = {"runs": runs_data, "more": bool(paginator.more), "per_page": per_page}
    data = _sanitize(data, replacements)
    _write_fixture(scenario, "list_runs", data, f"{entity}/{project}")


def record_list_runs_error(
    api: wandb_public.Api,
    entity: str,
    project: str,
    scenario: str,
    replacements: dict[str, str],
) -> None:
    try:
        paginator = api.runs(path=f"{entity}/{project}")
        items = list(paginator[0:50])
    except Exception as exc:  # noqa: BLE001
        data = _sanitize(
            {"exception_type": type(exc).__name__, "exception_message": str(exc)},
            replacements,
        )
        _write_fixture(scenario, "list_runs_error", data, f"{entity}/{project}")
        return
    data = _sanitize({"returned_empty": len(items) == 0, "count": len(items)}, replacements)
    _write_fixture(scenario, "list_runs_error", data, f"{entity}/{project}")
    print(
        "No exception was raised. Recorded that the call instead returned "
        f"{len(items)} run(s) — this is itself the finding for a "
        "permission-denied-vs-empty-page scenario; check fixture.json.",
        file=sys.stderr,
    )


def record_sweeps(
    api: wandb_public.Api,
    entity: str,
    project: str,
    scenario: str,
    replacements: dict[str, str],
) -> None:
    project_handle = api.project(project, entity)
    sweeps_data = []
    for sweep in project_handle.sweeps():
        sweeps_data.append(
            {
                "id": sweep.id,
                "config": dict(sweep.config or {}),
                "run_ids": [r.id for r in (sweep.runs or [])],
            }
        )
    data = {"sweeps": sweeps_data}
    data = _sanitize(data, replacements)
    _write_fixture(scenario, "sweeps", data, f"{entity}/{project}")


def record_auth_error(bad_api_key: str, scenario: str, replacements: dict[str, str]) -> None:
    try:
        bad_api = wandb_public.Api(api_key=bad_api_key)
        _ = bad_api.default_entity  # forces auth
    except Exception as exc:  # noqa: BLE001
        data = _sanitize(
            {"exception_type": type(exc).__name__, "exception_message": str(exc)},
            replacements,
        )
        _write_fixture(scenario, "auth_error", data, "auth (default_entity)")
        return
    print(
        "No exception was raised with the bad key — it may have "
        "authenticated successfully. Nothing written.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--scenario", required=True, help="Fixture directory name.")
        p.add_argument("--entity", default=None, help="Defaults to your W&B default entity.")
        p.add_argument(
            "--sanitize-project",
            default=None,
            help="If set, replace this literal project name with <project> in the fixture.",
        )

    p_run = sub.add_parser("run", help="Record a single run + optional metric history.")
    add_common(p_run)
    p_run.add_argument("--project", required=True)
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--metric", default=None, help="Metric to scan_history for.")

    p_run_err = sub.add_parser("run-error", help="Record a real failure fetching a run.")
    add_common(p_run_err)
    p_run_err.add_argument("--project", required=True)
    p_run_err.add_argument("--run-id", required=True)

    p_list = sub.add_parser("list-runs", help="Record a project's run listing (one page).")
    add_common(p_list)
    p_list.add_argument("--project", required=True)
    p_list.add_argument("--per-page", type=int, default=50)

    p_list_err = sub.add_parser("list-runs-error", help="Record a real failure listing runs.")
    add_common(p_list_err)
    p_list_err.add_argument("--project", required=True)

    p_sweeps = sub.add_parser("sweeps", help="Record a project's sweeps.")
    add_common(p_sweeps)
    p_sweeps.add_argument("--project", required=True)

    p_auth = sub.add_parser("auth-error", help="Record a real auth failure with a bad key.")
    p_auth.add_argument("--scenario", required=True)
    p_auth.add_argument("--bad-api-key", required=True)

    args = parser.parse_args()

    if args.command == "auth-error":
        record_auth_error(args.bad_api_key, args.scenario, replacements={})
        return

    api = wandb_public.Api()
    entity = args.entity or api.default_entity
    replacements: dict[str, str] = {entity: "<entity>"}
    if getattr(args, "sanitize_project", None):
        replacements[args.sanitize_project] = "<project>"

    if args.command == "run":
        record_run(
            api, entity, args.project, args.run_id, args.scenario, args.metric, replacements
        )
    elif args.command == "run-error":
        record_run_error(api, entity, args.project, args.run_id, args.scenario, replacements)
    elif args.command == "list-runs":
        record_list_runs(api, entity, args.project, args.scenario, args.per_page, replacements)
    elif args.command == "list-runs-error":
        record_list_runs_error(api, entity, args.project, args.scenario, replacements)
    elif args.command == "sweeps":
        record_sweeps(api, entity, args.project, args.scenario, replacements)


if __name__ == "__main__":
    main()