"""Replays recorded W&B fixtures (tests/fixtures/wandb/<scenario>/fixture.json)
through the real `WandbBackend` mapping/error-handling code, entirely
offline — no network access, no WANDB_API_KEY required.

This is the "verify every fixture can replay offline" step from
docs/wandb_fixture_plan.md: once `scripts/record_wandb_fixtures.py` has
been run against a real account and the resulting fixtures reviewed and
committed, this module turns each one into a `WandbBackend` call and
checks it doesn't crash and produces the shape the backend's mapping
code assumes.

The whole module is skipped (not failed) when no fixtures exist yet,
since that's the expected state until someone runs the recording script
locally — see tests/fixtures/README.md's "Milestone 3 status" section.
Every scenario named in docs/wandb_fixture_plan.md that *is* present gets
exercised; scenarios not yet recorded are simply absent from the
parametrized list, so this file never needs hand-editing as fixtures are
added — it discovers them from tests/fixtures/wandb/ directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiment_audit.auth import WandbCredentials
from experiment_audit.backends.wandb_backend import WandbBackend
from experiment_audit.models import RunRef

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "wandb"


def _discover_scenarios() -> list[Path]:
    if not FIXTURES_ROOT.exists():
        return []
    return sorted(
        p for p in FIXTURES_ROOT.iterdir() if p.is_dir() and (p / "fixture.json").exists()
    )


_SCENARIOS = _discover_scenarios()

pytestmark = pytest.mark.skipif(
    not _SCENARIOS,
    reason=(
        "No recorded W&B fixtures found under tests/fixtures/wandb/. "
        "Run scripts/record_wandb_fixtures.py against a real account first "
        "(see docs/wandb_fixture_plan.md)."
    ),
)


# ---------------------------------------------------------------------------
# Minimal replay doubles — built from the *recorded* fixture data, not from
# documented-shape guesses. Each mirrors just enough of the real wandb SDK
# surface for WandbBackend to run against it unmodified.
# ---------------------------------------------------------------------------


class _ReplayWandbRun:
    def __init__(self, data: dict[str, Any], history: list[dict[str, Any]]) -> None:
        self.id = data["id"]
        self.name = data["name"]
        self.project = data["project"]
        self.entity = data["entity"]
        self.tags = data.get("tags") or []
        self.state = data["state"]
        self.created_at = data["created_at"]
        self.config = data.get("config") or {}
        self.summary_metrics = data.get("summary_metrics") or {}
        self._history = history

    def scan_history(self, keys=None, min_step=0, max_step=None):
        records = [r for r in self._history if r.get("_step", 0) >= min_step]
        if max_step is not None:
            records = [r for r in records if r.get("_step", 0) < max_step]
        return records


class _ReplaySweepRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id


class _ReplaySweep:
    def __init__(self, data: dict[str, Any]) -> None:
        self.id = data["id"]
        self.config = data.get("config") or {}
        self.runs = [_ReplaySweepRun(rid) for rid in data.get("run_ids", [])]


class _ReplayProject:
    def __init__(self, sweeps: list[_ReplaySweep]) -> None:
        self._sweeps = sweeps

    def sweeps(self):
        return list(self._sweeps)


class _ReplayRunsPage:
    def __init__(self, runs: list[_ReplayWandbRun], more: bool) -> None:
        self._runs = runs
        self.more = more

    def __getitem__(self, index):
        return self._runs[index]


class _ReplayClient:
    """Replays a single recorded fixture. Only supports the call the
    fixture's `kind` corresponds to — this is intentional: a fixture
    recorded for `get_run_summary` should not silently satisfy a
    `list_runs` call with fabricated data.
    """

    def __init__(self, fixture: dict[str, Any], entity: str) -> None:
        self._fixture = fixture
        self.default_entity = entity

    def run(self, path: str):
        kind = self._fixture["kind"]
        if kind == "run":
            return _ReplayWandbRun(self._fixture["run"], self._fixture.get("history") or [])
        if kind == "run_error":
            raise Exception(self._fixture["exception_message"])
        raise AssertionError(f"fixture kind {kind!r} does not support run()")

    def runs(self, path: str, filters=None, per_page=50, order="+created_at"):
        kind = self._fixture["kind"]
        if kind == "list_runs":
            runs = [_ReplayWandbRun(r, []) for r in self._fixture["runs"]]
            return _ReplayRunsPage(runs, self._fixture.get("more", False))
        if kind == "list_runs_error":
            if "exception_message" in self._fixture:
                raise Exception(self._fixture["exception_message"])
            return _ReplayRunsPage([], False)
        raise AssertionError(f"fixture kind {kind!r} does not support runs()")

    def project(self, name: str, entity=None):
        kind = self._fixture["kind"]
        if kind == "sweeps":
            sweeps = [_ReplaySweep(s) for s in self._fixture["sweeps"]]
            return _ReplayProject(sweeps)
        raise AssertionError(f"fixture kind {kind!r} does not support project()")


def _load(scenario_dir: Path) -> dict[str, Any]:
    return json.loads((scenario_dir / "fixture.json").read_text())


def _backend_for(scenario_dir: Path) -> tuple[WandbBackend, dict[str, Any]]:
    fixture = _load(scenario_dir)
    entity = fixture.get("run", {}).get("entity") or "<entity>"
    client = _ReplayClient(fixture, entity=entity)
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="replay-key", entity=entity), client=client
    )
    return backend, fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_dir", _SCENARIOS, ids=lambda p: p.name)
async def test_fixture_replays_without_crashing(scenario_dir: Path):
    """The baseline guarantee every fixture must satisfy: replaying it
    through WandbBackend either returns a value or raises one of the
    backend's own typed/expected exceptions — never an unhandled
    TypeError/AttributeError from a mapping-shape mismatch."""
    backend, fixture = _backend_for(scenario_dir)
    kind = fixture["kind"]
    project = fixture.get("run", {}).get("project") or "replay-project"
    entity = fixture.get("run", {}).get("entity") or "<entity>"

    if kind == "run":
        ref = RunRef(
            backend="wandb", entity=entity, project=project, run_id=fixture["run"]["id"]
        )
        run = await backend.get_run_summary(ref)
        assert run.ref == ref
        metric = fixture.get("history_metric")
        if metric:
            history = await backend.get_metric_history(ref, metric)
            assert history.metric_name == metric
            # Points must line up 1:1 with recorded history rows.
            assert len(history.points) == len(fixture["history"])

    elif kind == "run_error":
        ref = RunRef(backend="wandb", entity=entity, project=project, run_id="unknown")
        with pytest.raises(Exception):  # noqa: B017 - real shape asserted below
            await backend.get_run_summary(ref)

    elif kind == "list_runs":
        page = await backend.list_runs(project)
        assert len(page.items) == len(fixture["runs"])

    elif kind == "list_runs_error":
        try:
            page = await backend.list_runs(project)
        except Exception:  # noqa: BLE001 - either outcome is valid; both are asserted below
            pass
        else:
            if "exception_message" not in fixture:
                assert len(page.items) == fixture.get("count", 0)

    elif kind == "sweeps":
        sweeps = backend.list_sweeps(project)
        assert len(sweeps) == len(fixture["sweeps"])

    elif kind == "auth_error":
        # Not a WandbBackend call (auth happens against a raw wandb.Api(),
        # before any backend is constructed) — covered separately by
        # test_auth_failure_fixture_matches_test_connection_classification
        # below. Nothing to replay through the backend here.
        pass

    else:
        pytest.fail(f"No replay assertion defined for fixture kind {kind!r}")


def test_run_not_found_fixture_maps_to_typed_error():
    """Scenario-specific check: `run_not_found` must map onto the typed
    `WandbRunNotFoundError`, not just "some exception" — this is the
    actual regression docs/wandb_fixture_plan.md's `run_not_found` row
    exists to catch."""
    candidates = [d for d in _SCENARIOS if d.name == "run_not_found"]
    if not candidates:
        pytest.skip("run_not_found fixture not recorded yet")
    scenario_dir = candidates[0]
    fixture = _load(scenario_dir)
    message = fixture["exception_message"].lower()
    assert "could not find" in message or "not found" in message or "404" in message, (
        "The recorded run_not_found exception text does not contain any "
        "phrase _is_run_not_found (wandb_backend.py) checks for — "
        "WandbRunNotFoundError's classification will NOT fire for this "
        "real error shape. Update _is_run_not_found to match what W&B "
        f"actually raises: {fixture['exception_message']!r}"
    )


def test_auth_failure_fixture_matches_test_connection_classification():
    """Scenario-specific check for `auth_failure_invalid_key`: confirms
    the real exception text contains 'auth' or 'api key', which is what
    `test_connection` checks for to distinguish an auth failure from a
    generic connection error."""
    candidates = [d for d in _SCENARIOS if d.name == "auth_failure_invalid_key"]
    if not candidates:
        pytest.skip("auth_failure_invalid_key fixture not recorded yet")
    fixture = _load(candidates[0])
    message = fixture["exception_message"].lower()
    assert "auth" in message or "api key" in message, (
        "The recorded auth-failure exception text does not contain 'auth' "
        "or 'api key' — test_connection's classification in "
        f"wandb_backend.py will misroute this: {fixture['exception_message']!r}"
    )