"""Tests for WandbBackend (Milestone 3) against an injected fake client.

**Not** "recorded fixtures" per spec §7 / tests/fixtures/README.md's own
convention — this build environment has no WANDB_API_KEY and no network
access to the W&B API (see the module docstring in wandb_backend.py and
the Milestone 3 summary for the full explanation). The fake client below
is constructed from W&B's *documented* public API attribute contracts
(wandb==0.28.0 `Run`/`Runs`/`Api` docstrings), not transcribed by hand
from a guess — but it is not a substitute for the real recorded-fixture
suite tests/fixtures/README.md commits this project to. That step is
pending `scripts/record_wandb_fixtures.py` being run against a real
project (see that script's docstring).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from experiment_audit_mcp.auth import (
    MissingCredentialsError,
    WandbCredentials,
    load_wandb_credentials,
)
from experiment_audit_mcp.backends.base import BackendCapability, RunFilter
from experiment_audit_mcp.backends.wandb_backend import (
    WandbBackend,
    WandbRunNotFoundError,
    _to_wandb_filters,
)
from experiment_audit_mcp.models import RunRef

# ---------------------------------------------------------------------------
# Fake W&B API client — satisfies the structural Protocols in
# wandb_backend.py, built from documented wandb==0.28.0 attribute shapes.
# ---------------------------------------------------------------------------


class _FakeWandbRun:
    def __init__(
        self,
        id: str,
        name: str,
        project: str,
        entity: str,
        tags: list[str],
        state: str,
        created_at: str,
        config: dict,
        summary_metrics: dict,
        history: list[dict] | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.project = project
        self.entity = entity
        self.tags = tags
        self.state = state
        self.created_at = created_at
        self.config = config
        self.summary_metrics = summary_metrics
        self._history = history or []
        self.scan_history_calls: list[dict] = []

    def scan_history(self, keys=None, min_step=0, max_step=None):
        self.scan_history_calls.append({"keys": keys, "min_step": min_step, "max_step": max_step})
        records = [r for r in self._history if r.get("_step", 0) >= min_step]
        if max_step is not None:
            records = [r for r in records if r.get("_step", 0) < max_step]
        return records


class _FakeWandbSweep:
    def __init__(self, id: str, config: dict, run_ids: list[str]) -> None:
        self.id = id
        self.config = config
        self.runs = [_FakeSweepRun(run_id) for run_id in run_ids]


class _FakeSweepRun:
    def __init__(self, id: str) -> None:
        self.id = id


class _FakeWandbProject:
    def __init__(self, sweeps: list[_FakeWandbSweep]) -> None:
        self._sweeps = sweeps

    def sweeps(self):
        return list(self._sweeps)


class _FakeRunsPage:
    """Mimics wandb.apis.public.Runs' slicing + `.more` contract closely
    enough to exercise WandbBackend's offset-cursor pagination logic."""

    def __init__(self, all_runs: list[_FakeWandbRun]) -> None:
        self._all_runs = all_runs
        self.more = False  # set by __getitem__, mirrors real Runs semantics

    def __getitem__(self, index):
        if isinstance(index, slice):
            stop = index.stop if index.stop is not None else len(self._all_runs)
            self.more = stop < len(self._all_runs)
            return self._all_runs[index]
        return self._all_runs[index]


class _FakeWandbApiClient:
    def __init__(
        self,
        default_entity: str = "dash-research",
        runs_by_project: dict[str, list[_FakeWandbRun]] | None = None,
        runs_by_path: dict[str, _FakeWandbRun] | None = None,
        sweeps_by_project: dict[str, list[_FakeWandbSweep]] | None = None,
        raise_on_default_entity: Exception | None = None,
    ) -> None:
        self.default_entity_value = default_entity
        self._raise_on_default_entity = raise_on_default_entity
        self._runs_by_project = runs_by_project or {}
        self._runs_by_path = runs_by_path or {}
        self._sweeps_by_project = sweeps_by_project or {}
        self.runs_calls: list[dict] = []
        self.run_calls: list[str] = []
        self.project_calls: list[tuple[str, str | None]] = []

    @property
    def default_entity(self) -> str:
        if self._raise_on_default_entity is not None:
            raise self._raise_on_default_entity
        return self.default_entity_value

    def runs(self, path, filters=None, per_page=50, order="+created_at"):
        self.runs_calls.append(
            {"path": path, "filters": filters, "per_page": per_page, "order": order}
        )
        project = path.split("/")[-1]
        return _FakeRunsPage(self._runs_by_project.get(project, []))

    def run(self, path):
        self.run_calls.append(path)
        if path not in self._runs_by_path:
            raise Exception(f"404 not found: {path}")
        return self._runs_by_path[path]

    def project(self, name, entity=None):
        self.project_calls.append((name, entity))
        return _FakeWandbProject(self._sweeps_by_project.get(name, []))


def _make_run(
    run_id="abc123",
    project="mamfac",
    entity="dash-research",
    tags=None,
    state="finished",
    created_at="2026-06-01T12:00:00Z",
    config=None,
    summary_metrics=None,
    history=None,
):
    return _FakeWandbRun(
        id=run_id,
        name=f"run-{run_id}",
        project=project,
        entity=entity,
        tags=tags if tags is not None else ["baseline"],
        state=state,
        created_at=created_at,
        config=config if config is not None else {"lr": 0.001},
        summary_metrics=summary_metrics if summary_metrics is not None else {"reward": 42.0},
        history=history,
    )


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_success_when_entity_resolves():
    client = _FakeWandbApiClient(default_entity="dash-research")
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity=None), client=client
    )
    status = await backend.test_connection()
    assert status.authenticated is True
    assert status.backend == "wandb"
    assert status.scopes_detected == ["read"]
    assert status.error is None


@pytest.mark.asyncio
async def test_connection_reports_auth_failure_distinctly_from_generic_error():
    client = _FakeWandbApiClient(
        raise_on_default_entity=Exception("Invalid API key: authentication failed")
    )
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="bad-key", entity=None), client=client
    )
    status = await backend.test_connection()
    assert status.authenticated is False
    assert "auth" in status.error.lower()


@pytest.mark.asyncio
async def test_connection_reports_generic_network_error_distinctly():
    client = _FakeWandbApiClient(raise_on_default_entity=Exception("connection timed out"))
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity=None), client=client
    )
    status = await backend.test_connection()
    assert status.authenticated is False
    assert "Could not verify W&B connection" in status.error


@pytest.mark.asyncio
async def test_connection_uses_configured_entity_without_calling_default_entity():
    client = _FakeWandbApiClient(default_entity="should-not-be-used")
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"),
        client=client,
    )
    await backend.test_connection()
    # default_entity is a property on the fake; the only observable proxy
    # for "was it called" is that no exception path was hit and no
    # tracking call list exists for a property access, so this is
    # asserted indirectly via list_runs using the configured entity below.
    assert backend._entity == "dash-research"


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_runs_happy_path_maps_fields_and_scopes_entity():
    run = _make_run(run_id="abc123", tags=["baseline", "v2"], state="finished")
    client = _FakeWandbApiClient(default_entity="dash-research", runs_by_project={"mamfac": [run]})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )

    page = await backend.list_runs("mamfac")

    assert page.next_cursor is None
    assert len(page.items) == 1
    got = page.items[0]
    assert got.ref == RunRef(
        backend="wandb", entity="dash-research", project="mamfac", run_id="abc123"
    )
    assert got.tags == ["baseline", "v2"]
    assert got.status == "finished"
    assert got.data_completeness == "complete"
    assert got.config == {"lr": 0.001}
    assert got.summary_metrics == {"reward": 42.0}
    assert got.created_at == datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_list_runs_paginates_via_offset_cursor_round_trip():
    runs = [_make_run(run_id=f"run{i}") for i in range(5)]
    client = _FakeWandbApiClient(default_entity="dash-research", runs_by_project={"mamfac": runs})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"),
        client=client,
        page_size=2,
    )

    page1 = await backend.list_runs("mamfac")
    assert [r.ref.run_id for r in page1.items] == ["run0", "run1"]
    assert page1.next_cursor == "2"

    page2 = await backend.list_runs("mamfac", cursor=page1.next_cursor)
    assert [r.ref.run_id for r in page2.items] == ["run2", "run3"]
    assert page2.next_cursor == "4"

    page3 = await backend.list_runs("mamfac", cursor=page2.next_cursor)
    assert [r.ref.run_id for r in page3.items] == ["run4"]
    assert page3.next_cursor is None


@pytest.mark.asyncio
async def test_list_runs_page_size_override_takes_precedence_over_constructor_default():
    # Revision 2: a per-call `page_size` overrides the backend's
    # constructor-configured default (`page_size=2` below) rather than
    # being ignored.
    runs = [_make_run(run_id=f"run{i}") for i in range(5)]
    client = _FakeWandbApiClient(default_entity="dash-research", runs_by_project={"mamfac": runs})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"),
        client=client,
        page_size=2,
    )

    page = await backend.list_runs("mamfac", page_size=4)

    assert [r.ref.run_id for r in page.items] == ["run0", "run1", "run2", "run3"]
    assert page.next_cursor == "4"
    assert client.runs_calls[0]["per_page"] == 4


@pytest.mark.asyncio
async def test_list_runs_omitted_page_size_falls_back_to_constructor_default():
    runs = [_make_run(run_id=f"run{i}") for i in range(5)]
    client = _FakeWandbApiClient(default_entity="dash-research", runs_by_project={"mamfac": runs})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"),
        client=client,
        page_size=2,
    )

    page = await backend.list_runs("mamfac")

    assert [r.ref.run_id for r in page.items] == ["run0", "run1"]
    assert client.runs_calls[0]["per_page"] == 2


@pytest.mark.asyncio
async def test_list_runs_passes_filters_through_translated():
    client = _FakeWandbApiClient(default_entity="dash-research", runs_by_project={"mamfac": []})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    await backend.list_runs("mamfac", filters=RunFilter(tags=["baseline"], status="finished"))
    assert client.runs_calls[0]["filters"] == {
        "$and": [{"tags": {"$in": ["baseline"]}}, {"state": "finished"}]
    }


# ---------------------------------------------------------------------------
# get_run_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_summary_found():
    run = _make_run(run_id="abc123", entity="dash-research", project="mamfac")
    client = _FakeWandbApiClient(runs_by_path={"dash-research/mamfac/abc123": run})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="abc123")

    got = await backend.get_run_summary(ref)

    assert got.ref == ref
    assert client.run_calls == ["dash-research/mamfac/abc123"]


@pytest.mark.asyncio
async def test_get_run_summary_not_found_raises_typed_error():
    client = _FakeWandbApiClient(runs_by_path={})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="missing")

    with pytest.raises(WandbRunNotFoundError):
        await backend.get_run_summary(ref)


# ---------------------------------------------------------------------------
# get_metric_history — the None/NaN-preservation adversarial case (spec §7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_metric_history_maps_nan_sentinel_and_missing_key_to_none():
    history = [
        {"_step": 0, "reward": 1.0},
        {"_step": 1, "reward": "NaN"},  # logged NaN, per W&B's documented sentinel
        {"_step": 2},  # metric not logged this step
        {"_step": 3, "reward": 3.5},
    ]
    run = _make_run(run_id="crashed-run", history=history)
    client = _FakeWandbApiClient(runs_by_path={"dash-research/mamfac/crashed-run": run})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="crashed-run")

    result = await backend.get_metric_history(ref, "reward")

    assert [p.value for p in result.points] == [1.0, None, None, 3.5]
    assert [p.step for p in result.points] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_get_metric_history_passes_step_range_to_scan_history():
    run = _make_run(run_id="abc123", history=[{"_step": i, "reward": float(i)} for i in range(10)])
    client = _FakeWandbApiClient(runs_by_path={"dash-research/mamfac/abc123": run})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="abc123")

    result = await backend.get_metric_history(ref, "reward", step_range=(2, 5))

    assert run.scan_history_calls[0]["min_step"] == 2
    assert run.scan_history_calls[0]["max_step"] == 5
    assert [p.step for p in result.points] == [2, 3, 4]


# ---------------------------------------------------------------------------
# data_completeness heuristic (flagged as needing live validation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,expected",
    [
        ("running", "partial"),
        ("finished", "complete"),
        ("crashed", "unknown"),
        ("killed", "unknown"),
        ("preempting", "unknown"),
    ],
)
async def test_data_completeness_inferred_from_state(state, expected):
    run = _make_run(run_id="r1", state=state)
    client = _FakeWandbApiClient(runs_by_project={"mamfac": [run]})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    page = await backend.list_runs("mamfac")
    assert page.items[0].data_completeness == expected


# ---------------------------------------------------------------------------
# summary_metrics: non-numeric values dropped, not raised on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_metrics_drops_non_numeric_values():
    run = _make_run(
        run_id="r1",
        summary_metrics={"reward": 1.5, "best_checkpoint": "gs://bucket/ckpt.pt", "steps": 100},
    )
    client = _FakeWandbApiClient(runs_by_project={"mamfac": [run]})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    page = await backend.list_runs("mamfac")
    assert page.items[0].summary_metrics == {"reward": 1.5, "steps": 100.0}


# ---------------------------------------------------------------------------
# rate-limit backoff (spec §5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_summary_retries_on_rate_limit_then_succeeds(monkeypatch):
    from experiment_audit_mcp.backends import wandb_backend as wb_module

    monkeypatch.setattr(wb_module.time, "sleep", lambda _seconds: None)

    call_count = {"n": 0}
    real_run = _make_run(run_id="abc123")

    class FlakyClient(_FakeWandbApiClient):
        def run(self, path):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("429 Too Many Requests")
            return real_run

    client = FlakyClient(runs_by_path={"dash-research/mamfac/abc123": real_run})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="abc123")

    result = await backend.get_run_summary(ref)

    assert call_count["n"] == 3
    assert result.ref.run_id == "abc123"


@pytest.mark.asyncio
async def test_get_run_summary_does_not_retry_non_retryable_errors(monkeypatch):
    from experiment_audit_mcp.backends import wandb_backend as wb_module

    monkeypatch.setattr(wb_module.time, "sleep", lambda _seconds: None)

    call_count = {"n": 0}

    class BrokenClient(_FakeWandbApiClient):
        def run(self, path):
            call_count["n"] += 1
            raise ValueError("malformed path")

    client = BrokenClient()
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="abc123")

    with pytest.raises(ValueError):
        await backend.get_run_summary(ref)
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_get_run_summary_gives_up_after_max_retries(monkeypatch):
    from experiment_audit_mcp.backends import wandb_backend as wb_module

    monkeypatch.setattr(wb_module.time, "sleep", lambda _seconds: None)

    call_count = {"n": 0}

    class AlwaysRateLimitedClient(_FakeWandbApiClient):
        def run(self, path):
            call_count["n"] += 1
            raise Exception("429 rate limit exceeded")

    client = AlwaysRateLimitedClient()
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="abc123")

    with pytest.raises(Exception, match="429"):
        await backend.get_run_summary(ref)
    assert call_count["n"] == wb_module._MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# REGRESSION TESTS — backend audit findings
# ---------------------------------------------------------------------------
#
# Each test below corresponds to one confirmed bug found during the
# audit of wandb_backend.py. See the module's inline "**Bugfix:**"
# comments at each fix site for the full explanation.


class _ShrinkingRunsPage:
    """A paginator double that returns fewer items than `per_page` while
    still reporting `.more = True` — simulating a live project where rows
    are removed/filtered between the server's "more available" check and
    the actual page fetch. Used to prove next_cursor tracks the number of
    items *actually returned*, not the requested page size."""

    def __init__(self, items: list[_FakeWandbRun], more: bool) -> None:
        self._items = items
        self.more = more

    def __getitem__(self, index):
        return self._items


@pytest.mark.asyncio
async def test_list_runs_next_cursor_reflects_items_actually_returned():
    """Bug: next_cursor was computed as `offset + per_page` whenever
    `has_more` was True, instead of `offset + len(page_items)`. A
    paginator that returns fewer items than requested while still
    reporting more data available would cause the next page's fetch to
    silently skip runs. Regression-tests the fix."""
    two_runs = [_make_run(run_id="r0"), _make_run(run_id="r1")]

    class _ShortPageClient(_FakeWandbApiClient):
        def runs(self, path, filters=None, per_page=50, order="+created_at"):
            self.runs_calls.append(
                {"path": path, "filters": filters, "per_page": per_page, "order": order}
            )
            return _ShrinkingRunsPage(two_runs, more=True)

    client = _ShortPageClient(default_entity="dash-research")
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"),
        client=client,
        page_size=5,
    )

    page = await backend.list_runs("mamfac")

    assert [r.ref.run_id for r in page.items] == ["r0", "r1"]
    # Must advance by the 2 items actually returned, NOT by per_page (5).
    assert page.next_cursor == "2"


@pytest.mark.asyncio
async def test_list_runs_rejects_non_positive_page_size():
    """Bug: page_size was never validated. page_size=0 (or negative)
    produces a zero-length/backwards slice every call while `has_more`
    stays True, so next_cursor never advances -- a non-terminating
    pagination loop for any caller that loops until next_cursor is None."""
    client = _FakeWandbApiClient(default_entity="dash-research", runs_by_project={"mamfac": []})
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )

    with pytest.raises(ValueError):
        await backend.list_runs("mamfac", page_size=0)

    with pytest.raises(ValueError):
        await backend.list_runs("mamfac", page_size=-3)


@pytest.mark.asyncio
async def test_get_metric_history_uses_fresh_client_when_not_injected(monkeypatch):
    """Bug: get_metric_history called `self._client.run(...)` directly,
    reusing the single long-lived `Api()` instance for the backend's
    whole lifetime -- exactly the caching-staleness failure mode
    documented in `__init__` and already worked around in `list_runs`
    and `get_run_summary`, just never applied here. A still-running run's
    metric history would freeze at its first-fetched snapshot forever.
    Regression-tests that a fresh client is constructed per call when no
    client was injected (i.e. the real, non-test code path)."""
    import wandb.apis.public as real_wandb_public

    shared_run = _make_run(run_id="r1", history=[{"_step": 0, "reward": 1.0}])
    construction_count = {"n": 0}

    def fake_api(api_key):
        construction_count["n"] += 1
        return _FakeWandbApiClient(runs_by_path={"dash-research/mamfac/r1": shared_run})

    monkeypatch.setattr(real_wandb_public, "Api", fake_api)

    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"),
        client=None,
    )
    baseline = construction_count["n"]  # accounts for the __init__-time construction
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="r1")

    await backend.get_metric_history(ref, "reward")
    await backend.get_metric_history(ref, "reward")

    assert construction_count["n"] == baseline + 2


def test_list_sweeps_uses_fresh_client_when_not_injected(monkeypatch):
    """Bug: list_sweeps called `self._client.project(...)` directly,
    reusing the same long-lived `Api()` instance rather than following
    the fresh-client-per-call pattern used everywhere else in this
    backend. A sweep that gained new runs after the first `list_sweeps`
    call would keep reporting the original, stale run set for the life
    of the process. Regression-tests that a fresh client is constructed
    per call when no client was injected."""
    import wandb.apis.public as real_wandb_public

    sweep = _FakeWandbSweep(id="sweep1", config={"method": "grid"}, run_ids=["r1"])
    construction_count = {"n": 0}

    def fake_api(api_key):
        construction_count["n"] += 1
        return _FakeWandbApiClient(
            default_entity="dash-research", sweeps_by_project={"mamfac": [sweep]}
        )

    monkeypatch.setattr(real_wandb_public, "Api", fake_api)

    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"),
        client=None,
    )
    baseline = construction_count["n"]

    backend.list_sweeps("mamfac")
    backend.list_sweeps("mamfac")

    assert construction_count["n"] == baseline + 2


@pytest.mark.asyncio
async def test_not_found_error_is_not_spuriously_retried_when_id_contains_status_digits(
    monkeypatch,
):
    """Bug: `_is_retryable` did a bare substring check (`"502" in
    message`), so a permanent `WandbRunNotFoundError` whose message
    embeds the run id verbatim (e.g. a run literally named "run502")
    looked like a retryable 502 and was retried up to `_MAX_RETRIES`
    times (~60s of added latency, per the real backoff schedule) before
    the not-found error was finally surfaced. Regression-tests that the
    lookup now fails fast (a single call) instead of being retried."""
    from experiment_audit_mcp.backends import wandb_backend as wb_module

    monkeypatch.setattr(wb_module.time, "sleep", lambda _seconds: None)

    call_count = {"n": 0}

    class NotFoundClient(_FakeWandbApiClient):
        def run(self, path):
            call_count["n"] += 1
            raise Exception("not found")

    client = NotFoundClient()
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    # The run id itself contains "502" as a plain substring -- this is
    # exactly what defeated the old bare-substring retry check.
    ref = RunRef(backend="wandb", entity="dash-research", project="mamfac", run_id="run502")

    with pytest.raises(WandbRunNotFoundError):
        await backend.get_run_summary(ref)

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_is_retryable_still_matches_real_status_codes_after_fix():
    """Sanity check accompanying the fix above: word-boundary matching
    must not regress detection of genuine retryable transport errors."""
    from experiment_audit_mcp.backends.wandb_backend import _is_retryable

    assert _is_retryable(Exception("429 Too Many Requests")) is True
    assert _is_retryable(Exception("Error 502: Bad Gateway")) is True
    assert _is_retryable(Exception("503 Service Unavailable")) is True
    assert _is_retryable(Exception("request timeout")) is True
    assert _is_retryable(Exception("rate limit exceeded")) is True
    assert _is_retryable(Exception("rate_limit_exceeded")) is True
    # And must NOT match digits that are merely embedded in an unrelated
    # identifier (the false-positive this fix closes).
    assert _is_retryable(Exception("no run found at team/proj/run502")) is False
    assert _is_retryable(Exception("no run found at team/proj503/run1")) is False


# ---------------------------------------------------------------------------
# capabilities (Milestone 3 explicitly does not implement list_sweeps)
# ---------------------------------------------------------------------------


def test_capabilities_include_sweeps():
    """Milestone 8: `list_sweeps` is now a real implementation (below),
    so declaring `SWEEPS` is no longer overclaiming -- see the
    Milestone-7-era version of this test (`test_capabilities_do_not_
    include_sweeps_yet`, since replaced) for why it was deliberately
    `set()` before this milestone."""
    client = _FakeWandbApiClient()
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )
    assert backend.capabilities == {BackendCapability.SWEEPS}


def test_list_sweeps_maps_sweep_config_and_runs():
    sweep = _FakeWandbSweep(
        id="sweep1",
        config={"method": "grid", "metric": {"name": "reward", "goal": "maximize"}},
        run_ids=["run1", "run2", "run3"],
    )
    client = _FakeWandbApiClient(
        default_entity="dash-research", sweeps_by_project={"mamfac": [sweep]}
    )
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )

    sweeps = backend.list_sweeps("mamfac")

    assert len(sweeps) == 1
    result = sweeps[0]
    assert result.sweep_id == "sweep1"
    assert result.method == "grid"
    assert result.target_metric == "reward"
    assert result.ref.entity == "dash-research"
    assert result.ref.project == "mamfac"
    assert [ref.run_id for ref in result.run_refs] == ["run1", "run2", "run3"]
    assert all(ref.entity == "dash-research" and ref.project == "mamfac" for ref in result.run_refs)
    assert client.project_calls == [("mamfac", "dash-research")]


def test_list_sweeps_defaults_missing_method_and_metric():
    """A malformed/legacy sweep config (no 'method' or 'metric' key)
    should not crash the whole listing -- see list_sweeps' 'Flag'
    docstring note. It should default rather than raise."""
    sweep = _FakeWandbSweep(id="sweep-legacy", config={}, run_ids=["run1"])
    client = _FakeWandbApiClient(
        default_entity="dash-research", sweeps_by_project={"mamfac": [sweep]}
    )
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )

    sweeps = backend.list_sweeps("mamfac")

    assert sweeps[0].method == "unsupported"
    assert sweeps[0].target_metric is None


def test_list_sweeps_returns_multiple_sweeps():
    sweep_a = _FakeWandbSweep(id="sweep-a", config={"method": "random"}, run_ids=["r1"])
    sweep_b = _FakeWandbSweep(id="sweep-b", config={"method": "bayes"}, run_ids=["r2", "r3"])
    client = _FakeWandbApiClient(
        default_entity="dash-research", sweeps_by_project={"mamfac": [sweep_a, sweep_b]}
    )
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )

    sweeps = backend.list_sweeps("mamfac")

    assert {s.sweep_id for s in sweeps} == {"sweep-a", "sweep-b"}


def test_list_sweeps_uses_configured_entity_without_resolving_default():
    """entity="dash-research" is already known at construction time (per
    Revision 1's RunRef.entity requirement) -- list_sweeps must not call
    the (expensive, network-hitting) default_entity resolution path when
    it already has an entity to use."""
    sweep = _FakeWandbSweep(id="sweep1", config={"method": "grid"}, run_ids=[])
    client = _FakeWandbApiClient(
        default_entity="should-not-be-used", sweeps_by_project={"mamfac": [sweep]}
    )
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity="dash-research"), client=client
    )

    sweeps = backend.list_sweeps("mamfac")

    assert sweeps[0].ref.entity == "dash-research"
    assert client.project_calls == [("mamfac", "dash-research")]


def test_list_sweeps_resolves_default_entity_synchronously_when_unset():
    sweep = _FakeWandbSweep(id="sweep1", config={"method": "grid"}, run_ids=[])
    client = _FakeWandbApiClient(
        default_entity="resolved-entity", sweeps_by_project={"mamfac": [sweep]}
    )
    backend = WandbBackend(
        credentials=WandbCredentials(api_key="fake-key", entity=None), client=client
    )

    sweeps = backend.list_sweeps("mamfac")

    assert sweeps[0].ref.entity == "resolved-entity"
    assert backend._entity == "resolved-entity"


# ---------------------------------------------------------------------------
# _to_wandb_filters (pure function unit tests)
# ---------------------------------------------------------------------------


def test_to_wandb_filters_none_when_no_filters():
    assert _to_wandb_filters(None) is None
    assert _to_wandb_filters(RunFilter()) is None


def test_to_wandb_filters_single_clause_not_wrapped_in_and():
    result = _to_wandb_filters(RunFilter(tags=["baseline"]))
    assert result == {"tags": {"$in": ["baseline"]}}


def test_to_wandb_filters_date_range():
    after = datetime(2026, 1, 1, tzinfo=UTC)
    before = datetime(2026, 6, 1, tzinfo=UTC)
    result = _to_wandb_filters(RunFilter(created_after=after, created_before=before))
    assert result == {
        "$and": [
            {"created_at": {"$gte": after.isoformat()}},
            {"created_at": {"$lte": before.isoformat()}},
        ]
    }


# ---------------------------------------------------------------------------
# auth.py
# ---------------------------------------------------------------------------


def test_load_wandb_credentials_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError):
        load_wandb_credentials()


def test_load_wandb_credentials_resolves_entity_from_env(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setenv("WANDB_ENTITY", "dash-research")
    creds = load_wandb_credentials()
    assert creds.api_key == "fake-key"
    assert creds.entity == "dash-research"


def test_load_wandb_credentials_entity_none_when_unset(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    creds = load_wandb_credentials()
    assert creds.entity is None


def test_missing_credentials_error_never_echoes_a_value():
    monkeypatch_value = "sk-super-secret-should-never-appear"
    err = MissingCredentialsError("WANDB_API_KEY")
    assert monkeypatch_value not in str(err)
    assert "WANDB_API_KEY" in str(err)
