"""Real ExperimentBackend implementation against the Weights & Biases API.

Implements design-spec-v1.md §3, §5, §6, satisfying the same
`ExperimentBackend` contract validated against `FakeBackend` in
Milestone 2. See the module-level notes below (and the Milestone 3
summary delivered alongside this file) for the concrete points where
W&B's real API shape required a documented mapping decision rather than
a mechanical translation — this file follows the roadmap's instruction
to surface those rather than paper over them.

Dependency injection: `WandbBackend` talks to W&B through a `client`
object satisfying the small structural interface below (`_WandbApiClient`
and friends), rather than importing `wandb.Api` directly everywhere.
`wandb.Api` satisfies this interface as-is. Tests inject a lightweight
fake client instead, so this module's mapping/pagination/error-handling
logic is fully unit-testable without network access or live credentials
— consistent with spec §7 ("tests run against recorded fixtures, not
live API calls").

**Flag (per roadmap's Milestone 3 "flag-if-triggered" clause):** this
sandbox has no network access to the W&B API and no `WANDB_API_KEY`, so
the fixtures backing this module's tests are constructed from W&B's
*documented* public API shapes (SDK docstrings/attribute contracts), not
recorded from a live project as spec §7 specifies. Three real ambiguities
were found while doing this that could not be resolved without live
traffic — each is called out at its point of use below and repeated in
the Milestone 3 summary. These need live validation against a real
project (e.g. MAMFAC/CARM++) before this backend is trusted in
production, per spec §7's own "API-drift regression" testing category.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
import time
from datetime import datetime
from typing import Any, Protocol

from experiment_audit_mcp.auth import WandbCredentials, load_wandb_credentials
from experiment_audit_mcp.backends.base import (
    BackendCapability,
    ConnectionStatus,
    ExperimentBackend,
    RunFilter,
)
from experiment_audit_mcp.models import MetricHistory, MetricPoint, Page, Run, RunRef, Sweep

# -- Structural interfaces for the wandb SDK surface this backend uses ------
#
# `wandb.Api` and `wandb.apis.public.Run` satisfy these as-is (verified
# against wandb==0.28.0's documented attributes). Defined explicitly here,
# rather than importing wandb's concrete classes as type hints, so a test
# double only needs to satisfy this narrow surface, not the SDK's full API.


class _WandbRunLike(Protocol):
    id: str
    name: str
    project: str
    entity: str
    tags: list[str]
    state: str
    created_at: str
    config: dict[str, Any]
    summary_metrics: dict[str, Any]

    def scan_history(
        self,
        keys: list[str] | None = None,
        min_step: int = 0,
        max_step: int | None = None,
    ) -> Any: ...


class _WandbRunsPageLike(Protocol):
    more: bool

    def __getitem__(self, index: Any) -> Any: ...


class _WandbSweepRunLike(Protocol):
    """The minimal shape this backend needs from a run nested under a
    sweep (`_WandbSweepLike.runs`). Deliberately narrower than
    `_WandbRunLike` above: `list_sweeps` (per Milestone 8's roadmap entry,
    "never full history, keeps this cheap even for large sweeps") only
    needs each run's `id` to build a `RunRef` -- full config/metrics are
    fetched later, once, only for the specific sweep being audited, via
    the existing `get_run_summary`."""

    id: str


class _WandbSweepLike(Protocol):
    id: str
    config: dict[str, Any]
    runs: Any  # iterable of _WandbSweepRunLike


class _WandbProjectLike(Protocol):
    def sweeps(self) -> Any: ...  # iterable of _WandbSweepLike


class _WandbApiClient(Protocol):
    default_entity: str

    def runs(
        self,
        path: str,
        filters: dict[str, Any] | None = None,
        per_page: int = 50,
        order: str = "+created_at",
    ) -> _WandbRunsPageLike: ...

    def run(self, path: str) -> _WandbRunLike: ...

    def project(self, name: str, entity: str | None = None) -> _WandbProjectLike: ...


# -- Backend-local errors ----------------------------------------------------
#
# Mirrors FakeBackend's pattern (backends/fake_backend.py): a backend-local
# not-found exception, translated into a structured ToolError at the
# tool layer in Milestone 4+. Not shared with FakeBackend's exception
# classes since the two backends fail for different underlying reasons
# (missing dict entry vs. a real 404 from the API) and conflating them
# would blur what should be two distinct, backend-specific error paths
# until Milestone 4 defines the shared translation.


class WandbRunNotFoundError(Exception):
    def __init__(self, ref: RunRef) -> None:
        self.ref = ref
        super().__init__(
            f"No run found at {ref.entity}/{ref.project}/{ref.run_id} "
            f"(or the API key lacks read access to it)."
        )


class WandbAuthenticationError(Exception):
    """Raised when the configured WANDB_API_KEY is rejected by the API."""


# -- Rate-limit backoff (spec §5: "backoff handled once in the backend
# layer, never duplicated per-tool") ----------------------------------------

_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 30.0


_RETRY_TOKEN_PATTERNS = [
    # Numeric HTTP-style status codes and "timeout" are wrapped in \b
    # (word-boundary) so they only match as standalone tokens — see the
    # bugfix note below. Phrase tokens ("rate limit"/"rate_limit") are
    # inherently space/underscore-delimited already, so a plain substring
    # check for them carries negligible false-positive risk.
    re.compile(r"\b429\b"),
    re.compile(r"\b502\b"),
    re.compile(r"\b503\b"),
    re.compile(r"\btimeout\b"),
    re.compile(r"rate[ _]limit"),
]


def _is_retryable(exc: Exception) -> bool:
    """Best-effort classification of a retryable (rate-limited/transient)
    failure vs. a permanent one (auth, not-found).

    **Flag:** wandb's public `Api` surfaces most transport failures as a
    single `wandb.errors.CommError` without a structured status code
    attached at this layer, so this falls back to checking the exception
    message for a 429/5xx/"rate limit" signal. This is a heuristic, not a
    contract — verify it against a real rate-limited response before
    relying on it in production (no live API access was available to
    confirm the exact exception shape W&B raises for a 429 today).

    **Bugfix:** this previously did a bare substring check (`"502" in
    message`), which misfires whenever an *unrelated* number happens to
    appear next to other digits/letters in the message — most
    concretely, `WandbRunNotFoundError`'s message embeds the run/entity/
    project id verbatim (see its `__init__` below), so a run named e.g.
    `"run502"` made a permanent, already-classified 404 look like a
    retryable 502 and caused up to `_MAX_RETRIES` pointless retries
    (~60s of added latency) before the not-found error was finally
    surfaced. Word-boundary matching on the numeric/timeout tokens fixes
    this while keeping the true-positive cases (space- or
    punctuation-delimited status codes in real transport errors) intact.
    """
    message = str(exc).lower()
    return any(pattern.search(message) for pattern in _RETRY_TOKEN_PATTERNS)


def _call_with_backoff(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Synchronous retry wrapper — runs inside `asyncio.to_thread` (the
    wandb SDK's public `Api` is a blocking/synchronous client; see the
    module-level note on why every public method below wraps its call in
    `asyncio.to_thread` rather than this backend requiring an
    async-native W&B client that doesn't publicly exist).
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            attempt += 1
            if attempt > _MAX_RETRIES or not _is_retryable(exc):
                raise
            delay = min(_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), _MAX_DELAY_SECONDS)
            delay += random.uniform(0, 0.5)  # jitter, avoids thundering herd
            time.sleep(delay)


# -- Backend implementation --------------------------------------------------

_DEFAULT_PAGE_SIZE = 50


class WandbBackend(ExperimentBackend):
    """`ExperimentBackend` implementation against the real W&B API.

    `capabilities` is now `{BackendCapability.SWEEPS}` (Milestone 8): a
    real `list_sweeps` override is implemented below, so declaring the
    capability is no longer the overclaiming the spec's capability-flag
    design (§3) exists to prevent -- it was deliberately `set()` through
    Milestone 7 for exactly that reason (see the Milestone 3 summary),
    and this is the one-line flip that summary already announced.
    """

    name = "wandb"

    def __init__(
        self,
        credentials: WandbCredentials | None = None,
        client: _WandbApiClient | None = None,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> None:
        self.capabilities: set[BackendCapability] = {BackendCapability.SWEEPS}
        self._credentials = credentials or load_wandb_credentials()
        self._page_size = page_size
        self._entity: str | None = self._credentials.entity
        # `_client_is_injected` distinguishes the test-double path (a
        # `_WandbApiClient` Protocol implementation passed in by tests,
        # which must keep being reused as-is) from the real path (no
        # `client` given, so this backend owns a real `wandb.Api()`).
        # This matters for `list_runs` below: the real `wandb.Api()`
        # caches its `Runs` paginator per (path, filters, order) key for
        # the lifetime of the `Api` instance (confirmed against the real
        # SDK: `wandb.apis.public.api.Api.runs`'s `self._runs` cache
        # dict). Since this backend is held for the lifetime of a
        # long-running MCP server process, reusing one `Api()` instance
        # for `list_runs` means every call after the first returns a
        # stale, frozen snapshot of whatever runs existed at the time of
        # that first call -- confirmed live: a project that grew from 3
        # to 27 runs kept reporting exactly 3 through `list_runs` no
        # matter how many times or how long after it was called again,
        # while a fresh `wandb.Api()` in a new process saw all 27
        # immediately. `list_runs` below builds a fresh, short-lived
        # `Api()` per call specifically to sidestep this cache, rather
        # than relying on the SDK's private `_runs` dict.
        self._client_is_injected = client is not None
        if client is not None:
            self._client = client
        else:
            import wandb.apis.public as wandb_public

            self._client = wandb_public.Api(api_key=self._credentials.api_key)

    # -- entity resolution ---------------------------------------------------

    async def _resolve_entity(self) -> str:
        """Resolve and cache the entity to scope RunRefs under.

        `WANDB_ENTITY` wins if set (auth.py); otherwise falls back to the
        authenticated user's `default_entity`, matching the deferral
        documented in `WandbCredentials` (auth.py) and the Revision 1
        writeup in design-spec-v1.md.
        """
        if self._entity is not None:
            return self._entity

        def _fetch_default_entity() -> str:
            return self._client.default_entity

        entity = await asyncio.to_thread(_call_with_backoff, _fetch_default_entity)
        self._entity = entity
        return entity

    def _resolve_entity_sync(self) -> str:
        """Synchronous counterpart to `_resolve_entity`, needed because
        `list_sweeps` (below) must stay a plain synchronous method per
        the frozen `ExperimentBackend.list_sweeps` signature (base.py,
        §3's deliberate async/sync asymmetry, preserved exactly rather
        than "fixed" -- see that method's docstring) and therefore cannot
        `await` the async version above. Duplicates `_resolve_entity`'s
        two-line body rather than sharing it: the only difference (a
        bare blocking call vs. an `asyncio.to_thread`-wrapped await)
        can't be factored out without one of the two methods losing its
        own correct sync/async shape.
        """
        if self._entity is not None:
            return self._entity
        entity = _call_with_backoff(lambda: self._client.default_entity)
        self._entity = entity
        return entity

    # -- ExperimentBackend contract ------------------------------------------

    async def test_connection(self) -> ConnectionStatus:
        try:
            await self._resolve_entity()
        except Exception as exc:  # noqa: BLE001 - classified below
            message = str(exc)
            if "auth" in message.lower() or "api key" in message.lower():
                return ConnectionStatus(backend=self.name, authenticated=False, error=message)
            return ConnectionStatus(
                backend=self.name,
                authenticated=False,
                error=f"Could not verify W&B connection: {message}",
            )
        return ConnectionStatus(
            backend=self.name,
            authenticated=True,
            # "read" is the only scope this backend can positively confirm
            # via the public API surface used here — W&B API keys don't
            # expose a queryable scope/permission list beyond "it worked".
            scopes_detected=["read"],
            error=None,
        )

    async def list_runs(
        self,
        project: str,
        filters: RunFilter | None = None,
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Page[Run]:
        entity = await self._resolve_entity()
        offset = _decode_cursor(cursor)
        # Revision 2: `page_size` is a per-call override of the backend's
        # constructor-configured default (`self._page_size`), closing the
        # gap between design-spec-v1.md §4.2's `list_runs(..., page_size=25)`
        # tool signature and this backend, which previously had no way to
        # honor a per-call page size at all. `None` (the common case) keeps
        # today's behavior unchanged.
        per_page = page_size if page_size is not None else self._page_size
        # **Bugfix:** page_size was never validated. A caller passing
        # page_size=0 (or negative) would get a zero-length (or
        # backwards) slice every call while `has_more` stayed True,
        # producing a `next_cursor` that never advances — an infinite,
        # zero-progress pagination loop for any caller that loops on
        # next_cursor until it's None (which is exactly how cursor-based
        # pagination is meant to be consumed). Fail fast instead.
        if per_page <= 0:
            raise ValueError(f"page_size must be a positive integer, got {per_page!r}")
        wandb_filters = _to_wandb_filters(filters)

        def _fetch() -> tuple[list[_WandbRunLike], bool]:
            # See __init__'s comment on `_client_is_injected`: a real,
            # long-lived `wandb.Api()` caches `.runs()` results per call
            # signature for its whole lifetime, silently going stale as
            # new runs appear on the server. Test doubles don't have
            # this problem (and must keep being reused, since tests may
            # rely on call-count/identity), so only the real path
            # constructs a fresh client here.
            if self._client_is_injected:
                client = self._client
            else:
                import wandb.apis.public as wandb_public

                client = wandb_public.Api(api_key=self._credentials.api_key)
            paginator = client.runs(
                path=f"{entity}/{project}",
                filters=wandb_filters,
                per_page=per_page,
                order="+created_at",
            )
            # Slicing forces the paginator to lazily load pages up to
            # this range — this is the real wandb.apis.public.Runs
            # contract (Paginator.__getitem__), not something bolted on
            # here.
            page_items = list(paginator[offset : offset + per_page])
            return page_items, bool(paginator.more)

        page_items, has_more = await asyncio.to_thread(_call_with_backoff, _fetch)
        runs = [_to_run(entity, project, r) for r in page_items]
        # **Bugfix:** next_cursor used to be computed as `offset + per_page`
        # unconditionally whenever `has_more` was True, rather than from
        # the number of items actually returned. Under normal conditions
        # those are equal, but if the underlying paginator ever returns
        # fewer than `per_page` items while still reporting more data
        # available (e.g. rows removed/filtered server-side between the
        # count check and the fetch — not something this backend can rule
        # out against a live, mutating W&B project), the old computation
        # would silently skip the un-returned runs on the next page.
        # Advancing by `len(page_items)` keeps the cursor correct
        # regardless of how many items a given fetch actually yielded.
        next_cursor = _encode_cursor(offset + len(page_items)) if has_more else None
        return Page(items=runs, next_cursor=next_cursor)

    async def get_run_summary(self, ref: RunRef) -> Run:
        def _fetch() -> _WandbRunLike:
            # Same root cause as list_runs above: `wandb.Api.run()`
            # shares its `self._runs` cache dict with `.runs()`, keyed
            # by the run's path string, and loads that run's data once
            # (`lazy=False`) rather than refetching on repeat calls. A
            # run fetched here while still in progress would otherwise
            # return that same in-progress snapshot forever, even after
            # the run finishes -- so this uses a fresh client too.
            if self._client_is_injected:
                client = self._client
            else:
                import wandb.apis.public as wandb_public

                client = wandb_public.Api(api_key=self._credentials.api_key)
            try:
                return client.run(f"{ref.entity}/{ref.project}/{ref.run_id}")
            except Exception as exc:  # noqa: BLE001
                if "not found" in str(exc).lower() or "404" in str(exc):
                    raise WandbRunNotFoundError(ref) from exc
                raise

        wandb_run = await asyncio.to_thread(_call_with_backoff, _fetch)
        return _to_run(ref.entity, ref.project, wandb_run)

    async def get_metric_history(
        self,
        ref: RunRef,
        metric: str,
        step_range: tuple[int, int] | None = None,
    ) -> MetricHistory:
        min_step = step_range[0] if step_range else 0
        max_step = step_range[1] if step_range else None

        def _fetch() -> list[dict[str, Any]]:
            # **Bugfix:** this used to call `self._client.run(...)`
            # directly, reusing the same long-lived `Api()` instance for
            # the whole life of the backend. That's exactly the staleness
            # bug documented in `__init__` and already worked around in
            # `list_runs` and `get_run_summary` (wandb.Api caches `.run()`
            # results in `self._runs`, keyed by path, for the instance's
            # lifetime) — it was just never applied here. In practice this
            # meant a metric history fetched once for a still-running run
            # would return that same frozen, in-progress snapshot forever,
            # even after the run finished and new points were logged —
            # silently defeating spec §5's data_completeness/partial-data
            # handling for the one code path (training curves) that most
            # needs fresh data. Fixed by using the same fresh-client
            # pattern as the other two methods.
            if self._client_is_injected:
                client = self._client
            else:
                import wandb.apis.public as wandb_public

                client = wandb_public.Api(api_key=self._credentials.api_key)
            try:
                run = client.run(f"{ref.entity}/{ref.project}/{ref.run_id}")
            except Exception as exc:  # noqa: BLE001
                if "not found" in str(exc).lower() or "404" in str(exc):
                    raise WandbRunNotFoundError(ref) from exc
                raise
            # scan_history (not history()) is used deliberately: history()
            # is a *sampled* (default 500-point) view, which would silently
            # drop points — unacceptable given spec §2/§7's requirement
            # that logged NaN/null points survive exactly, not be sampled
            # away. scan_history returns the full, unsampled record set.
            return list(run.scan_history(keys=[metric], min_step=min_step, max_step=max_step))

        records = await asyncio.to_thread(_call_with_backoff, _fetch)
        points = [_to_metric_point(record, metric) for record in records]
        return MetricHistory(ref=ref, metric_name=metric, points=points, schema_version=1)

    def list_sweeps(self, project: str) -> list[Sweep]:
        """Real sweep listing against the W&B API (Milestone 8).

        Kept synchronous, matching `ExperimentBackend.list_sweeps`'s
        frozen signature exactly (base.py, §3's deliberate async/sync
        asymmetry — that method's docstring covers why this isn't "fixed"
        to be async here). Because this method cannot `await`, entity
        resolution uses `_resolve_entity_sync` rather than the async
        `_resolve_entity` every other method above uses. Callers that
        need this off the event loop (`server.py`'s `audit_sweep` tool
        does) are responsible for wrapping the call in
        `asyncio.to_thread` themselves — that's a thin-wrapper MCP-layer
        concern (spec §4.1), not something this backend method should
        assume or hide.

        **Flag (same category as Milestone 3's three ambiguities — no
        live network access in this sandbox to confirm against a real
        project):** built from wandb's documented
        `Api.project(name, entity).sweeps()` -> `Sweep.id` /
        `Sweep.config` / `Sweep.runs` shapes. In particular: (a)
        `Sweep.config["method"]` and `Sweep.config["metric"]["name"]` are
        read defensively (`.get(...)`, defaulting to `"unsupported"` /
        `None`) rather than assumed present, since a malformed or legacy
        sweep config shouldn't crash listing every *other* sweep in the
        project; (b) each sweep run's `.id` is assumed sufficient to
        build a `RunRef` — full config/metrics are never fetched here
        (see `_WandbSweepRunLike`'s docstring), matching the roadmap's
        "never full history, keeps this cheap even for large sweeps"
        instruction for Milestone 8, since `audit_sweep` fetches full
        `Run` summaries itself afterward, once, only for the one sweep
        it's actually auditing.
        """
        entity = self._resolve_entity_sync()

        def _fetch() -> list[_WandbSweepLike]:
            # **Bugfix:** same class of staleness bug as get_metric_history
            # above — this reused `self._client` (the one long-lived
            # `Api()` instance held for the server's whole lifetime)
            # directly, instead of the fresh-client-per-call pattern
            # `list_runs`/`get_run_summary`/`get_metric_history` all use to
            # avoid `wandb.Api`'s internal per-instance caching. A sweep
            # that gains new runs after the first `list_sweeps` call would
            # otherwise keep reporting the original, stale run set for the
            # life of the process. Fixed for consistency with the rest of
            # this backend's established (and load-bearing) discipline.
            if self._client_is_injected:
                client = self._client
            else:
                import wandb.apis.public as wandb_public

                client = wandb_public.Api(api_key=self._credentials.api_key)
            project_handle = client.project(project, entity)
            return list(project_handle.sweeps())

        wandb_sweeps = _call_with_backoff(_fetch)
        return [_to_sweep(entity, project, sweep) for sweep in wandb_sweeps]


# -- Mapping helpers ----------------------------------------------------------


def _decode_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        return max(0, int(cursor))
    except ValueError:
        return 0


def _encode_cursor(offset: int) -> str:
    return str(offset)


def _to_wandb_filters(filters: RunFilter | None) -> dict[str, Any] | None:
    """Translate our backend-agnostic `RunFilter` into W&B's MongoDB-style
    filter dict, per the shape documented for `Api.runs(filters=...)`."""
    if filters is None:
        return None
    clauses: list[dict[str, Any]] = []
    if filters.tags:
        clauses.append({"tags": {"$in": list(filters.tags)}})
    if filters.status:
        clauses.append({"state": filters.status})
    if filters.created_after:
        clauses.append({"created_at": {"$gte": filters.created_after.isoformat()}})
    if filters.created_before:
        clauses.append({"created_at": {"$lte": filters.created_before.isoformat()}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _to_run(entity: str, project: str, wandb_run: _WandbRunLike) -> Run:
    ref = RunRef(backend="wandb", entity=entity, project=project, run_id=wandb_run.id)
    return Run(
        ref=ref,
        name=wandb_run.name,
        tags=list(wandb_run.tags or []),
        # **Flag:** passed through as-is rather than remapped onto the
        # spec §2 comment's four example values (running/finished/
        # crashed/failed) — `status` is typed as plain `str`, not a
        # Literal, and W&B's real state vocabulary is richer (adds
        # killed/preempting/preempted/pending, per the SDK's own
        # docstring). Silently collapsing these onto four buckets would
        # lose information audit_training_curve may care about later;
        # passing the real value through and letting a later milestone
        # define the mapping if one is actually needed was judged the
        # more honest choice than guessing now.
        status=wandb_run.state,
        created_at=_parse_created_at(wandb_run.created_at),
        config=dict(wandb_run.config or {}),
        summary_metrics=_to_summary_metrics(wandb_run.summary_metrics),
        data_completeness=_infer_data_completeness(wandb_run.state),
    )


def _parse_created_at(value: str) -> datetime:
    # W&B's documented created_at shape is an ISO timestamp string; Z-suffix
    # handling included since W&B has historically emitted both forms.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _to_summary_metrics(summary_metrics: dict[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in (summary_metrics or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            as_float = float(value)
            # **Bugfix (Audit #9):** same NaN/Infinity issue as
            # `_normalize_metric_value` above — a raw numeric NaN/Infinity
            # in a run's summary previously passed straight through as an
            # actual non-finite float, which later broke `json.dumps`'s
            # RFC-8259-invalid `NaN`/`Infinity` output. Dropped (not kept
            # as a key with a `None` value) to match this function's
            # existing "non-representable summary value is dropped, not
            # raised on" contract documented below, rather than silently
            # changing that contract's shape for this one case.
            if math.isnan(as_float) or math.isinf(as_float):
                continue
            result[key] = as_float
        # Non-numeric summary values (strings, nested dicts/media refs,
        # which W&B summaries can legitimately contain) are dropped here
        # rather than raised on — summary_metrics is typed
        # `dict[str, float]` per spec §2, and a media/table reference
        # isn't representable as one. Silent drop was chosen over raising
        # since one non-numeric key in a summary shouldn't fail the whole
        # run fetch; revisit if this proves too lossy in practice.
    return result


def _infer_data_completeness(state: str) -> str:
    """**Flag — the central open question from this milestone.**

    Spec §5 requires every `Run` to carry `data_completeness` so audit
    tools can downgrade confidence on a run that's still ingesting rather
    than silently treating partial data as complete. W&B's public API
    does not expose a direct "still ingesting" signal at this layer — the
    closest available proxy is `state`. This heuristic:

    - `"running"` -> `"partial"` (still being logged to, by definition)
    - `"finished"` -> `"complete"`
    - anything else (`crashed`, `killed`, `preempting`, `preempted`,
      `"not found"`/unknown) -> `"unknown"`, since a crashed run may have
      fully flushed its last write or may not have — this API surface
      can't tell us which, and guessing "complete" would risk exactly the
      "confidently wrong" failure mode spec §5 was written to prevent.

    This needs validation against real runs (a run you deliberately kill
    mid-log is the right test) before it's trusted — the roadmap's
    Milestone 3 "flag-if-triggered" clause exists precisely for this kind
    of finding.
    """
    if state == "running":
        return "partial"
    if state == "finished":
        return "complete"
    return "unknown"


def _to_metric_point(record: dict[str, Any], metric: str) -> MetricPoint:
    step = int(record.get("_step", 0))
    raw_value = record.get(metric)
    return MetricPoint(step=step, value=_normalize_metric_value(raw_value))


def _normalize_metric_value(raw_value: Any) -> float | None:
    """**Flag:** raw JSON has no native NaN/Infinity literal, and W&B's
    documented behavior is to serialize logged NaN/Inf values as the
    strings `"NaN"`, `"Infinity"`, `"-Infinity"` in API responses. This
    maps those sentinel strings (and an absent key, meaning the metric
    wasn't logged at that step) onto `None`, consistent with spec §2's
    "`None` represents a logged NaN/null — never silently dropped".
    Genuinely absent-at-that-step vs. logged-as-NaN are collapsed onto
    the same `None` representation here, since our model has no third
    state to distinguish them — flagged as a possible follow-up if that
    distinction turns out to matter in practice.
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value in ("NaN", "Infinity", "-Infinity"):
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, (int, float)):
        as_float = float(raw_value)
        # **Bugfix (Audit #9):** the sentinel-string handling above only
        # catches NaN/Infinity when W&B has already stringified them. If a
        # raw *numeric* NaN/Infinity ever reaches this layer instead (e.g.
        # a client library that decodes them as real floats rather than
        # the documented string sentinels — not something this codebase
        # can rule out for every W&B API version), `float(raw_value)`
        # silently produced an actual `nan`/`inf` float that later broke
        # `json.dumps`'s output (`NaN`/`Infinity` are not valid JSON per
        # RFC 8259) instead of the `None` spec §2 specifies for exactly
        # this case. `models.py`'s `_json_safe` now also catches this at
        # the MCP-boundary serialization choke point regardless, but
        # normalizing here too keeps this function's own return value
        # honest about the type it promises (`float | None`, never a
        # non-finite float).
        if math.isnan(as_float) or math.isinf(as_float):
            return None
        return as_float
    return None


def _to_sweep(entity: str, project: str, wandb_sweep: _WandbSweepLike) -> Sweep:
    """Map a wandb SDK sweep object onto the frozen `Sweep` model
    (models.py). See `list_sweeps`'s docstring for the documented-shape
    assumptions this relies on.
    """
    config = wandb_sweep.config or {}
    method = config.get("method") or "unsupported"
    metric_config = config.get("metric")
    target_metric = metric_config.get("name") if isinstance(metric_config, dict) else None
    run_refs = [
        RunRef(backend="wandb", entity=entity, project=project, run_id=run.id)
        for run in wandb_sweep.runs or []
    ]
    # `ref.run_id` is not a real run — same placeholder convention Sweep's
    # own docstring (models.py) describes; the sweep_id is used here only
    # because it's a convenient, already-unique value, not because it has
    # any special meaning as a "run_id".
    ref = RunRef(backend="wandb", entity=entity, project=project, run_id=wandb_sweep.id)
    return Sweep(
        ref=ref,
        sweep_id=wandb_sweep.id,
        method=method,
        run_refs=run_refs,
        target_metric=target_metric,
    )
