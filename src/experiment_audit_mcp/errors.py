"""Structured tool error taxonomy: ToolError and the error_type literal set.

Implements design-spec-v1.md §5 exactly. `partial_data` exists as its own
`error_type` (distinct from `run_not_found` and from silently returning
incomplete data as if complete) so a caller can distinguish "this run
doesn't exist" from "this run exists but W&B is still ingesting it" —
conflating the two was the review gap this revision closed. Note that
`partial_data` on a `Run` is primarily surfaced via
`Run.data_completeness` (models.py, spec §2) with audit tools downgrading
their own `confidence`; the `partial_data` error_type here is for cases
where a tool must refuse outright rather than proceed with a downgraded
confidence (the exact boundary is a Milestone 4+ concern, once real tools
exist to draw it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorType = Literal[
    "auth_failed",
    # **Flag (audit finding, dead/unreachable in the current MCP layer):**
    # `server.py`'s `_translate_backend_error` — the sole place that
    # turns a caught backend exception into a `ToolError` — has no
    # branch that ever produces `"rate_limited"`. This is consistent
    # with that function's own docstring ("Rate-limiting is deliberately
    # *not* classified here... by the time an exception reaches this
    # function, the backend has already exhausted its retries"), so this
    # is not a missed classification bug — it's an intentional
    # consequence of where backoff lives. But that means no code path in
    # this repository has ever constructed a `"rate_limited"` ToolError,
    # and nothing exercises one in the test suite; any client written
    # today to specifically branch on `error_type == "rate_limited"`
    # (e.g. to apply its own backoff) is branching on a value this server
    # cannot currently produce. Kept in the literal set as a reservation
    # for a future backend that can report remaining rate-limit budget
    # directly (with a real `retry_after_seconds`) rather than removed,
    # since removing a value from a frozen (spec §5) taxonomy is a
    # breaking change to any client already validating against it — but
    # flagged here so the gap is visible at the point anyone would look
    # to use this value, not just in an audit document.
    "rate_limited",
    "run_not_found",
    "backend_unsupported_capability",
    "insufficient_samples",
    "partial_data",
    "unknown",
]


@dataclass
class ToolError:
    """Structured error returned by a tool instead of raising a bare
    exception across the MCP boundary — spec §5."""

    error_type: ErrorType
    message: str
    recoverable: bool
    retry_after_seconds: int | None = None
