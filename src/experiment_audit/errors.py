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
