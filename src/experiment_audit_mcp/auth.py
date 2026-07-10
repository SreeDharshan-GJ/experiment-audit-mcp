"""Credential handling: env-var based auth for the W&B backend.

Implements design-spec-v1.md §6's env-var convention (`WANDB_API_KEY`,
matching the `GITHUB_PERSONAL_ACCESS_TOKEN`-style pattern established by
other MCP servers) and the fail-fast validation requirement — credentials
are resolved once, eagerly, rather than deferring the failure to the
first API call, so `test_connection` running automatically on server
start (§6) surfaces a bad/missing key immediately.

Also resolves the default `entity` needed to construct fully-scoped
`RunRef`s per Revision 1 (see design-spec-v1.md Revision Log).
`WANDB_ENTITY` is optional — if unset, `WandbBackend` falls back to the
authenticated user's default entity as reported by the W&B API at
connection time (see `wandb_backend.py`).

Secrets are never logged or included in any exception message: this
module only ever reports *whether* WANDB_API_KEY is present, never its
value, matching spec §6's "no logging of secrets" requirement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_API_KEY_VAR = "WANDB_API_KEY"
_ENTITY_VAR = "WANDB_ENTITY"


class MissingCredentialsError(Exception):
    """Raised when a required W&B credential is not present in the environment.

    Deliberately does not echo any environment value back in the message
    — only the name of the missing variable — so this exception can never
    leak a partially-set or malformed secret.
    """

    def __init__(self, var_name: str) -> None:
        self.var_name = var_name
        super().__init__(
            f"{var_name} is not set. experiment-audit-mcp requires a W&B API "
            f"key to connect to the real W&B API. Set the {var_name} "
            f"environment variable to a read-only API key (recommended) "
            f"before starting the server. See design-spec-v1.md §6 for the "
            f"read-only-key setup guidance."
        )


@dataclass(frozen=True)
class WandbCredentials:
    """Resolved W&B credentials.

    `entity` is `None` here if `WANDB_ENTITY` was not set — `WandbBackend`
    resolves it against the API's authenticated-user default entity at
    connect time in that case, rather than this module guessing or
    requiring it up front (a project can reasonably be accessed under a
    team entity distinct from the API key owner's personal default).
    """

    api_key: str
    entity: str | None


def load_wandb_credentials() -> WandbCredentials:
    """Fail-fast credential resolution.

    Raises `MissingCredentialsError` immediately if `WANDB_API_KEY` is
    unset, rather than deferring the failure to the first API call —
    this is what lets `test_connection` running on server start (spec §6)
    give an immediate, specific diagnosis instead of a lazy failure deep
    inside the first real tool call.
    """
    api_key = os.environ.get(_API_KEY_VAR)
    if not api_key:
        raise MissingCredentialsError(_API_KEY_VAR)
    entity = os.environ.get(_ENTITY_VAR) or None
    return WandbCredentials(api_key=api_key, entity=entity)
