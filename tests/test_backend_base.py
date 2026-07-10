"""Tests for experiment_audit_mcp.backends.base and .errors — Milestone 2.

Covers, per design-spec-v1.md §3, §5 and the roadmap's Milestone 2
deliverables:
- BackendCapability / NotSupportedError carry the right information
- ExperimentBackend cannot be instantiated without implementing every
  abstract method (proves the ABC contract is actually enforced, not
  just documented)
- ToolError covers the full error_type literal set from spec §5
"""

import pytest

from experiment_audit_mcp.backends.base import (
    BackendCapability,
    ConnectionStatus,
    ExperimentBackend,
    NotSupportedError,
    RunFilter,
)
from experiment_audit_mcp.errors import ToolError


def test_backend_capability_values():
    assert BackendCapability.SWEEPS.value == "sweeps"
    assert BackendCapability.ARTIFACTS.value == "artifacts"


def test_not_supported_error_carries_backend_and_capability():
    err = NotSupportedError("mlflow", BackendCapability.SWEEPS)
    assert err.backend_name == "mlflow"
    assert err.capability is BackendCapability.SWEEPS
    assert "mlflow" in str(err)
    assert "sweeps" in str(err)


def test_experiment_backend_cannot_be_instantiated_directly():
    # ABC with abstract methods must refuse direct instantiation — this
    # is what forces every real backend to implement the full contract
    # rather than silently inheriting a no-op.
    with pytest.raises(TypeError):
        ExperimentBackend()  # type: ignore[abstract]


def test_incomplete_backend_subclass_cannot_be_instantiated():
    class IncompleteBackend(ExperimentBackend):
        name = "incomplete"
        capabilities: set[BackendCapability] = set()
        # Missing test_connection, list_runs, get_run_summary,
        # get_metric_history — must fail to instantiate.

    with pytest.raises(TypeError):
        IncompleteBackend()  # type: ignore[abstract]


def test_default_list_sweeps_raises_not_supported_error():
    class MinimalBackend(ExperimentBackend):
        name = "minimal"
        capabilities: set[BackendCapability] = set()

        async def test_connection(self) -> ConnectionStatus:
            return ConnectionStatus(backend=self.name, authenticated=True)

        async def list_runs(self, project, filters=None, cursor=None, page_size=None):
            raise NotImplementedError

        async def get_run_summary(self, ref):
            raise NotImplementedError

        async def get_metric_history(self, ref, metric, step_range=None):
            raise NotImplementedError

    backend = MinimalBackend()
    with pytest.raises(NotSupportedError) as exc_info:
        backend.list_sweeps("some-project")
    assert exc_info.value.backend_name == "minimal"
    assert exc_info.value.capability is BackendCapability.SWEEPS


def test_run_filter_defaults_apply_no_filtering():
    filters = RunFilter()
    assert filters.tags is None
    assert filters.status is None
    assert filters.created_after is None
    assert filters.created_before is None


def test_connection_status_to_dict_matches_spec_shape():
    status = ConnectionStatus(
        backend="wandb", authenticated=True, scopes_detected=["read"]
    )
    d = status.to_dict()
    assert d == {
        "backend": "wandb",
        "authenticated": True,
        "scopes_detected": ["read"],
        "error": None,
    }


# ---------------------------------------------------------------------------
# ToolError
# ---------------------------------------------------------------------------

ALL_ERROR_TYPES = [
    "auth_failed",
    "rate_limited",
    "run_not_found",
    "backend_unsupported_capability",
    "insufficient_samples",
    "partial_data",
    "unknown",
]


@pytest.mark.parametrize("error_type", ALL_ERROR_TYPES)
def test_tool_error_accepts_every_spec_error_type(error_type):
    err = ToolError(error_type=error_type, message="details", recoverable=False)
    assert err.error_type == error_type
    assert err.retry_after_seconds is None


def test_tool_error_retry_after_seconds_optional_and_settable():
    err = ToolError(
        error_type="rate_limited",
        message="slow down",
        recoverable=True,
        retry_after_seconds=30,
    )
    assert err.retry_after_seconds == 30
