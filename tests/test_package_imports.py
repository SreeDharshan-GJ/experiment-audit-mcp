"""Milestone 0 sanity check: the package installs and imports cleanly.

Real tests begin in Milestone 1 (models.py). This file exists only so
CI has a non-empty, passing suite during scaffolding, and is deleted
once Milestone 1's real tests make it redundant.
"""

import experiment_audit


def test_package_has_version():
    assert hasattr(experiment_audit, "__version__")
