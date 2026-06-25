"""Contract-test isolation from root conftest autouse fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_tool_registry():
    """Override root autouse fixture — contract tests must not import app.core."""
    yield
