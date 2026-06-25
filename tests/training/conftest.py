"""Isolated conftest for training unit tests (avoids full app imports)."""

import pytest


@pytest.fixture(autouse=True)
def reset_tool_registry():
    yield


@pytest.fixture(autouse=True)
def cleanup_async_resources():
    yield


@pytest.fixture(autouse=True)
def setup_test_env():
    yield
