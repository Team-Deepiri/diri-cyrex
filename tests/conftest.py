"""
Shared pytest configuration and fixtures for all tests.

Agent-level mocks (FakeLLMProvider, MockVectorStore, MockRedis, FakeToolRegistry,
AgentTestHarness) come from diri-agent-testing-utils, the shared testing library
used across all Deepiri AI services. Cyrex-specific fixtures (orchestrator, tool
samples, connection cleanup) are defined here.
"""
import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock
from typing import Generator

# diri-agent-testing-utils: shared mocks and test harness
from diri_agent_testing_utils import (
    FakeLLMProvider,
    MockVectorStore,
    MockRedis,
    MockMemoryManager,
    FakeToolRegistry,
    AgentTestHarness,
)

# Set test environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NODE_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

# Disable LangSmith tracing during tests to prevent timeouts
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""


# ---------------------------------------------------------------------------
# Environment / connection cleanup (cyrex-specific)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_async_resources():
    """
    Cleanup async resources after each test.
    Closes connections to prevent hanging.
    Note: pytest-asyncio handles event loop cleanup automatically.
    """
    yield
    # Close any Milvus connections that might be open
    try:
        from pymilvus import connections

        if connections and hasattr(connections, "has_connection"):
            try:
                if connections.has_connection("default"):
                    connections.disconnect("default")
            except Exception:
                pass
    except (ImportError, AttributeError, Exception):
        pass


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NODE_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOCAL_LLM_BACKEND", "ollama")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3:8b")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-minimum-32-characters-long-for-testing")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    # Disable Milvus by default to prevent connection attempts
    monkeypatch.setenv("MILVUS_HOST", "")
    monkeypatch.setenv("MILVUS_PORT", "")


# ---------------------------------------------------------------------------
# LLM provider — from diri-agent-testing-utils
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_provider():
    """
    Deterministic fake LLM provider using diri-agent-testing-utils.

    Uses FakeLLMProvider which records all calls and returns a configurable
    default response.  Override responses per-test with:
        mock_llm_provider.response_map["prompt fragment"] = "desired response"
    """
    provider = FakeLLMProvider(default_response="Test response from LLM")
    # Expose health_check and is_available in the shape cyrex code expects
    provider.health_check = Mock(
        return_value={"status": "healthy", "backend": "fake", "model": "test-model"}
    )
    provider.is_available = Mock(return_value=True)
    provider.config = Mock()
    provider.config.model_name = "test-model"
    provider.config.backend = "fake"
    provider.model = "test-model"
    return provider


# ---------------------------------------------------------------------------
# Vector store — from diri-agent-testing-utils
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_store():
    """
    In-memory vector store using diri-agent-testing-utils.

    MockVectorStore does substring matching (no real embeddings).  The
    asimilarity_search / similarity_search methods return dicts; wrap in
    LangChain Documents only if a test needs that specific type.
    """
    return MockVectorStore()


# ---------------------------------------------------------------------------
# Redis mock — from diri-agent-testing-utils
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """
    In-memory Redis mock using diri-agent-testing-utils.

    Supports get/set/delete/exists/expire/keys/flushdb with TTL expiry.
    Use this instead of AsyncMock(redis) to get realistic key-value behaviour.
    """
    return MockRedis()


# ---------------------------------------------------------------------------
# Memory manager — from diri-agent-testing-utils
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_manager():
    """In-memory agent memory manager using diri-agent-testing-utils."""
    return MockMemoryManager()


# ---------------------------------------------------------------------------
# Tool registry — cyrex-specific (uses real ToolRegistry) + shared FakeToolRegistry
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_tool_registry():
    """
    Isolated FakeToolRegistry from diri-agent-testing-utils.

    Tracks every execute() call with timestamps.  Use this for testing
    agent behaviour without pulling in cyrex's default tools.
    """
    return FakeToolRegistry()


@pytest.fixture
def clean_tool_registry():
    """Cyrex ToolRegistry instance with no default tools loaded."""
    from app.core.tool_registry import ToolRegistry

    return ToolRegistry(load_defaults=False)


@pytest.fixture(autouse=True)
def reset_tool_registry():
    """Reset cyrex's global tool registry before each test."""
    from app.core.tool_registry import get_tool_registry

    yield
    registry = get_tool_registry()
    registry.reset()


# ---------------------------------------------------------------------------
# Agent test harness — from diri-agent-testing-utils
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_harness_factory():
    """
    Factory fixture that wraps any cyrex agent in AgentTestHarness.

    Usage:
        async def test_my_agent(agent_harness_factory):
            harness = agent_harness_factory(my_agent_instance)
            trace = await harness.step("What tasks do I have today?")
            harness.assert_response_contains(trace, "task")
            harness.assert_no_error(trace)
    """
    return AgentTestHarness


# ---------------------------------------------------------------------------
# Sample LangChain tools (cyrex-specific)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_weather_tool():
    """Sample weather tool for testing."""
    from langchain_core.tools import Tool

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny and 72°F with light winds."

    return Tool(
        name="get_weather",
        description="Get the current weather for a city. Input should be the city name.",
        func=get_weather,
    )


@pytest.fixture
def sample_calculator_tool():
    """Sample calculator tool for testing."""
    from langchain_core.tools import Tool

    def calculator(expression: str) -> str:
        try:
            result = eval(expression)  # noqa: S307
            return f"The result is {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    return Tool(
        name="calculator",
        description="Evaluate a mathematical expression.",
        func=calculator,
    )


@pytest.fixture
def sample_search_tool():
    """Sample search tool for testing."""
    from langchain_core.tools import Tool

    def search_knowledge(query: str) -> str:
        return f"Found information about: {query}"

    return Tool(
        name="search_knowledge",
        description="Search internal knowledge base.",
        func=search_knowledge,
    )


# ---------------------------------------------------------------------------
# Orchestrator (cyrex-specific, uses shared mocks)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_orchestrator(mock_llm_provider, mock_vector_store):
    """WorkflowOrchestrator wired with fake LLM and in-memory vector store."""
    from app.core.orchestrator import WorkflowOrchestrator

    return WorkflowOrchestrator(
        llm_provider=mock_llm_provider,
        vector_store=mock_vector_store,
    )
