"""
Shared pytest configuration and fixtures for all tests.

Agent-level mocks and harness fixtures are loaded from
diri-agent-testing-utils (shared across Deepiri services). Cyrex-specific
fixtures (orchestrator wiring, tool samples, connection cleanup) are defined here.
"""
import os
from unittest.mock import Mock

import pytest
from diri_agent_testing_utils import FakeLLMProvider

# Load shared pytest fixtures from diri-agent-testing-utils.
pytest_plugins = ("diri_agent_testing_utils.fixtures.pytest_fixtures",)

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
# Tool registry — cyrex-specific (uses real ToolRegistry)
# ---------------------------------------------------------------------------


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
# Agent test harness — alias shared fixture name for cyrex tests
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_harness_factory(agent_test_harness):
    """
    Factory fixture that wraps any cyrex agent in AgentTestHarness.

    Usage:
        async def test_my_agent(agent_harness_factory):
            harness = agent_harness_factory(my_agent_instance)
            trace = await harness.step("What tasks do I have today?")
            harness.assert_response_contains(trace, "task")
            harness.assert_no_error(trace)
    """
    return agent_test_harness


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
