"""
Shared pytest configuration and fixtures for all tests
"""
import pytest
import asyncio
import os
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Generator

# Set test environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NODE_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

# Disable LangSmith tracing during tests to prevent timeouts
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""


@pytest.fixture(autouse=True)
def cleanup_async_resources():
    """
    Cleanup async resources after each test.
    Closes connections to prevent hanging.
    Note: pytest-asyncio handles event loop cleanup automatically.
    """
    yield
    # Close any Milvus connections that might be open
    # This is the most common source of hanging connections
    try:
        from pymilvus import connections
        if connections and hasattr(connections, 'has_connection'):
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
    # Disable LangSmith tracing during tests
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    # Disable Milvus by default to prevent connection attempts
    monkeypatch.setenv("MILVUS_HOST", "")
    monkeypatch.setenv("MILVUS_PORT", "")


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing - uses AsyncMock for async methods"""
    mock = Mock()
    mock.get_llm.return_value = Mock()
    mock.ainvoke = AsyncMock(return_value="Test response from LLM")
    mock.stream = Mock(return_value=iter(["Test", " response", " chunks"]))
    mock.health_check.return_value = {
        "status": "healthy",
        "backend": "mock",
        "model": "test-model"
    }
    mock.is_available.return_value = True
    mock.config = Mock()
    mock.config.model_name = "test-model"
    mock.config.backend = "mock"
    mock.model = "test-model"
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing - uses AsyncMock for async methods"""
    from langchain_core.documents import Document
    
    mock = Mock()
    mock.get_retriever.return_value = Mock()
    # Use AsyncMock for async methods to prevent hanging
    mock.asimilarity_search = AsyncMock(return_value=[
        Document(page_content="Test document content", metadata={})
    ])
    mock.similarity_search = Mock(return_value=[
        Document(page_content="Test document content", metadata={})
    ])
    mock.stats.return_value = {
        "collection_name": "test",
        "num_entities": 1,
        "dimension": 384
    }
    return mock


@pytest.fixture
def clean_tool_registry():
    """Fixture that provides a clean tool registry without default tools"""
    from app.core.tool_registry import ToolRegistry
    return ToolRegistry(load_defaults=False)


@pytest.fixture(autouse=True)
def reset_tool_registry():
    """Reset tool registry before each test to ensure isolation"""
    from app.core.tool_registry import get_tool_registry
    yield
    # After test, reset the global registry
    registry = get_tool_registry()
    registry.reset()


@pytest.fixture
def sample_weather_tool():
    """Sample weather tool for testing"""
    from langchain_core.tools import Tool
    
    def get_weather(city: str) -> str:
        """Get the current weather for a city. Input should be the city name."""
        return f"The weather in {city} is sunny and 72°F with light winds."
    
    return Tool(
        name="get_weather",
        description="Get the current weather for a city. Input should be the city name.",
        func=get_weather
    )


@pytest.fixture
def sample_calculator_tool():
    """Sample calculator tool for testing"""
    from langchain_core.tools import Tool
    
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression. Input should be a valid Python expression."""
        try:
            result = eval(expression)
            return f"The result is {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return Tool(
        name="calculator",
        description="Evaluate a mathematical expression. Input should be a valid Python expression.",
        func=calculator
    )


@pytest.fixture
def sample_search_tool():
    """Sample search tool for testing"""
    from langchain_core.tools import Tool
    
    def search_knowledge(query: str) -> str:
        """Search internal knowledge base. Input should be a search query."""
        return f"Found information about: {query}"
    
    return Tool(
        name="search_knowledge",
        description="Search internal knowledge base. Input should be a search query.",
        func=search_knowledge
    )


@pytest.fixture
def mock_orchestrator(mock_llm_provider, mock_vector_store):
    """Mock orchestrator for testing"""
    from app.core.orchestrator import WorkflowOrchestrator
    
    orchestrator = WorkflowOrchestrator(
        llm_provider=mock_llm_provider,
        vector_store=mock_vector_store
    )
    return orchestrator

