"""
Integration layer for external services
Local LLMs, vector stores, and external APIs
"""

from typing import Any

__all__ = [
    "LocalLLMProvider",
    "MilvusVectorStore",
    "RAGBridge",
    "OpenAIProvider",
    "get_openai_provider",
]

LocalLLMProvider: Any
MilvusVectorStore: Any
RAGBridge: Any
OpenAIProvider: Any
get_openai_provider: Any


def __getattr__(name: str) -> Any:
    """Load optional integration providers lazily to avoid package import cycles."""
    if name == "LocalLLMProvider":
        from .local_llm import LocalLLMProvider

        return LocalLLMProvider
    if name == "MilvusVectorStore":
        from .milvus_store import MilvusVectorStore

        return MilvusVectorStore
    if name == "RAGBridge":
        from .rag_bridge import RAGBridge

        return RAGBridge
    if name in {"OpenAIProvider", "get_openai_provider"}:
        from .openai_wrapper import OpenAIProvider, get_openai_provider

        return {
            "OpenAIProvider": OpenAIProvider,
            "get_openai_provider": get_openai_provider,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

