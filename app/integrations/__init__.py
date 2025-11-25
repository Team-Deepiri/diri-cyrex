"""
Integration layer for external services
Local LLMs, vector stores, and external APIs
"""

from .local_llm import LocalLLMProvider
from .milvus_store import MilvusVectorStore
from .rag_bridge import RAGBridge
from .openai_wrapper import OpenAIProvider, get_openai_provider

__all__ = [
    'LocalLLMProvider',
    'MilvusVectorStore',
    'RAGBridge',
]

