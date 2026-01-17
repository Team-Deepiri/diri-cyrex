"""
Milvus Vector Store Integration.

This module provides a clean, modular interface for working with
Milvus vector databases in the Language Intelligence Platform.

Usage:
    from app.integrations.milvus import get_milvus_store, MilvusVectorStore

    # Get a store instance (cached)
    store = get_milvus_store(collection_name="my_collection")

    # Add documents
    from langchain_core.documents import Document
    docs = [Document(page_content="Hello world", metadata={"source": "test"})]
    ids = store.add_documents(docs)

    # Search
    results = store.similarity_search("hello", k=5)
"""
from .store import MilvusVectorStore, get_milvus_store
from .connection import MilvusConnectionManager, CircuitState
from .schema import (
    COLLECTION_TYPES,
    CollectionType,
    get_collection_schema,
    get_default_collections,
    get_collection_types_dict,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_INDEX_PARAMS,
)
from .exceptions import (
    MilvusError,
    MilvusConnectionError,
    MilvusCollectionError,
    MilvusUnavailableError,
)

__all__ = [
    # Store
    "MilvusVectorStore",
    "get_milvus_store",
    # Connection
    "MilvusConnectionManager",
    "CircuitState",
    # Schema
    "COLLECTION_TYPES",
    "CollectionType",
    "get_collection_schema",
    "get_default_collections",
    "get_collection_types_dict",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_INDEX_PARAMS",
    # Exceptions
    "MilvusError",
    "MilvusConnectionError",
    "MilvusCollectionError",
    "MilvusUnavailableError",
]
