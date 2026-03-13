"""
Milvus Vector Store Integration - Backwards Compatibility Module.

DEPRECATED: This module is maintained for backwards compatibility only.
Use `from app.integrations.milvus import ...` instead.

This module re-exports all public APIs from the new modular milvus package.
"""
import warnings

warnings.warn(
    "milvus_store module is deprecated. Use 'from app.integrations.milvus import ...' instead",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new module for backwards compatibility
from .milvus import (
    # Main store
    MilvusVectorStore,
    get_milvus_store,
    # Connection management
    MilvusConnectionManager,
    CircuitState,
    # Schema and types
    COLLECTION_TYPES,
    CollectionType,
    get_collection_schema,
    get_default_collections,
    get_collection_types_dict,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_INDEX_PARAMS,
    # Exceptions
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
