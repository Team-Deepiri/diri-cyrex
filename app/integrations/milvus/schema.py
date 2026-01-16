"""
Collection schemas and type definitions for Milvus.

Centralizes all collection definitions for the Language Intelligence Platform.
"""
from dataclasses import dataclass
from typing import Dict, List

from pymilvus import FieldSchema, CollectionSchema, DataType


# Default embedding dimension (sentence-transformers/all-MiniLM-L6-v2)
DEFAULT_EMBEDDING_DIM = 384

# Default index parameters for HNSW
DEFAULT_INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}


@dataclass
class CollectionType:
    """Definition for a collection type"""
    name: str
    description: str
    use_case: str


# All available collection types for Language Intelligence Platform
COLLECTION_TYPES: Dict[str, CollectionType] = {
    "regulatory_documents": CollectionType(
        name="regulatory_documents",
        description="Regulatory language evolution tracking",
        use_case="Track regulatory language changes over time, identify new regulations, monitor updates"
    ),
    "contracts": CollectionType(
        name="contracts",
        description="Contract clause evolution and intelligence",
        use_case="Track contract clause changes across versions, analyze contract intelligence"
    ),
    "leases": CollectionType(
        name="leases",
        description="Lease abstraction and management",
        use_case="Extract and manage lease terms, abstract lease data, map to regulations"
    ),
    "obligations": CollectionType(
        name="obligations",
        description="Obligation tracking and dependency graphs",
        use_case="Map cascading obligations across contracts and leases, track obligations with deadlines and owners"
    ),
    "clauses": CollectionType(
        name="clauses",
        description="Clause extraction and evolution",
        use_case="Track individual clause changes and patterns, clause-level analysis"
    ),
    "compliance_patterns": CollectionType(
        name="compliance_patterns",
        description="Compliance pattern mining and prediction",
        use_case="Identify patterns in compliance failures, predict risks, flag high-risk language patterns"
    ),
    "version_drift": CollectionType(
        name="version_drift",
        description="Contract version drift detection",
        use_case="Detect divergence from expected language standards, explain what changed and why"
    ),
    "knowledge_base": CollectionType(
        name="knowledge_base",
        description="Domain-specific knowledge base for language intelligence",
        use_case="Store domain-specific knowledge, best practices, reference materials for regulatory/contract/lease intelligence"
    ),
    "deepiri_knowledge": CollectionType(
        name="deepiri_knowledge",
        description="General knowledge base",
        use_case="Default collection for document storage"
    ),
}


def get_collection_schema(dimension: int = DEFAULT_EMBEDDING_DIM) -> CollectionSchema:
    """
    Get the standard collection schema for Milvus.

    Args:
        dimension: Embedding vector dimension (default: 384)

    Returns:
        CollectionSchema with id, text, metadata, and embedding fields
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    ]
    return CollectionSchema(fields=fields, description="Deepiri vector store collection")


def get_default_collections() -> List[str]:
    """
    Get list of collections to create at startup.

    Returns:
        List of collection names for the Language Intelligence Platform
    """
    return [
        "regulatory_documents",
        "contracts",
        "leases",
        "obligations",
        "clauses",
        "compliance_patterns",
        "version_drift",
    ]


def get_collection_types_dict() -> Dict[str, Dict[str, str]]:
    """
    Get collection types as a dictionary (for API responses).

    Returns:
        Dictionary with collection type info for serialization
    """
    return {
        name: {
            "name": ct.name,
            "description": ct.description,
            "use_case": ct.use_case,
        }
        for name, ct in COLLECTION_TYPES.items()
    }
