"""
Collection Management API
REST API for managing Milvus collections for Language Intelligence Platform
"""
from fastapi import APIRouter, HTTPException, Request
from ..logging_config import get_logger

logger = get_logger("cyrex.api.collection_management")

router = APIRouter(prefix="/api/v1/documents", tags=["Collection Management"])


# ============================================================================
# Collection Type Definitions for Language Intelligence Platform
# ============================================================================

COLLECTION_TYPES = {
    "regulatory_documents": {
        "name": "regulatory_documents",
        "description": "Regulatory language evolution tracking",
        "use_case": "Track regulatory language changes over time, identify new regulations, monitor updates",
    },
    "contracts": {
        "name": "contracts",
        "description": "Contract clause evolution and intelligence",
        "use_case": "Track contract clause changes across versions, analyze contract intelligence",
    },
    "leases": {
        "name": "leases",
        "description": "Lease abstraction and management",
        "use_case": "Extract and manage lease terms, abstract lease data, map to regulations",
    },
    "obligations": {
        "name": "obligations",
        "description": "Obligation tracking and dependency graphs",
        "use_case": "Map cascading obligations across contracts and leases, track obligations with deadlines and owners",
    },
    "clauses": {
        "name": "clauses",
        "description": "Clause extraction and evolution",
        "use_case": "Track individual clause changes and patterns, clause-level analysis",
    },
    "compliance_patterns": {
        "name": "compliance_patterns",
        "description": "Compliance pattern mining and prediction",
        "use_case": "Identify patterns in compliance failures, predict risks, flag high-risk language patterns",
    },
    "version_drift": {
        "name": "version_drift",
        "description": "Contract version drift detection",
        "use_case": "Detect divergence from expected language standards, explain what changed and why",
    },
    "knowledge_base": {
        "name": "knowledge_base",
        "description": "Domain-specific knowledge base for language intelligence",
        "use_case": "Store domain-specific knowledge, best practices, reference materials for regulatory/contract/lease intelligence",
    },
}


# ============================================================================
# Collection Management Endpoints
# ============================================================================

@router.get("/collections")
async def list_collections(request: Request):
    """
    List all Milvus collections
    
    **Returns:** List of all collections in Milvus with statistics
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        from pymilvus import connections, utility
        from ..integrations.milvus_store import MilvusConnectionManager
        from ..settings import settings
        
        # Connect to Milvus
        connection_manager = MilvusConnectionManager(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            connection_alias="collection_list"
        )
        
        try:
            connection_manager.connect()
            
            # List all collections
            collections = utility.list_collections()
            
            # Get stats for each collection
            collection_info = []
            for collection_name in collections:
                try:
                    from pymilvus import Collection
                    collection = Collection(collection_name)
                    collection.load()
                    stats = {
                        "name": collection_name,
                        "num_entities": collection.num_entities,
                        "exists": True,
                    }
                    collection_info.append(stats)
                except Exception as e:
                    logger.warning(f"Could not get stats for collection {collection_name}: {e}")
                    collection_info.append({
                        "name": collection_name,
                        "exists": True,
                        "error": str(e),
                    })
            
            return {
                "success": True,
                "collections": collection_info,
                "count": len(collections),
                "request_id": request_id,
            }
        finally:
            connection_manager.disconnect()
    
    except Exception as e:
        logger.error(f"Error listing collections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/stats")
async def get_collection_stats(collection_name: str, request: Request):
    """
    Get statistics for a specific collection
    
    **Returns:** Collection statistics including entity count, health status
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        from ..integrations.milvus_store import get_milvus_store
        
        # Get vector store for the collection
        vector_store = get_milvus_store(collection_name=collection_name)
        
        # Get stats
        stats = vector_store.stats()
        
        return {
            "success": True,
            "collection_name": collection_name,
            "stats": stats,
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/types")
async def get_collection_types(request: Request):
    """
    Get available collection types for Language Intelligence Platform
    
    **Returns:** List of collection types and their descriptions
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    return {
        "success": True,
        "collection_types": list(COLLECTION_TYPES.values()),
        "description": "Available collections for Regulatory Contract Lease Language Intelligence Platform",
        "request_id": request_id,
    }

