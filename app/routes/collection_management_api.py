"""
Collection Management API
REST API for managing Milvus collections for Language Intelligence Platform
"""
from fastapi import APIRouter, HTTPException, Request
from ..logging_config import get_logger
from ..integrations.milvus import get_collection_types_dict

logger = get_logger("cyrex.api.collection_management")

router = APIRouter(prefix="/api/v1/documents", tags=["Collection Management"])


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
        from ..integrations.milvus import MilvusConnectionManager
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
        from ..integrations.milvus import get_milvus_store
        
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
        "collection_types": list(get_collection_types_dict().values()),
        "description": "Available collections for Regulatory Contract Lease Language Intelligence Platform",
        "request_id": request_id,
    }

