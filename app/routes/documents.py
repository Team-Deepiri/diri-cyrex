"""
Document Management API Routes
REST API for full CRUD operations on Milvus vector store documents
"""
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from ..integrations.milvus_store import get_milvus_store
from ..logging_config import get_logger, ErrorLogger

router = APIRouter()
logger = get_logger("cyrex.documents_api")
error_logger = ErrorLogger()


# ============================================================================
# Request/Response Models
# ============================================================================

class DocumentAddRequest(BaseModel):
    """Request model for adding a single document"""
    content: str = Field(..., description="Document content/text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    collection_name: Optional[str] = Field(default="deepiri_knowledge", description="Collection name")


class DocumentBatchAddRequest(BaseModel):
    """Request model for batch adding documents"""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents with 'content' and 'metadata'")
    collection_name: Optional[str] = Field(default="deepiri_knowledge", description="Collection name")
    batch_size: int = Field(default=100, description="Batch processing size")


class DocumentSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query")
    k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    collection_name: Optional[str] = Field(default="deepiri_knowledge", description="Collection name")


class DocumentUpdateRequest(BaseModel):
    """Request model for updating a document"""
    content: Optional[str] = Field(None, description="New document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New/updated metadata")


class DocumentDeleteByFilterRequest(BaseModel):
    """Request model for bulk deletion by filters"""
    filters: Dict[str, Any] = Field(..., description="Metadata filters for deletion")
    collection_name: Optional[str] = Field(default="deepiri_knowledge", description="Collection name")


class DocumentResponse(BaseModel):
    """Response model for single document operations"""
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for document listing"""
    success: bool
    data: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int
    request_id: Optional[str] = None


class DocumentSearchResponse(BaseModel):
    """Response model for search operations"""
    success: bool
    data: List[Dict[str, Any]]
    query: str
    count: int
    request_id: Optional[str] = None


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/api/documents", response_model=DocumentResponse)
async def add_document(req: DocumentAddRequest, request: Request):
    """
    Add a single document to the vector store
    
    **Example:**
    ```json
    {
        "content": "This is a document about AI",
        "metadata": {"category": "AI", "author": "John"},
        "collection_name": "deepiri_knowledge"
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=req.collection_name)
        
        # Create document
        from langchain_core.documents import Document
        doc = Document(page_content=req.content, metadata=req.metadata)
        
        # Add document
        doc_ids = await store.aadd_documents([doc])
        
        logger.info(f"Added document to collection '{req.collection_name}'")
        
        return {
            "success": True,
            "message": "Document added successfully",
            "data": {
                "id": doc_ids[0] if doc_ids else None,
                "collection": req.collection_name
            },
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/documents")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/documents/batch", response_model=DocumentResponse)
async def batch_add_documents(req: DocumentBatchAddRequest, request: Request):
    """
    Batch add multiple documents to the vector store
    
    **Example:**
    ```json
    {
        "documents": [
            {"content": "Doc 1", "metadata": {"type": "A"}},
            {"content": "Doc 2", "metadata": {"type": "B"}}
        ],
        "collection_name": "deepiri_knowledge",
        "batch_size": 100
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=req.collection_name)
        
        # Batch add documents
        result = await store.abatch_add_documents(req.documents, batch_size=req.batch_size)
        
        logger.info(f"Batch added {result['added']} documents to collection '{req.collection_name}'")
        
        return {
            "success": True,
            "message": f"Added {result['added']} documents",
            "data": result,
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/documents/batch")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    query: str = Query(..., description="Search query"),
    k: int = Query(default=5, ge=1, le=100, description="Number of results"),
    collection_name: str = Query(default="deepiri_knowledge", description="Collection name"),
    request: Request = None
):
    """
    Search for documents using semantic similarity
    
    **Example:**
    ```
    GET /api/documents/search?query=AI technology&k=5&collection_name=deepiri_knowledge
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=collection_name)
        
        # Search documents
        results = await store.asimilarity_search(query=query, k=k)
        
        # Format results
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "id": doc.metadata.get("id", None)
            }
            for doc in results
        ]
        
        logger.info(f"Search returned {len(formatted_results)} results for query: '{query}'")
        
        return {
            "success": True,
            "data": formatted_results,
            "query": query,
            "count": len(formatted_results),
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/documents/search")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    collection_name: str = Query(default="deepiri_knowledge", description="Collection name"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max documents to return"),
    offset: int = Query(default=0, ge=0, description="Number of documents to skip"),
    request: Request = None
):
    """
    List documents with pagination
    
    **Example:**
    ```
    GET /api/documents?collection_name=deepiri_knowledge&limit=10&offset=0
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=collection_name)
        
        # List documents
        documents = await store.alist_documents(filters=None, limit=limit, offset=offset)
        
        # Get total count
        total = await store.acount_documents()
        
        logger.info(f"Listed {len(documents)} documents from collection '{collection_name}'")
        
        return {
            "success": True,
            "data": documents,
            "total": total,
            "offset": offset,
            "limit": limit,
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/documents")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/collections", response_model=DocumentResponse)
async def list_collections(request: Request = None):
    """
    List all available collections in Milvus

    **Example:**
    ```
    GET /api/documents/collections
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'

    try:
        from ..integrations.milvus_store import MilvusVectorStore

        # List all collections
        collections = await MilvusVectorStore.alist_all_collections()

        logger.info(f"Listed {len(collections)} collections")

        return {
            "success": True,
            "data": {
                "collections": collections,
                "count": len(collections)
            },
            "request_id": request_id
        }

    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/documents/collections")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/collections/{collection_name}/info", response_model=DocumentResponse)
async def get_collection_info(collection_name: str, request: Request = None):
    """
    Get detailed information about a specific collection

    **Example:**
    ```
    GET /api/documents/collections/deepiri_knowledge/info
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'

    try:
        from ..integrations.milvus_store import MilvusVectorStore

        # Get collection info
        info = await MilvusVectorStore.aget_collection_info(collection_name)

        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])

        logger.info(f"Retrieved info for collection '{collection_name}'")

        return {
            "success": True,
            "data": info,
            "request_id": request_id
        }

    except HTTPException:
        raise
    except Exception as e:
        error_logger.log_api_error(e, request_id, f"/api/documents/collections/{collection_name}/info")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    collection_name: str = Query(default="deepiri_knowledge", description="Collection name"),
    request: Request = None
):
    """
    Get a specific document by ID
    
    **Example:**
    ```
    GET /api/documents/12345?collection_name=deepiri_knowledge
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=collection_name)
        
        # Get document by ID
        document = await store.aget_document_by_id(doc_id)
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        logger.info(f"Retrieved document {doc_id} from collection '{collection_name}'")
        
        return {
            "success": True,
            "data": document,
            "request_id": request_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_logger.log_api_error(e, request_id, f"/api/documents/{doc_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/documents/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: str,
    req: DocumentUpdateRequest,
    collection_name: str = Query(default="deepiri_knowledge", description="Collection name"),
    request: Request = None
):
    """
    Update a document's content and/or metadata
    
    **Note:** Milvus updates are implemented as delete + re-add, so the document ID will change.
    
    **Example:**
    ```json
    PUT /api/documents/12345?collection_name=deepiri_knowledge
    {
        "content": "Updated content",
        "metadata": {"status": "reviewed"}
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=collection_name)
        
        # Update document
        result = await store.aupdate_document(doc_id, content=req.content, metadata=req.metadata)
        
        if not result.get("updated"):
            raise HTTPException(status_code=404, detail=result.get("error", "Update failed"))
        
        logger.info(f"Updated document {doc_id} in collection '{collection_name}'")
        
        return {
            "success": True,
            "message": "Document updated successfully",
            "data": result,
            "request_id": request_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_logger.log_api_error(e, request_id, f"/api/documents/{doc_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/documents/{doc_id}", response_model=DocumentResponse)
async def delete_document(
    doc_id: str,
    collection_name: str = Query(default="deepiri_knowledge", description="Collection name"),
    request: Request = None
):
    """
    Delete a specific document by ID
    
    **Example:**
    ```
    DELETE /api/documents/12345?collection_name=deepiri_knowledge
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=collection_name)
        
        # Delete document
        result = await store.adelete(ids=[doc_id])
        
        logger.info(f"Deleted document {doc_id} from collection '{collection_name}'")
        
        return {
            "success": True,
            "message": "Document deleted successfully",
            "data": result,
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, f"/api/documents/{doc_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/documents", response_model=DocumentResponse)
async def delete_documents_by_filter(req: DocumentDeleteByFilterRequest, request: Request):
    """
    Bulk delete documents matching metadata filters
    
    **Example:**
    ```json
    DELETE /api/documents
    {
        "filters": {"status": "draft", "author": "john"},
        "collection_name": "deepiri_knowledge"
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=req.collection_name)
        
        # Delete by filter
        result = await store.adelete_by_filter(req.filters)
        
        logger.info(f"Deleted {result['deleted']} documents by filter from collection '{req.collection_name}'")
        
        return {
            "success": True,
            "message": f"Deleted {result['deleted']} documents",
            "data": result,
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/api/documents (bulk delete)")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/stats/{collection_name}", response_model=DocumentResponse)
async def get_collection_stats(collection_name: str, request: Request = None):
    """
    Get statistics for a collection
    
    **Example:**
    ```
    GET /api/documents/stats/deepiri_knowledge
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=collection_name)
        
        # Get stats
        stats = store.stats()
        
        logger.info(f"Retrieved stats for collection '{collection_name}'")
        
        return {
            "success": True,
            "data": stats,
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, f"/api/documents/stats/{collection_name}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/health/{collection_name}", response_model=DocumentResponse)
async def health_check(collection_name: str, request: Request = None):
    """
    Perform health check on a collection
    
    **Example:**
    ```
    GET /api/documents/health/deepiri_knowledge
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    try:
        # Get milvus store
        store = get_milvus_store(collection_name=collection_name)
        
        # Health check
        health = store.health_check()
        
        logger.info(f"Health check for collection '{collection_name}': {'healthy' if health.get('healthy') else 'unhealthy'}")
        
        return {
            "success": True,
            "data": health,
            "request_id": request_id
        }
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, f"/api/documents/health/{collection_name}")
        raise HTTPException(status_code=500, detail=str(e))
