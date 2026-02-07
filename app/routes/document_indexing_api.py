"""
Document Indexing API
REST API for indexing B2B documents into Milvus RAG system
"""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import tempfile
import os
from pathlib import Path

from ..services.document_indexing_service import (
    get_document_indexing_service,
    B2BDocumentType,
    DocumentFormat,
)
from ..logging_config import get_logger

logger = get_logger("cyrex.api.document_indexing")

router = APIRouter(prefix="/api/v1/documents", tags=["Document Indexing"])


# ============================================================================
# Request/Response Models
# ============================================================================

class IndexTextRequest(BaseModel):
    """Request to index text content"""
    text: str = Field(..., description="Text content to index")
    document_id: Optional[str] = Field(None, description="Optional document ID")
    title: str = Field("Untitled Document", description="Document title")
    doc_type: str = Field("other", description="Document type")
    industry: str = Field("generic", description="Industry niche")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchIndexRequest(BaseModel):
    """Request to index multiple files"""
    files: List[Dict[str, Any]] = Field(..., description="List of file info dicts")
    industry: str = Field("generic", description="Industry for all documents")
    chunk_size: Optional[int] = Field(1000, description="Chunk size in characters")
    chunk_overlap: Optional[int] = Field(200, description="Chunk overlap")


class SearchDocumentsRequest(BaseModel):
    """Request to search documents"""
    query: str = Field(..., description="Search query")
    doc_types: Optional[List[str]] = Field(None, description="Filter by document types")
    industry: Optional[str] = Field(None, description="Filter by industry")
    top_k: int = Field(5, description="Number of results")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class DeleteDocumentRequest(BaseModel):
    """Request to delete a document"""
    document_id: str = Field(..., description="Document ID to delete")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/index/text")
async def index_text(req: IndexTextRequest, request: Request):
    """
    Index text content directly
    
    **Use Case:** Index API responses, database content, or generated text
    
    **Example:**
    ```json
    {
        "text": "Company policy: All employees must...",
        "title": "Employee Handbook 2024",
        "doc_type": "policy",
        "industry": "corporate",
        "metadata": {"department": "HR", "version": "2024.1"}
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        service = await get_document_indexing_service()
        
        # Convert doc_type string to enum
        try:
            doc_type = B2BDocumentType(req.doc_type)
        except ValueError:
            doc_type = B2BDocumentType.OTHER
        
        indexed_doc = await service.index_text(
            text=req.text,
            document_id=req.document_id,
            title=req.title,
            doc_type=doc_type,
            industry=req.industry,
            metadata=req.metadata,
        )
        
        return {
            "success": True,
            "document_id": indexed_doc.document_id,
            "title": indexed_doc.title,
            "chunk_count": indexed_doc.chunk_count,
            "indexed_at": indexed_doc.indexed_at.isoformat(),
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error indexing text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/file")
async def index_file_upload(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    doc_type: str = Form("other"),
    industry: str = Form("generic"),
    metadata: Optional[str] = Form(None),  # JSON string
    request: Request = None,
):
    """
    Upload and index a file
    
    **Supported Formats:** PDF, DOCX, TXT, MD, CSV, JSON, HTML, XLSX, PPTX
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/documents/index/file" \
      -F "file=@invoice.pdf" \
      -F "title=Invoice #12345" \
      -F "doc_type=invoice" \
      -F "industry=manufacturing"
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    # Parse metadata if provided
    metadata_dict = {}
    if metadata:
        try:
            import json
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning(f"Invalid metadata JSON: {metadata}")
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        service = await get_document_indexing_service()
        
        # Create temp file
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()  # Ensure data is written to disk
            temp_file_path = temp_file.name
        # File handle is now closed, but file remains on disk (delete=False)
        
        # Convert doc_type string to enum
        try:
            doc_type_enum = B2BDocumentType(doc_type)
        except ValueError:
            doc_type_enum = B2BDocumentType.OTHER
        
        # Index file
        indexed_doc = await service.index_file(
            file_path=temp_file_path,
            document_id=document_id,
            title=title or file.filename,
            doc_type=doc_type_enum,
            industry=industry,
            metadata=metadata_dict,
        )
        
        # Format response to match blueprint expected output
        return {
            "success": True,
            "message": "Document indexed successfully!",
            "document_id": indexed_doc.document_id,
            "title": indexed_doc.title,
            "chunks": indexed_doc.chunk_count,
            "format": indexed_doc.format.value,
            "file_size": indexed_doc.file_size,
            "indexed_at": indexed_doc.indexed_at.isoformat() if indexed_doc.indexed_at else None,
            "doc_type": doc_type,
            "industry": industry,
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error indexing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@router.post("/index/batch")
async def index_batch(req: BatchIndexRequest, request: Request):
    """
    Index multiple files in batch
    
    **Use Case:** Bulk import of company documents, historical data, etc.
    
    **Example:**
    ```json
    {
        "files": [
            {
                "file_path": "/data/invoices/inv_001.pdf",
                "title": "Invoice #001",
                "doc_type": "invoice",
                "metadata": {"vendor": "Acme Corp"}
            },
            {
                "file_path": "/data/manuals/manual_xyz.pdf",
                "title": "Equipment Manual XYZ",
                "doc_type": "manual"
            }
        ],
        "industry": "manufacturing"
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        service = await get_document_indexing_service()
        
        # Convert file info
        files_info = []
        for file_info in req.files:
            file_dict = {
                "file_path": file_info["file_path"],
                "document_id": file_info.get("document_id"),
                "title": file_info.get("title"),
                "doc_type": None,
                "metadata": file_info.get("metadata", {}),
            }
            
            # Convert doc_type if provided
            if file_info.get("doc_type"):
                try:
                    file_dict["doc_type"] = B2BDocumentType(file_info["doc_type"])
                except ValueError:
                    file_dict["doc_type"] = B2BDocumentType.OTHER
            
            files_info.append(file_dict)
        
        # Index batch
        result = await service.index_batch(
            files=files_info,
            industry=req.industry,
        )
        
        return {
            "success": result["success_rate"] > 0.9,
            "total": result["total"],
            "success_count": result["success_count"],
            "failed_count": result["failed_count"],
            "success_rate": result["success_rate"],
            "successful": result["successful"],
            "failed": result["failed"][:10],  # First 10 errors
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error batch indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(req: SearchDocumentsRequest, request: Request):
    """
    Search indexed documents
    
    **Use Cases:**
    - "What are our payment terms with vendor X?"
    - "Find all invoices from Q4 2024"
    - "What's the procedure for equipment maintenance?"
    
    **Example:**
    ```json
    {
        "query": "What are the payment terms?",
        "doc_types": ["contract", "agreement"],
        "industry": "manufacturing",
        "top_k": 5
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        service = await get_document_indexing_service()
        
        # Convert doc_types
        doc_types_enum = None
        if req.doc_types:
            doc_types_enum = []
            for dt in req.doc_types:
                try:
                    doc_types_enum.append(B2BDocumentType(dt))
                except ValueError:
                    pass
        
        # Search
        results = await service.search(
            query=req.query,
            doc_types=doc_types_enum,
            industry=req.industry,
            top_k=req.top_k,
            metadata_filters=req.metadata_filters,
        )
        
        return {
            "success": True,
            "query": req.query,
            "results_count": len(results),
            "results": results,
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
async def delete_document(req: DeleteDocumentRequest, request: Request):
    """Delete a document and all its chunks"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        service = await get_document_indexing_service()
        success = await service.delete_document(req.document_id)
        
        return {
            "success": success,
            "document_id": req.document_id,
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class BatchDeleteDocumentsRequest(BaseModel):
    """Request to delete multiple documents"""
    document_ids: List[str] = Field(..., description="List of document IDs to delete")


@router.post("/delete/batch")
async def batch_delete_documents(req: BatchDeleteDocumentsRequest, request: Request):
    """Delete multiple documents and all their chunks"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        service = await get_document_indexing_service()
        deleted_count = 0
        failed = []
        
        for document_id in req.document_ids:
            try:
                success = await service.delete_document(document_id)
                if success:
                    deleted_count += 1
                else:
                    failed.append(document_id)
            except Exception as e:
                logger.warning(f"Failed to delete document {document_id}: {e}")
                failed.append(document_id)
        
        return {
            "success": deleted_count > 0,
            "deleted_count": deleted_count,
            "failed_count": len(failed),
            "failed": failed,
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error batch deleting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_statistics(request: Request):
    """
    Get indexing statistics
    
    **Returns:**
    - Total documents indexed
    - Total chunks
    - Documents by type
    - Vector store statistics
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        service = await get_document_indexing_service()
        stats = await service.get_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents(
    doc_type: Optional[str] = None,
    industry: Optional[str] = None,
    request: Request = None,
):
    """List all indexed documents with optional filters"""
    request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
    
    try:
        service = await get_document_indexing_service()
        
        result = await service.list_indexed_documents(
            doc_type=doc_type,
            industry=industry,
        )
        
        return {
            "success": True,
            "count": result.get("total", 0),
            "documents": result.get("documents", []),
            "request_id": request_id,
        }
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}")
async def get_document(document_id: str, request: Request):
    """Get metadata for a specific indexed document"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        service = await get_document_indexing_service()
        doc = await service.get_indexed_document(document_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "document": doc,
            "request_id": request_id,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(request: Request):
    """Health check for document indexing service"""
    try:
        service = await get_document_indexing_service()
        stats = await service.get_statistics()
        
        return {
            "status": "healthy",
            "service": "Document Indexing API",
            "collection": stats.get("collection_name"),
            "total_documents": stats.get("total_documents", 0),
            "vector_store_healthy": stats.get("vector_store_stats", {}).get("healthy", False),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }