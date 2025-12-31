"""
Universal RAG API Endpoints
REST API for document indexing, retrieval, and generation across all industries
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from deepiri_modelkit.rag import DocumentType, IndustryNiche, RAGQuery
except ImportError:
    # Fallback definitions
    DocumentType = str
    IndustryNiche = str
    RAGQuery = dict

from ..integrations.universal_rag_engine import create_universal_rag_engine
from ..logging_config import get_logger

logger = get_logger("cyrex.api.universal_rag")
router = APIRouter(prefix="/api/v1/universal-rag", tags=["Universal RAG"])

# Global RAG engines (one per industry)
_rag_engines: Dict[str, Any] = {}


def get_rag_engine(industry: str):
    """Get or create RAG engine for industry"""
    if industry not in _rag_engines:
        try:
            industry_enum = IndustryNiche(industry)
        except (ValueError, TypeError):
            industry_enum = IndustryNiche.GENERIC
        
        _rag_engines[industry] = create_universal_rag_engine(industry=industry_enum)
        logger.info(f"Created RAG engine for industry: {industry}")
    
    return _rag_engines[industry]


# Request/Response Models

class IndexDocumentRequest(BaseModel):
    """Request to index a document"""
    id: str = Field(..., description="Unique document ID")
    content: str = Field(..., description="Document content")
    doc_type: str = Field(..., description="Document type (regulation, manual, faq, etc.)")
    industry: str = Field(default="generic", description="Industry niche")
    title: Optional[str] = Field(None, description="Document title")
    source: Optional[str] = Field(None, description="Document source")
    created_at: Optional[str] = Field(None, description="Creation date (ISO format)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchIndexRequest(BaseModel):
    """Request to index multiple documents"""
    documents: List[IndexDocumentRequest]
    industry: str = Field(default="generic", description="Industry for all documents")


class SearchRequest(BaseModel):
    """Request to search documents"""
    query: str = Field(..., description="Search query")
    industry: Optional[str] = Field(None, description="Filter by industry")
    doc_types: Optional[List[str]] = Field(None, description="Filter by document types")
    top_k: Optional[int] = Field(None, description="Number of results to return")
    date_range_start: Optional[str] = Field(None, description="Start date filter (ISO format)")
    date_range_end: Optional[str] = Field(None, description="End date filter (ISO format)")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Custom metadata filters")


class GenerateRequest(BaseModel):
    """Request to generate response with RAG"""
    query: str = Field(..., description="User query")
    industry: Optional[str] = Field(None, description="Industry context")
    doc_types: Optional[List[str]] = Field(None, description="Document types to search")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    llm_prompt_template: Optional[str] = Field(None, description="Custom prompt template")


class DeleteDocumentsRequest(BaseModel):
    """Request to delete documents"""
    doc_ids: List[str] = Field(..., description="Document IDs to delete")
    industry: str = Field(default="generic", description="Industry")


# API Endpoints

@router.post("/index")
async def index_document(req: IndexDocumentRequest, request: Request):
    """
    Index a single document
    
    **Example Request:**
    ```json
    {
        "id": "reg_osha_2024_001",
        "content": "OSHA regulation text...",
        "doc_type": "regulation",
        "industry": "manufacturing",
        "title": "OSHA Safety Standards 2024",
        "source": "osha.gov",
        "metadata": {
            "regulation_number": "29 CFR 1910",
            "effective_date": "2024-01-01"
        }
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Get RAG engine for industry
        engine = get_rag_engine(req.industry)
        
        # Create Document object
        try:
            doc_type_enum = DocumentType(req.doc_type)
        except (ValueError, TypeError):
            doc_type_enum = DocumentType.OTHER
        
        try:
            industry_enum = IndustryNiche(req.industry)
        except (ValueError, TypeError):
            industry_enum = IndustryNiche.GENERIC
        
        from deepiri_modelkit.rag import Document
        document = Document(
            id=req.id,
            content=req.content,
            doc_type=doc_type_enum,
            industry=industry_enum,
            title=req.title,
            source=req.source,
            created_at=datetime.fromisoformat(req.created_at) if req.created_at else None,
            metadata=req.metadata,
        )
        
        # Index document
        success = engine.index_document(document)
        
        if success:
            return {
                "success": True,
                "doc_id": req.id,
                "message": "Document indexed successfully",
                "request_id": request_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to index document")
    
    except Exception as e:
        logger.error(f"Error indexing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/batch")
async def index_documents_batch(req: BatchIndexRequest, request: Request):
    """
    Index multiple documents in batch
    
    **Use Case:** Bulk importing regulations, manuals, or historical data
    
    **Example:** Index 1000 maintenance logs at once
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        engine = get_rag_engine(req.industry)
        
        # Convert requests to Document objects
        from deepiri_modelkit.rag import Document
        documents = []
        
        for doc_req in req.documents:
            try:
                doc_type_enum = DocumentType(doc_req.doc_type)
            except (ValueError, TypeError):
                doc_type_enum = DocumentType.OTHER
            
            try:
                industry_enum = IndustryNiche(req.industry)
            except (ValueError, TypeError):
                industry_enum = IndustryNiche.GENERIC
            
            document = Document(
                id=doc_req.id,
                content=doc_req.content,
                doc_type=doc_type_enum,
                industry=industry_enum,
                title=doc_req.title,
                source=doc_req.source,
                created_at=datetime.fromisoformat(doc_req.created_at) if doc_req.created_at else None,
                metadata=doc_req.metadata,
            )
            documents.append(document)
        
        # Batch index
        result = engine.index_documents(documents)
        
        return {
            "success": result["success"],
            "indexed_count": result["indexed_count"],
            "total_requested": len(req.documents),
            "collection": result.get("collection"),
            "request_id": request_id
        }
    
    except Exception as e:
        logger.error(f"Error batch indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(req: SearchRequest, request: Request):
    """
    Search for relevant documents
    
    **Use Cases:**
    - "Is this repair necessary?" (searches maintenance logs, manuals)
    - "Is this covered by policy?" (searches insurance policies, claims)
    - "What are the safety requirements?" (searches regulations, procedures)
    
    **Example Request:**
    ```json
    {
        "query": "What are the fire safety requirements for manufacturing facilities?",
        "industry": "manufacturing",
        "doc_types": ["regulation", "safety_guideline"],
        "top_k": 5
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Determine industry
        industry = req.industry or "generic"
        engine = get_rag_engine(industry)
        
        # Build RAG query
        try:
            industry_enum = IndustryNiche(industry) if req.industry else None
        except (ValueError, TypeError):
            industry_enum = None
        
        doc_types_enum = None
        if req.doc_types:
            doc_types_enum = []
            for dt in req.doc_types:
                try:
                    doc_types_enum.append(DocumentType(dt))
                except (ValueError, TypeError):
                    pass
        
        date_range = None
        if req.date_range_start and req.date_range_end:
            date_range = (
                datetime.fromisoformat(req.date_range_start),
                datetime.fromisoformat(req.date_range_end)
            )
        
        rag_query = RAGQuery(
            query=req.query,
            industry=industry_enum,
            doc_types=doc_types_enum,
            top_k=req.top_k,
            date_range=date_range,
            metadata_filters=req.metadata_filters,
        )
        
        # Retrieve documents
        results = engine.retrieve(rag_query)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "document": {
                    "id": result.document.id,
                    "title": result.document.title,
                    "content": result.document.content[:500] + "..." if len(result.document.content) > 500 else result.document.content,
                    "doc_type": result.document.doc_type.value if hasattr(result.document.doc_type, 'value') else result.document.doc_type,
                    "industry": result.document.industry.value if hasattr(result.document.industry, 'value') else result.document.industry,
                    "source": result.document.source,
                    "created_at": result.document.created_at.isoformat() if result.document.created_at else None,
                    "metadata": result.document.metadata,
                },
                "score": result.score,
                "rerank_score": result.rerank_score,
            })
        
        return {
            "success": True,
            "query": req.query,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "request_id": request_id
        }
    
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_with_rag(req: GenerateRequest, request: Request):
    """
    Generate response using RAG (Retrieval-Augmented Generation)
    
    **Use Case:** Answer questions using company knowledge base
    
    **Example:**
    - Input: "How do I replace the compressor belt on Model XYZ?"
    - Output: Detailed instructions from equipment manual
    
    **Example Request:**
    ```json
    {
        "query": "How do I replace the compressor belt on Model XYZ?",
        "industry": "manufacturing",
        "doc_types": ["manual", "procedure"],
        "top_k": 3
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        industry = req.industry or "generic"
        engine = get_rag_engine(industry)
        
        # Build query
        try:
            industry_enum = IndustryNiche(industry) if req.industry else None
        except (ValueError, TypeError):
            industry_enum = None
        
        doc_types_enum = None
        if req.doc_types:
            doc_types_enum = []
            for dt in req.doc_types:
                try:
                    doc_types_enum.append(DocumentType(dt))
                except (ValueError, TypeError):
                    pass
        
        rag_query = RAGQuery(
            query=req.query,
            industry=industry_enum,
            doc_types=doc_types_enum,
            top_k=req.top_k,
        )
        
        # Retrieve relevant documents
        retrieved = engine.retrieve(rag_query)
        
        # Generate response with context
        generation_result = engine.generate_with_context(
            query=req.query,
            retrieved_docs=retrieved,
            llm_prompt_template=req.llm_prompt_template
        )
        
        return {
            "success": True,
            "query": req.query,
            "prompt": generation_result["prompt"],
            "context": generation_result["context"],
            "retrieved_count": generation_result["retrieved_count"],
            "retrieved_docs": generation_result["retrieved_docs"],
            "request_id": request_id
        }
    
    except Exception as e:
        logger.error(f"Error generating with RAG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents(req: DeleteDocumentsRequest, request: Request):
    """Delete documents by IDs"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        engine = get_rag_engine(req.industry)
        success = engine.delete_documents(req.doc_ids)
        
        return {
            "success": success,
            "deleted_count": len(req.doc_ids),
            "request_id": request_id
        }
    
    except Exception as e:
        logger.error(f"Error deleting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{industry}")
async def get_statistics(industry: str, request: Request):
    """
    Get statistics about indexed documents for an industry
    
    **Example Response:**
    ```json
    {
        "collection_name": "deepiri_manufacturing_rag",
        "industry": "manufacturing",
        "num_entities": 15420,
        "mode": "milvus",
        "healthy": true
    }
    ```
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        engine = get_rag_engine(industry)
        stats = engine.get_statistics()
        
        return {
            "success": True,
            "stats": stats,
            "request_id": request_id
        }
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Universal RAG API",
        "engines_loaded": list(_rag_engines.keys()),
    }

