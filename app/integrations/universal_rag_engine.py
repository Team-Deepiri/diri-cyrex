"""
Universal RAG Engine Implementation for Cyrex
Concrete implementation using Milvus vector store
"""

from typing import List, Dict, Any, Optional
import os
from datetime import datetime

# Import from deepiri-modelkit (shared library)
try:
    from deepiri_modelkit.rag import (
        UniversalRAGEngine as BaseRAGEngine,
        Document,
        DocumentType,
        IndustryNiche,
        RAGConfig,
        RAGQuery,
        RetrievalResult,
        DocumentProcessor,
        get_processor,
    )
except ImportError:
    # Fallback if modelkit not installed
    from ...integrations.company_data_automation import logger
    logger.warning("deepiri-modelkit not available, using local definitions")
    # Use local definitions as fallback
    BaseRAGEngine = object
    Document = dict
    DocumentType = str
    IndustryNiche = str
    RAGConfig = dict
    RAGQuery = dict
    RetrievalResult = dict

from ..integrations.milvus_store import get_milvus_store, MilvusVectorStore
from ..logging_config import get_logger

logger = get_logger("cyrex.universal_rag")


class UniversalRAGEngine(BaseRAGEngine):
    """
    Production RAG engine for Cyrex using Milvus
    Supports all industries and document types
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store: Optional[MilvusVectorStore] = None
        self.reranker = None
        super().__init__(config)
    
    def _initialize(self):
        """Initialize RAG components"""
        try:
            # Initialize Milvus vector store
            self.vector_store = get_milvus_store(
                collection_name=self.config.collection_name,
                host=self.config.vector_db_host,
                port=self.config.vector_db_port,
            )
            logger.info(
                "Universal RAG engine initialized",
                industry=self.config.industry.value,
                collection=self.config.collection_name
            )
            
            # Initialize reranker if enabled
            if self.config.use_reranking:
                self._load_reranker()
        
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}", exc_info=True)
            raise
    
    def _load_reranker(self):
        """Load cross-encoder reranker"""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.config.reranker_model)
            logger.info("Reranker loaded", model=self.config.reranker_model)
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")
            self.config.use_reranking = False
    
    def index_document(self, document: Document) -> bool:
        """Index a single document"""
        try:
            # Convert to LangChain Document format
            from langchain_core.documents import Document as LCDocument
            
            lc_doc = LCDocument(
                page_content=document.content,
                metadata={
                    "id": document.id,
                    "doc_type": document.doc_type.value if hasattr(document.doc_type, 'value') else document.doc_type,
                    "industry": document.industry.value if hasattr(document.industry, 'value') else document.industry,
                    "title": document.title,
                    "source": document.source,
                    "created_at": document.created_at.isoformat() if document.created_at else None,
                    "updated_at": document.updated_at.isoformat() if document.updated_at else None,
                    "author": document.author,
                    "version": document.version,
                    "chunk_index": document.chunk_index,
                    "total_chunks": document.total_chunks,
                    **document.metadata,
                }
            )
            
            # Add to vector store
            self.vector_store.add_documents([lc_doc])
            
            logger.info(
                "Document indexed",
                doc_id=document.id,
                doc_type=document.doc_type.value if hasattr(document.doc_type, 'value') else document.doc_type
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to index document {document.id}: {e}", exc_info=True)
            return False
    
    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Index multiple documents in batch"""
        try:
            from langchain_core.documents import Document as LCDocument
            
            lc_docs = []
            for doc in documents:
                lc_doc = LCDocument(
                    page_content=doc.content,
                    metadata={
                        "id": doc.id,
                        "doc_type": doc.doc_type.value if hasattr(doc.doc_type, 'value') else doc.doc_type,
                        "industry": doc.industry.value if hasattr(doc.industry, 'value') else doc.industry,
                        "title": doc.title,
                        "source": doc.source,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                        "author": doc.author,
                        "version": doc.version,
                        "chunk_index": doc.chunk_index,
                        "total_chunks": doc.total_chunks,
                        **doc.metadata,
                    }
                )
                lc_docs.append(lc_doc)
            
            # Batch add to vector store
            self.vector_store.add_documents(lc_docs)
            
            logger.info(
                "Batch documents indexed",
                count=len(documents),
                industry=self.config.industry.value
            )
            
            return {
                "success": True,
                "indexed_count": len(documents),
                "collection": self.config.collection_name,
            }
        
        except Exception as e:
            logger.error(f"Failed to batch index documents: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "indexed_count": 0,
            }
    
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """Retrieve relevant documents"""
        try:
            # Build filter expression for Milvus
            filter_dict = self._build_filter(query)
            
            # Perform similarity search
            top_k = query.top_k or self.config.top_k
            results = self.vector_store.similarity_search_with_score(
                query=query.query,
                k=top_k,
                filter=filter_dict,
            )
            
            # Convert to RetrievalResult
            retrieval_results = []
            for lc_doc, score in results:
                # Reconstruct Document from LangChain document
                doc = self._lc_doc_to_document(lc_doc)
                
                result = RetrievalResult(
                    document=doc,
                    score=float(score),
                )
                retrieval_results.append(result)
            
            # Apply reranking if enabled
            if self.config.use_reranking and self.reranker and len(retrieval_results) > 1:
                retrieval_results = self._rerank(query.query, retrieval_results)
            
            # Filter by similarity threshold
            retrieval_results = [
                r for r in retrieval_results
                if r.rerank_score >= self.config.similarity_threshold
                if self.config.use_reranking
                else r.score >= self.config.similarity_threshold
            ]
            
            logger.info(
                "Documents retrieved",
                query=query.query[:100],
                count=len(retrieval_results)
            )
            
            return retrieval_results
        
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}", exc_info=True)
            return []
    
    def _build_filter(self, query: RAGQuery) -> Optional[Dict[str, Any]]:
        """Build Milvus filter expression from query"""
        filters = {}
        
        # Industry filter
        if query.industry:
            filters["industry"] = query.industry.value if hasattr(query.industry, 'value') else query.industry
        
        # Document type filter
        if query.doc_types:
            doc_type_values = [
                dt.value if hasattr(dt, 'value') else dt
                for dt in query.doc_types
            ]
            if len(doc_type_values) == 1:
                filters["doc_type"] = doc_type_values[0]
            else:
                # Milvus supports in operator
                filters["doc_type"] = {"$in": doc_type_values}
        
        # Date range filter
        if query.date_range:
            start_date, end_date = query.date_range
            filters["created_at"] = {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat(),
            }
        
        # Custom metadata filters
        if query.metadata_filters:
            filters.update(query.metadata_filters)
        
        return filters if filters else None
    
    def _lc_doc_to_document(self, lc_doc) -> Document:
        """Convert LangChain Document to our Document model"""
        metadata = lc_doc.metadata
        
        return Document(
            id=metadata.get("id", "unknown"),
            content=lc_doc.page_content,
            doc_type=DocumentType(metadata.get("doc_type", "other")),
            industry=IndustryNiche(metadata.get("industry", "generic")),
            title=metadata.get("title"),
            source=metadata.get("source"),
            created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else None,
            updated_at=datetime.fromisoformat(metadata["updated_at"]) if metadata.get("updated_at") else None,
            author=metadata.get("author"),
            version=metadata.get("version"),
            chunk_index=metadata.get("chunk_index"),
            total_chunks=metadata.get("total_chunks"),
            metadata={k: v for k, v in metadata.items() if k not in [
                "id", "doc_type", "industry", "title", "source",
                "created_at", "updated_at", "author", "version",
                "chunk_index", "total_chunks"
            ]},
        )
    
    def _rerank(self, query: str, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank candidates using cross-encoder"""
        try:
            pairs = [[query, c.document.content] for c in candidates]
            scores = self.reranker.predict(pairs)
            
            for i, candidate in enumerate(candidates):
                candidate.rerank_score = float(scores[i])
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x.rerank_score, reverse=True)
            
            logger.debug(f"Reranked {len(candidates)} candidates")
            return candidates
        
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates
    
    def generate_with_context(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        llm_prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using retrieved context"""
        # Build context from retrieved documents
        context_parts = []
        for i, result in enumerate(retrieved_docs[:5]):  # Use top 5
            doc = result.document
            context_parts.append(
                f"[Document {i+1}] {doc.title or 'Untitled'}\n"
                f"Type: {doc.doc_type.value if hasattr(doc.doc_type, 'value') else doc.doc_type}\n"
                f"Content: {doc.content}\n"
                f"Source: {doc.source or 'Unknown'}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Build prompt
        if llm_prompt_template:
            prompt = llm_prompt_template.format(context=context, query=query)
        else:
            prompt = self._default_prompt_template(context, query)
        
        return {
            "prompt": prompt,
            "context": context,
            "retrieved_count": len(retrieved_docs),
            "retrieved_docs": [r.to_dict() for r in retrieved_docs],
        }
    
    def _default_prompt_template(self, context: str, query: str) -> str:
        """Default prompt template"""
        return f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Answer: Please provide a detailed answer based on the context above. If the context doesn't contain relevant information, please state that clearly."""
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.vector_store.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}", exc_info=True)
            return False
    
    def update_document(self, doc_id: str, document: Document) -> bool:
        """Update an existing document"""
        try:
            # Delete old version
            self.delete_documents([doc_id])
            # Index new version
            return self.index_document(document)
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}", exc_info=True)
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        try:
            stats = self.vector_store.stats()
            return {
                "collection_name": self.config.collection_name,
                "industry": self.config.industry.value,
                "vector_db_type": self.config.vector_db_type,
                **stats,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {"error": str(e)}


# Factory function
def create_universal_rag_engine(
    industry: IndustryNiche = IndustryNiche.GENERIC,
    collection_name: Optional[str] = None,
    **kwargs
) -> UniversalRAGEngine:
    """
    Create a universal RAG engine for specific industry
    
    Args:
        industry: Industry niche
        collection_name: Custom collection name (optional)
        **kwargs: Additional configuration
        
    Returns:
        Configured RAGEngine instance
    """
    config = RAGConfig(
        industry=industry,
        collection_name=collection_name or f"deepiri_{industry.value}_rag",
        vector_db_host=os.getenv("MILVUS_HOST", "milvus"),
        vector_db_port=int(os.getenv("MILVUS_PORT", "19530")),
        **kwargs
    )
    
    engine = UniversalRAGEngine(config)
    return engine

