"""
Enhanced Universal RAG Engine with Advanced Features
Integrates caching, monitoring, async processing, and advanced retrieval
"""

from typing import List, Dict, Any, Optional
import os
import asyncio
from datetime import datetime
from enum import Enum

# Import base classes
try:
    from deepiri_modelkit.rag import (
        UniversalRAGEngine as BaseRAGEngine,
        Document,
        DocumentType,
        IndustryNiche,
        RAGConfig,
        RAGQuery,
        RetrievalResult,
    )
    from deepiri_modelkit.rag.advanced_retrieval import (
        AdvancedRetrievalPipeline,
        SynonymQueryExpander,
        MultiQueryRetriever,
    )
    from deepiri_modelkit.rag.caching import (
        AdvancedCacheManager,
        EmbeddingCache,
        QueryResultCache,
    )
    from deepiri_modelkit.rag.monitoring import (
        RAGMonitor,
        PerformanceTimer,
    )
    from deepiri_modelkit.rag.async_processing import (
        AsyncDocumentIndexer,
        BatchProcessingConfig,
        BatchProcessingResult,
    )
except ImportError:
    # Fallback
    BaseRAGEngine = object
    Document = dict
    DocumentType = str
    
    # Create fallback enum for IndustryNiche
    class IndustryNiche(Enum):
        """Fallback IndustryNiche enum when deepiri-modelkit is not available"""
        INSURANCE = "insurance"
        MANUFACTURING = "manufacturing"
        PROPERTY_MANAGEMENT = "property_management"
        HEALTHCARE = "healthcare"
        CONSTRUCTION = "construction"
        AUTOMOTIVE = "automotive"
        ENERGY = "energy"
        LOGISTICS = "logistics"
        RETAIL = "retail"
        HOSPITALITY = "hospitality"
        GENERIC = "generic"
    
    RAGConfig = dict
    RAGQuery = dict
    RetrievalResult = dict
    AdvancedRetrievalPipeline = None
    AdvancedCacheManager = None
    RAGMonitor = None

from ..integrations.milvus_store import get_milvus_store, MilvusVectorStore
from ..utils.cache import get_redis_client
from ..logging_config import get_logger

logger = get_logger("cyrex.enhanced_rag")


class EnhancedUniversalRAGEngine(BaseRAGEngine):
    """
    Enhanced RAG engine with:
    - Advanced caching (Redis + in-memory)
    - Performance monitoring
    - Async batch processing
    - Query expansion and multi-query retrieval
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        config: RAGConfig,
        enable_caching: bool = True,
        enable_monitoring: bool = True,
        enable_query_expansion: bool = True,
        enable_multi_query: bool = True
    ):
        self.config = config
        self.vector_store: Optional[MilvusVectorStore] = None
        self.reranker = None
        
        # Advanced features
        self.enable_caching = enable_caching
        self.enable_monitoring = enable_monitoring
        self.enable_query_expansion = enable_query_expansion
        self.enable_multi_query = enable_multi_query
        
        # Initialize components
        self.cache_manager = None
        self.embedding_cache = None
        self.query_cache = None
        self.monitor = None
        self.advanced_retrieval = None
        
        super().__init__(config)
    
    def _initialize(self):
        """Initialize all RAG components"""
        try:
            # Initialize Milvus vector store
            self.vector_store = get_milvus_store(
                collection_name=self.config.collection_name,
                host=self.config.vector_db_host,
                port=self.config.vector_db_port,
            )
            logger.info(
                "Enhanced RAG engine initialized",
                industry=self.config.industry.value if hasattr(self.config.industry, 'value') else self.config.industry,
                collection=self.config.collection_name
            )
            
            # Initialize caching
            if self.enable_caching:
                self._initialize_caching()
            
            # Initialize monitoring
            if self.enable_monitoring:
                self._initialize_monitoring()
            
            # Initialize reranker
            if self.config.use_reranking:
                self._load_reranker()
            
            # Initialize advanced retrieval
            if self.enable_query_expansion or self.enable_multi_query:
                self._initialize_advanced_retrieval()
        
        except Exception as e:
            logger.error(f"Failed to initialize enhanced RAG engine: {e}", exc_info=True)
            raise
    
    def _initialize_caching(self):
        """Initialize caching layer"""
        try:
            redis_client = get_redis_client()
            self.cache_manager = AdvancedCacheManager(
                redis_client=redis_client,
                default_ttl=3600,
                max_size=10000
            )
            self.embedding_cache = EmbeddingCache(self.cache_manager)
            self.query_cache = QueryResultCache(self.cache_manager)
            logger.info("Caching layer initialized")
        except Exception as e:
            logger.warning(f"Caching initialization failed: {e}")
            self.enable_caching = False
    
    def _initialize_monitoring(self):
        """Initialize monitoring"""
        try:
            self.monitor = RAGMonitor(max_history=10000)
            logger.info("Monitoring initialized")
        except Exception as e:
            logger.warning(f"Monitoring initialization failed: {e}")
            self.enable_monitoring = False
    
    def _initialize_advanced_retrieval(self):
        """Initialize advanced retrieval pipeline"""
        try:
            query_expander = SynonymQueryExpander() if self.enable_query_expansion else None
            
            self.advanced_retrieval = AdvancedRetrievalPipeline(
                base_retriever=self,
                query_expander=query_expander,
                use_multi_query=self.enable_multi_query,
                use_cache=self.enable_caching,
                cache_manager=self.cache_manager
            )
            logger.info("Advanced retrieval pipeline initialized")
        except Exception as e:
            logger.warning(f"Advanced retrieval initialization failed: {e}")
            self.enable_query_expansion = False
            self.enable_multi_query = False
    
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
        start_time = datetime.now()
        
        try:
            # Check embedding cache
            embedding = None
            if self.enable_caching and self.embedding_cache:
                embedding = self.embedding_cache.get(document.content)
            
            # Generate embedding if not cached
            if embedding is None:
                from langchain_core.documents import Document as LCDocument
                lc_doc = LCDocument(
                    page_content=document.content,
                    metadata=self._document_to_metadata(document)
                )
                
                # Add to vector store (generates embedding internally)
                self.vector_store.add_documents([lc_doc])
                
                # Cache embedding if possible
                if self.enable_caching and self.embedding_cache:
                    # Get embedding from vector store if available
                    try:
                        # This would require accessing the embedding from the store
                        # For now, we'll let the store handle caching
                        pass
                    except Exception:
                        pass
            
            # Record metrics
            if self.enable_monitoring and self.monitor:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.monitor.record_indexing(
                    operation_type="index",
                    num_documents=1,
                    processing_time_ms=processing_time,
                    success=True
                )
            
            logger.info("Document indexed", doc_id=document.id)
            return True
        
        except Exception as e:
            logger.error(f"Failed to index document {document.id}: {e}", exc_info=True)
            
            # Record error
            if self.enable_monitoring and self.monitor:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.monitor.record_indexing(
                    operation_type="index",
                    num_documents=1,
                    processing_time_ms=processing_time,
                    success=False,
                    error=str(e)
                )
            
            return False
    
    async def index_documents_async(
        self,
        documents: List[Document],
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> BatchProcessingResult:
        """Index documents asynchronously in batches"""
        async def index_doc(doc: Document) -> bool:
            return self.index_document(doc)
        
        config = BatchProcessingConfig(
            batch_size=batch_size,
            max_concurrent_batches=5
        )
        
        indexer = AsyncDocumentIndexer(index_doc, config)
        return await indexer.index_documents(documents, progress_callback)
    
    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Index multiple documents (sync wrapper for async)"""
        if not documents:
            return {"success": True, "indexed_count": 0}
        
        # Use async version
        try:
            result = asyncio.run(self.index_documents_async(documents))
            return {
                "success": result.success_rate > 0.9,  # 90%+ success
                "indexed_count": result.successful_items,
                "failed_count": result.failed_items,
                "processing_time_seconds": result.processing_time_seconds,
                "errors": result.errors[:10]  # First 10 errors
            }
        except Exception as e:
            logger.error(f"Batch indexing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "indexed_count": 0
            }
    
    def retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """Retrieve documents with advanced features"""
        start_time = datetime.now()
        cache_hit = False
        reranking_used = False
        query_expansion_used = False
        
        try:
            # Check query cache
            if self.enable_caching and self.query_cache:
                cached = self.query_cache.get(query)
                if cached:
                    cache_hit = True
                    if self.enable_monitoring and self.monitor:
                        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
                        self.monitor.record_retrieval(
                            query=query,
                            results=cached,
                            retrieval_time_ms=retrieval_time,
                            cache_hit=True
                        )
                    return cached
            
            # Use advanced retrieval if enabled
            if self.advanced_retrieval:
                results = self.advanced_retrieval.retrieve(query)
                query_expansion_used = self.enable_query_expansion
            else:
                # Fallback to base retrieval
                results = self._base_retrieve(query)
            
            # Apply reranking if enabled
            if self.config.use_reranking and self.reranker and len(results) > 1:
                results = self._rerank(query.query, results)
                reranking_used = True
            
            # Cache results
            if self.enable_caching and self.query_cache:
                self.query_cache.set(query, results)
            
            # Record metrics
            if self.enable_monitoring and self.monitor:
                retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
                self.monitor.record_retrieval(
                    query=query,
                    results=results,
                    retrieval_time_ms=retrieval_time,
                    cache_hit=cache_hit,
                    reranking_used=reranking_used,
                    query_expansion_used=query_expansion_used
                )
            
            return results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return []
    
    def _base_retrieve(self, query: RAGQuery) -> List[RetrievalResult]:
        """Base retrieval implementation"""
        try:
            # Build filter
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
                doc = self._lc_doc_to_document(lc_doc)
                result = RetrievalResult(
                    document=doc,
                    score=float(score),
                )
                retrieval_results.append(result)
            
            return retrieval_results
        
        except Exception as e:
            logger.error(f"Base retrieval failed: {e}", exc_info=True)
            return []
    
    def _rerank(self, query: str, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank candidates using cross-encoder"""
        try:
            pairs = [[query, c.document.content] for c in candidates]
            scores = self.reranker.predict(pairs)
            
            for i, candidate in enumerate(candidates):
                candidate.rerank_score = float(scores[i])
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x.rerank_score, reverse=True)
            
            return candidates
        
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates
    
    def _build_filter(self, query: RAGQuery) -> Optional[Dict[str, Any]]:
        """Build Milvus filter expression"""
        filters = {}
        
        if query.industry:
            filters["industry"] = query.industry.value if hasattr(query.industry, 'value') else query.industry
        
        if query.doc_types:
            doc_type_values = [
                dt.value if hasattr(dt, 'value') else dt
                for dt in query.doc_types
            ]
            if len(doc_type_values) == 1:
                filters["doc_type"] = doc_type_values[0]
            else:
                filters["doc_type"] = {"$in": doc_type_values}
        
        if query.date_range:
            start_date, end_date = query.date_range
            filters["created_at"] = {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat(),
            }
        
        if query.metadata_filters:
            filters.update(query.metadata_filters)
        
        return filters if filters else None
    
    def _document_to_metadata(self, document: Document) -> Dict[str, Any]:
        """Convert Document to metadata dict"""
        return {
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
    
    def _lc_doc_to_document(self, lc_doc) -> Document:
        """Convert LangChain Document to Document"""
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
    
    def generate_with_context(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        llm_prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using retrieved context"""
        context_parts = []
        for i, result in enumerate(retrieved_docs[:5]):
            doc = result.document
            context_parts.append(
                f"[Document {i+1}] {doc.title or 'Untitled'}\n"
                f"Type: {doc.doc_type.value if hasattr(doc.doc_type, 'value') else doc.doc_type}\n"
                f"Content: {doc.content}\n"
                f"Source: {doc.source or 'Unknown'}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
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
            
            # Invalidate cache
            if self.enable_caching and self.query_cache:
                # Invalidate all queries (simplified - could be more targeted)
                self.cache_manager.clear(namespace="rag:queries")
            
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}", exc_info=True)
            return False
    
    def update_document(self, doc_id: str, document: Document) -> bool:
        """Update an existing document"""
        try:
            self.delete_documents([doc_id])
            return self.index_document(document)
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}", exc_info=True)
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        try:
            stats = self.vector_store.stats()
            
            result = {
                "collection_name": self.config.collection_name,
                "industry": self.config.industry.value if hasattr(self.config.industry, 'value') else self.config.industry,
                "vector_db_type": self.config.vector_db_type,
                **stats,
            }
            
            # Add monitoring stats
            if self.enable_monitoring and self.monitor:
                result["monitoring"] = self.monitor.get_performance_report()
            
            # Add cache stats
            if self.enable_caching and self.cache_manager:
                result["cache"] = self.cache_manager.get_stats()
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {"error": str(e)}


# Factory function
def create_enhanced_rag_engine(
    industry: IndustryNiche = IndustryNiche.GENERIC,
    collection_name: Optional[str] = None,
    enable_caching: bool = True,
    enable_monitoring: bool = True,
    enable_query_expansion: bool = True,
    enable_multi_query: bool = True,
    **kwargs
) -> EnhancedUniversalRAGEngine:
    """
    Create enhanced universal RAG engine
    
    Args:
        industry: Industry niche
        collection_name: Custom collection name
        enable_caching: Enable Redis caching
        enable_monitoring: Enable performance monitoring
        enable_query_expansion: Enable query expansion
        enable_multi_query: Enable multi-query retrieval
        **kwargs: Additional configuration
        
    Returns:
        Configured EnhancedUniversalRAGEngine
    """
    config = RAGConfig(
        industry=industry,
        collection_name=collection_name or f"deepiri_{industry.value}_rag",
        vector_db_host=os.getenv("MILVUS_HOST", "milvus"),
        vector_db_port=int(os.getenv("MILVUS_PORT", "19530")),
        **kwargs
    )
    
    engine = EnhancedUniversalRAGEngine(
        config=config,
        enable_caching=enable_caching,
        enable_monitoring=enable_monitoring,
        enable_query_expansion=enable_query_expansion,
        enable_multi_query=enable_multi_query
    )
    
    return engine

