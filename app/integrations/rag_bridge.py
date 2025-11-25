"""
RAG Bridge
Bridges existing KnowledgeRetrievalEngine with new LangChain orchestration
Seamless integration between old and new systems
"""
from typing import List, Dict, Optional, Any
from ..logging_config import get_logger

logger = get_logger("cyrex.rag_bridge")

# LangChain imports with graceful fallbacks
try:
    from langchain_core.documents import Document
    HAS_LANGCHAIN = True
except ImportError:
    logger.warning("LangChain core documents not available")
    Document = None
    HAS_LANGCHAIN = False

from ..services.knowledge_retrieval_engine import KnowledgeRetrievalEngine
from .milvus_store import MilvusVectorStore


class RAGBridge:
    """
    Bridge between existing KnowledgeRetrievalEngine and new LangChain system
    Provides unified interface for RAG operations
    """
    
    def __init__(
        self,
        knowledge_engine: Optional[KnowledgeRetrievalEngine] = None,
        milvus_store: Optional[MilvusVectorStore] = None,
    ):
        self.knowledge_engine = knowledge_engine
        self.milvus_store = milvus_store
        self.logger = logger
    
    async def retrieve(
        self,
        query: str,
        knowledge_bases: Optional[List[str]] = None,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents using either system
        
        Args:
            query: Search query
            knowledge_bases: Optional list of knowledge base names
            top_k: Number of results
            filters: Optional filters
        
        Returns:
            List of relevant documents
        """
        # Prefer Milvus if available
        if self.milvus_store:
            try:
                docs = await self.milvus_store.asimilarity_search(
                    query=query,
                    k=top_k,
                    filter=filters,
                )
                self.logger.debug(f"Retrieved {len(docs)} documents from Milvus")
                return docs
            except Exception as e:
                self.logger.warning(f"Milvus retrieval failed, falling back to knowledge engine: {e}")
        
        # Fallback to knowledge engine
        if self.knowledge_engine:
            try:
                docs = self.knowledge_engine.retrieve(
                    query=query,
                    knowledge_bases=knowledge_bases,
                    top_k=top_k,
                    filters=filters,
                )
                self.logger.debug(f"Retrieved {len(docs)} documents from knowledge engine")
                return docs
            except Exception as e:
                self.logger.error(f"Knowledge engine retrieval failed: {e}", exc_info=True)
                return []
        
        self.logger.warning("No RAG system available")
        return []
    
    async def add_documents(
        self,
        content: str,
        metadata: Dict,
        knowledge_base: str = "project_context",
    ):
        """Add documents to RAG system"""
        # Try Milvus first
        if self.milvus_store:
            try:
                doc = Document(page_content=content, metadata=metadata)
                await self.milvus_store.aadd_documents([doc])
                self.logger.info(f"Added document to Milvus: {knowledge_base}")
                return
            except Exception as e:
                self.logger.warning(f"Milvus add failed, falling back: {e}")
        
        # Fallback to knowledge engine
        if self.knowledge_engine:
            try:
                self.knowledge_engine.add_document(
                    content=content,
                    metadata=metadata,
                    knowledge_base=knowledge_base,
                )
                self.logger.info(f"Added document to knowledge engine: {knowledge_base}")
            except Exception as e:
                self.logger.error(f"Failed to add document: {e}", exc_info=True)
                raise


def get_rag_bridge(
    knowledge_engine: Optional[KnowledgeRetrievalEngine] = None,
    milvus_store: Optional[MilvusVectorStore] = None,
) -> RAGBridge:
    """Get RAG bridge instance"""
    return RAGBridge(knowledge_engine, milvus_store)

