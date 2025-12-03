"""
Milvus Vector Store Integration
Production-grade vector database for RAG and semantic search
Integrates with existing KnowledgeRetrievalEngine
"""
from typing import List, Dict, Optional, Any
import os
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.milvus_store")

# LangChain imports with graceful fallbacks
HAS_LANGCHAIN_MILVUS = False
try:
    from langchain_community.vectorstores import Milvus
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    HAS_LANGCHAIN_MILVUS = True
except ImportError as e:
    logger.warning(f"LangChain Milvus integration not available: {e}")
    Milvus = None
    HuggingFaceEmbeddings = None
    Document = None
    Embeddings = None

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    HAS_PYMILVUS = True
except ImportError as e:
    logger.warning(f"pymilvus not available: {e}")
    connections = None
    Collection = None
    FieldSchema = None
    CollectionSchema = None
    DataType = None
    utility = None
    HAS_PYMILVUS = False


class MilvusVectorStore:
    """
    Production Milvus vector store wrapper
    Manages collections, embeddings, and retrieval operations
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: Optional[Embeddings] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dimension: Optional[int] = None,
    ):
        self.collection_name = collection_name
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        
        # Initialize embeddings
        if embedding_model:
            self.embeddings = embedding_model
        else:
            if not HuggingFaceEmbeddings:
                raise ImportError("HuggingFaceEmbeddings not available. Install langchain-community.")
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            logger.info(f"Using embedding model: {model_name}")
        
        # Get embedding dimension
        if dimension:
            self.dimension = dimension
        else:
            # Test embedding to get dimension
            test_embedding = self.embeddings.embed_query("test")
            self.dimension = len(test_embedding)
        
        # Try to connect to Milvus - fallback to in-memory if unavailable
        if not HAS_PYMILVUS or not connections:
            logger.warning("pymilvus not available, using in-memory fallback")
            self.milvus_available = False
            self.collection = None
            self.langchain_store = None
            self._memory_docs = []
            return
        
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=2.0,  # Fast timeout for tests
            )
            self.milvus_available = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(
                f"Milvus unavailable, falling back to in-memory store: {e}"
            )
            self.milvus_available = False
            self.collection = None
            self.langchain_store = None
            # Initialize in-memory fallback
            self._memory_docs = []
            return
        
        # Initialize or get collection
        try:
            self.collection = self._get_or_create_collection()
        except Exception as e:
            logger.warning(f"Failed to initialize collection, using in-memory fallback: {e}")
            self.milvus_available = False
            self.collection = None
            self.langchain_store = None
            self._memory_docs = []
            return
        
        # Initialize LangChain Milvus wrapper with timeout
        try:
            self.langchain_store = Milvus(
                embedding_function=self.embeddings,
                connection_args={
                    "host": self.host,
                    "port": self.port,
                    "timeout": 5.0
                },
                collection_name=self.collection_name,
            )
        except Exception as e:
            logger.warning(f"Failed to create LangChain Milvus wrapper, using in-memory fallback: {e}")
            self.milvus_available = False
            self.collection = None
            self.langchain_store = None
            self._memory_docs = []
    
    def _connect(self):
        """Connect to Milvus server with timeout (deprecated - now handled in __init__)"""
        # This method is kept for backwards compatibility but connection is now in __init__
        pass
    
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            else:
                collection = self._create_collection()
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Load collection into memory
            collection.load()
            return collection
        
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}", exc_info=True)
            raise
    
    def _create_collection(self) -> Collection:
        """Create new Milvus collection with schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Deepiri vector store: {self.collection_name}"
        )
        
        collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index for vector search
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        return collection
    
    def add_documents(self, documents: List[Document], **kwargs):
        """Add documents to vector store"""
        if not self.milvus_available:
            # In-memory fallback
            self._memory_docs.extend(documents)
            return [f"memory_doc_{i}" for i in range(len(self._memory_docs) - len(documents), len(self._memory_docs))]
        
        try:
            return self.langchain_store.add_documents(documents, **kwargs)
        except Exception as e:
            logger.error(f"Failed to add documents: {e}", exc_info=True)
            raise
    
    async def aadd_documents(self, documents: List[Document], **kwargs):
        """Async add documents"""
        if not self.milvus_available:
            # In-memory fallback
            self._memory_docs.extend(documents)
            return [f"memory_doc_{i}" for i in range(len(self._memory_docs) - len(documents), len(self._memory_docs))]
        
        try:
            if hasattr(self.langchain_store, 'aadd_documents'):
                return await self.langchain_store.aadd_documents(documents, **kwargs)
            else:
                # Fallback to sync
                import asyncio
                return await asyncio.to_thread(self.add_documents, documents, **kwargs)
        except Exception as e:
            logger.error(f"Failed to async add documents: {e}", exc_info=True)
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents"""
        if not self.milvus_available:
            # Simple in-memory fallback - return first k documents
            return self._memory_docs[:k]
        
        try:
            return self.langchain_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise
    
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """Async similarity search"""
        if not self.milvus_available:
            # Simple in-memory fallback
            return self._memory_docs[:k]
        
        try:
            if hasattr(self.langchain_store, 'asimilarity_search'):
                return await self.langchain_store.asimilarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                    **kwargs
                )
            else:
                import asyncio
                return await asyncio.to_thread(
                    self.similarity_search,
                    query=query,
                    k=k,
                    filter=filter,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Async similarity search failed: {e}", exc_info=True)
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """Search with similarity scores"""
        if not self.milvus_available:
            # In-memory fallback with dummy scores
            return [(doc, 0.5) for doc in self._memory_docs[:k]]
        
        try:
            return self.langchain_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}", exc_info=True)
            raise
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs):
        """Delete documents by IDs"""
        if not self.milvus_available:
            # In-memory fallback - remove by index if ids provided
            if ids:
                # Simple implementation - remove by position
                for doc_id in ids:
                    if doc_id.startswith("memory_doc_"):
                        idx = int(doc_id.split("_")[-1])
                        if 0 <= idx < len(self._memory_docs):
                            self._memory_docs.pop(idx)
            return {"deleted": len(ids) if ids else 0}
        
        try:
            return self.langchain_store.delete(ids=ids, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}", exc_info=True)
            raise
    
    def get_retriever(self, **kwargs):
        """Get LangChain retriever from vector store"""
        if not self.milvus_available:
            # Simple in-memory retriever
            class SimpleRetriever:
                def __init__(self, store):
                    self.store = store
                
                def get_relevant_documents(self, query):
                    return self.store._memory_docs
                
                async def aget_relevant_documents(self, query):
                    return self.store._memory_docs
            
            return SimpleRetriever(self)
        
        return self.langchain_store.as_retriever(**kwargs)
    
    def stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.milvus_available:
            return {
                "collection_name": self.collection_name,
                "num_entities": len(self._memory_docs),
                "dimension": self.dimension,
                "host": self.host,
                "port": self.port,
                "mode": "in-memory"
            }
        
        try:
            stats = self.collection.num_entities
            return {
                "collection_name": self.collection_name,
                "num_entities": stats,
                "dimension": self.dimension,
                "host": self.host,
                "port": self.port,
                "mode": "milvus"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {"error": str(e)}


def get_milvus_store(
    collection_name: str,
    embedding_model: Optional[Embeddings] = None,
    **kwargs
) -> MilvusVectorStore:
    """
    Factory function to get Milvus vector store
    
    Args:
        collection_name: Name of the collection
        embedding_model: Optional custom embedding model
        **kwargs: Additional configuration
    
    Returns:
        Configured MilvusVectorStore instance
    """
    return MilvusVectorStore(
        collection_name=collection_name,
        embedding_model=embedding_model,
        **kwargs
    )

