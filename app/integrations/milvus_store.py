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
HAS_BASE_RETRIEVER = False
try:
    from langchain_community.vectorstores import Milvus
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    HAS_LANGCHAIN_MILVUS = True
    HAS_BASE_RETRIEVER = True
except ImportError as e:
    logger.warning(f"LangChain Milvus integration not available: {e}")
    Milvus = None
    Document = None
    Embeddings = None
    BaseRetriever = None
    CallbackManagerForRetrieverRun = None

# Use modern langchain-huggingface (eliminates deprecation warnings)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HAS_HUGGINGFACE_EMBEDDINGS = True
except ImportError:
    # Fallback to deprecated version if langchain-huggingface not available
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HAS_HUGGINGFACE_EMBEDDINGS = True
        logger.warning("Using deprecated langchain_community.embeddings.HuggingFaceEmbeddings. Install langchain-huggingface.")
    except ImportError:
        HuggingFaceEmbeddings = None
        HAS_HUGGINGFACE_EMBEDDINGS = False
        logger.warning("HuggingFaceEmbeddings not available. Install langchain-huggingface.")

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
        
        # Initialize embeddings with robust error handling for PyTorch meta tensor issues
        if embedding_model:
            self.embeddings = embedding_model
        else:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            
            embedding_initialized = False
            last_error = None
            
            # Method 1: Try HuggingFaceEmbeddings if available
            if HuggingFaceEmbeddings:
                try:
                    self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
                    logger.info(f"Using embedding model: {model_name} (HuggingFaceEmbeddings)")
                    embedding_initialized = True
                except Exception as e:
                    last_error = e
                    logger.warning(f"HuggingFaceEmbeddings failed: {e}, trying robust wrapper")
            
            # Method 2: Use robust embeddings wrapper (bypasses HuggingFaceEmbeddings issues)
            if not embedding_initialized:
                try:
                    from .embeddings_wrapper import get_robust_embeddings
                    self.embeddings = get_robust_embeddings(model_name)
                    logger.info(f"Using embedding model: {model_name} (RobustEmbeddings wrapper)")
                    embedding_initialized = True
                except Exception as e2:
                    last_error = e2
                    logger.error(f"RobustEmbeddings initialization failed: {e2}")
            
            if not embedding_initialized:
                raise RuntimeError(f"Failed to initialize embeddings after all attempts: {last_error}") from last_error
        
        # Get embedding dimension
        if dimension:
            self.dimension = dimension
        else:
            # Test embedding to get dimension
            try:
                test_embedding = self.embeddings.embed_query("test")
                self.dimension = len(test_embedding)
            except Exception as e:
                # If test embedding fails, use default dimension for all-MiniLM-L6-v2
                logger.warning(f"Failed to get embedding dimension from test query: {e}, using default 384")
                self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        
        # Try to connect to Milvus - fallback to in-memory if unavailable
        if not HAS_PYMILVUS or not connections:
            logger.warning("pymilvus not available, using in-memory fallback")
            self.milvus_available = False
            self.collection = None
            self.langchain_store = None
            self._memory_docs = []
            return
        
        # Use separate connection aliases to avoid conflicts
        self.connection_alias = "default"
        self.langchain_connection_alias = f"langchain_{self.collection_name}"
        
        try:
            # Connect with default alias for PyMilvus operations
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port,
                timeout=10.0,  # Increased timeout for better reliability
            )
            self.milvus_available = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port} (alias: {self.connection_alias})")
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
        
        # Initialize LangChain Milvus wrapper with explicit connection configuration
        # Use a separate connection alias to avoid conflicts with PyMilvus connection
        try:
            # Create a separate connection for LangChain to avoid channel conflicts
            try:
                # Disconnect any existing connection with this alias first
                if connections.has_connection(self.langchain_connection_alias):
                    connections.disconnect(self.langchain_connection_alias)
                
                # Create a new connection specifically for LangChain
                connections.connect(
                    alias=self.langchain_connection_alias,
                    host=self.host,
                    port=self.port,
                    timeout=10.0,  # Increased timeout
                )
                logger.debug(f"Created LangChain connection alias: {self.langchain_connection_alias}")
            except Exception as conn_err:
                logger.warning(f"Failed to create separate LangChain connection: {conn_err}, using default")
                self.langchain_connection_alias = self.connection_alias
            
            # Try new langchain_milvus package first, fallback to deprecated Milvus
            try:
                from langchain_milvus import Milvus as MilvusVectorStore
                # langchain_milvus creates MilvusClient internally which may need uri or connection_args
                # Try both uri and connection_args approaches
                logger.debug(f"Initializing LangChain Milvus wrapper with host={self.host}, port={self.port}")
                
                # Ensure connection is established before creating wrapper
                # The wrapper might create its own connection, so we ensure ours is ready
                self._ensure_connection()
                
                # Try to initialize the wrapper - it may create its own connection
                # If it fails (e.g., defaults to localhost), we'll catch and use fallback
                # Try multiple connection methods since langchain_milvus API may vary
                wrapper_initialized = False
                last_error = None
                
                # Method 1: Try with uri parameter (MilvusClient supports this)
                if not wrapper_initialized:
                    try:
                        uri = f"http://{self.host}:{self.port}"
                        self.langchain_store = MilvusVectorStore(
                            embedding_function=self.embeddings,
                            uri=uri,
                            collection_name=self.collection_name,
                            vector_field="embedding",  # Use our field name, not default "vector"
                            auto_id=True,
                        )
                        logger.info(f"LangChain Milvus wrapper initialized successfully with uri={uri}")
                        wrapper_initialized = True
                    except (TypeError, ValueError) as uri_err:
                        # Parameter error - uri not supported, try next method
                        logger.debug(f"uri parameter not supported, trying connection_args: {uri_err}")
                        last_error = uri_err
                    except Exception as uri_err:
                        # Connection error - check if it's a localhost issue
                        error_msg = str(uri_err).lower()
                        if "localhost" in error_msg:
                            logger.debug(f"uri method connected to localhost instead of {self.host}, trying connection_args: {uri_err}")
                            last_error = uri_err
                        else:
                            # Other error, but might still work with different method
                            logger.debug(f"uri method failed, trying connection_args: {uri_err}")
                            last_error = uri_err
                
                # Method 2: Try with connection_args
                if not wrapper_initialized:
                    try:
                        connection_args = {
                            "host": self.host,  # Explicitly set host (not localhost)
                            "port": self.port,
                            "timeout": 10.0,  # Increased timeout for better reliability
                        }
                        self.langchain_store = MilvusVectorStore(
                            embedding_function=self.embeddings,
                            connection_args=connection_args,
                            collection_name=self.collection_name,
                            vector_field="embedding",  # Use our field name, not default "vector"
                            auto_id=True,
                        )
                        logger.info(f"LangChain Milvus wrapper initialized successfully with connection_args (host={self.host}, port={self.port})")
                        wrapper_initialized = True
                    except Exception as conn_args_err:
                        error_msg = str(conn_args_err).lower()
                        if "localhost" in error_msg:
                            logger.debug(f"connection_args connected to localhost instead of {self.host}, trying host/port: {conn_args_err}")
                        else:
                            logger.debug(f"connection_args failed, trying host/port parameters: {conn_args_err}")
                        last_error = conn_args_err
                
                # Method 3: Removed - langchain_milvus.Milvus doesn't support host/port as separate parameters
                
                # If all methods failed, raise the last error
                if not wrapper_initialized:
                    if last_error:
                        error_msg = str(last_error).lower()
                        if "localhost" in error_msg or "closed channel" in error_msg or "connection" in error_msg:
                            logger.warning(f"LangChain Milvus wrapper failed to connect (may be using wrong host or channel closed). Error: {last_error}")
                        else:
                            logger.warning(f"LangChain Milvus wrapper initialization failed: {last_error}")
                        raise last_error  # Re-raise to be caught by outer except block
                    else:
                        # Should not happen, but handle gracefully
                        raise RuntimeError("Failed to initialize LangChain Milvus wrapper: all connection methods failed")
            except ImportError:
                # Fallback to deprecated version - it expects "vector" field by default
                # We'll create the collection ourselves and let LangChain use it
                # Note: This may still log errors about "vector" field, but they're harmless
                # since we manage the collection schema ourselves
                logger.warning("Using deprecated langchain_community.vectorstores.Milvus. Install langchain-milvus for better compatibility.")
                connection_args = {
                    "host": self.host,  # Explicitly set host (not localhost)
                    "port": self.port,
                    "timeout": 10.0,
                }
                self.langchain_store = Milvus(
                    embedding_function=self.embeddings,
                    connection_args=connection_args,
                    collection_name=self.collection_name,
                )
        except Exception as e:
            logger.warning(f"Failed to create LangChain Milvus wrapper, using in-memory fallback: {e}", exc_info=True)
            # Don't mark as unavailable - PyMilvus connection still works
            # Just disable LangChain wrapper features
            self.langchain_store = None
    
    def _connect(self):
        """Connect to Milvus server with timeout (deprecated - now handled in __init__)"""
        # This method is kept for backwards compatibility but connection is now in __init__
        # Ensure connections are still alive
        self._ensure_connection()
    
    def _ensure_connection(self):
        """Ensure Milvus connections are alive, reconnect if needed"""
        if not HAS_PYMILVUS or not connections:
            return
        
        try:
            # Check default connection
            if not connections.has_connection(self.connection_alias):
                logger.info(f"Reconnecting to Milvus at {self.host}:{self.port}")
                connections.connect(
                    alias=self.connection_alias,
                    host=self.host,
                    port=self.port,
                    timeout=10.0,
                )
            
            # Check LangChain connection if it exists
            if hasattr(self, 'langchain_connection_alias') and self.langchain_connection_alias != self.connection_alias:
                if not connections.has_connection(self.langchain_connection_alias):
                    logger.info(f"Reconnecting LangChain connection to Milvus at {self.host}:{self.port}")
                    connections.connect(
                        alias=self.langchain_connection_alias,
                        host=self.host,
                        port=self.port,
                        timeout=10.0,
                    )
        except Exception as e:
            logger.warning(f"Failed to ensure Milvus connections: {e}")
    
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
        
        # Ensure connection is alive before searching
        self._ensure_connection()
        
        try:
            results = []
            if self.langchain_store:
                results = self.langchain_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                    **kwargs
                )
            else:
                # Fallback to in-memory if LangChain wrapper not available
                logger.warning("LangChain wrapper not available, using in-memory fallback")
                results = self._memory_docs[:k]
            
            # Handle empty collection gracefully (no documents indexed yet)
            if not results:
                logger.debug(f"No documents found in collection '{self.collection_name}' (collection may be empty)")
                return []
            
            return results
        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's an empty collection error (not a connection error)
            if "empty" in error_msg or "no entities" in error_msg or "collection is empty" in error_msg:
                logger.debug(f"Collection '{self.collection_name}' is empty - no documents indexed yet")
                return []
            
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            # Try to reconnect and retry once
            try:
                self._ensure_connection()
                if self.langchain_store:
                    results = self.langchain_store.similarity_search(
                        query=query,
                        k=k,
                        filter=filter,
                        **kwargs
                    )
                    if not results:
                        logger.debug(f"No documents found in collection '{self.collection_name}' after retry")
                        return []
                    return results
            except Exception as retry_error:
                retry_error_msg = str(retry_error).lower()
                if "empty" in retry_error_msg or "no entities" in retry_error_msg:
                    logger.debug(f"Collection '{self.collection_name}' is empty - no documents indexed yet")
                    return []
                logger.error(f"Similarity search retry failed: {retry_error}")
            # Fallback to in-memory
            return self._memory_docs[:k]
    
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
        
        # Ensure connection is alive before searching
        self._ensure_connection()
        
        try:
            results = []
            if self.langchain_store and hasattr(self.langchain_store, 'asimilarity_search'):
                results = await self.langchain_store.asimilarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                    **kwargs
                )
            elif self.langchain_store:
                import asyncio
                results = await asyncio.to_thread(
                    self.similarity_search,
                    query=query,
                    k=k,
                    filter=filter,
                    **kwargs
                )
            else:
                # Fallback to in-memory if LangChain wrapper not available
                logger.warning("LangChain wrapper not available, using in-memory fallback")
                results = self._memory_docs[:k]
            
            # Handle empty collection gracefully (no documents indexed yet)
            if not results:
                logger.debug(f"No documents found in collection '{self.collection_name}' (collection may be empty)")
                return []
            
            return results
        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's an empty collection error (not a connection error)
            if "empty" in error_msg or "no entities" in error_msg or "collection is empty" in error_msg:
                logger.debug(f"Collection '{self.collection_name}' is empty - no documents indexed yet")
                return []
            
            logger.error(f"Async similarity search failed: {e}", exc_info=True)
            # Try to reconnect and retry once
            try:
                self._ensure_connection()
                if self.langchain_store and hasattr(self.langchain_store, 'asimilarity_search'):
                    results = await self.langchain_store.asimilarity_search(
                        query=query,
                        k=k,
                        filter=filter,
                        **kwargs
                    )
                    if not results:
                        logger.debug(f"No documents found in collection '{self.collection_name}' after retry")
                        return []
                    return results
            except Exception as retry_error:
                retry_error_msg = str(retry_error).lower()
                if "empty" in retry_error_msg or "no entities" in retry_error_msg:
                    logger.debug(f"Collection '{self.collection_name}' is empty - no documents indexed yet")
                    return []
                logger.error(f"Async similarity search retry failed: {retry_error}")
            # Fallback to in-memory
            return self._memory_docs[:k]
    
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
            # Simple in-memory retriever that inherits from BaseRetriever
            # This makes it compatible with LangChain's pipe operator (|)
            if HAS_BASE_RETRIEVER and BaseRetriever:
                class SimpleRetriever(BaseRetriever):
                    store: Any  # MilvusVectorStore instance
                    
                    def _get_relevant_documents(
                        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
                    ) -> List[Document]:
                        """Get relevant documents for a query"""
                        return self.store._memory_docs[:4]  # Return first 4 docs
                    
                    async def _aget_relevant_documents(
                        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
                    ) -> List[Document]:
                        """Async get relevant documents for a query"""
                        return self.store._memory_docs[:4]  # Return first 4 docs
                
                return SimpleRetriever(store=self)
            else:
                # Fallback: wrap in RunnableLambda if BaseRetriever not available
                logger.warning("BaseRetriever not available, using simple callable")
                return lambda query: self._memory_docs[:4]
        
        if not self.langchain_store:
            # If LangChain wrapper is not available, create a simple retriever that uses our search methods
            logger.warning("LangChain Milvus wrapper not available, using fallback retriever")
            if not HAS_BASE_RETRIEVER:
                raise RuntimeError("LangChain retriever not available. Milvus connection may have failed.")
            
            # Create a custom retriever that wraps our similarity_search method
            class MilvusRetriever(BaseRetriever):
                store: Any  # MilvusVectorStore instance
                k: int = 4  # default number of retrieved items
                search_params: Optional[Dict[str, Any]] = None
                filter: Optional[str] = None
                output_fields: Optional[List[str]] = None
                
                class Config:
                    arbitrary_types_allowed = True  # Allow non-pydantic objects like MilvusVectorStore
                    extra = "ignore"  # Ignore additional fields without errors
                
                def __init__(self, store: 'MilvusVectorStore', **kwargs):
                    # Initialize with proper field values
                    # Extract known fields and pass to super
                    k_value = kwargs.pop('k', 4)
                    search_params = kwargs.pop('search_params', None)
                    filter_str = kwargs.pop('filter', None)
                    output_fields = kwargs.pop('output_fields', None)
                    # Pass only defined fields to super (extra fields are ignored due to extra="ignore")
                    super().__init__(
                        store=store,
                        k=k_value,
                        search_params=search_params,
                        filter=filter_str,
                        output_fields=output_fields
                    )
                
                def _get_relevant_documents(
                    self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
                ) -> List[Document]:
                    """Get relevant documents using store's similarity_search"""
                    try:
                        results = self.store.similarity_search(query, k=self.k)
                        return results if results else []
                    except Exception as e:
                        logger.error(f"Error in retriever search: {e}")
                        return []
                
                async def _aget_relevant_documents(
                    self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
                ) -> List[Document]:
                    """Async get relevant documents"""
                    try:
                        results = await self.store.asimilarity_search(query, k=self.k)
                        return results if results else []
                    except Exception as e:
                        logger.error(f"Error in async retriever search: {e}")
                        return []
            
            return MilvusRetriever(self, **kwargs)
        
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

