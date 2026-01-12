"""
Milvus Vector Store Integration
Production-grade vector database for RAG and semantic search
Integrates with existing KnowledgeRetrievalEngine
"""
from typing import List, Dict, Optional, Any
import os
import warnings
from pydantic import ConfigDict
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.milvus_store")

# Suppress AsyncMilvusClient initialization warning - it's expected when initializing in sync context
# The async client will be created automatically when async methods are called
warnings.filterwarnings("ignore", message=".*AsyncMilvusClient.*no running event loop.*", category=Warning)

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
        
        # Always initialize in-memory fallback first to ensure it exists
        self._memory_docs = []
        self.milvus_available = False
        self.collection = None
        self.langchain_store = None
        
        # Initialize embeddings with robust error handling for PyTorch meta tensor issues
        if embedding_model:
            self.embeddings = embedding_model
            logger.info("Using provided embedding model")
        else:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            
            embedding_initialized = False
            last_error = None
            errors = []
            
            # Method 1: Try HuggingFaceEmbeddings if available
            if HuggingFaceEmbeddings:
                try:
                    logger.info(f"Attempting to initialize {model_name} with HuggingFaceEmbeddings...")
                    self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
                    # Test the embedding to ensure it works
                    test_embedding = self.embeddings.embed_query("test")
                    if test_embedding and len(test_embedding) > 0:
                        logger.info(f"Successfully initialized embedding model: {model_name} (HuggingFaceEmbeddings, dim={len(test_embedding)})")
                        embedding_initialized = True
                    else:
                        raise RuntimeError("Embedding test returned empty result")
                except Exception as e:
                    error_msg = str(e)
                    errors.append(f"HuggingFaceEmbeddings: {error_msg}")
                    last_error = e
                    if "meta tensor" in error_msg.lower() or "to_empty" in error_msg.lower():
                        logger.warning(f"HuggingFaceEmbeddings failed with meta tensor error: {e}, trying robust wrapper")
                    else:
                        logger.warning(f"HuggingFaceEmbeddings failed: {e}, trying robust wrapper")
            
            # Method 2: Use robust embeddings wrapper (bypasses HuggingFaceEmbeddings issues)
            if not embedding_initialized:
                try:
                    logger.info(f"Attempting to initialize {model_name} with RobustEmbeddings wrapper...")
                    from .embeddings_wrapper import get_robust_embeddings
                    self.embeddings = get_robust_embeddings(model_name)
                    # Test the embedding to ensure it works
                    test_embedding = self.embeddings.embed_query("test")
                    if test_embedding and len(test_embedding) > 0:
                        logger.info(f"Successfully initialized embedding model: {model_name} (RobustEmbeddings wrapper, dim={len(test_embedding)})")
                        embedding_initialized = True
                    else:
                        raise RuntimeError("Embedding test returned empty result")
                except Exception as e2:
                    error_msg = str(e2)
                    errors.append(f"RobustEmbeddings: {error_msg}")
                    last_error = e2
                    logger.error(f"RobustEmbeddings initialization failed: {e2}", exc_info=True)
            
            if not embedding_initialized:
                error_summary = "; ".join(errors) if errors else str(last_error)
                raise RuntimeError(
                    f"Failed to initialize embeddings after all attempts. "
                    f"Model: {model_name}. "
                    f"Errors: {error_summary}. "
                    f"This may be due to PyTorch meta tensor issues, missing dependencies, or insufficient resources."
                ) from last_error
        
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
            logger.info(f"Attempting to connect to Milvus at {self.host}:{self.port}...")
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port,
                timeout=10.0,  # Increased timeout for better reliability
            )
            
            # Verify connection by checking server version
            try:
                from pymilvus import __version__ as pymilvus_version
                logger.info(f"Connected to Milvus at {self.host}:{self.port} (pymilvus version: {pymilvus_version})")
            except Exception:
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
            self.milvus_available = True
        except Exception as e:
            error_msg = str(e).lower()
            if "connection refused" in error_msg or "cannot connect" in error_msg:
                logger.warning(
                    f"Milvus connection refused at {self.host}:{self.port}. "
                    f"Ensure Milvus is running. Falling back to in-memory store."
                )
            elif "timeout" in error_msg:
                logger.warning(
                    f"Milvus connection timeout at {self.host}:{self.port}. "
                    f"Server may be slow or unavailable. Falling back to in-memory store."
                )
            else:
                logger.warning(
                    f"Milvus unavailable at {self.host}:{self.port}: {e}. "
                    f"Falling back to in-memory store."
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
            # _memory_docs already initialized above
            return
        
        # Initialize LangChain Milvus wrapper with explicit connection configuration
        # Use a separate connection alias to avoid conflicts with PyMilvus connection
        try:
            # Ensure default connection is also set up (MilvusClient may use "default" alias)
            # This is important because langchain_milvus.Milvus creates MilvusClient internally
            # which might default to using the "default" connection alias if connection_args aren't properly passed
            try:
                # Disconnect default if it exists and reconnect to ensure correct host
                if connections.has_connection("default"):
                    try:
                        connections.disconnect("default")
                    except Exception:
                        pass
                
                # Create/update default connection to use correct host
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    timeout=10.0,
                )
                logger.debug(f"Ensured default connection alias points to {self.host}:{self.port}")
            except Exception as default_conn_err:
                logger.debug(f"Could not set up default connection alias: {default_conn_err}")
            
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
                
                # Method 2: Try with connection_args (host and port explicitly)
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
                            logger.debug(f"connection_args connected to localhost instead of {self.host}, trying direct host/port: {conn_args_err}")
                        else:
                            logger.debug(f"connection_args failed, trying direct host/port parameters: {conn_args_err}")
                        last_error = conn_args_err
                
                # Method 3: Try with host and port as direct keyword arguments (if supported)
                if not wrapper_initialized:
                    try:
                        self.langchain_store = MilvusVectorStore(
                            embedding_function=self.embeddings,
                            host=self.host,  # Pass host directly
                            port=self.port,  # Pass port directly
                            collection_name=self.collection_name,
                            vector_field="embedding",
                            auto_id=True,
                        )
                        logger.info(f"LangChain Milvus wrapper initialized successfully with direct host/port (host={self.host}, port={self.port})")
                        wrapper_initialized = True
                    except (TypeError, ValueError) as direct_err:
                        # Parameter error - host/port not supported as direct params
                        logger.debug(f"Direct host/port parameters not supported: {direct_err}")
                        last_error = direct_err
                    except Exception as direct_err:
                        error_msg = str(direct_err).lower()
                        if "localhost" in error_msg:
                            logger.debug(f"Direct host/port method connected to localhost instead of {self.host}: {direct_err}")
                            last_error = direct_err
                        else:
                            logger.debug(f"Direct host/port method failed: {direct_err}")
                            last_error = direct_err
                
                # Method 4: Try with using parameter to reference existing default connection
                if not wrapper_initialized:
                    try:
                        self.langchain_store = MilvusVectorStore(
                            embedding_function=self.embeddings,
                            using="default",  # Use the default connection we set up
                            collection_name=self.collection_name,
                            vector_field="embedding",
                            auto_id=True,
                        )
                        logger.info(f"LangChain Milvus wrapper initialized successfully with using='default' connection")
                        wrapper_initialized = True
                    except (TypeError, ValueError) as using_err:
                        # Parameter error - using not supported
                        logger.debug(f"Using parameter not supported: {using_err}")
                        last_error = using_err
                    except Exception as using_err:
                        error_msg = str(using_err).lower()
                        if "localhost" in error_msg:
                            logger.debug(f"Using method connected to localhost instead of {self.host}: {using_err}")
                            last_error = using_err
                        else:
                            logger.debug(f"Using method failed: {using_err}")
                            last_error = using_err
                
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
            # Mark as unavailable if connection fails
            self.milvus_available = False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Milvus connection and collection.
        
        Returns:
            Dictionary with health status, connection info, and collection stats
        """
        health_status = {
            "healthy": False,
            "milvus_available": self.milvus_available,
            "connection": {
                "host": self.host,
                "port": self.port,
                "connected": False,
            },
            "collection": {
                "name": self.collection_name,
                "exists": False,
                "loaded": False,
                "num_entities": 0,
            },
            "embeddings": {
                "initialized": self.embeddings is not None,
                "dimension": self.dimension if hasattr(self, 'dimension') else None,
            },
            "errors": [],
        }
        
        if not HAS_PYMILVUS:
            health_status["errors"].append("pymilvus not available")
            return health_status
        
        # Check connection
        try:
            if connections.has_connection(self.connection_alias):
                health_status["connection"]["connected"] = True
                health_status["healthy"] = True
            else:
                health_status["errors"].append(f"Connection '{self.connection_alias}' not established")
        except Exception as e:
            health_status["errors"].append(f"Connection check failed: {e}")
        
        # Check collection
        if self.milvus_available and self.collection:
            try:
                health_status["collection"]["exists"] = True
                health_status["collection"]["num_entities"] = self.collection.num_entities
                
                # Check if collection is loaded
                try:
                    # Try to query to see if collection is loaded
                    if hasattr(self.collection, 'is_empty'):
                        health_status["collection"]["loaded"] = True
                    else:
                        health_status["collection"]["loaded"] = True  # Assume loaded if we can access it
                except Exception:
                    health_status["collection"]["loaded"] = False
                    health_status["errors"].append("Collection may not be loaded")
            except Exception as e:
                health_status["errors"].append(f"Collection check failed: {e}")
                health_status["healthy"] = False
        
        # Check embeddings
        if self.embeddings:
            try:
                # Test embedding generation
                test_embedding = self.embeddings.embed_query("health check")
                if test_embedding and len(test_embedding) > 0:
                    health_status["embeddings"]["dimension"] = len(test_embedding)
                else:
                    health_status["errors"].append("Embedding test returned empty result")
                    health_status["healthy"] = False
            except Exception as e:
                health_status["errors"].append(f"Embedding test failed: {e}")
                health_status["healthy"] = False
        
        # Overall health is True only if all components are healthy
        if health_status["errors"]:
            health_status["healthy"] = False
        
        return health_status
    
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
        # Ensure _memory_docs exists (safety check)
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
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
        # Ensure _memory_docs exists (safety check)
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
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
        # Ensure _memory_docs exists (safety check)
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
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
            # Ensure _memory_docs exists (safety check)
            if not hasattr(self, '_memory_docs'):
                self._memory_docs = []
            return self._memory_docs[:k]
    
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """Async similarity search"""
        # Ensure _memory_docs exists (safety check)
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
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
            # Ensure _memory_docs exists (safety check)
            if not hasattr(self, '_memory_docs'):
                self._memory_docs = []
            return self._memory_docs[:k]
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """Search with similarity scores"""
        # Ensure _memory_docs exists (safety check)
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
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
        # Ensure _memory_docs exists (safety check)
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
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
    
    async def adelete(self, ids: Optional[List[str]] = None, **kwargs):
        """Async delete documents by IDs"""
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
        if not self.milvus_available:
            if ids:
                for doc_id in ids:
                    if doc_id.startswith("memory_doc_"):
                        idx = int(doc_id.split("_")[-1])
                        if 0 <= idx < len(self._memory_docs):
                            self._memory_docs.pop(idx)
            return {"deleted": len(ids) if ids else 0}
        
        try:
            import asyncio
            return await asyncio.to_thread(self.delete, ids, **kwargs)
        except Exception as e:
            logger.error(f"Failed to async delete documents: {e}", exc_info=True)
            raise
    
    def delete_by_filter(self, filters: Dict[str, Any]) -> Dict[str, int]:
        """
        Delete documents matching metadata filters
        
        Args:
            filters: Dictionary of metadata filters (e.g., {"user_id": "123", "type": "pattern"})
        
        Returns:
            Dictionary with count of deleted documents
        """
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
        if not self.milvus_available:
            original_count = len(self._memory_docs)
            self._memory_docs = [
                doc for doc in self._memory_docs 
                if not all(doc.metadata.get(k) == v for k, v in filters.items())
            ]
            deleted_count = original_count - len(self._memory_docs)
            logger.info(f"Deleted {deleted_count} documents by filter (in-memory)")
            return {"deleted": deleted_count}
        
        try:
            if HAS_PYMILVUS and self.collection:
                expr_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        expr_parts.append(f'metadata["{key}"] == "{value}"')
                    elif isinstance(value, (int, float)):
                        expr_parts.append(f'metadata["{key}"] == {value}')
                    elif isinstance(value, bool):
                        expr_parts.append(f'metadata["{key}"] == {str(value).lower()}')
                
                if not expr_parts:
                    logger.warning("No valid filter expressions, aborting deletion")
                    return {"deleted": 0}
                
                expr = " && ".join(expr_parts)
                result = self.collection.delete(expr)
                deleted_count = result.delete_count if hasattr(result, 'delete_count') else 0
                logger.info(f"Deleted {deleted_count} documents by filter from Milvus")
                return {"deleted": deleted_count}
            else:
                logger.warning("PyMilvus not available for filter-based deletion")
                return {"deleted": 0}
        except Exception as e:
            logger.error(f"Failed to delete by filter: {e}", exc_info=True)
            raise
    
    async def adelete_by_filter(self, filters: Dict[str, Any]) -> Dict[str, int]:
        """Async delete documents by filter"""
        try:
            import asyncio
            return await asyncio.to_thread(self.delete_by_filter, filters)
        except Exception as e:
            logger.error(f"Failed to async delete by filter: {e}", exc_info=True)
            raise
    
    def update_document(
        self, 
        doc_id: str, 
        content: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update existing document content and/or metadata
        Note: Milvus doesn't support true updates, so this deletes and re-adds the document
        
        Args:
            doc_id: Document ID to update
            content: New content (if None, keeps existing)
            metadata: New metadata (if None, keeps existing; otherwise merges with existing)
        
        Returns:
            Dictionary with update status
        """
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
        if not self.milvus_available:
            if doc_id.startswith("memory_doc_"):
                idx = int(doc_id.split("_")[-1])
                if 0 <= idx < len(self._memory_docs):
                    doc = self._memory_docs[idx]
                    if content:
                        doc.page_content = content
                    if metadata:
                        doc.metadata.update(metadata)
                    logger.info(f"Updated document {doc_id} (in-memory)")
                    return {"updated": True, "doc_id": doc_id}
            return {"updated": False, "error": "Document not found"}
        
        try:
            if content is None and metadata is None:
                logger.warning("No content or metadata provided for update")
                return {"updated": False, "error": "Nothing to update"}
            
            logger.warning("Milvus update implemented as delete + re-add. Document ID will change.")
            self.delete(ids=[doc_id])
            new_doc = Document(
                page_content=content if content else "",
                metadata=metadata if metadata else {}
            )
            new_ids = self.add_documents([new_doc])
            
            logger.info(f"Updated document {doc_id} (new ID: {new_ids[0] if new_ids else 'unknown'})")
            return {
                "updated": True, 
                "old_doc_id": doc_id,
                "new_doc_id": new_ids[0] if new_ids else None,
                "note": "Document was deleted and re-added with new ID"
            }
        except Exception as e:
            logger.error(f"Failed to update document: {e}", exc_info=True)
            raise
    
    async def aupdate_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async update document"""
        try:
            import asyncio
            return await asyncio.to_thread(self.update_document, doc_id, content, metadata)
        except Exception as e:
            logger.error(f"Failed to async update document: {e}", exc_info=True)
            raise
    
    def list_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List documents with optional filtering and pagination
        
        Args:
            filters: Optional metadata filters
            limit: Maximum number of documents to return
            offset: Number of documents to skip
        
        Returns:
            List of document dictionaries with id, content preview, and metadata
        """
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
        if not self.milvus_available:
            docs = self._memory_docs
            
            if filters:
                docs = [
                    doc for doc in docs
                    if all(doc.metadata.get(k) == v for k, v in filters.items())
                ]
            
            paginated = docs[offset:offset + limit]
            
            result = []
            for i, doc in enumerate(paginated):
                result.append({
                    "id": f"memory_doc_{offset + i}",
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "content_length": len(doc.page_content),
                    "metadata": doc.metadata
                })
            
            logger.info(f"Listed {len(result)} documents (in-memory)")
            return result
        
        try:
            if HAS_PYMILVUS and self.collection:
                expr = None
                if filters:
                    expr_parts = []
                    for key, value in filters.items():
                        if isinstance(value, str):
                            expr_parts.append(f'metadata["{key}"] == "{value}"')
                        elif isinstance(value, (int, float)):
                            expr_parts.append(f'metadata["{key}"] == {value}')
                    
                    if expr_parts:
                        expr = " && ".join(expr_parts)
                
                query_result = self.collection.query(
                    expr=expr or "",
                    output_fields=["id", "text", "metadata"],
                    limit=limit,
                    offset=offset
                )
                
                result = []
                for entity in query_result:
                    content = entity.get("text", "")
                    result.append({
                        "id": str(entity.get("id", "")),
                        "content_preview": content[:200] + "..." if len(content) > 200 else content,
                        "content_length": len(content),
                        "metadata": entity.get("metadata", {})
                    })
                
                logger.info(f"Listed {len(result)} documents from Milvus")
                return result
            else:
                logger.warning("PyMilvus not available for listing")
                return []
        except Exception as e:
            logger.error(f"Failed to list documents: {e}", exc_info=True)
            return []
    
    async def alist_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Async list documents"""
        try:
            import asyncio
            return await asyncio.to_thread(self.list_documents, filters, limit, offset)
        except Exception as e:
            logger.error(f"Failed to async list documents: {e}", exc_info=True)
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific document by ID with full content
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document dictionary or None if not found
        """
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
        if not self.milvus_available:
            if doc_id.startswith("memory_doc_"):
                idx = int(doc_id.split("_")[-1])
                if 0 <= idx < len(self._memory_docs):
                    doc = self._memory_docs[idx]
                    return {
                        "id": doc_id,
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
            return None
        
        try:
            if HAS_PYMILVUS and self.collection:
                result = self.collection.query(
                    expr=f"id == {doc_id}",
                    output_fields=["id", "text", "metadata"],
                    limit=1
                )
                
                if result:
                    entity = result[0]
                    return {
                        "id": str(entity.get("id", "")),
                        "content": entity.get("text", ""),
                        "metadata": entity.get("metadata", {})
                    }
            
            return None
        except Exception as e:
            logger.error(f"Failed to get document by ID: {e}", exc_info=True)
            return None
    
    async def aget_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Async get document by ID"""
        try:
            import asyncio
            return await asyncio.to_thread(self.get_document_by_id, doc_id)
        except Exception as e:
            logger.error(f"Failed to async get document by ID: {e}", exc_info=True)
            raise
    
    def batch_add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Optimized batch addition of documents
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
            batch_size: Number of documents to process per batch
        
        Returns:
            Dictionary with count of added documents and their IDs
        """
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
        try:
            all_ids = []
            
            doc_objects = [
                Document(
                    page_content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                )
                for doc in documents
            ]
            
            for i in range(0, len(doc_objects), batch_size):
                batch = doc_objects[i:i + batch_size]
                ids = self.add_documents(batch)
                all_ids.extend(ids)
            
            logger.info(f"Batch added {len(all_ids)} documents")
            return {
                "added": len(all_ids),
                "ids": all_ids
            }
        except Exception as e:
            logger.error(f"Failed to batch add documents: {e}", exc_info=True)
            raise
    
    async def abatch_add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Async batch add documents"""
        try:
            all_ids = []
            
            doc_objects = [
                Document(
                    page_content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                )
                for doc in documents
            ]
            
            for i in range(0, len(doc_objects), batch_size):
                batch = doc_objects[i:i + batch_size]
                ids = await self.aadd_documents(batch)
                all_ids.extend(ids)
            
            logger.info(f"Async batch added {len(all_ids)} documents")
            return {
                "added": len(all_ids),
                "ids": all_ids
            }
        except Exception as e:
            logger.error(f"Failed to async batch add documents: {e}", exc_info=True)
            raise
    
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count total documents with optional filtering
        
        Args:
            filters: Optional metadata filters
        
        Returns:
            Count of documents
        """
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
        if not self.milvus_available:
            if not filters:
                return len(self._memory_docs)
            
            count = sum(
                1 for doc in self._memory_docs
                if all(doc.metadata.get(k) == v for k, v in filters.items())
            )
            return count
        
        try:
            if HAS_PYMILVUS and self.collection:
                if not filters:
                    return self.collection.num_entities
                
                expr_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        expr_parts.append(f'metadata["{key}"] == "{value}"')
                    elif isinstance(value, (int, float)):
                        expr_parts.append(f'metadata["{key}"] == {value}')
                
                if not expr_parts:
                    return 0
                
                expr = " && ".join(expr_parts)
                result = self.collection.query(expr=expr, output_fields=["id"])
                return len(result)
            
            return 0
        except Exception as e:
            logger.error(f"Failed to count documents: {e}", exc_info=True)
            return 0
    
    async def acount_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Async count documents"""
        try:
            import asyncio
            return await asyncio.to_thread(self.count_documents, filters)
        except Exception as e:
            logger.error(f"Failed to async count documents: {e}", exc_info=True)
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
                        # Ensure _memory_docs exists (safety check)
                        if not hasattr(self.store, '_memory_docs'):
                            self.store._memory_docs = []
                        return self.store._memory_docs[:4]  # Return first 4 docs
                    
                    async def _aget_relevant_documents(
                        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
                    ) -> List[Document]:
                        """Async get relevant documents for a query"""
                        # Ensure _memory_docs exists (safety check)
                        if not hasattr(self.store, '_memory_docs'):
                            self.store._memory_docs = []
                        return self.store._memory_docs[:4]  # Return first 4 docs
                
                return SimpleRetriever(store=self)
            else:
                # Fallback: wrap in RunnableLambda if BaseRetriever not available
                logger.warning("BaseRetriever not available, using simple callable")
                # Ensure _memory_docs exists (safety check)
                if not hasattr(self, '_memory_docs'):
                    self._memory_docs = []
                _memory_docs_ref = self._memory_docs  # Capture reference for lambda
                return lambda query: _memory_docs_ref[:4]
        
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
                
                model_config = ConfigDict(
                    arbitrary_types_allowed=True,  # Allow non-pydantic objects like MilvusVectorStore
                    extra="ignore"  # Ignore additional fields without errors
                )
                
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
        # Ensure _memory_docs exists (safety check)
        if not hasattr(self, '_memory_docs'):
            self._memory_docs = []
        
        if not self.milvus_available:
            return {
                "collection_name": self.collection_name,
                "num_entities": len(self._memory_docs),
                "dimension": self.dimension,
                "host": self.host,
                "port": self.port,
                "mode": "in-memory",
                "healthy": False,
            }
        
        try:
            # Ensure connection is alive
            self._ensure_connection()
            
            stats = self.collection.num_entities
            return {
                "collection_name": self.collection_name,
                "num_entities": stats,
                "dimension": self.dimension,
                "host": self.host,
                "port": self.port,
                "mode": "milvus",
                "healthy": self.milvus_available,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "healthy": False,
            }

    @staticmethod
    def list_all_collections(host: Optional[str] = None, port: Optional[int] = None) -> List[str]:
        """
        List all available collections in Milvus

        Args:
            host: Milvus host (default from settings)
            port: Milvus port (default from settings)

        Returns:
            List of collection names
        """
        if not HAS_PYMILVUS or not utility:
            logger.warning("pymilvus not available, cannot list collections")
            return []

        milvus_host = host or settings.MILVUS_HOST
        milvus_port = port or settings.MILVUS_PORT

        try:
            import uuid
            # Create temporary connection
            temp_alias = f"temp_list_{uuid.uuid4().hex[:8]}"

            connections.connect(
                alias=temp_alias,
                host=milvus_host,
                port=milvus_port,
                timeout=10.0
            )

            # List all collections
            collections = utility.list_collections(using=temp_alias)

            # Disconnect
            connections.disconnect(temp_alias)

            logger.info(f"Found {len(collections)} collections in Milvus")
            return collections

        except Exception as e:
            logger.error(f"Failed to list collections: {e}", exc_info=True)
            return []

    @staticmethod
    async def alist_all_collections(host: Optional[str] = None, port: Optional[int] = None) -> List[str]:
        """Async version of list_all_collections"""
        import asyncio
        return await asyncio.to_thread(MilvusVectorStore.list_all_collections, host, port)

    @staticmethod
    def get_collection_info(collection_name: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific collection without instantiating MilvusVectorStore

        Args:
            collection_name: Name of the collection
            host: Milvus host (default from settings)
            port: Milvus port (default from settings)

        Returns:
            Dictionary with collection info (schema, count, etc.)
        """
        if not HAS_PYMILVUS or not utility:
            return {"error": "pymilvus not available"}

        milvus_host = host or settings.MILVUS_HOST
        milvus_port = port or settings.MILVUS_PORT

        try:
            import uuid
            # Create temporary connection
            temp_alias = f"temp_info_{uuid.uuid4().hex[:8]}"

            connections.connect(
                alias=temp_alias,
                host=milvus_host,
                port=milvus_port,
                timeout=10.0
            )

            # Check if collection exists
            if not utility.has_collection(collection_name, using=temp_alias):
                connections.disconnect(temp_alias)
                return {"error": f"Collection '{collection_name}' not found"}

            # Get collection
            from pymilvus import Collection as PyMilvusCollection
            collection = PyMilvusCollection(name=collection_name, using=temp_alias)

            # Get schema info
            schema = collection.schema
            fields_info = []
            for field in schema.fields:
                field_dict = {
                    "name": field.name,
                    "type": str(field.dtype),
                    "is_primary": field.is_primary,
                }
                if hasattr(field, 'dim'):
                    field_dict["dimension"] = field.dim
                if hasattr(field, 'max_length'):
                    field_dict["max_length"] = field.max_length
                fields_info.append(field_dict)

            # Get count
            num_entities = collection.num_entities

            # Disconnect
            connections.disconnect(temp_alias)

            return {
                "collection_name": collection_name,
                "num_entities": num_entities,
                "description": schema.description,
                "fields": fields_info,
                "host": milvus_host,
                "port": milvus_port,
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}", exc_info=True)
            return {"error": str(e)}

    @staticmethod
    async def aget_collection_info(collection_name: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """Async version of get_collection_info"""
        import asyncio
        return await asyncio.to_thread(MilvusVectorStore.get_collection_info, collection_name, host, port)


# Module-level cache for MilvusVectorStore instances (prevents recreating on every request)
_store_cache: Dict[str, MilvusVectorStore] = {}
_cache_lock = None  # Will be initialized as threading.Lock() when needed


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
