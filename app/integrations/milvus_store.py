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

# Suppress pymilvus gRPC channel error logging - we handle these errors gracefully with reconnection
# These errors are expected when connections are being reestablished
import logging
_pymilvus_logger = logging.getLogger("pymilvus")
# Set to WARNING level to suppress ERROR logs for channel errors (we handle them)
_pymilvus_logger.setLevel(logging.WARNING)

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
            
            try:
                logger.info(f"Initializing {model_name} with cached RobustEmbeddings wrapper...")
                from .embeddings_wrapper import get_robust_embeddings
                self.embeddings = get_robust_embeddings(model_name)
                # Don't test embedding here to avoid reloading - trust the cache
                logger.info(f"Successfully initialized embedding model: {model_name} (cached RobustEmbeddings)")
                embedding_initialized = True
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
            # Check if connection already exists and is valid before creating new one
            connection_exists = False
            try:
                if connections.has_connection(self.connection_alias):
                    # Connection exists, verify it's still valid by trying a simple operation
                    try:
                        # Try to list collections to verify connection is alive
                        utility.list_collections()
                        connection_exists = True
                        logger.debug(f"Reusing existing Milvus connection: {self.connection_alias}")
                    except (ValueError, Exception) as verify_error:
                        error_msg = str(verify_error).lower()
                        error_type = type(verify_error).__name__
                        # Check for gRPC channel errors
                        is_channel_error = (
                            "closed channel" in error_msg or 
                            "rpc" in error_msg or 
                            "cannot invoke" in error_msg or
                            "channel closed" in error_msg or
                            (error_type == "ValueError" and "channel" in error_msg)
                        )
                        
                        if is_channel_error:
                            # Connection exists but channel is closed, disconnect and reconnect
                            logger.debug(f"Existing connection has closed channel, reconnecting...")
                            try:
                                connections.disconnect(self.connection_alias)
                            except Exception:
                                pass
                            connection_exists = False
                        else:
                            # Other error, might still be valid, but reconnect to be safe
                            logger.debug(f"Connection verification failed: {verify_error}, reconnecting...")
                            try:
                                connections.disconnect(self.connection_alias)
                            except Exception:
                                pass
                            connection_exists = False
            except Exception:
                # has_connection check failed, assume no connection
                connection_exists = False
            
            if not connection_exists:
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
                        # Suppress common compatibility errors - fallback works fine
                        if "unexpected keyword argument" in error_msg or "using" in error_msg:
                            logger.debug(f"LangChain Milvus wrapper not compatible with this version (common): {last_error}")
                        elif "localhost" in error_msg or "closed channel" in error_msg or "connection" in error_msg:
                            logger.debug(f"LangChain Milvus wrapper failed to connect (may be using wrong host or channel closed): {last_error}")
                        else:
                            logger.debug(f"LangChain Milvus wrapper initialization failed: {last_error}")
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
            # Suppress verbose errors for known compatibility issues
            error_msg = str(e).lower()
            if "unexpected keyword argument" in error_msg or "using" in error_msg:
                logger.debug(f"LangChain Milvus wrapper not compatible, using fallback: {e}")
            else:
                logger.debug(f"Failed to create LangChain Milvus wrapper, using fallback: {e}")
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
            # Check default connection - verify it's actually alive, not just registered
            connection_needs_reconnect = True
            try:
                if connections.has_connection(self.connection_alias):
                    # Connection exists, verify it's still valid
                    try:
                        # Try a simple operation to verify connection is alive
                        utility.list_collections()
                        connection_needs_reconnect = False
                    except (ValueError, Exception) as verify_error:
                        error_msg = str(verify_error).lower()
                        error_type = type(verify_error).__name__
                        # Check for gRPC channel errors
                        is_channel_error = (
                            "closed channel" in error_msg or 
                            "rpc" in error_msg or 
                            "cannot invoke" in error_msg or
                            "channel closed" in error_msg or
                            (error_type == "ValueError" and "channel" in error_msg)
                        )
                        
                        if is_channel_error:
                            # Connection channel is closed, need to reconnect
                            logger.debug(f"Connection channel closed, will reconnect: {verify_error}")
                            try:
                                connections.disconnect(self.connection_alias)
                            except Exception:
                                pass
                            connection_needs_reconnect = True
                        else:
                            # Other error, might be transient, but reconnect to be safe
                            logger.debug(f"Connection verification failed, will reconnect: {verify_error}")
                            try:
                                connections.disconnect(self.connection_alias)
                            except Exception:
                                pass
                            connection_needs_reconnect = True
            except Exception:
                # has_connection check failed, assume no connection
                connection_needs_reconnect = True
            
            if connection_needs_reconnect:
                logger.info(f"Reconnecting to Milvus at {self.host}:{self.port}")
                connections.connect(
                    alias=self.connection_alias,
                    host=self.host,
                    port=self.port,
                    timeout=10.0,
                )
                # Verify the connection is actually working after reconnect
                try:
                    utility.list_collections()
                    logger.debug("Milvus connection verified after reconnect")
                except Exception as verify_err:
                    error_msg = str(verify_err).lower()
                    is_channel_error = (
                        "closed channel" in error_msg or 
                        "rpc" in error_msg or 
                        "cannot invoke" in error_msg or
                        "channel closed" in error_msg
                    )
                    if is_channel_error:
                        logger.warning(f"Connection verification failed after reconnect: {verify_err}. Connection may be unstable.")
                    else:
                        # Non-channel error might be acceptable (e.g., no collections yet)
                        logger.debug(f"Connection verification returned non-critical error: {verify_err}")
            
            # Check LangChain connection if it exists
            if hasattr(self, 'langchain_connection_alias') and self.langchain_connection_alias != self.connection_alias:
                langchain_needs_reconnect = True
                try:
                    if connections.has_connection(self.langchain_connection_alias):
                        # Verify it's alive
                        try:
                            utility.list_collections()
                            langchain_needs_reconnect = False
                        except Exception:
                            # Channel might be closed, reconnect
                            try:
                                connections.disconnect(self.langchain_connection_alias)
                            except Exception:
                                pass
                            langchain_needs_reconnect = True
                except Exception:
                    langchain_needs_reconnect = True
                
                if langchain_needs_reconnect:
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
        
        # Check embeddings - just verify it's initialized, don't actually generate embeddings
        # (generating embeddings in health check causes model to reload repeatedly)
        if self.embeddings:
            try:
                # Just check if the embeddings object has the required methods
                if hasattr(self.embeddings, 'embed_query') and hasattr(self.embeddings, 'embed_documents'):
                    # If we have a cached dimension, use it
                    if hasattr(self, 'dimension') and self.dimension:
                        health_status["embeddings"]["dimension"] = self.dimension
                    health_status["embeddings"]["status"] = "initialized"
                else:
                    health_status["errors"].append("Embeddings object missing required methods")
                    health_status["healthy"] = False
            except Exception as e:
                health_status["errors"].append(f"Embedding check failed: {e}")
                health_status["healthy"] = False
        
        # Overall health is True only if all components are healthy
        if health_status["errors"]:
            health_status["healthy"] = False
        
        return health_status
    
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Ensure connection is alive right before checking collection
                # This prevents closed channel errors that can occur between operations
                self._ensure_connection()
                
                # Check if collection exists - this can fail if RPC channel is closed
                collection_exists = False
                try:
                    collection_exists = utility.has_collection(self.collection_name)
                except (ValueError, Exception) as check_error:
                    error_msg = str(check_error).lower()
                    error_type = type(check_error).__name__
                    # Check for gRPC channel errors - these can be ValueError or other exceptions
                    is_channel_error = (
                        "closed channel" in error_msg or 
                        "rpc" in error_msg or 
                        "cannot invoke" in error_msg or
                        "channel closed" in error_msg or
                        (error_type == "ValueError" and "channel" in error_msg)
                    )
                    
                    if is_channel_error:
                        # Connection channel was closed, reconnect and retry
                        logger.debug(f"gRPC channel error during has_collection check (attempt {attempt + 1}/{max_retries}): {check_error}")
                        if attempt < max_retries - 1:
                            # Reconnect and retry
                            try:
                                if connections.has_connection(self.connection_alias):
                                    connections.disconnect(self.connection_alias)
                            except Exception:
                                pass
                            
                            connections.connect(
                                alias=self.connection_alias,
                                host=self.host,
                                port=self.port,
                                timeout=10.0,
                            )
                            logger.debug(f"Reconnected to Milvus, retrying collection check...")
                            continue  # Retry the has_collection check
                        else:
                            # Last attempt failed, try to get collection anyway (might exist)
                            logger.debug(f"Failed to check collection existence after {max_retries} attempts, attempting to get collection directly")
                            collection_exists = None  # Unknown state, try to get it
                    else:
                        # Other error, re-raise
                        raise
                
                # If we know the collection exists, use it
                if collection_exists is True:
                    collection = Collection(self.collection_name)
                    logger.info(f"Using existing collection: {self.collection_name}")
                elif collection_exists is False:
                    # Collection doesn't exist, create it
                    collection = self._create_collection()
                    logger.info(f"Created new collection: {self.collection_name}")
                else:
                    # Unknown state (collection_exists is None), try to get it
                    try:
                        collection = Collection(self.collection_name)
                        logger.info(f"Using existing collection: {self.collection_name} (existence check failed, but collection accessible)")
                    except Exception as get_error:
                        # Collection doesn't exist, create it
                        logger.info(f"Collection not accessible, creating new one: {get_error}")
                        collection = self._create_collection()
                        logger.info(f"Created new collection: {self.collection_name}")
                
                # Load collection into memory
                try:
                    collection.load()
                except (ValueError, Exception) as load_error:
                    error_msg = str(load_error).lower()
                    error_type = type(load_error).__name__
                    # Check for gRPC channel errors
                    is_channel_error = (
                        "closed channel" in error_msg or 
                        "rpc" in error_msg or 
                        "cannot invoke" in error_msg or
                        "channel closed" in error_msg or
                        (error_type == "ValueError" and "channel" in error_msg)
                    )
                    
                    if is_channel_error:
                        # Connection closed during load, reconnect and retry
                        if attempt < max_retries - 1:
                            logger.debug(f"gRPC channel error during collection.load() (attempt {attempt + 1}/{max_retries}), reconnecting...")
                            try:
                                if connections.has_connection(self.connection_alias):
                                    connections.disconnect(self.connection_alias)
                            except Exception:
                                pass
                            
                            connections.connect(
                                alias=self.connection_alias,
                                host=self.host,
                                port=self.port,
                                timeout=10.0,
                            )
                            # Re-get the collection after reconnection
                            collection = Collection(self.collection_name)
                            collection.load()
                        else:
                            raise
                    else:
                        raise
                
                return collection
            
            except (ValueError, Exception) as e:
                if attempt < max_retries - 1:
                    error_msg = str(e).lower()
                    error_type = type(e).__name__
                    # Check for gRPC channel errors
                    is_channel_error = (
                        "closed channel" in error_msg or 
                        "rpc" in error_msg or 
                        "cannot invoke" in error_msg or
                        "channel closed" in error_msg or
                        (error_type == "ValueError" and "channel" in error_msg)
                    )
                    
                    if is_channel_error:
                        logger.debug(f"gRPC channel error during collection operation (attempt {attempt + 1}/{max_retries}): {e}")
                        # Reconnect and retry
                        try:
                            if connections.has_connection(self.connection_alias):
                                connections.disconnect(self.connection_alias)
                        except Exception:
                            pass
                        
                        connections.connect(
                            alias=self.connection_alias,
                            host=self.host,
                            port=self.port,
                            timeout=10.0,
                        )
                        logger.info(f"Reconnected to Milvus, retrying collection operation...")
                        continue
                
                # Last attempt or non-recoverable error
                logger.error(f"Failed to get/create collection after {attempt + 1} attempts: {e}", exc_info=True)
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


# Global cache for MilvusVectorStore instances (keyed by collection_name)
_milvus_store_cache = {}

def get_milvus_store(
    collection_name: str,
    embedding_model: Optional[Embeddings] = None,
    **kwargs
) -> MilvusVectorStore:
    """
    Factory function to get Milvus vector store (cached singleton per collection)
    
    Args:
        collection_name: Name of the collection
        embedding_model: Optional custom embedding model
        **kwargs: Additional configuration
    
    Returns:
        Configured MilvusVectorStore instance (cached)
    """
    global _milvus_store_cache
    
    # Use collection_name as cache key
    if collection_name not in _milvus_store_cache:
        logger.info(f"Creating new MilvusVectorStore instance for collection: {collection_name}")
        _milvus_store_cache[collection_name] = MilvusVectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            **kwargs
        )
    else:
        logger.debug(f"Returning cached MilvusVectorStore instance for collection: {collection_name}")
    
    return _milvus_store_cache[collection_name]

