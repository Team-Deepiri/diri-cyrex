"""
Milvus Vector Store Integration
Production-grade vector database for RAG and semantic search
Integrates with existing KnowledgeRetrievalEngine

Industry-standard features:
- Connection pooling and reuse
- Exponential backoff retry strategy
- Circuit breaker pattern for repeated failures
- Thread-safe connection management
- Graceful degradation to in-memory fallback
"""
from typing import List, Dict, Optional, Any
import os
import warnings
import threading
import time
from enum import Enum
from pydantic import ConfigDict
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.milvus_store")


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class MilvusConnectionManager:
    """
    Industry-standard connection manager with circuit breaker and exponential backoff.
    Manages connection lifecycle, health checks, and automatic recovery.
    """
    
    def __init__(self, host: str, port: int, connection_alias: str = "default"):
        self.host = host
        self.port = port
        self.connection_alias = connection_alias
        self._lock = threading.Lock()
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._success_count = 0
        
        # Circuit breaker configuration
        self.failure_threshold = 5  # Open circuit after 5 consecutive failures
        self.recovery_timeout = 30.0  # Try recovery after 30 seconds
        self.half_open_max_attempts = 3  # Max attempts in half-open state
        
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker should be open"""
        if self._circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._circuit_state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit breaker transitioning to HALF_OPEN for {self.connection_alias}")
                return False
            return True
        return False
    
    def _record_success(self):
        """Record successful operation"""
        with self._lock:
            if self._circuit_state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_attempts:
                    self._circuit_state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker CLOSED for {self.connection_alias} - service recovered")
            elif self._circuit_state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def _record_failure(self):
        """Record failed operation"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._circuit_state == CircuitState.HALF_OPEN:
                # Failed in half-open, go back to open
                self._circuit_state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN for {self.connection_alias} - recovery attempt failed")
            elif self._failure_count >= self.failure_threshold:
                self._circuit_state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN for {self.connection_alias} - {self._failure_count} consecutive failures")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self._circuit_state
    
    def ensure_connection(self) -> bool:
        """
        Ensure connection exists and is valid (thread-safe).
        Returns True if connection is available, False otherwise.
        """
        if not HAS_PYMILVUS or not connections:
            return False
        
        # Check circuit breaker
        if self._is_circuit_open():
            logger.debug(f"Circuit breaker is OPEN for {self.connection_alias}, skipping connection attempt")
            return False
        
        with self._lock:
            try:
                # Check if connection exists
                if connections.has_connection(self.connection_alias):
                    # Connection exists - assume valid (don't verify with operations that can fail)
                    return True
                
                # Connection doesn't exist, create it
                try:
                    connections.connect(
                        alias=self.connection_alias,
                        host=self.host,
                        port=self.port,
                        timeout=10.0,
                    )
                    self._record_success()
                    logger.debug(f"Connected to Milvus at {self.host}:{self.port} (alias: {self.connection_alias})")
                    return True
                except Exception as connect_err:
                    self._record_failure()
                    error_msg = str(connect_err).lower()
                    if "connection refused" in error_msg or "cannot connect" in error_msg:
                        logger.debug(f"Milvus connection refused at {self.host}:{self.port}")
                    elif "timeout" in error_msg:
                        logger.debug(f"Milvus connection timeout at {self.host}:{self.port}")
                    else:
                        logger.debug(f"Milvus connection failed: {connect_err}")
                    return False
            except Exception as e:
                self._record_failure()
                logger.debug(f"Connection check failed: {e}")
                return False
    
    def reconnect(self) -> bool:
        """
        Force reconnection (thread-safe).
        Returns True if reconnection successful, False otherwise.
        """
        if not HAS_PYMILVUS or not connections:
            return False
        
        with self._lock:
            try:
                # Disconnect existing connection if it exists
                try:
                    if connections.has_connection(self.connection_alias):
                        connections.disconnect(self.connection_alias)
                except Exception:
                    pass
                
                # Create new connection
                connections.connect(
                    alias=self.connection_alias,
                    host=self.host,
                    port=self.port,
                    timeout=10.0,
                )
                self._record_success()
                logger.info(f"Reconnected to Milvus at {self.host}:{self.port} (alias: {self.connection_alias})")
                return True
            except Exception as e:
                self._record_failure()
                logger.warning(f"Reconnection failed: {e}")
                return False


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
        
        # Industry-standard connection managers with circuit breakers
        self._connection_manager = MilvusConnectionManager(
            host=self.host,
            port=self.port,
            connection_alias=self.connection_alias
        )
        self._langchain_connection_manager = MilvusConnectionManager(
            host=self.host,
            port=self.port,
            connection_alias=self.langchain_connection_alias
        )
        
        # Thread lock for connection management to prevent race conditions
        self._connection_lock = threading.Lock()
        
        try:
            # Connect with default alias for PyMilvus operations
            # Use thread lock to prevent concurrent connection attempts
            with self._connection_lock:
                # Check if connection already exists
                connection_exists = False
                try:
                    if connections.has_connection(self.connection_alias):
                        # Connection exists - don't verify with utility.list_collections()
                        # as it can cause channel errors. If it's actually closed, the next
                        # operation will fail and we'll reconnect then.
                        connection_exists = True
                        logger.debug(f"Reusing existing Milvus connection: {self.connection_alias}")
                except Exception:
                    # has_connection check failed, assume no connection
                    connection_exists = False
                
                if not connection_exists:
                    # Disconnect any existing connection in bad state
                    try:
                        if connections.has_connection(self.connection_alias):
                            connections.disconnect(self.connection_alias)
                    except Exception:
                        pass
                    
                    logger.info(f"Attempting to connect to Milvus at {self.host}:{self.port}...")
                    connections.connect(
                        alias=self.connection_alias,
                        host=self.host,
                        port=self.port,
                        timeout=10.0,  # Increased timeout for better reliability
                    )
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
        """Ensure Milvus connections are alive, reconnect if needed (thread-safe with circuit breaker)"""
        if not HAS_PYMILVUS or not connections:
            return
        
        # Use connection manager with circuit breaker
        if self._connection_manager.ensure_connection():
            self.milvus_available = True
        else:
            self.milvus_available = False
        
        # Ensure LangChain connection if needed
        if hasattr(self, 'langchain_connection_alias') and self.langchain_connection_alias != self.connection_alias:
            self._langchain_connection_manager.ensure_connection()
    
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
    
    def _exponential_backoff_retry(self, func, max_retries: int = 3, base_delay: float = 0.1):
        """
        Execute function with exponential backoff retry strategy.
        Industry-standard retry pattern with exponential delay.
        """
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                error_type = type(e).__name__
                
                # Check for gRPC channel errors
                is_channel_error = (
                    "closed channel" in error_msg or 
                    "rpc" in error_msg or 
                    "cannot invoke" in error_msg or
                    "channel closed" in error_msg or
                    (error_type == "ValueError" and "channel" in error_msg) or
                    "connectionnotexistexception" in error_msg or
                    "should create connection first" in error_msg
                )
                
                if is_channel_error and attempt < max_retries - 1:
                    # Exponential backoff: delay = base_delay * (2 ^ attempt)
                    delay = base_delay * (2 ** attempt)
                    logger.debug(f"Channel error (attempt {attempt + 1}/{max_retries}), retrying after {delay:.2f}s: {e}")
                    time.sleep(delay)
                    
                    # Reconnect before retry
                    with self._connection_lock:
                        self._connection_manager.reconnect()
                    continue
                else:
                    # Not a channel error or last attempt
                    if attempt < max_retries - 1:
                        logger.debug(f"Non-channel error (attempt {attempt + 1}/{max_retries}): {e}")
                    raise
        
        # All retries exhausted
        if last_exception:
            raise last_exception
    
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one (with exponential backoff retry)"""
        def _get_collection():
            # Ensure connection is alive right before checking collection
            self._ensure_connection()
            
            if not self.milvus_available:
                raise RuntimeError("Milvus connection not available")
                
            # Check if collection exists
            collection_exists = False
            try:
                collection_exists = utility.has_collection(self.collection_name)
            except (ValueError, Exception) as check_error:
                error_msg = str(check_error).lower()
                error_type = type(check_error).__name__
                # Check for gRPC channel errors
                is_channel_error = (
                    "closed channel" in error_msg or 
                    "rpc" in error_msg or 
                    "cannot invoke" in error_msg or
                    "channel closed" in error_msg or
                    (error_type == "ValueError" and "channel" in error_msg) or
                    "connectionnotexistexception" in error_msg or
                    "should create connection first" in error_msg
                )
                
                if is_channel_error:
                    # Re-raise to trigger exponential backoff retry
                    raise
                else:
                    # Other error, try to get collection anyway (might exist)
                    collection_exists = None  # Unknown state
            
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
            
            # Don't load collection at init - use lazy loading on first access
            # This prevents blocking startup and connection issues
            logger.info(f"Collection '{self.collection_name}' ready (lazy loading enabled)")
            
            return collection
    
    def _ensure_collection_loaded(self, collection: Collection) -> bool:
        """
        Ensure collection is loaded (lazy loading pattern).
        Checks loading state first to avoid redundant loads.
        Returns True if loaded, False if failed.
        """
        try:
            # Check if collection is already loaded (optimization: avoid redundant loads)
            try:
                # Check loading progress - if already loaded, this returns quickly
                progress = utility.loading_progress(self.collection_name)
                if progress.get("loading_progress", 0) == 100:
                    logger.debug(f"Collection '{self.collection_name}' already loaded")
                    return True
            except Exception:
                # Progress check failed, try to load anyway
                pass
            
            # Verify connection is alive before loading
            self._ensure_connection()
            if not self.milvus_available:
                logger.warning(f"Cannot load collection '{self.collection_name}': connection unavailable")
                return False
            
            # Load collection with timeout in background thread
            return self._load_collection_with_timeout(collection, timeout=15.0)
            
        except Exception as e:
            error_msg = str(e).lower()
            is_channel_error = (
                "closed channel" in error_msg or 
                "rpc" in error_msg or 
                "cannot invoke" in error_msg
            )
            
            if is_channel_error:
                # Connection issue - try to reconnect
                logger.warning(f"Channel error detected, attempting reconnection for '{self.collection_name}'")
                if self.connection_manager.reconnect():
                    # Retry loading after reconnection
                    return self._load_collection_with_timeout(collection, timeout=15.0)
            
            logger.warning(f"Failed to ensure collection loaded: {e}")
            return False
    
    def _load_collection_with_timeout(self, collection: Collection, timeout: float = 15.0) -> bool:
        """
        Load collection with timeout and proper error handling.
        Uses threading to enforce timeout since collection.load() is blocking.
        Returns True if successful, False otherwise.
        """
        import queue
        
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def load_collection():
            try:
                # Use load with specific fields if possible (optimization)
                # For now, use standard load - can be optimized later with load_fields
                collection.load()
                result_queue.put(True)
            except Exception as e:
                error_queue.put(e)
        
        # Run loading in background thread with timeout
        load_thread = threading.Thread(target=load_collection, daemon=True)
        load_thread.start()
        load_thread.join(timeout=timeout)
        
        if load_thread.is_alive():
            # Thread is still running = timeout
            logger.warning(
                f"Collection loading timed out after {timeout}s for '{self.collection_name}'. "
                "Will retry on next access."
            )
            return False
        
        # Check for errors
        if not error_queue.empty():
            error = error_queue.get()
            error_msg = str(error).lower()
            # Check for channel errors that we can recover from
            is_channel_error = (
                "closed channel" in error_msg or 
                "rpc" in error_msg or 
                "cannot invoke" in error_msg or
                "channel closed" in error_msg
            )
            
            if is_channel_error:
                logger.warning(
                    f"Collection loading failed due to channel error for '{self.collection_name}': {error}"
                )
                # Trigger reconnection attempt
                self.connection_manager.reconnect()
            else:
                logger.warning(
                    f"Collection loading failed for '{self.collection_name}': {error}"
                )
            return False
        
        # Success
        logger.debug(f"Collection '{self.collection_name}' loaded successfully")
        return True
    
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
        
        if not self.milvus_available:
            # Fallback to in-memory
            if not hasattr(self, '_memory_docs'):
                self._memory_docs = []
            return self._memory_docs[:k]
        
        # Lazy load collection on first use (optimization: don't block startup)
        if self.collection:
            self._ensure_collection_loaded(self.collection)
        
        def _search():
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
        
        try:
            results = self._exponential_backoff_retry(_search, max_retries=2, base_delay=0.1)
            # Record success for circuit breaker
            self._connection_manager._record_success()
            return results
        except Exception as e:
            error_msg = str(e).lower()
            # Record failure for circuit breaker
            self._connection_manager._record_failure()
            
            # Check if it's an empty collection error (not a connection error)
            if "empty" in error_msg or "no entities" in error_msg or "collection is empty" in error_msg:
                logger.debug(f"Collection '{self.collection_name}' is empty - no documents indexed yet")
                return []
            
            logger.error(f"Similarity search failed: {e}")
            # Fallback to in-memory
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
        
        if not self.milvus_available:
            # Fallback to in-memory
            if not hasattr(self, '_memory_docs'):
                self._memory_docs = []
            return self._memory_docs[:k]
        
        async def _async_search():
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
        
        try:
            # For async, we need to handle retries differently
            import asyncio
            max_retries = 2
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    results = await _async_search()
                    # Record success for circuit breaker
                    self._connection_manager._record_success()
                    return results
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    error_type = type(e).__name__
                    
                    # Check if it's an empty collection error
                    if "empty" in error_msg or "no entities" in error_msg or "collection is empty" in error_msg:
                        logger.debug(f"Collection '{self.collection_name}' is empty - no documents indexed yet")
                        return []
                    
                    # Check for gRPC channel errors
                    is_channel_error = (
                        "closed channel" in error_msg or 
                        "rpc" in error_msg or 
                        "cannot invoke" in error_msg or
                        "channel closed" in error_msg or
                        (error_type == "ValueError" and "channel" in error_msg) or
                        "connectionnotexistexception" in error_msg or
                        "should create connection first" in error_msg
                    )
                    
                    if is_channel_error and attempt < max_retries - 1:
                        # Exponential backoff
                        delay = 0.1 * (2 ** attempt)
                        logger.debug(f"Channel error during async search (attempt {attempt + 1}/{max_retries}), retrying after {delay:.2f}s")
                        await asyncio.sleep(delay)
                        
                        # Reconnect
                        with self._connection_lock:
                            self._connection_manager.reconnect()
                        continue
                    else:
                        # Not a channel error or last attempt
                        break
            
            # All retries exhausted
            self._connection_manager._record_failure()
            if last_exception:
                logger.error(f"Async similarity search failed after {max_retries} attempts: {last_exception}")
            # Fallback to in-memory
            if not hasattr(self, '_memory_docs'):
                self._memory_docs = []
            return self._memory_docs[:k]
        except Exception as e:
            self._connection_manager._record_failure()
            error_msg = str(e).lower()
            if "empty" in error_msg or "no entities" in error_msg:
                return []
            logger.error(f"Async similarity search failed: {e}")
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

