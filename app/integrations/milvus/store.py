"""
Milvus Vector Store implementation.

Provides document CRUD operations with connection management,
automatic collection creation, and in-memory fallback.
"""
import os
import json
import threading
from typing import Optional, List, Dict, Any

from langchain_core.documents import Document

from .connection import MilvusConnectionManager
from .schema import get_collection_schema, DEFAULT_EMBEDDING_DIM, DEFAULT_INDEX_PARAMS
from .exceptions import MilvusConnectionError
from ...settings import settings
from ...logging_config import get_logger

logger = get_logger("cyrex.milvus.store")

# Try to import pymilvus
try:
    from pymilvus import Collection, utility, connections
    HAS_PYMILVUS = True
except ImportError:
    HAS_PYMILVUS = False
    Collection = None
    utility = None
    connections = None
    logger.warning("pymilvus not available")


class MilvusVectorStore:
    """
    Vector store implementation for Milvus.

    Provides document CRUD operations with:
    - Connection management with circuit breaker
    - Automatic collection creation
    - In-memory fallback when Milvus unavailable
    - Both sync and async operations
    """

    def __init__(
        self,
        collection_name: str = "deepiri_knowledge",
        host: Optional[str] = None,
        port: Optional[int] = None,
        embedding_model=None,
        dimension: int = DEFAULT_EMBEDDING_DIM,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Milvus collection
            host: Milvus server host (defaults to settings.MILVUS_HOST)
            port: Milvus server port (defaults to settings.MILVUS_PORT)
            embedding_model: Optional pre-configured embedding model
            dimension: Embedding vector dimension (default: 384)
        """
        self.collection_name = collection_name
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.dimension = dimension

        # State
        self._lock = threading.RLock()
        self._memory_docs: List[Document] = []
        self._using_fallback = False
        self.collection: Optional[Collection] = None

        # Initialize embeddings
        self.embeddings = self._init_embeddings(embedding_model)
        if self.embeddings:
            # Update dimension from actual embedding
            try:
                test_embedding = self.embeddings.embed_query("test")
                self.dimension = len(test_embedding)
            except Exception:
                pass

        # Initialize connection
        self._connection = MilvusConnectionManager(
            host=self.host,
            port=self.port,
            connection_alias=f"store_{collection_name}"
        )

        # Try to connect and initialize collection
        self._initialize()

    @property
    def milvus_available(self) -> bool:
        """Check if Milvus is available"""
        return self._connection.is_connected and self.collection is not None

    @property
    def using_fallback(self) -> bool:
        """Check if using in-memory fallback"""
        return self._using_fallback

    def _init_embeddings(self, embedding_model):
        """Initialize embedding model"""
        if embedding_model:
            return embedding_model

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
            return None

    def _initialize(self):
        """Initialize connection and collection"""
        if not HAS_PYMILVUS:
            logger.warning(
                "[FALLBACK MODE] pymilvus not available, using in-memory fallback. "
                "Documents will NOT persist!"
            )
            self._using_fallback = True
            return

        if not self._connection.connect():
            logger.warning(
                f"[FALLBACK MODE] Cannot connect to Milvus at {self.host}:{self.port}, "
                f"using in-memory fallback. Documents will NOT persist!"
            )
            self._using_fallback = True
            return

        try:
            self.collection = self._get_or_create_collection()
            self._using_fallback = False
            logger.info(f"Initialized collection '{self.collection_name}' on Milvus")
        except Exception as e:
            logger.error(f"Failed to initialize collection '{self.collection_name}': {e}")
            logger.warning(
                f"[FALLBACK MODE] Using in-memory fallback. Documents will NOT persist!"
            )
            self._using_fallback = True

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        alias = self._connection.connection_alias

        if utility.has_collection(self.collection_name, using=alias):
            collection = Collection(self.collection_name, using=alias)
            collection.load()
            return collection

        # Create new collection
        schema = get_collection_schema(self.dimension)
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=alias
        )
        collection.create_index(
            field_name="embedding",
            index_params=DEFAULT_INDEX_PARAMS
        )
        collection.load()
        logger.info(f"Created collection '{self.collection_name}'")
        return collection

    # ========== Document Operations ==========

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents to the store.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of document IDs
        """
        if self._using_fallback:
            logger.warning(
                f"[FALLBACK MODE] Adding {len(documents)} documents to in-memory store. "
                f"Documents will NOT persist!"
            )
            start_idx = len(self._memory_docs)
            self._memory_docs.extend(documents)
            return [f"memory_doc_{i}" for i in range(start_idx, start_idx + len(documents))]

        if not self._connection.ensure_connection():
            raise MilvusConnectionError("Cannot connect to Milvus")

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        embeddings = [self.embeddings.embed_query(text) for text in texts]

        result = self.collection.insert([texts, metadatas, embeddings])
        self.collection.flush()

        doc_ids = [str(pk) for pk in result.primary_keys] if hasattr(result, 'primary_keys') else []
        logger.info(f"Added {len(documents)} documents to '{self.collection_name}'")
        return doc_ids

    async def aadd_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Async add documents"""
        import asyncio
        return await asyncio.to_thread(self.add_documents, documents, **kwargs)

    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query string
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of matching Document objects
        """
        if self._using_fallback:
            logger.warning("[FALLBACK MODE] Searching in-memory store")
            return self._memory_docs[:k]

        if not self._connection.ensure_connection():
            raise MilvusConnectionError("Cannot connect to Milvus")

        query_embedding = self.embeddings.embed_query(query)

        search_params = {"metric_type": "L2", "params": {"ef": 64}}
        expr = self._build_filter_expr(filters) if filters else None

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["text", "metadata"]
        )

        documents = []
        for hits in results:
            for hit in hits:
                # Get metadata - handle both dict and JSON string
                raw_metadata = hit.entity.get("metadata", {})
                if isinstance(raw_metadata, str):
                    try:
                        metadata = json.loads(raw_metadata)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to parse metadata JSON", metadata=raw_metadata[:100] if raw_metadata else None)
                        metadata = {}
                else:
                    metadata = raw_metadata or {}
                
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata=metadata
                )
                doc.metadata["id"] = str(hit.id)
                doc.metadata["score"] = hit.distance
                documents.append(doc)

        return documents

    async def asimilarity_search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        """Async similarity search"""
        import asyncio
        return await asyncio.to_thread(self.similarity_search, query, k, filters)

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dict with id, content, metadata or None if not found
        """
        if self._using_fallback:
            if doc_id.startswith("memory_doc_"):
                try:
                    idx = int(doc_id.split("_")[-1])
                    if 0 <= idx < len(self._memory_docs):
                        doc = self._memory_docs[idx]
                        return {"id": doc_id, "content": doc.page_content, "metadata": doc.metadata}
                except (ValueError, IndexError):
                    pass
            return None

        if not self._connection.ensure_connection():
            return None

        try:
            results = self.collection.query(
                expr=f"id == {doc_id}",
                output_fields=["text", "metadata"]
            )
            if results:
                return {
                    "id": doc_id,
                    "content": results[0].get("text", ""),
                    "metadata": results[0].get("metadata", {})
                }
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
        return None

    async def aget_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Async get document by ID"""
        import asyncio
        return await asyncio.to_thread(self.get_document_by_id, doc_id)

    def list_documents(self, filters: Optional[Dict] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List documents with pagination.

        Args:
            filters: Optional metadata filters
            limit: Maximum documents to return
            offset: Number of documents to skip

        Returns:
            List of document dicts
        """
        if self._using_fallback:
            docs = self._memory_docs[offset:offset + limit]
            return [
                {"id": f"memory_doc_{offset + i}", "content": doc.page_content, "metadata": doc.metadata}
                for i, doc in enumerate(docs)
            ]

        if not self._connection.ensure_connection():
            return []

        try:
            expr = self._build_filter_expr(filters) if filters else ""
            results = self.collection.query(
                expr=expr or "id > 0",
                output_fields=["text", "metadata"],
                limit=limit,
                offset=offset
            )
            return [
                {"id": str(r.get("id")), "content": r.get("text", ""), "metadata": r.get("metadata", {})}
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    async def alist_documents(self, filters: Optional[Dict] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Async list documents"""
        import asyncio
        return await asyncio.to_thread(self.list_documents, filters, limit, offset)

    def delete(self, ids: List[str], **kwargs) -> Dict[str, Any]:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            Dict with deletion result
        """
        if self._using_fallback:
            deleted = 0
            for doc_id in ids:
                if doc_id.startswith("memory_doc_"):
                    try:
                        idx = int(doc_id.split("_")[-1])
                        if 0 <= idx < len(self._memory_docs):
                            self._memory_docs[idx] = None
                            deleted += 1
                    except (ValueError, IndexError):
                        pass
            self._memory_docs = [d for d in self._memory_docs if d is not None]
            return {"deleted": deleted}

        if not self._connection.ensure_connection():
            raise MilvusConnectionError("Cannot connect to Milvus")

        expr = f"id in [{','.join(ids)}]"
        self.collection.delete(expr)
        self.collection.flush()
        return {"deleted": len(ids)}

    async def adelete(self, ids: List[str], **kwargs) -> Dict[str, Any]:
        """Async delete"""
        import asyncio
        return await asyncio.to_thread(self.delete, ids, **kwargs)

    def update_document(self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Update a document (delete + re-add).

        Args:
            doc_id: Document ID to update
            content: New content (optional)
            metadata: New/merged metadata (optional)

        Returns:
            Dict with update result including new ID
        """
        existing = self.get_document_by_id(doc_id)
        if not existing:
            return {"updated": False, "error": "Document not found"}

        new_content = content if content is not None else existing["content"]
        new_metadata = {**existing.get("metadata", {}), **(metadata or {})}

        self.delete([doc_id])

        doc = Document(page_content=new_content, metadata=new_metadata)
        new_ids = self.add_documents([doc])

        return {"updated": True, "old_id": doc_id, "new_id": new_ids[0] if new_ids else None}

    async def aupdate_document(self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Async update"""
        import asyncio
        return await asyncio.to_thread(self.update_document, doc_id, content, metadata)

    def count_documents(self, filters: Optional[Dict] = None) -> int:
        """
        Count documents in the collection.

        Args:
            filters: Optional metadata filters (not implemented for Milvus)

        Returns:
            Number of documents
        """
        if self._using_fallback:
            return len(self._memory_docs)

        if not self._connection.ensure_connection():
            return 0

        # Query the collection directly to get accurate count
        # This is more reliable than collection.num_entities which can be stale
        try:
            # Query with a high limit to count actual entities
            results = self.collection.query(
                expr="id >= 0",  # Query all entities
                output_fields=["id"],
                limit=16384  # Milvus max limit
            )
            count = len(results)
            logger.debug(f"Counted {count} entities via query")
            return count
        except Exception as e:
            logger.warning(f"Failed to query count, falling back to num_entities: {e}")
            # Fallback to num_entities if query fails
            try:
                self.collection.flush()
                self.collection.load()
            except Exception as load_error:
                logger.debug(f"Load during count fallback failed (non-critical): {load_error}")
            return self.collection.num_entities

    async def acount_documents(self, filters: Optional[Dict] = None) -> int:
        """Async count"""
        import asyncio
        return await asyncio.to_thread(self.count_documents, filters)

    def delete_by_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete documents matching filters.

        Args:
            filters: Metadata filters for matching documents

        Returns:
            Dict with deletion result
        """
        if self._using_fallback:
            original_count = len(self._memory_docs)
            self._memory_docs = [
                d for d in self._memory_docs
                if not all(d.metadata.get(k) == v for k, v in filters.items())
            ]
            return {"deleted": original_count - len(self._memory_docs)}

        if not self._connection.ensure_connection():
            raise MilvusConnectionError("Cannot connect to Milvus")

        # For JSON fields, Milvus delete doesn't support the same filter syntax as query
        # So we need to query first to get IDs, then delete by IDs
        try:
            # Build filter expression for query (this works for queries)
            expr = self._build_filter_expr(filters) if filters else ""
            
            if not expr:
                logger.warning("No filter expression provided for delete")
                return {"deleted": 0}
            
            # Query to get all matching document IDs
            results = self.collection.query(
                expr=expr,
                output_fields=["id"],
                limit=16384  # Milvus max limit
            )
            
            if not results:
                logger.info("No documents found matching filter")
                return {"deleted": 0}
            
            # Extract IDs
            ids_to_delete = [r.get("id") for r in results if r.get("id") is not None]
            
            if not ids_to_delete:
                logger.warning("No valid IDs found to delete")
                return {"deleted": 0}
            
            # Delete by IDs (this works reliably)
            # Milvus delete expression: "id in [1, 2, 3, ...]"
            ids_str = ", ".join(str(id_val) for id_val in ids_to_delete)
            delete_expr = f"id in [{ids_str}]"
            
            self.collection.delete(delete_expr)
            self.collection.flush()
            
            logger.info(f"Deleted {len(ids_to_delete)} documents by IDs")
            return {"deleted": len(ids_to_delete)}
                
        except Exception as e:
            logger.error(f"Failed to delete by filter: {e}", exc_info=True)
            raise

    async def adelete_by_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Async delete by filter"""
        import asyncio
        return await asyncio.to_thread(self.delete_by_filter, filters)

    # ========== Batch Operations ==========

    def batch_add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """
        Batch add documents.

        Args:
            documents: List of dicts with 'content' and 'metadata' keys
            batch_size: Number of documents per batch

        Returns:
            Dict with added/failed counts
        """
        added = 0
        failed = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            docs = [
                Document(page_content=d.get("content", ""), metadata=d.get("metadata", {}))
                for d in batch
            ]
            try:
                self.add_documents(docs)
                added += len(docs)
            except Exception as e:
                logger.error(f"Batch add failed: {e}")
                failed += len(docs)

        return {"added": added, "failed": failed}

    async def abatch_add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """Async batch add"""
        import asyncio
        return await asyncio.to_thread(self.batch_add_documents, documents, batch_size)

    # ========== Stats & Health ==========

    def stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "collection_name": self.collection_name,
            "using_fallback": self._using_fallback,
            "milvus_available": self.milvus_available,
            "num_entities": self.count_documents(),
            "connection_state": self._connection.circuit_state.value,
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "healthy": self.milvus_available or self._using_fallback,
            "milvus_connected": self._connection.is_connected,
            "collection_loaded": self.collection is not None,
            "using_fallback": self._using_fallback,
            "embeddings_available": self.embeddings is not None,
        }

    # ========== Static Methods ==========

    @staticmethod
    def list_all_collections(host: Optional[str] = None, port: Optional[int] = None) -> List[str]:
        """
        List all collections in Milvus.

        Args:
            host: Milvus host (defaults to settings)
            port: Milvus port (defaults to settings)

        Returns:
            List of collection names
        """
        if not HAS_PYMILVUS:
            logger.warning("pymilvus not available, cannot list collections")
            return []

        h = host or settings.MILVUS_HOST
        p = port or settings.MILVUS_PORT

        try:
            alias = "list_collections_temp"
            if connections.has_connection(alias):
                connections.disconnect(alias)
            connections.connect(alias=alias, host=h, port=p, timeout=10.0)
            collection_list = utility.list_collections(using=alias)
            connections.disconnect(alias)
            return collection_list
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    @staticmethod
    async def alist_all_collections(host: Optional[str] = None, port: Optional[int] = None) -> List[str]:
        """Async list all collections"""
        import asyncio
        return await asyncio.to_thread(MilvusVectorStore.list_all_collections, host, port)

    @staticmethod
    def get_collection_info(collection_name: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """
        Get collection info.

        Args:
            collection_name: Name of the collection
            host: Milvus host (defaults to settings)
            port: Milvus port (defaults to settings)

        Returns:
            Dict with collection info or error
        """
        if not HAS_PYMILVUS:
            return {"error": "pymilvus not available"}

        h = host or settings.MILVUS_HOST
        p = port or settings.MILVUS_PORT

        try:
            alias = f"info_{collection_name}_temp"
            if connections.has_connection(alias):
                connections.disconnect(alias)
            connections.connect(alias=alias, host=h, port=p, timeout=10.0)

            if not utility.has_collection(collection_name, using=alias):
                connections.disconnect(alias)
                return {"error": f"Collection '{collection_name}' not found"}

            collection = Collection(collection_name, using=alias)
            info = {
                "name": collection_name,
                "num_entities": collection.num_entities,
                "schema": str(collection.schema),
            }
            connections.disconnect(alias)
            return info
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    @staticmethod
    async def aget_collection_info(collection_name: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """Async get collection info"""
        import asyncio
        return await asyncio.to_thread(MilvusVectorStore.get_collection_info, collection_name, host, port)

    # ========== Helpers ==========

    def _build_filter_expr(self, filters: Dict[str, Any]) -> str:
        """Build Milvus filter expression from dict"""
        if not filters:
            return ""

        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f'metadata["{key}"] == "{value}"')
            else:
                conditions.append(f'metadata["{key}"] == {value}')

        return " && ".join(conditions)


# ========== Factory Function ==========

_store_cache: Dict[str, MilvusVectorStore] = {}
_cache_lock = threading.Lock()


def get_milvus_store(
    collection_name: str = "deepiri_knowledge",
    embedding_model=None,
    force_new: bool = False,
    **kwargs
) -> MilvusVectorStore:
    """
    Get or create a MilvusVectorStore instance (with caching).

    Args:
        collection_name: Name of the collection
        embedding_model: Optional embedding model
        force_new: Force creation of new instance
        **kwargs: Additional arguments for MilvusVectorStore

    Returns:
        MilvusVectorStore instance
    """
    with _cache_lock:
        if force_new or collection_name not in _store_cache:
            _store_cache[collection_name] = MilvusVectorStore(
                collection_name=collection_name,
                embedding_model=embedding_model,
                **kwargs
            )
        return _store_cache[collection_name]
