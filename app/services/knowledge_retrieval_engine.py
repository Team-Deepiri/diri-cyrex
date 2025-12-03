"""
Knowledge Retrieval Engine
LangChain-based RAG orchestration for knowledge retrieval
Integrates with vector stores and LLMs for context-aware generation
"""
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma, Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    HAS_COMPRESSION = True
except ImportError:
    HAS_COMPRESSION = False
    ContextualCompressionRetriever = None
    LLMChainExtractor = None
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import OpenAIEmbeddings
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    ChatOpenAI = None
    OpenAIEmbeddings = None
import os
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.knowledge_retrieval_engine")


class KnowledgeRetrievalEngine:
    """
    RAG orchestration engine using LangChain for knowledge retrieval.
    Manages multiple knowledge bases and provides unified retrieval interface.
    """
    
    def __init__(
        self,
        vector_store_type: str = "chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_compression: bool = True
    ):
        self.vector_store_type = vector_store_type
        self.use_compression = use_compression
        
        # Initialize embeddings
        if embedding_model.startswith("sentence-transformers"):
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        else:
            if not settings.OPENAI_API_KEY:
                logger.warning("OpenAI API key not set, using HuggingFace embeddings")
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            else:
                self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        
        # Initialize knowledge bases - handle failures gracefully
        self.knowledge_bases = {}
        for kb_name in ["user_patterns", "project_context", "ability_templates", 
                        "rules_knowledge", "historical_abilities"]:
            try:
                kb = self._init_kb(kb_name)
                if kb is not None:
                    self.knowledge_bases[kb_name] = kb
                else:
                    logger.warning(f"Knowledge base '{kb_name}' not initialized (connection failed)")
            except Exception as e:
                logger.warning(f"Failed to initialize knowledge base '{kb_name}': {e}")
                # Continue with other knowledge bases
        
        # Initialize compressor if enabled - prefer OpenAI, fallback to local LLM
        self.compressor = None
        if use_compression:
            llm = None
            
            # Try OpenAI first
            if HAS_OPENAI and settings.OPENAI_API_KEY:
                try:
                    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=settings.OPENAI_API_KEY)
                    logger.info("Using OpenAI for document compression")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI for compression: {e}, trying local LLM")
            
            # Fallback to local LLM
            if not llm:
                try:
                    from ..integrations.local_llm import get_local_llm
                    local_llm = get_local_llm()
                    
                    if local_llm and local_llm.is_available():
                        llm = local_llm.get_langchain_llm()
                        logger.info("Using local LLM for document compression")
                except Exception as e:
                    logger.warning(f"Failed to initialize local LLM for compression: {e}")
            
            # Create compressor if we have an LLM
            if llm:
                try:
                    self.compressor = LLMChainExtractor.from_llm(llm)
                except Exception as e:
                    logger.warning(f"Failed to create compressor: {e}. Compression disabled.")
            else:
                logger.warning("Compression disabled: No LLM available")
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        logger.info(f"Knowledge retrieval engine initialized with {len(self.knowledge_bases)} knowledge bases")
    
    def _init_kb(self, collection_name: str):
        """Initialize a knowledge base"""
        if self.vector_store_type == "chroma":
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            os.makedirs(f"{persist_dir}/{collection_name}", exist_ok=True)
            return Chroma(
                persist_directory=f"{persist_dir}/{collection_name}",
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
        elif self.vector_store_type == "milvus":
            milvus_host = os.getenv("MILVUS_HOST", "localhost")
            milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
            
            # Use thread-based timeout since Milvus connection can hang
            import threading
            import queue
            
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def init_milvus():
                try:
                    kb = Milvus(
                        embedding_function=self.embeddings,
                        connection_args={
                            "host": milvus_host,
                            "port": milvus_port,
                            "timeout": 5.0
                        },
                        collection_name=collection_name
                    )
                    result_queue.put(kb)
                except Exception as e:
                    error_queue.put(e)
            
            # Run initialization in thread with timeout
            thread = threading.Thread(target=init_milvus, daemon=True)
            thread.start()
            thread.join(timeout=5.0)  # 5 second timeout
            
            if thread.is_alive():
                # Thread is still running = timeout
                logger.warning(f"Milvus initialization for '{collection_name}' timed out after 5 seconds")
                return None
            
            # Check for errors
            if not error_queue.empty():
                e = error_queue.get()
                logger.warning(f"Failed to initialize Milvus knowledge base '{collection_name}': {e}")
                return None
            
            # Get result
            if not result_queue.empty():
                return result_queue.get()
            
            # No result and no error = something went wrong
            logger.warning(f"Failed to initialize Milvus knowledge base '{collection_name}': Unknown error")
            return None
        else:
            raise ValueError(f"Unknown vector store type: {self.vector_store_type}")
    
    def add_document(
        self,
        content: str,
        metadata: Dict,
        knowledge_base: str = "project_context"
    ):
        """Add document to knowledge base"""
        if knowledge_base not in self.knowledge_bases:
            raise ValueError(f"Unknown knowledge base: {knowledge_base}")
        
        try:
            # Split text
            docs = self.text_splitter.create_documents([content], [metadata])
            
            # Add to vector store
            kb = self.knowledge_bases[knowledge_base]
            kb.add_documents(docs)
            kb.persist()
            
            logger.info(f"Added document to {knowledge_base}: {len(docs)} chunks")
        
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise
    
    def retrieve(
        self,
        query: str,
        knowledge_bases: Optional[List[str]] = None,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents from knowledge bases
        
        Args:
            query: Search query
            knowledge_bases: List of KBs to search (None = all)
            top_k: Number of results per KB
            filters: Optional metadata filters
        
        Returns:
            List of relevant documents
        """
        if knowledge_bases is None:
            knowledge_bases = list(self.knowledge_bases.keys())
        
        all_results = []
        
        for kb_name in knowledge_bases:
            if kb_name not in self.knowledge_bases:
                continue
            
            kb = self.knowledge_bases[kb_name]
            retriever = kb.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": top_k,
                    "filter": filters
                }
            )
            
            # Apply compression if enabled
            if self.compressor:
                retriever = ContextualCompressionRetriever(
                    base_compressor=self.compressor,
                    base_retriever=retriever
                )
            
            try:
                docs = retriever.get_relevant_documents(query)
                # Add KB name to metadata
                for doc in docs:
                    doc.metadata["knowledge_base"] = kb_name
                all_results.extend(docs)
            except Exception as e:
                logger.error(f"Retrieval failed for {kb_name}: {str(e)}")
        
        # Deduplicate and sort by relevance
        seen = set()
        unique_results = []
        for doc in all_results:
            doc_id = f"{doc.page_content[:50]}_{doc.metadata.get('knowledge_base', 'unknown')}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
        
        return unique_results[:top_k * len(knowledge_bases)]
    
    def retrieve_formatted(
        self,
        query: str,
        knowledge_bases: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """Retrieve and format documents as context string"""
        docs = self.retrieve(query, knowledge_bases, top_k)
        
        formatted = []
        for i, doc in enumerate(docs):
            kb = doc.metadata.get("knowledge_base", "unknown")
            formatted.append(f"Document {i+1} (from {kb}):\n{doc.page_content}\n")
        
        return "\n".join(formatted)
    
    def add_user_pattern(
        self,
        user_id: str,
        pattern: str,
        metadata: Dict
    ):
        """Add user behavior pattern to knowledge base"""
        content = f"User {user_id} pattern: {pattern}"
        metadata.update({
            "user_id": user_id,
            "type": "user_pattern",
            "pattern": pattern
        })
        self.add_document(content, metadata, "user_patterns")
    
    def add_ability_template(
        self,
        ability_name: str,
        description: str,
        steps: List[str],
        metadata: Dict
    ):
        """Add ability template to knowledge base"""
        content = f"""
Ability: {ability_name}
Description: {description}
Steps:
{chr(10).join(f"- {step}" for step in steps)}
"""
        metadata.update({
            "ability_name": ability_name,
            "type": "ability_template"
        })
        self.add_document(content, metadata, "ability_templates")
    
    def add_project_context(
        self,
        project_id: str,
        context: str,
        metadata: Dict
    ):
        """Add project context to knowledge base"""
        metadata.update({
            "project_id": project_id,
            "type": "project_context"
        })
        self.add_document(context, metadata, "project_context")


# Singleton instance
_knowledge_retrieval_engine = None

def get_knowledge_retrieval_engine() -> KnowledgeRetrievalEngine:
    """Get singleton KnowledgeRetrievalEngine instance"""
    global _knowledge_retrieval_engine
    if _knowledge_retrieval_engine is None:
        _knowledge_retrieval_engine = KnowledgeRetrievalEngine()
    return _knowledge_retrieval_engine

