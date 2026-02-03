"""
Document Indexing Service
Handles document ingestion, processing, chunking, and indexing into Milvus
for the Language Intelligence Platform
"""

from typing import List, Dict, Optional, Any
from enum import Enum
from pathlib import Path

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local imports
from ..logging_config import get_logger
from ..integrations.milvus_store import get_milvus_store

logger = get_logger("cyrex.document_indexing")

# Document type enum 
class B2BDocumentType(str, Enum):
    """Document types for Language Intelligence Platform"""
    LEGAL_DOCUMENT = "legal_document"  # General legal document
    CONTRACT = "contract"               # Contract documents
    LEASE = "lease"                     # Lease documents
    REGULATION = "regulation"           # Regulatory documents
    AMENDMENT = "amendment"             # Amendments
    COMPLIANCE_REPORT = "compliance_report"  # Compliance reports
    TEMPLATE = "template"               # Templates (for drift detection)

# Document format enum - matches MD file
class DocumentFormat(str, Enum):
    """Supported file formats"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"

class DocumentIndexingService:
    """
    Service for indexing documents into Milvus for RAG retrieval.
    Handles file parsing, chunking, embedding, and indexing.
    Supports the Language Intelligence Platform use cases.
    """
    
    def __init__(
        self,
        collection_name: str, # Required: Name of the Milvus collection
        chunk_size: int = 1500, # Optional: Size of text chunks (default 1500 for legal documents)
        chunk_overlap: int = 300, # Optional: Overlap between chunks (default 300 for clause preservation)
        chunking_strategy: str = "paragraph", # Optional: Chunking strategy - "paragraph" (preserves clauses), "sentence", or "character"
        use_enhanced_rag: bool = True # Optional: Whether to use enhanced RAG features
    ):
        """
        Initialize DocumentIndexingService.
        
        Args:
            collection_name: Name of the Milvus collection
            chunk_size: Size of text chunks (default 1500 for legal documents)
            chunk_overlap: Overlap between chunks (default 300 for clause preservation)
            chunking_strategy: Chunking strategy - "paragraph" (preserves clauses), 
                             "sentence", or "character"
            use_enhanced_rag: Whether to use enhanced RAG features
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.use_enhanced_rag = use_enhanced_rag
        
        # Initialize vector store (connects to Milvus)
        try:
            self.vector_store = get_milvus_store(
                collection_name=collection_name
            )
            logger.info(
                "Vector store initialized",
                collection=collection_name
            )
        except Exception as e:
            logger.error(
                "Failed to initialize vector store",
                collection=collection_name,
                error=str(e)
            )
            raise
        
        # Initialize text splitter based on strategy
        # Paragraph strategy preserves clause boundaries (important for legal docs)
        if chunking_strategy == "paragraph":
            separators = ["\n\n", "\n", " ", ""]
        elif chunking_strategy == "sentence":
            separators = ["\n", ". ", " ", ""]
        else:  # character
            separators = [" ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
        
        logger.info(
            "DocumentIndexingService initialized",
            collection=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy
        )

    
    def _parse_file(self, file_path: str) -> str:
        """
        Parse file and extract text content.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Extracted text content
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path_obj.suffix.lower().lstrip('.')
        logger.info("Parsing file", file_path=file_path, format=file_ext)
        
        try:
            if file_ext == 'pdf':
                return self._parse_pdf(file_path)
            elif file_ext in ['docx', 'doc']:
                return self._parse_docx(file_path)
            elif file_ext in ['txt', 'md']:
                return self._parse_txt(file_path)
            elif file_ext == 'csv':
                return self._parse_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error("Failed to parse file", file_path=file_path, error=str(e))
            raise

    def _parse_csv(self, file_path: str) -> str:
        """Parse CSV file"""
        try:
            import csv
            text_lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    text_lines.append(" | ".join(row))  # Join columns with pipe separator
            text = "\n".join(text_lines)
            logger.info("Extracted text from CSV", file_path=file_path, length=len(text))
            return text
        except Exception as e:
            logger.error("Failed to parse CSV", file_path=file_path, error=str(e))
            raise
    
    def _parse_pdf(self, file_path: str) -> str:
        """Parse PDF file"""
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages])
            logger.info("Extracted text from PDF", file_path=file_path, length=len(text))
            return text
        except Exception as e:
            logger.error("Failed to parse PDF", file_path=file_path, error=str(e))
            raise
    
    def _parse_docx(self, file_path: str) -> str:
        """Parse DOCX file"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info("Extracted text from DOCX", file_path=file_path, length=len(text))
            return text
        except Exception as e:
            logger.error("Failed to parse DOCX", file_path=file_path, error=str(e))
            raise
    
    def _parse_txt(self, file_path: str) -> str:
        """Parse TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info("Extracted text from TXT", file_path=file_path, length=len(text))
            return text
        except Exception as e:
            logger.error("Failed to parse TXT", file_path=file_path, error=str(e))
            raise
    
    def _chunk_document(
        self,
        text: str,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Split document text into smaller chunks for vector indexing.
        
        Takes the full document text and splits it into chunks of approximately
        chunk_size characters, preserving paragraph boundaries when possible.
        Each chunk is wrapped in a LangChain Document object with metadata.
        
        Args:
            text: The full document text to chunk
            document_id: Unique identifier for the document
            metadata: Document metadata to preserve in each chunk (version, 
                    document_type, etc.)
        
        Returns:
            List of Document objects, each containing:
            - page_content: A chunk of text (~chunk_size characters)
            - metadata: Original metadata plus chunk_index and total_chunks
        """
        documents = []
        chunks = self.text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata, # Business metadata from MD file (version, etc.)
                 # Technical metadata (not in MD file, but useful)
                "chunk_index": i,
                "document_id": document_id,
                "total_chunks": len(chunks)
            }
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        return documents

    async def index_file(
        self,
        file_path: str,
        document_id: str,
        title: str,
        doc_type: B2BDocumentType,
        industry: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index a file into Milvus for RAG retrieval.
        
        Orchestrates the complete pipeline: parse → chunk → embed → store.
        Supports PDF, DOCX, TXT, MD, and CSV formats.
        
        Args:
            file_path: Path to the file to index
            document_id: Unique document identifier
            title: Document title
            doc_type: Document type (LEASE, CONTRACT, REGULATION, etc.)
            industry: Industry classification
            metadata: Optional metadata dict (version, document-specific fields, etc.)
                    See MD file for complete metadata schemas.
        
        Returns:
            Dict with document_id, title, chunk_count, format, and status.
        
        Raises:
                FileNotFoundError: If file doesn't exist
                ValueError: If file format is unsupported
                RuntimeError: If indexing fails
        """
        try:
            # Parse file
            text = self._parse_file(file_path)
            
            # Get file extension for return value
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            
            # Build metadata
            full_metadata = metadata or {}
            full_metadata.update({
                "document_id": document_id,
                "title": title,
                "doc_type": doc_type.value,
                "industry": industry
            })
            
            # Chunk document
            documents = self._chunk_document(text, document_id, full_metadata)
            
            # Store in Milvus
            self.vector_store.add_documents(documents)
            
            logger.info(
                "Document indexed successfully",
                document_id=document_id,
                chunk_count=len(documents),
                format=file_ext
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "chunk_count": len(documents),
                "format": file_ext,
                "message": "Document indexed successfully"
            }
            
        except FileNotFoundError as e:
            logger.error(
                "File not found",
                file_path=file_path,
                document_id=document_id,
                error=str(e)
            )
            raise  # Re-raise specific error
            
        except ValueError as e:
            logger.error(
                "Invalid file format or parameter",
                file_path=file_path,
                document_id=document_id,
                error=str(e)
            )
            raise  # Re-raise specific error
            
        except Exception as e:
            logger.error(
                "Failed to index document",
                file_path=file_path,
                document_id=document_id,
                error=str(e),
                exc_info=True  # Include stack trace
            )
            raise RuntimeError(f"Failed to index document: {str(e)}") from e

    async def search(
        self,
        query: str,
        doc_types: Optional[List[B2BDocumentType]] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search indexed documents using semantic similarity.
        
        Performs vector similarity search to find the most relevant document chunks
        matching the query. Supports filtering by document type and metadata.
        
        Args:
            query: Search query text
            doc_types: Optional list of document types to filter by
            top_k: Number of results to return (default: 5)
            metadata_filters: Optional metadata filters (e.g., {"lease_id": "LEASE-001"})
        
        Returns:
            List of dictionaries containing:
            - content: Chunk text content
            - title: Document title
            - doc_type: Document type
            - document_id: Document identifier
            - chunk_index: Chunk index within document
            - score: Similarity score (0-1, higher = more relevant)
            - metadata: Full chunk metadata
        """
        try:
            # Build filter for document types if specified
            search_filter = None
            if doc_types:
                doc_type_values = [dt.value for dt in doc_types]
                # Milvus filter expression: doc_type in ["lease", "contract", ...]
                if len(doc_type_values) == 1:
                    search_filter = {"doc_type": doc_type_values[0]}
                else:
                    # For multiple types, we'll filter in post-processing
                    # (Milvus filter expressions can be complex)
                    search_filter = None
            
            # Merge with additional metadata filters
            if metadata_filters:
                if search_filter:
                    search_filter.update(metadata_filters)
                else:
                    search_filter = metadata_filters
            
            # Perform similarity search
            documents = self.vector_store.similarity_search(
                query=query,
                k=top_k * 2,  # Get more results to filter if needed
                filter=search_filter
            )
            
            if not documents:
                logger.info("No documents found", query=query[:100])
                return []
            
            # Convert Document objects to dict format
            results = []
            for doc in documents:
                metadata = doc.metadata or {}
                
                # Filter by doc_types if specified (post-processing)
                if doc_types:
                    doc_type_value = metadata.get("doc_type")
                    if doc_type_value not in [dt.value for dt in doc_types]:
                        continue
                
                # Apply additional metadata filters if any
                if metadata_filters:
                    matches = all(
                        metadata.get(key) == value
                        for key, value in metadata_filters.items()
                    )
                    if not matches:
                        continue
                
                # Calculate similarity score (if available from vector store)
                # Note: Milvus similarity_search doesn't return scores directly,
                # but we can use a placeholder or query with scores separately
                score = metadata.get("score", 0.0)  # Will be 0.0 if not available
                
                result = {
                    "content": doc.page_content,
                    "title": metadata.get("title", "Unknown"),
                    "doc_type": metadata.get("doc_type", "unknown"),
                    "document_id": metadata.get("document_id", "unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "total_chunks": metadata.get("total_chunks", 0),
                    "score": score,
                    "metadata": metadata
                }
                results.append(result)
            
            # Limit to top_k after filtering
            results = results[:top_k]
            
            logger.info(
                "Search completed",
                query=query[:100],
                results_count=len(results),
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Search failed",
                query=query[:100],
                error=str(e),
                exc_info=True
            )
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about indexed documents.
        
        Returns:
            Dictionary containing:
            - collection_name: Name of the Milvus collection
            - num_entities: Total number of document chunks indexed
            - milvus_available: Whether Milvus is connected
            - using_fallback: Whether using in-memory fallback
            - connection_state: Connection state
            - healthy: Overall health status
            - embeddings_available: Whether embeddings are available
        """
        try:
            # Get basic stats from vector store
            stats = self.vector_store.stats()
            
            # Add health check info
            health = self.vector_store.health_check()
            
            result = {
                "collection_name": self.collection_name,
                "num_entities": stats.get("num_entities", 0),
                "milvus_available": stats.get("milvus_available", False),
                "using_fallback": stats.get("using_fallback", False),
                "connection_state": stats.get("connection_state", "unknown"),
                "healthy": health.get("healthy", False),
                "embeddings_available": health.get("embeddings_available", False),
            }
            
            logger.info("Statistics retrieved", collection=self.collection_name, num_entities=result["num_entities"])
            return result
            
        except Exception as e:
            logger.error("Failed to get statistics", error=str(e), exc_info=True)
            return {
                "error": str(e),
                "collection_name": self.collection_name
            }
# Singleton instance
_document_indexing_service: Optional[DocumentIndexingService] = None

def get_document_indexing_service(
    collection_name: str = "language_intelligence_documents",
    **kwargs
) -> DocumentIndexingService:
    """
    Get singleton DocumentIndexingService instance.
    
    Args:
        collection_name: Name of the Milvus collection
        **kwargs: Additional arguments passed to DocumentIndexingService.__init__
    
    Returns:
        DocumentIndexingService instance
    """
    global _document_indexing_service
    if _document_indexing_service is None:
        _document_indexing_service = DocumentIndexingService(
            collection_name=collection_name,
            **kwargs
        )
    return _document_indexing_service