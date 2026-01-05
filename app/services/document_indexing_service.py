"""
Document Indexing Service for B2B Data
Comprehensive document processing, chunking, and indexing into Milvus
Supports multiple file formats and B2B document types
"""
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import asyncio
import json
import uuid
import mimetypes
from ..integrations.milvus_store import get_milvus_store, MilvusVectorStore
import asyncio
from typing import Optional, callable
from ..integrations.universal_rag_engine import create_universal_rag_engine
from ..integrations.enhanced_universal_rag_engine import create_enhanced_rag_engine
from ..logging_config import get_logger
from ..settings import settings

logger = get_logger("cyrex.document_indexing")


class DocumentFormat(str, Enum):
    """Supported document formats"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "markdown"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    XML = "xml"
    XLSX = "xlsx"
    PPTX = "pptx"
    EMAIL = "email"
    IMAGE = "image"  # OCR support


class B2BDocumentType(str, Enum):
    """B2B document types"""
    # Contracts & Legal
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"
    LEGAL_DOCUMENT = "legal_document"
    
    # Financial
    INVOICE = "invoice"
    RECEIPT = "receipt"
    QUOTE = "quote"
    PURCHASE_ORDER = "purchase_order"
    FINANCIAL_REPORT = "financial_report"
    BUDGET = "budget"
    
    # Operations
    PROCEDURE = "procedure"
    MANUAL = "manual"
    POLICY = "policy"
    GUIDELINE = "guideline"
    STANDARD = "standard"
    SPECIFICATION = "specification"
    
    # Communication
    EMAIL = "email"
    MEMO = "memo"
    REPORT = "report"
    PRESENTATION = "presentation"
    MEETING_NOTES = "meeting_notes"
    
    # Vendor/Supplier
    VENDOR_PROFILE = "vendor_profile"
    SUPPLIER_AGREEMENT = "supplier_agreement"
    CATALOG = "catalog"
    PRICE_LIST = "price_list"
    
    # Compliance
    REGULATION = "regulation"
    COMPLIANCE_REPORT = "compliance_report"
    AUDIT_REPORT = "audit_report"
    CERTIFICATE = "certificate"
    
    # Knowledge Base
    FAQ = "faq"
    KNOWLEDGE_BASE = "knowledge_base"
    WIKI = "wiki"
    DOCUMENTATION = "documentation"
    
    # Other
    OTHER = "other"


@dataclass
class DocumentChunk:
    """Chunk of a document"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedDocument:
    """Indexed document metadata"""
    document_id: str
    title: str
    doc_type: B2BDocumentType
    format: DocumentFormat
    source: str
    industry: str
    file_size: int
    page_count: Optional[int] = None
    chunk_count: int = 0
    indexed_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Document chunking strategies"""
    
    @staticmethod
    def chunk_by_size(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[str]:
        """Chunk text by character size"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '.\n', '! ', '?\n', '?\n']:
                    last_punct = chunk.rfind(punct)
                    if last_punct > chunk_size * 0.5:  # At least 50% of chunk
                        chunk = chunk[:last_punct + 1]
                        end = start + len(chunk)
                        break
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
        
        return chunks
    
    @staticmethod
    def chunk_by_paragraphs(text: str, max_chunk_size: int = 2000) -> List[str]:
        """Chunk by paragraphs"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for '\n\n'
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def chunk_by_sentences(text: str, sentences_per_chunk: int = 10) -> List[str]:
        """Chunk by sentence count"""
        import re
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = '. '.join(sentences[i:i + sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk.strip() + '.')
        
        return chunks


class DocumentProcessor:
    """Process various document formats"""
    
    @staticmethod
    async def extract_text(file_path: str, format: DocumentFormat) -> str:
        """Extract text from file based on format"""
        try:
            if format == DocumentFormat.PDF:
                return await DocumentProcessor._extract_pdf(file_path)
            elif format == DocumentFormat.DOCX:
                return await DocumentProcessor._extract_docx(file_path)
            elif format == DocumentFormat.TXT:
                return await DocumentProcessor._extract_txt(file_path)
            elif format == DocumentFormat.MD:
                return await DocumentProcessor._extract_txt(file_path)
            elif format == DocumentFormat.CSV:
                return await DocumentProcessor._extract_csv(file_path)
            elif format == DocumentFormat.JSON:
                return await DocumentProcessor._extract_json(file_path)
            elif format == DocumentFormat.HTML:
                return await DocumentProcessor._extract_html(file_path)
            elif format == DocumentFormat.XLSX:
                return await DocumentProcessor._extract_xlsx(file_path)
            else:
                logger.warning(f"Unsupported format: {format}, trying text extraction")
                return await DocumentProcessor._extract_txt(file_path)
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise
    
    @staticmethod
    async def _extract_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
                return '\n'.join(text)
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = []
                    for page in pdf.pages:
                        text.append(page.extract_text())
                    return '\n'.join(text)
            except ImportError:
                raise ImportError("PyPDF2 or pdfplumber required for PDF extraction")
    
    @staticmethod
    async def _extract_docx(file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            from docx import Document
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("python-docx required for DOCX extraction")
    
    @staticmethod
    async def _extract_txt(file_path: str) -> str:
        """Extract text from plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    @staticmethod
    async def _extract_csv(file_path: str) -> str:
        """Extract text from CSV"""
        import csv
        text_parts = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                text_parts.append(', '.join(row))
        return '\n'.join(text_parts)
    
    @staticmethod
    async def _extract_json(file_path: str) -> str:
        """Extract text from JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    
    @staticmethod
    async def _extract_html(file_path: str) -> str:
        """Extract text from HTML"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text(separator='\n', strip=True)
        except ImportError:
            # Fallback: simple regex extraction
            import re
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()
                text = re.sub(r'<[^>]+>', '', html)
                return text
    
    @staticmethod
    async def _extract_xlsx(file_path: str) -> str:
        """Extract text from Excel"""
        try:
            import pandas as pd
            df = pd.read_excel(file_path, sheet_name=None)
            text_parts = []
            for sheet_name, sheet_df in df.items():
                text_parts.append(f"Sheet: {sheet_name}")
                text_parts.append(sheet_df.to_string())
            return '\n\n'.join(text_parts)
        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel extraction")
    
    @staticmethod
    def detect_format(file_path: str) -> DocumentFormat:
        """Detect document format from file extension"""
        ext = Path(file_path).suffix.lower()
        format_map = {
            '.pdf': DocumentFormat.PDF,
            '.docx': DocumentFormat.DOCX,
            '.doc': DocumentFormat.DOCX,
            '.txt': DocumentFormat.TXT,
            '.md': DocumentFormat.MD,
            '.csv': DocumentFormat.CSV,
            '.json': DocumentFormat.JSON,
            '.html': DocumentFormat.HTML,
            '.htm': DocumentFormat.HTML,
            '.xml': DocumentFormat.XML,
            '.xlsx': DocumentFormat.XLSX,
            '.xls': DocumentFormat.XLSX,
            '.pptx': DocumentFormat.PPTX,
            '.ppt': DocumentFormat.PPTX,
        }
        return format_map.get(ext, DocumentFormat.TXT)


class DocumentIndexingService:
    """
    Comprehensive document indexing service for B2B data
    Handles file upload, processing, chunking, and indexing into Milvus
    """
    
    def __init__(
        self,
        collection_name: str = "b2b_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "size",  # "size", "paragraph", "sentence"
        use_enhanced_rag: bool = True,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.use_enhanced_rag = use_enhanced_rag
        
        self.vector_store: Optional[MilvusVectorStore] = None
        self.rag_engine = None
        self.chunker = DocumentChunker()
        self.processor = DocumentProcessor()
        
        self._indexed_documents: Dict[str, IndexedDocument] = {}
        self.logger = logger
    
    async def initialize(self):
        """Initialize vector store and RAG engine"""
        try:
            # Initialize Milvus vector store (sync function, but we're in async context)
            self.vector_store = await asyncio.to_thread(
                get_milvus_store,
                collection_name=self.collection_name,
            )
            
            # Initialize RAG engine (sync functions)
            if self.use_enhanced_rag:
                from ..integrations.enhanced_universal_rag_engine import IndustryNiche
                self.rag_engine = await asyncio.to_thread(
                    create_enhanced_rag_engine,
                    industry=IndustryNiche.GENERIC,
                    collection_name=self.collection_name,
                    enable_caching=True,
                    enable_monitoring=True,
                )
            else:
                from ..integrations.universal_rag_engine import IndustryNiche
                self.rag_engine = await asyncio.to_thread(
                    create_universal_rag_engine,
                    industry=IndustryNiche.GENERIC,
                    collection_name=self.collection_name,
                )
            
            self.logger.info(f"Document indexing service initialized: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize indexing service: {e}", exc_info=True)
            raise
    
    async def index_file(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        doc_type: Optional[B2BDocumentType] = None,
        industry: str = "generic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IndexedDocument:
        """
        Index a file into Milvus
        
        Args:
            file_path: Path to file
            document_id: Optional document ID (auto-generated if not provided)
            title: Document title
            doc_type: Document type
            industry: Industry niche
            metadata: Additional metadata
        
        Returns:
            IndexedDocument with metadata
        """
        if not self.vector_store:
            await self.initialize()
        
        document_id = document_id or str(uuid.uuid4())
        file_path_obj = Path(file_path)
        
        # Detect format
        format_type = self.processor.detect_format(file_path)
        
        # Extract text
        self.logger.info(f"Extracting text from {file_path} (format: {format_type.value})")
        text = await self.processor.extract_text(file_path, format_type)
        
        if not text.strip():
            raise ValueError(f"No text extracted from {file_path}")
        
        # Determine document type if not provided
        if not doc_type:
            doc_type = self._infer_document_type(file_path_obj.name, text)
        
        # Determine title if not provided
        if not title:
            title = file_path_obj.stem
        
        # Chunk document
        chunks = self._chunk_document(text)
        
        # Index chunks
        from deepiri_modelkit.rag import Document, DocumentType, IndustryNiche
        
        indexed_chunks = 0
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                id=f"{document_id}_chunk_{i}",
                content=chunk_text,
                doc_type=DocumentType(doc_type.value),
                industry=IndustryNiche(industry),
                title=f"{title} (Chunk {i+1}/{len(chunks)})",
                source=file_path,
                created_at=datetime.utcnow(),
                metadata={
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "format": format_type.value,
                    **(metadata or {}),
                },
            )
            
            success = self.rag_engine.index_document(chunk_doc)
            if success:
                indexed_chunks += 1
        
        # Create indexed document record
        indexed_doc = IndexedDocument(
            document_id=document_id,
            title=title,
            doc_type=doc_type,
            format=format_type,
            source=file_path,
            industry=industry,
            file_size=file_path_obj.stat().st_size,
            chunk_count=indexed_chunks,
            metadata=metadata or {},
        )
        
        self._indexed_documents[document_id] = indexed_doc
        
        self.logger.info(
            f"Indexed document: {document_id}",
            title=title,
            chunks=indexed_chunks,
            format=format_type.value
        )
        
        return indexed_doc
    
    async def index_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        title: str = "Untitled Document",
        doc_type: B2BDocumentType = B2BDocumentType.OTHER,
        industry: str = "generic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IndexedDocument:
        """Index raw text content"""
        if not self.vector_store:
            await self.initialize()
        
        document_id = document_id or str(uuid.uuid4())
        
        # Chunk document
        chunks = self._chunk_document(text)
        
        # Index chunks
        from deepiri_modelkit.rag import Document, DocumentType, IndustryNiche
        
        indexed_chunks = 0
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                id=f"{document_id}_chunk_{i}",
                content=chunk_text,
                doc_type=DocumentType(doc_type.value),
                industry=IndustryNiche(industry),
                title=f"{title} (Chunk {i+1}/{len(chunks)})",
                source="text_input",
                created_at=datetime.utcnow(),
                metadata={
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(metadata or {}),
                },
            )
            
            success = self.rag_engine.index_document(chunk_doc)
            if success:
                indexed_chunks += 1
        
        indexed_doc = IndexedDocument(
            document_id=document_id,
            title=title,
            doc_type=doc_type,
            format=DocumentFormat.TXT,
            source="text_input",
            industry=industry,
            file_size=len(text.encode('utf-8')),
            chunk_count=indexed_chunks,
            metadata=metadata or {},
        )
        
        self._indexed_documents[document_id] = indexed_doc
        
        return indexed_doc
    
    async def index_batch(
        self,
        files: List[Dict[str, Any]],
        industry: str = "generic",
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Index multiple files in batch
        
        Args:
            files: List of file info dicts with keys: file_path, document_id (opt), title (opt), doc_type (opt), metadata (opt)
            industry: Industry for all documents
            progress_callback: Optional callback(progress, total, current_file)
        
        Returns:
            Batch indexing result
        """
        if not self.vector_store:
            await self.initialize()
        
        results = {
            "successful": [],
            "failed": [],
            "total": len(files),
        }
        
        for i, file_info in enumerate(files):
            try:
                indexed_doc = await self.index_file(
                    file_path=file_info["file_path"],
                    document_id=file_info.get("document_id"),
                    title=file_info.get("title"),
                    doc_type=file_info.get("doc_type"),
                    industry=industry,
                    metadata=file_info.get("metadata"),
                )
                results["successful"].append({
                    "document_id": indexed_doc.document_id,
                    "title": indexed_doc.title,
                    "chunks": indexed_doc.chunk_count,
                })
                
                if progress_callback:
                    progress_callback(i + 1, len(files), file_info.get("file_path", "unknown"))
            
            except Exception as e:
                error_info = {
                    "file_path": file_info.get("file_path", "unknown"),
                    "error": str(e),
                }
                results["failed"].append(error_info)
                self.logger.error(f"Failed to index file: {error_info}")
        
        results["success_count"] = len(results["successful"])
        results["failed_count"] = len(results["failed"])
        results["success_rate"] = results["success_count"] / results["total"] if results["total"] > 0 else 0
        
        return results
    
    def _chunk_document(self, text: str) -> List[str]:
        """Chunk document using configured strategy"""
        if self.chunking_strategy == "paragraph":
            return self.chunker.chunk_by_paragraphs(text, max_chunk_size=self.chunk_size)
        elif self.chunking_strategy == "sentence":
            return self.chunker.chunk_by_sentences(text, sentences_per_chunk=self.chunk_size // 100)
        else:  # "size" default
            return self.chunker.chunk_by_size(text, self.chunk_size, self.chunk_overlap)
    
    def _infer_document_type(self, filename: str, text: str) -> B2BDocumentType:
        """Infer document type from filename and content"""
        filename_lower = filename.lower()
        text_lower = text.lower()[:500]  # Check first 500 chars
        
        # Check filename patterns
        if 'invoice' in filename_lower or 'inv_' in filename_lower:
            return B2BDocumentType.INVOICE
        elif 'contract' in filename_lower or 'agreement' in filename_lower:
            return B2BDocumentType.CONTRACT
        elif 'manual' in filename_lower or 'guide' in filename_lower:
            return B2BDocumentType.MANUAL
        elif 'policy' in filename_lower:
            return B2BDocumentType.POLICY
        elif 'procedure' in filename_lower:
            return B2BDocumentType.PROCEDURE
        elif 'report' in filename_lower:
            return B2BDocumentType.REPORT
        elif 'quote' in filename_lower or 'quotation' in filename_lower:
            return B2BDocumentType.QUOTE
        elif 'po' in filename_lower or 'purchase_order' in filename_lower:
            return B2BDocumentType.PURCHASE_ORDER
        
        # Check content patterns
        if 'invoice number' in text_lower or 'bill to' in text_lower:
            return B2BDocumentType.INVOICE
        elif 'terms and conditions' in text_lower or 'this agreement' in text_lower:
            return B2BDocumentType.CONTRACT
        elif 'step 1' in text_lower and 'step 2' in text_lower:
            return B2BDocumentType.PROCEDURE
        
        return B2BDocumentType.OTHER
    
    async def search(
        self,
        query: str,
        doc_types: Optional[List[B2BDocumentType]] = None,
        industry: Optional[str] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search indexed documents"""
        if not self.rag_engine:
            await self.initialize()
        
        from deepiri_modelkit.rag import RAGQuery, DocumentType, IndustryNiche
        
        doc_types_enum = None
        if doc_types:
            doc_types_enum = [DocumentType(dt.value) for dt in doc_types]
        
        industry_enum = IndustryNiche(industry) if industry else None
        
        rag_query = RAGQuery(
            query=query,
            industry=industry_enum,
            doc_types=doc_types_enum,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )
        
        results = self.rag_engine.retrieve(rag_query)
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "document_id": result.document.metadata.get("document_id"),
                "chunk_index": result.document.metadata.get("chunk_index"),
                "title": result.document.title,
                "content": result.document.content,
                "doc_type": result.document.doc_type.value if hasattr(result.document.doc_type, 'value') else result.document.doc_type,
                "score": result.score,
                "rerank_score": result.rerank_score,
                "source": result.document.source,
            })
        
        return formatted
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        if not self.rag_engine:
            await self.initialize()
        
        # Find all chunk IDs for this document
        # Note: This requires querying or maintaining a mapping
        # For now, we'll use a pattern-based approach
        
        # Search for chunks with this document_id in metadata
        results = await self.search(
            query="",  # Empty query to get all
            metadata_filters={"document_id": document_id},
            top_k=10000,  # Large number to get all chunks
        )
        
        chunk_ids = [r.get("document_id", "") + f"_chunk_{r.get('chunk_index', 0)}" for r in results]
        
        if chunk_ids:
            success = self.rag_engine.delete_documents(chunk_ids)
            if success:
                self._indexed_documents.pop(document_id, None)
            return success
        
        return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        if not self.vector_store:
            await self.initialize()
        
        stats = self.vector_store.stats()
        
        # Count by document type
        doc_type_counts = {}
        for doc in self._indexed_documents.values():
            doc_type = doc.doc_type.value
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
        
        return {
            "collection_name": self.collection_name,
            "total_documents": len(self._indexed_documents),
            "total_chunks": sum(doc.chunk_count for doc in self._indexed_documents.values()),
            "documents_by_type": doc_type_counts,
            "vector_store_stats": stats,
        }
    
    def get_indexed_document(self, document_id: str) -> Optional[IndexedDocument]:
        """Get metadata for an indexed document"""
        return self._indexed_documents.get(document_id)
    
    def list_indexed_documents(
        self,
        doc_type: Optional[B2BDocumentType] = None,
        industry: Optional[str] = None,
    ) -> List[IndexedDocument]:
        """List all indexed documents with optional filters"""
        docs = list(self._indexed_documents.values())
        
        if doc_type:
            docs = [d for d in docs if d.doc_type == doc_type]
        
        if industry:
            docs = [d for d in docs if d.industry == industry]
        
        return docs


# ============================================================================
# Singleton Instance
# ============================================================================

_indexing_service: Optional[DocumentIndexingService] = None


async def get_document_indexing_service(
    collection_name: str = "b2b_documents",
    **kwargs
) -> DocumentIndexingService:
    """Get or create document indexing service singleton"""
    global _indexing_service
    if _indexing_service is None:
        _indexing_service = DocumentIndexingService(
            collection_name=collection_name,
            **kwargs
        )
        await _indexing_service.initialize()
    return _indexing_service

