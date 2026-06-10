"""
Advanced Document Analyzer with ML-based Classification and Extraction
Uses LLM and ML models for document type classification, layout analysis, and structured extraction
"""
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import re
import json

logger = logging.getLogger(__name__)


class DocumentCategory(str, Enum):
    """Document categories for classification"""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    REPORT = "report"
    FORM = "form"
    CONTRACT = "contract"
    LETTER = "letter"
    STATEMENT = "statement"
    RESUME = "resume"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    GENERAL = "general"
    UNKNOWN = "unknown"


class LayoutElement:
    """Represents a layout element in a document"""
    def __init__(self, element_type: str, content: str, bbox: Optional[Tuple[float, float, float, float]] = None):
        self.element_type = element_type  # 'header', 'paragraph', 'table', 'list', 'footer', 'title'
        self.content = content
        self.bbox = bbox  # (x1, y1, x2, y2) bounding box
        self.confidence = 1.0


class AdvancedDocumentAnalyzer:
    """Advanced analyzer using ML/LLM for document understanding"""
    
    def __init__(self):
        self.category_keywords = {
            DocumentCategory.INVOICE: ['invoice', 'bill', 'amount due', 'total', 'invoice number', 'invoice date'],
            DocumentCategory.RECEIPT: ['receipt', 'payment', 'paid', 'transaction', 'date', 'total'],
            DocumentCategory.REPORT: ['report', 'summary', 'analysis', 'findings', 'conclusion'],
            DocumentCategory.FORM: ['form', 'application', 'please fill', 'signature', 'date'],
            DocumentCategory.CONTRACT: ['contract', 'agreement', 'terms', 'conditions', 'party'],
            DocumentCategory.STATEMENT: ['statement', 'account', 'balance', 'transaction', 'statement date'],
            DocumentCategory.RESUME: ['resume', 'cv', 'experience', 'education', 'skills', 'objective'],
        }
    
    async def classify_document(
        self,
        text: str,
        filename: str,
        use_llm: bool = True,
    ) -> Tuple[DocumentCategory, float]:
        """
        Classify document type using keyword matching and optionally LLM
        
        Args:
            text: Extracted text from document
            filename: Original filename
            use_llm: Whether to use LLM for classification (more accurate)
        
        Returns:
            Tuple of (DocumentCategory, confidence_score)
        """
        # First try keyword-based classification
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower or keyword in filename_lower)
            if score > 0:
                scores[category] = score / len(keywords)
        
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            if best_category[1] > 0.3:  # Threshold for keyword-based classification
                return best_category[0], min(best_category[1] * 0.8, 0.8)  # Cap at 0.8 for keyword-based
        
        # Use LLM for more accurate classification if available
        if use_llm:
            try:
                category = await self._classify_with_llm(text[:2000], filename)  # Use first 2000 chars
                if category:
                    return category, 0.9  # High confidence for LLM classification
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, falling back to keyword matching")
        
        # Fallback to general
        return DocumentCategory.GENERAL, 0.5
    
    async def _classify_with_llm(self, text: str, filename: str) -> Optional[DocumentCategory]:
        """Classify document using LLM"""
        try:
            from ..integrations.ollama_container import get_ollama_client, ChatMessage, GenerationOptions
            
            ollama = await get_ollama_client()
            
            prompt = f"""Analyze this document and classify it into one of these categories:
- invoice
- receipt
- report
- form
- contract
- letter
- statement
- resume
- presentation
- spreadsheet
- general

Document filename: {filename}
Document text preview: {text[:1000]}

Respond with ONLY the category name (lowercase, one word)."""
            
            messages = [
                ChatMessage(role="system", content="You are a document classification expert. Respond with only the category name."),
                ChatMessage(role="user", content=prompt),
            ]
            
            options = GenerationOptions(temperature=0.1, num_predict=50)  # Low temperature for classification
            result = await ollama.chat(messages, model="llama3:8b", options=options)
            
            response = result.response.strip().lower()
            
            # Map response to category
            category_map = {
                'invoice': DocumentCategory.INVOICE,
                'receipt': DocumentCategory.RECEIPT,
                'report': DocumentCategory.REPORT,
                'form': DocumentCategory.FORM,
                'contract': DocumentCategory.CONTRACT,
                'letter': DocumentCategory.LETTER,
                'statement': DocumentCategory.STATEMENT,
                'resume': DocumentCategory.RESUME,
                'cv': DocumentCategory.RESUME,
                'presentation': DocumentCategory.PRESENTATION,
                'spreadsheet': DocumentCategory.SPREADSHEET,
                'general': DocumentCategory.GENERAL,
            }
            
            for key, category in category_map.items():
                if key in response:
                    return category
            
            return DocumentCategory.GENERAL
            
        except Exception as e:
            logger.error(f"LLM classification error: {e}", exc_info=True)
            return None
    
    async def detect_layout(
        self,
        text: str,
        tables: List[List[List[str]]],
        use_ml: bool = False,
    ) -> List[LayoutElement]:
        """
        Detect document layout structure
        
        Args:
            text: Full document text
            tables: Extracted tables
            use_ml: Whether to use ML models (future enhancement)
        
        Returns:
            List of LayoutElement objects
        """
        elements = []
        lines = text.split('\n')
        
        # Simple heuristic-based layout detection
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Detect headers (short lines, all caps, or with special formatting)
            if len(line_stripped) < 100 and (
                line_stripped.isupper() or
                line_stripped.startswith('#') or
                re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line_stripped)
            ):
                elements.append(LayoutElement('header', line_stripped))
            
            # Detect titles (first non-empty line, or lines with title case)
            elif i < 5 and len(line_stripped) < 200:
                elements.append(LayoutElement('title', line_stripped))
            
            # Detect lists (lines starting with bullets, numbers, or dashes)
            elif re.match(r'^[\s]*[•\-\*\d+\.\)]\s+', line_stripped):
                elements.append(LayoutElement('list', line_stripped))
            
            # Regular paragraphs
            else:
                elements.append(LayoutElement('paragraph', line_stripped))
        
        # Add tables as layout elements
        for table in tables:
            table_text = '\n'.join([' | '.join(row) for row in table[:5]])  # First 5 rows
            elements.append(LayoutElement('table', table_text))
        
        return elements
    
    async def extract_key_value_pairs(
        self,
        text: str,
        document_category: DocumentCategory,
        use_llm: bool = True,
    ) -> Dict[str, str]:
        """
        Extract key-value pairs from document
        
        Args:
            text: Document text
            document_category: Classified document category
            use_llm: Whether to use LLM for extraction
        
        Returns:
            Dictionary of key-value pairs
        """
        kvp = {}
        
        # Pattern-based extraction for common fields
        patterns = {
            'invoice_number': r'(?:invoice\s*(?:#|number|no\.?|num\.?):?\s*)([A-Z0-9\-]+)',
            'date': r'(?:date:?\s*)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            'total': r'(?:total:?\s*\$?)([\d,]+\.?\d*)',
            'amount_due': r'(?:amount\s+due:?\s*\$?)([\d,]+\.?\d*)',
            'due_date': r'(?:due\s+date:?\s*)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            'vendor': r'(?:vendor|from|seller):?\s*([A-Z][A-Za-z\s&,\.]+)',
            'customer': r'(?:customer|to|buyer|bill\s+to):?\s*([A-Z][A-Za-z\s&,\.]+)',
            'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'phone': r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
        }
        
        text_lower = text.lower()
        for key, pattern in patterns.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                value = match.group(1) if match.groups() else match.group(0)
                if value and value not in kvp.values():  # Avoid duplicates
                    kvp[key] = value.strip()
                    break  # Take first match
        
        # Use LLM for more sophisticated extraction
        if use_llm and len(text) > 100:
            try:
                llm_kvp = await self._extract_kvp_with_llm(text[:3000], document_category)
                kvp.update(llm_kvp)
            except Exception as e:
                logger.warning(f"LLM KVP extraction failed: {e}")
        
        return kvp
    
    async def _extract_kvp_with_llm(
        self,
        text: str,
        document_category: DocumentCategory,
    ) -> Dict[str, str]:
        """Extract key-value pairs using LLM"""
        try:
            from ..integrations.ollama_container import get_ollama_client, ChatMessage, GenerationOptions
            
            ollama = await get_ollama_client()
            
            # Category-specific extraction prompts
            extraction_prompts = {
                DocumentCategory.INVOICE: """Extract key information from this invoice:
- invoice_number
- invoice_date
- due_date
- vendor_name
- customer_name
- subtotal
- tax
- total
- amount_due""",
                DocumentCategory.RECEIPT: """Extract key information from this receipt:
- receipt_number
- date
- merchant_name
- total_amount
- payment_method""",
                DocumentCategory.CONTRACT: """Extract key information from this contract:
- contract_number
- effective_date
- expiration_date
- party_a
- party_b
- contract_value""",
            }
            
            base_prompt = extraction_prompts.get(
                document_category,
                "Extract key-value pairs from this document. Focus on important fields like dates, names, amounts, and identifiers."
            )
            
            prompt = f"""{base_prompt}

Document text:
{text[:2000]}

Respond with a JSON object containing the extracted fields. Use null for missing fields.
Example: {{"invoice_number": "INV-123", "date": "2024-01-15", "total": "1000.00"}}"""
            
            messages = [
                ChatMessage(role="system", content="You are a document extraction expert. Respond with valid JSON only."),
                ChatMessage(role="user", content=prompt),
            ]
            
            options = GenerationOptions(temperature=0.1, num_predict=500)
            result = await ollama.chat(messages, model="llama3:8b", options=options)
            
            # Parse JSON response
            response_text = result.response.strip()
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            try:
                kvp = json.loads(response_text)
                # Filter out null values and ensure all values are strings
                return {k: str(v) for k, v in kvp.items() if v is not None}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM JSON response: {response_text[:200]}")
                return {}
            
        except Exception as e:
            logger.error(f"LLM KVP extraction error: {e}", exc_info=True)
            return {}
    
    def get_column_mapping(
        self,
        document_category: DocumentCategory,
        tables: List[List[List[str]]],
        key_value_pairs: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Get smart column mapping based on document type
        
        Args:
            document_category: Classified document category
            tables: Extracted tables
            key_value_pairs: Extracted key-value pairs
        
        Returns:
            Dictionary with column mapping suggestions
        """
        mappings = {
            DocumentCategory.INVOICE: {
                'suggested_columns': ['Item', 'Description', 'Quantity', 'Unit Price', 'Total'],
                'kvp_columns': ['Invoice Number', 'Invoice Date', 'Due Date', 'Vendor', 'Customer', 'Subtotal', 'Tax', 'Total', 'Amount Due'],
                'start_row': 1,
            },
            DocumentCategory.RECEIPT: {
                'suggested_columns': ['Item', 'Price', 'Quantity', 'Total'],
                'kvp_columns': ['Receipt Number', 'Date', 'Merchant', 'Total Amount', 'Payment Method'],
                'start_row': 1,
            },
            DocumentCategory.REPORT: {
                'suggested_columns': ['Section', 'Content', 'Page'],
                'kvp_columns': ['Report Title', 'Date', 'Author', 'Summary'],
                'start_row': 1,
            },
            DocumentCategory.FORM: {
                'suggested_columns': ['Field', 'Value'],
                'kvp_columns': list(key_value_pairs.keys()),
                'start_row': 1,
            },
        }
        
        return mappings.get(document_category, {
            'suggested_columns': ['Column 1', 'Column 2', 'Column 3'],
            'kvp_columns': list(key_value_pairs.keys()),
            'start_row': 1,
        })


# Singleton instance
_analyzer: Optional[AdvancedDocumentAnalyzer] = None


def get_advanced_analyzer() -> AdvancedDocumentAnalyzer:
    """Get singleton AdvancedDocumentAnalyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = AdvancedDocumentAnalyzer()
    return _analyzer

