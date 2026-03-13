"""
Universal Invoice Processing Engine
Processes invoices across all 6 industries with industry-specific handling

This is the UNIVERSAL engine that works for:
- Property Management (HVAC, plumbing, electrical invoices)
- Corporate Procurement (purchase orders, supplier invoices)
- P&C Insurance (auto body shop, home repair contractor invoices)
- General Contractors (subcontractor invoices, material supplier invoices)
- Retail/E-Commerce (freight carrier invoices, warehouse vendor invoices)
- Law Firms (expert witness invoices, e-discovery vendor invoices)

Architecture:
- OCR for all invoice formats
- Multimodal AI for invoice parsing
- Industry-specific LoRA adapters for context
- Structured data extraction
- One engine, all industries
"""
from typing import Dict, List, Optional, Any, BinaryIO, Union
from datetime import datetime
from dataclasses import dataclass, field
import json
import asyncio
import base64
import io

from ..core.types import IndustryNiche
from ..logging_config import get_logger
from .document_verification_service import DocumentVerificationService

logger = get_logger("cyrex.invoice_parser")


@dataclass
class InvoiceLineItem:
    """Invoice line item structure"""
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None
    service_category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedInvoice:
    """Processed invoice structure"""
    invoice_id: str
    vendor_name: str
    vendor_id: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[datetime] = None
    total_amount: float = 0.0
    line_items: List[InvoiceLineItem] = field(default_factory=list)
    service_category: Optional[str] = None
    work_description: Optional[str] = None
    property_address: Optional[str] = None
    payment_terms: Optional[str] = None
    due_date: Optional[datetime] = None
    industry: IndustryNiche = IndustryNiche.GENERIC
    extraction_confidence: float = 0.0
    extraction_method: str = "llm"
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None


class UniversalInvoiceProcessor:
    """
    Universal Invoice Processing Engine
    
    Processes invoices from all 6 industries:
    1. Property Management
    2. Corporate Procurement
    3. P&C Insurance
    4. General Contractors
    5. Retail/E-Commerce
    6. Law Firms
    
    Uses:
    - OCR for text extraction
    - Multimodal AI for intelligent parsing
    - Industry-specific LoRA adapters for context
    - Structured data extraction
    """
    
    def __init__(self):
        self.logger = logger
        self.document_verification = DocumentVerificationService()
        self._lora_adapters = {}  # Cache for LoRA adapters
        
    async def process_invoice(
        self,
        invoice_content: Union[str, bytes, BinaryIO],
        industry: IndustryNiche = IndustryNiche.GENERIC,
        invoice_format: str = "text",  # text, pdf, image, json
        use_lora: bool = True,
        **kwargs
    ) -> ProcessedInvoice:
        """
        Process invoice from any industry
        
        Args:
            invoice_content: Invoice content (text, bytes, or file-like)
            industry: Industry niche for context
            invoice_format: Format of invoice (text, pdf, image, json)
            use_lora: Whether to use industry-specific LoRA adapter
            **kwargs: Additional processing options
            
        Returns:
            ProcessedInvoice with extracted structured data
        """
        try:
            # Step 1: Extract raw data using document verification
            if invoice_format == "text":
                document_content = invoice_content if isinstance(invoice_content, str) else invoice_content.read().decode('utf-8')
            elif invoice_format == "json":
                # Already structured
                raw_data = json.loads(invoice_content) if isinstance(invoice_content, str) else invoice_content
                return self._process_structured_invoice(raw_data, industry)
            else:
                # PDF or image - would use OCR here
                # For now, assume text extraction already done
                document_content = str(invoice_content)
            
            # Step 2: Extract structured data
            extraction_result = await self.document_verification.extract_invoice_data(
                document_content=document_content,
                document_type="invoice",
                industry=industry
            )
            
            if not extraction_result.get("success"):
                raise ValueError(f"Failed to extract invoice data: {extraction_result.get('error')}")
            
            extracted_data = extraction_result.get("extracted_data", {})
            
            # Step 3: Enhance with industry-specific LoRA adapter if available
            if use_lora:
                extracted_data = await self._enhance_with_lora(
                    extracted_data,
                    industry,
                    document_content
                )
            
            # Step 4: Build ProcessedInvoice
            processed = self._build_processed_invoice(
                extracted_data,
                industry,
                extraction_result.get("confidence", 0.0),
                extraction_result.get("extraction_method", "llm")
            )
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Invoice processing failed: {e}", exc_info=True)
            raise
    
    async def process_invoices_batch(
        self,
        invoices: List[Dict[str, Any]],
        industry: IndustryNiche = IndustryNiche.GENERIC,
        **kwargs
    ) -> List[ProcessedInvoice]:
        """
        Process multiple invoices in batch
        
        Args:
            invoices: List of invoice data dicts
            industry: Industry niche
            **kwargs: Additional processing options
            
        Returns:
            List of ProcessedInvoice objects
        """
        tasks = [
            self.process_invoice(
                invoice_data.get("content", ""),
                industry=invoice_data.get("industry", industry),
                invoice_format=invoice_data.get("format", "text"),
                **kwargs
            )
            for invoice_data in invoices
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process invoice {i}: {result}")
                continue
            processed.append(result)
        
        return processed
    
    async def _enhance_with_lora(
        self,
        extracted_data: Dict[str, Any],
        industry: IndustryNiche,
        document_content: str
    ) -> Dict[str, Any]:
        """
        Enhance extracted data with industry-specific LoRA adapter
        
        This would load and use a LoRA adapter trained for the specific industry
        to improve extraction accuracy and add industry-specific fields
        """
        try:
            # Load LoRA adapter for industry (if available)
            lora_adapter = await self._get_lora_adapter(industry)
            
            if not lora_adapter:
                # No LoRA adapter available, return original data
                return extracted_data
            
            # Use LoRA adapter to enhance extraction
            # This would call the LoRA-enhanced model for better context understanding
            # For now, return original data (implementation would go here)
            
            return extracted_data
            
        except Exception as e:
            self.logger.warning(f"LoRA enhancement failed: {e}, using original extraction")
            return extracted_data
    
    async def _get_lora_adapter(self, industry: IndustryNiche):
        """
        Get LoRA adapter for industry (cached)
        
        In production, this would load from model registry (MLflow/S3)
        """
        if industry in self._lora_adapters:
            return self._lora_adapters[industry]
        
        # Load from model registry (would implement actual loading)
        # For now, return None (no LoRA adapters loaded)
        return None
    
    def _build_processed_invoice(
        self,
        extracted_data: Dict[str, Any],
        industry: IndustryNiche,
        confidence: float,
        extraction_method: str
    ) -> ProcessedInvoice:
        """Build ProcessedInvoice from extracted data"""
        import uuid
        
        # Parse line items
        line_items = []
        raw_line_items = extracted_data.get("line_items", [])
        for item in raw_line_items:
            line_items.append(InvoiceLineItem(
                description=item.get("description", ""),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total=item.get("total"),
                service_category=item.get("service_category"),
                metadata=item.get("metadata", {})
            ))
        
        # Parse dates
        invoice_date = None
        if extracted_data.get("invoice_date"):
            try:
                from dateutil import parser
                invoice_date = parser.parse(extracted_data["invoice_date"])
            except:
                pass
        
        due_date = None
        if extracted_data.get("due_date"):
            try:
                from dateutil import parser
                due_date = parser.parse(extracted_data["due_date"])
            except:
                pass
        
        return ProcessedInvoice(
            invoice_id=str(uuid.uuid4()),
            vendor_name=extracted_data.get("vendor_name", ""),
            vendor_id=extracted_data.get("vendor_id"),
            invoice_number=extracted_data.get("invoice_number"),
            invoice_date=invoice_date,
            total_amount=float(extracted_data.get("total_amount", 0.0)),
            line_items=line_items,
            service_category=extracted_data.get("service_category"),
            work_description=extracted_data.get("work_description"),
            property_address=extracted_data.get("property_address"),
            payment_terms=extracted_data.get("payment_terms"),
            due_date=due_date,
            industry=industry,
            extraction_confidence=confidence,
            extraction_method=extraction_method,
            metadata=extracted_data.get("metadata", {}),
            raw_data=extracted_data
        )
    
    def _process_structured_invoice(
        self,
        raw_data: Dict[str, Any],
        industry: IndustryNiche
    ) -> ProcessedInvoice:
        """Process already-structured invoice data"""
        return self._build_processed_invoice(
            raw_data,
            industry,
            confidence=1.0,
            extraction_method="structured"
        )


# Singleton instance
_processor_instance: Optional[UniversalInvoiceProcessor] = None


def get_universal_invoice_processor() -> UniversalInvoiceProcessor:
    """Get singleton instance of Universal Invoice Processor"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = UniversalInvoiceProcessor()
    return _processor_instance

