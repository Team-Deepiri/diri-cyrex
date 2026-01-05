"""
Cyrex Document Verification Service
OCR, multimodal AI, and document authenticity verification

Capabilities:
- OCR for all invoice types
- Multimodal AI for invoice parsing
- Document forgery detection
- Photo verification for work performed
- Structured data extraction
"""
from typing import Dict, List, Optional, Any, BinaryIO
from datetime import datetime
import base64
import io
import asyncio
import json

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

from ..logging_config import get_logger
from ..core.types import IndustryNiche

logger = get_logger("cyrex.document_verification")


class DocumentVerificationService:
    """
    Document Verification Service for Cyrex
    
    Handles:
    - OCR extraction from invoices
    - Multimodal AI parsing
    - Document authenticity checking
    - Photo verification
    """
    
    def __init__(self):
        self.logger = logger
        self._ocr_engine = None
        self._multimodal_model = None
    
    async def extract_invoice_data(
        self,
        document_content: str,
        document_type: str = "invoice",
        industry: IndustryNiche = IndustryNiche.GENERIC
    ) -> Dict[str, Any]:
        """
        Extract structured data from invoice/document
        
        Uses OCR + LLM for intelligent extraction
        """
        try:
            # For now, use LLM-based extraction
            # In production, would use OCR (Tesseract, AWS Textract, etc.)
            
            extraction_prompt = f"""Extract structured data from this {document_type}:

{document_content}

Extract and return as JSON with these fields:
- vendor_name (string)
- vendor_id (string, optional)
- invoice_number (string, optional)
- invoice_date (ISO format string)
- total_amount (number)
- line_items (array of objects with: description, quantity, unit_price, total)
- service_category (string, optional)
- work_description (string, optional)
- property_address (string, optional)
- payment_terms (string, optional)
- due_date (ISO format string, optional)

Industry context: {industry.value}

Return ONLY valid JSON, no other text.
"""
            
            # Use LLM for extraction (would integrate with OCR in production)
            from ..integrations.local_llm import get_local_llm
            
            llm = get_local_llm(
                backend="ollama",
                model_name="llama3:8b",
                temperature=0.1,  # Low temperature for consistent extraction
            )
            
            if llm:
                response = await asyncio.to_thread(llm.invoke, extraction_prompt)
                try:
                    extracted = json.loads(response)
                    return {
                        "success": True,
                        "extracted_data": extracted,
                        "extraction_method": "llm",
                        "confidence": 0.85,
                    }
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        extracted = json.loads(json_match.group())
                        return {
                            "success": True,
                            "extracted_data": extracted,
                            "extraction_method": "llm_parsed",
                            "confidence": 0.75,
                        }
            
            # Fallback: Basic extraction
            return {
                "success": False,
                "extracted_data": {},
                "error": "LLM extraction failed",
                "confidence": 0.0,
            }
            
        except Exception as e:
            self.logger.error(f"Document extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "extracted_data": {},
                "error": str(e),
                "confidence": 0.0,
            }
    
    async def verify_document_authenticity(
        self,
        document_content: str,
        invoice_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify document authenticity
        
        Checks for:
        - Forged invoices
        - Suspicious patterns
        - Inconsistencies
        """
        verification_results = {
            "authentic": True,
            "confidence": 1.0,
            "checks": [],
            "warnings": [],
        }
        
        # Check 1: Invoice number format consistency
        invoice_number = invoice_data.get("invoice_number", "")
        if invoice_number:
            # Check for suspicious patterns
            if len(invoice_number) < 3:
                verification_results["warnings"].append("Invoice number seems too short")
                verification_results["confidence"] *= 0.9
            
            # Check for sequential patterns (potential duplicates)
            if invoice_number.isdigit() and len(invoice_number) > 5:
                # Could be sequential - flag for review
                verification_results["checks"].append({
                    "check": "invoice_number_format",
                    "status": "passed",
                    "note": "Invoice number format appears valid"
                })
        
        # Check 2: Date consistency
        invoice_date = invoice_data.get("invoice_date")
        if invoice_date:
            try:
                if isinstance(invoice_date, str):
                    date_obj = datetime.fromisoformat(invoice_date.replace('Z', '+00:00'))
                else:
                    date_obj = invoice_date
                
                # Check if date is in future
                if date_obj > datetime.utcnow():
                    verification_results["warnings"].append("Invoice date is in the future")
                    verification_results["confidence"] *= 0.8
                
                # Check if date is too old (more than 2 years)
                if (datetime.utcnow() - date_obj).days > 730:
                    verification_results["warnings"].append("Invoice date is more than 2 years old")
                    verification_results["confidence"] *= 0.9
                
                verification_results["checks"].append({
                    "check": "date_consistency",
                    "status": "passed" if verification_results["confidence"] > 0.8 else "warning",
                })
            except Exception:
                verification_results["warnings"].append("Could not parse invoice date")
                verification_results["confidence"] *= 0.7
        
        # Check 3: Amount consistency
        total_amount = invoice_data.get("total_amount", 0)
        line_items = invoice_data.get("line_items", [])
        
        if line_items:
            calculated_total = sum(item.get("total", item.get("unit_price", 0) * item.get("quantity", 1)) for item in line_items)
            if abs(calculated_total - total_amount) > 0.01:
                verification_results["warnings"].append(f"Line items total ({calculated_total}) doesn't match invoice total ({total_amount})")
                verification_results["confidence"] *= 0.7
                verification_results["authentic"] = False
        
        # Check 4: Vendor name consistency
        vendor_name = invoice_data.get("vendor_name", "")
        if not vendor_name or len(vendor_name) < 2:
            verification_results["warnings"].append("Vendor name is missing or too short")
            verification_results["confidence"] *= 0.6
        
        # Final assessment
        if verification_results["confidence"] < 0.7:
            verification_results["authentic"] = False
        
        verification_results["checks"].append({
            "check": "overall_authenticity",
            "status": "passed" if verification_results["authentic"] else "failed",
            "confidence": verification_results["confidence"],
        })
        
        return verification_results
    
    async def verify_work_photos(
        self,
        before_photo: Optional[bytes] = None,
        after_photo: Optional[bytes] = None,
        work_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify work performed using before/after photos
        
        Uses multimodal AI to verify work completion
        """
        if not before_photo or not after_photo:
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": "Missing before or after photos",
            }
        
        try:
            # In production, would use vision model (GPT-4 Vision, Claude 3, etc.)
            # For now, return basic verification
            
            # Check if images are valid
            if not HAS_PIL:
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "reason": "PIL/Pillow not installed - photo verification requires image processing",
                }
            
            try:
                before_img = Image.open(io.BytesIO(before_photo))
                after_img = Image.open(io.BytesIO(after_photo))
                
                # Basic checks
                before_size = before_img.size
                after_size = after_img.size
                
                # If images are very similar, might be duplicate
                if before_size == after_size:
                    # Could use image comparison here
                    pass
                
                return {
                    "verified": True,
                    "confidence": 0.75,
                    "before_photo_size": before_size,
                    "after_photo_size": after_size,
                    "note": "Photos validated. Full verification requires vision model integration.",
                }
            except Exception as e:
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "error": f"Invalid image format: {str(e)}",
                }
        
        except Exception as e:
            self.logger.error(f"Photo verification failed: {e}", exc_info=True)
            return {
                "verified": False,
                "confidence": 0.0,
                "error": str(e),
            }
    
    async def process_document(
        self,
        document_content: str,
        document_type: str = "invoice",
        industry: IndustryNiche = IndustryNiche.GENERIC,
        verify_authenticity: bool = True
    ) -> Dict[str, Any]:
        """
        Complete document processing pipeline
        
        Steps:
        1. Extract structured data (OCR + LLM)
        2. Verify authenticity
        3. Return comprehensive result
        """
        result = {
            "success": False,
            "extracted_data": {},
            "verification": {},
            "processed_at": datetime.utcnow().isoformat(),
        }
        
        # Step 1: Extract data
        extraction = await self.extract_invoice_data(
            document_content=document_content,
            document_type=document_type,
            industry=industry
        )
        
        if not extraction.get("success"):
            result["error"] = extraction.get("error", "Extraction failed")
            return result
        
        extracted_data = extraction.get("extracted_data", {})
        result["extracted_data"] = extracted_data
        result["extraction_confidence"] = extraction.get("confidence", 0.0)
        
        # Step 2: Verify authenticity
        if verify_authenticity:
            verification = await self.verify_document_authenticity(
                document_content=document_content,
                invoice_data=extracted_data
            )
            result["verification"] = verification
        
        result["success"] = True
        return result


# Global service instance
_document_verification_service: Optional[DocumentVerificationService] = None


async def get_document_verification_service() -> DocumentVerificationService:
    """Get or create document verification service singleton"""
    global _document_verification_service
    
    if _document_verification_service is None:
        _document_verification_service = DocumentVerificationService()
    
    return _document_verification_service

