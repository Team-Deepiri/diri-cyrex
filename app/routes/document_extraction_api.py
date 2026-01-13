"""
Document Extraction API Routes
FastAPI endpoints for extracting text from documents (PDF, DOCX, images)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..logging_config import get_logger
from ..services.document_verification_service import DocumentVerificationService

logger = get_logger("cyrex.api.document_extraction")

router = APIRouter(prefix="/document-extraction", tags=["Document Extraction"])


class ExtractTextRequest(BaseModel):
    documentUrl: str
    documentType: Optional[str] = "pdf"  # pdf, docx, image, text


@router.post("/extract-text")
async def extract_text(request: ExtractTextRequest):
    """Extract text from document (PDF, DOCX, image)"""
    try:
        document_service = DocumentVerificationService()
        
        # Extract text based on document type
        if request.documentType == "pdf" or request.documentType is None:
            result = await document_service.extract_text_from_pdf(request.documentUrl)
        elif request.documentType == "docx" or request.documentType == "doc":
            result = await document_service.extract_text_from_docx(request.documentUrl)
        elif request.documentType == "image":
            result = await document_service.extract_text_from_image(request.documentUrl)
        else:
            # Try PDF as default
            result = await document_service.extract_text_from_pdf(request.documentUrl)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Text extraction failed")
            )
        
        return {
            "success": True,
            "text": result.get("text", ""),
            "documentUrl": request.documentUrl,
            "documentType": request.documentType or "pdf",
            "extractionMethod": result.get("extraction_method", "unknown"),
            "pageCount": result.get("page_count"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text extraction error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

