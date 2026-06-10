"""
Cyrex Guard API Routes
REST API endpoints for Cyrex Guard - vendor fraud detection across 6 industries

Endpoints for:
- Vendor Invoice Intelligence (invoice processing)
- Market Rate Intelligence (pricing benchmarks)
- Industry Adapter Loader (LoRA management)
- Vendor Fraud Analyzer (fraud detection)
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from enum import Enum
import json

from ..services.invoice_parser import (
    get_universal_invoice_processor,
    ProcessedInvoice,
    InvoiceLineItem
)
from ..services.pricing_benchmark import (
    get_pricing_benchmark_engine,
    PricingBenchmark,
    PriceComparison,
    PricingTier
)
from ..services.lora_loader import (
    get_industry_lora_service,
    LoRAAdapterInfo,
    LoRAAdapterStatus
)
from ..services.fraud_detector import (
    get_universal_fraud_detection_service,
    FraudDetectionResult,
    FraudIndicator
)
from ..core.types import IndustryNiche, VendorFraudType, RiskLevel
from ..logging_config import get_logger

logger = get_logger("cyrex.api.cyrex_guard")

router = APIRouter(prefix="/cyrex-guard", tags=["cyrex-guard"])


# =============================================================================
# Request/Response Models
# =============================================================================

class IndustryEnum(str, Enum):
    """Industry choices for API"""
    property_management = "property_management"
    corporate_procurement = "corporate_procurement"
    insurance_pc = "insurance_pc"
    general_contractors = "general_contractors"
    retail_ecommerce = "retail_ecommerce"
    law_firms = "law_firms"
    generic = "generic"


def industry_enum_to_niche(industry: IndustryEnum) -> IndustryNiche:
    """Convert API enum to IndustryNiche"""
    mapping = {
        IndustryEnum.property_management: IndustryNiche.PROPERTY_MANAGEMENT,
        IndustryEnum.corporate_procurement: IndustryNiche.CORPORATE_PROCUREMENT,
        IndustryEnum.insurance_pc: IndustryNiche.INSURANCE_PC,
        IndustryEnum.general_contractors: IndustryNiche.GENERAL_CONTRACTORS,
        IndustryEnum.retail_ecommerce: IndustryNiche.RETAIL_ECOMMERCE,
        IndustryEnum.law_firms: IndustryNiche.LAW_FIRMS,
        IndustryEnum.generic: IndustryNiche.GENERIC,
    }
    return mapping.get(industry, IndustryNiche.GENERIC)


# =============================================================================
# Universal Invoice Processing API
# =============================================================================

class ProcessInvoiceRequest(BaseModel):
    """Request to process an invoice"""
    invoice_content: str = Field(..., description="Invoice content (text, JSON, or base64)")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    invoice_format: str = Field("text", description="Format: text, json, pdf, image")
    use_lora: bool = Field(True, description="Use industry-specific LoRA adapter")


class ProcessInvoiceResponse(BaseModel):
    """Response from invoice processing"""
    success: bool
    invoice_id: str
    vendor_name: str
    vendor_id: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_amount: float
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    service_category: Optional[str] = None
    industry: str
    extraction_confidence: float
    extraction_method: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchProcessInvoicesRequest(BaseModel):
    """Request to process multiple invoices"""
    invoices: List[Dict[str, Any]] = Field(..., description="List of invoice data")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    use_lora: bool = Field(True, description="Use industry-specific LoRA adapter")


@router.post("/invoice/process", response_model=ProcessInvoiceResponse)
async def process_invoice(request: ProcessInvoiceRequest):
    """
    Process a single invoice
    
    Extracts structured data from invoice using OCR + multimodal AI
    """
    try:
        processor = get_universal_invoice_processor()
        industry_niche = industry_enum_to_niche(request.industry)
        
        processed = await processor.process_invoice(
            invoice_content=request.invoice_content,
            industry=industry_niche,
            invoice_format=request.invoice_format,
            use_lora=request.use_lora
        )
        
        return ProcessInvoiceResponse(
            success=True,
            invoice_id=processed.invoice_id,
            vendor_name=processed.vendor_name,
            vendor_id=processed.vendor_id,
            invoice_number=processed.invoice_number,
            invoice_date=processed.invoice_date.isoformat() if processed.invoice_date else None,
            total_amount=processed.total_amount,
            line_items=[
                {
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total": item.total,
                    "service_category": item.service_category
                }
                for item in processed.line_items
            ],
            service_category=processed.service_category,
            industry=processed.industry.value,
            extraction_confidence=processed.extraction_confidence,
            extraction_method=processed.extraction_method,
            metadata=processed.metadata
        )
    except Exception as e:
        logger.error(f"Invoice processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Invoice processing failed: {str(e)}")


@router.post("/invoice/process-batch")
async def process_invoices_batch(request: BatchProcessInvoicesRequest):
    """
    Process multiple invoices in batch
    """
    try:
        processor = get_universal_invoice_processor()
        industry_niche = industry_enum_to_niche(request.industry)
        
        processed = await processor.process_invoices_batch(
            invoices=request.invoices,
            industry=industry_niche,
            use_lora=request.use_lora
        )
        
        return {
            "success": True,
            "processed_count": len(processed),
            "invoices": [
                {
                    "invoice_id": inv.invoice_id,
                    "vendor_name": inv.vendor_name,
                    "total_amount": inv.total_amount,
                    "extraction_confidence": inv.extraction_confidence
                }
                for inv in processed
            ]
        }
    except Exception as e:
        logger.error(f"Batch invoice processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# =============================================================================
# Pricing Benchmark API
# =============================================================================

class ComparePriceRequest(BaseModel):
    """Request to compare price against benchmark"""
    invoice_price: float = Field(..., description="Price to compare")
    service_category: str = Field(..., description="Service category")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    location: Optional[str] = Field(None, description="Geographic location")
    invoice_date: Optional[str] = Field(None, description="Invoice date (ISO format)")


class GetBenchmarkRequest(BaseModel):
    """Request to get pricing benchmark"""
    service_category: str = Field(..., description="Service category")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    location: Optional[str] = Field(None, description="Geographic location")


@router.post("/pricing/compare")
async def compare_price(request: ComparePriceRequest):
    """
    Compare invoice price against market benchmarks
    """
    try:
        engine = get_pricing_benchmark_engine()
        industry_niche = industry_enum_to_niche(request.industry)
        
        invoice_date = None
        if request.invoice_date:
            from dateutil import parser
            invoice_date = parser.parse(request.invoice_date)
        
        comparison = await engine.compare_price(
            invoice_price=request.invoice_price,
            service_category=request.service_category,
            industry=industry_niche,
            location=request.location,
            invoice_date=invoice_date
        )
        
        return {
            "success": True,
            "invoice_price": comparison.invoice_price,
            "deviation_percent": comparison.deviation_percent,
            "deviation_amount": comparison.deviation_amount,
            "tier": comparison.tier.value,
            "is_overpriced": comparison.is_overpriced,
            "confidence": comparison.confidence,
            "benchmark": {
                "median_price": comparison.benchmark.median_price,
                "mean_price": comparison.benchmark.mean_price,
                "percentile_25": comparison.benchmark.percentile_25,
                "percentile_75": comparison.benchmark.percentile_75,
                "percentile_90": comparison.benchmark.percentile_90,
                "sample_count": comparison.benchmark.sample_count
            },
            "recommendations": comparison.recommendations
        }
    except Exception as e:
        logger.error(f"Price comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Price comparison failed: {str(e)}")


@router.post("/pricing/benchmark")
async def get_benchmark(request: GetBenchmarkRequest):
    """
    Get pricing benchmark for service category
    """
    try:
        engine = get_pricing_benchmark_engine()
        industry_niche = industry_enum_to_niche(request.industry)
        
        benchmark = await engine.get_benchmark(
            service_category=request.service_category,
            industry=industry_niche,
            location=request.location
        )
        
        if not benchmark:
            raise HTTPException(status_code=404, detail="Benchmark not found - insufficient data")
        
        return {
            "success": True,
            "service_category": benchmark.service_category,
            "industry": benchmark.industry.value,
            "location": benchmark.location,
            "median_price": benchmark.median_price,
            "mean_price": benchmark.mean_price,
            "min_price": benchmark.min_price,
            "max_price": benchmark.max_price,
            "percentile_25": benchmark.percentile_25,
            "percentile_75": benchmark.percentile_75,
            "percentile_90": benchmark.percentile_90,
            "sample_count": benchmark.sample_count,
            "date_range_start": benchmark.date_range_start.isoformat() if benchmark.date_range_start else None,
            "date_range_end": benchmark.date_range_end.isoformat() if benchmark.date_range_end else None,
            "last_updated": benchmark.last_updated.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get benchmark failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Get benchmark failed: {str(e)}")


# =============================================================================
# Industry LoRA Adapter API
# =============================================================================

class LoadLoRARequest(BaseModel):
    """Request to load LoRA adapter"""
    industry: IndustryEnum = Field(..., description="Industry for LoRA adapter")
    adapter_id: Optional[str] = Field(None, description="Specific adapter ID (optional)")
    force_reload: bool = Field(False, description="Force reload even if already loaded")


@router.post("/lora/load")
async def load_lora_adapter(request: LoadLoRARequest):
    """
    Load industry-specific LoRA adapter
    """
    try:
        service = get_industry_lora_service()
        industry_niche = industry_enum_to_niche(request.industry)
        
        success = await service.load_adapter(
            industry=industry_niche,
            adapter_id=request.adapter_id,
            force_reload=request.force_reload
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="LoRA adapter not found or failed to load")
        
        adapter_info = await service.get_adapter_status(industry_niche)
        
        return {
            "success": True,
            "industry": request.industry.value,
            "adapter_id": adapter_info.adapter_id if adapter_info else None,
            "status": adapter_info.status.value if adapter_info else "loaded",
            "loaded_at": adapter_info.loaded_at.isoformat() if adapter_info and adapter_info.loaded_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load LoRA adapter failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Load LoRA adapter failed: {str(e)}")


@router.post("/lora/unload")
async def unload_lora_adapter(industry: IndustryEnum):
    """
    Unload industry-specific LoRA adapter
    """
    try:
        service = get_industry_lora_service()
        industry_niche = industry_enum_to_niche(industry)
        
        success = await service.unload_adapter(industry_niche)
        
        return {
            "success": success,
            "industry": industry.value
        }
    except Exception as e:
        logger.error(f"Unload LoRA adapter failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unload LoRA adapter failed: {str(e)}")


@router.get("/lora/status")
async def get_lora_status(industry: IndustryEnum):
    """
    Get LoRA adapter status for industry
    """
    try:
        service = get_industry_lora_service()
        industry_niche = industry_enum_to_niche(industry)
        
        adapter_info = await service.get_adapter_status(industry_niche)
        is_loaded = await service.is_adapter_loaded(industry_niche)
        
        if not adapter_info:
            return {
                "success": True,
                "industry": industry.value,
                "status": "not_loaded",
                "is_loaded": False
            }
        
        return {
            "success": True,
            "industry": industry.value,
            "adapter_id": adapter_info.adapter_id,
            "adapter_name": adapter_info.adapter_name,
            "version": adapter_info.version,
            "status": adapter_info.status.value,
            "is_loaded": is_loaded,
            "loaded_at": adapter_info.loaded_at.isoformat() if adapter_info.loaded_at else None,
            "base_model": adapter_info.base_model
        }
    except Exception as e:
        logger.error(f"Get LoRA status failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Get LoRA status failed: {str(e)}")


@router.get("/lora/list")
async def list_lora_adapters():
    """
    List all available LoRA adapters
    """
    try:
        service = get_industry_lora_service()
        adapters = await service.list_available_adapters()
        
        return {
            "success": True,
            "adapters": [
                {
                    "industry": adapter.industry.value,
                    "adapter_id": adapter.adapter_id,
                    "adapter_name": adapter.adapter_name,
                    "version": adapter.version,
                    "status": adapter.status.value,
                    "base_model": adapter.base_model
                }
                for adapter in adapters
            ]
        }
    except Exception as e:
        logger.error(f"List LoRA adapters failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"List LoRA adapters failed: {str(e)}")


# =============================================================================
# Universal Fraud Detection API
# =============================================================================

class DetectFraudRequest(BaseModel):
    """Request to detect fraud in invoice"""
    invoice: ProcessInvoiceResponse = Field(..., description="Processed invoice")
    use_vendor_intelligence: bool = Field(True, description="Use vendor history")
    use_pricing_benchmark: bool = Field(True, description="Compare against benchmarks")
    use_anomaly_detection: bool = Field(True, description="Use ML anomaly detection")
    use_pattern_matching: bool = Field(True, description="Use rule-based patterns")


@router.post("/fraud/detect")
async def detect_fraud(request: DetectFraudRequest):
    """
    Detect fraud in processed invoice
    
    Uses multiple detection methods:
    - Pricing benchmark comparison
    - Vendor intelligence (cross-industry patterns)
    - Anomaly detection (ML)
    - Pattern matching (rules)
    """
    try:
        fraud_service = get_universal_fraud_detection_service()
        processor = get_universal_invoice_processor()
        
        # Convert API response back to ProcessedInvoice
        # (In production, would have better serialization)
        industry_niche = industry_enum_to_niche(IndustryEnum(request.invoice.industry))
        
        # Reconstruct ProcessedInvoice from response
        # For now, we'll need to process the invoice again or store it
        # This is a simplified version - in production would cache processed invoices
        
        # Create a minimal ProcessedInvoice for fraud detection
        from ..services.invoice_parser import ProcessedInvoice, InvoiceLineItem
        
        line_items = [
            InvoiceLineItem(
                description=item.get("description", ""),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total=item.get("total"),
                service_category=item.get("service_category")
            )
            for item in request.invoice.line_items
        ]
        
        invoice_date = None
        if request.invoice.invoice_date:
            from dateutil import parser
            invoice_date = parser.parse(request.invoice.invoice_date)
        
        processed_invoice = ProcessedInvoice(
            invoice_id=request.invoice.invoice_id,
            vendor_name=request.invoice.vendor_name,
            vendor_id=request.invoice.vendor_id,
            invoice_number=request.invoice.invoice_number,
            invoice_date=invoice_date,
            total_amount=request.invoice.total_amount,
            line_items=line_items,
            service_category=request.invoice.service_category,
            industry=industry_niche,
            extraction_confidence=request.invoice.extraction_confidence,
            extraction_method=request.invoice.extraction_method,
            metadata=request.invoice.metadata
        )
        
        result = await fraud_service.detect_fraud(
            invoice=processed_invoice,
            use_vendor_intelligence=request.use_vendor_intelligence,
            use_pricing_benchmark=request.use_pricing_benchmark,
            use_anomaly_detection=request.use_anomaly_detection,
            use_pattern_matching=request.use_pattern_matching
        )
        
        return {
            "success": True,
            "invoice_id": result.invoice_id,
            "vendor_id": result.vendor_id,
            "fraud_detected": result.fraud_detected,
            "risk_score": result.risk_score,
            "risk_level": result.risk_level.value,
            "fraud_indicators": [
                {
                    "fraud_type": ind.fraud_type.value,
                    "severity": ind.severity,
                    "confidence": ind.confidence,
                    "description": ind.description,
                    "evidence": ind.evidence,
                    "recommendation": ind.recommendation
                }
                for ind in result.fraud_indicators
            ],
            "price_comparison": {
                "deviation_percent": result.price_comparison.deviation_percent if result.price_comparison else None,
                "is_overpriced": result.price_comparison.is_overpriced if result.price_comparison else None,
                "tier": result.price_comparison.tier.value if result.price_comparison else None
            } if result.price_comparison else None,
            "recommendations": result.recommendations,
            "confidence": result.confidence
        }
    except Exception as e:
        logger.error(f"Fraud detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fraud detection failed: {str(e)}")


@router.post("/fraud/detect-full")
async def detect_fraud_full(
    invoice_content: str = Form(...),
    industry: IndustryEnum = Form(IndustryEnum.property_management),
    invoice_format: str = Form("text"),
    use_lora: bool = Form(True),
    use_vendor_intelligence: bool = Form(True),
    use_pricing_benchmark: bool = Form(True),
    use_anomaly_detection: bool = Form(True),
    use_pattern_matching: bool = Form(True)
):
    """
    Full fraud detection pipeline: process invoice + detect fraud
    
    This endpoint combines invoice processing and fraud detection in one call
    """
    try:
        processor = get_universal_invoice_processor()
        fraud_service = get_universal_fraud_detection_service()
        industry_niche = industry_enum_to_niche(industry)
        
        # Process invoice
        processed = await processor.process_invoice(
            invoice_content=invoice_content,
            industry=industry_niche,
            invoice_format=invoice_format,
            use_lora=use_lora
        )
        
        # Detect fraud
        result = await fraud_service.detect_fraud(
            invoice=processed,
            use_vendor_intelligence=use_vendor_intelligence,
            use_pricing_benchmark=use_pricing_benchmark,
            use_anomaly_detection=use_anomaly_detection,
            use_pattern_matching=use_pattern_matching
        )
        
        return {
            "success": True,
            "invoice": {
                "invoice_id": processed.invoice_id,
                "vendor_name": processed.vendor_name,
                "total_amount": processed.total_amount,
                "service_category": processed.service_category
            },
            "fraud_detection": {
                "fraud_detected": result.fraud_detected,
                "risk_score": result.risk_score,
                "risk_level": result.risk_level.value,
                "fraud_indicators_count": len(result.fraud_indicators),
                "recommendations": result.recommendations
            }
        }
    except Exception as e:
        logger.error(f"Full fraud detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Full fraud detection failed: {str(e)}")

