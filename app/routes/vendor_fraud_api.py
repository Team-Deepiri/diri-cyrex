"""
Vendor Fraud Detection API Routes
REST API endpoints for Cyrex vendor fraud detection system
"""
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
from dataclasses import asdict

from ..agents.implementations.vendor_fraud_agent import VendorFraudAgent
from ..agents.agent_factory import AgentFactory
from ..core.types import AgentConfig, AgentRole, IndustryNiche, RiskLevel
from ..integrations.universal_rag_engine import create_universal_rag_engine
from ..logging_config import get_logger

logger = get_logger("cyrex.api.vendor_fraud")

router = APIRouter(prefix="/vendor-fraud", tags=["vendor-fraud"])


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


class LineItem(BaseModel):
    """Invoice line item"""
    description: str
    quantity: float = 1.0
    unit_price: float
    total: Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.total is None:
            self.total = self.quantity * self.unit_price


class InvoiceData(BaseModel):
    """Invoice data for analysis"""
    vendor_name: str = Field(..., description="Name of the vendor")
    vendor_id: Optional[str] = Field(None, description="Unique vendor identifier")
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[str] = Field(None, description="Invoice date (ISO format)")
    total_amount: float = Field(..., description="Total invoice amount")
    line_items: List[LineItem] = Field(default_factory=list, description="Invoice line items")
    service_category: Optional[str] = Field(None, description="Service category (e.g., hvac_repair)")
    property_address: Optional[str] = Field(None, description="Property address if applicable")
    work_description: Optional[str] = Field(None, description="Description of work performed")


class AnalyzeInvoiceRequest(BaseModel):
    """Request to analyze an invoice"""
    invoice: InvoiceData
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    include_vendor_history: bool = Field(True, description="Include vendor history in analysis")
    session_id: Optional[str] = Field(None, description="Session ID for context")


class VendorProfileRequest(BaseModel):
    """Request for vendor profile"""
    vendor_id: str = Field(..., description="Vendor identifier")
    vendor_name: Optional[str] = Field(None, description="Vendor name")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")


class PricingBenchmarkRequest(BaseModel):
    """Request for pricing benchmark"""
    service_type: str = Field(..., description="Service type (e.g., hvac_repair, plumbing_repair)")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    price_to_check: Optional[float] = Field(None, description="Price to compare against benchmark")


class DocumentIngestionRequest(BaseModel):
    """Request to ingest a document"""
    content: str = Field(..., description="Document content")
    title: str = Field(..., description="Document title")
    doc_type: str = Field("vendor_invoice", description="Document type")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryKnowledgeBaseRequest(BaseModel):
    """Request to query knowledge base"""
    query: str = Field(..., description="Query string")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    top_k: int = Field(5, description="Number of results to return")


class ChatRequest(BaseModel):
    """Chat request for conversational interaction"""
    message: str = Field(..., description="User message")
    industry: IndustryEnum = Field(IndustryEnum.property_management, description="Industry context")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


# =============================================================================
# Agent Management
# =============================================================================

# Cache for agent instances
_agent_cache: Dict[str, VendorFraudAgent] = {}


async def get_vendor_fraud_agent(
    industry: IndustryNiche = IndustryNiche.PROPERTY_MANAGEMENT,
    session_id: Optional[str] = None
) -> VendorFraudAgent:
    """Get or create a VendorFraudAgent for the specified industry"""
    cache_key = f"{industry.value}_{session_id or 'default'}"
    
    if cache_key not in _agent_cache:
        # Create agent configuration
        config = AgentConfig(
            role=AgentRole.FRAUD_DETECTOR,
            name=f"Cyrex Vendor Fraud Detector ({industry.value})",
            description="Detects vendor fraud across multiple industries",
            capabilities=[
                "invoice_analysis",
                "vendor_intelligence",
                "pricing_benchmarks",
                "fraud_detection",
                "risk_assessment",
            ],
            tools=[
                "analyze_invoice",
                "get_vendor_profile",
                "check_pricing_benchmark",
                "query_knowledge_base",
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=4000,
        )
        
        # Create agent
        agent = VendorFraudAgent(
            agent_config=config,
            llm_provider=None,  # Will use default Ollama
            session_id=session_id,
            industry=industry,
        )
        
        _agent_cache[cache_key] = agent
        logger.info(f"Created VendorFraudAgent", industry=industry.value, session_id=session_id)
    
    return _agent_cache[cache_key]


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/analyze-invoice")
async def analyze_invoice(request: AnalyzeInvoiceRequest):
    """
    Analyze an invoice for potential fraud
    
    This endpoint performs comprehensive fraud analysis including:
    - Price comparison against market benchmarks
    - Duplicate invoice detection
    - Vendor history review
    - Risk score calculation
    - Recommendations for action
    """
    logger.info("Invoice analysis requested",
               vendor=request.invoice.vendor_name,
               industry=request.industry.value)
    
    try:
        # Get agent for industry
        industry_niche = IndustryNiche(request.industry.value)
        agent = await get_vendor_fraud_agent(industry_niche, request.session_id)
        
        # Convert invoice to dict
        invoice_dict = {
            "vendor_name": request.invoice.vendor_name,
            "vendor_id": request.invoice.vendor_id,
            "invoice_number": request.invoice.invoice_number,
            "invoice_date": request.invoice.invoice_date,
            "total_amount": request.invoice.total_amount,
            "line_items": [
                {
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total": item.total,
                }
                for item in request.invoice.line_items
            ],
            "service_category": request.invoice.service_category,
            "property_address": request.invoice.property_address,
            "work_description": request.invoice.work_description,
        }
        
        # Analyze invoice
        result = await agent.analyze_invoice(
            invoice_data=invoice_dict,
            industry=industry_niche,
        )
        
        return {
            "success": True,
            "analysis": result,
            "analyzed_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Invoice analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vendor-profile")
async def get_vendor_profile(request: VendorProfileRequest):
    """
    Get vendor profile and intelligence
    
    Returns vendor history, risk score, and fraud indicators
    """
    logger.info("Vendor profile requested", vendor_id=request.vendor_id)
    
    try:
        industry_niche = IndustryNiche(request.industry.value)
        agent = await get_vendor_fraud_agent(industry_niche)
        
        # Query vendor profile
        result = await agent._tool_get_vendor_profile(request.vendor_id)
        
        return {
            "success": True,
            "vendor_profile": result,
            "retrieved_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Vendor profile request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pricing-benchmark")
async def check_pricing_benchmark(request: PricingBenchmarkRequest):
    """
    Check pricing against market benchmarks
    
    Returns market rate data for the specified service type
    """
    logger.info("Pricing benchmark requested",
               service_type=request.service_type,
               industry=request.industry.value)
    
    try:
        from ..agents.tools.vendor_fraud_tools import get_pricing_benchmark
        
        result = await get_pricing_benchmark(
            service_type=request.service_type,
            industry=request.industry.value,
        )
        
        # If price provided, calculate deviation
        if request.price_to_check and result.get("found") and result.get("benchmark"):
            benchmark = result["benchmark"]
            avg_price = benchmark.get("avg", 0)
            if avg_price > 0:
                deviation = ((request.price_to_check - avg_price) / avg_price) * 100
                result["price_checked"] = request.price_to_check
                result["deviation_percent"] = round(deviation, 1)
                result["status"] = "above_market" if request.price_to_check > benchmark.get("max", 0) else (
                    "below_market" if request.price_to_check < benchmark.get("min", 0) else "within_range"
                )
        
        return {
            "success": True,
            "benchmark": result,
            "retrieved_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Pricing benchmark request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest-document")
async def ingest_document(request: DocumentIngestionRequest):
    """
    Ingest a document into the knowledge base
    
    Documents are indexed in Milvus for RAG retrieval
    """
    logger.info("Document ingestion requested",
               title=request.title,
               industry=request.industry.value)
    
    try:
        industry_niche = IndustryNiche(request.industry.value)
        
        # Get or create RAG engine
        rag_engine = create_universal_rag_engine(
            industry=industry_niche,
            collection_name=f"cyrex_{industry_niche.value}_vendor_fraud"
        )
        
        # Create document
        from ..integrations.universal_rag_engine import Document, DocumentType
        import uuid
        
        doc = Document(
            id=str(uuid.uuid4()),
            content=request.content,
            doc_type=DocumentType(request.doc_type) if hasattr(DocumentType, request.doc_type.upper()) else "other",
            industry=industry_niche,
            title=request.title,
            source="api_ingestion",
            created_at=datetime.utcnow(),
            metadata=request.metadata,
        )
        
        # Index document
        success = rag_engine.index_document(doc)
        
        return {
            "success": success,
            "document_id": doc.id,
            "message": "Document indexed successfully" if success else "Failed to index document",
            "indexed_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_knowledge_base(request: QueryKnowledgeBaseRequest):
    """
    Query the knowledge base for relevant information
    
    Uses RAG to retrieve relevant documents and context
    """
    logger.info("Knowledge base query",
               query=request.query[:100],
               industry=request.industry.value)
    
    try:
        industry_niche = IndustryNiche(request.industry.value)
        agent = await get_vendor_fraud_agent(industry_niche)
        
        # Query RAG
        context = await agent._query_rag(request.query)
        
        return {
            "success": True,
            "query": request.query,
            "results": context if context else "No relevant documents found",
            "industry": request.industry.value,
            "retrieved_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Knowledge base query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Conversational interface with the fraud detection agent
    
    Send natural language queries and get intelligent responses
    """
    logger.info("Chat request",
               message=request.message[:100],
               industry=request.industry.value)
    
    try:
        industry_niche = IndustryNiche(request.industry.value)
        agent = await get_vendor_fraud_agent(industry_niche, request.session_id)
        
        # Invoke agent
        response = await agent.invoke(
            input_text=request.message,
            context=request.context,
            use_tools=True,
        )
        
        return {
            "success": True,
            "response": response.content,
            "confidence": response.confidence,
            "tool_calls": response.tool_calls,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/industries")
async def list_industries():
    """
    List supported industries
    
    Returns all industries supported by the vendor fraud detection system
    """
    industries = [
        {
            "id": "property_management",
            "name": "Property Management",
            "description": "HVAC, plumbing, electrical contractor fraud detection",
            "vendor_types": ["HVAC contractors", "Plumbers", "Electricians", "General maintenance"],
        },
        {
            "id": "corporate_procurement",
            "name": "Corporate Procurement",
            "description": "Supplier invoice and purchase order fraud detection",
            "vendor_types": ["IT services", "Consulting", "Office supplies", "Equipment vendors"],
        },
        {
            "id": "insurance_pc",
            "name": "P&C Insurance",
            "description": "Auto body shop and home repair contractor fraud",
            "vendor_types": ["Auto body shops", "Roofing contractors", "Restoration companies"],
        },
        {
            "id": "general_contractors",
            "name": "General Contractors",
            "description": "Subcontractor and material supplier fraud",
            "vendor_types": ["Electrical subs", "Plumbing subs", "Material suppliers"],
        },
        {
            "id": "retail_ecommerce",
            "name": "Retail & E-Commerce",
            "description": "Freight carrier and warehouse vendor fraud",
            "vendor_types": ["Freight carriers", "LTL carriers", "Warehouse vendors"],
        },
        {
            "id": "law_firms",
            "name": "Law Firms",
            "description": "Expert witness and e-discovery vendor fraud",
            "vendor_types": ["Expert witnesses", "E-discovery vendors", "Court reporters"],
        },
    ]
    
    return {
        "success": True,
        "industries": industries,
        "count": len(industries),
    }


@router.get("/health")
async def health_check():
    """
    Health check for vendor fraud detection service
    """
    return {
        "status": "healthy",
        "service": "Cyrex Vendor Fraud Detection",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/analytics")
async def get_analytics(
    industry: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get comprehensive analytics
    
    Returns:
    - Total invoices analyzed
    - Fraud detection rate
    - High-risk vendors
    - Cross-industry network effects
    - Total amount analyzed
    """
    try:
        from ..services.vendor_intelligence_service import get_vendor_intelligence_service
        
        intel_service = await get_vendor_intelligence_service()
        
        industry_niche = IndustryNiche(industry) if industry else None
        
        date_range = None
        if start_date and end_date:
            date_range = (
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date)
            )
        
        analytics = await intel_service.get_analytics(
            industry=industry_niche,
            date_range=date_range
        )
        
        return {
            "success": True,
            "analytics": analytics,
        }
        
    except Exception as e:
        logger.error(f"Analytics request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vendors")
async def list_vendors(
    query: Optional[str] = None,
    industry: Optional[str] = None,
    risk_level: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
):
    """
    Search and list vendors
    
    Supports filtering by:
    - Query (vendor name search)
    - Industry
    - Risk level
    - Status (active, flagged, blocked, monitored)
    """
    try:
        from ..services.vendor_intelligence_service import get_vendor_intelligence_service
        from ..core.types import RiskLevel
        from ..services.vendor_intelligence_service import VendorStatus
        
        intel_service = await get_vendor_intelligence_service()
        
        industry_niche = IndustryNiche(industry) if industry else None
        risk_level_enum = RiskLevel(risk_level) if risk_level else None
        status_enum = VendorStatus(status) if status else None
        
        vendors = await intel_service.search_vendors(
            query=query,
            industry=industry_niche,
            risk_level=risk_level_enum,
            status=status_enum,
            limit=limit
        )
        
        return {
            "success": True,
            "vendors": [asdict(v) for v in vendors],
            "count": len(vendors),
        }
        
    except Exception as e:
        logger.error(f"Vendor search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vendors/{vendor_id}")
async def get_vendor_details(vendor_id: str):
    """
    Get comprehensive vendor details including cross-industry intelligence
    """
    try:
        from ..services.vendor_intelligence_service import get_vendor_intelligence_service
        
        intel_service = await get_vendor_intelligence_service()
        
        vendor = await intel_service.get_vendor_profile(vendor_id)
        if not vendor:
            raise HTTPException(status_code=404, detail="Vendor not found")
        
        cross_industry = await intel_service.get_cross_industry_vendors(vendor_id)
        
        return {
            "success": True,
            "vendor": asdict(vendor),
            "cross_industry_intelligence": cross_industry,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vendor details request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-document")
async def verify_document(
    document_content: str = Form(...),
    document_type: str = Form("invoice"),
    industry: str = Form("property_management"),
    verify_authenticity: bool = Form(True)
):
    """
    Verify document authenticity and extract data
    
    Uses OCR + multimodal AI for document processing
    """
    try:
        from ..services.document_verification_service import get_document_verification_service
        
        verification_service = await get_document_verification_service()
        industry_niche = IndustryNiche(industry)
        
        result = await verification_service.process_document(
            document_content=document_content,
            document_type=document_type,
            industry=industry_niche,
            verify_authenticity=verify_authenticity
        )
        
        return {
            "success": result.get("success", False),
            "extracted_data": result.get("extracted_data", {}),
            "verification": result.get("verification", {}),
            "extraction_confidence": result.get("extraction_confidence", 0.0),
        }
        
    except Exception as e:
        logger.error(f"Document verification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

