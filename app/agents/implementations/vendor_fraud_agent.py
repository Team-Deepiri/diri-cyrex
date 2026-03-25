"""
Cyrex Vendor Fraud Detection Agent
LangGraph-based multi-agent workflow for detecting vendor fraud across industries

This agent analyzes invoices, detects fraud patterns, and provides vendor intelligence
using RAG pipeline with Milvus for document-based expertise.
"""
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
import asyncio
import operator
import json

from ..base_agent import BaseAgent, AgentResponse
from ...core.types import (
    AgentConfig, AgentRole, AgentStatus, MemoryType,
    IndustryNiche, VendorFraudType, RiskLevel
)
from ...logging_config import get_logger

logger = get_logger("cyrex.agent.vendor_fraud")

# LangGraph imports
HAS_LANGGRAPH = False
try:
    from langgraph.graph import StateGraph, END, START
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    HAS_LANGGRAPH = True
except ImportError:
    logger.warning("LangGraph not available, using fallback mode")
    StateGraph = None
    END = None
    START = None
    BaseMessage = dict
    HumanMessage = dict
    AIMessage = dict
    SystemMessage = dict


# =============================================================================
# State Definition for LangGraph Workflow
# =============================================================================

class VendorFraudState(TypedDict):
    """State structure for vendor fraud detection workflow"""
    messages: List[Any]
    workflow_id: str
    session_id: Optional[str]
    industry: str
    # Input data
    invoice_data: Optional[Dict[str, Any]]
    vendor_id: Optional[str]
    document_content: Optional[str]
    query: str
    # Processing results
    extracted_data: Optional[Dict[str, Any]]
    pricing_analysis: Optional[Dict[str, Any]]
    vendor_profile: Optional[Dict[str, Any]]
    fraud_indicators: List[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    # Final output
    fraud_detected: bool
    risk_level: str
    recommendations: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


# =============================================================================
# Vendor Fraud Agent Implementation
# =============================================================================

class VendorFraudAgent(BaseAgent):
    """
    Cyrex Vendor Fraud Detection Agent
    
    Capabilities:
    - Invoice fraud detection (overpricing, duplicates, phantom work)
    - Vendor intelligence and risk scoring
    - Pricing benchmark analysis
    - Document verification
    - Industry-specific analysis (Property Mgmt, Procurement, Insurance, etc.)
    
    Uses LangGraph for multi-step workflow:
    1. Document Processing → Extract invoice/document data
    2. Vendor Intelligence → Look up vendor history and profile
    3. Pricing Analysis → Compare against market benchmarks
    4. Fraud Detection → Identify fraud patterns
    5. Risk Assessment → Calculate overall risk score
    """
    
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_provider=None,
        session_id: Optional[str] = None,
        industry: IndustryNiche = IndustryNiche.GENERIC,
    ):
        super().__init__(agent_config, llm_provider, session_id)
        self.industry = industry
        self.rag_engine = None
        self.graph = None
        
        # Build LangGraph workflow
        if HAS_LANGGRAPH:
            self._build_workflow()
        
        # Register vendor fraud detection tools
        self._register_fraud_tools()
    
    def _default_prompt_template(self) -> str:
        """Vendor fraud detection prompt template"""
        return f"""You are Cyrex, an expert Vendor Fraud Detection AI Agent.

Your role: {self.role.value}
Industry focus: {self.industry.value}

You are an expert in detecting vendor and supplier fraud across multiple industries:
- Property Management (HVAC, plumbing, electrical contractor fraud)
- Corporate Procurement (supplier invoice fraud)
- P&C Insurance (auto body shop, home repair contractor fraud)
- General Contractors (subcontractor and material supplier fraud)
- Retail/E-Commerce (freight carrier and warehouse vendor fraud)
- Law Firms (expert witness and e-discovery vendor fraud)

Your capabilities:
1. Invoice Fraud Detection - Detect overpriced invoices, duplicate billing, phantom work
2. Vendor Intelligence - Build vendor profiles, track performance, detect bad actors
3. Pricing Benchmark Analysis - Compare prices against market rates
4. Document Verification - Verify invoice authenticity and detect forgeries
5. Risk Assessment - Calculate vendor risk scores and predict fraud probability

Current task: {{task}}
Context: {{context}}

Provide detailed analysis with specific findings, evidence, and recommendations.
Always cite sources from the knowledge base when available.
"""
    
    def _register_fraud_tools(self):
        """Register vendor fraud detection tools"""
        self.register_tool(
            "analyze_invoice",
            self._tool_analyze_invoice,
            "Analyze an invoice for fraud indicators (overpricing, duplicates, suspicious patterns)"
        )
        self.register_tool(
            "get_vendor_profile",
            self._tool_get_vendor_profile,
            "Get vendor profile including history, performance, and risk score"
        )
        self.register_tool(
            "check_pricing_benchmark",
            self._tool_check_pricing_benchmark,
            "Compare price against market benchmarks for the service/product"
        )
        self.register_tool(
            "search_similar_invoices",
            self._tool_search_similar_invoices,
            "Search for similar invoices to detect duplicates or patterns"
        )
        self.register_tool(
            "calculate_risk_score",
            self._tool_calculate_risk_score,
            "Calculate overall fraud risk score for a vendor or invoice"
        )
        self.register_tool(
            "query_knowledge_base",
            self._tool_query_knowledge_base,
            "Query the RAG knowledge base for industry-specific information"
        )
    
    def _build_workflow(self):
        """Build LangGraph workflow for vendor fraud detection"""
        if not HAS_LANGGRAPH:
            return
        
        try:
            workflow = StateGraph(VendorFraudState)
            
            # Add nodes for each processing step
            workflow.add_node("document_processor", self._node_document_processor)
            workflow.add_node("vendor_intelligence", self._node_vendor_intelligence)
            workflow.add_node("pricing_analyzer", self._node_pricing_analyzer)
            workflow.add_node("fraud_detector", self._node_fraud_detector)
            workflow.add_node("risk_assessor", self._node_risk_assessor)
            
            # Set entry point
            workflow.set_entry_point("document_processor")
            
            # Add edges for workflow flow
            workflow.add_edge("document_processor", "vendor_intelligence")
            workflow.add_edge("vendor_intelligence", "pricing_analyzer")
            workflow.add_edge("pricing_analyzer", "fraud_detector")
            workflow.add_edge("fraud_detector", "risk_assessor")
            workflow.add_edge("risk_assessor", END)
            
            # Compile workflow
            self.graph = workflow.compile()
            logger.info("VendorFraudAgent LangGraph workflow built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build LangGraph workflow: {e}", exc_info=True)
            self.graph = None
    
    # =========================================================================
    # LangGraph Workflow Nodes
    # =========================================================================
    
    async def _node_document_processor(self, state: VendorFraudState) -> VendorFraudState:
        """Node 1: Process and extract data from invoice/document"""
        logger.info("Document processor node executing", workflow_id=state.get("workflow_id"))
        
        try:
            invoice_data = state.get("invoice_data") or {}
            document_content = state.get("document_content") or ""
            
            # Extract structured data from invoice
            extracted = {
                "vendor_name": invoice_data.get("vendor_name", "Unknown"),
                "vendor_id": invoice_data.get("vendor_id", state.get("vendor_id")),
                "invoice_number": invoice_data.get("invoice_number"),
                "invoice_date": invoice_data.get("invoice_date"),
                "total_amount": invoice_data.get("total_amount", 0),
                "line_items": invoice_data.get("line_items", []),
                "service_category": invoice_data.get("service_category"),
                "property_address": invoice_data.get("property_address"),
                "work_description": invoice_data.get("work_description", document_content),
                "industry": state.get("industry", self.industry.value),
            }
            
            # Use LLM to extract additional details if document content provided
            if document_content and self.llm:
                extraction_prompt = f"""Extract key information from this invoice/document:

{document_content}

Extract and return as JSON:
- vendor_name
- invoice_number
- total_amount
- line_items (list with description, quantity, unit_price, total)
- service_category
- any suspicious elements or notes
"""
                try:
                    llm_response = await asyncio.to_thread(self.llm.invoke, extraction_prompt)
                    # Parse and merge LLM extraction
                    try:
                        llm_extracted = json.loads(llm_response)
                        extracted.update(llm_extracted)
                    except json.JSONDecodeError:
                        extracted["llm_notes"] = llm_response
                except Exception as e:
                    logger.warning(f"LLM extraction failed: {e}")
            
            state["extracted_data"] = extracted
            state["metadata"]["document_processed_at"] = datetime.utcnow().isoformat()
            
            logger.info("Document processing complete", 
                       vendor=extracted.get("vendor_name"),
                       amount=extracted.get("total_amount"))
            
        except Exception as e:
            logger.error(f"Document processor failed: {e}", exc_info=True)
            state["extracted_data"] = {"error": str(e)}
        
        return state
    
    async def _node_vendor_intelligence(self, state: VendorFraudState) -> VendorFraudState:
        """Node 2: Gather vendor intelligence and history"""
        logger.info("Vendor intelligence node executing", workflow_id=state.get("workflow_id"))
        
        try:
            extracted = state.get("extracted_data") or {}
            vendor_id = extracted.get("vendor_id") or state.get("vendor_id")
            vendor_name = extracted.get("vendor_name", "Unknown")
            industry = IndustryNiche(state.get("industry", self.industry.value))
            
            # Get vendor intelligence from service
            from ...services.vendor_intelligence_service import get_vendor_intelligence_service
            
            intel_service = await get_vendor_intelligence_service()
            vendor_profile_obj = await intel_service.get_or_create_vendor(
                vendor_id=vendor_id,
                vendor_name=vendor_name,
                industry=industry
            )
            
            # Query RAG for additional context
            vendor_context = await self._query_rag(
                f"vendor profile history performance {vendor_name} {vendor_id}"
            )
            
            # Build comprehensive vendor profile
            vendor_profile = {
                "vendor_id": vendor_profile_obj.vendor_id,
                "vendor_name": vendor_profile_obj.vendor_name,
                "known_industries": vendor_profile_obj.industries_served,
                "total_invoices_analyzed": vendor_profile_obj.total_invoices_analyzed,
                "fraud_flags_count": vendor_profile_obj.fraud_flags_count,
                "fraud_flags_by_industry": vendor_profile_obj.fraud_flags_by_industry,
                "cross_industry_flags": vendor_profile_obj.cross_industry_flags,
                "flagged_by_industries": vendor_profile_obj.flagged_by_industries,
                "average_invoice_amount": vendor_profile_obj.average_invoice_amount,
                "average_price_deviation": vendor_profile_obj.average_price_deviation,
                "pricing_deviation_history": vendor_profile_obj.pricing_deviation_history[-10:],  # Last 10
                "current_risk_score": vendor_profile_obj.current_risk_score,
                "risk_level": vendor_profile_obj.risk_level,
                "risk_history": vendor_profile_obj.risk_history[-5:],  # Last 5
                "status": vendor_profile_obj.status,
                "first_seen": vendor_profile_obj.first_seen.isoformat(),
                "last_activity": vendor_profile_obj.last_activity.isoformat(),
                "rag_context": vendor_context,
                "network_effects": vendor_profile_obj.cross_industry_flags > 0,
            }
            
            state["vendor_profile"] = vendor_profile
            
            logger.info("Vendor intelligence gathered",
                       vendor=vendor_name,
                       flags=vendor_profile_obj.fraud_flags_count,
                       risk_score=vendor_profile_obj.current_risk_score,
                       industries=len(vendor_profile_obj.industries_served))
            
        except Exception as e:
            logger.error(f"Vendor intelligence failed: {e}", exc_info=True)
            state["vendor_profile"] = {"error": str(e)}
        
        return state
    
    async def _node_pricing_analyzer(self, state: VendorFraudState) -> VendorFraudState:
        """Node 3: Analyze pricing against benchmarks"""
        logger.info("Pricing analyzer node executing", workflow_id=state.get("workflow_id"))
        
        try:
            extracted = state.get("extracted_data") or {}
            industry = state.get("industry", self.industry.value)
            
            total_amount = extracted.get("total_amount", 0)
            service_category = extracted.get("service_category", "general")
            line_items = extracted.get("line_items", [])
            
            # Query RAG for pricing benchmarks
            benchmark_context = await self._query_rag(
                f"pricing benchmark market rate {service_category} {industry}"
            )
            
            # Build pricing analysis
            pricing_analysis = {
                "invoice_total": total_amount,
                "service_category": service_category,
                "industry": industry,
                "benchmark_context": benchmark_context,
                "line_item_analysis": [],
                "price_deviation_percent": 0,
                "overpriced_items": [],
                "market_rate_estimate": 0,
            }
            
            # Analyze each line item if available
            for item in line_items:
                item_analysis = {
                    "description": item.get("description", "Unknown"),
                    "charged_price": item.get("total", item.get("unit_price", 0)),
                    "market_estimate": None,
                    "deviation_percent": None,
                    "flag": False,
                }
                
                # Use industry-specific logic for common services
                charged = item_analysis["charged_price"]
                if charged > 0:
                    # Simple heuristic - can be enhanced with actual benchmark data
                    # For now, flag items significantly above average
                    if charged > 1000:  # High-value items get more scrutiny
                        item_analysis["flag"] = True
                        pricing_analysis["overpriced_items"].append(item_analysis)
                
                pricing_analysis["line_item_analysis"].append(item_analysis)
            
            # Calculate overall deviation if we have benchmark data
            if total_amount > 0:
                # This would be enhanced with actual benchmark database
                # For now, use RAG context to inform analysis
                if benchmark_context and "overcharge" in benchmark_context.lower():
                    pricing_analysis["price_deviation_percent"] = 30  # Flag as potentially overpriced
            
            state["pricing_analysis"] = pricing_analysis
            
            logger.info("Pricing analysis complete",
                       total=total_amount,
                       overpriced_count=len(pricing_analysis.get("overpriced_items", [])))
            
        except Exception as e:
            logger.error(f"Pricing analyzer failed: {e}", exc_info=True)
            state["pricing_analysis"] = {"error": str(e)}
        
        return state
    
    async def _node_fraud_detector(self, state: VendorFraudState) -> VendorFraudState:
        """Node 4: Detect fraud patterns and indicators"""
        logger.info("Fraud detector node executing", workflow_id=state.get("workflow_id"))
        
        try:
            extracted = state.get("extracted_data") or {}
            vendor_profile = state.get("vendor_profile") or {}
            pricing_analysis = state.get("pricing_analysis") or {}
            
            fraud_indicators = []
            
            # Check for inflated invoice
            if pricing_analysis.get("price_deviation_percent", 0) > 20:
                fraud_indicators.append({
                    "type": VendorFraudType.INFLATED_INVOICE.value,
                    "severity": "high" if pricing_analysis.get("price_deviation_percent", 0) > 40 else "medium",
                    "description": f"Invoice price is {pricing_analysis.get('price_deviation_percent', 0)}% above market rate",
                    "evidence": pricing_analysis.get("overpriced_items", []),
                })
            
            # Check for duplicate billing patterns
            # This would query the RAG for similar invoices
            similar_invoices_context = await self._query_rag(
                f"invoice {extracted.get('invoice_number')} {extracted.get('vendor_name')} duplicate"
            )
            if similar_invoices_context and "duplicate" in similar_invoices_context.lower():
                fraud_indicators.append({
                    "type": VendorFraudType.DUPLICATE_BILLING.value,
                    "severity": "high",
                    "description": "Potential duplicate billing detected",
                    "evidence": similar_invoices_context,
                })
            
            # Check vendor history for previous flags
            if vendor_profile.get("fraud_flags_count", 0) > 0:
                fraud_indicators.append({
                    "type": "vendor_history_flag",
                    "severity": "medium",
                    "description": f"Vendor has {vendor_profile.get('fraud_flags_count')} previous fraud flags",
                    "evidence": vendor_profile.get("previous_flags"),
                })
            
            # Check for suspicious patterns in work description
            work_desc = extracted.get("work_description", "")
            if work_desc:
                suspicious_keywords = ["emergency", "urgent", "after hours", "premium"]
                for keyword in suspicious_keywords:
                    if keyword in work_desc.lower():
                        fraud_indicators.append({
                            "type": VendorFraudType.PRICE_GOUGING.value,
                            "severity": "low",
                            "description": f"Work description contains '{keyword}' which may justify premium pricing but requires verification",
                            "evidence": keyword,
                        })
                        break
            
            state["fraud_indicators"] = fraud_indicators
            state["fraud_detected"] = len(fraud_indicators) > 0
            
            logger.info("Fraud detection complete",
                       indicators_found=len(fraud_indicators),
                       fraud_detected=state["fraud_detected"])
            
        except Exception as e:
            logger.error(f"Fraud detector failed: {e}", exc_info=True)
            state["fraud_indicators"] = []
            state["fraud_detected"] = False
        
        return state
    
    async def _node_risk_assessor(self, state: VendorFraudState) -> VendorFraudState:
        """Node 5: Calculate overall risk score and recommendations"""
        logger.info("Risk assessor node executing", workflow_id=state.get("workflow_id"))
        
        try:
            fraud_indicators = state.get("fraud_indicators", [])
            vendor_profile = state.get("vendor_profile") or {}
            pricing_analysis = state.get("pricing_analysis") or {}
            
            # Calculate risk score (0-100)
            risk_score = 0
            
            # Add points for each fraud indicator
            severity_weights = {"low": 10, "medium": 25, "high": 40, "critical": 60}
            for indicator in fraud_indicators:
                risk_score += severity_weights.get(indicator.get("severity", "low"), 10)
            
            # Add points for vendor history
            risk_score += vendor_profile.get("fraud_flags_count", 0) * 15
            
            # Add points for pricing deviation
            price_deviation = pricing_analysis.get("price_deviation_percent", 0)
            if price_deviation > 50:
                risk_score += 30
            elif price_deviation > 30:
                risk_score += 20
            elif price_deviation > 20:
                risk_score += 10
            
            # Cap at 100
            risk_score = min(risk_score, 100)
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = RiskLevel.CRITICAL.value
            elif risk_score >= 50:
                risk_level = RiskLevel.HIGH.value
            elif risk_score >= 25:
                risk_level = RiskLevel.MEDIUM.value
            else:
                risk_level = RiskLevel.LOW.value
            
            # Generate recommendations
            recommendations = []
            if risk_score >= 70:
                recommendations.append("URGENT: Halt payment pending investigation")
                recommendations.append("Conduct vendor audit and request itemized breakdown")
                recommendations.append("Compare with alternative vendor quotes")
            elif risk_score >= 50:
                recommendations.append("Request detailed documentation before payment")
                recommendations.append("Verify work completion with on-site inspection")
                recommendations.append("Check vendor references and history")
            elif risk_score >= 25:
                recommendations.append("Review line items for accuracy")
                recommendations.append("Compare against historical invoices from this vendor")
            else:
                recommendations.append("Invoice appears within normal parameters")
                recommendations.append("Standard approval process recommended")
            
            # Build risk assessment
            risk_assessment = {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "confidence_score": 0.85 if len(fraud_indicators) > 0 else 0.75,
                "indicators_count": len(fraud_indicators),
                "assessment_date": datetime.utcnow().isoformat(),
                "industry": state.get("industry", self.industry.value),
            }
            
            state["risk_assessment"] = risk_assessment
            state["risk_level"] = risk_level
            state["recommendations"] = recommendations
            state["confidence_score"] = risk_assessment["confidence_score"]
            
            logger.info("Risk assessment complete",
                       risk_score=risk_score,
                       risk_level=risk_level,
                       recommendations_count=len(recommendations))
            
        except Exception as e:
            logger.error(f"Risk assessor failed: {e}", exc_info=True)
            state["risk_assessment"] = {"error": str(e)}
            state["risk_level"] = RiskLevel.MEDIUM.value
            state["recommendations"] = ["Manual review required due to processing error"]
            state["confidence_score"] = 0.5
        
        return state
    
    # =========================================================================
    # Tool Implementations
    # =========================================================================
    
    async def _tool_analyze_invoice(
        self,
        invoice_data: Dict[str, Any],
        industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze an invoice for fraud indicators"""
        return await self.analyze_invoice(
            invoice_data=invoice_data,
            industry=IndustryNiche(industry) if industry else self.industry
        )
    
    async def _tool_get_vendor_profile(self, vendor_id: str) -> Dict[str, Any]:
        """Get vendor profile from intelligence database"""
        context = await self._query_rag(f"vendor profile {vendor_id}")
        return {
            "vendor_id": vendor_id,
            "profile_data": context,
            "retrieved_at": datetime.utcnow().isoformat(),
        }
    
    async def _tool_check_pricing_benchmark(
        self,
        service_type: str,
        price: float,
        industry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check price against market benchmarks"""
        industry_val = industry or self.industry.value
        context = await self._query_rag(f"pricing benchmark {service_type} {industry_val}")
        return {
            "service_type": service_type,
            "submitted_price": price,
            "industry": industry_val,
            "benchmark_context": context,
            "checked_at": datetime.utcnow().isoformat(),
        }
    
    async def _tool_search_similar_invoices(
        self,
        vendor_id: str,
        amount: float,
        date_range_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Search for similar invoices to detect duplicates"""
        context = await self._query_rag(
            f"invoice vendor:{vendor_id} amount:{amount} similar duplicate"
        )
        return [{
            "search_query": f"vendor:{vendor_id} amount:{amount}",
            "context": context,
            "searched_at": datetime.utcnow().isoformat(),
        }]
    
    async def _tool_calculate_risk_score(
        self,
        vendor_id: str,
        fraud_indicators: List[str]
    ) -> Dict[str, Any]:
        """Calculate fraud risk score"""
        base_score = len(fraud_indicators) * 15
        return {
            "vendor_id": vendor_id,
            "risk_score": min(base_score, 100),
            "indicators_count": len(fraud_indicators),
            "calculated_at": datetime.utcnow().isoformat(),
        }
    
    async def _tool_query_knowledge_base(
        self,
        query: str,
        industry: Optional[str] = None
    ) -> str:
        """Query the RAG knowledge base"""
        return await self._query_rag(query)
    
    # =========================================================================
    # RAG Integration
    # =========================================================================
    
    async def _query_rag(self, query: str) -> str:
        """Query the RAG engine for relevant context"""
        try:
            if not self.rag_engine:
                await self._initialize_rag()
            
            if self.rag_engine:
                from ...integrations.universal_rag_engine import RAGQuery
                rag_query = RAGQuery(
                    query=query,
                    industry=self.industry,
                    top_k=5,
                )
                results = self.rag_engine.retrieve(rag_query)
                if results:
                    return "\n\n".join([r.document.content for r in results[:3]])
            
            return ""
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            return ""
    
    async def _initialize_rag(self):
        """Initialize RAG engine"""
        try:
            from ...integrations.universal_rag_engine import create_universal_rag_engine
            self.rag_engine = create_universal_rag_engine(
                industry=self.industry,
                collection_name=f"cyrex_{self.industry.value}_vendor_fraud"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize RAG engine: {e}")
            self.rag_engine = None
    
    # =========================================================================
    # Main Analysis Methods
    # =========================================================================
    
    async def analyze_invoice(
        self,
        invoice_data: Dict[str, Any],
        industry: Optional[IndustryNiche] = None,
        document_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main method to analyze an invoice for fraud
        
        Args:
            invoice_data: Invoice data dictionary
            industry: Industry niche for analysis
            document_content: Raw document content (optional)
        
        Returns:
            Complete fraud analysis result
        """
        industry = industry or self.industry
        
        # Build initial state
        initial_state: VendorFraudState = {
            "messages": [],
            "workflow_id": f"fraud_analysis_{datetime.utcnow().timestamp()}",
            "session_id": self.session_id,
            "industry": industry.value,
            "invoice_data": invoice_data,
            "vendor_id": invoice_data.get("vendor_id"),
            "document_content": document_content,
            "query": f"Analyze invoice from {invoice_data.get('vendor_name', 'Unknown')}",
            "extracted_data": None,
            "pricing_analysis": None,
            "vendor_profile": None,
            "fraud_indicators": [],
            "risk_assessment": None,
            "fraud_detected": False,
            "risk_level": RiskLevel.LOW.value,
            "recommendations": [],
            "confidence_score": 0.0,
            "metadata": {},
        }
        
        # Execute workflow
        if self.graph:
            try:
                result = await self.graph.ainvoke(initial_state)
                analysis_result = self._format_analysis_result(result)
                
                # Record in vendor intelligence database
                await self._record_analysis(invoice_data, analysis_result, industry)
                
                return analysis_result
            except Exception as e:
                logger.error(f"LangGraph workflow failed: {e}", exc_info=True)
                # Fall back to sequential execution
        
        # Fallback: Sequential execution
        analysis_result = await self._sequential_analysis(initial_state)
        
        # Record in vendor intelligence database
        await self._record_analysis(invoice_data, analysis_result, industry)
        
        return analysis_result
    
    async def _record_analysis(
        self,
        invoice_data: Dict[str, Any],
        analysis_result: Dict[str, Any],
        industry: IndustryNiche
    ):
        """Record analysis in vendor intelligence database"""
        try:
            from ...services.vendor_intelligence_service import get_vendor_intelligence_service
            
            intel_service = await get_vendor_intelligence_service()
            await intel_service.record_invoice_analysis(
                invoice_data=invoice_data,
                analysis_result=analysis_result,
                industry=industry
            )
            
            logger.info("Analysis recorded in vendor intelligence database",
                       vendor=invoice_data.get("vendor_name"),
                       fraud_detected=analysis_result.get("fraud_detected"))
        except Exception as e:
            logger.warning(f"Failed to record analysis: {e}")
    
    async def _sequential_analysis(self, state: VendorFraudState) -> Dict[str, Any]:
        """Sequential fallback when LangGraph not available"""
        state = await self._node_document_processor(state)
        state = await self._node_vendor_intelligence(state)
        state = await self._node_pricing_analyzer(state)
        state = await self._node_fraud_detector(state)
        state = await self._node_risk_assessor(state)
        return self._format_analysis_result(state)
    
    def _format_analysis_result(self, state: VendorFraudState) -> Dict[str, Any]:
        """Format the analysis result for API response"""
        return {
            "success": True,
            "workflow_id": state.get("workflow_id"),
            "industry": state.get("industry"),
            "fraud_detected": state.get("fraud_detected", False),
            "risk_level": state.get("risk_level", RiskLevel.LOW.value),
            "risk_score": state.get("risk_assessment", {}).get("risk_score", 0),
            "confidence_score": state.get("confidence_score", 0.0),
            "fraud_indicators": state.get("fraud_indicators", []),
            "recommendations": state.get("recommendations", []),
            "extracted_data": state.get("extracted_data"),
            "vendor_profile": state.get("vendor_profile"),
            "pricing_analysis": state.get("pricing_analysis"),
            "risk_assessment": state.get("risk_assessment"),
            "metadata": state.get("metadata", {}),
            "analyzed_at": datetime.utcnow().isoformat(),
        }
    
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task - required by BaseAgent"""
        task_type = task.get("type", "analyze_invoice")
        
        if task_type == "analyze_invoice":
            return await self.analyze_invoice(
                invoice_data=task.get("invoice_data", {}),
                industry=IndustryNiche(task.get("industry")) if task.get("industry") else None,
                document_content=task.get("document_content"),
            )
        elif task_type == "query":
            response = await self.invoke(
                input_text=task.get("query", ""),
                context=context,
            )
            return response.to_dict()
        else:
            return {"error": f"Unknown task type: {task_type}"}

