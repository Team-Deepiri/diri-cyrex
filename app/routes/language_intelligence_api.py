"""
Language Intelligence API Routes
FastAPI endpoints for lease abstraction and contract intelligence
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from ..logging_config import get_logger
from ..services.document_processors.lease_processor import LeaseProcessor
from ..services.document_processors.contract_processor import ContractProcessor
from ..services.clause_evolution_tracker import ClauseEvolutionTracker
from ..services.obligation_dependency_graph import ObligationDependencyGraph
from ..integrations.llm_providers import get_llm_provider
import json
import re

logger = get_logger("cyrex.api.language_intelligence")

router = APIRouter(prefix="/language-intelligence", tags=["Language Intelligence"])


# ============================================
# PHASE 1: LEASE ABSTRACTION
# ============================================

class AbstractLeaseRequest(BaseModel):
    leaseId: str
    documentText: str
    documentUrl: str
    leaseNumber: Optional[str] = None
    tenantName: Optional[str] = None
    propertyAddress: Optional[str] = None


@router.post("/lease/abstract")
async def abstract_lease(request: AbstractLeaseRequest):
    """Process lease document and return abstracted terms"""
    try:
        processor = LeaseProcessor()
        result = await processor.process(
            document_text=request.documentText,
            document_url=request.documentUrl,
            lease_id=request.leaseId,
            lease_number=request.leaseNumber,
        )
        
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Lease abstraction error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# PHASE 2: CONTRACT INTELLIGENCE
# ============================================

class AbstractContractRequest(BaseModel):
    contractId: str
    documentText: str
    documentUrl: str
    contractNumber: Optional[str] = None
    partyA: Optional[str] = None
    partyB: Optional[str] = None
    versionNumber: int = 1


@router.post("/contract/abstract")
async def abstract_contract(request: AbstractContractRequest):
    """Process contract document and return abstracted terms"""
    try:
        processor = ContractProcessor()
        result = await processor.process(
            document_text=request.documentText,
            document_url=request.documentUrl,
            contract_id=request.contractId,
            version_number=request.versionNumber,
        )
        
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Contract abstraction error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class TrackClauseEvolutionRequest(BaseModel):
    contractId: str
    oldVersionClauses: List[Dict[str, Any]]
    newVersionClauses: List[Dict[str, Any]]
    oldVersionNumber: int
    newVersionNumber: int


@router.post("/contract/track-clause-evolution")
async def track_clause_evolution(request: TrackClauseEvolutionRequest):
    """Track clause changes between contract versions"""
    try:
        tracker = ClauseEvolutionTracker()
        result = await tracker.track_clause_changes(
            contract_id=request.contractId,
            old_version_clauses=request.oldVersionClauses,
            new_version_clauses=request.newVersionClauses,
            old_version_number=request.oldVersionNumber,
            new_version_number=request.newVersionNumber,
        )
        
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Clause evolution tracking error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class BuildDependencyGraphRequest(BaseModel):
    contractId: str
    obligations: List[Dict[str, Any]]
    contracts: Optional[List[str]] = None
    leases: Optional[List[str]] = None


@router.post("/contract/build-dependency-graph")
async def build_dependency_graph(request: BuildDependencyGraphRequest):
    """Build obligation dependency graph"""
    try:
        graph_builder = ObligationDependencyGraph()
        result = await graph_builder.build_graph(
            obligations=request.obligations,
            contracts=request.contracts,
            leases=request.leases,
        )
        
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Dependency graph error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class FindCascadingObligationsRequest(BaseModel):
    obligationId: str
    maxDepth: int = 5


@router.post("/obligations/find-cascading")
async def find_cascading_obligations(request: FindCascadingObligationsRequest):
    """Find obligations that cascade from a given obligation"""
    try:
        graph_builder = ObligationDependencyGraph()
        result = await graph_builder.find_cascading_obligations(
            obligation_id=request.obligationId,
            max_depth=request.maxDepth,
        )
        
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Cascade analysis error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class CompareVersionsRequest(BaseModel):
    oldAbstractedTerms: Dict[str, Any]
    newAbstractedTerms: Dict[str, Any]


@router.post("/contract/compare-versions")
async def compare_contract_versions(request: CompareVersionsRequest):
    """Compare two contract versions"""
    try:
        llm_provider = get_llm_provider()
        llm = llm_provider.get_llm()
        
        comparison_prompt = f"""Compare these two contract versions and identify all changes:

OLD VERSION:
{json.dumps(request.oldAbstractedTerms, indent=2)}

NEW VERSION:
{json.dumps(request.newAbstractedTerms, indent=2)}

Return a JSON object with:
- "summary": A brief summary of changes
- "significant": boolean indicating if changes are significant
- "changes": Array of specific changes with:
  - "field": The field that changed
  - "oldValue": Previous value
  - "newValue": New value
  - "changeType": "ADDED", "MODIFIED", "DELETED"
  - "impact": "HIGH", "MEDIUM", "LOW"

Return ONLY valid JSON."""
        
        response = await llm.ainvoke(comparison_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            comparison_result = json.loads(json_match.group())
        else:
            comparison_result = {
                "summary": "Version comparison completed",
                "significant": False,
                "changes": [],
            }
        
        return {"success": True, "data": comparison_result}
    except Exception as e:
        logger.error("Version comparison error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lease/compare-versions")
async def compare_lease_versions(request: CompareVersionsRequest):
    """Compare two lease versions"""
    try:
        llm_provider = get_llm_provider()
        llm = llm_provider.get_llm()
        
        comparison_prompt = f"""Compare these two lease versions and identify all changes:

OLD VERSION:
{json.dumps(request.oldAbstractedTerms, indent=2)}

NEW VERSION:
{json.dumps(request.newAbstractedTerms, indent=2)}

Return a JSON object with:
- "summary": A brief summary of changes
- "significant": boolean indicating if changes are significant
- "changes": Array of specific changes with:
  - "field": The field that changed (e.g., "financialTerms.baseRent", "keyDates.leaseEndDate")
  - "oldValue": Previous value
  - "newValue": New value
  - "changeType": "ADDED", "MODIFIED", "DELETED"
  - "impact": "HIGH", "MEDIUM", "LOW"

Return ONLY valid JSON."""
        
        response = await llm.ainvoke(comparison_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            comparison_result = json.loads(json_match.group())
        else:
            comparison_result = {
                "summary": "Version comparison completed",
                "significant": False,
                "changes": [],
            }
        
        return {"success": True, "data": comparison_result}
    except Exception as e:
        logger.error("Lease version comparison error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

