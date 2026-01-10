"""
Contract Document Processor
Extracts structured data from contract documents using LLM
"""
from typing import Dict, Any, Optional, List
import json
import re
from datetime import datetime

from app.logging_config import get_logger
from app.integrations.llm_providers import get_llm_provider
from app.integrations.rag_pipeline import RAGPipeline

logger = get_logger("cyrex.contract_processor")


class ContractProcessor:
    """
    Process contract documents and extract structured terms
    
    Extracts:
    - Contract parties and details
    - Key clauses with types
    - Financial terms
    - Obligations
    - Termination conditions
    - Liability and indemnification terms
    """
    
    def __init__(self, llm_provider=None, rag_pipeline: Optional[RAGPipeline] = None):
        self.llm_provider = llm_provider or get_llm_provider()
        self.rag_pipeline = rag_pipeline
        
        self.extraction_prompt = """You are an expert contract analyst. Extract structured data from the following contract document.

Contract Document Text:
{document_text}

Extract the following information and return as JSON:

{{
  "parties": {{
    "partyA": {{
      "name": "string",
      "entityType": "CORPORATION|LLC|INDIVIDUAL|PARTNERSHIP",
      "contactInfo": {{"email": "string", "phone": "string"}}
    }},
    "partyB": {{
      "name": "string",
      "entityType": "CORPORATION|LLC|INDIVIDUAL|PARTNERSHIP",
      "contactInfo": {{"email": "string", "phone": "string"}}
    }}
  }},
  "contractDetails": {{
    "contractType": "SERVICE|SUPPLY|MSA|NDA|LICENSE|PARTNERSHIP|OTHER",
    "jurisdiction": "string",
    "governingLaw": "string",
    "effectiveDate": "YYYY-MM-DD",
    "expirationDate": "YYYY-MM-DD",
    "autoRenewal": boolean,
    "renewalTerms": "string"
  }},
  "financialTerms": {{
    "paymentTerms": "string",
    "paymentSchedule": [
      {{"amount": number, "currency": "USD", "dueDate": "YYYY-MM-DD", "milestone": "string"}}
    ],
    "lateFees": "string",
    "terminationFees": "string",
    "penalties": "string"
  }},
  "clauses": [
    {{
      "clauseNumber": "string",
      "clauseType": "TERMINATION|PAYMENT|LIABILITY|INDEMNIFICATION|CONFIDENTIALITY|NON_COMPETE|FORCE_MAJEURE|DISPUTE_RESOLUTION|INTELLECTUAL_PROPERTY|WARRANTY|OTHER",
      "clauseTitle": "string",
      "clauseText": "string",
      "appliesTo": "PARTY_A|PARTY_B|BOTH",
      "section": "string",
      "pageNumber": number
    }}
  ],
  "obligations": [
    {{
      "description": "string",
      "obligationType": "PAYMENT|DELIVERY|PERFORMANCE|NOTIFICATION|COMPLIANCE|RENEWAL|TERMINATION|CONFIDENTIALITY|OTHER",
      "party": "PARTY_A|PARTY_B|BOTH",
      "deadline": "YYYY-MM-DD",
      "frequency": "ONE_TIME|MONTHLY|QUARTERLY|ANNUAL",
      "amount": number,
      "currency": "USD",
      "conditions": "string",
      "triggers": ["string"],
      "dependencies": ["string"]
    }}
  ],
  "terminationTerms": {{
    "terminationRights": [
      {{"party": "PARTY_A|PARTY_B|BOTH", "conditions": "string", "noticeRequired": number}}
    ],
    "terminationPenalties": "string",
    "survivalClauses": ["string"]
  }},
  "liabilityTerms": {{
    "limitationOfLiability": "string",
    "indemnification": [
      {{"indemnifyingParty": "PARTY_A|PARTY_B", "indemnifiedParty": "PARTY_A|PARTY_B", "scope": "string"}}
    ],
    "insuranceRequirements": [
      {{"type": "GENERAL_LIABILITY|PROFESSIONAL|ERRORS_OMISSIONS", "minimumCoverage": number, "requiredBy": "PARTY_A|PARTY_B"}}
    ]
  }},
  "intellectualProperty": {{
    "ownership": "string",
    "licenses": [
      {{"licensor": "PARTY_A|PARTY_B", "licensee": "PARTY_A|PARTY_B", "scope": "string"}}
    ],
    "restrictions": ["string"]
  }},
  "disputeResolution": {{
    "governingLaw": "string",
    "jurisdiction": "string",
    "arbitration": boolean,
    "arbitrationRules": "string",
    "mediation": boolean
  }}
}}

Be precise and extract only information that is explicitly stated in the document. If information is not available, use null.
Return ONLY valid JSON, no additional text."""

    async def process(
        self,
        document_text: str,
        document_url: str,
        contract_id: Optional[str] = None,
        version_number: int = 1,
    ) -> Dict[str, Any]:
        """
        Process contract document and extract structured terms
        """
        start_time = datetime.now()
        
        try:
            logger.info("Processing contract document",
                       document_url=document_url,
                       contract_id=contract_id,
                       version_number=version_number)
            
            # Use RAG for similar contract examples
            context = ""
            if self.rag_pipeline:
                try:
                    similar_contracts = await self.rag_pipeline.query(
                        query=f"contract clauses obligations {contract_id or ''}",
                        top_k=3
                    )
                    if similar_contracts:
                        context = "\n\nSimilar contract examples:\n" + "\n".join(
                            [doc.get("content", "") for doc in similar_contracts]
                        )
                except Exception as e:
                    logger.warning("RAG retrieval failed", error=str(e))
            
            # Format prompt
            prompt = self.extraction_prompt.format(
                document_text=document_text[:50000]
            )
            
            if context:
                prompt += context
            
            # Call LLM
            llm = self.llm_provider.get_llm()
            response = await llm.ainvoke(prompt)
            
            # Parse JSON response
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in LLM response")
            
            abstracted_terms = json.loads(json_str)
            abstracted_terms = self._validate_and_clean(abstracted_terms)
            
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info("Contract processing completed",
                       contract_id=contract_id,
                       processing_time_ms=processing_time_ms,
                       clauses_count=len(abstracted_terms.get("clauses", [])),
                       obligations_count=len(abstracted_terms.get("obligations", [])))
            
            return {
                "abstractedTerms": abstracted_terms,
                "keyClauses": abstracted_terms.get("clauses", []),
                "financialTerms": abstracted_terms.get("financialTerms", {}),
                "obligations": abstracted_terms.get("obligations", []),
                "terminationTerms": abstracted_terms.get("terminationTerms", {}),
                "liabilityTerms": abstracted_terms.get("liabilityTerms", {}),
                "intellectualProperty": abstracted_terms.get("intellectualProperty", {}),
                "disputeResolution": abstracted_terms.get("disputeResolution", {}),
                "confidence": self._calculate_confidence(abstracted_terms),
                "processingTimeMs": processing_time_ms,
            }
            
        except Exception as e:
            logger.error("Error processing contract document",
                        document_url=document_url,
                        error=str(e))
            raise
    
    def _validate_and_clean(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted data"""
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [item for item in value if item is not None]
            elif isinstance(value, dict):
                data[key] = self._validate_and_clean(value)
        return data
    
    def _calculate_confidence(self, abstracted_terms: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        required_fields = ["parties", "contractDetails", "clauses"]
        populated_fields = sum(1 for field in required_fields if abstracted_terms.get(field))
        base_confidence = populated_fields / len(required_fields)
        
        if abstracted_terms.get("obligations"):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

