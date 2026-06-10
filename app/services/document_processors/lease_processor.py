"""
Lease Document Processor
Extracts structured data from lease documents using LLM
"""
from typing import Dict, Any, Optional, List
import json
import re
from datetime import datetime

from app.logging_config import get_logger
from app.integrations.llm_providers import get_llm_provider
from app.integrations.rag_pipeline import RAGPipeline

logger = get_logger("cyrex.lease_processor")


class LeaseProcessor:
    """
    Process lease documents and extract structured terms
    
    Extracts:
    - Financial terms (rent, deposit, escalations)
    - Key dates (start, end, renewal options)
    - Property details
    - Obligations (tenant and landlord)
    - Key clauses
    """
    
    def __init__(self, llm_provider=None, rag_pipeline: Optional[RAGPipeline] = None):
        self.llm_provider = llm_provider or get_llm_provider()
        self.rag_pipeline = rag_pipeline
        
        self.extraction_prompt = """You are an expert lease analyst. Extract structured data from the following lease document.

Lease Document Text:
{document_text}

Extract the following information and return as JSON:

{{
  "financialTerms": {{
    "baseRent": {{
      "amount": number,
      "currency": "USD",
      "frequency": "monthly|quarterly|annual"
    }},
    "securityDeposit": {{
      "amount": number,
      "currency": "USD"
    }},
    "escalations": [
      {{
        "type": "CPI|FIXED|MARKET",
        "percentage": number,
        "effectiveDate": "YYYY-MM-DD"
      }}
    ],
    "additionalCharges": [
      {{
        "type": "CAM|UTILITIES|TAXES|INSURANCE|OTHER",
        "description": "string",
        "amount": number,
        "frequency": "monthly|quarterly|annual|one_time"
      }}
    ]
  }},
  "keyDates": {{
    "leaseStartDate": "YYYY-MM-DD",
    "leaseEndDate": "YYYY-MM-DD",
    "renewalOptions": [
      {{
        "optionNumber": number,
        "termMonths": number,
        "noticeDays": number,
        "rentAdjustment": "string"
      }}
    ]
  }},
  "propertyDetails": {{
    "squareFootage": number,
    "propertyType": "OFFICE|RETAIL|INDUSTRIAL|RESIDENTIAL|OTHER",
    "address": "string",
    "suiteNumber": "string"
  }},
  "parties": {{
    "tenant": {{
      "name": "string",
      "entityType": "CORPORATION|LLC|INDIVIDUAL|PARTNERSHIP",
      "contactInfo": {{"email": "string", "phone": "string"}}
    }},
    "landlord": {{
      "name": "string",
      "entityType": "CORPORATION|LLC|INDIVIDUAL|PARTNERSHIP",
      "contactInfo": {{"email": "string", "phone": "string"}}
    }}
  }},
  "keyClauses": [
    {{
      "clauseType": "TERMINATION|ASSIGNMENT|MAINTENANCE|INSURANCE|DEFAULT|OTHER",
      "title": "string",
      "summary": "string",
      "fullText": "string",
      "appliesTo": "TENANT|LANDLORD|BOTH"
    }}
  ],
  "obligations": [
    {{
      "description": "string",
      "obligationType": "PAYMENT|MAINTENANCE|NOTIFICATION|COMPLIANCE|INSURANCE|TAX|UTILITY|REPAIR|INSPECTION|OTHER",
      "party": "TENANT|LANDLORD|BOTH",
      "deadline": "YYYY-MM-DD",
      "frequency": "ONE_TIME|MONTHLY|QUARTERLY|ANNUAL|ONGOING",
      "amount": number,
      "currency": "USD",
      "sourceClause": "string",
      "confidence": 0.0-1.0
    }}
  ],
  "insuranceRequirements": [
    {{
      "type": "GENERAL_LIABILITY|PROPERTY|WORKERS_COMP|OTHER",
      "minimumCoverage": number,
      "requiredBy": "TENANT|LANDLORD",
      "evidenceRequired": boolean
    }}
  ],
  "maintenanceResponsibilities": {{
    "tenant": ["string"],
    "landlord": ["string"],
    "shared": ["string"]
  }},
  "assignmentAndSubletting": {{
    "assignmentAllowed": boolean,
    "sublettingAllowed": boolean,
    "conditions": "string",
    "landlordConsentRequired": boolean
  }},
  "terminationTerms": {{
    "earlyTerminationAllowed": boolean,
    "terminationPenalty": "string",
    "noticeRequired": number,
    "noticeUnit": "DAYS|MONTHS"
  }}
}}

Be precise and extract only information that is explicitly stated in the document. If information is not available, use null.
Return ONLY valid JSON, no additional text."""

    async def process(
        self,
        document_text: str,
        document_url: str,
        lease_id: Optional[str] = None,
        lease_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process lease document and extract structured terms
        """
        start_time = datetime.now()
        
        try:
            logger.info("Processing lease document",
                       document_url=document_url,
                       lease_id=lease_id,
                       lease_number=lease_number)
            
            # Use RAG for similar lease examples
            context = ""
            if self.rag_pipeline:
                try:
                    similar_leases = await self.rag_pipeline.query(
                        query=f"lease terms financial obligations {lease_number or ''}",
                        top_k=3
                    )
                    if similar_leases:
                        context = "\n\nSimilar lease examples:\n" + "\n".join(
                            [doc.get("content", "") for doc in similar_leases]
                        )
                except Exception as e:
                    logger.warning("RAG retrieval failed", error=str(e))
            
            # Format prompt
            prompt = self.extraction_prompt.format(
                document_text=document_text[:50000]  # Limit text length
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
            
            logger.info("Lease processing completed",
                       lease_id=lease_id,
                       processing_time_ms=processing_time_ms,
                       obligations_count=len(abstracted_terms.get("obligations", [])))
            
            return {
                "abstractedTerms": abstracted_terms,
                "financialTerms": abstracted_terms.get("financialTerms", {}),
                "keyDates": abstracted_terms.get("keyDates", {}),
                "propertyDetails": abstracted_terms.get("propertyDetails", {}),
                "keyClauses": abstracted_terms.get("keyClauses", []),
                "obligations": abstracted_terms.get("obligations", []),
                "confidence": self._calculate_confidence(abstracted_terms),
                "processingTimeMs": processing_time_ms,
            }
            
        except Exception as e:
            logger.error("Error processing lease document",
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
        required_fields = ["financialTerms", "keyDates", "parties"]
        populated_fields = sum(1 for field in required_fields if abstracted_terms.get(field))
        base_confidence = populated_fields / len(required_fields)
        
        if abstracted_terms.get("obligations"):
            base_confidence += 0.1
        
        if abstracted_terms.get("keyClauses"):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

