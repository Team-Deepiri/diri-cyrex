"""
Vendor Fraud Detection Tools
Specialized tools for vendor fraud analysis, pricing benchmarks, and document verification
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import asyncio
from ...logging_config import get_logger

logger = get_logger("cyrex.tools.vendor_fraud")


# =============================================================================
# Industry-Specific Pricing Benchmarks
# =============================================================================

INDUSTRY_BENCHMARKS = {
    "property_management": {
        "hvac_repair": {"min": 150, "avg": 350, "max": 800, "unit": "per visit"},
        "hvac_installation": {"min": 3000, "avg": 5500, "max": 12000, "unit": "per unit"},
        "plumbing_repair": {"min": 100, "avg": 300, "max": 600, "unit": "per visit"},
        "plumbing_emergency": {"min": 200, "avg": 500, "max": 1000, "unit": "per visit"},
        "electrical_repair": {"min": 100, "avg": 250, "max": 500, "unit": "per visit"},
        "electrical_panel_upgrade": {"min": 1500, "avg": 3000, "max": 6000, "unit": "per panel"},
        "roof_repair": {"min": 300, "avg": 800, "max": 2000, "unit": "per repair"},
        "roof_replacement": {"min": 5000, "avg": 12000, "max": 25000, "unit": "per roof"},
        "appliance_repair": {"min": 100, "avg": 200, "max": 400, "unit": "per appliance"},
        "general_maintenance": {"min": 50, "avg": 100, "max": 200, "unit": "per hour"},
    },
    "corporate_procurement": {
        "it_services": {"min": 75, "avg": 150, "max": 300, "unit": "per hour"},
        "consulting": {"min": 150, "avg": 300, "max": 600, "unit": "per hour"},
        "office_supplies": {"min": 0, "avg": 0, "max": 0, "unit": "varies"},  # Too variable
        "software_licenses": {"min": 0, "avg": 0, "max": 0, "unit": "per user/year"},
        "facilities_maintenance": {"min": 50, "avg": 100, "max": 200, "unit": "per hour"},
    },
    "insurance_pc": {
        "auto_body_labor": {"min": 50, "avg": 75, "max": 125, "unit": "per hour"},
        "auto_parts_markup": {"min": 0, "avg": 30, "max": 50, "unit": "percent"},
        "roof_repair_claim": {"min": 300, "avg": 1500, "max": 5000, "unit": "per claim"},
        "water_damage_restoration": {"min": 1500, "avg": 4500, "max": 12000, "unit": "per claim"},
        "fire_damage_restoration": {"min": 5000, "avg": 20000, "max": 100000, "unit": "per claim"},
    },
    "general_contractors": {
        "electrical_subcontractor": {"min": 65, "avg": 95, "max": 150, "unit": "per hour"},
        "plumbing_subcontractor": {"min": 60, "avg": 90, "max": 140, "unit": "per hour"},
        "hvac_subcontractor": {"min": 70, "avg": 100, "max": 160, "unit": "per hour"},
        "drywall_subcontractor": {"min": 2, "avg": 3, "max": 5, "unit": "per sqft"},
        "painting_subcontractor": {"min": 3, "avg": 5, "max": 8, "unit": "per sqft"},
        "concrete_material": {"min": 100, "avg": 150, "max": 200, "unit": "per cubic yard"},
        "lumber": {"min": 400, "avg": 600, "max": 900, "unit": "per 1000 board feet"},
    },
    "retail_ecommerce": {
        "ltl_freight": {"min": 100, "avg": 300, "max": 800, "unit": "per shipment"},
        "ftl_freight": {"min": 1500, "avg": 2500, "max": 4000, "unit": "per load"},
        "last_mile_delivery": {"min": 5, "avg": 10, "max": 20, "unit": "per package"},
        "warehouse_storage": {"min": 10, "avg": 25, "max": 50, "unit": "per pallet/month"},
        "fulfillment_fee": {"min": 2, "avg": 4, "max": 8, "unit": "per order"},
    },
    "law_firms": {
        "expert_witness": {"min": 300, "avg": 600, "max": 1500, "unit": "per hour"},
        "e_discovery_review": {"min": 100, "avg": 250, "max": 400, "unit": "per hour"},
        "court_reporter": {"min": 200, "avg": 400, "max": 800, "unit": "per day"},
        "investigation_services": {"min": 75, "avg": 150, "max": 300, "unit": "per hour"},
        "legal_research": {"min": 50, "avg": 100, "max": 200, "unit": "per hour"},
    },
}


# =============================================================================
# Fraud Pattern Definitions
# =============================================================================

FRAUD_PATTERNS = {
    "inflated_pricing": {
        "description": "Invoice price significantly above market rate",
        "threshold_percent": 30,
        "severity": "high",
        "indicators": [
            "Price more than 30% above market average",
            "No justification for premium pricing",
            "Vendor history shows pattern of overcharging",
        ],
    },
    "duplicate_billing": {
        "description": "Same work billed multiple times",
        "severity": "critical",
        "indicators": [
            "Multiple invoices for same service on same date",
            "Invoice numbers with sequential patterns",
            "Work description matches previous invoice",
        ],
    },
    "phantom_work": {
        "description": "Billing for work not performed",
        "severity": "critical",
        "indicators": [
            "No work order or service request on file",
            "No before/after documentation",
            "Tenant/property manager unaware of work",
        ],
    },
    "unnecessary_services": {
        "description": "Vendor recommending unnecessary work",
        "severity": "medium",
        "indicators": [
            "Full replacement recommended when repair would suffice",
            "Multiple return visits for same issue",
            "Scope creep from original work order",
        ],
    },
    "kickback_scheme": {
        "description": "Internal staff receiving kickbacks",
        "severity": "critical",
        "indicators": [
            "Single vendor receives disproportionate work",
            "Approvals bypassing normal process",
            "Vendor only works with specific staff",
        ],
    },
}


# =============================================================================
# Tool Functions
# =============================================================================

async def analyze_invoice_for_fraud(
    invoice_data: Dict[str, Any],
    industry: str = "property_management"
) -> Dict[str, Any]:
    """
    Analyze an invoice for potential fraud indicators
    
    Args:
        invoice_data: Invoice details including line items
        industry: Industry context for benchmarks
    
    Returns:
        Fraud analysis result with indicators and recommendations
    """
    logger.info("Analyzing invoice for fraud", 
               vendor=invoice_data.get("vendor_name"),
               industry=industry)
    
    result = {
        "fraud_indicators": [],
        "risk_score": 0,
        "recommendations": [],
        "pricing_analysis": [],
    }
    
    line_items = invoice_data.get("line_items", [])
    benchmarks = INDUSTRY_BENCHMARKS.get(industry, {})
    
    total_deviation = 0
    items_analyzed = 0
    
    for item in line_items:
        description = item.get("description", "").lower()
        price = item.get("total", item.get("unit_price", 0))
        
        # Find matching benchmark
        for service_type, benchmark in benchmarks.items():
            if service_type.replace("_", " ") in description:
                max_price = benchmark["max"]
                avg_price = benchmark["avg"]
                
                if max_price > 0 and price > max_price:
                    deviation = ((price - avg_price) / avg_price) * 100 if avg_price > 0 else 100
                    total_deviation += deviation
                    items_analyzed += 1
                    
                    result["pricing_analysis"].append({
                        "item": item.get("description"),
                        "charged": price,
                        "benchmark_avg": avg_price,
                        "benchmark_max": max_price,
                        "deviation_percent": round(deviation, 1),
                        "status": "above_market" if price > max_price else "within_range",
                    })
                    
                    if deviation > 30:
                        result["fraud_indicators"].append({
                            "type": "inflated_pricing",
                            "severity": "high" if deviation > 50 else "medium",
                            "item": item.get("description"),
                            "deviation": f"{round(deviation, 1)}% above average",
                        })
                break
    
    # Calculate overall risk score
    if items_analyzed > 0:
        avg_deviation = total_deviation / items_analyzed
        result["risk_score"] = min(int(avg_deviation), 100)
    
    # Add recommendations based on findings
    if result["risk_score"] >= 50:
        result["recommendations"].append("Request itemized breakdown with labor hours and material costs")
        result["recommendations"].append("Obtain competitive quotes from alternative vendors")
        result["recommendations"].append("Verify work completion before payment")
    elif result["risk_score"] >= 25:
        result["recommendations"].append("Review invoice details carefully")
        result["recommendations"].append("Compare with historical invoices from this vendor")
    
    logger.info("Invoice analysis complete",
               risk_score=result["risk_score"],
               indicators=len(result["fraud_indicators"]))
    
    return result


async def get_pricing_benchmark(
    service_type: str,
    industry: str = "property_management"
) -> Dict[str, Any]:
    """
    Get pricing benchmark for a service type
    
    Args:
        service_type: Type of service (e.g., "hvac_repair", "plumbing_repair")
        industry: Industry context
    
    Returns:
        Benchmark pricing data
    """
    benchmarks = INDUSTRY_BENCHMARKS.get(industry, {})
    
    # Try exact match first
    if service_type in benchmarks:
        return {
            "service_type": service_type,
            "industry": industry,
            "benchmark": benchmarks[service_type],
            "found": True,
        }
    
    # Try fuzzy match
    service_lower = service_type.lower().replace(" ", "_")
    for key, value in benchmarks.items():
        if key in service_lower or service_lower in key:
            return {
                "service_type": service_type,
                "matched_to": key,
                "industry": industry,
                "benchmark": value,
                "found": True,
            }
    
    return {
        "service_type": service_type,
        "industry": industry,
        "found": False,
        "message": f"No benchmark found for {service_type} in {industry}",
    }


async def check_duplicate_invoices(
    vendor_id: str,
    invoice_number: str,
    amount: float,
    date: str,
    existing_invoices: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Check for potential duplicate invoices
    
    Args:
        vendor_id: Vendor identifier
        invoice_number: Invoice number to check
        amount: Invoice amount
        date: Invoice date
        existing_invoices: List of existing invoices to check against
    
    Returns:
        Duplicate check result
    """
    duplicates = []
    
    for existing in existing_invoices:
        # Check for exact duplicates
        if existing.get("invoice_number") == invoice_number:
            duplicates.append({
                "type": "exact_duplicate",
                "matching_invoice": existing.get("invoice_number"),
                "severity": "critical",
            })
            continue
        
        # Check for amount and date match
        if (existing.get("vendor_id") == vendor_id and
            existing.get("amount") == amount and
            existing.get("date") == date):
            duplicates.append({
                "type": "potential_duplicate",
                "matching_invoice": existing.get("invoice_number"),
                "reason": "Same vendor, amount, and date",
                "severity": "high",
            })
        
        # Check for similar amount within date range
        if (existing.get("vendor_id") == vendor_id and
            abs(existing.get("amount", 0) - amount) < amount * 0.05):  # Within 5%
            duplicates.append({
                "type": "similar_invoice",
                "matching_invoice": existing.get("invoice_number"),
                "reason": "Similar amount from same vendor",
                "severity": "medium",
            })
    
    return {
        "vendor_id": vendor_id,
        "invoice_number": invoice_number,
        "duplicates_found": len(duplicates),
        "duplicates": duplicates,
        "is_duplicate": len([d for d in duplicates if d["type"] == "exact_duplicate"]) > 0,
        "needs_review": len(duplicates) > 0,
    }


async def calculate_vendor_risk_score(
    vendor_id: str,
    vendor_history: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate overall risk score for a vendor
    
    Args:
        vendor_id: Vendor identifier
        vendor_history: Historical data about the vendor
    
    Returns:
        Vendor risk assessment
    """
    risk_score = 0
    risk_factors = []
    
    # Check fraud flags
    fraud_flags = vendor_history.get("fraud_flags_count", 0)
    if fraud_flags > 0:
        risk_score += fraud_flags * 20
        risk_factors.append(f"{fraud_flags} previous fraud flags")
    
    # Check pricing deviation history
    avg_deviation = vendor_history.get("average_price_deviation", 0)
    if avg_deviation > 30:
        risk_score += 25
        risk_factors.append(f"Average {avg_deviation}% price deviation")
    elif avg_deviation > 20:
        risk_score += 15
        risk_factors.append(f"Moderate price deviation ({avg_deviation}%)")
    
    # Check complaint history
    complaints = vendor_history.get("complaints_count", 0)
    if complaints > 5:
        risk_score += 20
        risk_factors.append(f"{complaints} complaints on record")
    elif complaints > 2:
        risk_score += 10
        risk_factors.append(f"{complaints} complaints on record")
    
    # Check invoice dispute rate
    dispute_rate = vendor_history.get("dispute_rate", 0)
    if dispute_rate > 0.1:
        risk_score += 15
        risk_factors.append(f"{dispute_rate * 100}% invoice dispute rate")
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = "critical"
    elif risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 25:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "vendor_id": vendor_id,
        "risk_score": min(risk_score, 100),
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendation": _get_vendor_recommendation(risk_level),
        "assessed_at": datetime.utcnow().isoformat(),
    }


def _get_vendor_recommendation(risk_level: str) -> str:
    """Get recommendation based on risk level"""
    recommendations = {
        "critical": "BLOCK: Do not approve new invoices. Conduct vendor audit.",
        "high": "REVIEW: All invoices require manager approval and documentation verification.",
        "medium": "MONITOR: Regular spot checks on invoices. Track pricing patterns.",
        "low": "STANDARD: Normal approval process. Continue monitoring.",
    }
    return recommendations.get(risk_level, "REVIEW: Manual assessment required.")


async def register_vendor_fraud_tools(agent):
    """Register vendor fraud tools with an agent"""
    agent.register_tool(
        "analyze_invoice_fraud",
        analyze_invoice_for_fraud,
        "Analyze an invoice for fraud indicators including overpricing, duplicates, and suspicious patterns"
    )
    agent.register_tool(
        "get_pricing_benchmark",
        get_pricing_benchmark,
        "Get market pricing benchmarks for a specific service type and industry"
    )
    agent.register_tool(
        "check_duplicate_invoices",
        check_duplicate_invoices,
        "Check if an invoice may be a duplicate of an existing invoice"
    )
    agent.register_tool(
        "calculate_vendor_risk",
        calculate_vendor_risk_score,
        "Calculate the overall fraud risk score for a vendor based on their history"
    )
    
    logger.info("Vendor fraud tools registered", agent_id=agent.agent_id)





