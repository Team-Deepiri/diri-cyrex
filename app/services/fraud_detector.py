"""
Vendor Fraud Analyzer
Detects vendor fraud across all 6 industries using unified models

This service provides:
- Anomaly detection for invoice fraud
- Pattern matching for vendor fraud
- Predictive risk scoring
- Cross-industry fraud pattern detection

Works for all 6 industries:
1. Property Management
2. Corporate Procurement
3. P&C Insurance
4. General Contractors
5. Retail/E-Commerce
6. Law Firms

Architecture:
- One model architecture, industry-specific LoRA adapters
- Anomaly detection (autoencoder-based)
- Pattern matching (rule-based + ML)
- Predictive risk scoring (ML models)
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json

from ..core.types import IndustryNiche, VendorFraudType, RiskLevel
from ..logging_config import get_logger
from .invoice_parser import ProcessedInvoice, get_universal_invoice_processor
from .pricing_benchmark import get_pricing_benchmark_engine, PriceComparison
from .vendor_intelligence_service import get_vendor_intelligence_service, VendorProfile
from .lora_loader import get_industry_lora_service

logger = get_logger("cyrex.fraud_detector")


@dataclass
class FraudIndicator:
    """Fraud indicator detected"""
    fraud_type: VendorFraudType
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class FraudDetectionResult:
    """Fraud detection result"""
    invoice_id: str
    vendor_id: Optional[str]
    fraud_detected: bool
    risk_score: float  # 0.0 to 100.0
    risk_level: RiskLevel
    fraud_indicators: List[FraudIndicator] = field(default_factory=list)
    price_comparison: Optional[PriceComparison] = None
    vendor_profile: Optional[VendorProfile] = None
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalFraudDetectionService:
    """
    Vendor Fraud Analyzer
    
    Detects vendor fraud using:
    1. Anomaly detection (autoencoder-based)
    2. Pattern matching (rule-based + ML)
    3. Pricing benchmark comparison
    4. Vendor intelligence (cross-industry patterns)
    5. Predictive risk scoring
    """
    
    def __init__(self):
        self.logger = logger
        self.invoice_processor = get_universal_invoice_processor()
        self.pricing_engine = get_pricing_benchmark_engine()
        self.vendor_intelligence = get_vendor_intelligence_service()
        self.lora_service = get_industry_lora_service()
        self._anomaly_detector = None  # Would load autoencoder model
        self._risk_scorer = None  # Would load risk scoring model
        
    async def detect_fraud(
        self,
        invoice: ProcessedInvoice,
        use_vendor_intelligence: bool = True,
        use_pricing_benchmark: bool = True,
        use_anomaly_detection: bool = True,
        use_pattern_matching: bool = True
    ) -> FraudDetectionResult:
        """
        Detect fraud in invoice
        
        Args:
            invoice: Processed invoice
            use_vendor_intelligence: Use vendor history and cross-industry data
            use_pricing_benchmark: Compare against market rates
            use_anomaly_detection: Use ML anomaly detection
            use_pattern_matching: Use rule-based pattern matching
            
        Returns:
            FraudDetectionResult with detection findings
        """
        try:
            fraud_indicators = []
            risk_factors = []
            
            # 1. Pricing benchmark comparison
            if use_pricing_benchmark and invoice.service_category:
                price_comparison = await self.pricing_engine.compare_price(
                    invoice_price=invoice.total_amount,
                    service_category=invoice.service_category,
                    industry=invoice.industry,
                    location=invoice.property_address
                )
                
                if price_comparison.is_overpriced:
                    fraud_indicators.append(FraudIndicator(
                        fraud_type=VendorFraudType.INFLATED_INVOICE,
                        severity=min(1.0, abs(price_comparison.deviation_percent) / 100.0),
                        confidence=price_comparison.confidence,
                        description=f"Invoice price is {price_comparison.deviation_percent:.1f}% above market median",
                        evidence={
                            "deviation_percent": price_comparison.deviation_percent,
                            "deviation_amount": price_comparison.deviation_amount,
                            "benchmark_median": price_comparison.benchmark.median_price,
                            "tier": price_comparison.tier.value
                        },
                        recommendation="Request detailed breakdown and compare to market rates"
                    ))
                    risk_factors.append(price_comparison.deviation_percent / 100.0)
            else:
                price_comparison = None
            
            # 2. Vendor intelligence check
            vendor_profile = None
            if use_vendor_intelligence and invoice.vendor_id:
                vendor_profile = await self.vendor_intelligence.get_vendor_profile(invoice.vendor_id)
                
                if vendor_profile:
                    # Check for cross-industry flags
                    if vendor_profile.cross_industry_flags > 0:
                        fraud_indicators.append(FraudIndicator(
                            fraud_type=VendorFraudType.INFLATED_INVOICE,
                            severity=0.7,
                            confidence=0.8,
                            description=f"Vendor has {vendor_profile.cross_industry_flags} fraud flags across {len(vendor_profile.flagged_by_industries)} industries",
                            evidence={
                                "cross_industry_flags": vendor_profile.cross_industry_flags,
                                "flagged_industries": vendor_profile.flagged_by_industries,
                                "fraud_history": vendor_profile.fraud_flags_count
                            },
                            recommendation="High-risk vendor - review carefully"
                        ))
                        risk_factors.append(0.7)
                    
                    # Check fraud history
                    if vendor_profile.fraud_flags_count > 3:
                        fraud_indicators.append(FraudIndicator(
                            fraud_type=VendorFraudType.INFLATED_INVOICE,
                            severity=0.6,
                            confidence=0.75,
                            description=f"Vendor has {vendor_profile.fraud_flags_count} previous fraud flags",
                            evidence={
                                "fraud_flags_count": vendor_profile.fraud_flags_count,
                                "fraud_types": vendor_profile.fraud_types_detected
                            },
                            recommendation="Vendor with fraud history - verify all charges"
                        ))
                        risk_factors.append(0.6)
                    
                    # Check risk score
                    if vendor_profile.current_risk_score > 70:
                        fraud_indicators.append(FraudIndicator(
                            fraud_type=VendorFraudType.INFLATED_INVOICE,
                            severity=0.5,
                            confidence=0.7,
                            description=f"Vendor has high risk score: {vendor_profile.current_risk_score:.1f}/100",
                            evidence={
                                "risk_score": vendor_profile.current_risk_score,
                                "risk_level": vendor_profile.risk_level
                            },
                            recommendation="High-risk vendor - additional review recommended"
                        ))
                        risk_factors.append(vendor_profile.current_risk_score / 100.0)
            
            # 3. Pattern matching (rule-based)
            if use_pattern_matching:
                pattern_indicators = await self._detect_fraud_patterns(invoice)
                fraud_indicators.extend(pattern_indicators)
                for indicator in pattern_indicators:
                    risk_factors.append(indicator.severity)
            
            # 4. Anomaly detection (ML-based)
            if use_anomaly_detection:
                anomaly_score = await self._detect_anomalies(invoice)
                if anomaly_score > 0.7:
                    fraud_indicators.append(FraudIndicator(
                        fraud_type=VendorFraudType.INFLATED_INVOICE,
                        severity=anomaly_score,
                        confidence=0.65,
                        description="Invoice shows anomalous patterns compared to historical data",
                        evidence={"anomaly_score": anomaly_score},
                        recommendation="Review invoice for unusual patterns"
                    ))
                    risk_factors.append(anomaly_score)
            
            # Calculate overall risk score
            if risk_factors:
                risk_score = sum(risk_factors) / len(risk_factors) * 100.0
                risk_score = min(100.0, risk_score)  # Cap at 100
            else:
                risk_score = 0.0
            
            # Determine risk level
            if risk_score >= 80:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 60:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 40:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Determine if fraud detected
            fraud_detected = len(fraud_indicators) > 0 and risk_score >= 40
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                fraud_indicators,
                risk_score,
                price_comparison,
                vendor_profile
            )
            
            # Calculate confidence
            if fraud_indicators:
                confidence = sum(ind.confidence for ind in fraud_indicators) / len(fraud_indicators)
            else:
                confidence = 1.0 - (risk_score / 100.0)  # Higher confidence for lower risk
            
            return FraudDetectionResult(
                invoice_id=invoice.invoice_id,
                vendor_id=invoice.vendor_id,
                fraud_detected=fraud_detected,
                risk_score=risk_score,
                risk_level=risk_level,
                fraud_indicators=fraud_indicators,
                price_comparison=price_comparison,
                vendor_profile=vendor_profile,
                recommendations=recommendations,
                confidence=confidence,
                metadata={
                    "industry": invoice.industry.value,
                    "service_category": invoice.service_category,
                    "total_amount": invoice.total_amount
                }
            )
            
        except Exception as e:
            self.logger.error(f"Fraud detection failed: {e}", exc_info=True)
            raise
    
    async def _detect_fraud_patterns(self, invoice: ProcessedInvoice) -> List[FraudIndicator]:
        """Detect fraud using rule-based pattern matching"""
        indicators = []
        
        # Check for duplicate line items
        if len(invoice.line_items) > 1:
            descriptions = [item.description.lower() for item in invoice.line_items]
            if len(descriptions) != len(set(descriptions)):
                indicators.append(FraudIndicator(
                    fraud_type=VendorFraudType.DUPLICATE_BILLING,
                    severity=0.8,
                    confidence=0.9,
                    description="Duplicate line items detected",
                    evidence={"duplicate_count": len(descriptions) - len(set(descriptions))},
                    recommendation="Review for duplicate charges"
                ))
        
        # Check for suspicious round numbers (common in fraud)
        if invoice.total_amount % 1000 == 0 and invoice.total_amount > 1000:
            indicators.append(FraudIndicator(
                fraud_type=VendorFraudType.INFLATED_INVOICE,
                severity=0.3,
                confidence=0.4,
                description="Invoice total is a round number (potential fraud indicator)",
                evidence={"total_amount": invoice.total_amount},
                recommendation="Verify invoice details"
            ))
        
        # Check for missing critical information
        missing_fields = []
        if not invoice.invoice_date:
            missing_fields.append("invoice_date")
        if not invoice.vendor_name:
            missing_fields.append("vendor_name")
        if not invoice.line_items:
            missing_fields.append("line_items")
        
        if missing_fields:
            indicators.append(FraudIndicator(
                fraud_type=VendorFraudType.FORGED_DOCUMENTS,
                severity=0.5,
                confidence=0.6,
                description=f"Missing critical invoice fields: {', '.join(missing_fields)}",
                evidence={"missing_fields": missing_fields},
                recommendation="Request complete invoice documentation"
            ))
        
        return indicators
    
    async def _detect_anomalies(self, invoice: ProcessedInvoice) -> float:
        """
        Detect anomalies using ML models (autoencoder)
        
        Returns anomaly score (0.0 to 1.0)
        """
        try:
            # In production, would use loaded autoencoder model
            # For now, return placeholder based on heuristics
            
            # Simple heuristic: check if invoice amount is unusual
            # In production, would use actual autoencoder
            if invoice.total_amount > 100000:
                return 0.6  # Large invoices are somewhat anomalous
            elif invoice.total_amount < 10:
                return 0.4  # Very small invoices are somewhat anomalous
            else:
                return 0.2  # Normal range
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            return 0.0
    
    def _generate_recommendations(
        self,
        fraud_indicators: List[FraudIndicator],
        risk_score: float,
        price_comparison: Optional[PriceComparison],
        vendor_profile: Optional[VendorProfile]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_score >= 80:
            recommendations.append("CRITICAL: Immediate review required - high fraud risk detected")
        elif risk_score >= 60:
            recommendations.append("HIGH RISK: Detailed review recommended")
        
        if price_comparison and price_comparison.is_overpriced:
            recommendations.extend(price_comparison.recommendations)
        
        if vendor_profile and vendor_profile.cross_industry_flags > 0:
            recommendations.append(f"Vendor flagged in {len(vendor_profile.flagged_by_industries)} other industries")
        
        # Add specific recommendations from fraud indicators
        for indicator in fraud_indicators:
            if indicator.recommendation:
                recommendations.append(indicator.recommendation)
        
        if not recommendations:
            recommendations.append("No significant fraud indicators detected - standard review process")
        
        return recommendations


# Singleton instance
_fraud_service_instance: Optional[UniversalFraudDetectionService] = None


def get_universal_fraud_detection_service() -> UniversalFraudDetectionService:
    """Get singleton instance of Vendor Fraud Analyzer"""
    global _fraud_service_instance
    if _fraud_service_instance is None:
        _fraud_service_instance = UniversalFraudDetectionService()
    return _fraud_service_instance

