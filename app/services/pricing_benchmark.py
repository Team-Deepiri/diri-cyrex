"""
Universal Pricing Benchmark Engine
Real-time market rate comparisons across all 6 industries

This engine aggregates pricing from all industries to create:
- Real-time market rate comparisons
- Industry-specific pricing data
- Location-based pricing adjustments
- Historical pricing trends
- Cross-industry pricing intelligence

Works for:
- Property Management (HVAC, plumbing, electrical rates)
- Corporate Procurement (supplier pricing)
- P&C Insurance (auto body shop, home repair rates)
- General Contractors (subcontractor, material rates)
- Retail/E-Commerce (freight carrier rates)
- Law Firms (expert witness, e-discovery rates)
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.types import IndustryNiche
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger

logger = get_logger("cyrex.pricing_benchmark")


class PricingTier(str, Enum):
    """Pricing tier levels"""
    LOW = "low"  # Bottom 25%
    MEDIUM = "medium"  # 25-75%
    HIGH = "high"  # Top 25%
    OUTLIER = "outlier"  # Extreme values


@dataclass
class PricingBenchmark:
    """Pricing benchmark data"""
    service_category: str
    industry: IndustryNiche
    location: Optional[str] = None
    # Pricing statistics
    median_price: float = 0.0
    mean_price: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_90: float = 0.0
    # Sample size
    sample_count: int = 0
    # Time range
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriceComparison:
    """Price comparison result"""
    invoice_price: float
    benchmark: PricingBenchmark
    deviation_percent: float  # Positive = overpriced, Negative = underpriced
    deviation_amount: float
    tier: PricingTier
    is_overpriced: bool
    confidence: float
    recommendations: List[str] = field(default_factory=list)


class PricingBenchmarkEngine:
    """
    Universal Pricing Benchmark Engine
    
    Aggregates pricing from all industries to provide:
    1. Real-time market rate comparisons
    2. Industry-specific pricing
    3. Location-based adjustments
    4. Historical pricing trends
    5. Cross-industry pricing intelligence
    """
    
    def __init__(self):
        self.logger = logger
        self._postgres = None
        self._cache: Dict[str, PricingBenchmark] = {}
        self._cache_ttl = timedelta(hours=1)
        
    async def initialize(self):
        """Initialize database connection"""
        if self._postgres is None:
            self._postgres = await get_postgres_manager()
    
    async def compare_price(
        self,
        invoice_price: float,
        service_category: str,
        industry: IndustryNiche,
        location: Optional[str] = None,
        invoice_date: Optional[datetime] = None
    ) -> PriceComparison:
        """
        Compare invoice price against market benchmarks
        
        Args:
            invoice_price: Price from invoice
            service_category: Service category (e.g., "HVAC repair", "freight shipping")
            industry: Industry niche
            location: Geographic location (optional)
            invoice_date: Invoice date for historical comparison
            
        Returns:
            PriceComparison with deviation analysis
        """
        await self.initialize()
        
        # Get benchmark
        benchmark = await self.get_benchmark(
            service_category=service_category,
            industry=industry,
            location=location,
            invoice_date=invoice_date
        )
        
        if not benchmark or benchmark.sample_count == 0:
            return PriceComparison(
                invoice_price=invoice_price,
                benchmark=benchmark or PricingBenchmark(service_category=service_category, industry=industry),
                deviation_percent=0.0,
                deviation_amount=0.0,
                tier=PricingTier.MEDIUM,
                is_overpriced=False,
                confidence=0.0,
                recommendations=["Insufficient benchmark data"]
            )
        
        # Calculate deviation
        median_price = benchmark.median_price
        deviation_amount = invoice_price - median_price
        deviation_percent = (deviation_amount / median_price * 100) if median_price > 0 else 0.0
        
        # Determine tier
        if invoice_price <= benchmark.percentile_25:
            tier = PricingTier.LOW
        elif invoice_price <= benchmark.percentile_75:
            tier = PricingTier.MEDIUM
        elif invoice_price <= benchmark.percentile_90:
            tier = PricingTier.HIGH
        else:
            tier = PricingTier.OUTLIER
        
        # Determine if overpriced (above 75th percentile)
        is_overpriced = invoice_price > benchmark.percentile_75
        
        # Calculate confidence based on sample size
        confidence = min(1.0, benchmark.sample_count / 100.0)  # More samples = higher confidence
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            invoice_price,
            benchmark,
            deviation_percent,
            tier,
            is_overpriced
        )
        
        return PriceComparison(
            invoice_price=invoice_price,
            benchmark=benchmark,
            deviation_percent=deviation_percent,
            deviation_amount=deviation_amount,
            tier=tier,
            is_overpriced=is_overpriced,
            confidence=confidence,
            recommendations=recommendations
        )
    
    async def get_benchmark(
        self,
        service_category: str,
        industry: IndustryNiche,
        location: Optional[str] = None,
        invoice_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Optional[PricingBenchmark]:
        """
        Get pricing benchmark for service category
        
        Args:
            service_category: Service category
            industry: Industry niche
            location: Geographic location (optional)
            invoice_date: Date for historical comparison
            use_cache: Whether to use cached benchmark
            
        Returns:
            PricingBenchmark or None if insufficient data
        """
        await self.initialize()
        
        # Check cache
        cache_key = f"{industry.value}:{service_category}:{location or 'global'}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.utcnow() - cached.last_updated < self._cache_ttl:
                return cached
        
        # Query database for pricing data
        # This would query the vendor_intelligence database for historical invoice prices
        benchmark = await self._calculate_benchmark_from_db(
            service_category=service_category,
            industry=industry,
            location=location,
            invoice_date=invoice_date
        )
        
        if benchmark:
            self._cache[cache_key] = benchmark
        
        return benchmark
    
    async def _calculate_benchmark_from_db(
        self,
        service_category: str,
        industry: IndustryNiche,
        location: Optional[str] = None,
        invoice_date: Optional[datetime] = None
    ) -> Optional[PricingBenchmark]:
        """
        Calculate benchmark from database
        
        Queries historical invoice data to calculate pricing statistics
        """
        try:
            # Query invoices for this service category and industry
            # This is a simplified query - actual implementation would be more complex
            
            query = """
            SELECT 
                total_amount,
                invoice_date,
                location
            FROM agent_interactions
            WHERE metadata->>'service_category' = $1
            AND metadata->>'industry' = $2
            AND metadata->>'total_amount' IS NOT NULL
            """
            
            params = [service_category, industry.value]
            
            if location:
                query += " AND metadata->>'location' = $3"
                params.append(location)
            
            if invoice_date:
                # Use historical data within 6 months
                date_start = invoice_date - timedelta(days=180)
                query += f" AND invoice_date >= ${len(params) + 1}"
                params.append(date_start)
            
            query += " ORDER BY invoice_date DESC LIMIT 1000"
            
            rows = await self._postgres.fetch(query, params)
            
            if not rows or len(rows) < 5:  # Need at least 5 samples
                return None
            
            # Extract prices
            prices = []
            for row in rows:
                try:
                    amount = float(row.get("total_amount") or row.get("metadata", {}).get("total_amount", 0))
                    if amount > 0:
                        prices.append(amount)
                except (ValueError, TypeError):
                    continue
            
            if len(prices) < 5:
                return None
            
            # Calculate statistics
            prices_sorted = sorted(prices)
            n = len(prices_sorted)
            
            median_price = prices_sorted[n // 2] if n % 2 == 1 else (prices_sorted[n // 2 - 1] + prices_sorted[n // 2]) / 2
            mean_price = sum(prices) / n
            min_price = prices_sorted[0]
            max_price = prices_sorted[-1]
            percentile_25 = prices_sorted[int(n * 0.25)]
            percentile_75 = prices_sorted[int(n * 0.75)]
            percentile_90 = prices_sorted[int(n * 0.90)]
            
            # Get date range
            dates = [row.get("invoice_date") for row in rows if row.get("invoice_date")]
            date_range_start = min(dates) if dates else None
            date_range_end = max(dates) if dates else None
            
            return PricingBenchmark(
                service_category=service_category,
                industry=industry,
                location=location,
                median_price=median_price,
                mean_price=mean_price,
                min_price=min_price,
                max_price=max_price,
                percentile_25=percentile_25,
                percentile_75=percentile_75,
                percentile_90=percentile_90,
                sample_count=n,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate benchmark: {e}", exc_info=True)
            return None
    
    def _generate_recommendations(
        self,
        invoice_price: float,
        benchmark: PricingBenchmark,
        deviation_percent: float,
        tier: PricingTier,
        is_overpriced: bool
    ) -> List[str]:
        """Generate recommendations based on price comparison"""
        recommendations = []
        
        if is_overpriced:
            if deviation_percent > 50:
                recommendations.append(f"Price is {deviation_percent:.1f}% above market median - HIGH RISK")
                recommendations.append("Consider requesting detailed breakdown and justification")
            elif deviation_percent > 25:
                recommendations.append(f"Price is {deviation_percent:.1f}% above market median - MODERATE RISK")
                recommendations.append("Verify pricing with vendor and compare to market rates")
            else:
                recommendations.append(f"Price is {deviation_percent:.1f}% above market median")
                recommendations.append("Price is within acceptable range but above average")
        else:
            if deviation_percent < -25:
                recommendations.append(f"Price is {abs(deviation_percent):.1f}% below market median")
                recommendations.append("Price appears competitive - verify service quality")
            else:
                recommendations.append("Price is within market range")
        
        # Add tier-specific recommendations
        if tier == PricingTier.OUTLIER:
            recommendations.append("Price is in outlier range - requires investigation")
        elif tier == PricingTier.HIGH:
            recommendations.append("Price is in high tier - review vendor pricing history")
        
        return recommendations
    
    async def update_benchmark(
        self,
        service_category: str,
        industry: IndustryNiche,
        price: float,
        location: Optional[str] = None,
        invoice_date: Optional[datetime] = None
    ):
        """
        Update benchmark with new price data
        
        This would be called after processing each invoice to keep benchmarks current
        """
        # Invalidate cache
        cache_key = f"{industry.value}:{service_category}:{location or 'global'}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        # Store price data in database for future benchmark calculations
        # This would insert into a pricing_data table
        self.logger.debug(f"Updated benchmark data: {service_category} in {industry.value} with price {price}")


# Singleton instance
_engine_instance: Optional[PricingBenchmarkEngine] = None


def get_pricing_benchmark_engine() -> PricingBenchmarkEngine:
    """Get singleton instance of Pricing Benchmark Engine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PricingBenchmarkEngine()
    return _engine_instance

