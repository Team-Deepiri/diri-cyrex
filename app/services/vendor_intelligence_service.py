"""
Cyrex Vendor Intelligence Service
Comprehensive vendor intelligence database with cross-industry tracking

This is the CORE of Cyrex - not just fraud detection, but complete vendor intelligence:
- Cross-industry vendor profiles
- Performance tracking across all industries
- Predictive risk scoring
- Network effects (vendors flagged in one industry help all)
- Pricing benchmark aggregation
- Fraud pattern detection
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import uuid
import json
from enum import Enum

from ..core.types import IndustryNiche, RiskLevel, VendorFraudType
from ..database.postgres import get_postgres_manager
from ..logging_config import get_logger
from ..integrations.universal_rag_engine import create_universal_rag_engine

logger = get_logger("cyrex.vendor_intelligence")


class VendorStatus(str, Enum):
    """Vendor status"""
    ACTIVE = "active"
    FLAGGED = "flagged"
    BLOCKED = "blocked"
    MONITORED = "monitored"
    VERIFIED = "verified"


@dataclass
class VendorProfile:
    """Comprehensive vendor profile"""
    vendor_id: str
    vendor_name: str
    # Cross-industry tracking
    industries_served: List[str] = field(default_factory=list)
    total_invoices_analyzed: int = 0
    total_invoice_amount: float = 0.0
    # Fraud tracking
    fraud_flags_count: int = 0
    fraud_flags_by_industry: Dict[str, int] = field(default_factory=dict)
    fraud_types_detected: List[str] = field(default_factory=list)
    # Performance metrics
    average_invoice_amount: float = 0.0
    average_price_deviation: float = 0.0
    pricing_deviation_history: List[float] = field(default_factory=list)
    # Risk scoring
    current_risk_score: float = 0.0
    risk_level: str = RiskLevel.LOW.value
    risk_history: List[Dict[str, Any]] = field(default_factory=list)
    # Network effects
    cross_industry_flags: int = 0
    flagged_by_industries: List[str] = field(default_factory=list)
    # Metadata
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    status: str = VendorStatus.ACTIVE.value
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvoiceRecord:
    """Invoice record for tracking"""
    invoice_id: str
    vendor_id: str
    vendor_name: str
    industry: str
    invoice_number: Optional[str]
    invoice_date: datetime
    total_amount: float
    service_category: Optional[str]
    fraud_detected: bool
    risk_score: float
    risk_level: str
    fraud_indicators: List[Dict[str, Any]]
    analyzed_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class VendorIntelligenceService:
    """
    Cyrex Vendor Intelligence Service
    
    Core capabilities:
    1. Cross-industry vendor tracking
    2. Vendor profile management
    3. Predictive risk scoring
    4. Network effects (cross-industry flags)
    5. Performance analytics
    6. Pricing benchmark aggregation
    """
    
    def __init__(self):
        self.logger = logger
        self._rag_engine = None
        self._vendor_cache: Dict[str, VendorProfile] = {}
    
    async def initialize(self):
        """Initialize the service"""
        try:
            # Initialize database tables
            await self._create_tables()
            logger.info("Vendor Intelligence Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}", exc_info=True)
            raise
    
    async def _create_tables(self):
        """Create database tables for vendor intelligence"""
        postgres = await get_postgres_manager()
        
        # Vendors table
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS cyrex_vendors (
                vendor_id VARCHAR(255) PRIMARY KEY,
                vendor_name VARCHAR(500) NOT NULL,
                industries_served JSONB DEFAULT '[]',
                total_invoices_analyzed INTEGER DEFAULT 0,
                total_invoice_amount DECIMAL(15, 2) DEFAULT 0,
                fraud_flags_count INTEGER DEFAULT 0,
                fraud_flags_by_industry JSONB DEFAULT '{}',
                fraud_types_detected JSONB DEFAULT '[]',
                average_invoice_amount DECIMAL(15, 2) DEFAULT 0,
                average_price_deviation DECIMAL(10, 2) DEFAULT 0,
                pricing_deviation_history JSONB DEFAULT '[]',
                current_risk_score DECIMAL(5, 2) DEFAULT 0,
                risk_level VARCHAR(20) DEFAULT 'low',
                risk_history JSONB DEFAULT '[]',
                cross_industry_flags INTEGER DEFAULT 0,
                flagged_by_industries JSONB DEFAULT '[]',
                first_seen TIMESTAMP DEFAULT NOW(),
                last_activity TIMESTAMP DEFAULT NOW(),
                status VARCHAR(20) DEFAULT 'active',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_vendors_name ON cyrex_vendors(vendor_name);
            CREATE INDEX IF NOT EXISTS idx_vendors_risk ON cyrex_vendors(risk_level, current_risk_score);
            CREATE INDEX IF NOT EXISTS idx_vendors_status ON cyrex_vendors(status);
            CREATE INDEX IF NOT EXISTS idx_vendors_industries ON cyrex_vendors USING GIN(industries_served);
        """)
        
        # Invoices table
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS cyrex_invoices (
                invoice_id VARCHAR(255) PRIMARY KEY,
                vendor_id VARCHAR(255) NOT NULL,
                vendor_name VARCHAR(500) NOT NULL,
                industry VARCHAR(100) NOT NULL,
                invoice_number VARCHAR(255),
                invoice_date TIMESTAMP,
                total_amount DECIMAL(15, 2) NOT NULL,
                service_category VARCHAR(255),
                fraud_detected BOOLEAN DEFAULT FALSE,
                risk_score DECIMAL(5, 2) DEFAULT 0,
                risk_level VARCHAR(20) DEFAULT 'low',
                fraud_indicators JSONB DEFAULT '[]',
                analyzed_at TIMESTAMP DEFAULT NOW(),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_invoices_vendor ON cyrex_invoices(vendor_id);
            CREATE INDEX IF NOT EXISTS idx_invoices_industry ON cyrex_invoices(industry);
            CREATE INDEX IF NOT EXISTS idx_invoices_date ON cyrex_invoices(invoice_date);
            CREATE INDEX IF NOT EXISTS idx_invoices_fraud ON cyrex_invoices(fraud_detected);
        """)
        
        # Pricing benchmarks table
        await postgres.execute("""
            CREATE TABLE IF NOT EXISTS cyrex_pricing_benchmarks (
                benchmark_id VARCHAR(255) PRIMARY KEY,
                service_type VARCHAR(255) NOT NULL,
                industry VARCHAR(100) NOT NULL,
                location VARCHAR(255),
                min_price DECIMAL(15, 2),
                avg_price DECIMAL(15, 2) NOT NULL,
                max_price DECIMAL(15, 2),
                unit VARCHAR(50),
                sample_size INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT NOW(),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_benchmarks_service ON cyrex_pricing_benchmarks(service_type, industry);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_benchmarks_unique ON cyrex_pricing_benchmarks(service_type, industry, location);
        """)
        
        logger.info("Vendor intelligence database tables created")
    
    async def get_or_create_vendor(
        self,
        vendor_id: Optional[str],
        vendor_name: str,
        industry: IndustryNiche
    ) -> VendorProfile:
        """Get or create vendor profile"""
        # Generate vendor_id if not provided
        if not vendor_id:
            vendor_id = f"vendor_{uuid.uuid4().hex[:12]}"
        
        # Check cache first
        if vendor_id in self._vendor_cache:
            return self._vendor_cache[vendor_id]
        
        # Check database
        postgres = await get_postgres_manager()
        row = await postgres.fetchrow(
            "SELECT * FROM cyrex_vendors WHERE vendor_id = $1",
            vendor_id
        )
        
        if row:
            profile = self._row_to_vendor_profile(row)
        else:
            # Create new vendor
            profile = VendorProfile(
                vendor_id=vendor_id,
                vendor_name=vendor_name,
                industries_served=[industry.value],
            )
            await self.save_vendor(profile)
        
        self._vendor_cache[vendor_id] = profile
        return profile
    
    async def save_vendor(self, profile: VendorProfile):
        """Save vendor profile to database"""
        postgres = await get_postgres_manager()
        
        await postgres.execute("""
            INSERT INTO cyrex_vendors (
                vendor_id, vendor_name, industries_served, total_invoices_analyzed,
                total_invoice_amount, fraud_flags_count, fraud_flags_by_industry,
                fraud_types_detected, average_invoice_amount, average_price_deviation,
                pricing_deviation_history, current_risk_score, risk_level, risk_history,
                cross_industry_flags, flagged_by_industries, first_seen, last_activity,
                status, metadata, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, NOW()
            )
            ON CONFLICT (vendor_id) DO UPDATE SET
                vendor_name = EXCLUDED.vendor_name,
                industries_served = EXCLUDED.industries_served,
                total_invoices_analyzed = EXCLUDED.total_invoices_analyzed,
                total_invoice_amount = EXCLUDED.total_invoice_amount,
                fraud_flags_count = EXCLUDED.fraud_flags_count,
                fraud_flags_by_industry = EXCLUDED.fraud_flags_by_industry,
                fraud_types_detected = EXCLUDED.fraud_types_detected,
                average_invoice_amount = EXCLUDED.average_invoice_amount,
                average_price_deviation = EXCLUDED.average_price_deviation,
                pricing_deviation_history = EXCLUDED.pricing_deviation_history,
                current_risk_score = EXCLUDED.current_risk_score,
                risk_level = EXCLUDED.risk_level,
                risk_history = EXCLUDED.risk_history,
                cross_industry_flags = EXCLUDED.cross_industry_flags,
                flagged_by_industries = EXCLUDED.flagged_by_industries,
                last_activity = EXCLUDED.last_activity,
                status = EXCLUDED.status,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """,
            profile.vendor_id,
            profile.vendor_name,
            json.dumps(profile.industries_served),
            profile.total_invoices_analyzed,
            float(profile.total_invoice_amount),
            profile.fraud_flags_count,
            json.dumps(profile.fraud_flags_by_industry),
            json.dumps(profile.fraud_types_detected),
            float(profile.average_invoice_amount),
            float(profile.average_price_deviation),
            json.dumps(profile.pricing_deviation_history),
            float(profile.current_risk_score),
            profile.risk_level,
            json.dumps(profile.risk_history),
            profile.cross_industry_flags,
            json.dumps(profile.flagged_by_industries),
            profile.first_seen,
            profile.last_activity,
            profile.status,
            json.dumps(profile.metadata),
        )
    
    async def record_invoice_analysis(
        self,
        invoice_data: Dict[str, Any],
        analysis_result: Dict[str, Any],
        industry: IndustryNiche
    ) -> InvoiceRecord:
        """Record invoice analysis and update vendor profile"""
        invoice_id = f"inv_{uuid.uuid4().hex[:12]}"
        
        # Get or create vendor
        vendor = await self.get_or_create_vendor(
            vendor_id=invoice_data.get("vendor_id"),
            vendor_name=invoice_data.get("vendor_name", "Unknown"),
            industry=industry
        )
        
        # Create invoice record
        invoice_record = InvoiceRecord(
            invoice_id=invoice_id,
            vendor_id=vendor.vendor_id,
            vendor_name=vendor.vendor_name,
            industry=industry.value,
            invoice_number=invoice_data.get("invoice_number"),
            invoice_date=datetime.fromisoformat(invoice_data.get("invoice_date", datetime.utcnow().isoformat())) if isinstance(invoice_data.get("invoice_date"), str) else invoice_data.get("invoice_date", datetime.utcnow()),
            total_amount=float(invoice_data.get("total_amount", 0)),
            service_category=invoice_data.get("service_category"),
            fraud_detected=analysis_result.get("fraud_detected", False),
            risk_score=analysis_result.get("risk_score", 0),
            risk_level=analysis_result.get("risk_level", RiskLevel.LOW.value),
            fraud_indicators=analysis_result.get("fraud_indicators", []),
            analyzed_at=datetime.utcnow(),
            metadata=analysis_result.get("metadata", {}),
        )
        
        # Save invoice record
        await self._save_invoice_record(invoice_record)
        
        # Update vendor profile
        await self._update_vendor_from_invoice(vendor, invoice_record, analysis_result, industry)
        
        return invoice_record
    
    async def _save_invoice_record(self, invoice: InvoiceRecord):
        """Save invoice record to database"""
        postgres = await get_postgres_manager()
        
        await postgres.execute("""
            INSERT INTO cyrex_invoices (
                invoice_id, vendor_id, vendor_name, industry, invoice_number,
                invoice_date, total_amount, service_category, fraud_detected,
                risk_score, risk_level, fraud_indicators, analyzed_at, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
            )
        """,
            invoice.invoice_id,
            invoice.vendor_id,
            invoice.vendor_name,
            invoice.industry,
            invoice.invoice_number,
            invoice.invoice_date,
            float(invoice.total_amount),
            invoice.service_category,
            invoice.fraud_detected,
            float(invoice.risk_score),
            invoice.risk_level,
            json.dumps(invoice.fraud_indicators),
            invoice.analyzed_at,
            json.dumps(invoice.metadata),
        )
    
    async def _update_vendor_from_invoice(
        self,
        vendor: VendorProfile,
        invoice: InvoiceRecord,
        analysis_result: Dict[str, Any],
        industry: IndustryNiche
    ):
        """Update vendor profile based on invoice analysis"""
        # Update industries served
        if industry.value not in vendor.industries_served:
            vendor.industries_served.append(industry.value)
        
        # Update invoice statistics
        vendor.total_invoices_analyzed += 1
        vendor.total_invoice_amount += invoice.total_amount
        vendor.average_invoice_amount = vendor.total_invoice_amount / vendor.total_invoices_analyzed
        
        # Update fraud flags
        if invoice.fraud_detected:
            vendor.fraud_flags_count += 1
            if industry.value not in vendor.fraud_flags_by_industry:
                vendor.fraud_flags_by_industry[industry.value] = 0
            vendor.fraud_flags_by_industry[industry.value] += 1
            
            # Track fraud types
            for indicator in invoice.fraud_indicators:
                fraud_type = indicator.get("type")
                if fraud_type and fraud_type not in vendor.fraud_types_detected:
                    vendor.fraud_types_detected.append(fraud_type)
            
            # Network effects: Flag vendor across all industries
            if industry.value not in vendor.flagged_by_industries:
                vendor.flagged_by_industries.append(industry.value)
                vendor.cross_industry_flags += 1
        
        # Update pricing deviation
        pricing_analysis = analysis_result.get("pricing_analysis", {})
        if pricing_analysis:
            deviation = pricing_analysis.get("price_deviation_percent", 0)
            if deviation:
                vendor.pricing_deviation_history.append(deviation)
                # Keep last 100 deviations
                if len(vendor.pricing_deviation_history) > 100:
                    vendor.pricing_deviation_history = vendor.pricing_deviation_history[-100:]
                vendor.average_price_deviation = sum(vendor.pricing_deviation_history) / len(vendor.pricing_deviation_history)
        
        # Recalculate risk score
        await self._calculate_vendor_risk_score(vendor)
        
        # Update last activity
        vendor.last_activity = datetime.utcnow()
        
        # Update status based on risk
        if vendor.risk_level == RiskLevel.CRITICAL.value:
            vendor.status = VendorStatus.BLOCKED.value
        elif vendor.risk_level == RiskLevel.HIGH.value:
            vendor.status = VendorStatus.FLAGGED.value
        elif vendor.fraud_flags_count > 0:
            vendor.status = VendorStatus.MONITORED.value
        
        # Save updated profile
        await self.save_vendor(vendor)
    
    async def _calculate_vendor_risk_score(self, vendor: VendorProfile):
        """Calculate predictive risk score for vendor"""
        risk_score = 0.0
        
        # Base score from fraud flags
        risk_score += vendor.fraud_flags_count * 15
        
        # Cross-industry flags (network effects)
        risk_score += vendor.cross_industry_flags * 20
        
        # Pricing deviation
        if vendor.average_price_deviation > 50:
            risk_score += 30
        elif vendor.average_price_deviation > 30:
            risk_score += 20
        elif vendor.average_price_deviation > 20:
            risk_score += 10
        
        # Industry spread (vendors serving many industries are riskier if flagged)
        if len(vendor.flagged_by_industries) > 1:
            risk_score += len(vendor.flagged_by_industries) * 10
        
        # Recent activity (recent flags weigh more)
        if vendor.fraud_flags_count > 0:
            days_since_last_flag = (datetime.utcnow() - vendor.last_activity).days
            if days_since_last_flag < 30:
                risk_score += 15
            elif days_since_last_flag < 90:
                risk_score += 10
        
        # Cap at 100
        risk_score = min(risk_score, 100.0)
        vendor.current_risk_score = risk_score
        
        # Determine risk level
        if risk_score >= 70:
            vendor.risk_level = RiskLevel.CRITICAL.value
        elif risk_score >= 50:
            vendor.risk_level = RiskLevel.HIGH.value
        elif risk_score >= 25:
            vendor.risk_level = RiskLevel.MEDIUM.value
        else:
            vendor.risk_level = RiskLevel.LOW.value
        
        # Add to risk history
        vendor.risk_history.append({
            "risk_score": risk_score,
            "risk_level": vendor.risk_level,
            "timestamp": datetime.utcnow().isoformat(),
            "factors": {
                "fraud_flags": vendor.fraud_flags_count,
                "cross_industry_flags": vendor.cross_industry_flags,
                "price_deviation": vendor.average_price_deviation,
            }
        })
        # Keep last 50 risk assessments
        if len(vendor.risk_history) > 50:
            vendor.risk_history = vendor.risk_history[-50:]
    
    async def get_vendor_profile(self, vendor_id: str) -> Optional[VendorProfile]:
        """Get vendor profile by ID"""
        if vendor_id in self._vendor_cache:
            return self._vendor_cache[vendor_id]
        
        postgres = await get_postgres_manager()
        row = await postgres.fetchrow(
            "SELECT * FROM cyrex_vendors WHERE vendor_id = $1",
            vendor_id
        )
        
        if row:
            profile = self._row_to_vendor_profile(row)
            self._vendor_cache[vendor_id] = profile
            return profile
        
        return None
    
    async def search_vendors(
        self,
        query: Optional[str] = None,
        industry: Optional[IndustryNiche] = None,
        risk_level: Optional[RiskLevel] = None,
        status: Optional[VendorStatus] = None,
        limit: int = 50
    ) -> List[VendorProfile]:
        """Search vendors with filters"""
        postgres = await get_postgres_manager()
        
        conditions = []
        params = []
        param_idx = 1
        
        if query:
            conditions.append(f"vendor_name ILIKE ${param_idx}")
            params.append(f"%{query}%")
            param_idx += 1
        
        if industry:
            conditions.append(f"${param_idx} = ANY(industries_served)")
            params.append(industry.value)
            param_idx += 1
        
        if risk_level:
            conditions.append(f"risk_level = ${param_idx}")
            params.append(risk_level.value)
            param_idx += 1
        
        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status.value)
            param_idx += 1
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query_sql = f"""
            SELECT * FROM cyrex_vendors
            WHERE {where_clause}
            ORDER BY current_risk_score DESC, last_activity DESC
            LIMIT ${param_idx}
        """
        params.append(limit)
        
        rows = await postgres.fetch(query_sql, *params)
        
        return [self._row_to_vendor_profile(row) for row in rows]
    
    async def get_cross_industry_vendors(self, vendor_id: str) -> Dict[str, Any]:
        """Get vendor's cross-industry intelligence"""
        vendor = await self.get_vendor_profile(vendor_id)
        if not vendor:
            return {}
        
        return {
            "vendor_id": vendor.vendor_id,
            "vendor_name": vendor.vendor_name,
            "industries_served": vendor.industries_served,
            "cross_industry_flags": vendor.cross_industry_flags,
            "flagged_by_industries": vendor.flagged_by_industries,
            "fraud_flags_by_industry": vendor.fraud_flags_by_industry,
            "network_effect": vendor.cross_industry_flags > 0,
            "risk_score": vendor.current_risk_score,
            "risk_level": vendor.risk_level,
        }
    
    async def update_pricing_benchmark(
        self,
        service_type: str,
        industry: IndustryNiche,
        min_price: float,
        avg_price: float,
        max_price: float,
        unit: str,
        location: Optional[str] = None
    ):
        """Update pricing benchmark"""
        postgres = await get_postgres_manager()
        benchmark_id = f"bench_{uuid.uuid4().hex[:12]}"
        
        await postgres.execute("""
            INSERT INTO cyrex_pricing_benchmarks (
                benchmark_id, service_type, industry, location,
                min_price, avg_price, max_price, unit, last_updated
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (service_type, industry, COALESCE(location, ''))
            DO UPDATE SET
                min_price = EXCLUDED.min_price,
                avg_price = EXCLUDED.avg_price,
                max_price = EXCLUDED.max_price,
                unit = EXCLUDED.unit,
                sample_size = cyrex_pricing_benchmarks.sample_size + 1,
                last_updated = NOW()
        """,
            benchmark_id,
            service_type,
            industry.value,
            location,
            float(min_price),
            float(avg_price),
            float(max_price),
            unit,
        )
    
    async def get_pricing_benchmark(
        self,
        service_type: str,
        industry: IndustryNiche,
        location: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get pricing benchmark"""
        postgres = await get_postgres_manager()
        
        row = await postgres.fetchrow("""
            SELECT * FROM cyrex_pricing_benchmarks
            WHERE service_type = $1 AND industry = $2
            AND (location = $3 OR location IS NULL)
            ORDER BY location NULLS LAST
            LIMIT 1
        """,
            service_type,
            industry.value,
            location,
        )
        
        if row:
            return {
                "service_type": row["service_type"],
                "industry": row["industry"],
                "location": row["location"],
                "min_price": float(row["min_price"]) if row["min_price"] else None,
                "avg_price": float(row["avg_price"]),
                "max_price": float(row["max_price"]) if row["max_price"] else None,
                "unit": row["unit"],
                "sample_size": row["sample_size"],
                "last_updated": row["last_updated"].isoformat() if row["last_updated"] else None,
            }
        
        return None
    
    async def get_analytics(
        self,
        industry: Optional[IndustryNiche] = None,
        date_range: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """Get comprehensive analytics"""
        postgres = await get_postgres_manager()
        
        # Build date filter
        date_filter = ""
        params = []
        if date_range:
            date_filter = "AND invoice_date BETWEEN $1 AND $2"
            params = [date_range[0], date_range[1]]
        
        industry_filter = ""
        if industry:
            industry_filter = f"AND industry = ${len(params) + 1}"
            params.append(industry.value)
        
        # Total invoices
        total_invoices = await postgres.fetchval(
            f"SELECT COUNT(*) FROM cyrex_invoices WHERE 1=1 {date_filter} {industry_filter}",
            *params
        )
        
        # Fraud detected
        fraud_count = await postgres.fetchval(
            f"SELECT COUNT(*) FROM cyrex_invoices WHERE fraud_detected = TRUE {date_filter} {industry_filter}",
            *params
        )
        
        # Total amount analyzed
        total_amount = await postgres.fetchval(
            f"SELECT COALESCE(SUM(total_amount), 0) FROM cyrex_invoices WHERE 1=1 {date_filter} {industry_filter}",
            *params
        )
        
        # High-risk vendors
        high_risk_vendors = await postgres.fetchval(
            "SELECT COUNT(*) FROM cyrex_vendors WHERE risk_level IN ('high', 'critical')"
        )
        
        # Cross-industry flags
        cross_industry_flags = await postgres.fetchval(
            "SELECT SUM(cross_industry_flags) FROM cyrex_vendors"
        )
        
        return {
            "total_invoices_analyzed": total_invoices or 0,
            "fraud_detected_count": fraud_count or 0,
            "fraud_detection_rate": (fraud_count / total_invoices * 100) if total_invoices > 0 else 0,
            "total_amount_analyzed": float(total_amount) if total_amount else 0,
            "high_risk_vendors": high_risk_vendors or 0,
            "cross_industry_flags": cross_industry_flags or 0,
            "network_effects_active": cross_industry_flags > 0 if cross_industry_flags else False,
            "industry": industry.value if industry else "all",
            "date_range": {
                "start": date_range[0].isoformat() if date_range else None,
                "end": date_range[1].isoformat() if date_range else None,
            },
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def _row_to_vendor_profile(self, row) -> VendorProfile:
        """Convert database row to VendorProfile"""
        return VendorProfile(
            vendor_id=row["vendor_id"],
            vendor_name=row["vendor_name"],
            industries_served=json.loads(row["industries_served"]) if isinstance(row["industries_served"], str) else row["industries_served"],
            total_invoices_analyzed=row["total_invoices_analyzed"],
            total_invoice_amount=float(row["total_invoice_amount"]),
            fraud_flags_count=row["fraud_flags_count"],
            fraud_flags_by_industry=json.loads(row["fraud_flags_by_industry"]) if isinstance(row["fraud_flags_by_industry"], str) else row["fraud_flags_by_industry"],
            fraud_types_detected=json.loads(row["fraud_types_detected"]) if isinstance(row["fraud_types_detected"], str) else row["fraud_types_detected"],
            average_invoice_amount=float(row["average_invoice_amount"]),
            average_price_deviation=float(row["average_price_deviation"]),
            pricing_deviation_history=json.loads(row["pricing_deviation_history"]) if isinstance(row["pricing_deviation_history"], str) else row["pricing_deviation_history"],
            current_risk_score=float(row["current_risk_score"]),
            risk_level=row["risk_level"],
            risk_history=json.loads(row["risk_history"]) if isinstance(row["risk_history"], str) else row["risk_history"],
            cross_industry_flags=row["cross_industry_flags"],
            flagged_by_industries=json.loads(row["flagged_by_industries"]) if isinstance(row["flagged_by_industries"], str) else row["flagged_by_industries"],
            first_seen=row["first_seen"],
            last_activity=row["last_activity"],
            status=row["status"],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
        )


# Global service instance
_vendor_intelligence_service: Optional[VendorIntelligenceService] = None


async def get_vendor_intelligence_service() -> VendorIntelligenceService:
    """Get or create vendor intelligence service singleton"""
    global _vendor_intelligence_service
    
    if _vendor_intelligence_service is None:
        _vendor_intelligence_service = VendorIntelligenceService()
        await _vendor_intelligence_service.initialize()
    
    return _vendor_intelligence_service


