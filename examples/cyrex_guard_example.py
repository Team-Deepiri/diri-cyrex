"""
Cyrex Guard Examples
Demonstrates how to use Cyrex Guard for vendor fraud detection
"""
import asyncio
import json
from datetime import datetime

from app.services.invoice_parser import get_universal_invoice_processor
from app.services.pricing_benchmark import get_pricing_benchmark_engine
from app.services.lora_loader import get_industry_lora_service
from app.services.fraud_detector import get_universal_fraud_detection_service
from app.core.types import IndustryNiche


async def example_process_invoice():
    """Example: Process an invoice"""
    print("\n=== Example: Process Invoice ===\n")
    
    processor = get_universal_invoice_processor()
    
    # Example invoice text
    invoice_content = """
    Invoice #: INV-2026-001
    Vendor: ABC Plumbing Services
    Date: January 15, 2026
    Total: $1,250.00
    
    Line Items:
    1. Emergency pipe repair at 123 Main St
       Quantity: 1
       Unit Price: $1,250.00
       Total: $1,250.00
    
    Payment Terms: Net 30
    """
    
    processed = await processor.process_invoice(
        invoice_content=invoice_content,
        industry=IndustryNiche.PROPERTY_MANAGEMENT,
        invoice_format="text",
        use_lora=True
    )
    
    print(f"Invoice ID: {processed.invoice_id}")
    print(f"Vendor: {processed.vendor_name}")
    print(f"Total Amount: ${processed.total_amount:,.2f}")
    print(f"Service Category: {processed.service_category}")
    print(f"Extraction Confidence: {processed.extraction_confidence:.2%}")
    print(f"Line Items: {len(processed.line_items)}")
    
    for i, item in enumerate(processed.line_items, 1):
        print(f"  {i}. {item.description}: ${item.total or 0:,.2f}")


async def example_compare_price():
    """Example: Compare price against benchmark"""
    print("\n=== Example: Compare Price ===\n")
    
    engine = get_pricing_benchmark_engine()
    await engine.initialize()
    
    comparison = await engine.compare_price(
        invoice_price=1500.00,
        service_category="hvac_repair",
        industry=IndustryNiche.PROPERTY_MANAGEMENT,
        location="New York, NY"
    )
    
    print(f"Invoice Price: ${comparison.invoice_price:,.2f}")
    print(f"Market Median: ${comparison.benchmark.median_price:,.2f}")
    print(f"Deviation: {comparison.deviation_percent:+.1f}%")
    print(f"Tier: {comparison.tier.value}")
    print(f"Is Overpriced: {comparison.is_overpriced}")
    print(f"Confidence: {comparison.confidence:.2%}")
    print("\nRecommendations:")
    for rec in comparison.recommendations:
        print(f"  - {rec}")


async def example_get_benchmark():
    """Example: Get pricing benchmark"""
    print("\n=== Example: Get Pricing Benchmark ===\n")
    
    engine = get_pricing_benchmark_engine()
    await engine.initialize()
    
    benchmark = await engine.get_benchmark(
        service_category="plumbing_repair",
        industry=IndustryNiche.PROPERTY_MANAGEMENT,
        location="Los Angeles, CA"
    )
    
    if benchmark:
        print(f"Service Category: {benchmark.service_category}")
        print(f"Industry: {benchmark.industry.value}")
        print(f"Location: {benchmark.location or 'Global'}")
        print(f"\nPricing Statistics:")
        print(f"  Median: ${benchmark.median_price:,.2f}")
        print(f"  Mean: ${benchmark.mean_price:,.2f}")
        print(f"  Min: ${benchmark.min_price:,.2f}")
        print(f"  Max: ${benchmark.max_price:,.2f}")
        print(f"  25th Percentile: ${benchmark.percentile_25:,.2f}")
        print(f"  75th Percentile: ${benchmark.percentile_75:,.2f}")
        print(f"  90th Percentile: ${benchmark.percentile_90:,.2f}")
        print(f"\nSample Count: {benchmark.sample_count}")
    else:
        print("No benchmark data available (insufficient samples)")


async def example_lora_adapters():
    """Example: Manage LoRA adapters"""
    print("\n=== Example: LoRA Adapter Management ===\n")
    
    service = get_industry_lora_service()
    
    # List available adapters
    adapters = await service.list_available_adapters()
    print(f"Available Adapters: {len(adapters)}")
    for adapter in adapters:
        print(f"  - {adapter.industry.value}: {adapter.adapter_name} (v{adapter.version})")
    
    # Check status
    status = await service.get_adapter_status(IndustryNiche.PROPERTY_MANAGEMENT)
    if status:
        print(f"\nProperty Management Adapter Status: {status.status.value}")
        if status.loaded_at:
            print(f"Loaded At: {status.loaded_at}")
    else:
        print("\nProperty Management Adapter: Not configured")


async def example_detect_fraud():
    """Example: Detect fraud in invoice"""
    print("\n=== Example: Fraud Detection ===\n")
    
    processor = get_universal_invoice_processor()
    fraud_service = get_universal_fraud_detection_service()
    
    # Process invoice first
    invoice_content = """
    Invoice #: INV-2026-004
    Vendor: Suspicious Vendor Inc
    Date: January 15, 2026
    Total: $5,000.00
    
    Line Items:
    1. HVAC repair - $5,000.00
    """
    
    processed = await processor.process_invoice(
        invoice_content=invoice_content,
        industry=IndustryNiche.PROPERTY_MANAGEMENT,
        invoice_format="text"
    )
    
    print(f"Processed Invoice: {processed.invoice_id}")
    print(f"Vendor: {processed.vendor_name}")
    print(f"Amount: ${processed.total_amount:,.2f}\n")
    
    # Detect fraud
    result = await fraud_service.detect_fraud(
        invoice=processed,
        use_vendor_intelligence=True,
        use_pricing_benchmark=True,
        use_anomaly_detection=True,
        use_pattern_matching=True
    )
    
    print(f"Fraud Detected: {result.fraud_detected}")
    print(f"Risk Score: {result.risk_score:.1f}/100")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nFraud Indicators: {len(result.fraud_indicators)}")
    
    for i, indicator in enumerate(result.fraud_indicators, 1):
        print(f"\n  {i}. {indicator.fraud_type.value}")
        print(f"     Severity: {indicator.severity:.2%}")
        print(f"     Confidence: {indicator.confidence:.2%}")
        print(f"     Description: {indicator.description}")
        print(f"     Recommendation: {indicator.recommendation}")
    
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
    
    if result.price_comparison:
        print(f"\nPrice Comparison:")
        print(f"  Deviation: {result.price_comparison.deviation_percent:+.1f}%")
        print(f"  Tier: {result.price_comparison.tier.value}")


async def example_full_pipeline():
    """Example: Full pipeline from invoice to fraud detection"""
    print("\n=== Example: Full Pipeline ===\n")
    
    processor = get_universal_invoice_processor()
    fraud_service = get_universal_fraud_detection_service()
    pricing_engine = get_pricing_benchmark_engine()
    await pricing_engine.initialize()
    
    # Step 1: Process invoice
    invoice_content = """
    Invoice #: INV-2026-005
    Vendor: Quality HVAC Services
    Date: January 15, 2026
    Total: $1,800.00
    
    Line Items:
    1. HVAC system maintenance - $1,800.00
    
    Service Category: hvac_maintenance
    Property: 456 Oak Avenue, New York, NY
    """
    
    print("Step 1: Processing invoice...")
    processed = await processor.process_invoice(
        invoice_content=invoice_content,
        industry=IndustryNiche.PROPERTY_MANAGEMENT,
        invoice_format="text",
        use_lora=True
    )
    print(f"✓ Invoice processed: {processed.invoice_id}")
    
    # Step 2: Compare price
    print("\nStep 2: Comparing price against benchmark...")
    if processed.service_category:
        comparison = await pricing_engine.compare_price(
            invoice_price=processed.total_amount,
            service_category=processed.service_category,
            industry=processed.industry,
            location=processed.property_address
        )
        print(f"✓ Price comparison: {comparison.deviation_percent:+.1f}% deviation")
        print(f"  Tier: {comparison.tier.value}, Overpriced: {comparison.is_overpriced}")
    
    # Step 3: Detect fraud
    print("\nStep 3: Detecting fraud...")
    result = await fraud_service.detect_fraud(
        invoice=processed,
        use_vendor_intelligence=True,
        use_pricing_benchmark=True,
        use_anomaly_detection=True,
        use_pattern_matching=True
    )
    print(f"✓ Fraud detection complete")
    print(f"  Risk Score: {result.risk_score:.1f}/100")
    print(f"  Risk Level: {result.risk_level.value}")
    print(f"  Fraud Detected: {result.fraud_detected}")
    print(f"  Indicators: {len(result.fraud_indicators)}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Invoice: {processed.invoice_id}")
    print(f"Vendor: {processed.vendor_name}")
    print(f"Amount: ${processed.total_amount:,.2f}")
    print(f"Risk: {result.risk_level.value} ({result.risk_score:.1f}/100)")
    if result.fraud_detected:
        print("⚠️  FRAUD INDICATORS DETECTED - Review Required")
    else:
        print("✓ No significant fraud indicators")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("Cyrex Guard Examples")
    print("=" * 60)
    
    await example_process_invoice()
    await example_compare_price()
    await example_get_benchmark()
    await example_lora_adapters()
    await example_detect_fraud()
    await example_full_pipeline()
    
    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

