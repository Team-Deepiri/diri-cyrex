"""
Universal RAG Example
Demonstrates indexing and querying documents across industries
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.integrations.universal_rag_engine import create_universal_rag_engine
from deepiri_modelkit.rag import (
    Document,
    DocumentType,
    IndustryNiche,
    RAGQuery,
)
from datetime import datetime


def example_insurance():
    """Example: Insurance claims and policies"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Insurance Claims Processing")
    print("="*80)
    
    # Create insurance RAG engine
    engine = create_universal_rag_engine(
        industry=IndustryNiche.INSURANCE,
        collection_name="example_insurance_rag"
    )
    
    # Index homeowners policy
    policy = Document(
        id="policy_homeowners_2024",
        content="""
        Homeowners Insurance Policy - Coverage Details
        
        Water Damage: Covered up to $50,000 for sudden and accidental water damage
        including burst pipes, appliance malfunctions, and roof leaks.
        
        Exclusions: Flood damage is NOT covered under standard homeowners insurance.
        Requires separate flood insurance policy.
        
        Deductible: $1,000 per claim
        """,
        doc_type=DocumentType.POLICY,
        industry=IndustryNiche.INSURANCE,
        title="Standard Homeowners Policy",
        source="policy_documents/homeowners_2024.pdf",
        metadata={
            "policy_type": "homeowners",
            "coverage_limit": 50000,
        }
    )
    
    print("\n✓ Indexing homeowners policy...")
    success = engine.index_document(policy)
    print(f"  Status: {'Success' if success else 'Failed'}")
    
    # Query
    print("\n📋 Query: Is water damage from a burst pipe covered?")
    query = RAGQuery(
        query="Is water damage from a burst pipe covered?",
        industry=IndustryNiche.INSURANCE,
        doc_types=[DocumentType.POLICY],
        top_k=3
    )
    
    results = engine.retrieve(query)
    print(f"\n  Found {len(results)} relevant documents:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Document: {result.document.title}")
        print(f"    Score: {result.score:.3f}")
        print(f"    Answer: {result.document.content.strip()[:200]}...")


def example_manufacturing():
    """Example: Manufacturing equipment maintenance"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Manufacturing Equipment Maintenance")
    print("="*80)
    
    # Create manufacturing RAG engine
    engine = create_universal_rag_engine(
        industry=IndustryNiche.MANUFACTURING,
        collection_name="example_manufacturing_rag"
    )
    
    # Index equipment manual
    manual = Document(
        id="manual_compressor_xyz_2024",
        content="""
        Compressor Model XYZ-500 Maintenance Manual
        
        Belt Replacement Procedure:
        1. Power off compressor and disconnect from power source
        2. Remove safety guard (4 bolts)
        3. Loosen tensioner pulley
        4. Remove old belt
        5. Install new belt (Part #: BLT-XYZ-500)
        6. Adjust tension to 50-60 lbs
        7. Replace safety guard
        8. Test operation
        
        Belt Replacement Frequency: Every 2000 operating hours or 12 months
        """,
        doc_type=DocumentType.MANUAL,
        industry=IndustryNiche.MANUFACTURING,
        title="Compressor XYZ-500 Maintenance Manual",
        source="manufacturer/acmecorp",
        version="2024.1",
        metadata={
            "equipment_model": "XYZ-500",
            "manufacturer": "AcmeCorp",
        }
    )
    
    print("\n✓ Indexing equipment manual...")
    success = engine.index_document(manual)
    print(f"  Status: {'Success' if success else 'Failed'}")
    
    # Index maintenance log
    log = Document(
        id="maint_log_xyz500_2024_01",
        content="""
        Date: 2024-01-15
        Equipment: Compressor XYZ-500 (Serial: 123456)
        Maintenance Type: Preventive
        Technician: John Doe
        
        Work Performed:
        - Replaced compressor belt (Part #: BLT-XYZ-500)
        - Adjusted belt tension to 55 lbs
        - Tested operation - Normal
        
        Next Service: 2024-07-15 or 2000 hours
        """,
        doc_type=DocumentType.MAINTENANCE_LOG,
        industry=IndustryNiche.MANUFACTURING,
        title="XYZ-500 Belt Replacement - Jan 2024",
        source="maintenance_system",
        created_at=datetime(2024, 1, 15),
        metadata={
            "equipment_model": "XYZ-500",
            "serial_number": "123456",
            "maintenance_type": "preventive",
        }
    )
    
    print("\n✓ Indexing maintenance log...")
    success = engine.index_document(log)
    print(f"  Status: {'Success' if success else 'Failed'}")
    
    # Query 1: How-to
    print("\n📋 Query 1: How do I replace the belt on compressor XYZ-500?")
    query1 = RAGQuery(
        query="How do I replace the belt on compressor XYZ-500?",
        industry=IndustryNiche.MANUFACTURING,
        doc_types=[DocumentType.MANUAL],
        top_k=3
    )
    
    results1 = engine.retrieve(query1)
    print(f"\n  Found {len(results1)} relevant documents:")
    for i, result in enumerate(results1, 1):
        print(f"\n  Result {i}:")
        print(f"    Document: {result.document.title}")
        print(f"    Score: {result.score:.3f}")
        print(f"    Content: {result.document.content.strip()[:200]}...")
    
    # Query 2: Historical
    print("\n\n📋 Query 2: When was the last belt replacement on XYZ-500?")
    query2 = RAGQuery(
        query="When was the last belt replacement on XYZ-500?",
        industry=IndustryNiche.MANUFACTURING,
        doc_types=[DocumentType.MAINTENANCE_LOG],
        metadata_filters={"equipment_model": "XYZ-500"},
        top_k=5
    )
    
    results2 = engine.retrieve(query2)
    print(f"\n  Found {len(results2)} relevant documents:")
    for i, result in enumerate(results2, 1):
        print(f"\n  Result {i}:")
        print(f"    Document: {result.document.title}")
        print(f"    Date: {result.document.created_at}")
        print(f"    Content: {result.document.content.strip()[:200]}...")


def example_generate_with_context():
    """Example: Generate answer with RAG context"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Generate Answer with RAG Context")
    print("="*80)
    
    engine = create_universal_rag_engine(
        industry=IndustryNiche.MANUFACTURING
    )
    
    # Query and generate
    print("\n📋 Query: What maintenance is required for compressor XYZ-500?")
    query = RAGQuery(
        query="What maintenance is required for compressor XYZ-500?",
        industry=IndustryNiche.MANUFACTURING,
        doc_types=[DocumentType.MANUAL, DocumentType.MAINTENANCE_LOG],
        top_k=3
    )
    
    retrieved = engine.retrieve(query)
    
    print(f"\n  Retrieved {len(retrieved)} documents")
    
    generation = engine.generate_with_context(
        query="What maintenance is required for compressor XYZ-500?",
        retrieved_docs=retrieved
    )
    
    print("\n  Generated Prompt:")
    print("  " + "-"*76)
    print(f"{generation['prompt'][:500]}...")
    print("  " + "-"*76)


def example_statistics():
    """Example: Get RAG statistics"""
    print("\n" + "="*80)
    print("EXAMPLE 4: RAG Statistics")
    print("="*80)
    
    # Get stats for each industry
    for industry in [IndustryNiche.INSURANCE, IndustryNiche.MANUFACTURING]:
        engine = create_universal_rag_engine(industry=industry)
        stats = engine.get_statistics()
        
        print(f"\n{industry.value.upper()} Statistics:")
        print(f"  Collection: {stats.get('collection_name')}")
        print(f"  Documents: {stats.get('num_entities', 0)}")
        print(f"  Mode: {stats.get('mode', 'unknown')}")
        print(f"  Healthy: {stats.get('healthy', False)}")


def main():
    """Run all examples"""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  Universal RAG System - Demonstration".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    try:
        # Run examples
        example_insurance()
        example_manufacturing()
        example_generate_with_context()
        example_statistics()
        
        print("\n" + "="*80)
        print("✓ All examples completed successfully!")
        print("="*80)
        print("\nNext Steps:")
        print("  1. Try the REST API: http://localhost:8000/docs")
        print("  2. Read the guide: docs/UNIVERSAL_RAG_GUIDE.md")
        print("  3. Explore examples: docs/UNIVERSAL_RAG_EXAMPLES.md")
        print()
    
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

