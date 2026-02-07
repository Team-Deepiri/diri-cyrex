# Milvus Indexing Pipeline for LangChain/LangGraph Financial Analysis Agents

**Purpose**: Complete guide for indexing financial, legal, and regulatory documents into Milvus to support LangChain/LangGraph agents for the Language Intelligence Platform.

**Platform Features Supported**:
- Regulatory Language Evolution Tracker
- Contract Clause Evolution Tracker
- Corporate Lease Abstraction
- Obligation Dependency Graph
- Compliance Pattern Mining
- Version Drift Detection
- Obligation Tracker

**Architecture**: Document Ingestion → Processing → Embedding → Milvus Indexing → LangChain/LangGraph Agent RAG Retrieval

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Document Types for Language Intelligence Platform](#document-types-for-language-intelligence-platform)
4. [Indexing Pipeline Implementation](#indexing-pipeline-implementation)
5. [LangChain/LangGraph Agent Integration](#langchainlanggraph-agent-integration)
6. [Use Case Examples](#use-case-examples)
7. [Metadata Schema Requirements](#metadata-schema-requirements)
8. [Chunking Strategies](#chunking-strategies)
9. [API Endpoints](#api-endpoints)
10. [Best Practices](#best-practices)

---

## OVERVIEW

### What This Pipeline Does

The Milvus indexing pipeline transforms financial, legal, and regulatory documents into searchable vector embeddings for LangChain/LangGraph agents performing:

- **Regulatory Language Analysis**: Track how regulatory language evolves over time
- **Contract Clause Tracking**: Monitor clause changes across contract versions
- **Lease Abstraction**: Extract and structure lease terms for analysis
- **Obligation Mapping**: Map cascading obligations across contracts and leases
- **Compliance Pattern Mining**: Identify patterns in compliance failures
- **Version Drift Detection**: Detect divergence from expected language standards
- **Obligation Tracking**: Track obligations with deadlines and owners

### Pipeline Flow

```
Financial/Legal Documents → Document Ingestion → Text Extraction → 
Intelligent Chunking → Embedding Generation → Milvus Indexing → 
LangChain/LangGraph Agent RAG Retrieval → Analysis & Insights
```

### Key Capabilities

✅ **Multi-Document Type Support**: Leases, contracts, regulatory documents, amendments, compliance reports  
✅ **Version Tracking**: Index multiple versions of same document for drift detection  
✅ **Obligation Extraction**: Metadata extraction for obligation tracking  
✅ **Clause-Level Chunking**: Intelligent chunking that preserves clause boundaries  
✅ **Regulatory Linkage**: Metadata linking contracts/leases to regulations  
✅ **LangChain Integration**: Direct integration with LangChain vector stores  
✅ **LangGraph Workflow Support**: Ready for multi-agent LangGraph workflows

---

## ARCHITECTURE

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Document Ingestion Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Leases     │  │  Contracts   │  │ Regulatory  │        │
│  │   (PDF/DOCX) │  │  (PDF/DOCX)  │  │  Documents  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            Document Processing Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Parser      │  │  Clause      │  │  Obligation  │        │
│  │  (PDF/DOCX)   │  │  Extractor   │  │  Extractor  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Intelligent Chunking Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Clause-Aware│  │  Version     │  │  Context     │        │
│  │  Chunking    │  │  Tracking    │  │  Preservation│        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Embedding Generation Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Embedding   │  │   Batch      │  │  Legal/Fin   │        │
│  │    Model     │  │  Processing  │  │   Context   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Milvus Indexing Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Collection  │  │   Index      │  │   Metadata   │        │
│  │  Management  │  │  Management  │  │   Storage    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          LangChain/LangGraph Agent Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  LangChain   │  │  LangGraph   │  │   RAG        │        │
│  │  Retrievers  │  │  Workflows   │  │   Chains     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**1. Document Ingestion Layer**
- Accepts leases, contracts, regulatory documents, amendments
- Validates file formats and versions
- Routes to appropriate processors

**2. Document Processing Layer**
- Parses legal/financial documents
- Extracts clauses, obligations, terms
- Identifies document relationships (contract → lease, regulation → contract)
- Extracts version information

**3. Intelligent Chunking Layer**
- Clause-aware chunking (preserves clause boundaries)
- Version-aware chunking (tracks changes across versions)
- Context preservation (maintains document structure)

**4. Embedding Generation Layer**
- Generates vector embeddings optimized for legal/financial language
- Batch processing for efficiency
- Preserves semantic meaning of legal terms

**5. Milvus Indexing Layer**
- Creates/manages collections for different document types
- Indexes vectors with rich metadata
- Supports version tracking and relationship mapping

**6. LangChain/LangGraph Agent Layer**
- LangChain retrievers for semantic search
- LangGraph workflows for multi-step analysis
- RAG chains for document-augmented generation

---

## DOCUMENT TYPES FOR LANGUAGE INTELLIGENCE PLATFORM

### 1. Leases (`LEASE`)

**Purpose**: Corporate lease abstraction and analysis

**Metadata Required**:
```python
{
    "document_type": "lease",
    "lease_id": "LEASE-001",
    "tenant_name": "Acme Corp",
    "landlord_name": "Property LLC",
    "property_address": "123 Main St",
    "lease_start_date": "2024-01-01",
    "lease_end_date": "2029-12-31",
    "version": "1.0",
    "version_date": "2024-01-01",
    "previous_version_id": null,
    "regulatory_references": ["REG-2024-001"],  # Links to regulations
    "contract_references": ["CONTRACT-001"],    # Links to contracts
    "obligations": [
        {
            "obligation_id": "OBL-001",
            "type": "rent_payment",
            "deadline": "monthly",
            "owner": "tenant",
            "amount": 5000.00
        }
    ],
    "clauses": [
        {
            "clause_id": "CLAUSE-001",
            "type": "rent",
            "section": "Section 3.1",
            "text": "Tenant shall pay monthly rent..."
        }
    ]
}
```

### 2. Contracts (`CONTRACT`)

**Purpose**: Contract clause evolution tracking and obligation mapping

**Metadata Required**:
```python
{
    "document_type": "contract",
    "contract_id": "CONTRACT-001",
    "contract_name": "Service Agreement",
    "party_a": "Acme Corp",
    "party_b": "Vendor Inc",
    "effective_date": "2024-01-01",
    "expiration_date": "2026-12-31",
    "version": "2.1",
    "version_date": "2024-06-01",
    "previous_version_id": "CONTRACT-001-v2.0",
    "regulatory_references": ["REG-2024-002"],
    "lease_references": ["LEASE-001"],
    "obligations": [
        {
            "obligation_id": "OBL-002",
            "type": "payment",
            "deadline": "net_30",
            "owner": "party_b",
            "amount": 10000.00
        }
    ],
    "clauses": [
        {
            "clause_id": "CLAUSE-002",
            "type": "payment_terms",
            "section": "Section 4.2",
            "text": "Payment shall be due within 30 days...",
            "evolution_history": [
                {"version": "2.0", "text": "Payment shall be due within 45 days..."},
                {"version": "2.1", "text": "Payment shall be due within 30 days..."}
            ]
        }
    ]
}
```

### 3. Regulatory Documents (`REGULATION`)

**Purpose**: Regulatory language evolution tracking

**Metadata Required**:
```python
{
    "document_type": "regulation",
    "regulation_id": "REG-2024-001",
    "regulation_name": "Commercial Lease Standards Act",
    "jurisdiction": "State of California",
    "effective_date": "2024-01-01",
    "version": "2024.1",
    "version_date": "2024-01-01",
    "previous_version_id": "REG-2023-001",
    "impacted_contracts": ["CONTRACT-001", "CONTRACT-002"],
    "impacted_leases": ["LEASE-001"],
    "language_changes": [
        {
            "section": "Section 5.2",
            "old_text": "Landlords must provide...",
            "new_text": "Landlords shall provide...",
            "change_type": "mandatory_language",
            "impact": "high"
        }
    ]
}
```

### 4. Amendments (`AMENDMENT`)

**Purpose**: Track document modifications

**Metadata Required**:
```python
{
    "document_type": "amendment",
    "amendment_id": "AMEND-001",
    "amends_document_id": "LEASE-001",
    "amends_document_type": "lease",
    "amendment_date": "2024-06-01",
    "changes": [
        {
            "clause_id": "CLAUSE-001",
            "change_type": "modified",
            "old_text": "...",
            "new_text": "..."
        }
    ]
}
```

### 5. Compliance Reports (`COMPLIANCE_REPORT`)

**Purpose**: Compliance pattern mining

**Metadata Required**:
```python
{
    "document_type": "compliance_report",
    "report_id": "COMP-001",
    "report_date": "2024-06-01",
    "report_type": "quarterly",
    "compliance_status": "non_compliant",
    "violations": [
        {
            "violation_id": "VIOL-001",
            "regulation_id": "REG-2024-001",
            "contract_id": "CONTRACT-001",
            "violation_type": "deadline_missed",
            "pattern": "recurring"
        }
    ],
    "patterns": [
        {
            "pattern_id": "PATTERN-001",
            "pattern_type": "recurring_violation",
            "frequency": "monthly",
            "affected_documents": ["CONTRACT-001", "CONTRACT-002"]
        }
    ]
}
```

---

## INDEXING PIPELINE IMPLEMENTATION

### Using DocumentIndexingService

**Location**: `app/services/document_indexing_service.py`

**Initialization**:
```python
from app.services.document_indexing_service import (
    get_document_indexing_service,
    B2BDocumentType
)

# Initialize for Language Intelligence Platform
service = await get_document_indexing_service(
    collection_name="language_intelligence_documents",
    chunk_size=1500,  # Larger chunks for legal documents
    chunk_overlap=300,  # More overlap for clause preservation
    chunking_strategy="paragraph",  # Paragraph-aware for clauses
    use_enhanced_rag=True
)
```

### Indexing a Lease Document

```python
# Index lease for abstraction and obligation tracking
lease_doc = await service.index_file(
    file_path="lease_001.pdf",
    document_id="LEASE-001",
    title="Corporate Lease Agreement - 123 Main St",
    doc_type=B2BDocumentType.LEGAL_DOCUMENT,  # Or create LEASE type
    industry="legal",
    metadata={
        "document_type": "lease",
        "lease_id": "LEASE-001",
        "tenant_name": "Acme Corp",
        "landlord_name": "Property LLC",
        "property_address": "123 Main St",
        "lease_start_date": "2024-01-01",
        "lease_end_date": "2029-12-31",
        "version": "1.0",
        "version_date": "2024-01-01",
        "regulatory_references": ["REG-2024-001"],
        "contract_references": ["CONTRACT-001"],
        # Obligations will be extracted by agents
        # Clauses will be identified during chunking
    }
)
```

### Indexing a Contract with Version Tracking

```python
# Index contract version 2.1
contract_v21 = await service.index_file(
    file_path="contract_001_v2.1.pdf",
    document_id="CONTRACT-001-v2.1",
    title="Service Agreement v2.1",
    doc_type=B2BDocumentType.CONTRACT,
    industry="legal",
    metadata={
        "document_type": "contract",
        "contract_id": "CONTRACT-001",
        "version": "2.1",
        "version_date": "2024-06-01",
        "previous_version_id": "CONTRACT-001-v2.0",
        "party_a": "Acme Corp",
        "party_b": "Vendor Inc",
        "effective_date": "2024-01-01",
        "expiration_date": "2026-12-31",
        "regulatory_references": ["REG-2024-002"],
    }
)

# Index previous version for drift detection
contract_v20 = await service.index_file(
    file_path="contract_001_v2.0.pdf",
    document_id="CONTRACT-001-v2.0",
    title="Service Agreement v2.0",
    doc_type=B2BDocumentType.CONTRACT,
    industry="legal",
    metadata={
        "document_type": "contract",
        "contract_id": "CONTRACT-001",
        "version": "2.0",
        "version_date": "2024-01-01",
        "previous_version_id": "CONTRACT-001-v1.0",
    }
)
```

### Indexing Regulatory Documents

```python
# Index regulation for language evolution tracking
regulation = await service.index_file(
    file_path="regulation_2024_001.pdf",
    document_id="REG-2024-001",
    title="Commercial Lease Standards Act 2024",
    doc_type=B2BDocumentType.REGULATION,
    industry="legal",
    metadata={
        "document_type": "regulation",
        "regulation_id": "REG-2024-001",
        "regulation_name": "Commercial Lease Standards Act",
        "jurisdiction": "State of California",
        "effective_date": "2024-01-01",
        "version": "2024.1",
        "version_date": "2024-01-01",
        "previous_version_id": "REG-2023-001",
    }
)
```

---

## LANGCHAIN/LANGGRAPH AGENT INTEGRATION

### LangChain Retriever Setup

**Using MilvusVectorStore with LangChain**:

```python
from app.integrations.milvus_store import get_milvus_store
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Get Milvus vector store
vector_store = get_milvus_store(
    collection_name="language_intelligence_documents",
    host="localhost",
    port=19530
)

# Create LangChain retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": {
            "document_type": "lease"  # Filter by document type
        }
    }
)

# Use in LangChain chain
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

llm = Ollama(model="llama3:8b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query with context
result = qa_chain.invoke({
    "query": "What are the rent payment obligations in lease LEASE-001?"
})
```

### LangGraph Workflow Integration

**Using in LangGraph Multi-Agent Workflow**:

```python
from app.core.langgraph_workflow import LangGraphMultiAgentWorkflow
from app.integrations.milvus_store import get_milvus_store

# Initialize workflow with Milvus vector store
vector_store = get_milvus_store(
    collection_name="language_intelligence_documents"
)

workflow = LangGraphMultiAgentWorkflow(
    llm_provider=llm_provider,
    vector_store=vector_store,  # Pass vector store
    tool_registry=tool_registry,
    rag_pipeline=rag_pipeline,  # RAG pipeline for document retrieval
)

# Execute workflow that uses document retrieval
result = await workflow.execute(
    task="Analyze lease LEASE-001 and identify all obligations",
    user_id="user123"
)
```

### Custom LangChain Tool for Document Search

**Create tool for agents to search documents**:

```python
from langchain.tools import Tool
from app.services.document_indexing_service import get_document_indexing_service

async def search_documents_tool(query: str, doc_type: str = None) -> str:
    """Tool for agents to search indexed documents"""
    service = await get_document_indexing_service(
        collection_name="language_intelligence_documents"
    )
    
    doc_types = None
    if doc_type:
        from app.services.document_indexing_service import B2BDocumentType
        doc_types = [B2BDocumentType(doc_type)]
    
    results = await service.search(
        query=query,
        doc_types=doc_types,
        top_k=5
    )
    
    if not results:
        return "No relevant documents found."
    
    formatted = []
    for result in results:
        formatted.append(
            f"Document: {result['title']}\n"
            f"Type: {result['doc_type']}\n"
            f"Content: {result['content']}\n"
            f"Source: {result['source']}\n"
        )
    
    return "\n\n".join(formatted)

# Register as LangChain tool
document_search_tool = Tool(
    name="search_documents",
    description="Search indexed legal, financial, and regulatory documents. Use this to find leases, contracts, regulations, and compliance reports.",
    func=lambda q: asyncio.run(search_documents_tool(q))
)
```

### RAG Chain for Document Analysis

**Create RAG chain for specific use cases**:

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Setup memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create RAG chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)

# Use for lease abstraction
response = rag_chain.invoke({
    "question": "Extract all financial terms from lease LEASE-001 including rent amount, payment schedule, and security deposit."
})
```

---

## USE CASE EXAMPLES

### Use Case 1: Lease Abstraction Agent

**Goal**: Extract structured data from lease documents

```python
# 1. Index lease
await service.index_file(
    file_path="lease.pdf",
    document_id="LEASE-001",
    doc_type=B2BDocumentType.LEGAL_DOCUMENT,
    metadata={"document_type": "lease", "lease_id": "LEASE-001"}
)

# 2. Create LangChain agent with document retrieval
from langchain.agents import initialize_agent, AgentType

tools = [document_search_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 3. Agent abstracts lease
result = agent.run(
    "Find lease LEASE-001 and extract: "
    "1. Tenant name and landlord name "
    "2. Property address "
    "3. Lease term (start and end dates) "
    "4. Monthly rent amount "
    "5. Security deposit "
    "6. All payment obligations"
)
```

### Use Case 2: Contract Clause Evolution Tracker

**Goal**: Track how clauses change across contract versions

```python
# 1. Index multiple versions
versions = ["v1.0", "v2.0", "v2.1"]
for version in versions:
    await service.index_file(
        file_path=f"contract_{version}.pdf",
        document_id=f"CONTRACT-001-{version}",
        metadata={
            "contract_id": "CONTRACT-001",
            "version": version,
            "previous_version": versions[versions.index(version) - 1] if versions.index(version) > 0 else None
        }
    )

# 2. LangGraph workflow for clause comparison
workflow = LangGraphMultiAgentWorkflow(vector_store=vector_store)

result = await workflow.execute(
    task="Compare contract CONTRACT-001 versions 2.0 and 2.1. "
         "Identify all clause changes and explain what changed and why."
)
```

### Use Case 3: Regulatory Language Evolution Tracker

**Goal**: Track how regulatory language changes over time

```python
# 1. Index regulation versions
await service.index_file(
    file_path="regulation_2023.pdf",
    document_id="REG-2023-001",
    metadata={
        "regulation_id": "REG-2024-001",
        "version": "2023.1",
        "effective_date": "2023-01-01"
    }
)

await service.index_file(
    file_path="regulation_2024.pdf",
    document_id="REG-2024-001",
    metadata={
        "regulation_id": "REG-2024-001",
        "version": "2024.1",
        "effective_date": "2024-01-01",
        "previous_version_id": "REG-2023-001"
    }
)

# 2. Agent analyzes language evolution
agent.run(
    "Compare regulation REG-2024-001 versions 2023.1 and 2024.1. "
    "Identify all language changes, categorize them by type "
    "(mandatory language, definitions, requirements), and assess impact."
)
```

### Use Case 4: Obligation Dependency Graph

**Goal**: Map cascading obligations across documents

```python
# 1. Index related documents with obligation metadata
await service.index_file(
    file_path="regulation.pdf",
    document_id="REG-001",
    metadata={
        "document_type": "regulation",
        "obligations": [{"id": "OBL-REG-001", "type": "compliance_reporting"}]
    }
)

await service.index_file(
    file_path="contract.pdf",
    document_id="CONTRACT-001",
    metadata={
        "document_type": "contract",
        "regulatory_references": ["REG-001"],
        "obligations": [
            {"id": "OBL-CON-001", "type": "payment", "depends_on": "OBL-REG-001"}
        ]
    }
)

await service.index_file(
    file_path="lease.pdf",
    document_id="LEASE-001",
    metadata={
        "document_type": "lease",
        "contract_references": ["CONTRACT-001"],
        "obligations": [
            {"id": "OBL-LEASE-001", "type": "rent", "depends_on": "OBL-CON-001"}
        ]
    }
)

# 2. LangGraph workflow builds dependency graph
workflow.execute(
    task="Build obligation dependency graph for regulation REG-001. "
         "Show how obligations cascade from regulation → contract → lease."
)
```

### Use Case 5: Compliance Pattern Mining

**Goal**: Identify patterns in compliance failures

```python
# 1. Index compliance reports
for report in compliance_reports:
    await service.index_file(
        file_path=report["file"],
        document_id=report["id"],
        metadata={
            "document_type": "compliance_report",
            "compliance_status": report["status"],
            "violations": report["violations"],
            "patterns": report.get("patterns", [])
        }
    )

# 2. Agent mines patterns
agent.run(
    "Analyze all compliance reports from Q1 2024. "
    "Identify recurring violation patterns. "
    "Which regulations are most frequently violated? "
    "Which contracts/leases have the most violations? "
    "What patterns indicate high compliance risk?"
)
```

### Use Case 6: Version Drift Detection

**Goal**: Detect divergence from expected language standards

```python
# 1. Index standard template and actual contract
await service.index_file(
    file_path="template.pdf",
    document_id="TEMPLATE-001",
    metadata={
        "document_type": "template",
        "is_template": True
    }
)

await service.index_file(
    file_path="contract_actual.pdf",
    document_id="CONTRACT-001",
    metadata={
        "document_type": "contract",
        "template_id": "TEMPLATE-001"
    }
)

# 2. Agent detects drift
agent.run(
    "Compare contract CONTRACT-001 against template TEMPLATE-001. "
    "Identify all language drift - sections that differ from template. "
    "Categorize drifts by type (added clauses, modified language, removed sections). "
    "Flag any drifts that may create compliance risk."
)
```

### Use Case 7: Obligation Tracker

**Goal**: Track obligations with deadlines and owners

```python
# 1. Index documents with obligation metadata
await service.index_file(
    file_path="lease.pdf",
    document_id="LEASE-001",
    metadata={
        "document_type": "lease",
        "obligations": [
            {
                "obligation_id": "OBL-001",
                "type": "rent_payment",
                "deadline": "2024-07-01",
                "owner": "tenant",
                "amount": 5000.00,
                "frequency": "monthly"
            },
            {
                "obligation_id": "OBL-002",
                "type": "maintenance",
                "deadline": "2024-06-15",
                "owner": "landlord"
            }
        ]
    }
)

# 2. Agent tracks obligations
agent.run(
    "List all obligations for lease LEASE-001. "
    "Show deadlines, owners, and amounts. "
    "Which obligations are due in the next 30 days? "
    "Which obligations are impacted by regulation REG-2024-001?"
)
```

---

## METADATA SCHEMA REQUIREMENTS

### Required Metadata Fields

**For All Documents**:
```python
{
    "document_type": str,  # "lease", "contract", "regulation", etc.
    "document_id": str,    # Unique identifier
    "version": str,        # Version number
    "version_date": str,   # ISO date
    "previous_version_id": str | None,  # For version tracking
}
```

**For Leases**:
```python
{
    "lease_id": str,
    "tenant_name": str,
    "landlord_name": str,
    "property_address": str,
    "lease_start_date": str,
    "lease_end_date": str,
    "regulatory_references": List[str],  # Regulation IDs
    "contract_references": List[str],     # Contract IDs
}
```

**For Contracts**:
```python
{
    "contract_id": str,
    "contract_name": str,
    "party_a": str,
    "party_b": str,
    "effective_date": str,
    "expiration_date": str,
    "regulatory_references": List[str],
    "lease_references": List[str],
}
```

**For Regulations**:
```python
{
    "regulation_id": str,
    "regulation_name": str,
    "jurisdiction": str,
    "effective_date": str,
    "impacted_contracts": List[str],
    "impacted_leases": List[str],
}
```

**For Obligations**:
```python
{
    "obligation_id": str,
    "type": str,  # "rent_payment", "compliance_reporting", etc.
    "deadline": str,
    "owner": str,
    "amount": float | None,
    "depends_on": List[str],  # Other obligation IDs
}
```

**For Clauses**:
```python
{
    "clause_id": str,
    "type": str,  # "rent", "payment_terms", etc.
    "section": str,  # "Section 3.1"
    "text": str,
    "evolution_history": List[Dict],  # For clause evolution
}
```

---

## CHUNKING STRATEGIES

### Clause-Aware Chunking

**For Contracts and Leases**:
```python
# Use paragraph-based chunking to preserve clause boundaries
service = await get_document_indexing_service(
    collection_name="language_intelligence_documents",
    chunk_size=2000,  # Larger chunks for legal documents
    chunk_overlap=400,  # More overlap to preserve context
    chunking_strategy="paragraph"  # Preserves clause structure
)
```

### Version-Aware Chunking

**For Version Comparison**:
- Chunk at same boundaries across versions
- Include version metadata in each chunk
- Preserve section/clause identifiers

### Context Preservation

**For Obligation Tracking**:
- Include document context in chunks (document_id, version, section)
- Preserve relationships (obligation → clause → document)
- Maintain metadata links across chunks

---

## API ENDPOINTS

### Index Document

```bash
POST /api/v1/documents/index/file
Content-Type: multipart/form-data

file: <file>
document_id: LEASE-001
title: Corporate Lease Agreement
doc_type: legal_document
industry: legal
metadata: {
    "document_type": "lease",
    "lease_id": "LEASE-001",
    "version": "1.0",
    "regulatory_references": ["REG-2024-001"]
}
```

### Search Documents

```bash
POST /api/v1/documents/search
Content-Type: application/json

{
    "query": "rent payment obligations",
    "doc_types": ["lease"],
    "metadata_filters": {
        "lease_id": "LEASE-001"
    },
    "top_k": 10
}
```

### Get Document Versions

```bash
GET /api/v1/documents/{document_id}/versions
```

### Get Obligations

```bash
GET /api/v1/documents/{document_id}/obligations
```

---

## BEST PRACTICES

### 1. Version Tracking

- Always include `version` and `previous_version_id` in metadata
- Use consistent versioning scheme (semantic versioning recommended)
- Index all versions to enable drift detection

### 2. Relationship Mapping

- Link documents via `regulatory_references`, `contract_references`, `lease_references`
- Use consistent IDs across related documents
- Index relationships bidirectionally when possible

### 3. Obligation Extraction

- Extract obligations during indexing or via agents
- Include obligation dependencies in metadata
- Track deadlines and owners for obligation tracking

### 4. Clause Identification

- Preserve clause boundaries during chunking
- Include clause metadata (section, type, ID) in chunks
- Track clause evolution across versions

### 5. Collection Organization

- Use separate collections for different document types if needed
- Or use single collection with metadata filtering
- Consider collection per use case (leases, contracts, regulations)

### 6. LangChain/LangGraph Integration

- Use MilvusVectorStore's LangChain retriever interface
- Configure retrievers with appropriate filters
- Use RAG chains for document-augmented generation
- Leverage LangGraph for multi-step analysis workflows

---

## SUMMARY

### What's Ready ✅

- Document Indexing Service supports all document types
- Milvus integration is production-ready
- LangChain retriever interface available
- LangGraph workflow support exists
- Metadata schema supports all use cases

### What's Needed 🔧

1. **Enhanced Metadata Extraction**: Extract obligations, clauses, relationships during indexing
2. **Version Comparison Tools**: Utilities to compare document versions
3. **Relationship Graph Builder**: Build dependency graphs from indexed documents
4. **Pattern Mining Agents**: LangGraph agents for compliance pattern analysis
5. **Drift Detection Algorithms**: Compare documents against templates/standards

### Next Steps

1. Index sample documents (leases, contracts, regulations)
2. Test LangChain retrievers with document queries
3. Build LangGraph workflows for specific use cases
4. Implement metadata extraction for obligations and clauses
5. Create version comparison and drift detection tools

---

## STEP-BY-STEP BEGINNER'S GUIDE

### Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.9+ installed
- [ ] Docker and Docker Compose installed (for Milvus)
- [ ] diri-cyrex codebase cloned and set up
- [ ] Basic understanding of Python
- [ ] Terminal/command line access

### Step 1: Start Milvus Database

**1.1. Check if Milvus is already running:**

```bash
docker ps | grep milvus
```

If you see Milvus containers, skip to Step 2.

**1.2. Start Milvus using Docker Compose:**

```bash
# Navigate to deepiri-platform directory
cd deepiri-platform

# Start Milvus (if in docker-compose.yml)
docker-compose up -d milvus

# OR start standalone Milvus
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest \
  milvus run standalone
```

**1.3. Verify Milvus is running:**

```bash
# Check container status
docker ps | grep milvus

# Test connection (should return connection info)
curl http://localhost:9091/healthz
```

**Expected Output**: `{"status":"ok"}` or similar

---

### Step 2: Set Environment Variables

**2.1. Create or update `.env` file in `diri-cyrex/` directory:**

```bash
cd deepiri-platform/diri-cyrex
nano .env  # or use your preferred editor
```

**2.2. Add these variables:**

```bash
# Milvus Connection
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Document Collection Name
DOCUMENT_COLLECTION_NAME=language_intelligence_documents

# Embedding Model (optional - defaults to all-MiniLM-L6-v2)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# RAG Settings
RAG_ENABLED=true
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.7
```

**2.3. Save and exit the file**

---

### Step 3: Install Dependencies

**3.1. Navigate to diri-cyrex directory:**

```bash
cd deepiri-platform/diri-cyrex
```

**3.2. Create virtual environment (if not already created):**

```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3.3. Install required packages:**

```bash
pip install -r requirements.txt

# If specific packages are missing, install them:
pip install pymilvus langchain langchain-community sentence-transformers
pip install fastapi uvicorn python-multipart
pip install PyPDF2 python-docx pandas openpyxl
```

---

### Step 4: Test Milvus Connection

**4.1. Create a test script `test_milvus.py`:**

```python
# test_milvus.py
from pymilvus import connections, utility
import os

# Connect to Milvus
try:
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )
    print("✅ Successfully connected to Milvus!")
    
    # List existing collections
    collections = utility.list_collections()
    print(f"📚 Existing collections: {collections}")
    
except Exception as e:
    print(f"❌ Failed to connect to Milvus: {e}")
    print("💡 Make sure Milvus is running: docker ps | grep milvus")
```

**4.2. Run the test:**

```bash
python test_milvus.py
```

**Expected Output**: `✅ Successfully connected to Milvus!`

If you see an error, go back to Step 1 and verify Milvus is running.

---

### Step 5: Index Your First Document

**5.1. Create a test document directory:**

```bash
mkdir -p test_documents
```

**5.2. Add a test document:**

Create `test_documents/sample_lease.txt` with this content:

```
LEASE AGREEMENT

Lease ID: LEASE-001
Tenant: Acme Corporation
Landlord: Property Management LLC
Property Address: 123 Main Street, City, State 12345

Lease Term:
Start Date: January 1, 2024
End Date: December 31, 2029

Financial Terms:
Monthly Rent: $5,000.00
Security Deposit: $10,000.00
Payment Due: 1st of each month

Obligations:
1. Tenant shall pay monthly rent on or before the 1st of each month
2. Landlord shall maintain common areas
3. Tenant shall maintain property in good condition
4. Landlord shall provide 24-hour emergency maintenance

This lease is subject to Commercial Lease Standards Act 2024.
```

**5.3. Create indexing script `index_test_document.py`:**

```python
# index_test_document.py
import asyncio
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.document_indexing_service import (
    get_document_indexing_service,
    B2BDocumentType
)

async def index_test_document():
    print("🚀 Starting document indexing...")
    
    try:
        # Initialize service
        print("📦 Initializing Document Indexing Service...")
        service = await get_document_indexing_service(
            collection_name="language_intelligence_documents",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_strategy="paragraph"
        )
        print("✅ Service initialized!")
        
        # Index the test document
        print("📄 Indexing test document...")
        result = await service.index_file(
            file_path="test_documents/sample_lease.txt",
            document_id="LEASE-001",
            title="Sample Lease Agreement - 123 Main St",
            doc_type=B2BDocumentType.LEGAL_DOCUMENT,
            industry="legal",
            metadata={
                "document_type": "lease",
                "lease_id": "LEASE-001",
                "tenant_name": "Acme Corporation",
                "landlord_name": "Property Management LLC",
                "property_address": "123 Main Street",
                "lease_start_date": "2024-01-01",
                "lease_end_date": "2029-12-31",
                "version": "1.0"
            }
        )
        
        print(f"✅ Document indexed successfully!")
        print(f"   Document ID: {result.document_id}")
        print(f"   Title: {result.title}")
        print(f"   Chunks: {result.chunk_count}")
        print(f"   Format: {result.format.value}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(index_test_document())
    if result:
        print("\n🎉 Success! Your document is now indexed in Milvus!")
    else:
        print("\n💥 Failed to index document. Check the error above.")
```

**5.4. Run the indexing script:**

```bash
python index_test_document.py
```

**Expected Output**:
```
🚀 Starting document indexing...
📦 Initializing Document Indexing Service...
✅ Service initialized!
📄 Indexing test document...
✅ Document indexed successfully!
   Document ID: LEASE-001
   Title: Sample Lease Agreement - 123 Main St
   Chunks: 3
   Format: txt

🎉 Success! Your document is now indexed in Milvus!
```

---

### Step 6: Search Your Indexed Document

**6.1. Create search script `search_documents.py`:**

```python
# search_documents.py
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.document_indexing_service import get_document_indexing_service

async def search_documents():
    print("🔍 Searching indexed documents...")
    
    try:
        # Initialize service
        service = await get_document_indexing_service(
            collection_name="language_intelligence_documents"
        )
        
        # Search for documents
        query = "What are the rent payment obligations?"
        print(f"📝 Query: {query}\n")
        
        results = await service.search(
            query=query,
            top_k=5
        )
        
        if results:
            print(f"✅ Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"--- Result {i} ---")
                print(f"Title: {result['title']}")
                print(f"Type: {result['doc_type']}")
                print(f"Score: {result['score']:.4f}")
                print(f"Content: {result['content'][:200]}...")
                print()
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(search_documents())
```

**6.2. Run the search:**

```bash
python search_documents.py
```

**Expected Output**:
```
🔍 Searching indexed documents...
📝 Query: What are the rent payment obligations?

✅ Found 1 results:

--- Result 1 ---
Title: Sample Lease Agreement - 123 Main St (Chunk 1/3)
Type: legal_document
Score: 0.8234
Content: LEASE AGREEMENT

Lease ID: LEASE-001
Tenant: Acme Corporation
...
```

---

### Step 7: Use with LangChain

**7.1. Create LangChain test script `test_langchain_rag.py`:**

```python
# test_langchain_rag.py
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.integrations.milvus_store import get_milvus_store
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

async def test_langchain_rag():
    print("🤖 Testing LangChain RAG with indexed documents...")
    
    try:
        # Get Milvus vector store
        print("📦 Getting Milvus vector store...")
        vector_store = get_milvus_store(
            collection_name="language_intelligence_documents",
            host="localhost",
            port=19530
        )
        print("✅ Vector store ready!")
        
        # Create retriever
        print("🔍 Creating retriever...")
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        print("✅ Retriever created!")
        
        # Initialize LLM (using Ollama - make sure it's running)
        print("🧠 Initializing LLM...")
        try:
            llm = Ollama(model="llama3:8b")
            print("✅ LLM initialized!")
        except Exception as e:
            print(f"⚠️  LLM not available: {e}")
            print("💡 Install Ollama and pull llama3:8b model")
            return
        
        # Create RAG chain
        print("🔗 Creating RAG chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        print("✅ RAG chain created!")
        
        # Ask a question
        question = "What is the monthly rent amount in lease LEASE-001?"
        print(f"\n❓ Question: {question}\n")
        
        result = qa_chain.invoke({"query": question})
        
        print("💬 Answer:")
        print(result["result"])
        print(f"\n📚 Sources: {len(result['source_documents'])} documents")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_langchain_rag())
```

**7.2. Make sure Ollama is running (optional - for LLM):**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it (if installed)
ollama serve
```

**7.3. Run the LangChain test:**

```bash
python test_langchain_rag.py
```

---

### Step 8: Index a Real PDF Document

**8.1. Get a sample PDF (lease, contract, or regulation)**

Place it in `test_documents/` directory.

**8.2. Create script to index PDF `index_pdf.py`:**

```python
# index_pdf.py
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.document_indexing_service import (
    get_document_indexing_service,
    B2BDocumentType
)

async def index_pdf():
    # Get PDF file path from command line or use default
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test_documents/sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        print("💡 Usage: python index_pdf.py <path_to_pdf>")
        return
    
    print(f"📄 Indexing PDF: {pdf_path}")
    
    try:
        service = await get_document_indexing_service(
            collection_name="language_intelligence_documents"
        )
        
        result = await service.index_file(
            file_path=pdf_path,
            document_id=f"DOC-{os.path.basename(pdf_path)}",
            title=os.path.basename(pdf_path),
            doc_type=B2BDocumentType.LEGAL_DOCUMENT,
            industry="legal",
            metadata={
                "document_type": "lease",  # Change based on your document
                "version": "1.0"
            }
        )
        
        print(f"✅ Indexed: {result.document_id}")
        print(f"   Chunks: {result.chunk_count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(index_pdf())
```

**8.3. Run with your PDF:**

```bash
python index_pdf.py test_documents/your_lease.pdf
```

---

### Step 9: Get Statistics

**9.1. Create stats script `get_stats.py`:**

```python
# get_stats.py
import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.document_indexing_service import get_document_indexing_service

async def get_stats():
    try:
        service = await get_document_indexing_service(
            collection_name="language_intelligence_documents"
        )
        
        stats = await service.get_statistics()
        
        print("📊 Indexing Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(get_stats())
```

**9.2. Run stats:**

```bash
python get_stats.py
```

---

### Step 10: Use API Endpoints

**10.1. Start the FastAPI server:**

```bash
cd deepiri-platform/diri-cyrex
python -m uvicorn app.main:app --reload --port 8000
```

**10.2. Test indexing via API:**

```bash
# Index a file via API
curl -X POST "http://localhost:8000/api/v1/documents/index/file" \
  -F "file=@test_documents/sample_lease.txt" \
  -F "title=Test Lease" \
  -F "doc_type=legal_document" \
  -F "industry=legal" \
  -F 'metadata={"document_type": "lease", "lease_id": "LEASE-002"}'
```

**10.3. Search via API:**

```bash
curl -X POST "http://localhost:8000/api/v1/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "rent payment obligations",
    "top_k": 5
  }'
```

---

### Troubleshooting

#### Problem: "Cannot connect to Milvus"

**Solution**:
```bash
# Check if Milvus is running
docker ps | grep milvus

# If not running, start it
docker-compose up -d milvus

# Check Milvus logs
docker logs milvus-standalone
```

#### Problem: "Module not found" errors

**Solution**:
```bash
# Make sure you're in the right directory
cd deepiri-platform/diri-cyrex

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# Install missing packages
pip install <package_name>
```

#### Problem: "Collection not found" or "Empty results"

**Solution**:
```bash
# Check if documents are indexed
python get_stats.py

# If no documents, index one first
python index_test_document.py
```

#### Problem: "PDF parsing failed"

**Solution**:
```bash
# Install PDF parsing libraries
pip install PyPDF2 pdfplumber

# Try alternative parser
# Update document_indexing_service.py to use pdfplumber instead of PyPDF2
```

---

### Next Steps

Once you've completed these steps:

1. ✅ **Index more documents** - Add leases, contracts, regulations
2. ✅ **Build LangChain agents** - Create agents that use the indexed documents
3. ✅ **Create LangGraph workflows** - Build multi-step analysis workflows
4. ✅ **Extract metadata** - Add obligation and clause extraction
5. ✅ **Version tracking** - Index multiple versions of documents
6. ✅ **Build use cases** - Implement the 7 Language Intelligence Platform features

---

### Quick Reference Commands

```bash
# Start Milvus
docker-compose up -d milvus

# Check Milvus status
docker ps | grep milvus

# Test Milvus connection
python test_milvus.py

# Index a document
python index_test_document.py

# Search documents
python search_documents.py

# Get statistics
python get_stats.py

# Start API server
python -m uvicorn app.main:app --reload --port 8000
```

---

**Document Created**: January 2026  
**Version**: 1.0  
**Status**: Ready for LangChain/LangGraph Financial Analysis Agents
