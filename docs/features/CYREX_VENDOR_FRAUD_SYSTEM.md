# Cyrex Vendor Fraud Detection System

## Complete Implementation Guide

**Date**: January 4, 2026  
**Version**: 1.0.0  
**Architecture**: LangGraph + LangChain + Milvus RAG

---

## Overview

The Cyrex Vendor Fraud Detection System is a production-ready AI platform for detecting vendor and supplier fraud across six industries. It uses:

- **LangGraph** for multi-agent workflow orchestration
- **LangChain** for LLM integration and tool management
- **Milvus** for vector storage and RAG retrieval
- **Industry-specific LoRA adapters** for specialized analysis

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cyrex Vendor Fraud System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Frontend   │────▶│   REST API   │────▶│  LangGraph   │    │
│  │ (React/TS)   │     │  (FastAPI)   │     │  Workflow    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Multi-Agent Workflow (LangGraph)             │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                           │  │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐  │  │
│  │  │Document │──▶│ Vendor  │──▶│ Pricing │──▶│ Fraud   │  │  │
│  │  │Processor│   │ Intel   │   │ Analyzer│   │ Detector│  │  │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘  │  │
│  │                                                   │       │  │
│  │                                                   ▼       │  │
│  │                                           ┌─────────┐    │  │
│  │                                           │  Risk   │    │  │
│  │                                           │Assessor │    │  │
│  │                                           └─────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    RAG Knowledge Base                     │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │   Milvus Vector Store   │   Industry-Specific LoRA       │  │
│  │   - Vendor Profiles     │   - Property Management        │  │
│  │   - Pricing Benchmarks  │   - Corporate Procurement      │  │
│  │   - Fraud Patterns      │   - P&C Insurance              │  │
│  │   - Historical Data     │   - General Contractors        │  │
│  │                         │   - Retail/E-Commerce          │  │
│  │                         │   - Law Firms                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. VendorFraudAgent (`app/agents/implementations/vendor_fraud_agent.py`)

The main agent class that orchestrates fraud detection:

```python
from app.agents.implementations import VendorFraudAgent
from app.core.types import AgentConfig, AgentRole, IndustryNiche

# Create agent
config = AgentConfig(
    role=AgentRole.FRAUD_DETECTOR,
    name="Cyrex Vendor Fraud Detector",
    temperature=0.3,
)

agent = VendorFraudAgent(
    agent_config=config,
    industry=IndustryNiche.PROPERTY_MANAGEMENT,
)

# Analyze invoice
result = await agent.analyze_invoice(
    invoice_data={
        "vendor_name": "ABC Plumbing",
        "total_amount": 1500,
        "line_items": [
            {"description": "Emergency pipe repair", "total": 1500}
        ],
    },
    industry=IndustryNiche.PROPERTY_MANAGEMENT,
)
```

### 2. LangGraph Workflow

The agent uses a 5-node LangGraph workflow:

1. **Document Processor**: Extracts data from invoices
2. **Vendor Intelligence**: Gathers vendor history and profile
3. **Pricing Analyzer**: Compares prices against benchmarks
4. **Fraud Detector**: Identifies fraud patterns
5. **Risk Assessor**: Calculates risk score and recommendations

```python
# Workflow state
class VendorFraudState(TypedDict):
    messages: List[Any]
    workflow_id: str
    invoice_data: Dict[str, Any]
    extracted_data: Dict[str, Any]
    pricing_analysis: Dict[str, Any]
    vendor_profile: Dict[str, Any]
    fraud_indicators: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    fraud_detected: bool
    risk_level: str
    recommendations: List[str]
```

### 3. Industry-Specific Tools (`app/agents/tools/vendor_fraud_tools.py`)

Pre-built tools for fraud detection:

```python
# Available tools
- analyze_invoice_for_fraud()  # Main analysis
- get_pricing_benchmark()       # Market rate lookup
- check_duplicate_invoices()    # Duplicate detection
- calculate_vendor_risk_score() # Risk scoring
```

### 4. REST API (`app/routes/vendor_fraud_api.py`)

Full REST API for all operations:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vendor-fraud/analyze-invoice` | POST | Analyze invoice for fraud |
| `/vendor-fraud/vendor-profile` | POST | Get vendor intelligence |
| `/vendor-fraud/pricing-benchmark` | POST | Check pricing benchmarks |
| `/vendor-fraud/ingest-document` | POST | Add document to RAG |
| `/vendor-fraud/query` | POST | Query knowledge base |
| `/vendor-fraud/chat` | POST | Chat with agent |
| `/vendor-fraud/industries` | GET | List supported industries |
| `/vendor-fraud/health` | GET | Health check |

### 5. Frontend (`cyrex-interface/src/components/VendorFraud/`)

React components for the UI:

- `VendorFraudPanel.tsx` - Main panel component
- `api.ts` - API client
- `types.ts` - TypeScript types
- `VendorFraudPanel.css` - Styling

---

## Supported Industries

| Industry | ID | Common Fraud Types |
|----------|----|--------------------|
| Property Management | `property_management` | HVAC, plumbing, electrical contractor fraud |
| Corporate Procurement | `corporate_procurement` | Supplier invoice fraud, PO manipulation |
| P&C Insurance | `insurance_pc` | Auto body shop, home repair contractor fraud |
| General Contractors | `general_contractors` | Subcontractor and material supplier fraud |
| Retail & E-Commerce | `retail_ecommerce` | Freight carrier, warehouse vendor fraud |
| Law Firms | `law_firms` | Expert witness, e-discovery vendor fraud |

---

## Fraud Types Detected

1. **Inflated Invoices**: Prices 20-50% above market rates
2. **Phantom Work**: Billing for work never performed
3. **Duplicate Billing**: Same work billed multiple times
4. **Unnecessary Services**: Recommending unneeded repairs
5. **Kickback Schemes**: Internal staff receiving kickbacks
6. **Price Gouging**: Exploiting emergencies for excessive pricing
7. **Contract Non-Compliance**: Billing outside contract terms
8. **Forged Documents**: Fake invoices or work orders

---

## API Usage Examples

### Analyze Invoice

```bash
curl -X POST http://localhost:8000/vendor-fraud/analyze-invoice \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "invoice": {
      "vendor_name": "ABC Plumbing Services",
      "vendor_id": "vendor-123",
      "invoice_number": "INV-2026-001",
      "total_amount": 1500,
      "line_items": [
        {
          "description": "Emergency pipe repair",
          "quantity": 1,
          "unit_price": 1500
        }
      ],
      "service_category": "plumbing_emergency"
    },
    "industry": "property_management"
  }'
```

**Response:**

```json
{
  "success": true,
  "analysis": {
    "fraud_detected": true,
    "risk_level": "high",
    "risk_score": 65,
    "confidence_score": 0.85,
    "fraud_indicators": [
      {
        "type": "inflated_invoice",
        "severity": "high",
        "description": "Invoice price is 50% above market rate"
      }
    ],
    "recommendations": [
      "Request itemized breakdown with labor hours and material costs",
      "Obtain competitive quotes from alternative vendors",
      "Verify work completion before payment"
    ]
  }
}
```

### Check Pricing Benchmark

```bash
curl -X POST http://localhost:8000/vendor-fraud/pricing-benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "service_type": "plumbing_emergency",
    "industry": "property_management",
    "price_to_check": 1500
  }'
```

**Response:**

```json
{
  "success": true,
  "benchmark": {
    "service_type": "plumbing_emergency",
    "industry": "property_management",
    "found": true,
    "benchmark": {
      "min": 200,
      "avg": 500,
      "max": 1000,
      "unit": "per visit"
    },
    "price_checked": 1500,
    "deviation_percent": 200.0,
    "status": "above_market"
  }
}
```

### Chat with Agent

```bash
curl -X POST http://localhost:8000/vendor-fraud/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are common HVAC contractor fraud patterns?",
    "industry": "property_management"
  }'
```

### Ingest Document

```bash
curl -X POST http://localhost:8000/vendor-fraud/ingest-document \
  -H "Content-Type: application/json" \
  -d '{
    "title": "HVAC Pricing Guide 2026",
    "content": "Standard HVAC repair rates for residential properties...",
    "industry": "property_management",
    "doc_type": "pricing_guide"
  }'
```

---

## Running the System

### 1. Start Cyrex Backend

```bash
cd deepiri-platform/diri-cyrex

# With Docker
docker compose -f ../docker-compose.dev.yml up -d cyrex milvus

# Or locally
pip install -r requirements.txt
python -m app.main
```

### 2. Start Frontend (Optional)

```bash
cd cyrex-interface
npm install
npm run dev
```

### 3. Access API Documentation

Open http://localhost:8000/docs for Swagger UI

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...        # Optional: OpenAI API key
OLLAMA_BASE_URL=http://localhost:11434  # Local Ollama

# Vector Store
MILVUS_HOST=localhost
MILVUS_PORT=19530

# API
CYREX_API_KEY=your-api-key
```

### Agent Configuration

```python
AgentConfig(
    role=AgentRole.FRAUD_DETECTOR,
    name="Cyrex Vendor Fraud Detector",
    temperature=0.3,  # Lower for consistent analysis
    max_tokens=4000,  # Longer responses for detailed analysis
    capabilities=[
        "invoice_analysis",
        "vendor_intelligence",
        "pricing_benchmarks",
        "fraud_detection",
        "risk_assessment",
    ],
)
```

---

## LangGraph Workflow Flow

```
START
  │
  ▼
┌─────────────────────┐
│ 1. Document         │  Extract invoice data, parse line items
│    Processor        │  Use LLM to extract structured data
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ 2. Vendor           │  Query RAG for vendor history
│    Intelligence     │  Build vendor profile
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ 3. Pricing          │  Compare against market benchmarks
│    Analyzer         │  Calculate price deviations
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ 4. Fraud            │  Identify fraud patterns
│    Detector         │  Check for duplicates, phantom work
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ 5. Risk             │  Calculate risk score (0-100)
│    Assessor         │  Determine risk level, recommendations
└─────────────────────┘
  │
  ▼
 END
```

---

## Risk Scoring

### Risk Score Calculation

| Factor | Points |
|--------|--------|
| Each fraud indicator (low severity) | +10 |
| Each fraud indicator (medium severity) | +25 |
| Each fraud indicator (high severity) | +40 |
| Each fraud indicator (critical severity) | +60 |
| Previous vendor fraud flags | +15 each |
| Price deviation >50% | +30 |
| Price deviation 30-50% | +20 |
| Price deviation 20-30% | +10 |

### Risk Levels

| Score | Level | Recommended Action |
|-------|-------|-------------------|
| 0-24 | LOW | Standard approval process |
| 25-49 | MEDIUM | Review line items carefully |
| 50-69 | HIGH | Request documentation before payment |
| 70-100 | CRITICAL | Halt payment, conduct audit |

---

## RAG Knowledge Base

### Document Types

- **Vendor Invoices**: Historical invoice data for pattern detection
- **Pricing Guides**: Market rate benchmarks by service and location
- **Fraud Patterns**: Known fraud schemes and indicators
- **Vendor Profiles**: Vendor history, performance, and risk data
- **Industry Standards**: Building codes, regulations, standards

### Collections (Milvus)

- `cyrex_property_management_vendor_fraud`
- `cyrex_corporate_procurement_vendor_fraud`
- `cyrex_insurance_pc_vendor_fraud`
- `cyrex_general_contractors_vendor_fraud`
- `cyrex_retail_ecommerce_vendor_fraud`
- `cyrex_law_firms_vendor_fraud`

---

## Files Created

### Backend

```
app/
├── agents/
│   ├── implementations/
│   │   └── vendor_fraud_agent.py    # Main agent
│   ├── prompts/
│   │   └── vendor_fraud_prompts.py  # Industry prompts
│   └── tools/
│       └── vendor_fraud_tools.py    # Detection tools
├── routes/
│   └── vendor_fraud_api.py          # REST API
└── core/
    └── types.py                      # Extended with new types
```

### Frontend

```
cyrex-interface/src/components/VendorFraud/
├── VendorFraudPanel.tsx    # Main UI component
├── VendorFraudPanel.css    # Styles
├── api.ts                   # API client
├── types.ts                 # TypeScript types
└── index.ts                 # Exports
```

---

## Next Steps

1. **Populate Knowledge Base**: Ingest pricing guides, fraud patterns, historical data
2. **Train LoRA Adapters**: Fine-tune models for each industry
3. **Add More Tools**: Document OCR, image verification
4. **Implement Webhooks**: Real-time fraud alerts
5. **Build Analytics Dashboard**: Trend analysis, reporting
6. **Add Multi-Language Support**: Internationalization

---

## Support

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/vendor-fraud/health
- **Industries**: http://localhost:8000/vendor-fraud/industries

---

*Built with Cyrex AI Platform - January 2026*


