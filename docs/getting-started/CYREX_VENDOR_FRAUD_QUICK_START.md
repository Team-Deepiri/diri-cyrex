# Cyrex Vendor Fraud Detection - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Start the Backend

```bash
cd deepiri-platform/diri-cyrex

# Option A: Using Docker (Recommended)
docker compose -f ../docker-compose.dev.yml up -d cyrex milvus redis

# Option B: Local Development
pip install -r requirements.txt
python -m app.main
```

The API will be available at: `http://localhost:8000`

### Step 2: Start the Frontend

```bash
cd cyrex-interface
npm install
npm run dev
```

The UI will be available at: `http://localhost:5173` (or the port Vite assigns)

### Step 3: Access Vendor Fraud Detection

1. Open your browser to the frontend URL
2. Click **"Vendor Fraud Detection"** in the sidebar (💰 icon)
3. You'll see the full Vendor Fraud Detection panel!

---

## 📋 First Invoice Analysis

### Using the UI:

1. **Select Industry**: Choose from dropdown (e.g., "Property Management")
2. **Enter Invoice Details**:
   - Vendor Name: "ABC Plumbing Services"
   - Invoice Number: "INV-2026-001"
   - Service Category: "Plumbing Emergency"
   - Work Description: "Emergency pipe repair at 123 Main St"
3. **Add Line Items**:
   - Click "+ Add Line Item"
   - Description: "Emergency pipe repair"
   - Quantity: 1
   - Unit Price: 1500
4. **Click "🔍 Analyze Invoice"**
5. **View Results**:
   - Risk Level (color-coded badge)
   - Risk Score (0-100)
   - Fraud Indicators (if any)
   - Recommendations

### Using the API:

```bash
curl -X POST http://localhost:8000/vendor-fraud/analyze-invoice \
  -H "Content-Type: application/json" \
  -H "x-api-key: change-me" \
  -d '{
    "invoice": {
      "vendor_name": "ABC Plumbing Services",
      "invoice_number": "INV-2026-001",
      "total_amount": 1500,
      "line_items": [
        {
          "description": "Emergency pipe repair",
          "quantity": 1,
          "unit_price": 1500
        }
      ],
      "service_category": "plumbing_emergency",
      "work_description": "Emergency pipe repair at 123 Main St"
    },
    "industry": "property_management"
  }'
```

**Expected Response:**
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
        "description": "Invoice price is 200% above market rate"
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

---

## 🎯 Key Features

### 1. Invoice Analysis
- **Automatic fraud detection** across 6 industries
- **Pricing benchmark comparison**
- **Risk scoring** (0-100)
- **Actionable recommendations**

### 2. Chat with Agent
- Ask questions about vendor fraud
- Get industry-specific advice
- Query the knowledge base

### 3. Document Ingestion
- Add vendor invoices to knowledge base
- Upload pricing guides
- Index fraud patterns

### 4. Pricing Benchmarks
- Check market rates for services
- Compare quoted prices
- Get deviation percentages

---

## 🏭 Supported Industries

| Industry | Icon | Common Fraud Types |
|----------|------|-------------------|
| **Property Management** | 🏢 | HVAC, plumbing, electrical contractor fraud |
| **Corporate Procurement** | 📦 | Supplier invoice fraud, PO manipulation |
| **P&C Insurance** | 🛡️ | Auto body shop, home repair contractor fraud |
| **General Contractors** | 🏗️ | Subcontractor and material supplier fraud |
| **Retail & E-Commerce** | 🛒 | Freight carrier, warehouse vendor fraud |
| **Law Firms** | ⚖️ | Expert witness, e-discovery vendor fraud |

---

## 🔍 Fraud Types Detected

1. **Inflated Invoices** - Prices 20-50% above market
2. **Phantom Work** - Billing for work never performed
3. **Duplicate Billing** - Same work billed multiple times
4. **Unnecessary Services** - Recommending unneeded repairs
5. **Kickback Schemes** - Internal staff receiving kickbacks
6. **Price Gouging** - Exploiting emergencies
7. **Contract Non-Compliance** - Billing outside contract terms
8. **Forged Documents** - Fake invoices or work orders

---

## 📊 Risk Levels

| Score | Level | Color | Action |
|-------|-------|-------|--------|
| 0-24 | **LOW** | 🟢 Green | Standard approval |
| 25-49 | **MEDIUM** | 🟠 Orange | Review carefully |
| 50-69 | **HIGH** | 🔴 Red | Request documentation |
| 70-100 | **CRITICAL** | 🟣 Purple | Halt payment, audit |

---

## 🛠️ API Endpoints

### Analyze Invoice
```bash
POST /vendor-fraud/analyze-invoice
```

### Get Vendor Profile
```bash
POST /vendor-fraud/vendor-profile
```

### Check Pricing Benchmark
```bash
POST /vendor-fraud/pricing-benchmark
```

### Ingest Document
```bash
POST /vendor-fraud/ingest-document
```

### Query Knowledge Base
```bash
POST /vendor-fraud/query
```

### Chat with Agent
```bash
POST /vendor-fraud/chat
```

### List Industries
```bash
GET /vendor-fraud/industries
```

### Health Check
```bash
GET /vendor-fraud/health
```

---

## 📖 Example Workflows

### Workflow 1: Analyze Property Management Invoice

```bash
# 1. Analyze invoice
curl -X POST http://localhost:8000/vendor-fraud/analyze-invoice \
  -H "Content-Type: application/json" \
  -d '{
    "invoice": {
      "vendor_name": "HVAC Pro Services",
      "total_amount": 1200,
      "line_items": [
        {"description": "AC unit repair", "quantity": 1, "unit_price": 1200}
      ],
      "service_category": "hvac_repair"
    },
    "industry": "property_management"
  }'

# 2. Check pricing benchmark
curl -X POST http://localhost:8000/vendor-fraud/pricing-benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "service_type": "hvac_repair",
    "industry": "property_management",
    "price_to_check": 1200
  }'
```

### Workflow 2: Ingest Knowledge Base Document

```bash
# Add a pricing guide
curl -X POST http://localhost:8000/vendor-fraud/ingest-document \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Property Management Pricing Guide 2026",
    "content": "Standard HVAC repair rates: $150-800 per visit (average $350)...",
    "industry": "property_management",
    "doc_type": "pricing_guide"
  }'

# Query the knowledge base
curl -X POST http://localhost:8000/vendor-fraud/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are standard HVAC repair rates?",
    "industry": "property_management"
  }'
```

### Workflow 3: Chat with Agent

```bash
curl -X POST http://localhost:8000/vendor-fraud/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are common HVAC contractor fraud patterns?",
    "industry": "property_management"
  }'
```

---

## 🎨 UI Features

### Invoice Analysis Tab
- ✅ Vendor name and invoice number input
- ✅ Service category dropdown
- ✅ Dynamic line item management
- ✅ Real-time total calculation
- ✅ One-click analysis
- ✅ Visual risk indicators
- ✅ Fraud indicators list
- ✅ Recommendations display

### Chat Tab
- ✅ Conversational interface
- ✅ Session persistence
- ✅ Tool call visibility
- ✅ Message history

### Documents Tab
- ✅ Document ingestion
- ✅ Title and content input
- ✅ Industry selection
- ✅ Success notifications

### Benchmarks Tab
- ✅ Service type selection
- ✅ Price comparison
- ✅ Market rate display
- ✅ Deviation calculation

---

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
CYREX_API_KEY=your-api-key-here

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=sk-...  # Optional

# Vector Store
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis (for LangGraph checkpointing)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Frontend Configuration

Create `.env` file in `cyrex-interface/`:

```bash
VITE_CYREX_BASE_URL=http://localhost:8000
```

---

## 🐛 Troubleshooting

### Backend not starting?
```bash
# Check if port 8000 is available
netstat -an | grep 8000

# Check logs
docker compose logs cyrex
```

### Frontend not connecting?
```bash
# Verify API URL in browser console
# Check CORS settings
# Verify API key is set
```

### No fraud detected?
- Check if invoice data is complete
- Verify industry selection matches invoice type
- Ensure pricing benchmarks are available for service category

### RAG not working?
- Verify Milvus is running: `docker compose ps milvus`
- Check Milvus connection: `http://localhost:19530`
- Ingest some documents first

---

## 📚 Next Steps

1. **Populate Knowledge Base**: Ingest pricing guides and fraud patterns
2. **Train LoRA Adapters**: Fine-tune for your specific industry
3. **Add Custom Benchmarks**: Update pricing data for your region
4. **Integrate with Your Systems**: Connect to your invoice processing pipeline
5. **Set Up Alerts**: Configure webhooks for high-risk invoices

---

## 📞 Support

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/vendor-fraud/health
- **System Documentation**: See `docs/CYREX_VENDOR_FRAUD_SYSTEM.md`

---

**🎉 You're all set! Start detecting vendor fraud with Cyrex!**


