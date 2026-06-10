# Cyrex Guard Platform Implementation

**Date**: January 2026  
**Status**: ✅ **IMPLEMENTED**  
**Architecture**: Universal Platform + Industry-Specific LoRA Adapters  
**Product Name**: Cyrex Guard

---

## Overview

Cyrex Guard provides a unified vendor fraud detection platform that works across all 6 industries:

1. **Property Management** - HVAC, plumbing, electrical contractor fraud
2. **Corporate Procurement** - Supplier invoice fraud
3. **P&C Insurance** - Auto body shop, home repair contractor fraud
4. **General Contractors** - Subcontractor and material supplier fraud
5. **Retail/E-Commerce** - Freight carrier and warehouse vendor fraud
6. **Law Firms** - Expert witness and e-discovery vendor fraud

**Key Principle**: ONE platform, SIX industries, SAME fraud detection problem.

---

## Architecture

### Core Services (Runtime - diri-cyrex)

#### 1. Invoice Parser
**Location**: `app/services/invoice_parser.py`

**Purpose**: Process invoices from all industries using OCR + multimodal AI

**Features**:
- Supports multiple formats: text, JSON, PDF, image
- Industry-specific LoRA adapter integration
- Structured data extraction
- Batch processing support

**Usage**:
```python
from app.services.invoice_parser import get_universal_invoice_processor
from app.core.types import IndustryNiche

processor = get_universal_invoice_processor()
processed = await processor.process_invoice(
    invoice_content=invoice_text,
    industry=IndustryNiche.PROPERTY_MANAGEMENT,
    invoice_format="text",
    use_lora=True
)
```

**API Endpoint**: `POST /cyrex-guard/invoice/process`

#### 2. Pricing Benchmark
**Location**: `app/services/pricing_benchmark.py`

**Purpose**: Real-time market rate comparisons across all industries

**Features**:
- Industry-specific pricing data
- Location-based adjustments
- Historical pricing trends
- Cross-industry pricing intelligence

**Usage**:
```python
from app.services.pricing_benchmark import get_pricing_benchmark_engine

engine = get_pricing_benchmark_engine()
await engine.initialize()

comparison = await engine.compare_price(
    invoice_price=1500.00,
    service_category="hvac_repair",
    industry=IndustryNiche.PROPERTY_MANAGEMENT,
    location="New York, NY"
)
```

**API Endpoint**: `POST /cyrex-guard/pricing/compare`

#### 3. LoRA Loader
**Location**: `app/services/lora_loader.py`

**Purpose**: Load and manage industry-specific LoRA adapters for enhanced inference

**Features**:
- Dynamic adapter loading/unloading
- Model registry integration (MLflow/S3)
- Caching for fast inference
- Hot-swapping adapters

**Usage**:
```python
from app.services.lora_loader import get_industry_lora_service

service = get_industry_lora_service()
await service.load_adapter(IndustryNiche.PROPERTY_MANAGEMENT)
result = await service.infer_with_adapter(
    industry=IndustryNiche.PROPERTY_MANAGEMENT,
    prompt="Analyze this invoice..."
)
```

**API Endpoints**:
- `POST /cyrex-guard/lora/load` - Load adapter
- `POST /cyrex-guard/lora/unload` - Unload adapter
- `GET /cyrex-guard/lora/status` - Get adapter status
- `GET /cyrex-guard/lora/list` - List available adapters

#### 4. Fraud Detector
**Location**: `app/services/fraud_detector.py`

**Purpose**: Detect vendor fraud using multiple detection methods

**Features**:
- Anomaly detection (autoencoder-based)
- Pattern matching (rule-based + ML)
- Pricing benchmark comparison
- Vendor intelligence integration
- Predictive risk scoring

**Usage**:
```python
from app.services.fraud_detector import get_universal_fraud_detection_service

fraud_service = get_universal_fraud_detection_service()
result = await fraud_service.detect_fraud(
    invoice=processed_invoice,
    use_vendor_intelligence=True,
    use_pricing_benchmark=True,
    use_anomaly_detection=True,
    use_pattern_matching=True
)
```

**API Endpoints**:
- `POST /cyrex-guard/fraud/detect` - Detect fraud in processed invoice
- `POST /cyrex-guard/fraud/detect-full` - Full pipeline (process + detect)

---

### Training Pipelines (diri-helox)

#### 1. LoRA Industry Training Pipeline
**Location**: `pipelines/training/lora_industry_training.py`

**Purpose**: Train industry-specific LoRA adapters

**Features**:
- Fine-tunes base LLM (llama3:8b) with LoRA
- Industry-specific training data
- Exports to model registry

**Usage**:
```python
from pipelines.training.lora_industry_training import train_industry_lora, IndustryType

results = train_industry_lora(
    industry=IndustryType.PROPERTY_MANAGEMENT,
    training_data_path="data/processed/property_management/train.jsonl",
    validation_data_path="data/processed/property_management/val.jsonl"
)
```

#### 2. Fraud Detection Training Pipeline
**Location**: `pipelines/training/fraud_detection_training.py`

**Purpose**: Train ML models for fraud detection

**Model Types**:
- Anomaly Detector (Autoencoder)
- Pattern Matcher (Classification)
- Risk Scorer (Regression)

**Usage**:
```python
from pipelines.training.fraud_detection_training import train_fraud_detection_model

results = train_fraud_detection_model(
    model_type="anomaly_detector",
    training_data_path="data/processed/fraud_detection/train.jsonl"
)
```

#### 3. Vendor Risk Scoring Training Pipeline
**Location**: `pipelines/training/vendor_risk_scoring_training.py`

**Purpose**: Train models to predict vendor fraud risk

**Usage**:
```python
from pipelines.training.vendor_risk_scoring_training import train_vendor_risk_scorer

results = train_vendor_risk_scorer(
    training_data_path="data/processed/vendor_risk/train.jsonl"
)
```

---

## API Documentation

### Base URL
All endpoints are under `/cyrex-guard`

### Invoice Processing

#### Process Invoice
```http
POST /cyrex-guard/invoice/process
Content-Type: application/json

{
  "invoice_content": "Invoice text or JSON...",
  "industry": "property_management",
  "invoice_format": "text",
  "use_lora": true
}
```

#### Process Batch
```http
POST /cyrex-guard/invoice/process-batch
Content-Type: application/json

{
  "invoices": [...],
  "industry": "property_management",
  "use_lora": true
}
```

### Pricing Benchmarks

#### Compare Price
```http
POST /cyrex-guard/pricing/compare
Content-Type: application/json

{
  "invoice_price": 1500.00,
  "service_category": "hvac_repair",
  "industry": "property_management",
  "location": "New York, NY"
}
```

#### Get Benchmark
```http
POST /cyrex-guard/pricing/benchmark
Content-Type: application/json

{
  "service_category": "hvac_repair",
  "industry": "property_management",
  "location": "New York, NY"
}
```

### LoRA Adapters

#### Load Adapter
```http
POST /cyrex-guard/lora/load
Content-Type: application/json

{
  "industry": "property_management",
  "adapter_id": "optional-id",
  "force_reload": false
}
```

#### Get Status
```http
GET /cyrex-guard/lora/status?industry=property_management
```

### Fraud Detection

#### Detect Fraud (Full Pipeline)
```http
POST /cyrex-guard/fraud/detect-full
Content-Type: multipart/form-data

invoice_content: "Invoice text..."
industry: "property_management"
invoice_format: "text"
use_lora: true
use_vendor_intelligence: true
use_pricing_benchmark: true
use_anomaly_detection: true
use_pattern_matching: true
```

---

## Integration Flow

### Runtime Flow (Cyrex)

```
1. Invoice arrives → Universal Invoice Processor
   ↓
2. Extract structured data (OCR + LoRA)
   ↓
3. Pricing Benchmark Engine → Compare against market rates
   ↓
4. Vendor Intelligence Service → Check vendor history
   ↓
5. Universal Fraud Detection → Multiple detection methods
   ↓
6. Return fraud detection result with risk score
```

### Training Flow (Helox)

```
1. Collect training data → Industry-specific datasets
   ↓
2. Train LoRA adapters → Industry-specific fine-tuning
   ↓
3. Train fraud detection models → Anomaly, pattern, risk
   ↓
4. Export to model registry → MLflow/S3
   ↓
5. Cyrex loads models → Runtime inference
```

---

## Example Usage

See `examples/cyrex_guard_example.py` for complete examples:

- Process invoice
- Compare price
- Get benchmark
- Manage LoRA adapters
- Detect fraud
- Full pipeline

Run examples:
```bash
cd diri-cyrex
python examples/cyrex_guard_example.py
```

---

## Testing

Integration tests are in `tests/test_cyrex_guard.py`:

```bash
pytest tests/test_cyrex_guard.py -v
```

---

## Next Steps

1. ✅ Core services implemented
2. ✅ API routes created
3. ✅ Integration tests added
4. ✅ Example scripts created
5. ⏳ Train initial LoRA adapters
6. ⏳ Train fraud detection models
7. ⏳ Deploy to production
8. ⏳ Monitor and iterate

---

## Related Documentation

- [Vendor Fraud System](./CYREX_VENDOR_FRAUD_SYSTEM.md)
- [Architecture Overview](../architecture/ARCHITECTURE.md)
- [NICHES_FINALIZED.md](../../../My Deepiri Personal Docs/NICHES_FINALIZED.md)

