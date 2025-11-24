# Deepiri AI System - Complete Implementation Guide

## 🎯 Overview

The Deepiri AI System is a three-tier architecture fully integrated with LangChain, providing:
- **Maximum Reliability** (Tier 1): Predefined ability classification
- **High Creativity** (Tier 2): Dynamic ability generation
- **Adaptive Learning** (Tier 3): RL-based optimization

---

## 📁 File Structure

```
diri-cyrex/app/
├── services/
│   ├── command_router.py                  # Tier 1: BERT/DeBERTa command routing
│   ├── contextual_ability_engine.py       # Tier 2: LLM + RAG ability generation
│   ├── workflow_optimizer.py             # Tier 3: PPO RL workflow optimization
│   └── knowledge_retrieval_engine.py      # Knowledge retrieval with LangChain
├── routes/
│   └── intelligence_api.py                # API endpoints for all tiers
└── ml_models/
    ├── classifiers/
    │   └── ability_classifier.py         # (Legacy - use command_router)
    ├── generators/
    │   └── ability_generator.py          # (Legacy - use contextual_ability_engine)
    └── rl_agent/
        └── ppo_agent.py                  # (Legacy - use workflow_optimizer)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd diri-cyrex
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# .env file
OPENAI_API_KEY=sk-...
INTENT_CLASSIFIER_MODEL_PATH=./models/intent_classifier  # Optional
PRODUCTIVITY_AGENT_MODEL_PATH=./models/productivity_agent  # Optional
CHROMA_PERSIST_DIR=./chroma_db
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3. Start Service

```bash
python -m app.main
# or
uvicorn app.main:app --reload --port 8000
```

### 4. Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Intent classification
curl -X POST http://localhost:8000/agent/ai/classify-intent \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"command": "Create a task to refactor auth.ts", "user_role": "software_engineer"}'
```

### 5. (Optional) Launch the Cyrex Interface

For a UI-based experience, run the Cyrex Interface shipped in `diri-cyrex/cyrex-interface`:

```bash
cd diri-cyrex/cyrex-interface
npm install
npm run dev -- --host 0.0.0.0 --port 5175
# open http://localhost:5175
```

The interface exposes chat + all intelligence endpoints and is also available in `docker-compose.dev.yml` as the `cyrex-interface` service.

---

## 🔧 Service Details

### DeepiriIntentClassifier

**Purpose**: Classify user commands into predefined abilities

**Model**: Fine-tuned BERT/DeBERTa (microsoft/deberta-v3-base)

**Features**:
- 50+ predefined abilities
- Role-based filtering
- Parameter extraction
- Confidence scoring

**Example**:
```python
from app.services.command_router import get_command_router

router = get_command_router()
result = router.route_command(
    "Can you review this code?",
    user_role="software_engineer",
    top_k=3
)
```

### DeepiriAbilityGenerator

**Purpose**: Generate unique, contextual abilities on-the-fly

**Model**: GPT-4 Turbo + LangChain RAG

**Features**:
- LangChain orchestration
- Multiple knowledge bases
- Structured output (Pydantic)
- Alternative suggestions

**Example**:
```python
from app.services.contextual_ability_engine import get_contextual_ability_engine

engine = get_contextual_ability_engine()
result = engine.generate_ability(
    user_id="user123",
    user_command="Refactor to TypeScript",
    user_profile={"role": "engineer", "momentum": 450, "level": 15}
)
```

### DeepiriProductivityAgent

**Purpose**: Learn optimal productivity strategies

**Model**: PPO (Proximal Policy Optimization)

**Features**:
- Actor-Critic architecture
- 128D state encoding
- 50 action registry
- Reward-based learning

**Example**:
```python
from app.services.workflow_optimizer import get_workflow_optimizer

optimizer = get_workflow_optimizer()
recommendation = optimizer.recommend_action(user_data)
```

### DeepiriRAGOrchestrator

**Purpose**: Unified knowledge retrieval

**Features**:
- Multiple knowledge bases
- LangChain vector stores
- Document compression
- Context formatting

**Knowledge Bases**:
- `user_patterns`: User behavior patterns
- `project_context`: Project-specific context
- `ability_templates`: Pre-defined templates
- `rules_knowledge`: Business rules
- `historical_abilities`: Generated abilities

---

## 📊 Integration Flow

```
User Command
    ↓
[Intent Classifier] → High Confidence? → Execute Predefined Ability
    ↓ Low Confidence
[Ability Generator] → RAG Retrieval → LLM Generation → Execute Custom Ability
    ↓
[Productivity Agent] → Recommend Next Action
    ↓
User Feedback → Reward → Agent Learning
```

---

## 🎓 Training

### Intent Classifier

1. Collect training data (user commands → abilities)
2. Fine-tune BERT/DeBERTa
3. Deploy model

### Productivity Agent

1. Collect user interaction data
2. Train PPO agent offline
3. Deploy for online learning

See `DEEPIRI_AI_SYSTEM.md` for detailed training instructions.

---

## 📝 API Documentation

Full API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

All endpoints are under `/agent/ai/*`

---

## 🔗 Integration with Gamification

The AI system integrates seamlessly with the gamification system:

- **Intent Classification** → Execute predefined abilities → Award momentum
- **Ability Generation** → Create custom abilities → Charge momentum
- **Productivity Agent** → Recommend actions → Optimize momentum growth

---

For complete documentation, see `DEEPIRI_AI_SYSTEM.md`

