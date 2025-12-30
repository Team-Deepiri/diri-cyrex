
Architecture

### Tier 1: Command Routing (Maximum Reliability)
**Service**: `command_router.py`  
**Model**: Fine-tuned BERT/DeBERTa  
**Purpose**: Route user commands to predefined abilities

### Tier 2: Contextual Ability Generation (High Creativity)
**Service**: `contextual_ability_engine.py`  
**Model**: GPT-4/Claude + RAG (LangChain)  
**Purpose**: Generate unique, contextual abilities on-the-fly

### Tier 3: Workflow Optimization (Adaptive Learning)
**Service**: `workflow_optimizer.py`  
**Model**: PPO (Proximal Policy Optimization)  
**Purpose**: Learn optimal productivity strategies over time

### Knowledge Retrieval
**Service**: `knowledge_retrieval_engine.py`  
**Purpose**: Unified knowledge retrieval across multiple knowledge bases

---

## API Endpoints

Base URL: `/agent/intelligence`

### Tier 1: Intent Classification

#### `POST /intelligence/route-command`
Classify user command to predefined abilities.

**Request:**
```json
{
  "command": "Can you help me review this code for security issues?",
  "user_role": "software_engineer",
  "context": {
    "file_path": "auth.ts",
    "lines": 50
  },
  "min_confidence": 0.7,
  "top_k": 3
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "ability_id": "generate_code_review",
    "ability_name": "Generate Code Review",
    "category": "development",
    "confidence": 0.95,
    "parameters": {
      "file_path": "auth.ts",
      "focus_areas": ["security"],
      "review_type": "security"
    }
  }
}
```

### Tier 2: Ability Generation

#### `POST /intelligence/generate-ability`
Generate dynamic ability using LLM + RAG.

**Request:**
```json
{
  "user_id": "user123",
  "user_command": "I need to refactor this codebase to use TypeScript",
  "user_profile": {
    "role": "software_engineer",
    "momentum": 450,
    "level": 15,
    "active_boosts": ["focus"]
  },
  "project_context": {
    "language": "JavaScript",
    "files": 50,
    "estimated_size": "10k lines"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "ability_name": "TypeScript Migration Assistant",
    "description": "Converts JS/JSX files to TypeScript with type inference",
    "category": "automation",
    "steps": [
      "Analyze current JS files",
      "Generate TypeScript equivalents",
      "Add type annotations",
      "Fix type errors"
    ],
    "parameters": {
      "action": "migrate",
      "target": "codebase",
      "options": {
        "preserve_comments": true,
        "strict_mode": true
      }
    },
    "momentum_cost": 50,
    "estimated_duration": 120,
    "success_criteria": "All files converted with no type errors",
    "prerequisites": ["TypeScript installed", "Node.js >= 16"],
    "confidence": 0.87
  },
  "alternatives": [
    {
      "ability_name": "TypeScript Migration Assistant (Lite Version)",
      "momentum_cost": 30,
      "estimated_duration": 156
    }
  ]
}
```

### Tier 3: Productivity Agent

#### `POST /intelligence/recommend-action`
Get RL agent's recommended action.

**Request:**
```json
{
  "user_data": {
    "momentum": 450,
    "current_level": 15,
    "task_completion_rate": 0.85,
    "daily_streak": 7,
    "time_of_day": "afternoon",
    "work_intensity": 0.7,
    "stress_level": 0.3,
    "active_tasks": [1, 2, 3],
    "active_boosts": [],
    "recent_efficiency": 0.82
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "action_type": "boost",
    "ability_id": "activate_focus_boost",
    "category": "boost",
    "confidence": 0.82,
    "state_value": 0.75,
    "reasoning": "You're in high-efficiency mode. A focus boost now could maximize your productivity and help you complete 2-3 more tasks.",
    "expected_benefit": {
      "momentum_gain": 25,
      "time_saved": 30,
      "efficiency_boost": 0.15,
      "satisfaction_increase": 0.2
    }
  }
}
```

#### `POST /ai/agent/reward`
Record reward for RL training.

**Request:**
```json
{
  "outcome": {
    "task_completed": true,
    "efficiency": 0.92,
    "user_rating": 5,
    "time_saved": 15,
    "momentum_gained": 25,
    "user_frustrated": false,
    "ability_used": true
  }
}
```

### RAG Orchestration

#### `POST /ai/rag/index`
Index document in knowledge base.

**Request:**
```json
{
  "content": "User patterns show that focus boosts are most effective in the afternoon...",
  "metadata": {
    "user_id": "user123",
    "type": "user_pattern",
    "timestamp": "2024-01-15T10:00:00Z"
  },
  "knowledge_base": "user_patterns"
}
```

#### `POST /ai/rag/query`
Query knowledge bases.

**Request:**
```json
{
  "query": "What are effective focus boost strategies?",
  "knowledge_bases": ["user_patterns", "ability_templates"],
  "top_k": 5
}
```

---

## 🔧 Services

### DeepiriIntentClassifier

**Location**: `app/services/command_router.py`

**Features:**
- Fine-tuned BERT/DeBERTa model
- 50+ predefined abilities
- Role-based filtering
- Parameter extraction
- Confidence scoring

**Usage:**
```python
from app.services.command_router import get_command_router

router = get_command_router()
predictions = router.route_command(
    "Can you review this code?",
    user_role="software_engineer",
    top_k=3
)
```

### DeepiriAbilityGenerator

**Location**: `app/services/contextual_ability_engine.py`

**Features:**
- LangChain orchestration
- GPT-4/Claude integration
- RAG with multiple knowledge bases
- Structured output (Pydantic)
- Alternative generation

**Usage:**
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

**Location**: `app/services/workflow_optimizer.py`

**Features:**
- PPO reinforcement learning
- Actor-Critic architecture
- State encoding (128D)
- Action registry (50 actions)
- Reward computation
- Policy updates

**Usage:**
```python
from app.services.workflow_optimizer import get_workflow_optimizer

optimizer = get_workflow_optimizer()
recommendation = optimizer.recommend_action(user_data)
reward = optimizer.compute_reward(outcome)
optimizer.update(epochs=10)
```

### DeepiriRAGOrchestrator

**Location**: `app/services/knowledge_retrieval_engine.py`

**Features:**
- Multiple knowledge bases
- LangChain vector stores (Chroma/Milvus)
- Document compression
- Unified retrieval interface

**Knowledge Bases:**
- `user_patterns`: User behavior patterns
- `project_context`: Project-specific context
- `ability_templates`: Pre-defined ability templates
- `rules_knowledge`: Business rules and constraints
- `historical_abilities`: Previously generated abilities

**Usage:**
```python
from app.services.knowledge_retrieval_engine import get_knowledge_retrieval_engine

engine = get_knowledge_retrieval_engine()
engine.add_document(content, metadata, "user_patterns")
docs = engine.retrieve(query, ["user_patterns", "ability_templates"], top_k=5)
```

---

## 🚀 Integration with Gamification

### Automatic Ability Execution

When a user completes a task:
1. **Intent Classification** determines if task matches predefined ability
2. **Ability Generator** creates custom ability if needed
3. **Productivity Agent** recommends next optimal action
4. **RAG Orchestrator** retrieves relevant context

### Momentum Integration

- Classification: Low momentum cost (uses predefined abilities)
- Generation: Medium-high momentum cost (custom abilities)
- RL Agent: Optimizes for momentum growth

---

## 📊 Model Training

### Intent Classifier Training

```bash
# Collect training data
python train/scripts/collect_intent_data.py

# Fine-tune BERT
python train/scripts/train_intent_classifier.py \
    --model microsoft/deberta-v3-base \
    --data data/intent_classification.jsonl \
    --epochs 5
```

### Productivity Agent Training

```bash
# Collect user interaction data
python train/scripts/collect_rl_data.py

# Train PPO agent
python train/scripts/train_productivity_agent.py \
    --epochs 100 \
    --batch_size 64 \
    --gamma 0.99
```

---

## 🔒 Configuration

### Environment Variables

```bash
# OpenAI (for LLM generation)
OPENAI_API_KEY=sk-...

# Model Paths
INTENT_CLASSIFIER_MODEL_PATH=./models/intent_classifier
PRODUCTIVITY_AGENT_MODEL_PATH=./models/productivity_agent

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LangSmith (optional monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
```

---

## 📈 Performance Metrics

### Intent Classifier
- **Accuracy**: >90% on test set
- **Latency**: <100ms per request
- **Confidence Threshold**: 0.7 (configurable)

### Ability Generator
- **Relevance**: >85% of generated abilities used
- **Latency**: <3s per generation
- **Structured Output**: 100% valid JSON

### Productivity Agent
- **Recommendation Accuracy**: Measured by user acceptance rate
- **Reward Optimization**: Maximizes long-term productivity
- **Update Frequency**: Every 100 interactions

---

## 🎯 Use Cases

### Use Case 1: User says "Create a task to refactor auth.ts"
1. **Intent Classifier** → `create_objective` (confidence: 0.92)
2. **Parameters extracted**: title="refactor auth.ts", momentum_reward=10
3. **Execute**: Create objective via gamification API

### Use Case 2: User says "I want to migrate this codebase to TypeScript"
1. **Intent Classifier** → No high-confidence match
2. **Ability Generator** → Creates "TypeScript Migration Assistant"
3. **RAG** → Retrieves similar migration patterns
4. **Execute**: Custom ability with 50 momentum cost

### Use Case 3: User has high momentum, good efficiency
1. **Productivity Agent** → Recommends "activate_focus_boost"
2. **Reasoning**: "You're in high-efficiency mode. A focus boost now could maximize productivity."
3. **Expected Benefit**: +25 momentum, 30 min saved
4. **User accepts** → Reward recorded for agent learning

---

## 🔄 Workflow Integration

```
User Command
    ↓
Intent Classifier (Tier 1)
    ↓
[High Confidence?]
    ├─ Yes → Execute Predefined Ability
    └─ No → Ability Generator (Tier 2)
                ↓
            RAG Retrieval
                ↓
            LLM Generation
                ↓
            Execute Custom Ability
    ↓
Productivity Agent (Tier 3)
    ↓
Recommend Next Action
    ↓
User Feedback → Reward → Agent Learning
```

---

## 📝 Next Steps

1. **Collect Training Data**: User commands → abilities mapping
2. **Fine-tune Classifier**: Train on collected data
3. **Populate Knowledge Bases**: Index historical abilities, patterns
4. **Train RL Agent**: Collect user interaction data, train offline
5. **Deploy Online Learning**: Enable continual learning from user feedback

---

This three-tier system provides maximum reliability, creativity, and adaptive learning for Deepiri's gamification platform!

