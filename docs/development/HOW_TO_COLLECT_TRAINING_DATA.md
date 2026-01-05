# How to Collect Training Data for DIRI-CYREX

This guide shows you how to collect training data for all three tiers of the AI system.

---

## Quick Start: Enable Data Collection

### Step 1: Initialize Data Collection in Your API

Add data collection to your API endpoints. Here's how to instrument the command router:

```python
# In app/routes/intelligence_api.py or similar

from app.train.pipelines.data_collection_pipeline import get_data_collector

# When processing a user command:
def route_command(user_command: str, user_id: str):
    collector = get_data_collector()
    
    # Get prediction from classifier
    prediction = command_router.route(user_command, user_role="software_engineer")
    
    # Collect the prediction (even if not yet labeled)
    collector.collect_classification(
        task_text=user_command,
        description=None,
        prediction={
            'type': prediction.get('ability_id'),
            'complexity': prediction.get('complexity', 'medium'),
            'estimated_duration': prediction.get('duration', 30)
        },
        actual=None,  # Will be filled in later with user feedback
        feedback=None
    )
    
    # Also collect interaction
    collector.collect_interaction(
        user_id=user_id,
        action_type="command_routing",
        context={"command": user_command, "prediction": prediction},
        model_used="deberta-v3-base",
        response_time_ms=prediction.get('latency_ms', 0),
        success=prediction.get('confidence', 0) > 0.7,
        feedback=None
    )
    
    return prediction
```

---

## Method 1: Collect from Production (Real User Data)

### Instrument All API Endpoints

Add data collection to these key endpoints:

#### 1. Command Routing Endpoint

```python
# app/routes/intelligence_api.py

from app.train.pipelines.data_collection_pipeline import get_data_collector
from app.services.command_router import get_command_router
import time

@router.post("/intelligence/route-command")
async def route_command(request: CommandRequest):
    collector = get_data_collector()
    router = get_command_router()
    start_time = time.time()
    
    try:
        # Get prediction
        result = router.route(
            command=request.command,
            user_role=request.user_role,
            context=request.context,
            min_confidence=request.min_confidence
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Collect data
        collector.collect_classification(
            task_text=request.command,
            description=request.context.get('description'),
            prediction={
                'type': result.get('ability_id'),
                'complexity': result.get('complexity', 'medium'),
                'estimated_duration': result.get('duration', 30)
            },
            actual=None,  # User will provide feedback
            feedback=None
        )
        
        collector.collect_interaction(
            user_id=request.user_id,
            action_type="intent_classification",
            context={
                "command": request.command,
                "user_role": request.user_role,
                "prediction": result
            },
            model_used="deberta-v3-base",
            response_time_ms=latency_ms,
            success=result.get('confidence', 0) > 0.7,
            feedback=None
        )
        
        return result
        
    except Exception as e:
        # Still collect error data
        collector.collect_interaction(
            user_id=request.user_id,
            action_type="intent_classification",
            context={"command": request.command, "error": str(e)},
            model_used="deberta-v3-base",
            response_time_ms=(time.time() - start_time) * 1000,
            success=False,
            feedback=None
        )
        raise
```

#### 2. Ability Generation Endpoint

```python
@router.post("/intelligence/generate-ability")
async def generate_ability(request: AbilityGenerationRequest):
    collector = get_data_collector()
    engine = get_contextual_ability_engine()
    start_time = time.time()
    
    try:
        result = engine.generate_ability(
            user_id=request.user_id,
            user_command=request.user_command,
            user_profile=request.user_profile,
            project_context=request.project_context
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Collect ability generation data
        if result.get('success'):
            ability = result.get('ability', {})
            collector.collect_challenge_generation(
                task_text=request.user_command,
                challenge={
                    "description": ability.get('description'),
                    "type": ability.get('category'),
                    "difficulty": ability.get('complexity', 'medium'),
                    "pointsReward": ability.get('momentum_cost', 0)
                },
                user_engagement=None,  # Will be tracked later
                completion_rate=None,
                performance_score=None
            )
        
        collector.collect_interaction(
            user_id=request.user_id,
            action_type="ability_generation",
            context={
                "command": request.user_command,
                "generated_ability": result.get('ability'),
                "user_profile": request.user_profile
            },
            model_used=engine.llm_model_name,
            response_time_ms=latency_ms,
            success=result.get('success', False),
            feedback=None
        )
        
        return result
        
    except Exception as e:
        collector.collect_interaction(
            user_id=request.user_id,
            action_type="ability_generation",
            context={"command": request.user_command, "error": str(e)},
            model_used=engine.llm_model_name,
            response_time_ms=(time.time() - start_time) * 1000,
            success=False,
            feedback=None
        )
        raise
```

#### 3. Add Feedback Collection Endpoint

```python
@router.post("/intelligence/feedback")
async def collect_feedback(request: FeedbackRequest):
    """
    Collect user feedback on predictions and abilities.
    This is critical for training!
    """
    collector = get_data_collector()
    
    if request.feedback_type == "classification":
        # Update existing classification with feedback
        collector.collect_classification(
            task_text=request.original_command,
            description=None,
            prediction=request.original_prediction,
            actual={
                'type': request.correct_ability_id,
                'complexity': request.actual_complexity,
                'estimated_duration': request.actual_duration
            },
            feedback=request.rating  # 1-5 scale
        )
    
    elif request.feedback_type == "ability_generation":
        # Update ability generation with engagement metrics
        collector.collect_challenge_generation(
            task_text=request.original_command,
            challenge=request.generated_ability,
            user_engagement=request.engagement_score,  # 0-1
            completion_rate=1.0 if request.ability_used else 0.0,
            performance_score=request.performance_score  # 0-1
        )
    
    # Always collect interaction feedback
    collector.collect_interaction(
        user_id=request.user_id,
        action_type=request.feedback_type,
        context={
            "original_command": request.original_command,
            "feedback": request.rating,
            "comments": request.comments
        },
        model_used=request.model_used,
        response_time_ms=0,  # Not applicable for feedback
        success=request.rating >= 3,  # 3+ is success
        feedback=request.rating
    )
    
    return {"success": True, "message": "Feedback collected"}
```

---

## Method 2: Generate Synthetic Data (Quick Start)

If you don't have real user data yet, generate synthetic data to get started:

### Create Synthetic Data Generator

```python
# app/train/scripts/generate_synthetic_data.py

import json
import random
from pathlib import Path

# Predefined abilities (from command_router.py)
ABILITIES = [
    {"id": "summarize_text", "name": "Summarize Text", "category": "productivity"},
    {"id": "create_objective", "name": "Create Objective", "category": "gamification"},
    {"id": "activate_focus_boost", "name": "Activate Focus Boost", "category": "boost"},
    {"id": "generate_code_review", "name": "Generate Code Review", "category": "development"},
    {"id": "refactor_suggest", "name": "Refactor Suggestion", "category": "development"},
    {"id": "create_odyssey", "name": "Create Odyssey", "category": "gamification"},
    {"id": "schedule_break", "name": "Schedule Break", "category": "wellness"},
    # Add all 50 abilities...
]

# Example commands for each ability
COMMAND_TEMPLATES = {
    "summarize_text": [
        "Can you summarize this document?",
        "Give me a summary of this text",
        "What are the key points here?",
        "Make this shorter",
        "Summarize the main ideas"
    ],
    "create_objective": [
        "Create a task to refactor auth.ts",
        "I need to complete the login feature",
        "Add an objective for testing",
        "Create a goal to improve performance",
        "Set up a task for documentation"
    ],
    "activate_focus_boost": [
        "I need to focus",
        "Activate focus mode",
        "Help me concentrate",
        "Boost my focus",
        "I want to be more productive"
    ],
    "generate_code_review": [
        "Review this code for security issues",
        "Check my code for bugs",
        "Can you review this pull request?",
        "Analyze this code",
        "Find issues in this code"
    ],
    # Add templates for all abilities...
}

def generate_synthetic_classification_data(num_examples: int = 1000):
    """Generate synthetic command classification data"""
    data = []
    
    for ability in ABILITIES:
        ability_id = ability["id"]
        templates = COMMAND_TEMPLATES.get(ability_id, [])
        
        if not templates:
            continue
        
        # Generate multiple examples per ability
        examples_per_ability = num_examples // len(ABILITIES)
        
        for i in range(examples_per_ability):
            # Pick a random template
            base_command = random.choice(templates)
            
            # Add variations
            variations = [
                base_command,
                base_command.lower(),
                base_command.upper(),
                base_command + " please",
                "Please " + base_command.lower(),
                "I want to " + base_command.lower(),
                "Can you " + base_command.lower() + "?",
            ]
            
            command = random.choice(variations)
            
            # Add some noise/typos occasionally
            if random.random() < 0.1:  # 10% have typos
                command = command.replace("the", "teh").replace("you", "u")
            
            data.append({
                "text": command,
                "label": ability_id,
                "metadata": {
                    "category": ability["category"],
                    "complexity": random.choice(["low", "medium", "high"]),
                    "confidence": random.uniform(0.7, 1.0)
                }
            })
    
    return data

def generate_synthetic_ability_generation_data(num_examples: int = 500):
    """Generate synthetic ability generation examples"""
    data = []
    
    complex_commands = [
        "I want to migrate this codebase to TypeScript",
        "Help me build a REST API for user management",
        "Create a system to track employee productivity",
        "I need to refactor this entire module",
        "Build a dashboard for analytics",
        # Add more complex commands...
    ]
    
    for command in complex_commands * (num_examples // len(complex_commands)):
        # Generate a plausible ability description
        ability_description = f"Custom ability for: {command}"
        
        data.append({
            "input": command,
            "output": ability_description,
            "metadata": {
                "type": "custom",
                "difficulty": random.choice(["medium", "high"]),
                "engagement": random.uniform(0.6, 1.0),
                "completion_rate": random.uniform(0.5, 1.0),
                "performance": random.uniform(0.6, 1.0)
            }
        })
    
    return data

if __name__ == "__main__":
    # Generate classification data
    classification_data = generate_synthetic_classification_data(1000)
    
    output_path = Path("app/train/data/synthetic_classification.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in classification_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Generated {len(classification_data)} classification examples")
    print(f"Saved to {output_path}")
    
    # Generate ability generation data
    ability_data = generate_synthetic_ability_generation_data(500)
    
    output_path = Path("app/train/data/synthetic_ability_generation.jsonl")
    with open(output_path, 'w') as f:
        for item in ability_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Generated {len(ability_data)} ability generation examples")
    print(f"Saved to {output_path}")
```

Run it:
```bash
cd deepiri/diri-cyrex
python app/train/scripts/generate_synthetic_data.py
```

---

## Method 3: Manual Data Collection (Labeling)

### Create a Labeling Interface

```python
# app/train/scripts/label_data.py

import json
from pathlib import Path
from app.train.pipelines.data_collection_pipeline import get_data_collector

def label_unlabeled_data():
    """Interactive script to label unlabeled data"""
    collector = get_data_collector()
    
    # Get unlabeled classifications
    conn = sqlite3.connect(collector.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, task_text, predicted_type, predicted_complexity
        FROM task_classifications
        WHERE actual_type IS NULL
        LIMIT 100
    """)
    
    unlabeled = cursor.fetchall()
    
    print(f"Found {len(unlabeled)} unlabeled examples")
    
    for row in unlabeled:
        row_id, task_text, predicted_type, predicted_complexity = row
        
        print(f"\nCommand: {task_text}")
        print(f"Predicted: {predicted_type} ({predicted_complexity})")
        
        # Show available abilities
        print("\nAvailable abilities:")
        for i, ability in enumerate(ABILITIES[:10], 1):
            print(f"{i}. {ability['name']} ({ability['id']})")
        
        # Get user input
        correct_id = input("\nEnter correct ability ID (or 'skip'): ").strip()
        
        if correct_id.lower() == 'skip':
            continue
        
        # Update with correct label
        cursor.execute("""
            UPDATE task_classifications
            SET actual_type = ?, actual_complexity = ?
            WHERE id = ?
        """, (correct_id, predicted_complexity, row_id))
        
        conn.commit()
        print(f"Labeled as: {correct_id}\n")
    
    conn.close()
    print("Labeling complete!")

if __name__ == "__main__":
    label_unlabeled_data()
```

---

## Method 4: Export Data for Training

### Export Collected Data

```python
# app/train/scripts/export_training_data.py

from app.train.pipelines.data_collection_pipeline import get_data_collector
from pathlib import Path

def export_all_data():
    """Export all collected data for training"""
    collector = get_data_collector()
    
    # Create output directory
    output_dir = Path("app/train/data/exported")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export classification data
    classification_path = output_dir / "classification_training.jsonl"
    collector.export_for_training(str(classification_path), "classification")
    print(f"Exported classification data to {classification_path}")
    
    # Export ability generation data
    ability_path = output_dir / "ability_generation_training.jsonl"
    collector.export_for_training(str(ability_path), "challenge")
    print(f"Exported ability generation data to {ability_path}")
    
    # Export interaction data (for RL training)
    import sqlite3
    conn = sqlite3.connect(collector.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT user_id, action_type, context, model_used, 
               response_time_ms, success, feedback
        FROM user_interactions
        WHERE feedback IS NOT NULL OR success = 1
    """)
    
    interactions = cursor.fetchall()
    
    interaction_path = output_dir / "rl_training.jsonl"
    with open(interaction_path, 'w') as f:
        for row in interactions:
            data = {
                "user_id": row[0],
                "action_type": row[1],
                "context": json.loads(row[2]),
                "model_used": row[3],
                "response_time_ms": row[4],
                "success": bool(row[5]),
                "reward": row[6] if row[6] else (1.0 if row[5] else -1.0)
            }
            f.write(json.dumps(data) + '\n')
    
    conn.close()
    print(f"Exported {len(interactions)} interactions to {interaction_path}")

if __name__ == "__main__":
    export_all_data()
```

---

## Method 5: Continuous Data Collection

### Set Up Automatic Collection

Add middleware to automatically collect all API requests:

```python
# app/middleware/data_collection_middleware.py

from fastapi import Request
from app.train.pipelines.data_collection_pipeline import get_data_collector
import time
import json

async def data_collection_middleware(request: Request, call_next):
    """Middleware to automatically collect API request data"""
    collector = get_data_collector()
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Collect data
    latency_ms = (time.time() - start_time) * 1000
    
    # Only collect for AI endpoints
    if "/intelligence/" in str(request.url) or "/agent/" in str(request.url):
        try:
            body = await request.body()
            body_data = json.loads(body) if body else {}
            
            collector.collect_interaction(
                user_id=body_data.get('user_id', 'anonymous'),
                action_type=request.url.path,
                context={
                    "method": request.method,
                    "path": request.url.path,
                    "body": body_data,
                    "status_code": response.status_code
                },
                model_used="unknown",
                response_time_ms=latency_ms,
                success=response.status_code < 400,
                feedback=None
            )
        except Exception as e:
            # Don't break the request if collection fails
            pass
    
    return response
```

Add to your main app:

```python
# app/main.py

from app.middleware.data_collection_middleware import data_collection_middleware

app = FastAPI()

@app.middleware("http")
async def add_data_collection(request: Request, call_next):
    return await data_collection_middleware(request, call_next)
```

---

## Data Collection Checklist

### Immediate Actions (Week 1)

- [ ] Add data collection to command routing endpoint
- [ ] Add data collection to ability generation endpoint
- [ ] Create feedback collection endpoint
- [ ] Generate 1000 synthetic classification examples
- [ ] Generate 500 synthetic ability generation examples
- [ ] Set up automatic collection middleware

### Short Term (Month 1)

- [ ] Collect 5000+ real user commands
- [ ] Label at least 1000 commands manually
- [ ] Collect user feedback on 500+ predictions
- [ ] Track ability usage and outcomes
- [ ] Export data weekly for training

### Medium Term (Month 2-3)

- [ ] Collect 50,000+ interactions
- [ ] Have 10,000+ labeled examples
- [ ] Track ability generation success rates
- [ ] Collect RL training sequences (state-action-reward)
- [ ] Set up automated data quality checks

---

## Data Quality Guidelines

### Classification Data
- **Minimum**: 100 examples per ability (50 abilities = 5,000 examples)
- **Ideal**: 500+ examples per ability
- **Balance**: Ensure examples are balanced across abilities
- **Quality**: Only use examples with actual labels or high-confidence feedback

### Ability Generation Data
- **Minimum**: 1,000 input-output pairs
- **Ideal**: 10,000+ pairs
- **Quality**: Track engagement and completion rates
- **Diversity**: Cover different user roles, contexts, and complexity levels

### RL Training Data
- **Minimum**: 10,000 state-action-reward sequences
- **Ideal**: 100,000+ sequences
- **Quality**: Sequences should be from real user interactions
- **Coverage**: Cover different user states and scenarios

---

## Quick Commands

```bash
# Generate synthetic data
python app/train/scripts/generate_synthetic_data.py

# Label unlabeled data
python app/train/scripts/label_data.py

# Export all collected data
python app/train/scripts/export_training_data.py

# View data collection stats
sqlite3 app/train/data/collection.db "SELECT COUNT(*) FROM task_classifications;"
sqlite3 app/train/data/collection.db "SELECT COUNT(*) FROM user_interactions;"
```

---

## Next Steps

1. **Start collecting immediately**: Add data collection to your API endpoints
2. **Generate synthetic data**: Use synthetic data to start training while collecting real data
3. **Set up feedback loops**: Make it easy for users to provide feedback
4. **Automate collection**: Use middleware to collect everything automatically
5. **Export regularly**: Export data weekly for training

The more data you collect, the better your models will be!

