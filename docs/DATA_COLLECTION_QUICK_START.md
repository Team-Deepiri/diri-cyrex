# Data Collection Quick Start Guide

**Goal**: Start collecting training data immediately so you can train better models.

---

##  5-Minute Setup

### Step 1: Run the Quick Start Script

```bash
cd deepiri/diri-cyrex
python app/train/scripts/quick_start_data_collection.py
```

This will:
-  Check if data collection is set up
-  Generate initial synthetic data
-  Show you next steps

### Step 2: Verify It's Working

```bash
# Check if data was collected
python -c "
from app.train.pipelines.data_collection_pipeline import get_data_collector
import sqlite3
c = get_data_collector()
conn = sqlite3.connect(c.db_path)
print('Classifications:', conn.execute('SELECT COUNT(*) FROM task_classifications').fetchone()[0])
print('Interactions:', conn.execute('SELECT COUNT(*) FROM user_interactions').fetchone()[0])
"
```

You should see counts > 0 if it worked!

---

##  What Data You Need

### For Tier 1 (Intent Classifier)
- **What**: User commands -> Ability IDs
- **How Much**: 100+ examples per ability (50 abilities = 5,000+ examples)
- **Format**: `{"text": "Create a task", "label": "create_objective"}`

### For Tier 2 (Ability Generator)
- **What**: User commands -> Generated abilities
- **How Much**: 1,000+ input-output pairs
- **Format**: `{"input": "migrate to TypeScript", "output": "TypeScript Migration Assistant"}`

### For Tier 3 (RL Agent)
- **What**: State-action-reward sequences
- **How Much**: 10,000+ sequences
- **Format**: `{"state": {...}, "action": "activate_focus_boost", "reward": 10.0}`

---

##  Three Ways to Get Data

### Method 1: Synthetic Data (Start Today)

**Best for**: Getting started immediately, having baseline data

```bash
# Generate 1000 synthetic examples
python app/train/scripts/generate_synthetic_data.py
```

**Pros**: 
- Instant data
- Good for initial training
- No privacy concerns

**Cons**:
- Not as good as real data
- May not cover edge cases

### Method 2: Real User Data (Best Quality)

**Best for**: Production-quality models

1. **Instrument your API** (see `app/train/examples/integration_example.py`)
2. **Add feedback collection** (users label predictions)
3. **Let it run** (collect from real usage)

**Pros**:
- Real user patterns
- Covers edge cases
- High quality

**Cons**:
- Takes time to collect
- Needs user feedback
- Privacy considerations

### Method 3: Manual Labeling (Fill Gaps)

**Best for**: Labeling unlabeled data, fixing mistakes

```bash
# Label unlabeled predictions
python app/train/scripts/label_data.py
```

**Pros**:
- High quality labels
- Fixes prediction errors
- Curated dataset

**Cons**:
- Time consuming
- Manual work
- Doesn't scale

---

##  Integration Guide

### Add to Your API Endpoint

```python
from app.train.pipelines.data_collection_pipeline import get_data_collector

@router.post("/your-endpoint")
async def your_endpoint(request: YourRequest):
    collector = get_data_collector()
    
    # Your existing code...
    result = your_model.predict(request.input)
    
    # Add this - collect the prediction
    collector.collect_classification(
        task_text=request.input,
        description=None,
        prediction={
            'type': result.get('ability_id'),
            'complexity': 'medium',
            'estimated_duration': 30
        },
        actual=None,  # User will provide feedback
        feedback=None
    )
    
    return result
```

**See full examples**: `app/train/examples/integration_example.py`

---

##  Export Data for Training

### Export All Collected Data

```bash
python app/train/scripts/export_training_data.py
```

This creates:
- `app/train/data/exported/classification_training.jsonl`
- `app/train/data/exported/ability_generation_training.jsonl`
- `app/train/data/exported/rl_training.jsonl`

### Use in Training Scripts

```python
# In your training script
import json

with open('app/train/data/exported/classification_training.jsonl') as f:
    for line in f:
        data = json.loads(line)
        text = data['text']
        label = data['label']
        # Use for training...
```

---

##  Checklist

### Week 1: Get Started
- [ ] Run quick start script
- [ ] Generate 1000 synthetic examples
- [ ] Add data collection to 1 API endpoint
- [ ] Verify data is being saved

### Week 2: Expand Collection
- [ ] Add data collection to all AI endpoints
- [ ] Create feedback endpoint
- [ ] Collect 100+ real user commands
- [ ] Label 50+ examples manually

### Week 3: Scale Up
- [ ] Set up automatic collection middleware
- [ ] Collect 1000+ interactions
- [ ] Export data for first training run
- [ ] Monitor data quality

### Month 2: Production Ready
- [ ] Collect 10,000+ examples
- [ ] Have labeled dataset ready
- [ ] Set up automated exports
- [ ] Start training models

---

##  Next Steps

1. **Read the full guide**: `HOW_TO_COLLECT_TRAINING_DATA.md`
2. **See integration examples**: `app/train/examples/integration_example.py`
3. **Generate synthetic data**: `app/train/scripts/generate_synthetic_data.py`
4. **Export for training**: `app/train/scripts/export_training_data.py`

---

##  Pro Tips

1. **Start with synthetic data** - Get training running immediately
2. **Collect everything** - Better to have too much data than too little
3. **Get user feedback** - Labeled data is gold
4. **Export regularly** - Don't wait until you have "enough" data
5. **Monitor quality** - Bad data = bad models

---

## 🆘 Troubleshooting

### "No data being collected"
- Check that you're calling `collector.collect_*()` methods
- Verify database path is writable
- Check logs for errors

### "Database locked"
- Only one process should write to SQLite
- Use connection pooling or separate databases per process

### "Not enough data"
- Generate more synthetic data
- Add data collection to more endpoints
- Set up automatic collection middleware

---

##  Full Documentation

- **Complete Guide**: `HOW_TO_COLLECT_TRAINING_DATA.md`
- **Integration Examples**: `app/train/examples/integration_example.py`
- **Data Collection Pipeline**: `app/train/pipelines/data_collection_pipeline.py`

---

**Remember**: The best time to start collecting data was yesterday. The second best time is now! 

