# Training Quick Start - Run This Now

##  Immediate Steps (Copy & Paste)

### Option 1: All-in-One (Easiest)

```bash
cd deepiri/diri-cyrex
chmod +x app/train/scripts/quick_train.sh
./app/train/scripts/quick_train.sh
```

### Option 2: Step-by-Step

```bash
cd deepiri/diri-cyrex

# 1. Export collected data
python -c "
from app.train.pipelines.data_collection_pipeline import get_data_collector
from pathlib import Path
c = get_data_collector()
Path('app/train/data/exported').mkdir(parents=True, exist_ok=True)
c.export_for_training('app/train/data/exported/classification_training.jsonl', 'classification')
"

# 2. Prepare training data
python app/train/scripts/prepare_training_data.py

# 3. Train model
python app/train/scripts/train_intent_classifier.py --epochs 3 --batch-size 16
```

---

##  What You Need

### Minimum Requirements
-  Python 3.8+
-  PyTorch installed (`pip install torch transformers datasets scikit-learn`)
-  At least 100 training examples (or generate synthetic data)

### Recommended
-  GPU (CUDA) for faster training
-  5,000+ training examples for good accuracy
-  8GB+ RAM

---

##  What Gets Trained

**Tier 1: Intent Classifier**
- Input: User command (text)
- Output: Ability ID (0-49)
- Model: Fine-tuned DeBERTa
- Goal: >90% accuracy

---

##  Training Output

After training, you'll have:
```
app/train/models/intent_classifier/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── ability_map.json
└── training_info.json
```

---

##  Use Your Trained Model

```python
from app.services.command_router import get_command_router

# Use your trained model
router = get_command_router(
    model_path="app/train/models/intent_classifier"
)

# Test it
result = router.route(
    command="Create a task to refactor auth.ts",
    user_role="software_engineer"
)
print(result)  # Should return ability_id with high confidence
```

---

##  Quick Troubleshooting

### "No data found"
```bash
# Generate synthetic data
python app/train/scripts/generate_synthetic_data.py
```

### "CUDA out of memory"
```bash
# Use smaller batch size
python app/train/scripts/train_intent_classifier.py --batch-size 8
```

### "Low accuracy"
- Need more data (aim for 100+ examples per ability)
- Try more epochs: `--epochs 5`
- Check data quality

---

##  Full Documentation

- **Complete Guide**: `HOW_TO_START_TRAINING.md`
- **Data Collection**: `HOW_TO_COLLECT_TRAINING_DATA.md`
- **Integration**: `app/train/examples/deepiri_integration_example.py`

---

##  Success Checklist

- [ ] Data exported and prepared
- [ ] Training completed without errors
- [ ] Validation accuracy > 80% (target: >90%)
- [ ] Model saved to `app/train/models/intent_classifier`
- [ ] Tested inference with CommandRouter

---

**Start training now:**
```bash
cd deepiri/diri-cyrex && ./app/train/scripts/quick_train.sh
```

