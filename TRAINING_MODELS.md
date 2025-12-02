# 🚀 RUN TRAINING NOW - FINAL INSTRUCTIONS

## YOU ARE READY FOR LIFTOFF 🔥

Everything is set up. Here's how to launch:

---

## 🎯 OPTION 1: ONE COMMAND (RECOMMENDED)

### Open Terminal/PowerShell and run:

```bash
cd deepiri/diri-cyrex
python app/train/scripts/run_training_pipeline.py
```

**That's it.** Sit back and watch it:
1. Generate 5000 training examples ⚡
2. Prepare the data 📊
3. Train DeBERTa model 🧠
4. Evaluate performance 📈

**Time**: 15-30 minutes (GPU) or 1-2 hours (CPU)

---

## 🎯 OPTION 2: STEP BY STEP

### 1. Setup Environment (First Time Only)
```bash
python app/train/scripts/setup_training_env.py
```

### 2. Generate Data
```bash
python app/train/scripts/generate_synthetic_data.py
```

### 3. Train Model
```bash
python app/train/scripts/train_intent_classifier.py
```

### 4. Evaluate Model
```bash
python app/train/scripts/evaluate_trained_model.py
```

### 5. Test It Out!
```bash
python app/train/scripts/test_model_inference.py
```

---

## 📊 WHAT YOU'LL GET

### ✅ Trained Model
- Location: `app/train/models/intent_classifier`
- Size: ~500MB
- Accuracy: 85-95%

### ✅ Performance Report
- Location: `app/train/models/intent_classifier/evaluation_report.json`
- Includes: Accuracy, F1, Precision, Recall, Confusion Matrix

### ✅ 8 Task Categories
1. **Coding** - Programming tasks
2. **Writing** - Documents, emails, blogs
3. **Fitness** - Exercise, workouts
4. **Cleaning** - Organization, tidying
5. **Learning** - Study, courses, research
6. **Creative** - Design, art, content
7. **Administrative** - Scheduling, paperwork
8. **Social** - Friends, events, networking

---

## 🔥 ADVANCED OPTIONS

### More Training Data
```bash
python app/train/scripts/generate_synthetic_data.py --total-examples 10000
```

### Longer Training
```bash
python app/train/scripts/train_intent_classifier.py --epochs 5
```

### Smaller/Faster Model
```bash
python app/train/scripts/train_intent_classifier.py --model microsoft/deberta-v3-small
```

### PowerShell (Windows)
```powershell
.\app\train\scripts\run_training_pipeline.ps1 -Epochs 5 -TotalExamples 10000
```

---

## 💻 USE THE MODEL

After training, use it like this:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load
model = AutoModelForSequenceClassification.from_pretrained(
    "app/train/models/intent_classifier"
)
tokenizer = AutoTokenizer.from_pretrained(
    "app/train/models/intent_classifier"
)

# Predict
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits).item()
        conf = torch.softmax(outputs.logits, dim=-1).max().item()
    
    categories = ["coding", "writing", "fitness", "cleaning", 
                  "learning", "creative", "administrative", "social"]
    return {"category": categories[pred], "confidence": conf}

# Test
result = classify("Write unit tests for my API")
print(result)  # {"category": "coding", "confidence": 0.95}
```

---

## 📚 DOCUMENTATION

- **Main Guide**: `app/train/LIFTOFF.md`
- **Quick Start**: `app/train/TRAINING_QUICK_START.md`
- **Full Docs**: `app/train/README_TRAINING_PIPELINE.md`

---

## 🐛 TROUBLESHOOTING

### "Module not found"
```bash
pip install -r app/train/requirements.txt
```

### "CUDA out of memory"
```bash
python app/train/scripts/train_intent_classifier.py --batch-size 8
```

### Need help?
Check `app/train/LIFTOFF.md` for detailed troubleshooting

---

## 🚀 READY?

### RUN THIS NOW:

```bash
cd deepiri/diri-cyrex
python app/train/scripts/run_training_pipeline.py
```

## LIFTOFF IN 3... 2... 1... 🔥🚀

---

*After training completes, test your model with:*
```bash
python app/train/scripts/test_model_inference.py
```

**YOU'VE GOT THIS!** 💪

