# How to Start Training - Immediate Steps

This guide shows you exactly how to start training models **right now**.

---

## Step 1: Prepare Your Data (5 minutes)

### Export Collected Data

```bash
cd deepiri/diri-cyrex

# Export all collected data
python -c "
from app.train.pipelines.data_collection_pipeline import get_data_collector
from pathlib import Path

collector = get_data_collector()
output_dir = Path('app/train/data/exported')
output_dir.mkdir(parents=True, exist_ok=True)

# Export classification data (Tier 1)
collector.export_for_training(
    str(output_dir / 'classification_training.jsonl'),
    'classification'
)

# Export ability generation data (Tier 2)
collector.export_for_training(
    str(output_dir / 'ability_generation_training.jsonl'),
    'ability_generation'
)

print('Data exported!')
"
```

### Generate Synthetic Data (If Needed)

If you don't have enough real data yet:

```bash
python app/train/scripts/generate_synthetic_data.py
```

---

## Step 2: Prepare Training Dataset (5 minutes)

Create a script to format your data for training:

```bash
# Create the data preparation script
cat > app/train/scripts/prepare_training_data.py << 'EOF'
#!/usr/bin/env python3
"""Prepare training data from exported collections"""
import json
from pathlib import Path
from collections import Counter

def prepare_classification_data():
    """Prepare Tier 1 classification data"""
    input_file = Path("app/train/data/exported/classification_training.jsonl")
    output_file = Path("app/train/data/classification_train.jsonl")
    
    if not input_file.exists():
        print(f"⚠ {input_file} not found. Generate synthetic data first.")
        return
    
    # Load and format data
    data = []
    with open(input_file) as f:
        for line in f:
            item = json.loads(line)
            # Map ability names to IDs (you'll need to create this mapping)
            ability_id = map_ability_to_id(item['label'])
            data.append({
                "text": item['text'],
                "label": ability_id
            })
    
    # Split train/val
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save
    with open(output_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    val_file = Path("app/train/data/classification_val.jsonl")
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Prepared {len(train_data)} train, {len(val_data)} val examples")
    print(f"  Saved to {output_file} and {val_file}")

def map_ability_to_id(ability_name: str) -> int:
    """Map ability name to ID (0-49 for 50 abilities)"""
    # Create your ability mapping here
    ability_map = {
        "summarize_text": 0,
        "create_objective": 1,
        "activate_focus_boost": 2,
        "activate_velocity_boost": 3,
        "generate_code_review": 4,
        "refactor_suggest": 5,
        "create_odyssey": 6,
        "schedule_break": 7,
        # Add all 50 abilities...
    }
    return ability_map.get(ability_name, 0)

if __name__ == "__main__":
    prepare_classification_data()
EOF

python app/train/scripts/prepare_training_data.py
```

---

## Step 3: Train Tier 1 - Intent Classifier (30 minutes)

Create a working training script:

```bash
# Create the training script
cat > app/train/scripts/train_intent_classifier.py << 'EOF'
#!/usr/bin/env python3
"""
Train Tier 1: Intent Classifier (50 predefined abilities)
Fine-tunes DeBERTa for maximum reliability
"""
import os
import sys
import json
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_intent_classifier(
    model_name: str = "microsoft/deberta-v3-base",
    train_file: str = "app/train/data/classification_train.jsonl",
    val_file: str = "app/train/data/classification_val.jsonl",
    output_dir: str = "app/train/models/intent_classifier",
    num_abilities: int = 50,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Train intent classifier"""
    
    print("=" * 60)
    print("Training Tier 1: Intent Classifier")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Abilities: {num_abilities}")
    print(f"Epochs: {num_epochs}")
    print()
    
    # Check if data exists
    if not Path(train_file).exists():
        print(f"❌ Training file not found: {train_file}")
        print("   Run: python app/train/scripts/prepare_training_data.py")
        return
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('json', data_files={
        'train': train_file,
        'validation': val_file
    })
    
    print(f"✓ Train: {len(dataset['train'])} examples")
    print(f"✓ Val: {len(dataset['validation'])} examples")
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_abilities
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=128
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Evaluate
    print("\n📊 Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation F1: {eval_results['eval_f1']:.4f}")
    
    # Save
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save ability mapping
    ability_map = {
        i: f"ability_{i}" for i in range(num_abilities)
    }
    with open(f"{output_dir}/ability_map.json", 'w') as f:
        json.dump(ability_map, f, indent=2)
    
    print("✓ Training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Use this path in CommandRouter: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--train-file", default="app/train/data/classification_train.jsonl")
    parser.add_argument("--val-file", default="app/train/data/classification_val.jsonl")
    parser.add_argument("--output-dir", default="app/train/models/intent_classifier")
    parser.add_argument("--num-abilities", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    
    args = parser.parse_args()
    
    train_intent_classifier(
        model_name=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_abilities=args.num_abilities,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
EOF

# Make executable
chmod +x app/train/scripts/train_intent_classifier.py

# Run training
python app/train/scripts/train_intent_classifier.py \
    --model microsoft/deberta-v3-base \
    --epochs 3 \
    --batch-size 16
```

---

## Step 4: Use Your Trained Model

After training, update your CommandRouter to use the fine-tuned model:

```python
# In your code
from app.services.command_router import get_command_router

router = get_command_router(
    model_path="app/train/models/intent_classifier"  # Use your trained model
)
```

---

## Quick Start: All-in-One Script

Create this script to do everything at once:

```bash
cat > app/train/scripts/quick_train.sh << 'EOF'
#!/bin/bash
# Quick training script - does everything

set -e

echo "🚀 Deepiri Training Quick Start"
echo "================================"
echo ""

# Step 1: Export data
echo "Step 1: Exporting collected data..."
python -c "
from app.train.pipelines.data_collection_pipeline import get_data_collector
from pathlib import Path

collector = get_data_collector()
output_dir = Path('app/train/data/exported')
output_dir.mkdir(parents=True, exist_ok=True)

collector.export_for_training(
    str(output_dir / 'classification_training.jsonl'),
    'classification'
)
print('✓ Data exported')
"

# Step 2: Prepare data
echo ""
echo "Step 2: Preparing training data..."
python app/train/scripts/prepare_training_data.py

# Step 3: Train
echo ""
echo "Step 3: Training intent classifier..."
python app/train/scripts/train_intent_classifier.py \
    --epochs 3 \
    --batch-size 16

echo ""
echo "✅ Training complete!"
echo "Model saved to: app/train/models/intent_classifier"
EOF

chmod +x app/train/scripts/quick_train.sh

# Run it
./app/train/scripts/quick_train.sh
```

---

## Training Checklist

### Before Training
- [ ] Data collection is active
- [ ] Have at least 100 examples per ability (5,000+ total)
- [ ] Data is exported and formatted
- [ ] GPU available (optional, but faster)

### During Training
- [ ] Monitor training logs
- [ ] Check validation accuracy (target: >90%)
- [ ] Watch for overfitting
- [ ] Save checkpoints regularly

### After Training
- [ ] Evaluate on test set
- [ ] Test inference speed (<100ms target)
- [ ] Deploy model to production
- [ ] Update CommandRouter to use new model

---

## Next Steps

### Tier 2: Ability Generation
Once Tier 1 is working, train Tier 2:
- Fine-tune LLM for ability generation
- Train with RAG context
- See: `app/train/scripts/train_challenge_generator.py`

### Tier 3: RL Agent
Train the productivity optimizer:
- Collect RL sequences
- Train PPO agent
- See: `app/train/scripts/train_policy_network.py`

---

## Troubleshooting

### "Not enough data"
- Generate synthetic data: `python app/train/scripts/generate_synthetic_data.py`
- Collect more real data from production
- Use data augmentation

### "CUDA out of memory"
- Reduce batch size: `--batch-size 8`
- Use gradient accumulation
- Use smaller model: `--model distilbert-base-uncased`

### "Low accuracy"
- Check data quality
- Increase training epochs
- Try different learning rate
- Add more data

---

## Full Training Pipeline

For production training, see:
- `app/train/pipelines/full_training_pipeline.py`
- `app/train/pipelines/ml_training_pipeline.py`

---

**Start training now with:**
```bash
cd deepiri/diri-cyrex
./app/train/scripts/quick_train.sh
```

