# Quick Start: Training Pipeline

## 🚀 Quick Start

```bash
# 1. Generate synthetic data (7000 examples by default)
python app/train/scripts/generate_synthetic_data.py

# 2. Prepare training data
python app/train/scripts/prepare_training_data.py

# 3. Inspect datasets (recommended)
python app/train/scripts/inspect_datasets.py --all --quality

# 4. Run complete training pipeline
python app/train/scripts/run_training_pipeline.py
```

## 📋 What Changed

- **Dataset size**: 5,000 → **7,000** examples (+2,000)
- **Epochs**: 3 → **2**
- **Learning rate**: 2e-5 → **1e-5**
- **Ollama integration**: Enabled for better data augmentation
- **Confidence classes**: Redesigned with multiple attributes
- **Dataset inspection**: New tool for quality checks

## 🔧 Configuration

### With Ollama (Recommended)
```bash
# Ensure Ollama is running
ollama serve

# Pull model (if not already done)
ollama pull llama3:8b

# Run pipeline (Ollama enabled by default)
python app/train/scripts/run_training_pipeline.py
```

### Without Ollama
```bash
python app/train/scripts/run_training_pipeline.py --no-ollama
```

## 📊 Dataset Inspection

```bash
# Inspect all datasets
python app/train/scripts/inspect_datasets.py --all

# Quality check
python app/train/scripts/inspect_datasets.py --all --quality

# Compare datasets
python app/train/scripts/inspect_datasets.py --compare classification_train.jsonl classification_val.jsonl
```

## ⚠️ Important Reminders

1. **Prepare your data** before training
2. **Inspect datasets** to verify quality
3. **Check label distribution** is balanced
4. **Review confidence scores** in production

## 📁 Output Locations

- **Model**: `app/train/models/intent_classifier/`
- **Training data**: `app/train/data/classification_train.jsonl`
- **Validation data**: `app/train/data/classification_val.jsonl`
- **Test data**: `app/train/data/classification_test.jsonl`

## 🎯 Next Steps

After training:
1. Test model: `python app/train/scripts/test_model_inference.py`
2. Review evaluation report
3. Deploy using `CommandRouter` with confidence classes

