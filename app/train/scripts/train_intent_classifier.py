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
        labels, predictions, average='weighted', zero_division=0
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
    train_path = Path(train_file)
    val_path = Path(val_file)
    
    if not train_path.exists():
        print(f"❌ Training file not found: {train_file}")
        print("   Run: python app/train/scripts/prepare_training_data.py")
        print("   Or generate synthetic data first")
        return
    
    if not val_path.exists():
        print(f"⚠ Validation file not found: {val_file}")
        print("   Using train file for validation (not ideal)")
        val_path = train_path
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset('json', data_files={
            'train': str(train_path),
            'validation': str(val_path)
        })
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("   Make sure your JSONL file has 'text' and 'label' fields")
        return
    
    print(f"✓ Train: {len(dataset['train'])} examples")
    print(f"✓ Val: {len(dataset['validation'])} examples")
    
    if len(dataset['train']) < 100:
        print("⚠ Warning: Very small dataset. Consider generating more data.")
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_abilities
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure transformers is installed: pip install transformers")
        return
    
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
        remove_columns=[col for col in dataset['train'].column_names if col != 'label']
    )
    
    # Training arguments
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=str(output_path / "logs"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=min(100, len(dataset['train']) // batch_size),
        save_strategy="steps",
        save_steps=min(100, len(dataset['train']) // batch_size),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard by default
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
    print(f"   Device: {training_args.device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print()
    
    try:
        trainer.train()
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate
    print("\n📊 Evaluating...")
    eval_results = trainer.evaluate()
    print(f"\nResults:")
    print(f"  Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  Validation F1: {eval_results['eval_f1']:.4f}")
    print(f"  Validation Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Validation Recall: {eval_results['eval_recall']:.4f}")
    
    # Save
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save ability mapping (placeholder - update with real mapping)
    ability_map = {
        i: f"ability_{i}" for i in range(num_abilities)
    }
    with open(f"{output_dir}/ability_map.json", 'w') as f:
        json.dump(ability_map, f, indent=2)
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "num_abilities": num_abilities,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_examples": len(dataset['train']),
        "val_examples": len(dataset['validation']),
        "eval_accuracy": float(eval_results['eval_accuracy']),
        "eval_f1": float(eval_results['eval_f1'])
    }
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Use this path in CommandRouter: {output_dir}")
    print(f"\n   To use in production:")
    print(f"   from app.services.command_router import get_command_router")
    print(f"   router = get_command_router(model_path='{output_dir}')")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Tier 1 Intent Classifier")
    parser.add_argument("--model", default="microsoft/deberta-v3-base",
                       help="Base model to fine-tune")
    parser.add_argument("--train-file", default="app/train/data/classification_train.jsonl",
                       help="Training data file")
    parser.add_argument("--val-file", default="app/train/data/classification_val.jsonl",
                       help="Validation data file")
    parser.add_argument("--output-dir", default="app/train/models/intent_classifier",
                       help="Output directory for model")
    parser.add_argument("--num-abilities", type=int, default=50,
                       help="Number of predefined abilities")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    
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

