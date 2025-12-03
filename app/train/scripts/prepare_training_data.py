#!/usr/bin/env python3
"""Prepare training data from exported collections"""
import json
import sys
from pathlib import Path
from collections import Counter
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def map_ability_to_id(ability_name: str) -> int:
    """Map ability name to ID (0-7 for 8 task categories)"""
    # Task category mapping (8 categories)
    category_map = {
        "coding": 0,
        "writing": 1,
        "fitness": 2,
        "cleaning": 3,
        "learning": 4,
        "creative": 5,
        "administrative": 6,
        "social": 7
    }
    
    # If ability_name is already a number, return it
    try:
        ability_id = int(ability_name)
        if 0 <= ability_id < 8:
            return ability_id
    except ValueError:
        pass
    
    # Try to find in map
    if ability_name in category_map:
        return category_map[ability_name]
    
    # Try lowercase
    if ability_name.lower() in category_map:
        return category_map[ability_name.lower()]
    
    # Default to 0 if not found
    print(f"⚠ Warning: Unknown category '{ability_name}', mapping to 0 (coding)")
    return 0

def prepare_classification_data():
    """Prepare Tier 1 classification data"""
    # Check multiple possible input locations
    possible_inputs = [
        Path("app/train/data/classification_train.jsonl"),  # Already prepared
        Path("app/train/data/exported/classification_training.jsonl"),  # Exported data
        Path("app/train/data/synthetic_classification_train.jsonl"),  # Synthetic full format
    ]
    
    input_file = None
    for path in possible_inputs:
        if path.exists():
            input_file = path
            break
    
    output_train = Path("app/train/data/classification_train.jsonl")
    output_val = Path("app/train/data/classification_val.jsonl")
    
    # Create directories
    output_train.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_file:
        print(f"⚠ No training data found. Checking for existing prepared data...")
        # Check if data is already prepared
        if output_train.exists() and output_val.exists():
            print(f"✓ Training data already prepared!")
            print(f"  Train: {output_train}")
            print(f"  Val: {output_val}")
            return True
        
        print("   Options:")
        print("   1. Generate synthetic data:")
        print("      python3 app/train/scripts/generate_synthetic_data.py")
        print("   2. Export collected data:")
        print("      python -c \"from app.train.pipelines.data_collection_pipeline import get_data_collector; c = get_data_collector(); c.export_for_training('app/train/data/exported/classification_training.jsonl', 'classification')\"")
        return False
    
    # Load and format data
    data = []
    label_counts = Counter()
    
    print(f"Loading data from {input_file}...")
    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                
                # Extract text and label
                text = item.get('text', '')
                label = item.get('label', '')
                label_id = item.get('label_id', None)
                
                if not text:
                    print(f"⚠ Skipping line {line_num}: missing text")
                    continue
                
                # Handle label - could be string category or integer ID
                if label_id is not None:
                    # Already has label_id
                    ability_id = int(label_id)
                elif isinstance(label, int):
                    # Label is already an integer
                    ability_id = label
                elif isinstance(label, str):
                    # Label is a string category, map it
                    ability_id = map_ability_to_id(label)
                else:
                    print(f"⚠ Skipping line {line_num}: invalid label format")
                    continue
                
                # Validate label_id is in range [0, 7]
                if not (0 <= ability_id < 8):
                    print(f"⚠ Skipping line {line_num}: label_id {ability_id} out of range [0, 7]")
                    continue
                
                data.append({
                    "text": text,
                    "label": ability_id
                })
                label_counts[ability_id] += 1
                
            except json.JSONDecodeError as e:
                print(f"⚠ Skipping line {line_num}: JSON error - {e}")
                continue
            except Exception as e:
                print(f"⚠ Skipping line {line_num}: {e}")
                continue
    
    if not data:
        print("❌ No valid data found!")
        return False
    
    print(f"✓ Loaded {len(data)} examples")
    print(f"  Label distribution: {dict(label_counts.most_common(10))}")
    
    # Shuffle
    random.shuffle(data)
    
    # Split train/val (80/20)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save train
    print(f"\nSaving training data ({len(train_data)} examples)...")
    with open(output_train, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Save validation
    print(f"Saving validation data ({len(val_data)} examples)...")
    with open(output_val, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Data prepared!")
    print(f"  Train: {output_train}")
    print(f"  Val: {output_val}")
    print(f"\n  Next step: Run training")
    print(f"  python3 app/train/scripts/train_intent_classifier.py")
    
    return True

if __name__ == "__main__":
    success = prepare_classification_data()
    sys.exit(0 if success else 1)

