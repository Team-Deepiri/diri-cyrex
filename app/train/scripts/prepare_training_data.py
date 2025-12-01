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
    """Map ability name to ID (0-49 for 50 abilities)"""
    # Default ability mapping - update with your actual abilities
    ability_map = {
        "summarize_text": 0,
        "create_objective": 1,
        "activate_focus_boost": 2,
        "activate_velocity_boost": 3,
        "generate_code_review": 4,
        "refactor_suggest": 5,
        "create_odyssey": 6,
        "schedule_break": 7,
        # Add all 50 abilities here...
    }
    
    # If ability_name is already a number, return it
    try:
        ability_id = int(ability_name)
        if 0 <= ability_id < 50:
            return ability_id
    except ValueError:
        pass
    
    # Try to find in map
    if ability_name in ability_map:
        return ability_map[ability_name]
    
    # Try lowercase
    if ability_name.lower() in ability_map:
        return ability_map[ability_name.lower()]
    
    # Default to 0 if not found
    print(f"⚠ Warning: Unknown ability '{ability_name}', mapping to 0")
    return 0

def prepare_classification_data():
    """Prepare Tier 1 classification data"""
    input_file = Path("app/train/data/exported/classification_training.jsonl")
    output_train = Path("app/train/data/classification_train.jsonl")
    output_val = Path("app/train/data/classification_val.jsonl")
    
    # Create directories
    output_train.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"⚠ {input_file} not found.")
        print("   Options:")
        print("   1. Export collected data:")
        print("      python -c \"from app.train.pipelines.data_collection_pipeline import get_data_collector; c = get_data_collector(); c.export_for_training('app/train/data/exported/classification_training.jsonl', 'classification')\"")
        print("   2. Generate synthetic data:")
        print("      python app/train/scripts/generate_synthetic_data.py")
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
                
                if not text or not label:
                    print(f"⚠ Skipping line {line_num}: missing text or label")
                    continue
                
                # Map ability to ID
                ability_id = map_ability_to_id(label)
                
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
    print(f"  python app/train/scripts/train_intent_classifier.py")
    
    return True

if __name__ == "__main__":
    success = prepare_classification_data()
    sys.exit(0 if success else 1)

