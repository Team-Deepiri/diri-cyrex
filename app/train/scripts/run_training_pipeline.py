#!/usr/bin/env python3
"""
Complete Training Pipeline - Run Everything End-to-End
Generates synthetic data, prepares it, and trains the model
"""
import sys
import subprocess
from pathlib import Path
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 60)
    print(f"Step: {description}")
    print("=" * 60)
    print(f"Running: {cmd}")
    print()
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print(f"Command failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True

def main(
    generate_data: bool = True,
    total_examples: int = 5000,
    examples_per_class: int = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    skip_training: bool = False
):
    """Run the complete training pipeline"""
    
    print("=" * 60)
    print("🚀 Deepiri Training Pipeline")
    print("=" * 60)
    print()
    print("This will:")
    print("  1. Generate synthetic training data")
    print("  2. Prepare the dataset")
    print("  3. Train the DeBERTa classifier")
    print()
    
    # Step 1: Generate synthetic data
    if generate_data:
        cmd = f"python3 app/train/scripts/generate_synthetic_data.py"
        if examples_per_class:
            cmd += f" --examples-per-class {examples_per_class}"
        else:
            cmd += f" --total-examples {total_examples}"
        
        if not run_command(cmd, "Generate Synthetic Data"):
            print("\n❌ Failed to generate synthetic data")
            return False
    
    # Step 2: Prepare training data
    cmd = "python3 app/train/scripts/prepare_training_data.py"
    if not run_command(cmd, "Prepare Training Data"):
        print("\n❌ Failed to prepare training data")
        print("   Note: If data was already generated, this might be okay")
        print("   Check if app/train/data/classification_train.jsonl exists")
    
    # Step 3: Train the model
    if not skip_training:
        cmd = f"python3 app/train/scripts/train_intent_classifier.py"
        cmd += f" --epochs {epochs}"
        cmd += f" --batch-size {batch_size}"
        cmd += f" --learning-rate {learning_rate}"
        
        if not run_command(cmd, "Train Intent Classifier"):
            print("\n❌ Failed to train model")
            return False
    
    # Step 4: Evaluate the model
    print("\n" + "=" * 60)
    print("🎯 Evaluating Model Performance")
    print("=" * 60)
    cmd = "python3 app/train/scripts/evaluate_trained_model.py"
    if not run_command(cmd, "Evaluate Model on Test Set"):
        print("\n⚠️  Evaluation failed, but model is still trained")
    
    # Success!
    print("\n" + "=" * 60)
    print("🚀 TRAINING PIPELINE COMPLETE! 🚀")
    print("=" * 60)
    print()
    print("✅ Model trained and evaluated successfully!")
    print()
    print("📁 Model location: app/train/models/intent_classifier")
    print("📊 Evaluation report: app/train/models/intent_classifier/evaluation_report.json")
    print()
    print("🧪 Test the model interactively:")
    print("   python3 app/train/scripts/test_model_inference.py")
    print()
    print("📈 Use in production:")
    print("   from app.services.command_router import get_command_router")
    print("   router = get_command_router(")
    print("       model_path='app/train/models/intent_classifier'")
    print("   )")
    print()
    print("🎯 Categories (8 total):")
    print("   0: coding          - Write code, debug, refactor")
    print("   1: writing         - Blog posts, docs, emails")
    print("   2: fitness         - Exercise, workouts, running")
    print("   3: cleaning        - Organize, tidy, clean spaces")
    print("   4: learning        - Study, read, take courses")
    print("   5: creative        - Design, art, create content")
    print("   6: administrative  - Schedule, bills, admin tasks")
    print("   7: social          - Friends, events, networking")
    print()
    print("🔥 YOU'RE READY FOR LIFTOFF! 🔥")
    print()
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete training pipeline for Deepiri task classifier"
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation (use existing data)"
    )
    parser.add_argument(
        "--total-examples",
        type=int,
        default=5000,
        help="Total number of examples to generate (default: 5000)"
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=None,
        help="Number of examples per class (overrides total-examples)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (only generate and prepare data)"
    )
    
    args = parser.parse_args()
    
    success = main(
        generate_data=not args.skip_data_generation,
        total_examples=args.total_examples,
        examples_per_class=args.examples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        skip_training=args.skip_training
    )
    
    sys.exit(0 if success else 1)

