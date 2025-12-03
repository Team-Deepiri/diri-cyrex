#!/usr/bin/env python3
"""
Generate synthetic training data for task classification
Creates data for 8 categories: coding, writing, fitness, cleaning, learning, creative, administrative, social
"""
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Label mapping (8 categories)
LABEL_MAPPING = {
    "coding": 0,
    "writing": 1,
    "fitness": 2,
    "cleaning": 3,
    "learning": 4,
    "creative": 5,
    "administrative": 6,
    "social": 7
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

# Task templates for each category
TASK_TEMPLATES = {
    "coding": [
        "Write unit tests for my authentication API endpoints",
        "Debug the payment processing API",
        "Refactor the user service module",
        "Implement a new feature for user authentication",
        "Fix the bug in the database connection pool",
        "Review and optimize the search algorithm",
        "Set up CI/CD pipeline for the new service",
        "Create API documentation for the endpoints",
        "Migrate the codebase to TypeScript",
        "Write integration tests for the checkout flow",
        "Optimize database queries for better performance",
        "Implement error handling for the API",
        "Create a new microservice for notifications",
        "Update dependencies and fix security vulnerabilities",
        "Design and implement a caching layer",
        "Write code to handle file uploads",
        "Create a REST API endpoint for user profiles",
        "Fix memory leaks in the application",
        "Implement authentication middleware",
        "Write a script to migrate data to new schema",
        "Create a Dockerfile for the application",
        "Implement rate limiting for API endpoints",
        "Write unit tests for the validation logic",
        "Refactor legacy code to use modern patterns",
        "Set up monitoring and logging infrastructure"
    ],
    "writing": [
        "Write a blog post about machine learning trends",
        "Draft an email to the team about the project update",
        "Create documentation for the new feature",
        "Write a technical report on the system architecture",
        "Compose a proposal for the client meeting",
        "Write a summary of the quarterly results",
        "Draft a press release for the product launch",
        "Create user guides for the application",
        "Write meeting notes from the sprint planning",
        "Compose a response to customer feedback",
        "Write a case study on the project success",
        "Draft a newsletter for the team",
        "Create content for the company website",
        "Write a research paper on the findings",
        "Compose a grant proposal for funding",
        "Write a tutorial on how to use the system",
        "Draft a contract for the new partnership",
        "Create marketing copy for the campaign",
        "Write a review of the latest technology",
        "Compose a letter to stakeholders",
        "Write documentation for API endpoints",
        "Draft a presentation script for the conference",
        "Create a style guide for the documentation",
        "Write a white paper on industry trends",
        "Compose a thank you note to the team"
    ],
    "fitness": [
        "Do 30 minutes of cardio and stretching",
        "Go for a 5K run in the park",
        "Complete a full body strength training workout",
        "Do yoga for flexibility and relaxation",
        "Go to the gym for weight training",
        "Take a long walk in nature",
        "Do a HIIT workout at home",
        "Swim laps at the pool",
        "Go cycling for an hour",
        "Do a morning stretching routine",
        "Complete a core strength workout",
        "Go hiking on the weekend",
        "Do a pilates session",
        "Play basketball with friends",
        "Go for a jog around the neighborhood",
        "Do a bodyweight exercise routine",
        "Take a dance fitness class",
        "Go rock climbing at the gym",
        "Do a meditation and yoga session",
        "Complete a 10K training run",
        "Do a circuit training workout",
        "Go for a bike ride",
        "Do a flexibility and mobility routine",
        "Play tennis with a partner",
        "Complete a crossfit workout"
    ],
    "cleaning": [
        "Clean and organize my desk",
        "Vacuum the entire house",
        "Do the laundry and fold clothes",
        "Clean the kitchen and wash dishes",
        "Organize the closet and donate old clothes",
        "Deep clean the bathroom",
        "Tidy up the living room",
        "Clean the windows and mirrors",
        "Organize files and documents",
        "Clean out the refrigerator",
        "Dust all the furniture",
        "Mop the floors",
        "Clean the garage and organize tools",
        "Wash the car",
        "Organize the pantry and kitchen cabinets",
        "Clean the oven and stove",
        "Tidy up the bedroom",
        "Clean the carpets",
        "Organize the home office",
        "Clean the backyard and patio",
        "Wash all the bed linens",
        "Organize the bookshelf",
        "Clean the air vents and filters",
        "Tidy up the entryway",
        "Deep clean the entire house"
    ],
    "learning": [
        "Read a research paper on transformers",
        "Study for the certification exam",
        "Take an online course on machine learning",
        "Practice coding problems on LeetCode",
        "Read a chapter from the technical book",
        "Watch tutorial videos on the new framework",
        "Attend a workshop on cloud architecture",
        "Study the documentation for the API",
        "Practice a new programming language",
        "Read articles about best practices",
        "Take notes from the conference talk",
        "Study for the upcoming presentation",
        "Learn about the new technology stack",
        "Practice solving algorithm problems",
        "Read the latest industry research",
        "Study the system design patterns",
        "Take a course on data structures",
        "Learn about containerization and Docker",
        "Study the company's codebase",
        "Practice with the new development tools",
        "Read documentation for the library",
        "Study for the technical interview",
        "Learn about microservices architecture",
        "Practice with the database queries",
        "Study the security best practices"
    ],
    "creative": [
        "Design a logo for the new project",
        "Write a short story for the contest",
        "Create a video montage of the trip",
        "Compose a piece of music",
        "Paint a landscape scene",
        "Design a website mockup",
        "Write a poem about nature",
        "Create a digital art piece",
        "Design a poster for the event",
        "Write a screenplay for a short film",
        "Sketch ideas for the new product",
        "Create a photo collage",
        "Design a user interface mockup",
        "Write a song with lyrics",
        "Create an animation for the project",
        "Design a brand identity package",
        "Write a creative blog post",
        "Create a video tutorial",
        "Design a mobile app interface",
        "Write a children's story",
        "Create a portfolio website",
        "Design a marketing campaign visual",
        "Write a script for a podcast",
        "Create a digital illustration",
        "Design a book cover"
    ],
    "administrative": [
        "Schedule a meeting with the team",
        "File taxes for the year",
        "Pay bills and update budget",
        "Respond to important emails",
        "Update the project timeline",
        "Organize calendar for next week",
        "Review and approve expense reports",
        "Update employee records",
        "Schedule interviews for candidates",
        "Prepare agenda for the meeting",
        "File paperwork for the project",
        "Update the company policies",
        "Review contracts and agreements",
        "Organize files and documents",
        "Schedule appointments for the week",
        "Update the budget spreadsheet",
        "Respond to client inquiries",
        "Prepare reports for management",
        "Update the project status",
        "Organize team meeting notes",
        "File insurance claims",
        "Update contact information",
        "Schedule training sessions",
        "Review and process invoices",
        "Update the employee handbook"
    ],
    "social": [
        "Call a friend to catch up",
        "Plan a dinner party for friends",
        "Write a thank you note to someone",
        "Organize a team building event",
        "Reach out to a colleague for coffee",
        "Plan a birthday celebration",
        "Send a congratulatory message",
        "Organize a group outing",
        "Call family members",
        "Plan a weekend trip with friends",
        "Write a recommendation letter",
        "Organize a networking event",
        "Send holiday cards to friends",
        "Plan a surprise party",
        "Reach out to an old friend",
        "Organize a study group",
        "Plan a team lunch",
        "Send thank you emails",
        "Organize a community event",
        "Plan a game night",
        "Reach out to mentors",
        "Organize a volunteer activity",
        "Plan a family gathering",
        "Send invitations for the event",
        "Organize a farewell party"
    ]
}

def generate_variations(base_text: str, category: str, num_variations: int = 3) -> List[str]:
    """Generate variations of a base task text"""
    variations = [base_text]  # Include original
    
    # Add some simple variations
    if num_variations > 1:
        # Add "I need to" prefix
        variations.append(f"I need to {base_text.lower()}")
    
    if num_variations > 2:
        # Add "Can you help me" prefix
        variations.append(f"Can you help me {base_text.lower()}")
    
    if num_variations > 3:
        # Add "Please" prefix
        variations.append(f"Please {base_text.lower()}")
    
    if num_variations > 4:
        # Add urgency
        variations.append(f"{base_text} - urgent")
    
    return variations[:num_variations]

def generate_synthetic_dataset(
    total_examples: int = 5000,
    examples_per_class: int = None,
    output_dir: str = "app/train/data"
) -> Dict:
    """
    Generate synthetic dataset for task classification
    
    Args:
        total_examples: Total number of examples to generate
        examples_per_class: If specified, generate this many per class (overrides total_examples)
        output_dir: Directory to save the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate examples per class
    if examples_per_class:
        total_examples = examples_per_class * len(LABEL_MAPPING)
    
    examples_per_class = total_examples // len(LABEL_MAPPING)
    remainder = total_examples % len(LABEL_MAPPING)
    
    print("=" * 60)
    print("Generating Synthetic Training Data")
    print("=" * 60)
    print(f"Total examples: {total_examples}")
    print(f"Examples per class: ~{examples_per_class}")
    print(f"Categories: {list(LABEL_MAPPING.keys())}")
    print()
    
    all_data = []
    label_counts = Counter()
    
    # Generate data for each category
    for category, label_id in LABEL_MAPPING.items():
        templates = TASK_TEMPLATES[category]
        num_examples = examples_per_class + (1 if label_id < remainder else 0)
        
        print(f"Generating {num_examples} examples for '{category}'...")
        
        # Calculate how many variations per template
        variations_per_template = max(1, num_examples // len(templates))
        extra_variations = num_examples % len(templates)
        
        category_data = []
        for i, template in enumerate(templates):
            num_variations = variations_per_template + (1 if i < extra_variations else 0)
            variations = generate_variations(template, category, num_variations)
            
            for variation in variations:
                if len(category_data) >= num_examples:
                    break
                
                task_id = f"task_{len(all_data) + len(category_data):06d}"
                
                example = {
                    "id": task_id,
                    "text": variation,
                    "label": category,
                    "label_id": label_id,
                    "metadata": {
                        "length": len(variation),
                        "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                        "source": "synthetic"
                    }
                }
                
                category_data.append(example)
                label_counts[category] += 1
        
        all_data.extend(category_data)
        print(f"  ✓ Generated {len(category_data)} examples")
    
    # Shuffle the data
    random.shuffle(all_data)
    
    # Split into train (70%), validation (15%), test (15%)
    total = len(all_data)
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Save datasets
    train_file = output_path / "classification_train.jsonl"
    val_file = output_path / "classification_val.jsonl"
    test_file = output_path / "classification_test.jsonl"
    
    print(f"\nSaving datasets...")
    print(f"  Train: {len(train_data)} examples -> {train_file}")
    print(f"  Val: {len(val_data)} examples -> {val_file}")
    print(f"  Test: {len(test_data)} examples -> {test_file}")
    
    # Save in format expected by training script (text, label as integer)
    for file_path, data in [(train_file, train_data), (val_file, val_data), (test_file, test_data)]:
        with open(file_path, 'w') as f:
            for item in data:
                # Save in format expected by trainer: {"text": "...", "label": 0}
                f.write(json.dumps({
                    "text": item["text"],
                    "label": item["label_id"]
                }) + '\n')
    
    # Also save full format with metadata
    full_train_file = output_path / "synthetic_classification_train.jsonl"
    full_val_file = output_path / "synthetic_classification_val.jsonl"
    full_test_file = output_path / "synthetic_classification_test.jsonl"
    
    for file_path, data in [(full_train_file, train_data), (full_val_file, val_data), (full_test_file, test_data)]:
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    # Save label mapping
    label_map_file = output_path / "label_mapping.json"
    with open(label_map_file, 'w') as f:
        json.dump({
            "label2id": LABEL_MAPPING,
            "id2label": ID_TO_LABEL
        }, f, indent=2)
    
    # Save metadata
    metadata = {
        "dataset_name": "deepiri-task-classification-v1",
        "version": "1.0",
        "total_samples": total,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "num_classes": len(LABEL_MAPPING),
        "label_distribution": dict(label_counts),
        "avg_text_length": sum(len(item["text"]) for item in all_data) / len(all_data),
        "min_text_length": min(len(item["text"]) for item in all_data),
        "max_text_length": max(len(item["text"]) for item in all_data)
    }
    
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {total}")
    print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    print(f"\nLabel Distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} examples")
    print(f"\nFiles saved:")
    print(f"  Training data: {train_file}")
    print(f"  Validation data: {val_file}")
    print(f"  Test data: {test_file}")
    print(f"  Label mapping: {label_map_file}")
    print(f"  Metadata: {metadata_file}")
    print(f"\nNext step: Run training")
    print(f"  python3 app/train/scripts/train_intent_classifier.py")
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "metadata": metadata
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
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
        "--output-dir",
        type=str,
        default="app/train/data",
        help="Output directory for datasets (default: app/train/data)"
    )
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        total_examples=args.total_examples,
        examples_per_class=args.examples_per_class,
        output_dir=args.output_dir
    )

