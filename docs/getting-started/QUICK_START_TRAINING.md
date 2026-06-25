# Training Pipeline

> **Status: Pending Implementation**
>
> The training scripts are planned but not yet implemented. This directory is reserved for future training pipeline documentation.

## Planned Training Pipeline

Training scripts will be placed in `train/scripts/` (mapped to `/app/train` inside the Docker container). The planned pipeline includes:

### Phase 1: Intent Classifier Training
- Generate synthetic training data
- Fine-tune BERT/DeBERTa for intent classification
- Evaluate on validation set
- Deploy to `CommandRouter` service

### Phase 2: RL Agent Training
- Collect user interaction data
- Train PPO agent for workflow optimization
- Deploy to `WorkflowOptimizer` service

### Phase 3: Model Customization
- LoRA/QLoRA fine-tuning
- Domain-specific model adaptation
- Continuous improvement pipeline

## When Scripts Are Available

Once implemented, the training commands will be:

```bash
# Generate synthetic data
python train/scripts/generate_synthetic_data.py

# Run training pipeline
python train/scripts/run_training_pipeline.py
```

For local development, you can run Ollama and pull models:
```bash
ollama serve
ollama pull llama3:8b
```

## Current ML Models

Pre-trained models can be found in:
- `app/ml_models/classifiers/` — Intent classifiers (legacy)
- `app/ml_models/generators/` — Ability generators (legacy)
- `app/ml_models/rl_agent/` — RL agents (legacy)

The current production code uses services instead of these legacy model files:
- `app/services/command_router.py` — Intent classification
- `app/services/contextual_ability_engine.py` — Ability generation
- `app/services/workflow_optimizer.py` — Workflow optimization

## Related Documentation

- [LoRA Integration](../development/LORA_INTEGRATION.md)
- [LoRA/QLoRA System](../development/LORA_QLORA_SYSTEM.md)
- [Model Customization](../development/MODEL_CUSTOMIZATION.md)

