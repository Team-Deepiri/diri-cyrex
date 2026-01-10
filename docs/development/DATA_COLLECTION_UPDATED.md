# Data Collection System - Updated for Deepiri Platform

## What Changed

The data collection system has been **completely updated** to support Deepiri's full platform:

### New Data Types Collected

1. **Prompt-to-Tasks Engine** (Main Differentiator)
   - Every prompt -> tasks conversion
   - Execution plans, estimates, subtasks
   - User acceptance and completion tracking

2. **Tier 1: Intent Classification** (Enhanced)
   - Now includes `user_role` for role-based classification
   - Tracks confidence scores
   - 50 predefined abilities

3. **Tier 2: Role-based Ability Generation** (New)
   - Dynamic ability generation per role
   - RAG context tracking
   - Ability usage and engagement metrics

4. **Tier 3: RL Productivity Optimization** (New)
   - State-action-reward sequences
   - Productivity recommendations
   - Reward signal tracking

5. **Gamification System** (New)
   - Objectives (tasks with momentum)
   - Odysseys (project workflows)
   - Seasons (sprint cycles)
   - Momentum events (XP/levels)
   - Streak events (consistency)
   - Boost usage (power-ups)

## New Database Tables

The system now has **12 specialized tables**:

1. `task_classifications` - Tier 1 intent classification
2. `ability_generations` - Tier 2 role-based generation
3. `prompt_to_tasks` - Main differentiator engine
4. `rl_training_sequences` - Tier 3 RL training data
5. `productivity_recommendations` - Tier 3 recommendations
6. `objective_data` - Gamification objectives
7. `odyssey_data` - Gamification odysseys
8. `season_data` - Gamification seasons
9. `momentum_events` - Momentum/XP tracking
10. `streak_events` - Streak tracking
11. `boost_usage` - Boost effectiveness
12. `user_interactions` - General interactions

## New Collection Methods

### Prompt-to-Tasks Engine
```python
collector.collect_prompt_to_tasks(
    user_id="user123",
    user_role="software_engineer",
    prompt="Build login system",
    generated_tasks=[...],
    execution_plan={...},
    project_type="development"
)
```

### Role-based Ability Generation
```python
collector.collect_ability_generation(
    user_id="user123",
    user_role="software_engineer",
    user_command="I need to refactor this codebase",
    generated_ability={...},
    rag_context={...},
    model_used="gpt-4"
)
```

### RL Training Sequences
```python
collector.collect_rl_sequence(
    user_id="user123",
    state_data={...},
    action_taken="activate_focus_boost",
    reward=10.0,
    next_state_data={...}
)
```

### Gamification Data
```python
# Objectives
collector.collect_objective_data(...)

# Odysseys
collector.collect_odyssey_data(...)

# Seasons
collector.collect_season_data(...)

# Momentum
collector.collect_momentum_event(...)

# Streaks
collector.collect_streak_event(...)

# Boosts
collector.collect_boost_usage(...)
```

## Updated Export Methods

Export now supports:
- `classification` - Tier 1 data
- `ability_generation` - Tier 2 data
- `prompt_to_tasks` - Main engine data
- `rl_training` - Tier 3 sequences
- `gamification` - All gamification data

## Integration Examples

See the new comprehensive examples:
- `app/train/examples/deepiri_integration_example.py` - Full Deepiri integration
- `app/train/examples/integration_example.py` - Basic integration (still works)

## Quick Start

```bash
# 1. Check setup
python app/train/scripts/quick_start_data_collection.py

# 2. See integration examples
cat app/train/examples/deepiri_integration_example.py

# 3. Export data
python app/train/scripts/export_training_data.py
```

## Migration Notes

### Backward Compatibility
- Old `collect_challenge_generation()` still works
- Old `collect_classification()` still works (now enhanced)
- Old `collect_interaction()` still works (now enhanced)

### New Required Parameters
- `user_role` is now recommended for all collections
- `collect_ability_generation()` replaces `collect_challenge_generation()` for Tier 2

## Data Collection Priorities

### Priority 1: Prompt-to-Tasks Engine
**This is your main differentiator!**
- Collect every prompt -> tasks conversion
- Track user acceptance
- Track task completion rates
- This data is critical for training

### Priority 2: Three-Tier AI System
- **Tier 1**: 100+ examples per ability (5,000+ total)
- **Tier 2**: 1,000+ ability generation examples
- **Tier 3**: 10,000+ RL sequences

### Priority 3: Gamification System
- Collect all momentum events
- Track boost effectiveness
- Track streak patterns
- Use for RL reward signals

## Next Steps

1. **Instrument your endpoints** using `deepiri_integration_example.py`
2. **Add feedback collection** so users can label predictions
3. **Set up automatic collection** via middleware
4. **Export weekly** for training
5. **Monitor data quality** and fill gaps with synthetic data

## Full Documentation

- **Complete Guide**: `HOW_TO_COLLECT_TRAINING_DATA.md`
- **Quick Start**: `DATA_COLLECTION_QUICK_START.md`
- **Integration Examples**: `app/train/examples/deepiri_integration_example.py`

---

**The data collection system is now fully aligned with Deepiri's platform vision!** 

