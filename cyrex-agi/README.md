# Cyrex-AGI - Autonomous AI System

**Purpose**: Autonomous AI system with platform awareness and self-improvement capabilities

## Architecture

Cyrex-AGI is a separate service within `diri-cyrex` that:
- Observes platform events via streaming service
- Makes autonomous decisions
- Self-improves through Helox training pipelines
- Interacts with Cyrex runtime and platform services

## Structure

```
cyrex-agi/
├── app/
│   ├── core/              # AGI core engine
│   ├── awareness/         # Platform awareness
│   ├── decision_making/   # Autonomous decisions
│   ├── self_improvement/  # Self-improvement loops
│   └── integrations/
│       ├── streaming/     # Event consumption
│       ├── cyrex_bridge/  # Connection to Cyrex
│       └── platform/      # Platform interaction
└── Dockerfile
```

## Status

**Phase 5** - Placeholder structure. Full implementation planned for Phase 5 of roadmap.

## Related

- `diri-cyrex`: Runtime AI services (AGI interacts with)
- `diri-helox`: ML training (AGI uses for self-improvement)
- `deepiri-synapse`: Streaming service (AGI observes/acts via)
- `deepiri-modelkit`: Shared contracts
