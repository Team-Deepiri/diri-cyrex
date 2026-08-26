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

**Corrected 2026-08-07:** this is not an empty placeholder. `app/main.py` (~150 LOC) is a real,
running service — a Redis Streams consumer on port 8003, consumer group `cyrex-agi-pressure`,
subscribed to `pipeline.pressure.events` and `pipeline.artifact.invalidation`. It observes and
counts events (`GET /health` returns the counters); it does not yet make decisions or act on
what it observes. That's an honest **V1**, not Phase 5. V2 (splicing-enabled multi-agent) through
V5 (self-evolution proposals) remain unscheduled — see
[`docs/agi/CYREX_AGI_IMPLEMENTATION_PLAN_V2.md`](../docs/agi/CYREX_AGI_IMPLEMENTATION_PLAN_V2.md)
Wave 3 and [`docs/agi/STATUS.md`](../docs/agi/STATUS.md) for current build status.

## Related

- `diri-cyrex`: Runtime AI services (AGI interacts with)
- `diri-helox`: ML training (AGI uses for self-improvement)
- `deepiri-synapse`: Streaming service (AGI observes/acts via)
- `deepiri-modelkit`: Shared contracts
