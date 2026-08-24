# Elkedel integration (Cyrex)

Cyrex mounts Elkedel as a **separate** sensory / episodic-memory service.
Tools stay under the `elkedel.*` namespace — never folded into `cyrex.*`.

## Config

| Env | Default | Purpose |
|-----|---------|---------|
| `ELKEDEL_BASE_URL` | `http://elkedel:8765` | Runtime HTTP |
| `ELKEDEL_MCP_URL` | `http://elkedel-mcp:8766/mcp` | Streamable HTTP MCP |
| `ELKEDEL_API_KEY` | _(empty)_ | Shared secret (`x-api-key`) |
| `ELKEDEL_TIMEOUT_SEC` | `30` | Client timeout |
| `ELKEDEL_EYES_SYNC_ENABLED` | `true` (dev) | Persist visual identities → artifact store |

## Client + eyes

```python
from app.integrations.elkedel import get_elkedel_client

elkedel = get_elkedel_client()
await elkedel.eyes_start()
await elkedel.eyes_scene()
await elkedel.eyes_events()
```

Cyrex HTTP proxy: `/eyes/status|start|stop|scene|events|where`.

Background **ElkedelEyesSync** (`app/integrations/elkedel/artifact_sync.py`) writes
`VisualObservation` artifacts to Postgres when new identities spawn.

## Compose (dev)

```bash
docker compose -f docker-compose.dev.yml up -d postgres-cyrex elkedel elkedel-mcp cyrex
```

Memory lives on `postgres-cyrex` (`cyrex.elkedel_*` tables, migration `140_elkedel_memory.sql`).

Health: `/orchestration/health-comprehensive` → `services.elkedel`.

## Contract

See `deepiri-elkedel/docs/MCP.md` (contract version `0.2.0`).
