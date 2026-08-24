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

## Client

```python
from app.integrations.elkedel import get_elkedel_client

elkedel = get_elkedel_client()
await elkedel.ready()
await elkedel.stats()          # elkedel.stats
await elkedel.remember(jpeg)   # elkedel.remember
await elkedel.what_changed(0)  # elkedel.what_changed
```

## Compose

From `deepiri-platform`:

```bash
docker compose -f docker-compose.yml -f docker-compose.elkedel.yml up -d elkedel elkedel-mcp
```

Requires submodule `deepiri-elkedel` (see platform `.gitmodules`).

Health: `/orchestration/health-comprehensive` reports `services.elkedel`.

## Contract

See `deepiri-elkedel/docs/MCP.md` (contract version `0.2.0`).
