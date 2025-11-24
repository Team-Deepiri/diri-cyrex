# Cyrex Interface

A lightweight Vite + React dashboard to manually exercise the Cyrex intelligence API. It lives beside the FastAPI service and runs as its own container in the developer compose stack.

## Features

- Friendly chat surface powered by the `/agent/intelligence/generate-ability` endpoint  
- One-click forms for:
  - `/agent/intelligence/route-command`
  - `/agent/intelligence/generate-ability`
  - `/agent/intelligence/recommend-action`
  - `/agent/intelligence/knowledge/query`
- Inline instructions for server-side testing (`pytest`, `mypy`) and linting this UI

## Local development

```bash
cd deepiri/diri-cyrex/cyrex-interface
npm install
npm run dev -- --host 0.0.0.0 --port 5175
```

The app defaults to `http://localhost:8000` but you can override it (and the `x-api-key`) in the “Connection” panel.

## Docker / Compose

The service is added to `docker-compose.dev.yml` as `cyrex-interface` on port `5175`. Start it alongside `cyrex`:

```bash
cd deepiri
docker compose -f docker-compose.dev.yml up cyrex cyrex-interface
```

Then visit http://localhost:5175.

## Testing & Linting

This UI exposes the existing backend test commands so you can trigger them from the CLI:

- `pytest` – backend tests (run inside `diri-cyrex`)
- `mypy app` – backend type checks
- `npm run lint` – lint this UI

