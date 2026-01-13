# Quick Ollama Connection Fix

## Commands to Verify Ollama is Working

### 1. Check if Ollama container is running
```bash
docker ps | grep ollama
```

### 2. Check if Ollama is responding inside the container
```bash
docker exec deepiri-ollama-dev curl -s http://localhost:11434/api/tags
```

### 3. List models in Ollama container
```bash
docker exec deepiri-ollama-dev ollama list
```

### 4. Test connection from host (external port)
```bash
curl http://localhost:11435/api/tags
```

### 5. Test if Cyrex container can reach Ollama
```bash
# Test via service name (preferred)
docker exec deepiri-cyrex-dev curl -s http://ollama:11434/api/tags

# Test via container name (fallback)
docker exec deepiri-cyrex-dev curl -s http://deepiri-ollama-dev:11434/api/tags
```

### 6. Check Ollama logs for errors
```bash
docker logs deepiri-ollama-dev --tail 50
```

### 7. Check Cyrex logs for Ollama connection errors
```bash
docker logs deepiri-cyrex-dev --tail 100 | grep -i ollama
```

## Quick Fixes

### If Ollama container is not running:
```bash
docker compose -f docker-compose.dev.yml up -d ollama
```

### If models are missing:
```bash
docker exec deepiri-ollama-dev ollama pull llama3:8b
```

### Restart Cyrex to pick up connection changes:
```bash
docker compose -f docker-compose.dev.yml restart cyrex
```

### Check environment variable in Cyrex:
```bash
docker exec deepiri-cyrex-dev env | grep OLLAMA
```

## Expected Configuration

- **Container name**: `deepiri-ollama-dev`
- **Service name**: `ollama` (for Docker Compose networking)
- **Internal port**: `11434` (inside container)
- **External port**: `11435` (from host)
- **Environment variable**: `OLLAMA_BASE_URL=http://ollama:11434`

## Network Check

Verify both containers are on the same network:
```bash
docker network inspect deepiri-dev-network | grep -A 5 "deepiri-ollama-dev\|deepiri-cyrex-dev"
```

