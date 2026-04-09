#!/bin/bash
# Quick script to test Ollama connection and list models
# Usage: ./test-ollama-connection.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Testing Ollama Connection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CONTAINER_NAME="deepiri-ollama-dev"
CYREX_CONTAINER="deepiri-cyrex-dev"
RECOMMENDED_MODELS=("mistral:7b" "llama3:8b" "codellama:7b" "qwen2.5:7b" "qwen2.5-coder:7b" "deepseek-coder:6.7b" "phi3:mini")

if ! command -v docker >/dev/null 2>&1; then
    echo "❌ docker is required but not found"
    exit 1
fi

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ Container '$CONTAINER_NAME' is not running!"
    echo "   Start it with: docker compose -f docker-compose.dev.yml up -d ollama"
    exit 1
fi

echo "✅ Container '$CONTAINER_NAME' is running"
echo ""

# Test connection from inside container
echo "📡 Testing connection from inside container..."
if docker exec "$CONTAINER_NAME" curl -s -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is responding inside container"
else
    echo "❌ Ollama is not responding inside container"
    exit 1
fi
echo ""

# List models
echo "📋 Available models in Ollama:"
echo ""
MODEL_LIST_OUTPUT="$(docker exec "$CONTAINER_NAME" ollama list)"
echo "$MODEL_LIST_OUTPUT"
echo ""

# Check if at least one recommended model is present.
MODEL_MATCH=false
for model in "${RECOMMENDED_MODELS[@]}"; do
    if echo "$MODEL_LIST_OUTPUT" | grep -q "^${model}[[:space:]]"; then
        MODEL_MATCH=true
        break
    fi
done

if [ "$MODEL_MATCH" = true ]; then
    echo "✅ Recommended model set detected in container"
else
    echo "⚠️  None of the recommended models are installed yet"
    echo "   Recommended baseline: ${RECOMMENDED_MODELS[*]}"
fi
echo ""

# Test from host (external port)
echo "📡 Testing connection from host (external port 11435)..."
if curl -s -f http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is accessible from host on port 11435"
    echo ""
    echo "📋 Models accessible from host:"
    if command -v jq >/dev/null 2>&1; then
        curl -s http://localhost:11435/api/tags | jq -r '.models[]?.name // "No models found"'
    else
        curl -s http://localhost:11435/api/tags | grep -o '"name":"[^"]*"' | cut -d: -f2 | tr -d '"' || true
    fi
else
    echo "⚠️  Ollama is not accessible from host on port 11435"
    echo "   (This is OK if you're only accessing from inside Docker network)"
fi
echo ""

# Test Docker network connectivity
echo "📡 Testing Docker network connectivity..."
if ! docker ps --format '{{.Names}}' | grep -q "^${CYREX_CONTAINER}$"; then
    echo "⚠️  Cyrex container '${CYREX_CONTAINER}' is not running; skipping network probe"
elif docker exec "$CYREX_CONTAINER" curl -s -f http://ollama:11434/api/tags > /dev/null 2>&1 2>/dev/null; then
    echo "✅ Cyrex container can reach Ollama via service name 'ollama:11434'"
elif docker exec "$CYREX_CONTAINER" curl -s -f http://deepiri-ollama-dev:11434/api/tags > /dev/null 2>&1 2>/dev/null; then
    echo "✅ Cyrex container can reach Ollama via container name 'deepiri-ollama-dev:11434'"
else
    echo "❌ Cyrex container cannot reach Ollama"
    echo "   Check if both containers are on the same Docker network"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Connection test complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
