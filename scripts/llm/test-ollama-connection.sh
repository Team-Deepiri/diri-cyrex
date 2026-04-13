#!/bin/bash
# Quick script to test Ollama connection and list models
# Usage: ./test-ollama-connection.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Testing Ollama Connection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CONTAINER_NAME="deepiri-ollama-dev"

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
docker exec "$CONTAINER_NAME" ollama list
echo ""

# Test from host (external port)
echo "📡 Testing connection from host (external port 11435)..."
if curl -s -f http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is accessible from host on port 11435"
    echo ""
    echo "📋 Models accessible from host:"
    curl -s http://localhost:11435/api/tags | jq -r '.models[]?.name // "No models found"'
else
    echo "⚠️  Ollama is not accessible from host on port 11435"
    echo "   (This is OK if you're only accessing from inside Docker network)"
fi
echo ""

# Test Docker network connectivity
echo "📡 Testing Docker network connectivity..."
if docker exec deepiri-cyrex-dev curl -s -f http://ollama:11434/api/tags > /dev/null 2>&1 2>/dev/null; then
    echo "✅ Cyrex container can reach Ollama via service name 'ollama:11434'"
elif docker exec deepiri-cyrex-dev curl -s -f http://deepiri-ollama-dev:11434/api/tags > /dev/null 2>&1 2>/dev/null; then
    echo "✅ Cyrex container can reach Ollama via container name 'deepiri-ollama-dev:11434'"
else
    echo "❌ Cyrex container cannot reach Ollama"
    echo "   Check if both containers are on the same Docker network"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Connection test complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

