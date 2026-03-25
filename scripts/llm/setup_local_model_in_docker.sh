#!/bin/bash
# Setup script for local LLM (Ollama) as Docker container
# This script sets up Ollama as a Docker service and pulls the recommended model (llama3:8b)

set -e

echo "🚀 Setting up Ollama as Docker container for Cyrex..."
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    echo "   Please install Docker and try again"
    exit 1
fi

# Check if docker compose is available
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo "❌ Docker Compose is not available"
    echo "   Please install Docker Compose and try again"
    exit 1
fi

# Find the docker-compose file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.dev.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "❌ docker-compose.dev.yml not found at: $COMPOSE_FILE"
    echo "   Please run this script from the deepiri directory or ensure docker-compose.dev.yml exists"
    exit 1
fi

echo "📋 Using compose file: $COMPOSE_FILE"
echo ""

# Check if Ollama service is defined in compose file
if ! grep -q "ollama:" "$COMPOSE_FILE"; then
    echo "⚠️  Ollama service not found in docker-compose.dev.yml"
    echo "   The Ollama service should be defined in the compose file"
    echo "   Please ensure the ollama service is added to docker-compose.dev.yml"
    exit 1
fi

echo "📦 Pulling Ollama Docker image..."
docker pull ollama/ollama:latest

echo ""
echo "🔄 Starting Ollama container..."
cd "$PROJECT_ROOT"
$DOCKER_COMPOSE_CMD -f docker-compose.dev.yml up -d ollama

echo ""
echo "⏳ Waiting for Ollama to be ready..."
sleep 5

# Wait for Ollama to be ready (max 30 seconds)
MAX_WAIT=30
WAIT_TIME=0
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if docker exec deepiri-ollama-dev curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama container is ready!"
        break
    fi
    echo "   Waiting for Ollama to start... ($WAIT_TIME/$MAX_WAIT seconds)"
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "⚠️  Ollama container started but may not be fully ready yet"
    echo "   You can check status with: docker logs deepiri-ollama-dev"
fi

echo ""
echo "🔍 Checking if model already exists..."
MODEL_EXISTS=$(docker exec deepiri-ollama-dev ollama list 2>/dev/null | grep -q "llama3:8b" && echo "yes" || echo "no")

if [[ "$MODEL_EXISTS" == "yes" ]]; then
    echo "✅ Model llama3:8b already exists in container!"
    echo "   Skipping download..."
else
    echo "📥 Pulling recommended model: llama3:8b"
    echo "   This may take a few minutes depending on your internet connection... (4.7GB download)"
    echo "   Please wait, this is a one-time download..."
    
    # Pull model (without -it flag for non-interactive execution)
    if docker exec deepiri-ollama-dev ollama pull llama3:8b; then
        echo ""
        echo "✅ Model downloaded successfully!"
    else
        echo ""
        echo "❌ Failed to pull model. Please check your internet connection and try again:"
        echo "   docker exec -it deepiri-ollama-dev ollama pull llama3:8b"
        exit 1
    fi
fi

echo ""
echo "🔍 Verifying installation..."
docker exec deepiri-ollama-dev ollama list

echo ""
echo "✅ Ollama Docker setup complete!"
echo ""
echo "📝 Configuration:"
echo "   - Container name: deepiri-ollama-dev"
echo "   - Service URL: http://ollama:11434 (from other containers)"
echo "   - Host URL: http://localhost:11434 (from host machine)"
echo "   - Model: llama3:8b"
echo ""
echo "📝 Next steps:"
echo "   1. The Ollama service is already configured in docker-compose.dev.yml"
echo "   2. Cyrex will automatically connect to Ollama at http://ollama:11434"
echo "   3. Start Cyrex service:"
echo "      docker compose -f docker-compose.dev.yml up -d cyrex"
echo ""
echo "💡 Tips:"
echo "   - View Ollama logs: docker logs deepiri-ollama-dev"
echo "   - Restart Ollama: docker compose -f docker-compose.dev.yml restart ollama"
echo "   - Pull more models: docker exec -it deepiri-ollama-dev ollama pull <model-name>"
echo "   - Models are persisted in the ollama_dev_data Docker volume"

