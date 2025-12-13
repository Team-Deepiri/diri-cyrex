#!/bin/bash
# Check if Ollama container has models, and prompt to pull models if none exist

set -e

CONTAINER_NAME="deepiri-ollama-dev"

echo "🔍 Checking for Ollama models..."
echo ""

# Check if the specified container exists and is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Try to find any ollama container
    OLLAMA_CONTAINER=$(docker ps --format '{{.Names}}' | grep -i ollama | head -n 1)
    
    if [ -z "$OLLAMA_CONTAINER" ]; then
        echo "❌ Error: No Ollama container found"
        echo "   Expected: $CONTAINER_NAME"
        echo "   Please start the Ollama service first"
        exit 1
    else
        echo "⚠️  Container '$CONTAINER_NAME' not found, using '$OLLAMA_CONTAINER' instead"
        CONTAINER_NAME="$OLLAMA_CONTAINER"
    fi
fi

echo "📦 Using container: $CONTAINER_NAME"
echo ""

# List models in the container
echo "📋 Checking for installed models..."
MODELS=$(docker exec "$CONTAINER_NAME" ollama list 2>/dev/null || echo "")

# Check if there are any models (ollama list outputs header + models, so we check for more than 1 line)
MODEL_COUNT=$(echo "$MODELS" | grep -v "^NAME" | grep -v "^$" | wc -l | tr -d ' ')

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "⚠️  No models found in Ollama container"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  IMPORTANT: This project uses llama3:8b as the DEFAULT model"
    echo "   It is recommended to install llama3:8b for compatibility with this project"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Available models to install:"
    echo ""
    echo "  📌 RECOMMENDED (Project Default):"
    echo "    1) llama3:8b (4.7GB) - ⭐ DEFAULT - Used by this project"
    echo ""
    echo "  🦙 Llama Models:"
    echo "    2) llama3.2:1b (1.3GB) - Small, fast"
    echo "    3) llama3.2:3b (2.0GB) - Balanced"
    echo "    4) llama3.1:8b (4.7GB) - Latest Llama 3.1"
    echo "    5) llama3.1:70b (40GB) - Large, powerful"
    echo ""
    echo "  🌟 Mistral Models:"
    echo "    6) mistral:7b (4.1GB) - Efficient, high quality"
    echo "    7) mistral-nemo:12b (7.0GB) - Enhanced Mistral"
    echo ""
    echo "  💎 Gemma Models:"
    echo "    8) gemma2:2b (1.4GB) - Small, efficient"
    echo "    9) gemma2:9b (5.4GB) - Balanced"
    echo "   10) gemma2:27b (16GB) - Large, powerful"
    echo ""
    echo "  🧠 Phi Models:"
    echo "   11) phi3:mini (2.3GB) - Small, fast"
    echo "   12) phi3:medium (7.0GB) - Balanced"
    echo ""
    echo "  💻 Code Models:"
    echo "   13) codellama:7b (3.8GB) - Code generation"
    echo "   14) codellama:13b (7.3GB) - Larger code model"
    echo "   15) deepseek-coder:6.7b (4.1GB) - Advanced coding"
    echo ""
    echo "  🌟 Other Models:"
    echo "   16) qwen2.5:7b (4.4GB) - Alibaba's model"
    echo "   17) neural-chat:7b (4.1GB) - Conversational AI"
    echo ""
    echo "  🔧 Custom:"
    echo "   18) Enter custom model name"
    echo ""
    echo "Enter model number(s) to install (comma-separated, e.g., 1,6,13):"
    read -r selection
    
    # Parse comma-separated selection
    IFS=',' read -ra SELECTED <<< "$selection"
    
    # Model mapping
    declare -A MODEL_MAP=(
        ["1"]="llama3:8b"
        ["2"]="llama3.2:1b"
        ["3"]="llama3.2:3b"
        ["4"]="llama3.1:8b"
        ["5"]="llama3.1:70b"
        ["6"]="mistral:7b"
        ["7"]="mistral-nemo:12b"
        ["8"]="gemma2:2b"
        ["9"]="gemma2:9b"
        ["10"]="gemma2:27b"
        ["11"]="phi3:mini"
        ["12"]="phi3:medium"
        ["13"]="codellama:7b"
        ["14"]="codellama:13b"
        ["15"]="deepseek-coder:6.7b"
        ["16"]="qwen2.5:7b"
        ["17"]="neural-chat:7b"
    )
    
    MODELS_TO_PULL=()
    
    for num in "${SELECTED[@]}"; do
        num=$(echo "$num" | tr -d ' ') # Remove whitespace
        
        if [ "$num" == "18" ]; then
            echo ""
            echo "Enter custom model name (e.g., llama3:8b, mistral:7b):"
            read -r custom_model
            if [ -n "$custom_model" ]; then
                MODELS_TO_PULL+=("$custom_model")
            fi
        elif [ -n "${MODEL_MAP[$num]}" ]; then
            MODELS_TO_PULL+=("${MODEL_MAP[$num]}")
        else
            echo "⚠️  Invalid selection: $num (skipping)"
        fi
    done
    
    if [ ${#MODELS_TO_PULL[@]} -eq 0 ]; then
        echo ""
        echo "❌ No valid models selected. Exiting."
        exit 0
    fi
    
    echo ""
    echo "📥 Pulling ${#MODELS_TO_PULL[@]} model(s)..."
    echo "   This may take several minutes depending on your internet connection and model sizes..."
    echo ""
    
    SUCCESS_COUNT=0
    FAILED_COUNT=0
    
    for model in "${MODELS_TO_PULL[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📦 Pulling: $model"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        if docker exec -it "$CONTAINER_NAME" ollama pull "$model"; then
            echo ""
            echo "✅ Successfully pulled: $model"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo ""
            echo "❌ Failed to pull: $model"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
        echo ""
    done
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Summary: $SUCCESS_COUNT succeeded, $FAILED_COUNT failed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 Current models in container:"
    docker exec "$CONTAINER_NAME" ollama list
    echo ""
    
    if [ $SUCCESS_COUNT -gt 0 ]; then
        echo "✅ Model installation complete!"
    fi
    
    if [ $FAILED_COUNT -gt 0 ]; then
        echo "⚠️  Some models failed to install. You can try again later."
        exit 1
    fi
else
    echo "✅ Found $MODEL_COUNT model(s) in Ollama container:"
    echo ""
    echo "$MODELS"
    echo ""
    echo "ℹ️  To pull additional models, use:"
    echo "   docker exec -it $CONTAINER_NAME ollama pull <model-name>"
    echo ""
    echo "   Or run this script again to see the model selection menu"
fi

