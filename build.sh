#!/bin/bash
# Smart build script with GPU detection and CPU fallback

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${1:-Dockerfile}"
IMAGE_NAME="${2:-deepiri-dev-cyrex:latest}"

echo "🔍 Detecting GPU capabilities..."

# Detect GPU and get base image
BASE_IMAGE=$(bash "$SCRIPT_DIR/detect_gpu.sh")

echo "📦 Using base image: $BASE_IMAGE"
echo "🔨 Building Docker image with CUDA 12.8 support (automatic RTX 5080/5090 compatibility)..."

# Build with detected base image (CUDA 12.8 is now automatic)
docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --file "$SCRIPT_DIR/$DOCKERFILE" \
    --tag "$IMAGE_NAME" \
    "$SCRIPT_DIR"

echo "✅ Build complete!"
echo "📊 Image info:"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

