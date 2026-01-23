#!/bin/bash

# Script to build Docker images with multi-platform support
# Builds for both ARM64 and AMD64 so images work on any machine
#
# Usage:
#   ./scripts/build_docker.sh                    # Build multi-platform (requires --push for registry)
#   ./scripts/build_docker.sh --local            # Build for native platform only (can be loaded)
#   ./scripts/build_docker.sh --push <registry>  # Build and push multi-platform to registry

set -e

# Parse arguments
BUILD_MODE="multi"
REGISTRY=""
REGION="europe-west1"
PROJECT_ID="dtumlops-484310"

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            BUILD_MODE="local"
            shift
            ;;
        --push)
            BUILD_MODE="push"
            REGISTRY="$2"
            if [[ -z "$REGISTRY" ]]; then
                echo "ERROR: --push requires registry name"
                echo "Usage: ./scripts/build_docker.sh --push container-registry"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/build_docker.sh [--local|--push <registry>]"
            exit 1
            ;;
    esac
done

echo "Checking Docker daemon..."
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✓ Docker daemon is running"
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Detect native platform for local builds
HOST_ARCH=$(uname -m)
if [[ "$HOST_ARCH" == "arm64" || "$HOST_ARCH" == "aarch64" ]]; then
    NATIVE_PLATFORM="linux/arm64"
elif [[ "$HOST_ARCH" == "x86_64" || "$HOST_ARCH" == "amd64" ]]; then
    NATIVE_PLATFORM="linux/amd64"
else
    NATIVE_PLATFORM="linux/amd64"
fi

PLATFORMS="linux/amd64,linux/arm64"

images=(
    "dockerfiles/train.dockerfile:train:latest"
    "dockerfiles/train_tfidf.dockerfile:train-tfidf:latest"
    "dockerfiles/api.dockerfile:api:latest"
    "dockerfiles/evaluate.dockerfile:evaluate:latest"
    "monitoring_service/Dockerfile:monitoring:latest"
)

if [[ "$BUILD_MODE" == "local" ]]; then
    echo "Building Docker images for local use (platform: $NATIVE_PLATFORM)"
    echo "These images can be loaded and used locally."
    echo ""

    for image_spec in "${images[@]}"; do
        IFS=':' read -r dockerfile tag <<< "$image_spec"
        echo "Building $tag for $NATIVE_PLATFORM..."
        docker buildx build --platform "$NATIVE_PLATFORM" -f "$dockerfile" . -t "$tag" --load
        echo "✓ Built $tag"
        echo ""
    done

    echo "All Docker images built successfully for $NATIVE_PLATFORM (local use)!"

elif [[ "$BUILD_MODE" == "push" ]]; then
    echo "Building Docker images for multi-platform: $PLATFORMS"
    echo "Pushing to registry: $REGION-docker.pkg.dev/$PROJECT_ID/$REGISTRY"
    echo "This ensures images work on both ARM64 (Mac) and AMD64 (Linux/GCP) machines."
    echo ""

    for image_spec in "${images[@]}"; do
        IFS=':' read -r dockerfile tag <<< "$image_spec"
        image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY}/${tag}"
        echo "Building and pushing $tag for $PLATFORMS..."
        docker buildx build --platform "$PLATFORMS" -f "$dockerfile" . -t "$image_uri" -t "$tag" --push
        echo "✓ Built and pushed $image_uri"
        echo "  Also tagged as $tag (pull from registry to use)"
        echo ""
    done

    echo "All Docker images built and pushed successfully for $PLATFORMS!"
    echo "Images are now available in registry and will work on any platform!"
    echo "Pull with: docker pull ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY}/<image>:latest"

else
    echo "Building Docker images for multi-platform: $PLATFORMS"
    echo "⚠️  Note: Multi-platform images cannot be loaded with --load"
    echo "They will be built but not available locally."
    echo ""
    echo "Options:"
    echo "  --local              Build for native platform only (can be loaded)"
    echo "  --push <registry>    Build and push multi-platform to registry"
    echo ""

    for image_spec in "${images[@]}"; do
        IFS=':' read -r dockerfile tag <<< "$image_spec"
        echo "Building $tag for $PLATFORMS..."
        docker buildx build --platform "$PLATFORMS" -f "$dockerfile" . -t "$tag"
        echo "✓ Built $tag (use --push to make available)"
        echo ""
    done

    echo "Build complete. Use --push to push to registry, or --local for local use."
fi

echo ""
echo "To test the images:"
echo "  Training (PyTorch):    docker run --name experiment1 --rm train:latest"
echo "  Training (TF-IDF):     docker run --name experiment2 --rm train-tfidf:latest"
echo "  Evaluation:            docker run --name evaluate --rm -v \$(pwd)/trained_model.pt:/models/trained_model.pt -v \$(pwd)/data/processed:/data/processed evaluate:latest /models/trained_model.pt"
echo "  API:                   docker run --name api --rm -p 8000:8000 api:latest"
echo "  Monitoring:            docker run --name monitoring --rm -p 8001:8001 monitoring:latest"
