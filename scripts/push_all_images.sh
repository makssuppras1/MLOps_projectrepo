#!/bin/bash

# Script to build and push all Docker images to GCP Artifact Registry
# Builds multi-platform images (ARM64 + AMD64) so they work on any machine

set -e

# Configuration
REGION="europe-west1"
PROJECT_ID="dtumlops-484310"
REGISTRY="container-registry"
PLATFORMS="linux/amd64,linux/arm64"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building and Pushing All Docker Images${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY}"
echo "Platforms: ${PLATFORMS}"
echo ""

# Check Docker daemon
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Check if logged in to GCP
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "ERROR: Not logged in to GCP."
    echo "Please run: gcloud auth login"
    exit 1
fi

# Configure Docker to use gcloud as credential helper
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Define all images to build and push
images=(
    "dockerfiles/train.dockerfile:train:latest"
    "dockerfiles/train_tfidf.dockerfile:train-tfidf:latest"
    "dockerfiles/api.dockerfile:api:latest"
    "dockerfiles/evaluate.dockerfile:evaluate:latest"
    "monitoring_service/Dockerfile:monitoring:latest"
)

# Build and push each image
for image_spec in "${images[@]}"; do
    IFS=':' read -r dockerfile tag <<< "$image_spec"
    image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY}/${tag}"

    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}Building and pushing: ${tag}${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
    echo "Dockerfile: ${dockerfile}"
    echo "Tag: ${tag}"
    echo "Registry URI: ${image_uri}"
    echo "Platforms: ${PLATFORMS}"
    echo ""

    docker buildx build \
        --platform "${PLATFORMS}" \
        -f "${dockerfile}" \
        -t "${image_uri}" \
        -t "${tag}" \
        . \
        --push

    echo -e "${GREEN}✓ Successfully pushed ${image_uri}${NC}"
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All images pushed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Images are now available at:"
for image_spec in "${images[@]}"; do
    IFS=':' read -r dockerfile tag <<< "$image_spec"
    image_uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY}/${tag}"
    echo "  - ${image_uri}"
done
echo ""
echo "To pull and use an image:"
echo "  docker pull ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY}/train:latest"
echo ""
echo "These multi-platform images will work on:"
echo "  - ARM64 Macs (Apple Silicon)"
echo "  - AMD64 Linux machines"
echo "  - GCP Vertex AI / Cloud Run"
echo "  - Any machine with Docker"
