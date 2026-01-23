#!/bin/bash

# Script to build and test Docker images

set -e

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

echo "Building training Docker image..."
docker build -f dockerfiles/train.dockerfile . -t train:latest
echo "✓ Training image built successfully"
echo ""

echo "Building evaluation Docker image..."
docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
echo "✓ Evaluation image built successfully"
echo ""

echo "Building API Docker image..."
docker build -f dockerfiles/api.dockerfile . -t api:latest
echo "✓ API image built successfully"
echo ""

echo "All Docker images built successfully!"
echo ""
echo "To test the images:"
echo "  Training:    docker run --name experiment1 --rm train:latest"
echo "  Evaluation:  docker run --name evaluate --rm -v \$(pwd)/trained_model.pt:/models/trained_model.pt -v \$(pwd)/data/processed:/data/processed evaluate:latest /models/trained_model.pt"
echo "  API:         docker run --name api --rm -p 8000:8000 api:latest"
