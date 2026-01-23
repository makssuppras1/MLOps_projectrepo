#!/bin/bash
# Validation script for API deployment to Cloud Run
# Checks prerequisites and configuration before deploying

set -e

echo "=========================================="
echo "API DEPLOYMENT VALIDATION"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# 1. Check Python syntax
echo "1. CHECKING PYTHON SYNTAX"
echo "-------------------------"
if python3 -m py_compile tasks.py 2>/dev/null; then
    echo "✓ tasks.py syntax is valid"
else
    echo "✗ tasks.py has syntax errors"
    ((ERRORS++))
fi
echo ""

# 2. Check Docker is running
echo "2. CHECKING DOCKER"
echo "------------------"
if ! docker info > /dev/null 2>&1; then
    echo "✗ Docker daemon is not running"
    echo "  Please start Docker Desktop and try again"
    ((ERRORS++))
else
    echo "✓ Docker daemon is running"

    # Check buildx
    if docker buildx version > /dev/null 2>&1; then
        echo "✓ Docker buildx is available"

        # Check if a builder exists
        if docker buildx ls | grep -q .; then
            echo "✓ Docker buildx builders are configured"
        else
            echo "⚠ No buildx builders found. Creating default builder..."
            docker buildx create --name default --use 2>/dev/null || true
            docker buildx inspect --bootstrap 2>/dev/null || true
        fi
    else
        echo "✗ Docker buildx is not available"
        echo "  Please install Docker buildx"
        ((ERRORS++))
    fi
fi
echo ""

# 3. Check gcloud CLI
echo "3. CHECKING GCLOUD CLI"
echo "----------------------"
if ! command -v gcloud &> /dev/null; then
    echo "✗ gcloud CLI is not installed"
    echo "  Install from: https://cloud.google.com/sdk/docs/install"
    ((ERRORS++))
else
    echo "✓ gcloud CLI is installed"

    # Check authentication
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)
        echo "✓ Authenticated as: $ACTIVE_ACCOUNT"
    else
        echo "✗ Not authenticated with gcloud"
        echo "  Run: gcloud auth login"
        ((ERRORS++))
    fi

    # Check project
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
    if [[ -n "$PROJECT_ID" ]]; then
        echo "✓ GCP project is set: $PROJECT_ID"
    else
        echo "⚠ GCP project is not set"
        echo "  Run: gcloud config set project PROJECT_ID"
        ((WARNINGS++))
    fi
fi
echo ""

# 4. Check Docker authentication for GCP
echo "4. CHECKING DOCKER-GCP AUTHENTICATION"
echo "-------------------------------------"
REGION="europe-west1"
if docker pull "${REGION}-docker.pkg.dev/test/test/test:latest" 2>&1 | grep -q "authentication required\|unauthorized"; then
    echo "⚠ Docker may not be configured for GCP Artifact Registry"
    echo "  Run: gcloud auth configure-docker ${REGION}-docker.pkg.dev"
    ((WARNINGS++))
else
    # Try to check if we can access the registry (ignore pull errors, just check auth)
    if gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null; then
        echo "✓ Docker is configured for GCP Artifact Registry"
    else
        echo "⚠ Could not verify Docker-GCP configuration"
        ((WARNINGS++))
    fi
fi
echo ""

# 5. Check required files exist
echo "5. CHECKING REQUIRED FILES"
echo "--------------------------"
REQUIRED_FILES=(
    "tasks.py"
    "dockerfiles/api.dockerfile"
    "app/main.py"
    "pyproject.toml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ $file exists"
    else
        echo "✗ $file is missing"
        ((ERRORS++))
    fi
done
echo ""

# 6. Check invoke/deploy-api task exists
echo "6. CHECKING DEPLOY TASK"
echo "-----------------------"
if grep -q "@task" tasks.py && grep -q "def deploy_api" tasks.py; then
    echo "✓ deploy_api task is defined in tasks.py"

    # Check for multi-platform parameter
    if grep -q "multi_platform.*bool.*True" tasks.py; then
        echo "✓ Multi-platform support is enabled by default"
    else
        echo "⚠ Multi-platform parameter may not be set correctly"
        ((WARNINGS++))
    fi
else
    echo "✗ deploy_api task is not found in tasks.py"
    ((ERRORS++))
fi
echo ""

# 7. Check Dockerfile supports multi-platform
echo "7. CHECKING DOCKERFILE"
echo "----------------------"
if grep -q "TARGETPLATFORM\|ARG TARGETPLATFORM" dockerfiles/api.dockerfile; then
    echo "✓ Dockerfile supports multi-platform builds"
else
    echo "⚠ Dockerfile may not explicitly support multi-platform builds"
    ((WARNINGS++))
fi
echo ""

# Summary
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [[ $ERRORS -eq 0 ]]; then
    echo "✅ All critical checks passed!"
    echo ""
    echo "You can now deploy the API with:"
    echo "  uv run invoke deploy-api"
    echo ""
    if [[ $WARNINGS -gt 0 ]]; then
        echo "⚠️  Note: There are $WARNINGS warning(s) above. Review them before deploying."
    fi
    exit 0
else
    echo "❌ Found $ERRORS error(s). Please fix them before deploying."
    echo ""
    echo "Common fixes:"
    echo "  - Start Docker Desktop"
    echo "  - Run: gcloud auth login"
    echo "  - Run: gcloud config set project PROJECT_ID"
    echo "  - Run: gcloud auth configure-docker europe-west1-docker.pkg.dev"
    exit 1
fi
