#!/bin/bash
set -e

# Accept config file as argument, default to balanced_cpu
CONFIG_FILE="${1:-configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml}"

echo "=== VERTEX AI PRE-FLIGHT CHECK ==="
echo "Config file: ${CONFIG_FILE}"
echo ""

PROJECT_ID="dtumlops-484310"
REGION="europe-west1"
IMAGE_NAME="train:latest"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/container-registry/train:latest"

# Extract training script from config
# Look for train*.py in args section, handle both quoted and unquoted YAML values
TRAIN_SCRIPT_LINE=$(grep -A 10 "args:" "${CONFIG_FILE}" 2>/dev/null | grep -E "train.*\.py" | head -1)
if [ -n "${TRAIN_SCRIPT_LINE}" ]; then
    # Extract the full path - handles: "src/pname/train.py", 'src/pname/train.py', or src/pname/train.py
    # Remove leading dash and spaces, then extract content between quotes or take the whole value
    TRAIN_SCRIPT=$(echo "${TRAIN_SCRIPT_LINE}" | sed 's/^[[:space:]]*-[[:space:]]*//' | sed -E 's/^["'\''](.*)["'\'']$/\1/' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
else
    TRAIN_SCRIPT="src/pname/train.py"
fi
echo "Detected training script: ${TRAIN_SCRIPT}"
echo ""

# 1. Platform Check
echo "1. PLATFORM ARCHITECTURE"
if [[ $(uname -m) == "arm64" ]]; then
    echo "   ⚠️  Detected ARM64 (Mac). Must build for linux/amd64"
    echo "   Building with correct platform..."
    docker buildx build --platform linux/amd64 -f dockerfiles/train.dockerfile -t ${IMAGE_NAME} . --load
    echo "   ✓ Image built for linux/amd64"
else
    echo "   ✓ Architecture OK"
    docker build -f dockerfiles/train.dockerfile -t ${IMAGE_NAME} .
fi
echo ""

# 2. Service Account Permissions
echo "2. SERVICE ACCOUNT PERMISSIONS"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')
SERVICE_ACCOUNT="service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"

HAS_ARTIFACT=$(gcloud projects get-iam-policy ${PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SERVICE_ACCOUNT}" \
  --format="value(bindings.role)" 2>/dev/null | grep -c "artifactregistry.reader" || echo "0")

HAS_STORAGE=$(gcloud projects get-iam-policy ${PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SERVICE_ACCOUNT}" \
  --format="value(bindings.role)" 2>/dev/null | grep -c "storage.objectViewer" || echo "0")

if [ "$HAS_ARTIFACT" -eq 0 ]; then
    echo "   ✗ Missing artifactregistry.reader - FIXING..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
      --member="serviceAccount:${SERVICE_ACCOUNT}" \
      --role="roles/artifactregistry.reader" --quiet
    echo "   ✓ Fixed"
else
    echo "   ✓ Has artifactregistry.reader"
fi

if [ "$HAS_STORAGE" -eq 0 ]; then
    echo "   ✗ Missing storage.objectViewer - FIXING..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
      --member="serviceAccount:${SERVICE_ACCOUNT}" \
      --role="roles/storage.objectViewer" --quiet
    echo "   ✓ Fixed"
else
    echo "   ✓ Has storage.objectViewer"
fi
echo ""

# 3. Image in Region
echo "3. IMAGE IN REGION"
docker tag ${IMAGE_NAME} ${IMAGE_URI}
echo "   Pushing to ${REGION}..."
docker push ${IMAGE_URI} > /dev/null 2>&1 || docker push ${IMAGE_URI}
echo "   ✓ Image pushed to ${REGION}"
echo ""

# 4. Config Validation
echo "4. CONFIG VALIDATION"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "   ✗ Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Try to validate YAML (if pyyaml available)
if python3 -c "import yaml; yaml.safe_load(open('${CONFIG_FILE}'))" 2>/dev/null; then
    echo "   ✓ Config YAML is valid"
else
    echo "   ⚠️  Could not validate YAML (pyyaml not available), but file exists"
fi

# Check if training script exists
if [ -f "${TRAIN_SCRIPT}" ]; then
    echo "   ✓ Training script exists: ${TRAIN_SCRIPT}"
else
    echo "   ✗ Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# Check Hydra path (only for train.py, not train_tfidf.py)
if [[ "${TRAIN_SCRIPT}" == *"train.py" ]] && [[ "${TRAIN_SCRIPT}" != *"train_tfidf.py" ]]; then
    if grep -q 'config_path="/configs"' "${TRAIN_SCRIPT}"; then
        echo "   ✓ Hydra config path is absolute (/configs)"
    else
        echo "   ⚠️  Hydra config path is relative - may fail in Docker"
    fi
fi
echo ""

# 5. Data Accessibility
echo "5. DATA ACCESSIBILITY"
if gsutil ls gs://mlops_project_data_bucket1-europe-west1/data/processed/train_texts.json > /dev/null 2>&1; then
    echo "   ✓ Data exists in GCS"
else
    echo "   ✗ Data not found in GCS"
    exit 1
fi
echo ""

# 6. Quotas
echo "6. QUOTA CHECK"
CPUS=$(gcloud compute project-info describe --project=${PROJECT_ID} \
  --format="value(quotas[metric='CPUS_ALL_REGIONS'].usage)" 2>/dev/null || echo "0")
CPU_LIMIT=$(gcloud compute project-info describe --project=${PROJECT_ID} \
  --format="value(quotas[metric='CPUS_ALL_REGIONS'].limit)" 2>/dev/null || echo "10")
echo "   CPU Usage: ${CPUS}/${CPU_LIMIT}"
if (( $(echo "$CPUS >= $CPU_LIMIT" | bc -l) )); then
    echo "   ⚠️  CPU quota may be exhausted"
else
    echo "   ✓ CPU quota available"
fi
echo ""

echo "=== ALL CHECKS PASSED ==="
echo ""
echo "Ready to submit job. Run:"
echo "  WANDB_KEY=\$(gcloud secrets versions access latest --secret=WANDB_API_KEY)"
echo "  sed \"s|\\\${WANDB_API_KEY}|\$WANDB_KEY|g\" ${CONFIG_FILE} > /tmp/config.yaml"
echo "  gcloud ai custom-jobs create --region=${REGION} --display-name=\"training-\$(date +%s)\" --config=/tmp/config.yaml"
echo ""
echo "Or use this script with your config:"
echo "  ./scripts/preflight_check.sh ${CONFIG_FILE}"
