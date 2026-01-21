# Pre-Flight Checklist for Vertex AI Jobs

**ALWAYS RUN THIS BEFORE SUBMITTING ANY JOB**

## Quick Command

```bash
# Run before EVERY job submission
./scripts/preflight_check.sh
```

## 1. PLATFORM ARCHITECTURE (CRITICAL)

```bash
# Check your local architecture
uname -m
# If arm64 (Mac M1/M2), you MUST build for linux/amd64:
docker buildx build --platform linux/amd64 -f dockerfiles/train.dockerfile -t train:latest . --load

# Verify image platform
docker inspect train:latest | grep -A 5 "Architecture"
# Must show: "Architecture": "amd64"
```

**FIX**: Always build with `--platform linux/amd64` on Mac

## 2. SERVICE ACCOUNT PERMISSIONS (CRITICAL)

```bash
PROJECT_NUMBER=$(gcloud projects describe dtumlops-484310 --format='value(projectNumber)')
SERVICE_ACCOUNT="service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"

# Check permissions
gcloud projects get-iam-policy dtumlops-484310 \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SERVICE_ACCOUNT}" \
  --format="table(bindings.role)" | grep -E "artifactregistry|storage"

# Required roles:
# - roles/artifactregistry.reader
# - roles/storage.objectViewer
```

**FIX**: Grant missing permissions immediately

## 3. IMAGE EXISTS IN CORRECT REGION

```bash
# Check if image exists in the region you're using
REGION="europe-west1"  # or us-central1
gcloud artifacts docker images describe \
  ${REGION}-docker.pkg.dev/dtumlops-484310/container-registry/train:latest

# If fails, push it:
docker tag train:latest ${REGION}-docker.pkg.dev/dtumlops-484310/container-registry/train:latest
docker push ${REGION}-docker.pkg.dev/dtumlops-484310/container-registry/train:latest
```

**FIX**: Image must exist in same region as job

## 4. HYDRA CONFIG PATH

```bash
# Check config path in train.py
grep "config_path" src/pname/train.py
# For Docker: must be absolute "/configs"
# For VM: can be relative "../../configs"

# Verify configs are copied in Dockerfile
grep "COPY configs" dockerfiles/train.dockerfile
```

**FIX**: Use absolute path `/configs` for Docker containers

## 5. DATA ACCESSIBILITY

```bash
# Verify data exists in GCS
gsutil ls gs://mlops_project_data_bucket1-europe-west1/data/processed/train_texts.json

# For VM: Download data
gsutil -m cp gs://mlops_project_data_bucket1-europe-west1/data/processed/*.{json,pt} data/processed/
```

**FIX**: Data must be accessible (GCS mount or local copy)

## 6. QUOTAS

```bash
# Check CPU and instance quotas
gcloud compute project-info describe --project=dtumlops-484310 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | \
  grep -E "CPUS_ALL_REGIONS|INSTANCES"
```

**FIX**: Ensure quotas are not exhausted

## 7. CONFIG FILE STRUCTURE

```bash
# Verify YAML is valid
python3 -c "import yaml; yaml.safe_load(open('configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml'))"

# Check for region mismatches
grep -E "imageUri|outputUriPrefix" configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml
# Image region should match job region (or be accessible cross-region)
```

**FIX**: Ensure no region mismatches

## 8. ENVIRONMENT VARIABLES

```bash
# Verify WANDB_API_KEY will be substituted
cat configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml | grep WANDB_API_KEY
# Should have ${WANDB_API_KEY} placeholder

# Test substitution
WANDB_KEY=$(gcloud secrets versions access latest --secret=WANDB_API_KEY)
sed "s|\${WANDB_API_KEY}|$WANDB_KEY|g" configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml > /tmp/test.yaml
grep "value:" /tmp/test.yaml | grep -v "value: \"\""
```

**FIX**: Ensure secrets are accessible and will be substituted

## 9. DOCKERFILE VERIFICATION

```bash
# Check all required files are copied
grep -E "COPY|WORKDIR|ENTRYPOINT" dockerfiles/train.dockerfile

# Verify .dockerignore exists
test -f .dockerignore && echo "✓ .dockerignore exists" || echo "✗ Missing .dockerignore"
```

**FIX**: Ensure .dockerignore prevents bloat

## 10. PYTHON VERSION COMPATIBILITY

```bash
# Check pyproject.toml requires-python
grep "requires-python" pyproject.toml
# Should be compatible with base image Python version
```

**FIX**: Ensure Python version compatibility

---

## COMMON MISTAKES TO AVOID

1. ❌ Building on Mac without `--platform linux/amd64`
2. ❌ Image in wrong region
3. ❌ Missing service account permissions
4. ❌ Wrong Hydra config path (relative vs absolute)
5. ❌ Data not accessible
6. ❌ Quota exhausted
7. ❌ Region mismatch in config
8. ❌ Missing .dockerignore (huge images)
9. ❌ Python version incompatibility
10. ❌ WANDB_API_KEY not substituted
