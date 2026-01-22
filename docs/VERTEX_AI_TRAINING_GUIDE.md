# Vertex AI Training Guide

Quick guide to run training jobs on Google Cloud Vertex AI.

## ⚠️ CRITICAL: Pre-Flight Check Required

**BEFORE SUBMITTING ANY JOB, ALWAYS RUN:**

```bash
# Run before EVERY job submission
./scripts/preflight_check.sh
```

This automated script checks and fixes:
- Platform architecture (linux/amd64 for Mac)
- Service account permissions
- Image exists in correct region
- Config validation
- Data accessibility
- Quotas

**DO NOT SKIP THIS STEP** - It prevents common failures like platform mismatches, permission errors, and region issues.

## Prerequisites

1. **Preprocess data** (if not already done):
   ```bash
   uv run src/pname/data.py data/raw data/processed --num-categories 5
   ```

2. **Upload processed data to GCS**:
   ```bash
   gsutil -m cp data/processed/*.{json,pt} gs://mlops_project_data_bucket1-europe-west1/data/processed/
   ```

## Build and Push Docker Image

**⚠️ IMPORTANT: On Mac (ARM64), you MUST build for linux/amd64:**

```bash
# For Mac (ARM64) - REQUIRED
docker buildx build --platform linux/amd64 -f dockerfiles/train.dockerfile -t train:latest . --load

# For Linux (AMD64)
docker build -f dockerfiles/train.dockerfile . -t train:latest

# Tag and push
docker tag train:latest europe-west1-docker.pkg.dev/dtumlops-484310/container-registry/train:latest
docker push europe-west1-docker.pkg.dev/dtumlops-484310/container-registry/train:latest
```

**Or just run `./scripts/preflight_check.sh` which handles this automatically.**

## Submit Training Job

```bash
# Get WANDB key and create config
WANDB_KEY=$(gcloud secrets versions access latest --secret=WANDB_API_KEY)
sed "s|\${WANDB_API_KEY}|$WANDB_KEY|g" configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml > /tmp/config.yaml

# Submit job
gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name="balanced-training-$(date +%s)" \
  --config=/tmp/config.yaml
```

## Monitor Training

**Stream logs:**
```bash
gcloud ai custom-jobs stream-logs <JOB_ID> --region=europe-west1
```

**Check status:**
```bash
gcloud ai custom-jobs describe <JOB_ID> --region=europe-west1
```

**View in Console:**
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=dtumlops-484310

## Configuration

- **Machine**: `e2-highmem-4` with SPOT instances
- **Training**: 1 epoch, 2000 samples (balanced config)
- **Model Output**: Uses `AIP_MODEL_DIR` (automatically set by Vertex AI)
- **Output Path**: `gs://mlops_project_data_bucket1-europe-west1/experiments/balanced/trained_model.pt`
- **WandB**: Project `pname-arxiv-classifier`

## Available Configs

- `configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml` - Balanced training (2000 samples, 1 epoch, ~30-60 min)
- `configs/vertex_ai/vertex_ai_config_fast_cpu.yaml` - Fast training (2000 samples, 1 epoch, ~5-10 min)
