# GCP Cloud Training Guide

This guide shows you how to train your model on Google Cloud Platform.

## Prerequisites

1. **GCP Account**: You already have this ✅
2. **gcloud CLI**: Install and authenticate
   ```bash
   # Install gcloud (if not installed)
   # macOS: brew install google-cloud-sdk

   # Authenticate
   gcloud auth login
   gcloud auth application-default login

   # Set your project
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Enable required APIs**:
   ```bash
   gcloud services enable compute.googleapis.com
   gcloud services enable aiplatform.googleapis.com  # For Vertex AI
   gcloud services enable cloudbuild.googleapis.com  # For building Docker images
   ```

---

## Option 1: GCP Compute Engine with GPU (Recommended for Full Control)

### Step 1: Build and Push Docker Image to GCR

```bash
# Set your GCP project
export GCP_PROJECT_ID=your-project-id
gcloud config set project $GCP_PROJECT_ID

# Build and push Docker image
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/train:latest \
  --file dockerfiles/train.dockerfile

# Or build locally and push
docker build -f dockerfiles/train.dockerfile -t gcr.io/$GCP_PROJECT_ID/train:latest .
docker push gcr.io/$GCP_PROJECT_ID/train:latest
```

### Step 2: Create GPU Instance and Run Training

```bash
# Create a GPU instance (NVIDIA T4)
gcloud compute instances create training-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=cos-gpu \
  --image-project=cos-cloud \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd

# SSH into the instance
gcloud compute ssh training-instance --zone=us-central1-a

# On the instance, run:
docker run --gpus all \
  -v /tmp:/models \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  gcr.io/$GCP_PROJECT_ID/train:latest \
  training.epochs=10 training.batch_size=64 training.max_length=512

# Download results
gcloud compute scp training-instance:/tmp/trained_model.pt ./trained_model.pt --zone=us-central1-a

# Clean up
gcloud compute instances delete training-instance --zone=us-central1-a
```

### Automated Script

See `scripts/gcp_train.sh` for an automated version.

---

## Option 2: GCP Cloud Run Jobs (Serverless, Easy)

Cloud Run Jobs is perfect for one-off training jobs - no need to manage instances.

### Step 1: Build and Push Image

```bash
export GCP_PROJECT_ID=your-project-id
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/train:latest \
  --file dockerfiles/train.dockerfile
```

### Step 2: Create and Run Job

```bash
# Create a Cloud Run Job
gcloud run jobs create train-job \
  --image=gcr.io/$GCP_PROJECT_ID/train:latest \
  --region=us-central1 \
  --args="training.epochs=10,training.batch_size=64,training.max_length=512" \
  --set-env-vars="WANDB_API_KEY=$WANDB_API_KEY" \
  --memory=16Gi \
  --cpu=4 \
  --task-timeout=3600 \
  --max-retries=1

# Execute the job
gcloud run jobs execute train-job --region=us-central1

# View logs
gcloud run jobs executions logs read --region=us-central1
```

**Note**: Cloud Run Jobs doesn't support GPUs yet. Use Compute Engine or Vertex AI for GPU training.

---

## Option 3: Vertex AI Training (Managed ML Platform)

Vertex AI is GCP's managed ML platform - handles infrastructure automatically.

### Step 1: Prepare Training Code

Your code is already ready! Just need to package it.

### Step 2: Submit Training Job

```bash
# Using gcloud CLI
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=arxiv-classifier-training \
  --python-package-uris=gs://your-bucket/train-package.tar.gz \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,executor-image-uri=gcr.io/$GCP_PROJECT_ID/train:latest
```

### Step 3: Using Python SDK (Easier)

Create `train_vertex_ai.py`:

```python
from google.cloud import aiplatform

aiplatform.init(project="your-project-id", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name="arxiv-classifier-training",
    script_path="src/pname/train.py",
    container_uri=f"gcr.io/your-project-id/train:latest",
    requirements=["torch", "transformers", "wandb", "hydra-core", "loguru"]
)

job.run(
    args=["training.epochs=10", "training.batch_size=64"],
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)
```

---

## Option 4: WandB Launch with GCP

If you want to use WandB Launch with GCP:

### Setup

```bash
# Install WandB Launch
uv add "wandb[launch]"

# Authenticate with GCP
gcloud auth application-default login
```

### Create Launch Config (`wandb_launch_gcp.yaml`)

```yaml
resource: gcp-gke  # or gcp-vertex
resource_args:
  project: your-project-id
  region: us-central1
  machine_type: n1-standard-4
  accelerator_type: NVIDIA_TESLA_T4
  accelerator_count: 1
```

### Launch

```bash
wandb launch src/pname/train.py \
  --config wandb_launch_gcp.yaml \
  training.epochs=10 training.batch_size=64
```

---

## Data Handling on GCP

### Option A: Use DVC with GCS Remote

```bash
# Set up GCS as DVC remote
dvc remote add -d gcs gs://your-bucket/dvc-storage
dvc remote modify gcs projectname your-project-id

# Push data to GCS
dvc push

# On cloud instance, pull data
dvc pull
```

### Option B: Upload Data to GCS

```bash
# Upload processed data
gsutil -m cp -r data/processed gs://your-bucket/data/

# In training script, download:
gsutil -m cp -r gs://your-bucket/data/processed ./
```

### Option C: Mount GCS Bucket (FUSE)

```bash
# Install gcsfuse
# On instance:
gcsfuse your-bucket /mnt/gcs

# Use /mnt/gcs/data in training
```

---

## Cost Estimation

- **n1-standard-4 + T4 GPU**: ~$0.35-0.50/hour
- **n1-standard-8 + T4 GPU**: ~$0.50-0.70/hour
- **Cloud Run Jobs** (CPU only): ~$0.10-0.20/hour
- **Vertex AI**: Similar to Compute Engine + small platform fee

**Tip**: Use preemptible instances to save ~70%:
```bash
gcloud compute instances create ... --preemptible
```

---

## Quick Start Script

I've created `scripts/gcp_train.sh` - run it to automate the process!

```bash
chmod +x scripts/gcp_train.sh
./scripts/gcp_train.sh
```

---

## Recommended Setup

For your use case, I recommend:

1. **Quick experiments**: Cloud Run Jobs (CPU, fast setup)
2. **GPU training**: Compute Engine with GPU (full control)
3. **Production/automation**: Vertex AI (managed, scalable)

Let me know which option you prefer and I can help you set it up!
