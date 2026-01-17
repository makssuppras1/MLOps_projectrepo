# Quick Start: Train on GCP

## Prerequisites Check

1. **Install gcloud CLI** (if not installed):
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Or download from:
   # https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Set your project**:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

4. **Enable required APIs**:
   ```bash
   gcloud services enable compute.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

---

## Option 1: Automated Script (Easiest)

```bash
# Set your project ID
export GCP_PROJECT_ID=your-project-id
export WANDB_API_KEY=your-wandb-key  # Optional but recommended

# Run the automated script
./scripts/gcp_train.sh
```

The script will:
- ✅ Build and push Docker image to GCR
- ✅ Create a GPU instance
- ✅ Run training
- ✅ Download the trained model
- ✅ Clean up (optional)

---

## Option 2: Manual Steps

### Step 1: Build and Push Docker Image

```bash
export GCP_PROJECT_ID=your-project-id

gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/train:latest \
  --file dockerfiles/train.dockerfile
```

### Step 2: Create GPU Instance

```bash
gcloud compute instances create training-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=cos-gpu \
  --image-project=cos-cloud \
  --boot-disk-size=100GB
```

### Step 3: Run Training

```bash
gcloud compute ssh training-instance --zone=us-central1-a --command="
  docker pull gcr.io/$GCP_PROJECT_ID/train:latest
  docker run --gpus all \
    -v /tmp:/models \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    gcr.io/$GCP_PROJECT_ID/train:latest \
    training.epochs=10 training.batch_size=64
"
```

### Step 4: Download Model

```bash
gcloud compute scp training-instance:/tmp/trained_model.pt ./trained_model.pt \
  --zone=us-central1-a
```

### Step 5: Clean Up

```bash
gcloud compute instances delete training-instance --zone=us-central1-a
```

---

## Option 3: Cloud Run Jobs (No GPU, but Simple)

For CPU-only training (slower but easier):

```bash
# Build image
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/train:latest \
  --file dockerfiles/train.dockerfile

# Create and run job
gcloud run jobs create train-job \
  --image=gcr.io/$GCP_PROJECT_ID/train:latest \
  --region=us-central1 \
  --args="training.epochs=10,training.batch_size=32" \
  --set-env-vars="WANDB_API_KEY=$WANDB_API_KEY" \
  --memory=16Gi \
  --cpu=4

gcloud run jobs execute train-job --region=us-central1
```

---

## Cost Saving Tips

1. **Use preemptible instances** (save ~70%):
   ```bash
   gcloud compute instances create ... --preemptible
   ```

2. **Delete instances when done** (don't leave them running)

3. **Use appropriate instance sizes**:
   - Small experiments: `n1-standard-4` + T4
   - Larger models: `n1-standard-8` + T4

---

## Troubleshooting

### "Quota exceeded" error
```bash
# Check your quotas
gcloud compute project-info describe --project=$GCP_PROJECT_ID

# Request quota increase in GCP Console
```

### "Permission denied" errors
```bash
# Make sure you have the right roles
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="user:your-email@example.com" \
  --role="roles/compute.instanceAdmin"
```

### Docker not found on instance
The script automatically installs Docker, but if it fails:
```bash
gcloud compute ssh instance-name --zone=zone --command="curl -fsSL https://get.docker.com | sh"
```

---

## Next Steps

- See `GCP_TRAINING.md` for detailed options
- See `wandb_launch_gcp.yaml` for WandB Launch setup
- Check GCP Console for monitoring: https://console.cloud.google.com
