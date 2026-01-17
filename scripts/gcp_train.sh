#!/bin/bash

# GCP Training Script
# This script automates training on GCP Compute Engine with GPU

set -e

# Configuration - UPDATE THESE
# Defaults to current gcloud project if GCP_PROJECT_ID not set
GCP_PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo 'your-project-id')}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-training-instance-$(date +%s)}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
IMAGE_NAME="gcr.io/${GCP_PROJECT_ID}/train:latest"

# Training parameters
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-512}"

echo "🚀 GCP Training Setup"
echo "===================="
echo "Project: $GCP_PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found. Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: Not authenticated with GCP. Run: gcloud auth login"
    exit 1
fi

# Set project
echo "📋 Setting GCP project..."
gcloud config set project "$GCP_PROJECT_ID"

# Build and push Docker image
echo ""
echo "🐳 Building and pushing Docker image..."
gcloud builds submit --tag "$IMAGE_NAME" \
  --file dockerfiles/train.dockerfile \
  --quiet

echo "✅ Image built and pushed: $IMAGE_NAME"

# Create GPU instance
echo ""
echo "🖥️  Creating GPU instance..."
gcloud compute instances create "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=cos-gpu \
  --image-project=cos-cloud \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --quiet

echo "✅ Instance created: $INSTANCE_NAME"

# Wait for instance to be ready
echo ""
echo "⏳ Waiting for instance to be ready..."
sleep 30

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  Warning: WANDB_API_KEY not set. WandB logging will be disabled."
    WANDB_ENV=""
else
    WANDB_ENV="-e WANDB_API_KEY=$WANDB_API_KEY"
fi

# Run training
echo ""
echo "🎯 Starting training..."
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Max length: $MAX_LENGTH"
echo ""

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        echo 'Installing Docker...'
        curl -fsSL https://get.docker.com | sh
    fi

    # Pull image
    echo 'Pulling training image...'
    docker pull $IMAGE_NAME

    # Run training
    echo 'Starting training...'
    docker run --gpus all \
      -v /tmp:/models \
      $WANDB_ENV \
      $IMAGE_NAME \
      training.epochs=$EPOCHS training.batch_size=$BATCH_SIZE training.max_length=$MAX_LENGTH

    echo 'Training completed!'
"

# Download model
echo ""
echo "📥 Downloading trained model..."
gcloud compute scp "$INSTANCE_NAME:/tmp/trained_model.pt" \
  ./trained_model.pt \
  --zone="$ZONE" \
  --quiet

echo "✅ Model downloaded to: ./trained_model.pt"

# Clean up instance
echo ""
read -p "🗑️  Delete instance $INSTANCE_NAME? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
    echo "✅ Instance deleted"
else
    echo "ℹ️  Instance kept running. Delete manually with:"
    echo "   gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
fi

echo ""
echo "✅ Training complete!"
