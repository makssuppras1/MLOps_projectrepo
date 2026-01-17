# Cloud Training Guide

This guide explains how to train your model on cloud compute instead of locally.

## Option 1: WandB Launch (Recommended - Easiest)

Since you're already using WandB, this is the simplest option. WandB Launch can run your training on cloud providers automatically.

### Setup

1. **Install WandB Launch** (if not already installed):
   ```bash
   uv add wandb[launch]
   ```

2. **Configure cloud credentials**:
   - For AWS: Set up AWS credentials (`~/.aws/credentials` or environment variables)
   - For GCP: `gcloud auth application-default login`
   - For Azure: `az login`

3. **Create a launch config** (`wandb_launch.yaml`):
   ```yaml
   resource: aws-ec2  # or gcp-gke, azure-aci, kubernetes
   resource_args:
     instance_type: g4dn.xlarge  # GPU instance
     region: us-east-1
   ```

4. **Launch training**:
   ```bash
   wandb launch src/pname/train.py --config wandb_launch.yaml \
     training.epochs=10 training.batch_size=32
   ```

### Alternative: Use WandB Sweeps with Cloud Agents

You already have `configs/sweep.yaml`. You can run sweep agents on cloud:

```bash
# Create sweep
wandb sweep configs/sweep.yaml

# Run agent on cloud (after setting up cloud credentials)
wandb agent <sweep-id> --queue cloud
```

---

## Option 2: Docker + Cloud Compute

Your project already has Dockerfiles. You can run them on any cloud provider.

### AWS EC2 with GPU

1. **Launch an EC2 instance** (e.g., `g4dn.xlarge` with NVIDIA GPU)

2. **Build and push Docker image**:
   ```bash
   # Build locally
   docker build -f dockerfiles/train.dockerfile -t your-ecr-repo/train:latest .

   # Push to ECR (or Docker Hub)
   docker push your-ecr-repo/train:latest
   ```

3. **Run on EC2**:
   ```bash
   # SSH into EC2 instance
   docker pull your-ecr-repo/train:latest
   docker run --gpus all \
     -v $(pwd)/data:/data \
     -v $(pwd)/models:/models \
     -e WANDB_API_KEY=$WANDB_API_KEY \
     your-ecr-repo/train:latest \
     training.epochs=10 training.batch_size=32
   ```

### Google Cloud Platform (GCP)

1. **Use Cloud Run Jobs or Compute Engine**:
   ```bash
   # Build and push to GCR
   gcloud builds submit --tag gcr.io/YOUR_PROJECT/train:latest

   # Run on Cloud Run Jobs
   gcloud run jobs create train-job \
     --image gcr.io/YOUR_PROJECT/train:latest \
     --region us-central1 \
     --args "training.epochs=10,training.batch_size=32"
   ```

### Azure ML

1. **Use Azure ML Compute**:
   ```bash
   az ml compute create --name gpu-cluster --size Standard_NC6s_v3
   az ml job create --file train_job.yaml
   ```

---

## Option 3: Google Colab / Kaggle Notebooks (Free GPU)

For quick experiments with free GPU access:

1. **Upload your code to GitHub** (already done)

2. **Open Google Colab** and create a new notebook:
   ```python
   !git clone https://github.com/makssuppras1/MLOps_projectrepo.git
   %cd MLOps_projectrepo

   !pip install uv
   !uv sync --dev

   # Download data (or use DVC)
   !dvc pull  # if you have DVC remote configured

   # Train
   !uv run src/pname/train.py training.epochs=5 training.batch_size=32
   ```

3. **Enable GPU**: Runtime → Change runtime type → GPU (T4)

---

## Option 4: Managed ML Platforms

### AWS SageMaker

Create a training script wrapper and use SageMaker's training API:

```python
# train_sagemaker.py
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='src/pname/train.py',
    role='your-sagemaker-role',
    instance_type='ml.g4dn.xlarge',
    framework_version='2.0',
    py_version='py312',
    hyperparameters={
        'training.epochs': 10,
        'training.batch_size': 32
    }
)

estimator.fit({'training': 's3://your-bucket/data'})
```

### GCP Vertex AI

```python
from google.cloud import aiplatform

aiplatform.init(project='your-project', location='us-central1')

job = aiplatform.CustomTrainingJob(
    display_name='arxiv-classifier-training',
    script_path='src/pname/train.py',
    container_uri='gcr.io/cloud-aiplatform/training/pytorch-gpu.2-0:latest',
    requirements=['torch', 'transformers', 'wandb']
)

job.run(
    args=['training.epochs=10', 'training.batch_size=32'],
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

---

## Quick Start: WandB Launch (Recommended)

The easiest way to get started:

1. **Install WandB Launch**:
   ```bash
   uv add "wandb[launch]"
   ```

2. **Set up AWS credentials** (or GCP/Azure):
   ```bash
   aws configure
   # Or set environment variables:
   export AWS_ACCESS_KEY_ID=your-key
   export AWS_SECRET_ACCESS_KEY=your-secret
   ```

3. **Create launch config** (`wandb_launch.yaml`):
   ```yaml
   resource: aws-ec2
   resource_args:
     instance_type: g4dn.xlarge  # GPU instance
     region: us-east-1
     ami_id: ami-xxxxx  # Deep Learning AMI (optional)
   build:
     type: docker
     dockerfile_path: dockerfiles/train.dockerfile
   ```

4. **Launch**:
   ```bash
   wandb launch src/pname/train.py \
     --config wandb_launch.yaml \
     training.epochs=10 \
     training.batch_size=64 \
     training.max_length=512
   ```

---

## Cost Considerations

- **WandB Launch**: Pay for cloud compute + small WandB fee
- **AWS EC2 g4dn.xlarge**: ~$0.50-1.00/hour
- **GCP n1-standard-4 + T4**: ~$0.35-0.50/hour
- **Google Colab**: Free (with usage limits)
- **Kaggle**: Free (30 hours/week GPU)

---

## Data Considerations

Your data is tracked with DVC. For cloud training:

1. **Option A**: Pull data on cloud instance
   ```bash
   dvc pull  # On cloud instance
   ```

2. **Option B**: Upload to cloud storage (S3/GCS/Azure Blob)
   ```bash
   # Update data.dvc to point to cloud storage
   dvc remote add -d s3remote s3://your-bucket/data
   dvc push
   ```

3. **Option C**: Mount cloud storage in Docker
   ```bash
   docker run -v /mnt/s3:/data your-image
   ```

---

## Recommended Setup for Your Project

Given you already have:
- ✅ Dockerfiles
- ✅ WandB integration
- ✅ DVC for data

**Best option**: **WandB Launch** - it integrates seamlessly with your existing setup and handles infrastructure automatically.
