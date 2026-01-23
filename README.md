# ArXiv Scientific Paper Classifier - MLOps Pipeline

## 1. Project Overview

This project implements a complete MLOps pipeline for classifying scientific papers from the ArXiv dataset into research categories. The system uses both transformer-based (DistilBERT) and classical machine learning (TF-IDF + XGBoost) approaches for text classification.

**High-level ML Pipeline:**
1. **Data Collection**: Download ArXiv scientific papers dataset from Kaggle
2. **Data Versioning**: Track data versions using DVC with Google Cloud Storage backend
3. **Preprocessing**: Clean and transform text data into training-ready formats
4. **Training**: Train models locally or on Google Cloud Vertex AI
5. **Evaluation**: Assess model performance on test sets
6. **Deployment**: Serve models via FastAPI on Cloud Run
7. **Monitoring**: Track data drift and model performance in production

The project emphasizes reproducibility, automation, and monitoring throughout the ML lifecycle.

## 2. Prerequisites

### Operating System
- **macOS** (tested on macOS 14+)
- **Linux** (Ubuntu 20.04+)
- **Windows** (Windows 10/11 with WSL2 recommended)

### Required Software

1. **Python 3.12+**
   ```bash
   python --version  # Should show 3.12 or higher
   ```

2. **UV Package Manager** (fast Python package manager)
   ```bash
   # Install UV following: https://docs.astral.sh/uv/getting-started/installation/
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or on macOS:
   brew install uv
   ```

3. **Docker** (for containerized execution)
   ```bash
   docker --version
   # Install from: https://docs.docker.com/get-docker/
   ```

4. **DVC** (Data Version Control - installed via UV)
   - Automatically installed with project dependencies

5. **Git** (version control)
   ```bash
   git --version
   ```

6. **Google Cloud SDK** (for cloud operations - optional for local-only use)
   ```bash
   gcloud --version
   # Install from: https://cloud.google.com/sdk/docs/install
   ```

### Required Accounts

1. **Kaggle Account** (for dataset download)
   - Sign up at: https://www.kaggle.com/
   - Set up Kaggle API credentials:
     ```bash
     # Place kaggle.json in ~/.kaggle/kaggle.json
     # Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables
     ```

2. **Weights & Biases (W&B)** (for experiment tracking - optional)
   - Sign up at: https://wandb.ai/
   - Get API key from: https://wandb.ai/authorize
   - Set environment variable: `export WANDB_API_KEY=your_key_here`

3. **Google Cloud Platform (GCP)** (for cloud training and deployment - optional)
   - Create GCP project
   - Enable required APIs:
     - Vertex AI API
     - Cloud Storage API
     - Artifact Registry API
     - Cloud Build API
     - Cloud Run API
   - Set up service account with appropriate permissions
   - Authenticate: `gcloud auth login`

## 3. Repository Structure

```
MLOps_projectrepo/
├── .github/                  # GitHub Actions workflows (CI/CD)
│   └── workflows/
│       ├── tests.yaml        # Automated testing and linting
│       ├── deploy-api.yaml    # API deployment automation
│       └── cml_*.yaml        # Continuous ML workflows
├── app/                      # FastAPI inference application
│   └── main.py              # API server with /predict endpoint
├── ci/                       # Cloud Build configurations
│   ├── cloudbuild.yaml       # Training image builds
│   └── cloudbuild-api.yaml   # API image builds
├── configs/                  # Hydra configuration files
│   ├── config.yaml          # Main training config
│   ├── config_tfidf.yaml    # TF-IDF model config
│   ├── experiment/          # Experiment-specific configs
│   └── vertex_ai/          # Vertex AI job configurations
├── data/                     # Data directory (DVC-tracked)
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Preprocessed training data
├── dockerfiles/              # Docker image definitions
│   ├── train.dockerfile     # PyTorch training container
│   ├── train_tfidf.dockerfile # TF-IDF training container
│   ├── api.dockerfile       # API deployment container
│   └── evaluate.dockerfile  # Model evaluation container
├── monitoring/               # Data drift detection
│   ├── drift_monitor.py     # Drift detection logic
│   └── drift_robustness.py  # Model robustness testing
├── monitoring_service/        # Separate drift detection API
│   └── app/main.py         # Monitoring API endpoints
├── scripts/                  # Utility scripts
│   ├── download_dataset.sh  # Dataset download automation
│   └── preflight_check.sh   # Pre-flight checks for Vertex AI
├── src/                      # Source code
│   └── pname/               # Main package
│       ├── data.py          # Data download and preprocessing
│       ├── model.py          # DistilBERT model definition
│       ├── model_tfidf.py    # TF-IDF + XGBoost model
│       ├── train.py          # PyTorch training script
│       ├── train_tfidf.py    # TF-IDF training script
│       └── evaluate.py       # Model evaluation
├── tests/                    # Test suite
│   ├── test_data.py         # Data pipeline tests
│   ├── test_model.py        # Model tests
│   ├── test_training.py     # Training tests
│   └── integrationtests/    # API integration tests
├── pyproject.toml            # Project dependencies (UV)
├── uv.lock                   # Locked dependency versions
├── tasks.py                  # Invoke task definitions
└── README.md                 # This file
```

## 4. Environment Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd MLOps_projectrepo
```

### Step 2: Install Dependencies

The project uses **UV** for fast, reproducible dependency management. All dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

```bash
# Install all dependencies (main + dev)
uv sync --all-groups

# Verify installation
uv run python --version  # Should show Python 3.12+
```

**What this installs:**
- Core ML libraries: PyTorch, Transformers, XGBoost, scikit-learn
- MLOps tools: DVC, Hydra, Weights & Biases
- API framework: FastAPI, Uvicorn
- Development tools: pytest, ruff, black, pre-commit

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root (optional, but recommended):

```bash
# .env file
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=pname-arxiv-classifier
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# GCP Configuration (if using cloud features)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GCP_PROJECT_ID=dtumlops-484310
GCP_REGION=europe-west1
```

**Note:** The `.env` file is gitignored. Never commit API keys to the repository.

### Step 4: Verify Setup

```bash
# Check that invoke tasks are available
uv run invoke --list

# Run a quick test
uv run pytest tests/test_data.py -v
```

## 5. Data Setup

### Option A: Download Dataset Manually

The dataset can be downloaded using the provided script:

```bash
# Download ArXiv dataset from Kaggle
uv run sh scripts/download_dataset.sh
```

**Note:** This requires Kaggle API credentials. See [Prerequisites](#2-prerequisites) for setup instructions.

### Option B: Use DVC (Recommended for Reproducibility)

The project uses **DVC** (Data Version Control) to track data versions in Google Cloud Storage. This ensures all team members use the exact same dataset version.

#### First-Time DVC Setup

```bash
# DVC is already configured, but verify remote storage
uv run dvc remote list

# Should show: myremote -> gs://mlops_project_data_bucket1-europe-west1
```

#### Pull Data with DVC

```bash
# Pull the exact same data version used in training
uv run dvc pull

# Verify data was downloaded
ls -la data/raw/
ls -la data/processed/
```

**Expected files:**
- `data/raw/arXiv_scientific dataset.csv` - Raw dataset
- `data/processed/train_texts.json` - Training texts
- `data/processed/train_labels.pt` - Training labels
- `data/processed/test_texts.json` - Test texts
- `data/processed/test_labels.pt` - Test labels
- `data/processed/val_texts.json` - Validation texts (optional)
- `data/processed/val_labels.pt` - Validation labels (optional)
- `data/processed/category_mapping.json` - Category name mappings

#### Troubleshooting DVC

If DVC pull fails:

```bash
# 1. Verify GCP authentication
gcloud auth application-default login

# 2. Check DVC remote configuration
uv run dvc remote list

# 3. Verify GCS bucket access
gsutil ls gs://mlops_project_data_bucket1-europe-west1/

# 4. If cache is corrupted, clear and retry
rm -rf .dvc/cache
uv run dvc pull
```

### Preprocess Data

If you downloaded raw data but haven't preprocessed it:

```bash
# Preprocess raw data into training format
uv run invoke preprocess-data

# Or with automatic download if data is missing
uv run invoke preprocess-data --download
```

This creates the processed files in `data/processed/` that are required for training.

## 6. Training Locally

The project supports two model architectures:

### Option A: PyTorch Model (DistilBERT)

Train the transformer-based model:

```bash
# Train with default configuration
uv run invoke train

# Or directly:
uv run src/pname/train.py

# Train with custom experiment config
uv run src/pname/train.py experiment=fast

# Override specific hyperparameters
uv run src/pname/train.py training.epochs=5 training.batch_size=64 training.learning_rate=0.0001
```

**Training outputs:**
- Model checkpoint: `trained_model.pt` or `models/model.pth`
- Training logs: `outputs/` (Hydra automatically creates timestamped directories)
- W&B metrics: Available in your W&B dashboard (if `WANDB_API_KEY` is set)

### Option B: TF-IDF + XGBoost Model

Train the classical ML pipeline:

```bash
# Train with default TF-IDF config
uv run src/pname/train_tfidf.py

# Train with custom config
uv run src/pname/train_tfidf.py --config-name=config_tfidf training.epochs=10

# Override hyperparameters
uv run src/pname/train_tfidf.py --config-name=config_tfidf \
  model.max_features=10000 \
  model.max_depth=8 \
  training.epochs=20
```

**Training outputs:**
- Model checkpoint: `trained_model.pkl`
- Training metrics logged to W&B (if configured)

### Verify Training

```bash
# Check that model files were created
ls -la trained_model.* models/

# Evaluate the trained model
uv run src/pname/evaluate.py trained_model.pt  # For PyTorch
uv run src/pname/evaluate.py trained_model.pkl  # For TF-IDF
```

## 7. Docker Setup

### Build Docker Images

The project includes multiple Dockerfiles for different components:

```bash
# Build all images for local use (single platform, can be loaded)
uv run invoke docker-build-local

# Build multi-platform images and push to registry (for cloud deployment)
uv run invoke docker-build --push --registry container-registry
```

**Available images:**
- `train:latest` - PyTorch training container
- `train-tfidf:latest` - TF-IDF training container
- `api:latest` - FastAPI inference server
- `evaluate:latest` - Model evaluation container

### Run Training in Docker

```bash
# PyTorch training
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train:latest training.epochs=1

# TF-IDF training
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train-tfidf:latest training.epochs=1
```

**Important:** Ensure data is preprocessed before running training containers. The containers expect processed data files in `/data/processed/`.

## 8. Inference API

### Run API Locally

```bash
# Start API server
uv run invoke api

# Or directly:
uv run app/main.py --host 0.0.0.0 --port 8000

# With specific model
uv run app/main.py --model-path trained_model.pkl
```

The API will be available at `http://localhost:8000`.

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a subset of artificial intelligence."}'

# View monitoring report (if drift detection has been run)
curl http://localhost:8000/monitoring
```

### Run API in Docker

```bash
# Build API image
docker build -f dockerfiles/api.dockerfile -t api:latest .

# Run API container
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models:/models" \
  api:latest
```

### Deploy API to Cloud Run

```bash
# Deploy using invoke (recommended)
uv run invoke deploy-api

# Get the service URL
uv run invoke get-api-url
```

The API is automatically deployed via GitHub Actions when changes are pushed to `main` (see `.github/workflows/deploy-api.yaml`).

## 9. Monitoring and Drift Detection

### Build Reference Dataset

Before running drift detection, create a reference dataset from training data:

```bash
# Build reference dataset for drift comparison
uv run monitoring/drift_monitor.py build-reference

# With custom path
uv run monitoring/drift_monitor.py build-reference \
  --train-texts-path data/processed/train_texts.json
```

### Run Drift Detection

```bash
# Run drift monitoring (compares current data to reference)
uv run monitoring/drift_monitor.py monitor
```

This generates drift reports comparing production data to the training reference.

### View Drift Reports

Drift reports are available via the API:

```bash
# Get drift report HTML
curl http://localhost:8000/monitoring > drift_report.html

# Open in browser
open drift_report.html
```

### Run Monitoring Service

The project includes a separate monitoring service that can be deployed independently:

```bash
# Run monitoring service locally
uv run monitoring_service/app/main.py --host 0.0.0.0 --port 8001

# Test endpoints
curl http://localhost:8001/health
curl -X POST http://localhost:8001/drift/run
```

## 10. CI/CD Workflow

The project uses GitHub Actions for continuous integration and deployment.

### Automated Workflows

1. **Tests Workflow** (`.github/workflows/tests.yaml`)
   - Runs on every push and pull request
   - Tests across multiple OS (Ubuntu, Windows, macOS) and Python versions (3.12, 3.13)
   - Runs linting (Ruff) and formatting checks
   - Calculates code coverage (target: 70%+)
   - Builds Docker images on push to `main`

2. **API Deployment** (`.github/workflows/deploy-api.yaml`)
   - Triggers on changes to `app/`, `dockerfiles/api.dockerfile`, or model code
   - Builds and pushes API Docker image to Artifact Registry
   - Deploys to Cloud Run automatically
   - Comments deployment URL on pull requests

3. **Data Monitoring** (`.github/workflows/cml_data.yaml`)
   - Triggers on data changes (DVC)
   - Computes dataset statistics
   - Posts reports as PR comments

4. **Model Registry Monitoring** (`.github/workflows/cml_model_registry.yaml`)
   - Triggers on model registry changes
   - Validates model artifacts

### Local CI Checks

Before pushing, run CI checks locally:

```bash
# Run linting
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run coverage run -m pytest tests/
uv run coverage report -m
```

### Pre-commit Hooks

Install pre-commit hooks to catch issues before committing:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files
```

## 11. Cloud Training (Vertex AI)

### Prerequisites

1. **GCP Project Setup**
   ```bash
   # Set GCP project
   gcloud config set project dtumlops-484310

   # Set default region
   gcloud config set compute/region europe-west1
   ```

2. **Service Account Permissions**
   - Vertex AI User
   - Storage Object Viewer (for DVC data access)
   - Artifact Registry Writer (for Docker images)

3. **Pre-flight Check**
   ```bash
   # ALWAYS run before submitting jobs
   ./scripts/preflight_check.sh configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml
   ```

### Submit Training Job

```bash
# Submit job with default config
uv run invoke submit-job --config=configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml

# With custom display name
uv run invoke submit-job \
  --config=configs/vertex_ai/vertex_ai_config_tfidf.yaml \
  --display-name=my-training-run
```

### Monitor Job Status

```bash
# Check job status
uv run invoke job-status --job-id=<job_id>

# List recent jobs
uv run invoke list-jobs --limit=10

# Stream logs from running job
uv run invoke stream-logs --job-id=<job_id>
```

### Available Vertex AI Configs

- `vertex_ai_config_cpu.yaml` - CPU-only training (n1-highmem-4)
- `vertex_ai_config_gpu.yaml` - GPU training (n1-standard-4 with T4)
- `vertex_ai_config_tfidf.yaml` - TF-IDF training optimized
- `vertex_ai_config_fast.yaml` - Fast training with preemptible instances

## 12. Testing

### Run All Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test suites
uv run pytest tests/test_data.py -v
uv run pytest tests/test_model.py -v
uv run pytest tests/test_training.py -v
uv run pytest tests/test_tfidf_pipeline.py -v
```

### Integration Tests

```bash
# Start API server first (in another terminal)
uv run invoke api

# Run integration tests
uv run pytest tests/integrationtests/test_apis.py -v
```

### Code Coverage

```bash
# Run tests with coverage
uv run coverage run -m pytest tests/

# Generate coverage report
uv run coverage report -m

# Generate HTML report
uv run coverage html
# Open htmlcov/index.html in browser
```

## 13. Troubleshooting

### Common Issues

**Issue: DVC pull fails**
```bash
# Solution: Verify GCP authentication
gcloud auth application-default login
uv run dvc pull
```

**Issue: Docker build fails on macOS (ARM64)**
```bash
# Solution: Use multi-platform builds or specify platform
docker buildx build --platform linux/amd64 -f dockerfiles/train.dockerfile -t train:latest .
```

**Issue: Training fails with "Out of Memory"**
```bash
# Solution: Reduce batch size or use smaller dataset
uv run src/pname/train.py training.batch_size=8
```

**Issue: API can't find model**
```bash
# Solution: Specify model path explicitly
uv run app/main.py --model-path trained_model.pkl
```

**Issue: Vertex AI job fails**
```bash
# Solution: Always run pre-flight check first
./scripts/preflight_check.sh configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml
```

### Getting Help

- Check existing documentation in `docs/` directory
- Review GitHub Issues
- Check CI/CD logs in GitHub Actions
- Review W&B experiment logs for training issues

## 14. Next Steps

After completing setup:

1. **Train a model locally** to verify the pipeline works
2. **Run tests** to ensure code quality
3. **Start the API** and test predictions
4. **Set up monitoring** to track data drift
5. **Deploy to cloud** for production use

## 15. Additional Resources

- **Project Documentation**: See `docs/` directory for detailed guides
- **Invoke Tasks**: Run `uv run invoke --list` to see all available tasks
- **Configuration Files**: Explore `configs/` for experiment configurations
- **GitHub Actions**: Check `.github/workflows/` for CI/CD automation

---

**Project Status**: Active development
**License**: See `LICENSE` file
**Contributors**: See `CONTRIBUTION_ANALYSIS.md` for individual contributions
