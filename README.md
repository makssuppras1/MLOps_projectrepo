# Project description

The primary goal of this project is to build a functional MLOps pipeline for a scientific paper classification task. While the underlying machine learning problem is to categorize papers based on citation networks, the "product" of this project is the infrastructure surrounding the model, not the model itself.

We aim to take a transformer model, train it for our text classification purposes and wrap it in cool DevOps practices. The goal is to achieve three core milestones:

- **Reproducibility:** Ensuring anyone can run our code and get the same results.
- **Automation:** Using tools to handle training and testing automatically.
- **Monitoring:** Tracking our experiments so we can see what works without manual note-taking.

**Data Strategy**

We will start with the [arXiv Scientific Research Papers Dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset), which consists of 136,238 scientific papers. Our MLOps focus for this data will be on Data Version Control (DVC). Instead of just having a local folder of data, we will use DVC to track versions of our dataset. This way, if we update the data or add new papers later, we can "roll back" to previous versions just like we do with code in Git.

### Data Versioning with DVC

**Setup DVC (First Time):**
```bash
# Install dependencies (including DVC and dvc-gs for GCS support)
uv sync

# Initialize DVC (if not already done)
uv run dvc init
# Configure remote storage (GCS)
uv run dvc remote add -d myremote gs://mlops_project_data_bucket1-europe-west1
```

**Note:** DVC requires the `dvc-gs` package to work with Google Cloud Storage. This is already included in `pyproject.toml` dependencies.

**Track Data Files:**
```bash
# First, remove old data.dvc if it exists (tracks outdated data structure)
uv run dvc remove data.dvc 2>/dev/null || rm -f data.dvc

# Track raw data
uv run dvc add "data/raw/arXiv_scientific dataset.csv"

# Track processed data files
uv run dvc add data/processed/train_texts.json
uv run dvc add data/processed/train_labels.pt
uv run dvc add data/processed/test_texts.json
uv run dvc add data/processed/test_labels.pt
uv run dvc add data/processed/val_texts.json
uv run dvc add data/processed/val_labels.pt
uv run dvc add data/processed/category_mapping.json

# Note: If you get errors about .pt files being ignored, update .dvcignore
# to exclude model files (models/*.pt) but allow data files (data/**/*.pt)

# Commit DVC metadata to Git
git add "data/raw/arXiv_scientific dataset.csv.dvc" \
  data/processed/train_texts.json.dvc \
  data/processed/train_labels.pt.dvc \
  data/processed/test_texts.json.dvc \
  data/processed/test_labels.pt.dvc \
  data/processed/val_texts.json.dvc \
  data/processed/val_labels.pt.dvc \
  data/processed/category_mapping.json.dvc \
  .dvc/config
git commit -m "Track data with DVC"
```

**Push Data to GCS:**
```bash
uv run dvc push
```

**Troubleshooting DVC Issues:**

If you encounter errors like `ERROR: could not read '...dir'` (cache corruption):

```bash
# 1. Clear corrupted DVC cache and locks
rm -rf .dvc/cache .dvc/tmp/*
mkdir -p .dvc/cache

# 2. If you have an old data.dvc file tracking outdated data, remove it
# (This is safe - it only removes DVC tracking, not the actual data files)
uv run dvc remove data.dvc 2>/dev/null || rm -f data.dvc

# 3. Re-add files (use uv run to access DVC)
uv run dvc add <file>
```

If DVC is not found, ensure it's installed:
```bash
# Install dependencies (including DVC and dvc-gs for GCS support)
uv sync

# Use DVC via uv
uv run dvc --version
```

If you get `ERROR: gs is supported, but requires 'dvc-gs' to be installed`:
```bash
# Install dvc-gs plugin for Google Cloud Storage support
uv sync  # This will install dvc-gs from pyproject.toml
```

**Important:** Always use `uv run dvc` instead of just `dvc` to ensure you're using the project's DVC installation.

**Pull Data (For Teammates - 100% Reproducibility):**
```bash
# Clone the repo
git clone <repo-url>
cd MLOps_projectrepo

# Pull the exact same data version used in training
uv run dvc pull

# Now you have the exact same dataset version, ensuring 100% reproducibility!
```

**The "Flex":** Any teammate can run `dvc pull` and get the exact same dataset version you used for training. This ensures 100% reproducibility - no "it works on my machine" issues with data. DVC tracks data versions just like Git tracks code versions, but stores the actual data in GCS for efficient versioning.

### Development Setup

**Environment Setup:**

```bash
# Install all dependencies (including dev dependencies)
uv sync --dev

# Verify installation
uv run python --version
```

**Pre-commit Hooks:**

This project uses pre-commit hooks to ensure code quality before commits. Install them with:

```bash
# Install pre-commit hooks (runs automatically on git commit)
uv run pre-commit install

# Or run manually on all files
uv run pre-commit run --all-files

# Verify pre-commit hooks work
uv run pre-commit run --all-files
```

The hooks will automatically:
- Remove trailing whitespace
- Fix end-of-file issues
- Check YAML/JSON/TOML syntax
- Format code with `black` and `ruff`
- Sort imports with `isort`
- Run linting checks

**Models and Tools**

We will be using the DistilBERT base model from hugging face, and training it using our dataset for the purpose of classifying the research article categories based on their summary. We will keep the architecture simple so we can spend our energy on the following MLOps stack:

- **Docker:** We will containerize our code so it runs the same on every team member's laptop, avoiding "it works on my machine" errors. See the [Docker Build and Run](#docker-build-and-run) section for detailed instructions.
- **Weights & Biases (WandB):** We will use this to automatically log our loss curves and accuracy, making it easy to compare different training runs.
- **GitHub Actions:** We will implement basic CI (Continuous Integration) to automatically run "linter" checks (like Flake8) to ensure our code stays clean and readable.

By the end of the project, we expect to have a system where a single command can pull the data, build the environment, and train a reliable classification model.

# pname

Nej

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── ci/                       # CI/CD configuration files
│   ├── cloudbuild-api.yaml
│   └── cloudbuild.yaml
├── configs/                  # Configuration files
│   ├── experiment/          # Experiment configs
│   ├── vertex_ai/          # Vertex AI job configs
│   └── gcp/                # GCP-specific configs
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   ├── evaluate.dockerfile
│   ├── train.dockerfile
│   └── train_tfidf.dockerfile
├── Dockerfile                 # Root Dockerfile for API (production)
├── docs/                     # Documentation
│   ├── INVOKE_COMMANDS.md
│   ├── LOGGING_GUIDE.md
│   ├── MODEL_USAGE_GUIDE.md
│   ├── PRE_FLIGHT_CHECKLIST.md
│   ├── VERTEX_AI_TRAINING_GUIDE.md
│   ├── mkdocs.yaml
│   ├── profiling_guide.md
│   └── source/
│       └── index.md
├── scripts/                  # Utility scripts
│   ├── download_dataset.sh
│   └── preflight_check.sh
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports and generated artifacts
│   └── figures/
├── app/                      # FastAPI application
│   ├── __init__.py
│   └── main.py              # API server
├── src/                      # Source code
│   ├── pname/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── model_tfidf.py
│   │   ├── train.py
│   │   ├── train_tfidf.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file (uses uv for dependency management)
├── uv.lock                   # Locked dependencies
├── README.md                 # Project README
└── tasks.py                  # Project tasks (invoke commands)
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Running Scripts

The project includes several scripts for data processing, training, evaluation, and visualization. Scripts can be run directly using `uv run` or via the `invoke` task runner (use `uv run invoke <task>`).

### Download Dataset using with cURL

standing in the root of the repo

```bash
uv run sh scripts/download_dataset.sh
```

### Data Preprocessing

Before training, you need to preprocess the raw data. This script combines the split training files, normalizes the images, and saves processed data.

**Using invoke (recommended):**

```bash
uv run invoke preprocess-data
```

**Direct execution:**

```bash
uv run src/pname/data.py data/raw data/processed
```

This will:

- Load training data from `data/raw/train_images_0.pt` through `train_images_5.pt`
- Load test data from `data/raw/test_images.pt` and `test_target.pt`
- Normalize images using training statistics
- Save processed data to `data/processed/`

### Training the Model

**PyTorch Model (DistilBERT):**

To train the model with default parameters:

```bash
uv run invoke train
```

Or directly:

```bash
uv run src/pname/train.py
```

With custom parameters:

```bash
uv run src/pname/train.py --lr 0.001 --batch-size 64 --epochs 10
```

**TF-IDF + XGBoost Model:**

```bash
# Train TF-IDF model with default config
uv run src/pname/train_tfidf.py

# Train with custom config (using training.epochs - maps to model.n_estimators)
uv run src/pname/train_tfidf.py --config-name=config_tfidf training.epochs=10

# Override config values without editing files (recommended for quick changes)
uv run src/pname/train_tfidf.py --config-name=config_tfidf \
  model.max_features=10000 \
  model.max_depth=8 \
  training.epochs=10

# Or override model.n_estimators directly (number of boosting rounds)
uv run src/pname/train_tfidf.py --config-name=config_tfidf model.n_estimators=10

# Override multiple values at once
uv run src/pname/train_tfidf.py --config-name=config_tfidf \
  model.max_features=10000 \
  model.learning_rate=0.05 \
  model.max_depth=12 \
  training.epochs=20
```

**Note:** The `training.epochs` parameter is supported for compatibility with the PyTorch training script. It maps to `model.n_estimators` (the number of XGBoost boosting rounds). You can use either `training.epochs` or `model.n_estimators` - both work the same way.

**💡 Tip:** You can override any config value via Hydra CLI using `key=value` syntax. This is faster than editing config files and rebuilding Docker images. For example: `model.max_features=10000` overrides the value in `configs/model_tfidf.yaml`.

**Available training parameters (PyTorch):**

- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 10)

**Training outputs:**

- `models/model.pth`: Saved PyTorch model checkpoint
- `trained_model.pkl`: Saved TF-IDF model checkpoint
- `reports/figures/training_statistics.png`: Plot showing training loss and accuracy over time

### Evaluating the Model

To evaluate a trained model:

```bash
uv run src/pname/evaluate.py models/model.pth
```

This will:

- Load the saved model checkpoint
- Evaluate on the test set
- Print the test accuracy

### Visualizing Model Embeddings

To visualize model embeddings using t-SNE:

```bash
uv run src/pname/visualize.py models/model.pth
```

Or with a custom output filename:

```bash
uv run src/pname/visualize.py models/model.pth --figure-name my_embeddings.png
```

This generates a t-SNE visualization of the model's embeddings colored by digit class, saved to `reports/figures/`.

### Running the API Locally

**Start API Server:**

```bash
# Using invoke (recommended)
uv run invoke api

# Direct execution with default settings
uv run app/main.py

# With custom host/port
uv run app/main.py --host 0.0.0.0 --port 8000

# With model path
uv run app/main.py --model-path trained_model.pkl
```

**Test API Endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000/

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a subset of artificial intelligence."}'

# Load a model
curl -X POST "http://localhost:8000/load?model_path=trained_model.pkl"

# View monitoring report (if drift monitoring has been run)
curl http://localhost:8000/monitoring
```

### Testing

**Run All Tests:**

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/test_data.py -v
uv run pytest tests/test_model.py -v
uv run pytest tests/test_training.py -v
uv run pytest tests/test_tfidf_pipeline.py -v
uv run pytest tests/test_tfidf_standalone.py -v
uv run pytest tests/test_backward_compat.py -v
uv run pytest tests/monitoring/test_drift.py -v
```

**Run Integration Tests:**

```bash
# Start API server first (in another terminal)
uv run invoke api

# Then run integration tests
uv run pytest tests/integrationtests/test_apis.py -v
```

**Code Coverage:**

```bash
# Run tests with coverage
uv run coverage run -m pytest tests/

# Generate coverage report
uv run coverage report -m

# Generate HTML coverage report
uv run coverage html
# Open htmlcov/index.html in browser
```

### Code Quality

**Linting:**

```bash
# Run ruff linter
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix
```

**Formatting:**

```bash
# Check formatting
uv run ruff format --check .

# Format code
uv run ruff format .
```

**Verify Code Quality (All Checks):**

```bash
# Run all quality checks (linting + formatting)
uv run ruff check . && uv run ruff format --check .
```

### Data Statistics

**Generate Data Statistics:**

```bash
# Generate statistics about the dataset
uv run src/pname/data_stats.py
```

### Cloud Training on Vertex AI

**⚠️ CRITICAL: Before submitting any Vertex AI job, ALWAYS run:**

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

See `docs/VERTEX_AI_TRAINING_GUIDE.md` for full instructions.

**Submit Vertex AI Job:**

```bash
# Submit training job
uv run invoke submit-job --config=configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml

# Submit with custom display name
uv run invoke submit-job --config=configs/vertex_ai/vertex_ai_config_tfidf.yaml --display-name=my-training-run
```

**Check Job Status:**

```bash
# Check status of a specific job
uv run invoke job-status --job-id=<job_id>

# List recent jobs
uv run invoke list-jobs --limit=10
```

**Stream Job Logs:**

```bash
# Stream logs from a running job
uv run invoke stream-logs --job-id=<job_id>
```

### Docker Build and Run

**⚠️ Important: Path Handling**

If your project path contains spaces (e.g., "02476 Machine Learning Operations"), always use quoted paths in Docker commands:
- ✅ Use: `-v "$(pwd)/data:/data"`
- ❌ Avoid: `-v $(pwd)/data:/data` (will fail with spaces in path)

**⚠️ Important: Data Requirements**

Before running training containers, ensure your data is preprocessed:
- Required files in `data/processed/`: `train_texts.json`, `train_labels.pt`, `test_texts.json`, `test_labels.pt`, `category_mapping.json`
- If missing, run: `uv run invoke preprocess-data --download`
- The Docker container expects data to be mounted at `/data` (mapped from `$(pwd)/data`)

The project includes multiple Dockerfiles for different purposes. You can build and test Docker images locally before deploying to cloud environments.

#### Building Docker Images

**All images are built with multi-platform support (ARM64 + AMD64) by default**, ensuring they work on any machine (ARM64 Macs, AMD64 Linux, GCP, etc.).

**Using invoke (recommended):**

```bash
# Build multi-platform images and push to registry (works on any platform)
uv run invoke docker-build --push --registry container-registry

# Build multi-platform images without pushing (builds but not loadable locally)
uv run invoke docker-build

# Build for local use only (single platform, can be loaded)
uv run invoke docker-build-local
```

**Using build script:**

```bash
# Build multi-platform images and push to registry (recommended)
./scripts/build_docker.sh --push container-registry

# Build multi-platform images without pushing
./scripts/build_docker.sh

# Build for local use only (single platform, can be loaded)
./scripts/build_docker.sh --local
```

**Direct docker buildx commands:**

```bash
# Multi-platform build (works on any machine - ARM64 Mac, AMD64 Linux, GCP)
# Build and push to registry:
docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/train.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484310/container-registry/train:latest --push .

docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/train_tfidf.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484310/container-registry/train-tfidf:latest --push .

docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/api.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484310/container-registry/api:latest --push .

docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/evaluate.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484310/container-registry/evaluate:latest --push .

docker buildx build --platform linux/amd64,linux/arm64 -f monitoring_service/Dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484310/container-registry/monitoring:latest --push .

# For local development (single platform, can be loaded):
# On ARM64 Mac:
docker buildx build --platform linux/arm64 -f dockerfiles/train.dockerfile -t train:latest . --load

# On AMD64 Linux:
docker buildx build --platform linux/amd64 -f dockerfiles/train.dockerfile -t train:latest . --load
```

**Important Notes:**
- **Multi-platform images** (`linux/amd64,linux/arm64`) work on any machine but require `--push` to a registry (cannot use `--load`)
- **Single-platform images** can use `--load` for local development but only work on that platform
- **Default build** creates multi-platform images to ensure compatibility across all machines

#### Running Docker Containers Locally

**⚠️ Note:** All examples use quoted paths `"$(pwd)/..."` to handle directory paths with spaces correctly. If your project is in a path without spaces, the quotes are still safe to use.

**Training Container (PyTorch):**

```bash
# Run training with data mounted from local directory
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train:latest

# On ARM64 Mac, specify platform to suppress warning:
docker run --platform linux/amd64 --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train:latest

# With custom Hydra config override
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train:latest training.epochs=5 training.batch_size=64
```

**Training Container (TF-IDF):**

```bash
# Run TF-IDF training with data mounted from local directory
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train-tfidf:latest

# On ARM64 Mac, specify platform to suppress warning:
docker run --platform linux/amd64 --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train-tfidf:latest

# With custom Hydra config override
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train-tfidf:latest --config-name=config_tfidf training.epochs=1

# Override config values without rebuilding (recommended for quick changes)
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train-tfidf:latest --config-name=config_tfidf \
    model.max_features=10000 \
    training.epochs=1

# On ARM64 Mac with platform flag and config override
docker run --platform linux/amd64 --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train-tfidf:latest --config-name=config_tfidf training.epochs=1

# Override multiple config values at once
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  -v "$(pwd)/models:/models" \
  train-tfidf:latest --config-name=config_tfidf \
    model.max_features=10000 \
    model.max_depth=8 \
    training.epochs=10

# Note: Ensure data/processed/ contains required files before running:
# - train_texts.json, train_labels.pt
# - test_texts.json, test_labels.pt
# - category_mapping.json
# - val_texts.json, val_labels.pt (optional but recommended)
# If missing, run: uv run invoke preprocess-data --download
```

**API Container:**

```bash
# Run API server (exposes port 8000)
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/models:/models" \
  api:latest

# Test the API
curl http://localhost:8000/health
```

**Evaluation Container:**

```bash
# Run evaluation with model and data mounted
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/models:/models" \
  -v "$(pwd)/reports:/reports" \
  evaluate:latest /models/model.pth
```

#### Testing Docker Images Locally

**1. Verify image builds successfully:**

```bash
# Check that images exist
docker images | grep train
docker images | grep api
```

**2. Test training container:**

```bash
# Quick test run (will fail if data is missing, but confirms container works)
docker run --rm train:latest --help

# On ARM64 Mac, specify platform to suppress warning:
docker run --platform linux/amd64 --rm train:latest --help

# Quick test for TF-IDF training container
docker run --rm train-tfidf:latest --help

# On ARM64 Mac, specify platform to suppress warning:
docker run --platform linux/amd64 --rm train-tfidf:latest --help

# Full test with data (PyTorch)
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train:latest training.epochs=1 training.batch_size=8

# Full test with data (TF-IDF)
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest training.epochs=1

# On ARM64 Mac with platform flag (TF-IDF)
docker run --platform linux/amd64 --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest training.epochs=1
```

**3. Test API container:**

```bash
# Start API in background
docker run -d --name test-api -p 8000:8000 api:latest

# Check if it's running
docker ps | grep test-api

# Test health endpoint
curl http://localhost:8000/health

# View logs
docker logs test-api

# Stop and remove container
docker stop test-api && docker rm test-api
```

**4. Verify data mounts:**

```bash
# Check that data is accessible inside container
docker run --rm \
  -v "$(pwd)/data:/data" \
  train:latest ls -la /data

# Check that outputs directory is writable
docker run --rm \
  -v "$(pwd)/outputs:/outputs" \
  train:latest touch /outputs/test.txt && ls -la /outputs
```

**5. Test with minimal data (smoke test):**

```bash
# Create minimal test data structure
mkdir -p test_data/processed
echo '{"test": "data"}' > test_data/processed/test.json

# Run PyTorch container with test data
docker run --rm \
  -v "$(pwd)/test_data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train:latest training.epochs=1

# Run TF-IDF container with test data
docker run --rm \
  -v "$(pwd)/test_data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest training.epochs=1
```

#### Docker Troubleshooting

**Common Issues:**

- **Container exits with ExitCode=137** → Out of Memory (OOM). See `docs/OOM_FIX.md` for solutions.
- **Container exits with ExitCode=139** → Platform mismatch (segfault). See `PLATFORM_FIX_SUMMARY.md` for solutions.
- **Container can't find data** → Check mount paths and file permissions.
- **Training hangs** → Check if it's actually running (XGBoost can be slow) or if container was killed.

**Issue: Docker misinterprets path with spaces**

If you see errors like `Unable to find image 'semester/02476:latest'` or `pull access denied for semester/02476`, Docker is misinterpreting your path due to spaces.

```bash
# Problem: Path with spaces causes Docker to misinterpret arguments
# Error: Unable to find image 'semester/02476:latest'

# Solution 1: Always quote paths (recommended)
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest --help

# Solution 2: Use absolute path variable
PROJECT_ROOT="/Users/maks/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/11. semester/02476 Machine Learning Operations/MLOps_projectrepo"
docker run --rm \
  -v "${PROJECT_ROOT}/data:/data" \
  -v "${PROJECT_ROOT}/outputs:/outputs" \
  train-tfidf:latest --help

# Solution 3: Use cd to project root first, then use relative paths
cd "/Users/maks/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/11. semester/02476 Machine Learning Operations/MLOps_projectrepo"
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest --help
```

**Issue: Permission denied when writing to mounted volumes**

```bash
# Fix: Ensure directories exist and have correct permissions
mkdir -p outputs models reports
chmod 755 outputs models reports
```

**Issue: Platform mismatch errors on macOS**

**Solution 1: Multi-platform builds (recommended)**
```bash
# Build for both ARM64 (Mac) and AMD64 (GCP) - Docker auto-selects native architecture
docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/train.dockerfile -t train:latest --load .
docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest --load .

# For registry push (true multi-platform):
docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/train.dockerfile -t train:latest --push .
```

**Solution 2: Single platform builds**
```bash
# For GCP deployment (AMD64 only):
docker buildx build --platform linux/amd64 -f dockerfiles/train.dockerfile -t train:latest --load .
docker buildx build --platform linux/amd64 -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest --load .

# For local testing on Mac (ARM64 only, faster):
docker buildx build --platform linux/arm64 -f dockerfiles/train.dockerfile -t train:latest --load .
docker buildx build --platform linux/arm64 -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest --load .

# Alternative using build arg:
docker build --build-arg TARGETPLATFORM=linux/amd64 -f dockerfiles/train.dockerfile -t train:latest .
```

**Note:** When running AMD64 images on ARM64 Mac, Docker uses emulation which can be slower and may cause issues with native code (e.g., XGBoost). Building natively for ARM64 avoids these issues.
docker run --platform linux/amd64 --rm train-tfidf:latest --help
```

**Issue: Need to update config without rebuilding Docker image**

If you changed a config file but don't want to rebuild the entire Docker image:

```bash
# Solution 1: Override config via Hydra CLI (recommended - no rebuild needed)
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest --config-name=config_tfidf \
    model.max_features=10000 \
    training.epochs=1

# Solution 2: Fast rebuild (Docker uses layer caching - only rebuilds from configs/ onwards)
# This is still fast because Docker caches layers before COPY configs/
docker build -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest .

# Solution 3: Mount configs as volume (for development only - requires Dockerfile change)
# This allows live config updates without rebuilds, but is not recommended for production
```

**Issue: Training hangs or appears to stop during XGBoost training**

If training starts but seems to hang after "Starting training...":

```bash
# 0. First check if container was killed (OOM or crash)
# Run without --rm to inspect exit status:
docker run --name tfidf_check \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest training.epochs=1
docker inspect tfidf_check --format 'ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}}'
# ExitCode=137 + OOMKilled=true = Out of Memory (see OOM section above)
# ExitCode=139 = Segfault (platform mismatch - see PLATFORM_FIX_SUMMARY.md)
docker rm tfidf_check

# 1. Check if it's actually running (XGBoost can be slow with large datasets)
# The training might be running but not showing output. Wait a few minutes.
# Monitor in another terminal: docker stats

# 2. Use smaller dataset for testing (recommended for debugging)
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest --config-name=config_tfidf \
    training.epochs=1 \
    training.max_samples=1000 \
    model.max_features=1000

# 3. Check Docker container logs in real-time
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest --config-name=config_tfidf \
    training.epochs=1 \
    training.max_samples=5000 2>&1 | tee training.log

# 4. Monitor Docker container resource usage (in another terminal)
docker stats

# 5. Check if it's a memory issue - reduce features and samples
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest --config-name=config_tfidf \
    training.epochs=10 \
    training.max_samples=5000 \
    model.max_features=5000 \
    model.n_estimators=50

# 6. Run with Python unbuffered output to see progress immediately
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest python -u src/pname/train_tfidf.py \
    --config-name=config_tfidf \
    training.epochs=1 \
    training.max_samples=1000
```

**Note:** Training 82k samples with 10k features can take 5-15 minutes. The vectorization step alone can take 1-2 minutes. Be patient or use `training.max_samples` to test with smaller datasets first.

**Issue: Training fails with errors (XGBoost/TF-IDF)**

If training starts but fails with an error:

**Issue: Container can't find data files**

```bash
# Fix: Verify mount paths match Dockerfile expectations
# Training expects: /data (mapped from $(pwd)/data)
docker run --rm -v "$(pwd)/data:/data" train:latest ls /data
docker run --rm -v "$(pwd)/data:/data" train-tfidf:latest ls /data
```

**Issue: Container killed with ExitCode=137 (Out Of Memory)**

If your container exits with `ExitCode=137` and `OOMKilled=true`, the container ran out of memory:

```bash
# Check container exit status
docker inspect <container_name> --format 'ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}}'
# ExitCode=137 + OOMKilled=true = Out of Memory

# Solution 1: Increase Docker Desktop memory (recommended)
# Docker Desktop → Settings → Resources → Memory → Increase to 8GB+ → Apply & Restart

# Solution 2: Reduce memory usage by limiting samples
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest \
    training.epochs=1 \
    training.max_samples=20000  # Use 20k instead of 82k samples

# Solution 3: Reduce features
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest \
    training.epochs=1 \
    model.max_features=5000  # Use 5k instead of 10k features

# Memory requirements:
# - 82k samples × 10k features ≈ 5-6GB RAM minimum
# - 20k samples × 10k features ≈ 2GB RAM
# - 20k samples × 5k features ≈ 1GB RAM
```

See `docs/OOM_FIX.md` for detailed memory troubleshooting guide.

**Issue: Training fails after data loading**

If training starts but fails during model training (e.g., after "Starting training..." message):

```bash
# 1. Check if data files exist and are accessible
docker run --rm \
  -v "$(pwd)/data:/data" \
  train-tfidf:latest ls -la /data/processed/

# 2. Verify required data files exist
# Should see: train_texts.json, train_labels.pt, test_texts.json, test_labels.pt, category_mapping.json
docker run --rm \
  -v "$(pwd)/data:/data" \
  train-tfidf:latest sh -c "ls -la /data/processed/*.json /data/processed/*.pt 2>/dev/null || echo 'Files missing'"

# 3. Check if data needs to be preprocessed first
# Run preprocessing if data/processed/ is empty or missing files
uv run invoke preprocess-data --download

# 4. Check container exit status for OOM (ExitCode=137) or segfault (ExitCode=139)
# Run without --rm to inspect after crash:
docker run --name tfidf_debug \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest training.epochs=1
docker inspect tfidf_debug --format 'ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}}'
docker rm tfidf_debug

# 5. For XGBoost errors, check system dependencies (libomp on Mac)
# If you see libxgboost errors, install: brew install libomp

# 6. Run with more verbose logging to see the actual error
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest --config-name=config_tfidf training.epochs=1 2>&1 | tee training.log
```

**Issue: Out of disk space**

```bash
# Clean up unused Docker resources
docker system prune -a --volumes
```

#### Available Docker Images

| Image | Dockerfile | Purpose | Entrypoint |
|-------|-----------|---------|------------|
| `train:latest` | `dockerfiles/train.dockerfile` | PyTorch model training (DistilBERT) | `uv run src/pname/train.py` |
| `train-tfidf:latest` | `dockerfiles/train_tfidf.dockerfile` | TF-IDF + XGBoost model training | `uv run src/pname/train_tfidf.py` |
| `api:latest` | `dockerfiles/api.dockerfile` | FastAPI server | `uv run uvicorn app.main:app` |
| `evaluate:latest` | `dockerfiles/evaluate.dockerfile` | Model evaluation | `uv run src/pname/evaluate.py` |

**Note:** The root `Dockerfile` is for production API deployment and uses a different base image (Python 3.11 slim).

### Cloud Run Deployment

The API can be deployed to Google Cloud Run for production use. Deployment is automated via GitHub Actions when changes are pushed to `main`, or can be done manually.

#### Automated Deployment (GitHub Actions)

When you push changes to `main` that affect the API (`app/`, `dockerfiles/api.dockerfile`, `src/pname/model*.py`), the `.github/workflows/deploy-api.yaml` workflow will:

1. Build the API Docker image
2. Push it to Artifact Registry
3. Deploy to Cloud Run
4. Output the service URL

Check the Actions tab in GitHub to see deployment status and get the service URL.

#### Manual Deployment

**Using invoke (recommended):**

```bash
# Deploy API to Cloud Run
uv run invoke deploy-api

# Get the service URL after deployment
uv run invoke get-api-url
```

**Direct gcloud commands:**

```bash
# 1. Build and push Docker image
gcloud builds submit . \
  --config ci/cloudbuild-api.yaml \
  --substitutions=_IMAGE_NAME=inference-api:latest

# 2. Deploy to Cloud Run
gcloud run deploy arxiv-classifier-api \
  --image europe-west1-docker.pkg.dev/PROJECT_ID/container-registry/inference-api:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2

# 3. Get service URL
gcloud run services describe arxiv-classifier-api \
  --region europe-west1 \
  --format='value(status.url)'
```

#### Testing Deployed API

Once deployed, test the API:

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe arxiv-classifier-api \
  --region europe-west1 \
  --format='value(status.url)')

# Health check
curl $SERVICE_URL/health

# Make a prediction
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a subset of artificial intelligence."}'
```

#### Deployment Configuration

- **Service name:** `arxiv-classifier-api`
- **Region:** `europe-west1`
- **Memory:** 2Gi
- **CPU:** 2
- **Port:** 8000
- **Authentication:** Unauthenticated (public access)
- **Auto-scaling:** 0-10 instances

To customize deployment settings, edit `.github/workflows/deploy-api.yaml` or modify the `deploy_api` task in `tasks.py`.

#### Monitoring Service Deployment

**Manual Deployment:**

```bash
# Build and push monitoring service image
gcloud builds submit . \
  --config ci/cloudbuild-monitoring.yaml \
  --substitutions=_IMAGE_NAME=drift-detection-api:latest

# Deploy to Cloud Run
gcloud run deploy drift-detection-api \
  --image europe-west1-docker.pkg.dev/PROJECT_ID/container-registry/drift-detection-api:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8001 \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "GCS_BUCKET=mlops_project_data_bucket1-europe-west1,GCS_ARTIFACT_PREFIX=monitoring/drift_artifacts"

# Get service URL
gcloud run services describe drift-detection-api \
  --region europe-west1 \
  --format='value(status.url)'
```

**Test Monitoring Service:**

```bash
# Health check
curl $MONITORING_URL/health

# Run drift detection
curl -X POST $MONITORING_URL/drift/run

# Get drift report location
curl $MONITORING_URL/drift/report
```

### Load Testing

**Run Load Tests with Locust:**

```bash
# 1. Start API server first (in a separate terminal)
uv run invoke api

# 2. Run Locust load tests
uv run locust -f tests/performancetests/locustfile.py --host=http://localhost:8000

# 3. Open http://localhost:8089 in your browser
#    - Set number of users (e.g., 10)
#    - Set spawn rate (e.g., 2 users/second)
#    - Click "Start Swarming"
```

**Alternative: Use environment variable:**

```bash
export MYENDPOINT=http://localhost:8000
uv run locust -f tests/performancetests/locustfile.py
```

### Monitoring and Drift Detection

**Build Reference Dataset:**

```bash
# Build reference dataset from training data
uv run monitoring/drift_monitor.py build-reference

# With custom training data path
uv run monitoring/drift_monitor.py build-reference --train-texts-path data/processed/train_texts.json
```

**Run Drift Monitoring:**

```bash
# Run drift detection (requires reference data to be built first)
uv run monitoring/drift_monitor.py monitor
```

**Drift Robustness Testing:**

```bash
# Test model robustness to data drift
uv run monitoring/drift_robustness.py test trained_model.pkl

# With custom options
uv run monitoring/drift_robustness.py test trained_model.pkl \
  --model-type tfidf \
  --val-set-size 500 \
  --output-path monitoring/drift_robustness_report.json
```

**Run Monitoring Service Locally:**

```bash
# Start monitoring service API
uv run monitoring_service/app/main.py --host 0.0.0.0 --port 8001

# Test endpoints
curl http://localhost:8001/health
curl -X POST http://localhost:8001/drift/run
```

### Documentation

**Build Documentation:**

```bash
# Build static documentation
uv run invoke build-docs

# Documentation will be in build/ directory
```

**Serve Documentation Locally:**

```bash
# Start documentation server (auto-reload on changes)
uv run invoke serve-docs

# Open http://127.0.0.1:8000 in your browser
```

To see all available commands and options for each script:

```bash
uv run src/pname/train.py --help
uv run src/pname/evaluate.py --help
uv run src/pname/data.py --help
uv run src/pname/visualize.py --help
```

For invoke tasks (requires `uv run` prefix):

```bash
uv run invoke --list
uv run invoke --help <task-name>
```

### Example Workflow

1. **Preprocess the data:**

   ```bash
   uv run invoke preprocess-data
   ```

2. **Train the model:**

   ```bash
   uv run invoke train
   ```

   Or with custom parameters:

   ```bash
   uv run src/pname/train.py --epochs 10 --batch-size 64
   ```

3. **Evaluate the trained model:**

   ```bash
   uv run src/pname/evaluate.py models/model.pth
   ```

4. **Visualize embeddings:**

   ```bash
   uv run src/pname/visualize.py models/model.pth
   ```

5. **Check the outputs:**
   - Open `reports/figures/training_statistics.png` to view the training curves
   - Open `reports/figures/embeddings.png` to view the t-SNE visualization

## Reviewer Simulation: End-to-End Verification

This section provides the exact commands a reviewer would run to verify every component of the project works correctly. Execute these commands in order on a clean machine.

### Prerequisites Verification

```bash
# 1. Verify Python version
python --version  # Should be 3.12+

# 2. Verify uv is installed
uv --version

# 3. Verify Docker is installed
docker --version

# 4. Verify Git is installed
git --version
```

### Environment Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd MLOps_projectrepo

# 2. Install dependencies
uv sync --dev

# 3. Verify installation
uv run python --version
```

### Code Quality Verification

```bash
# 1. Run linting
uv run ruff check .

# 2. Check formatting
uv run ruff format --check .

# 3. Run pre-commit hooks
uv run pre-commit run --all-files
```

### Data Pipeline Verification

```bash
# 1. Download dataset (if not using DVC)
uv run sh scripts/download_dataset.sh

# 2. Pull data with DVC (if using DVC)
dvc pull

# 3. Preprocess data
uv run invoke preprocess-data

# 4. Verify processed data exists
ls -la data/processed/
```

### Training Verification

```bash
# 1. Train PyTorch model (quick test with 1 epoch)
uv run src/pname/train.py training.epochs=1 training.batch_size=8

# 2. Train TF-IDF model (quick test)
uv run src/pname/train_tfidf.py training.epochs=1

# 3. Verify models were created
ls -la trained_model.* models/
```

### Evaluation Verification

```bash
# 1. Evaluate PyTorch model
uv run src/pname/evaluate.py trained_model.pt

# 2. Evaluate TF-IDF model (if .pkl exists)
uv run src/pname/evaluate.py trained_model.pkl
```

### Testing Verification

```bash
# 1. Run all unit tests
uv run pytest tests/ -v

# 2. Run specific test suites
uv run pytest tests/test_data.py -v
uv run pytest tests/test_model.py -v
uv run pytest tests/test_training.py -v
uv run pytest tests/test_tfidf_pipeline.py -v
uv run pytest tests/test_backward_compat.py -v
uv run pytest tests/monitoring/test_drift.py -v

# 3. Run coverage
uv run coverage run -m pytest tests/
uv run coverage report -m
```

### API Verification

```bash
# 1. Start API server (in background or separate terminal)
uv run invoke api &
API_PID=$!

# 2. Wait for server to start
sleep 5

# 3. Test health endpoint
curl http://localhost:8000/health

# 4. Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a subset of artificial intelligence."}'

# 5. Run integration tests
uv run pytest tests/integrationtests/test_apis.py -v

# 6. Stop API server
kill $API_PID
```

### Docker Verification

```bash
# 1. Build Docker images
uv run invoke docker-build

# 2. Verify images exist
docker images | grep -E "(train|api|evaluate)"

# 3. Test PyTorch training container (quick smoke test)
docker run --rm train:latest --help

# 4. Test TF-IDF training container (quick smoke test)
docker run --rm train-tfidf:latest --help

# 5. Test API container
docker run -d --name test-api -p 8000:8000 api:latest
sleep 5
curl http://localhost:8000/health
docker stop test-api && docker rm test-api
```

### Monitoring Verification

```bash
# 1. Build reference dataset for drift monitoring
uv run monitoring/drift_monitor.py build-reference

# 2. Run drift monitoring
uv run monitoring/drift_monitor.py monitor

# 3. Verify drift report exists
ls -la monitoring/drift_report.html

# 4. Test drift robustness (if model exists)
uv run monitoring/drift_robustness.py test trained_model.pkl --val-set-size 100
```

### Documentation Verification

```bash
# 1. Build documentation
uv run invoke build-docs

# 2. Verify documentation was built
ls -la build/

# 3. Test documentation server (optional)
# uv run invoke serve-docs
# Then visit http://127.0.0.1:8000
```

### Load Testing Verification (Optional)

```bash
# 1. Start API server
uv run invoke api &
API_PID=$!
sleep 5

# 2. Run Locust (in another terminal or background)
uv run locust -f tests/performancetests/locustfile.py --host=http://localhost:8000 --headless -u 5 -r 1 -t 30s

# 3. Stop API server
kill $API_PID
```

### Summary

After running all verification commands above, you should have verified:

✅ Environment setup works
✅ Code quality checks pass
✅ Data pipeline works
✅ Training works (both PyTorch and TF-IDF)
✅ Evaluation works
✅ All tests pass
✅ API works locally
✅ Docker images build and run
✅ Monitoring/drift detection works
✅ Documentation builds
✅ Load testing works

**Expected Duration:** ~30-60 minutes depending on dataset size and training epochs.
