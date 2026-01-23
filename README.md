# Project description

The primary goal of this project is to build a functional MLOps pipeline for a scientific paper classification task. While the underlying machine learning problem is to categorize papers based on citation networks, the "product" of this project is the infrastructure surrounding the model, not the model itself.

We aim to take a transformer model, train it for our text classification purposes and wrap it in professional DevOps practices. The goal is to achieve three core milestones:

- **Reproducibility:** Ensuring anyone can run our code and get the same results.
- **Automation:** Using tools to handle training and testing automatically.
- **Monitoring:** Tracking our experiments so we can see what works without manual note-taking.

**Data Strategy**

We will start with the [arXiv Scientific Research Papers Dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset), which consists of 136,238 scientific papers. Our MLOps focus for this data will be on Data Version Control (DVC). Instead of just having a local folder of data, we will use DVC to track versions of our dataset. This way, if we update the data or add new papers later, we can "roll back" to previous versions just like we do with code in Git.

### Data Versioning with DVC

**Setup DVC (First Time):**
```bash
# Initialize DVC (if not already done)
dvc init
# Configure remote storage (GCS)
dvc remote add -d myremote gs://your-bucket-name/dvc-storage
```

**Track Data Files:**
```bash
# Track raw data
dvc add data/raw/train_texts.json
dvc add data/processed/train_texts.json

# Commit DVC metadata to Git
git add data/raw/train_texts.json.dvc data/processed/train_texts.json.dvc .dvc/config
git commit -m "Track data with DVC"
```

**Push Data to GCS:**
```bash
dvc push
```

**Pull Data (For Teammates - 100% Reproducibility):**
```bash
# Clone the repo
git clone <repo-url>
cd MLOps_projectrepo

# Pull the exact same data version used in training
dvc pull

# Now you have the exact same dataset version, ensuring 100% reproducibility!
```

**The "Flex":** Any teammate can run `dvc pull` and get the exact same dataset version you used for training. This ensures 100% reproducibility - no "it works on my machine" issues with data. DVC tracks data versions just like Git tracks code versions, but stores the actual data in GCS for efficient versioning.

**Models and Tools**

We will be using the DistilBERT base model from hugging face, and training it using our dataset for the purpose of classifying the research article categories based on their summary. We will keep the architecture simple so we can spend our energy on the following MLOps stack:

- **Docker:** We will containerize our code so it runs the same on every team member's laptop, avoiding "it works on my machine" errors.
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
│   └── train.dockerfile
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

**Available training parameters:**

- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 10)

**Training outputs:**

- `models/model.pth`: Saved model checkpoint
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

### Getting Help

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
