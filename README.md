# Project description

The primary goal of this project is to build a functional MLOps pipeline for a scientific paper classification task. While the underlying machine learning problem is to categorize papers based on citation networks, the "product" of this project is the infrastructure surrounding the model, not the model itself.

We aim to take a transformer model, train it for our text classification purposes and wrap it in professional DevOps practices. The goal is to achieve three core milestones:
- **Reproducibility:** Ensuring anyone can run our code and get the same results.
- **Automation:** Using tools to handle training and testing automatically.
- **Monitoring:** Tracking our experiments so we can see what works without manual note-taking.

**Data Strategy**

We will start with the [arXiv Scientific Research Papers Dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset), which consists of 136,238 scientific papers. Our MLOps focus for this data will be on Data Version Control (DVC). Instead of just having a local folder of data, we will use DVC to track versions of our dataset. This way, if we update the data or add new papers later, we can "roll back" to previous versions just like we do with code in Git.

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
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


## Running Scripts

The project includes several scripts for data processing, training, evaluation, and visualization. Scripts can be run directly using `uv run` or via the `invoke` task runner.

### Data Preprocessing

Before training, you need to preprocess the raw data. This script combines the split training files, normalizes the images, and saves processed data.

**Using invoke (recommended):**
```bash
invoke preprocess-data
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
invoke train
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

### Getting Help

To see all available commands and options for each script:
```bash
uv run src/pname/train.py --help
uv run src/pname/evaluate.py --help
uv run src/pname/data.py --help
uv run src/pname/visualize.py --help
```

For invoke tasks:
```bash
invoke --list
invoke --help <task-name>
```

### Example Workflow

1. **Preprocess the data:**
   ```bash
   invoke preprocess-data
   ```

2. **Train the model:**
   ```bash
   invoke train
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
