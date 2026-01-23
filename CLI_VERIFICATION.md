# CLI Verification Sequence for DTU MLOps Checklist (Week 1-3)

## Placeholders

- `<PYTHON_CMD>` = `uv run python` (or `python` if not using uv)
- `<PROJECT_ID>` = `dtumlops-484310` (from GCP config)
- `<REGION>` = `europe-west1` (from GCP config)
- `<BUCKET>` = `mlops_project_data_bucket1-europe-west1` (DVC remote)
- `<SERVICE_NAME>` = `arxiv-classifier-api` (Cloud Run service)
- `<MONITORING_SERVICE>` = `drift-detection-api` (Cloud Run monitoring service)
- `<IMAGE_NAME>` = `inference-api` (Docker image name)
- `<MODEL_PATH>` = path to trained model (e.g., `artifacts/model-*/trained_model.pkl` or `models/trained_model.pt`)

---

## Week 1

### Item: 1. Git repository created (M5 core)
**Commands:**
- `git remote -v`
- `git log --oneline -1`

**Pass condition:** Shows remote URL and at least one commit

---

### Item: 2. All team members have write access to GitHub repo (M5 core)
**Commands:**
- `gh api repos/:owner/:repo/collaborators --jq '.[].login'` (requires GitHub CLI)
- Alternative: `curl -s -H "Authorization: token $(gh auth token)" https://api.github.com/repos/<OWNER>/<REPO>/collaborators | jq -r '.[].login'`

**Pass condition:** Lists multiple collaborator usernames (exit code 0)

---

### Item: 3. Dedicated environment for packages (M2 core)
**Commands:**
- `test -f pyproject.toml && echo "pyproject.toml exists" || echo "Missing"`
- `test -f uv.lock && echo "uv.lock exists" || echo "Missing"`
- `uv --version`

**Pass condition:** All commands exit code 0

---

### Item: 4. Cookiecutter initial structure (M6 core)
**Commands:**
- `ls -d src/ tests/ configs/ data/ models/ 2>/dev/null | wc -l`
- `test -f src/pname/__init__.py && echo "Package structure OK"`

**Pass condition:** Count >= 4 directories and package init exists

---

### Item: 5. data.py downloads + preprocesses data (M6 core)
**Commands:**
- `uv run python src/pname/data.py download --help`
- `uv run python src/pname/data.py preprocess --help`

**Pass condition:** Both commands show help (exit code 0)

---

### Item: 6. model.py + train.py runs (M6 core)
**Commands:**
- `test -f src/pname/model.py && echo "model.py exists"`
- `test -f src/pname/train.py && echo "train.py exists"`
- `uv run python src/pname/train.py --help 2>&1 | head -5`

**Pass condition:** Files exist and train.py shows help or runs

---

### Item: 7. Dependencies pinned (requirements*.txt or pyproject.toml/uv.lock) (M2+M6 core)
**Commands:**
- `test -f uv.lock && echo "uv.lock exists"`
- `grep -E "^[a-zA-Z]" pyproject.toml | grep -E "==|>=" | head -3`

**Pass condition:** uv.lock exists and pyproject.toml shows version pins

---

### Item: 8. PEP8 compliance (M7 none-core)
**Commands:**
- `uv run ruff check src/ --select E,W --statistics | head -20`

**Pass condition:** Exit code 0 or shows only acceptable warnings

---

### Item: 9. Type hints + essential documentation (M7 none-core)
**Commands:**
- `grep -r "def " src/pname/*.py | head -3 | xargs -I {} grep -A 2 "{}" {} | grep -E "->|:" | head -5`
- `grep -r '"""' src/pname/*.py | wc -l`

**Pass condition:** Shows type hints and docstrings exist (count > 0)

---

### Item: 10. Data version control (DVC etc.) (M8 core)
**Commands:**
- `test -f .dvc/config && echo "DVC config exists"`
- `uv run dvc remote list`
- `test -f data/raw/*.csv.dvc && echo "Data tracked" || test -f data/processed/*.dvc && echo "Processed data tracked"`

**Pass condition:** DVC config exists, remote configured, and .dvc files present

---

### Item: 11. CLI commands/entry points (M9 none-core)
**Commands:**
- `test -f tasks.py && echo "tasks.py exists"`
- `uv run invoke --list`

**Pass condition:** tasks.py exists and shows available tasks

---

### Item: 12. Dockerfile(s) exist (M10 core)
**Commands:**
- `test -f Dockerfile && echo "Main Dockerfile exists"`
- `ls dockerfiles/*.dockerfile 2>/dev/null | wc -l`

**Pass condition:** Main Dockerfile exists and dockerfiles/ contains files

---

### Item: 13. Docker images build & run locally (M10 core)
**Commands:**
- `docker build -f Dockerfile -t test-build . 2>&1 | tail -5`
- `docker images test-build | grep test-build`

**Pass condition:** Build succeeds (exit code 0) and image listed

---

### Item: 14. Experiment config files exist (M11 none-core)
**Commands:**
- `ls configs/experiment/*.yaml 2>/dev/null | wc -l`
- `test -f configs/config.yaml && echo "Main config exists"`

**Pass condition:** Experiment configs exist (count > 0) and main config exists

---

### Item: 15. Hydra used for configs (M11 none-core)
**Commands:**
- `grep -r "hydra" pyproject.toml`
- `grep -r "@hydra\|hydra_main\|ConfigStore" src/ | head -3`

**Pass condition:** Hydra in dependencies and used in code

---

### Item: 16. Profiling exists (M12 none-core)
**Commands:**
- `test -f src/pname/profiler.py && echo "Profiler exists"`
- `grep -E "profiler|snakeviz|torch.profiler" pyproject.toml`

**Pass condition:** Profiler file exists and dependencies present

---

### Item: 17. Logging exists (M14 core)
**Commands:**
- `grep -r "loguru\|logging\|logger" src/pname/*.py | head -5`
- `grep "loguru" pyproject.toml`

**Pass condition:** Logging imports found and loguru in dependencies

---

### Item: 18. Weights & Biases logging exists (M14 core)
**Commands:**
- `grep "wandb" pyproject.toml`
- `grep -r "wandb\|WANDB" src/pname/*.py | head -3`

**Pass condition:** wandb in dependencies and used in code

---

### Item: 19. Hyperparameter sweep exists (M14 core)
**Commands:**
- `grep -r "sweep\|wandb.sweep\|hyperparameter" src/ configs/ | head -5`

**Pass condition:** Sweep configuration or code found

---

### Item: 20. PyTorch Lightning (if applicable) (M15 none-core)
**Commands:**
- `grep -i "lightning\|pytorch_lightning" pyproject.toml || echo "Not using Lightning"`

**Pass condition:** Either Lightning found or "Not using Lightning" (optional item)

---

## Week 2

### Item: 1. Unit tests for data (M16 core)
**Commands:**
- `test -f tests/test_data.py && echo "Data tests exist"`
- `uv run pytest tests/test_data.py -v --tb=short 2>&1 | tail -10`

**Pass condition:** Test file exists and tests pass (exit code 0)

---

### Item: 2. Unit tests for model/training (M16 core)
**Commands:**
- `test -f tests/test_model.py && echo "Model tests exist"`
- `uv run pytest tests/test_model.py -v --tb=short 2>&1 | tail -10`

**Pass condition:** Test file exists and tests pass (exit code 0)

---

### Item: 3. Code coverage (M16 core)
**Commands:**
- `grep "coverage" pyproject.toml`
- `uv run coverage run -m pytest tests/ && uv run coverage report -m | tail -5`

**Pass condition:** Coverage tool in deps and report shows coverage %

---

### Item: 4. CI running on GitHub (M17 core)
**Commands:**
- `test -f .github/workflows/tests.yaml && echo "CI workflow exists"`
- `gh workflow list 2>/dev/null || echo "GitHub CLI not available (check UI)"`

**Pass condition:** Workflow file exists (CLI check) or verify in GitHub UI

---

### Item: 5. CI caching + multi-OS/Python/(framework if relevant) (M17 core)
**Commands:**
- `grep -A 10 "matrix:" .github/workflows/tests.yaml | head -15`
- `grep -E "cache|actions/cache" .github/workflows/tests.yaml`

**Pass condition:** Matrix strategy found and caching configured

---

### Item: 6. Linting in CI (M17 core)
**Commands:**
- `grep -E "ruff|black|lint" .github/workflows/tests.yaml | head -5`

**Pass condition:** Linting step found in workflow

---

### Item: 7. Pre-commit hooks (M18 none-core)
**Commands:**
- `test -f .pre-commit-config.yaml && echo "Pre-commit config exists"`
- `uv run pre-commit --version 2>/dev/null || echo "Pre-commit not installed locally"`

**Pass condition:** Config file exists

---

### Item: 8. Workflow triggers on data changes (M19 none-core)
**Commands:**
- `grep -A 5 "paths:" .github/workflows/cml_data.yaml 2>/dev/null | grep -E "data/|\.dvc" || echo "No data trigger workflow"`

**Pass condition:** Paths include data/ or .dvc files, or workflow doesn't exist (optional)

---

### Item: 9. Workflow triggers on model registry changes (M19 none-core)
**Commands:**
- `grep -A 5 "paths:" .github/workflows/cml_model_registry.yaml 2>/dev/null | grep -E "model|artifact" || echo "No model registry trigger workflow"`

**Pass condition:** Paths include model/artifact files, or workflow doesn't exist (optional)

---

### Item: 10. GCP bucket + DVC remote (M21 core)
**Commands:**
- `gcloud config get-value project`
- `gcloud storage buckets describe gs://<BUCKET> --format="value(name)" 2>&1`
- `uv run dvc remote list`

**Pass condition:** Project set, bucket exists, DVC remote configured

---

### Item: 11. Workflow builds Docker images automatically (M21 core)
**Commands:**
- `grep -E "docker build|gcloud builds|cloudbuild" .github/workflows/*.yaml | head -3`
- `test -f ci/cloudbuild.yaml && echo "Cloud Build config exists"`

**Pass condition:** Build commands in workflows and Cloud Build config exists

---

### Item: 12. Training in GCP (Compute Engine or Vertex AI) (M21 core)
**Commands:**
- `grep -r "vertex\|compute.engine\|gcloud.*train" configs/ .github/workflows/ | head -3`
- `gcloud ai custom-jobs list --region=<REGION> --limit=1 2>&1 | head -5 || echo "No Vertex AI jobs (check manually)"`

**Pass condition:** Vertex AI/Compute configs found or jobs listed

---

### Item: 13. FastAPI inference app exists (M22 core)
**Commands:**
- `test -f app/main.py && echo "FastAPI app exists"`
- `grep "FastAPI\|@app\." app/main.py | head -3`

**Pass condition:** App file exists and FastAPI decorators found

---

### Item: 14. Deployed on GCP (Cloud Run/Functions) (M23 core)
**Commands:**
- `gcloud run services describe <SERVICE_NAME> --region=<REGION> --format="value(status.url)" 2>&1`
- `curl -s -o /dev/null -w "%{http_code}" $(gcloud run services describe <SERVICE_NAME> --region=<REGION> --format="value(status.url)")/health 2>&1 || echo "Health check failed"`

**Pass condition:** Service exists and returns HTTP 200 or 404 (service may be down)

---

### Item: 15. API tests + CI (M24 core)
**Commands:**
- `test -f tests/integrationtests/test_apis.py && echo "API tests exist"`
- `grep -E "test.*api|httpx|requests" tests/integrationtests/test_apis.py | head -3`

**Pass condition:** API test file exists and contains API test code

---

### Item: 16. Load test exists (M24 core)
**Commands:**
- `test -f tests/performancetests/locustfile.py && echo "Load test exists"`
- `grep "locust" pyproject.toml`

**Pass condition:** Locustfile exists and locust in dependencies

---

### Item: 17. Specialized deployment API (ONNX/BentoML) (M25 none-core)
**Commands:**
- `grep -r "onnx\|bentoml" pyproject.toml src/ | head -3 || echo "Not using specialized deployment"`

**Pass condition:** ONNX/BentoML found or "Not using" (optional item)

---

### Item: 18. Frontend for API (M26 none-core)
**Commands:**
- `grep -E "html|frontend|react|vue" app/main.py | head -3 || echo "No frontend"`

**Pass condition:** Frontend code found or "No frontend" (optional item)

---

## Week 3

### Item: 1. Drift robustness check (M27 core)
**Commands:**
- `test -f monitoring/drift_robustness.py && echo "Drift robustness exists"`
- `grep -r "drift_robustness\|run_drift_robustness" monitoring/ | head -3`

**Pass condition:** Drift robustness module exists and contains test function

---

### Item: 2. Input-output collection from deployed app (M27 core)
**Commands:**
- `grep -r "REQUEST_LOG\|api_requests\|collect.*data" app/main.py monitoring/ | head -5`
- `test -f monitoring/api_requests.csv && echo "Request log exists" || echo "Log file not found (may be empty)"`

**Pass condition:** Logging code found in app and monitoring modules

---

### Item: 3. Drift detection API deployed to cloud (M27 core)
**Commands:**
- `gcloud run services describe <MONITORING_SERVICE> --region=<REGION> --format="value(status.url)" 2>&1`
- `curl -s -o /dev/null -w "%{http_code}" $(gcloud run services describe <MONITORING_SERVICE> --region=<REGION> --format="value(status.url)")/health 2>&1 || curl -s $(gcloud run services describe <MONITORING_SERVICE> --region=<REGION> --format="value(status.url)")/docs 2>&1 | head -5`

**Pass condition:** Monitoring service exists and responds (HTTP 200 or docs page loads)

---

## Quick Verification Script

Save as `verify_checklist.sh` and run: `bash verify_checklist.sh`

```bash
#!/bin/bash
# Quick verification script (adjust placeholders first)

PROJECT_ID="dtumlops-484310"
REGION="europe-west1"
BUCKET="mlops_project_data_bucket1-europe-west1"
SERVICE_NAME="arxiv-classifier-api"
MONITORING_SERVICE="drift-detection-api"

echo "=== Week 1 ==="
echo "1. Git repo:" && git remote -v | head -1
echo "3. Environment:" && test -f uv.lock && echo "✓ uv.lock exists"
echo "10. DVC:" && uv run dvc remote list
echo "12. Docker:" && test -f Dockerfile && echo "✓ Dockerfile exists"

echo "=== Week 2 ==="
echo "1. Data tests:" && test -f tests/test_data.py && echo "✓ exists"
echo "10. GCP bucket:" && gcloud storage buckets describe gs://$BUCKET --format="value(name)" 2>&1 | head -1
echo "14. Cloud Run:" && gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)" 2>&1 | head -1

echo "=== Week 3 ==="
echo "1. Drift robustness:" && test -f monitoring/drift_robustness.py && echo "✓ exists"
echo "3. Monitoring API:" && gcloud run services describe $MONITORING_SERVICE --region=$REGION --format="value(status.url)" 2>&1 | head -1
```
