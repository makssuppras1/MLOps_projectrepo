# Invoke Commands Reference

Quick reference for common project tasks.

**Note:** All commands require `uv run` prefix:
```bash
uv run invoke <task-name>
```

## GCP / Vertex AI

# Preflight check
uv run invoke preflight-check
uv run invoke preflight-check --config=configs/vertex_ai/vertex_ai_config_tfidf.yaml

# Build and push Docker image
uv run invoke docker-build-and-push-gcp

# Submit training job
uv run invoke submit-job --config=configs/vertex_ai/vertex_ai_config_tfidf.yaml

# Monitor job
uv run invoke stream-logs --job-id=<JOB_ID>
uv run invoke job-status --job-id=<JOB_ID>
uv run invoke list-jobs

## Local Development

# Data and training
uv run invoke preprocess-data
uv run invoke train
uv run invoke evaluate --model-checkpoint=trained_model.pt

# API
uv run invoke api --host=0.0.0.0 --port=8000

# Tests
uv run invoke test

# Documentation
uv run invoke serve-docs

## See All Tasks

uv run invoke --list
uv run invoke --help <task-name>
