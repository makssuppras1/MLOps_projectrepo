# Monitoring Service

This directory contains the drift detection API service that can be deployed separately to Cloud Run.

## Structure

- `app/main.py`: FastAPI application with drift detection endpoints
- `Dockerfile`: Container image definition
- `.github/workflows/deploy-monitoring.yaml`: GitHub Actions deployment workflow

## Endpoints

- `GET /`: Health check
- `GET /health`: Detailed health status
- `POST /drift/run`: Run drift detection and return JSON results
- `GET /drift/report`: Get latest drift report location

## Deployment

The service is automatically deployed to Cloud Run when changes are pushed to `main` branch.

Manual deployment:
```bash
gcloud run deploy drift-detection-api \
  --image europe-west1-docker.pkg.dev/PROJECT_ID/container-registry/drift-detection-api:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8001
```

## Environment Variables

- `GCS_BUCKET`: GCS bucket for artifact storage (default: `mlops_project_data_bucket1-europe-west1`)
- `GCS_ARTIFACT_PREFIX`: Prefix for drift artifacts (default: `monitoring/drift_artifacts`)
