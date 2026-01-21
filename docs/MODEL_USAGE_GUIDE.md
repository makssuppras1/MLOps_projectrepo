# Model Usage Guide

## Finding Your Trained Model

After training completes on Vertex AI, the model is saved to `/outputs/trained_model.pt` in the container. However, you need to locate where Vertex AI stored it.

### Option 1: Download from Recent Job

Use the download script to find and download your model:

```bash
# List recent succeeded jobs
python scripts/download_model.py --list-jobs

# Download from a specific job ID
python scripts/download_model.py --job-id <JOB_ID>

# Or download from a known GCS path
python scripts/download_model.py --gcs-path gs://mlops_project_data_bucket1/trained_model.pt
```

### Option 2: Manual GCS Download

If you know the GCS path, download directly:

```bash
gsutil cp gs://<bucket>/<path>/trained_model.pt ./trained_model.pt
```

### Option 3: Check Vertex AI Console

1. Go to [Vertex AI Custom Jobs](https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=dtumlops-484310)
2. Find your completed job
3. Check the "Output directory" or "Artifacts" section
4. Download the model from there

## Using the Model in FastAPI

The FastAPI application now supports loading models from multiple locations:

### Automatic Loading (on startup)

The app automatically checks these locations in order:
1. `/outputs/trained_model.pt` (Docker containers)
2. `/gcs/mlops_project_data_bucket1-europe-west1/trained_model.pt` (GCS mount)
3. `/gcs/mlops_project_data_bucket1/trained_model.pt` (GCS mount)
4. `./trained_model.pt` (project root)
5. GCS paths (if `google-cloud-storage` is installed):
   - `gs://mlops_project_data_bucket1-europe-west1/trained_model.pt`
   - `gs://mlops_project_data_bucket1/trained_model.pt`

### Manual Loading via API

If the model isn't found automatically, load it manually:

```bash
# Load from local path
curl -X POST "http://localhost:8000/load?model_path=./trained_model.pt"

# Load from GCS path (requires google-cloud-storage)
curl -X POST "http://localhost:8000/load?model_path=gs://mlops_project_data_bucket1/trained_model.pt"
```

### Making Predictions

Once the model is loaded, make predictions:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Quantum computing and machine learning applications"}'
```

## Dependencies

Make sure you have the required dependencies:

```bash
uv sync  # This will install google-cloud-storage if not already installed
```

## Troubleshooting

### Model Not Found

If the model isn't found:
1. Check that the model file exists: `ls -lh trained_model.pt`
2. Verify GCS access: `gsutil ls gs://mlops_project_data_bucket1/`
3. Check application logs for loading attempts

### GCS Access Issues

If you get GCS access errors:
1. Authenticate: `gcloud auth application-default login`
2. Verify permissions: `gcloud projects get-iam-policy dtumlops-484310`
3. Check service account has `storage.objectViewer` role

### Model Loading Errors

If model loading fails:
1. Check model file size (should be ~253 MB based on training logs)
2. Verify PyTorch version compatibility
3. Check that the model architecture matches `MyAwesomeModel`

## Next Steps

1. **Download your model** using one of the methods above
2. **Place it in the project root** or specify the path
3. **Start the FastAPI app**: `uvicorn app.main:app --reload`
4. **Test the health endpoint**: `curl http://localhost:8000/health`
5. **Make predictions** using the `/predict` endpoint
