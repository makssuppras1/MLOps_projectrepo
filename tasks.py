import os
import re
import subprocess
import tempfile
from datetime import datetime

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "pname"
PYTHON_VERSION = "3.12"


# Project commands
@task
def download_data(ctx: Context, force: bool = False) -> None:
    """Download ArXiv dataset."""
    force_flag = " --force" if force else ""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py download{force_flag}", echo=True, pty=not WINDOWS)


@task
def preprocess_data(ctx: Context, download: bool = False) -> None:
    """Preprocess data. Use --download to automatically download if data is missing."""
    download_flag = " --download" if download else ""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data.py preprocess data/raw data/processed{download_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, model_checkpoint: str = "trained_model.pt") -> None:
    """Evaluate model."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py {model_checkpoint}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def visualize(ctx: Context, model_checkpoint: str = "trained_model.pt") -> None:
    """Visualize model embeddings."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/visualize.py {model_checkpoint}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def api(ctx: Context, host: str = "0.0.0.0", port: int = 8000, model_path: str = "") -> None:
    """Run API server."""
    cmd = f"uv run app/main.py --host {host} --port {port}"
    if model_path:
        cmd += f" --model-path {model_path}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t train-tfidf:latest . -f dockerfiles/train_tfidf.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# GCP / Vertex AI commands
@task
def preflight_check(ctx: Context, config: str = "configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml") -> None:
    """Run preflight check before submitting Vertex AI job."""
    ctx.run(f"./scripts/preflight_check.sh {config}", echo=True, pty=not WINDOWS)


@task
def docker_build_gcp(
    ctx: Context,
    dockerfile: str = "dockerfiles/train.dockerfile",
    tag: str = "train:latest",
    multi_platform: bool = True,
) -> None:
    """Build Docker image with multi-platform support (ARM64 + AMD64) by default.

    Args:
        dockerfile: Path to Dockerfile
        tag: Image tag
        multi_platform: If True, build for both linux/amd64 and linux/arm64 (default: True)
    """
    if multi_platform:
        # Multi-platform build (recommended - works on both ARM64 Mac and AMD64 GCP)
        # Note: --load only works with single platform, so we build for native platform
        import platform

        native_arch = "linux/arm64" if platform.machine() == "arm64" else "linux/amd64"
        ctx.run(
            f"docker buildx build --platform {native_arch} -f {dockerfile} -t {tag} . --load",
            echo=True,
            pty=not WINDOWS,
        )
        print(f"\n✓ Built {tag} for {native_arch} (native architecture)")
        print("  For multi-platform build with push to registry, use:")
        print(f"  docker buildx build --platform linux/amd64,linux/arm64 -f {dockerfile} -t {tag} . --push")
    else:
        # Single platform build (AMD64 only, for GCP)
        ctx.run(
            f"docker buildx build --platform linux/amd64 -f {dockerfile} -t {tag} . --load",
            echo=True,
            pty=not WINDOWS,
        )


@task
def docker_push_gcp(
    ctx: Context,
    tag: str = "train:latest",
    region: str = "europe-west1",
    project_id: str = "dtumlops-484310",
    registry: str = "container-registry",
) -> None:
    """Push Docker image to GCP Artifact Registry."""
    image_uri = f"{region}-docker.pkg.dev/{project_id}/{registry}/{tag}"
    ctx.run(f"docker tag {tag} {image_uri}", echo=True, pty=not WINDOWS)
    ctx.run(f"docker push {image_uri}", echo=True, pty=not WINDOWS)


@task
def docker_build_and_push_gcp(
    ctx: Context,
    dockerfile: str = "dockerfiles/train.dockerfile",
    tag: str = "train:latest",
    region: str = "europe-west1",
    project_id: str = "dtumlops-484310",
    registry: str = "container-registry",
) -> None:
    """Build and push Docker image to GCP in one command."""
    docker_build_gcp(ctx, dockerfile, tag)
    docker_push_gcp(ctx, tag, region, project_id, registry)


@task
def submit_job(
    ctx: Context,
    config: str = "configs/vertex_ai/vertex_ai_config_balanced_cpu.yaml",
    region: str = "europe-west1",
    display_name: str = "",
) -> str:
    """Submit a Vertex AI training job.

    Returns:
        Job ID (full resource path) if successful, empty string otherwise.
    """
    # Get WANDB key
    result = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest", "--secret=WANDB_API_KEY"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        ctx.run("echo 'Error: Failed to get WANDB_API_KEY from secrets'", echo=True)
        return ""

    wandb_key = result.stdout.strip()

    # Create temp config with substituted key
    with open(config, "r") as f:
        config_content = f.read().replace("${WANDB_API_KEY}", wandb_key)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(config_content)
        tmp_path = tmp.name

    try:
        if not display_name:
            timestamp = int(datetime.now().timestamp())
            config_basename = os.path.basename(config).replace(".yaml", "").replace("vertex_ai_config_", "")
            display_name = f"{config_basename}-{timestamp}"

        cmd = f"gcloud ai custom-jobs create --region={region} " f"--display-name={display_name} --config={tmp_path}"
        # Capture output to extract job ID
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            return ""

        output = result.stdout + result.stderr  # Job ID might be in either
        print(output)  # Show output to user

        # Extract job ID from output (format: "Created custom job: projects/.../customJobs/JOB_ID")
        # Try full resource path first
        match = re.search(r"projects/\d+/locations/[^/\s]+/customJobs/\d+", output)
        if match:
            return match.group(0)
        # Try just customJobs/ID
        match = re.search(r"customJobs/(\d+)", output)
        if match:
            return match.group(0)  # Returns "customJobs/123456"
        return ""
    finally:
        os.unlink(tmp_path)


@task
def stream_logs(ctx: Context, job_id: str, region: str = "europe-west1", project_id: str = "dtumlops-484310") -> None:
    """Stream logs from a Vertex AI job.

    Args:
        job_id: Job ID (can be just the numeric ID or full resource path).
        region: GCP region.
        project_id: GCP project ID (used if job_id is just numeric).
    """
    # Handle both full path and just numeric ID
    if job_id.startswith("projects/"):
        full_path = job_id
    elif job_id.startswith("customJobs/"):
        full_path = f"projects/{project_id}/locations/{region}/{job_id}"
    else:
        # Just numeric ID
        full_path = f"projects/{project_id}/locations/{region}/customJobs/{job_id}"

    ctx.run(
        f"gcloud ai custom-jobs stream-logs {full_path}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def deploy_and_stream(
    ctx: Context,
    config: str = "configs/vertex_ai/vertex_ai_config_tfidf.yaml",
    region: str = "europe-west1",
    project_id: str = "dtumlops-484310",  # Default to correct project ID
    wait_seconds: int = 5,
    dockerfile: str = "dockerfiles/train.dockerfile",
    tag: str = "train:latest",
    registry: str = "container-registry",
) -> None:
    """Build, push Docker image, submit job, wait, and stream logs.

    Args:
        config: Vertex AI config file path.
        region: GCP region.
        project_id: GCP project ID.
        wait_seconds: Seconds to wait before streaming logs.
        dockerfile: Dockerfile to build.
        tag: Docker image tag.
        registry: GCP Artifact Registry name.
    """
    import time

    # Step 1: Build and push Docker image
    ctx.run("echo 'Step 1: Building and pushing Docker image...'", echo=True)
    docker_build_and_push_gcp(ctx, dockerfile, tag, region, project_id, registry)

    # Step 2: Submit job
    ctx.run("echo 'Step 2: Submitting Vertex AI job...'", echo=True)
    job_id = submit_job(ctx, config, region)

    if not job_id:
        ctx.run("echo 'ERROR: Failed to get job ID from submission'", echo=True)
        return

    ctx.run(f"echo 'Job submitted: {job_id}'", echo=True)

    # Step 3: Wait
    ctx.run(f"echo 'Step 3: Waiting {wait_seconds} seconds before streaming logs...'", echo=True)
    time.sleep(wait_seconds)

    # Step 4: Stream logs
    ctx.run("echo 'Step 4: Streaming logs...'", echo=True)
    stream_logs(ctx, job_id, region, project_id)


@task
def job_status(ctx: Context, job_id: str, region: str = "europe-west1") -> None:
    """Check status of a Vertex AI job."""
    ctx.run(
        f"gcloud ai custom-jobs describe {job_id} --region={region} --format='yaml(state,error,createTime,updateTime)'",
        echo=True,
        pty=not WINDOWS,
    )


@task
def list_jobs(ctx: Context, region: str = "europe-west1", limit: int = 10) -> None:
    """List recent Vertex AI jobs."""
    ctx.run(
        f"gcloud ai custom-jobs list --region={region} --limit={limit} --format='table(displayName,state,createTime)'",
        echo=True,
        pty=not WINDOWS,
    )


@task
def deploy_api(
    ctx: Context,
    service_name: str = "arxiv-classifier-api",
    region: str = "europe-west1",
    project_id: str = "dtumlops-484310",
    image_name: str = "inference-api",
    allow_unauthenticated: bool = True,
) -> None:
    """Deploy API to Cloud Run.

    Builds the API Docker image, pushes it to Artifact Registry, and deploys to Cloud Run.

    Args:
        service_name: Name of the Cloud Run service.
        region: GCP region for deployment.
        project_id: GCP project ID.
        image_name: Name of the Docker image.
        allow_unauthenticated: If True, allows unauthenticated access to the service.
    """
    # Step 1: Build and push Docker image
    ctx.run("echo 'Step 1: Building and pushing API Docker image...'", echo=True)
    docker_build_and_push_gcp(
        ctx, "dockerfiles/api.dockerfile", f"{image_name}:latest", region, project_id, "container-registry"
    )

    # Step 2: Deploy to Cloud Run
    ctx.run(f"echo 'Step 2: Deploying to Cloud Run service: {service_name}...'", echo=True)

    image_url = f"europe-west1-docker.pkg.dev/{project_id}/container-registry/{image_name}:latest"

    deploy_cmd = (
        f"gcloud run deploy {service_name} "
        f"--image {image_url} "
        f"--platform managed "
        f"--region {region} "
        f"--port 8000 "
        f"--memory 2Gi "
        f"--cpu 2 "
        f"--timeout 300 "
        f"--max-instances 10 "
        f"--min-instances 0 "
        f"--set-env-vars REQUEST_LOG_PATH=/tmp/api_requests.csv"
    )

    if allow_unauthenticated:
        deploy_cmd += " --allow-unauthenticated"

    ctx.run(deploy_cmd, echo=True, pty=not WINDOWS)

    # Step 3: Get and display service URL
    ctx.run("echo 'Step 3: Retrieving service URL...'", echo=True)
    result = ctx.run(
        f"gcloud run services describe {service_name} --region={region} --format='value(status.url)'",
        echo=True,
        hide="stdout",
    )
    service_url = result.stdout.strip()

    ctx.run("echo '✅ API deployed successfully!'", echo=True)
    ctx.run(f"echo 'Service URL: {service_url}'", echo=True)
    ctx.run(f"echo 'Health check: {service_url}/health'", echo=True)
    ctx.run(
        f'echo \'Test: curl -X POST {service_url}/predict -H "Content-Type: application/json" -d \'{{"text":"test"}}\'\'',
        echo=True,
    )


@task
def get_api_url(
    ctx: Context,
    service_name: str = "arxiv-classifier-api",
    region: str = "europe-west1",
) -> None:
    """Get the URL of the deployed Cloud Run API service.

    Args:
        service_name: Name of the Cloud Run service.
        region: GCP region.
    """
    ctx.run(
        f"gcloud run services describe {service_name} --region={region} --format='value(status.url)'",
        echo=True,
        pty=not WINDOWS,
    )
