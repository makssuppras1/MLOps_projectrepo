# Repository Reorganization Proposal

## Executive Summary

Your repository structure is **already quite well-organized** following MLOps best practices. However, there are a few root-level files that should be moved or cleaned up to improve maintainability and clarity.

## Current Root Directory Analysis

### ✅ Files That SHOULD Remain in Root

These are standard project configuration files that belong in the root:

| File | Purpose | Status |
|------|---------|--------|
| `.gitignore` | Git ignore patterns | ✅ Keep |
| `.dockerignore` | Docker build ignore patterns | ✅ Keep |
| `.gcloudignore` | GCP deployment ignore patterns | ✅ Keep |
| `.dvcignore` | DVC ignore patterns | ✅ Keep |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration | ✅ Keep |
| `.python-version` | Python version specification | ✅ Keep |
| `pyproject.toml` | Python project configuration (dependencies, tools) | ✅ Keep |
| `uv.lock` | Locked dependency versions | ✅ Keep |
| `README.md` | Project documentation | ✅ Keep |
| `LICENSE` | License file | ✅ Keep |
| `tasks.py` | Invoke task definitions (project automation) | ✅ Keep |
| `data.dvc` | DVC metadata file for data versioning | ✅ Keep |
| `Dockerfile` | Production API Dockerfile | ✅ Keep |
| `.cursorrules` | Cursor IDE configuration | ✅ Keep |

### ⚠️ Files That SHOULD Be Moved

| Current Path | Proposed Path | Reason |
|--------------|---------------|--------|
| `artifact-cleanup-policy.json` | `configs/gcp/artifact-cleanup-policy.json` | GCP configuration belongs in configs directory |
| `deployment_summary.json` | `reports/deployment_summary.json` (or gitignore) | Generated deployment report - should be in reports or ignored |
| `test_summary.json` | `reports/test_summary.json` (or gitignore) | Generated test report - should be in reports or ignored |

### 🗑️ Files That SHOULD Be Gitignored (Generated Artifacts)

These files are generated during operations and should not be committed:

- `deployment_summary.json` - Generated during deployment
- `test_summary.json` - Generated during testing

**Note:** These are already partially covered by `.gitignore` patterns (`*.log`, `outputs/`), but should be explicitly added.

## Proposed Directory Structure

```
MLOps_projectrepo/
├── .github/                    # GitHub Actions workflows
│   ├── dependabot.yaml
│   └── workflows/
│       ├── cml_data.yaml
│       ├── cml_model_registry.yaml
│       └── tests.yaml
├── .devcontainer/              # VS Code dev container config
│   ├── devcontainer.json
│   ├── Dockerfile
│   └── post_create.sh
├── app/                        # FastAPI application
│   ├── __init__.py
│   └── main.py
├── ci/                         # CI/CD configuration
│   ├── cloudbuild-api.yaml
│   └── cloudbuild.yaml
├── configs/                    # Configuration files
│   ├── .gitkeep
│   ├── config.yaml
│   ├── config_tfidf.yaml
│   ├── model_conf.yaml
│   ├── model_tfidf.yaml
│   ├── training_conf.yaml
│   ├── training_conf_tfidf.yaml
│   ├── experiment/             # Experiment configs
│   │   ├── 2hour.yaml
│   │   ├── balanced.yaml
│   │   ├── exp1.yaml
│   │   ├── exp2.yaml
│   │   ├── fast.yaml
│   │   ├── null.yaml
│   │   ├── optimized_distilbert.yaml
│   │   └── tfidf_xgboost.yaml
│   ├── vertex_ai/              # Vertex AI job configs
│   │   ├── config.yaml
│   │   ├── gcp_workflow_spec.yaml
│   │   ├── vertex_ai_config_*.yaml
│   │   └── vertex_ai_train_*.yaml
│   └── gcp/                    # NEW: GCP-specific configs
│       └── artifact-cleanup-policy.json
├── data.dvc                    # DVC metadata (root level - correct)
├── dockerfiles/                # Dockerfiles for different services
│   ├── api.dockerfile
│   ├── evaluate.dockerfile
│   └── train.dockerfile
├── Dockerfile                   # Production API Dockerfile (root level - correct)
├── docs/                       # Documentation
│   ├── DOWNLOAD_MODEL.md
│   ├── INVOKE_COMMANDS.md
│   ├── LOGGING_GUIDE.md
│   ├── MODEL_USAGE_GUIDE.md
│   ├── PRE_FLIGHT_CHECKLIST.md
│   ├── README.md
│   ├── VERTEX_AI_TRAINING_GUIDE.md
│   ├── mkdocs.yaml
│   ├── profiling_guide.md
│   └── source/
│       └── index.md
├── monitoring/                 # Monitoring and drift detection
│   ├── __init__.py
│   ├── collect_current_data.py
│   ├── drift_monitor.py
│   └── schema.json
├── models/                     # Trained models (gitignored)
│   └── .gitkeep
├── notebooks/                  # Jupyter notebooks
│   └── .gitkeep
├── reports/                    # Reports and generated artifacts
│   ├── .gitkeep
│   ├── README.md
│   ├── report.py
│   ├── figures/                # Generated figures
│   │   ├── .gitkeep
│   │   └── *.png
│   ├── deployment_summary.json # MOVED: Generated deployment reports
│   └── test_summary.json       # MOVED: Generated test reports
├── scripts/                    # Utility scripts
│   ├── build_docker.sh
│   ├── download_dataset.sh
│   ├── download_model_from_wandb.py
│   └── preflight_check.sh
├── src/                        # Source code
│   └── pname/
│       ├── __init__.py
│       ├── data.py
│       ├── data_stats.py
│       ├── evaluate.py
│       ├── metrics.py
│       ├── model.py
│       ├── model_tfidf.py
│       ├── profiler.py
│       ├── train.py
│       ├── train_tfidf.py
│       ├── visualize.py
│       └── visualize_features.py
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_training.py
│   ├── integrationtests/
│   │   ├── __init__.py
│   │   └── test_apis.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── test_drift.py
│   └── performancetests/
│       ├── __init__.py
│       └── locustfile.py
├── .cursorrules                # Cursor IDE rules (root level - correct)
├── .dvc/                       # DVC internal files
│   ├── .gitignore
│   └── config
├── .gitignore                  # Git ignore patterns (root level - correct)
├── .dockerignore               # Docker ignore patterns (root level - correct)
├── .gcloudignore               # GCP ignore patterns (root level - correct)
├── .dvcignore                  # DVC ignore patterns (root level - correct)
├── .pre-commit-config.yaml     # Pre-commit hooks (root level - correct)
├── .python-version             # Python version (root level - correct)
├── LICENSE                     # License file (root level - correct)
├── pyproject.toml              # Python project config (root level - correct)
├── README.md                   # Project README (root level - correct)
├── tasks.py                    # Invoke tasks (root level - correct)
└── uv.lock                     # Locked dependencies (root level - correct)
```

## Detailed File Movement Plan

### 1. Move `artifact-cleanup-policy.json`

**From:** `artifact-cleanup-policy.json`
**To:** `configs/gcp/artifact-cleanup-policy.json`

**Justification:**
- This is a GCP Artifact Registry cleanup policy configuration
- Belongs with other GCP/cloud configurations
- Creates logical grouping: `configs/gcp/` for all GCP-specific configs
- Makes it easier to find and maintain cloud infrastructure configs

**Impact:** Low - this file is likely only used during GCP setup and not referenced in code.

### 2. Move `deployment_summary.json`

**From:** `deployment_summary.json`
**To:** `reports/deployment_summary.json` (or gitignore)

**Justification:**
- This is a generated artifact from deployment operations
- Contains deployment status, endpoints, and test results
- Should be in `reports/` directory with other generated reports
- Should also be added to `.gitignore` since it's generated

**Impact:** Low - appears to be a one-time deployment report, not referenced in code.

### 3. Move `test_summary.json`

**From:** `test_summary.json`
**To:** `reports/test_summary.json` (or gitignore)

**Justification:**
- This is a generated artifact from test runs
- Contains test results and notes
- Should be in `reports/` directory with other generated reports
- Should also be added to `.gitignore` since it's generated

**Impact:** Low - appears to be a test report artifact, not referenced in code.

## Recommended .gitignore Updates

Add these patterns to `.gitignore`:

```gitignore
# Generated reports and summaries
reports/deployment_summary.json
reports/test_summary.json
deployment_summary.json
test_summary.json
```

**Note:** The existing `.gitignore` already covers many patterns, but these specific files should be explicitly listed.

## Cleanup Actions Summary

### Actions to Take:

1. **Create new directory:**
   ```bash
   mkdir -p configs/gcp
   ```

2. **Move files:**
   ```bash
   mv artifact-cleanup-policy.json configs/gcp/
   mv deployment_summary.json reports/
   mv test_summary.json reports/
   ```

3. **Update .gitignore:**
   - Add explicit patterns for generated JSON reports

4. **Verify no code references:**
   - Search codebase for references to moved files
   - Update any scripts or documentation that reference these files

### Files to Consider for Future Cleanup:

- **`reports/figures/*.png`** - These appear to be generated figures. Consider if they should be gitignored (they currently are not, which might be intentional for documentation).

## Benefits of This Reorganization

1. **Clearer Structure:** GCP configs are grouped together
2. **Better Organization:** Generated reports are in the reports directory
3. **Reduced Root Clutter:** Only essential project files remain in root
4. **Easier Maintenance:** Related files are grouped logically
5. **Standard Compliance:** Follows MLOps best practices

## Compatibility Notes

- ✅ No breaking changes expected
- ✅ All existing workflows should continue to work
- ✅ Docker builds unaffected
- ✅ CI/CD pipelines unaffected
- ⚠️ If any scripts reference these files by absolute path, they'll need updates

## Next Steps

1. Review this proposal
2. Create backup branch: `git checkout -b backup-before-reorg`
3. Execute file moves
4. Update `.gitignore`
5. Test that all workflows still function
6. Commit changes

---

**Generated:** 2026-01-21
**Repository:** MLOps_projectrepo
**Analysis Date:** Current
