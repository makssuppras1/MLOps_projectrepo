# Quick Fix: Docker Platform Issue Summary

## 🎯 Root Cause

**You're running an AMD64 image on an ARM64 Mac using QEMU emulation.**

XGBoost's native C++ code crashes under emulation, causing a silent segfault. The container exits immediately after "Calling XGBoost fit()..." because the process is killed by the OS, not caught by Python exceptions.

## ✅ Immediate Fix (Copy-Paste)

```bash
# 1. Remove old AMD64 image
docker rmi train-tfidf:latest 2>/dev/null || true

# 2. Build native ARM64 (fast, stable, no emulation)
cd "/Users/maks/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/11. semester/02476 Machine Learning Operations/MLOps_projectrepo"

docker buildx build --platform linux/arm64 \
  -f dockerfiles/train_tfidf.dockerfile \
  -t train-tfidf:latest \
  --load .

# 3. Verify it's ARM64
docker image inspect train-tfidf:latest --format 'Architecture: {{.Architecture}}'
# Should output: arm64

# 4. Run training (now native, no crashes)
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/outputs:/outputs" \
  train-tfidf:latest training.epochs=1
```

## 📊 Key Concepts (From Docker Docs)

1. **Platform** = OS + Architecture (e.g., `linux/arm64`)
2. **Native builds** = Fast, stable (build for your host architecture)
3. **Emulation (QEMU)** = Slow, can crash with native code (running different architecture)
4. **Multi-platform images** = Manifest list with multiple variants (auto-selects native)
5. **`--load` limitation** = Only works with single-platform images

## 🔍 Diagnostic Commands

```bash
# Run full diagnostic
./scripts/diagnose_platform.sh train-tfidf:latest

# Quick checks
uname -m                                    # Your host: arm64
docker image inspect train-tfidf:latest --format '{{.Architecture}}'  # Image: should be arm64
docker run --rm train-tfidf:latest uname -m  # Container: should be aarch64 (ARM64)
```

## 📚 Full Documentation

See `docs/MULTI_PLATFORM_BUILD_GUIDE.md` for:
- Complete explanation of multi-platform builds
- All diagnostic commands
- Fixes for all scenarios (ARM64, AMD64, multi-platform)
- Troubleshooting guide

## ⚠️ Important Notes

- **For local development**: Always build native ARM64 (`--platform linux/arm64`)
- **For GCP deployment**: Build AMD64 separately and push to registry
- **Multi-platform builds**: Require `--push` (cannot use `--load`)
- **QEMU emulation**: Avoid for compute-heavy tasks (XGBoost, NumPy, etc.)

## ✅ Success Indicators

After applying the fix, you should see:
- ✅ No "platform mismatch" warnings
- ✅ Training completes successfully
- ✅ Model file created in `outputs/trained_model.pkl`
- ✅ Exit code 0
- ✅ Log shows "XGBoost training completed successfully"
