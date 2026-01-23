#!/bin/bash
# Docker Multi-Platform Diagnostic Script
# Based on: https://docs.docker.com/build/building/multi-platform/

set -e

echo "=========================================="
echo "DOCKER MULTI-PLATFORM DIAGNOSTIC"
echo "=========================================="
echo ""

# 1. Host Architecture
echo "1. HOST ARCHITECTURE"
echo "-------------------"
HOST_ARCH=$(uname -m)
HOST_OS=$(uname -s)
echo "Host OS: $HOST_OS"
echo "Host Architecture: $HOST_ARCH"
if [[ "$HOST_ARCH" == "arm64" || "$HOST_ARCH" == "aarch64" ]]; then
    echo "✓ Running on ARM64 (Apple Silicon or ARM Linux)"
elif [[ "$HOST_ARCH" == "x86_64" || "$HOST_ARCH" == "amd64" ]]; then
    echo "✓ Running on AMD64/x86_64"
else
    echo "⚠ Unknown architecture: $HOST_ARCH"
fi
echo ""

# 2. Docker Buildx Builders
echo "2. DOCKER BUILDX BUILDERS"
echo "-------------------------"
echo "Available builders:"
docker buildx ls
echo ""

# 3. Check if containerd image store is enabled (Docker Desktop)
echo "3. IMAGE STORE TYPE"
echo "-------------------"
if command -v docker &> /dev/null; then
    DOCKER_INFO=$(docker info 2>/dev/null | grep -i "storage driver" || echo "Unknown")
    echo "Storage driver: $DOCKER_INFO"
    echo "Note: For multi-platform images, containerd image store is recommended"
    echo "      Check Docker Desktop settings > Features in development > Use containerd"
fi
echo ""

# 4. QEMU/binfmt Support
echo "4. QEMU/BINFMT SUPPORT (for emulation)"
echo "--------------------------------------"
if [[ -d /proc/sys/fs/binfmt_misc ]]; then
    echo "Checking binfmt_misc registrations:"
    for qemu in /proc/sys/fs/binfmt_misc/qemu-*; do
        if [[ -f "$qemu" ]]; then
            FLAGS=$(grep flags "$qemu" 2>/dev/null | cut -d: -f2 || echo "N/A")
            if echo "$FLAGS" | grep -q "F"; then
                echo "  ✓ $(basename $qemu): Registered (F flag present)"
            else
                echo "  ⚠ $(basename $qemu): Registered but F flag missing"
            fi
        fi
    done
    if ! ls /proc/sys/fs/binfmt_misc/qemu-* 1>/dev/null 2>&1; then
        echo "  ⚠ No QEMU binfmt registrations found"
        echo "  To install: docker run --privileged --rm tonistiigi/binfmt --install all"
    fi
else
    echo "  ⚠ /proc/sys/fs/binfmt_misc not available (may be on macOS)"
    echo "  Docker Desktop on Mac includes QEMU automatically"
fi
echo ""

# 5. Image Platform Inspection
echo "5. IMAGE PLATFORM INSPECTION"
echo "----------------------------"
IMAGE_NAME="${1:-train-tfidf:latest}"
echo "Inspecting image: $IMAGE_NAME"
echo ""

if docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Local image found. Platform info:"
    docker image inspect "$IMAGE_NAME" --format '{{.Architecture}}/{{.Os}}' 2>/dev/null || echo "  ⚠ Could not determine platform"
    echo ""
    echo "Full platform details:"
    docker image inspect "$IMAGE_NAME" --format '{{json .Architecture}}' 2>/dev/null | jq -r '.' 2>/dev/null || docker image inspect "$IMAGE_NAME" | grep -i "architecture\|os" | head -5
else
    echo "  ⚠ Image '$IMAGE_NAME' not found locally"
fi
echo ""

# 6. Multi-platform Manifest Check
echo "6. MULTI-PLATFORM MANIFEST CHECK"
echo "--------------------------------"
if docker buildx imagetools inspect "$IMAGE_NAME" &>/dev/null 2>&1; then
    echo "Manifest list (multi-platform) info:"
    docker buildx imagetools inspect "$IMAGE_NAME" --format '{{json .}}' 2>/dev/null | jq -r '.manifests[] | "  Platform: \(.platform.os)/\(.platform.architecture)"' 2>/dev/null || echo "  Single-platform image or not in registry"
else
    echo "  ⚠ Cannot inspect manifest (image may not be in registry or buildx not configured)"
fi
echo ""

# 7. Buildx Driver Check
echo "7. BUILDX DRIVER CHECK"
echo "---------------------"
CURRENT_BUILDER=$(docker buildx ls | grep '*' | awk '{print $1}' || echo "default")
echo "Current builder: $CURRENT_BUILDER"
BUILDER_INFO=$(docker buildx inspect "$CURRENT_BUILDER" 2>/dev/null | grep -i "driver\|platform" | head -3 || echo "  Could not inspect builder")
echo "$BUILDER_INFO"
echo ""

# 8. Recommendations
echo "8. RECOMMENDATIONS"
echo "-----------------"
if [[ "$HOST_ARCH" == "arm64" || "$HOST_ARCH" == "aarch64" ]]; then
    echo "✓ You're on ARM64. For best performance:"
    echo "  - Build native ARM64 images: docker buildx build --platform linux/arm64 ..."
    echo "  - Avoid AMD64 emulation for compute-heavy tasks (XGBoost, compilation)"
    echo "  - Use multi-platform builds only if you need to push to registry for GCP"
fi

if docker image inspect "$IMAGE_NAME" &>/dev/null 2>&1; then
    IMG_ARCH=$(docker image inspect "$IMAGE_NAME" --format '{{.Architecture}}' 2>/dev/null || echo "unknown")
    if [[ "$HOST_ARCH" == "arm64" && "$IMG_ARCH" == "amd64" ]]; then
        echo ""
        echo "⚠ WARNING: You're running AMD64 image on ARM64 host"
        echo "  This requires QEMU emulation which can cause:"
        echo "  - 2-5x slower performance"
        echo "  - Crashes with native code (C/C++ libraries like XGBoost)"
        echo "  - Memory issues"
        echo ""
        echo "  SOLUTION: Rebuild for ARM64:"
        echo "    docker buildx build --platform linux/arm64 -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest --load ."
    fi
fi

echo ""
echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="
