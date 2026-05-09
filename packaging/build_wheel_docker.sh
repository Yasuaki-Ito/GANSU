#!/usr/bin/env bash
#
# Build a manylinux_2_28 wheel of GANSU inside a reproducible Docker image.
#
# The image is built once and cached locally; subsequent invocations just
# `docker run` and produce the wheel under host's wheelhouse/ directory.
#
# Usage:
#   bash packaging/build_wheel_docker.sh
#
# Env overrides:
#   IMAGE_TAG=gansu-wheel-builder:cuda12.6   # docker image name:tag
#   CUDA_ARCHS="80;86;89;90;100;120"          # GPU SM list passed to cmake
#   JOBS=$(nproc)                             # build parallelism

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="${IMAGE_TAG:-gansu-wheel-builder:cuda12.9}"
CUDA_ARCHS="${CUDA_ARCHS:-80;86;89;90;100;120}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# --- 1. Build (or reuse) the builder image --------------------------------
echo "==> Building Docker image: $IMAGE_TAG"
docker build \
    -f packaging/Dockerfile.wheel \
    -t "$IMAGE_TAG" \
    packaging/

# --- 2. Run the wheel build inside the container --------------------------
# The repo is mounted at /work; the cmake build tree is kept in a named
# Docker volume (gansu-cmake-build) mounted at /tmp/gansu-cmake-build so that
# (a) FetchContent's git checkouts never touch the host's bind-mounted
#     Dropbox/Windows filesystem, where DrvFs would mark the .git/objects
#     pack files read-only and prevent later cleanup, and
# (b) the cache survives across runs, making SKIP_BUILD=1 actually useful.
# Outputs (wheelhouse/, dist/, staged python/gansu/{lib,data}/) still appear
# on the host directly via the /work bind mount.
echo "==> Running wheel build (CUDA archs: $CUDA_ARCHS)"
docker run --rm \
    -v "$REPO_ROOT":/work \
    -v gansu-cmake-build:/tmp/gansu-cmake-build \
    -w /work \
    -e CUDA_ARCHS="$CUDA_ARCHS" \
    -e JOBS="$JOBS" \
    -e PLAT="manylinux_2_28_x86_64" \
    -e BUILD_DIR="/tmp/gansu-cmake-build" \
    -e SKIP_BUILD="${SKIP_BUILD:-0}" \
    "$IMAGE_TAG" \
    bash script/build_wheel.sh

echo
echo "=========================================================="
echo " Done. Wheel(s):"
ls -lh wheelhouse/
echo "=========================================================="
