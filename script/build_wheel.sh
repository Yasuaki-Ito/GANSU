#!/usr/bin/env bash
#
# Build a thin manylinux wheel of GANSU plus the matching native library
# (libgansu.so) as a separate asset for GitHub Releases.
#
# Outputs (under wheelhouse/ on the host):
#   gansu-<ver>-py3-none-manylinux_2_28_x86_64.whl   (~10 MB; uploaded to PyPI
#                                                     and to GitHub Releases)
#   libgansu-<ver>-linux-x86_64.so                   (~150 MB; uploaded only
#                                                     to GitHub Releases)
#   libgansu-<ver>-linux-x86_64.so.sha256            (checksum)
#
# The wheel ships a small `_native_meta.json` (version + sha256) so the
# Python loader can fetch the .so from the matching GitHub Release on first
# use, verify the hash, and cache it in `~/.cache/gansu/<ver>/`. Users who
# want an offline install instead can set the GANSU_LIB env var or drop the
# .so into the cache path manually.
#
# Prerequisites (install once):
#   pip install --upgrade build wheel auditwheel setuptools patchelf
#   # System: cmake >= 3.31, CUDA Toolkit 12.x with nvcc, gcc/g++ >= 11
#
# Usage:
#   bash script/build_wheel.sh                  # build all archs (default)
#   CUDA_ARCHS="80;90" bash script/build_wheel.sh   # custom arch list
#   SKIP_BUILD=1 bash script/build_wheel.sh     # skip cmake, reuse $BUILD_DIR
#
# Filesystem layout:
#   $BUILD_DIR  (default /tmp/gansu-cmake-build)  cmake build tree
#   $STAGE_DIR  (default /tmp/gansu-wheel-stage)  wheel input mirror
#   wheelhouse/  in repo, on host                  artefacts (only output)
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Configuration ---------------------------------------------------------
CUDA_ARCHS="${CUDA_ARCHS:-80;86;89;90;100;120}"
PLAT="${PLAT:-manylinux_2_28_x86_64}"
JOBS="${JOBS:-$(nproc)}"
SKIP_BUILD="${SKIP_BUILD:-0}"
BUILD_DIR="${BUILD_DIR:-/tmp/gansu-cmake-build}"
STAGE_DIR="${STAGE_DIR:-/tmp/gansu-wheel-stage}"

echo "==> CUDA archs: $CUDA_ARCHS"
echo "==> manylinux platform: $PLAT"
echo "==> CMake build dir: $BUILD_DIR"
echo "==> Wheel stage dir:  $STAGE_DIR"

# --- 1. Build libgansu.so --------------------------------------------------
if [[ "$SKIP_BUILD" != "1" ]]; then
    mkdir -p "$BUILD_DIR"
    find "$BUILD_DIR" -mindepth 1 -delete 2>/dev/null || true
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        -DENABLE_MULTI_GPU=ON \
        -DGANSU_BUNDLE_OPENBLAS=ON
    cmake --build "$BUILD_DIR" --target gansu_shared -j "$JOBS"
fi

if [[ ! -f "$BUILD_DIR/libgansu.so" ]]; then
    echo "ERROR: $BUILD_DIR/libgansu.so not found" >&2
    exit 1
fi

# --- 2. Compute version + SHA256 ------------------------------------------
VERSION=$(python -c "
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
")
PLATFORM_ID="linux-x86_64"
SHA256=$(sha256sum "$BUILD_DIR/libgansu.so" | cut -d' ' -f1)
echo "==> version: $VERSION   sha256: $SHA256"

# --- 3. Stage source for the thin wheel (no .so embedded) -----------------
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/python/gansu/data/basis" \
         "$STAGE_DIR/python/gansu/data/auxiliary_basis"

# Python source
cp python/gansu/*.py "$STAGE_DIR/python/gansu/"

# Build configuration
cp pyproject.toml "$STAGE_DIR/"
cp setup.py "$STAGE_DIR/"
[[ -f README.md ]] && cp README.md "$STAGE_DIR/"
[[ -f LICENSE ]] && cp LICENSE "$STAGE_DIR/"

# Native-library metadata: lets the Python loader fetch the right .so
# on first use and verify it.
cat > "$STAGE_DIR/python/gansu/_native_meta.json" <<EOF
{
  "version": "$VERSION",
  "platform": "$PLATFORM_ID",
  "sha256": "$SHA256"
}
EOF

# Fortran runtime for the statically-embedded OpenBLAS (GANSU_BUNDLE_OPENBLAS).
# libgansu.so links libopenblas.a statically but keeps a dynamic dependency on
# libgfortran.so.5 / libquadmath.so.0 (their static archives are not reliably
# PIC). Ship these two small libs in the wheel; the Python loader preloads them
# (RTLD_GLOBAL) before libgansu.so, satisfying its DT_NEEDED with no system
# OpenBLAS/Fortran runtime required. Resolve symlinks so we copy the real ELF.
mkdir -p "$STAGE_DIR/python/gansu/lib"
for soname in libquadmath.so.0 libgfortran.so.5; do
    src=$(gfortran -print-file-name="$soname" 2>/dev/null || true)
    if [[ -z "$src" || "$src" == "$soname" || ! -e "$src" ]]; then
        # -print-file-name returns the bare name when not found on the search path
        src=$(gcc -print-file-name="$soname" 2>/dev/null || true)
    fi
    if [[ -z "$src" || "$src" == "$soname" || ! -e "$src" ]]; then
        echo "ERROR: bundled runtime lib '$soname' not found (need gcc-gfortran)" >&2
        exit 1
    fi
    cp -L "$src" "$STAGE_DIR/python/gansu/lib/$soname"
    echo "==> Staged Fortran runtime: $soname  <-  $src"
done

# Basis sets (.gbs and .sad cache) — kept inside the thin wheel since
# they are only a few MB total and never need to vary at runtime.
shopt -s nullglob
for f in basis/*.gbs basis/*.sad; do
    [[ -e "$f" ]] && cp "$f" "$STAGE_DIR/python/gansu/data/basis/"
done
for f in auxiliary_basis/*.gbs; do
    [[ -e "$f" ]] && cp "$f" "$STAGE_DIR/python/gansu/data/auxiliary_basis/"
done
shopt -u nullglob

n_basis=$(ls "$STAGE_DIR/python/gansu/data/basis" 2>/dev/null | wc -l)
n_aux=$(ls "$STAGE_DIR/python/gansu/data/auxiliary_basis" 2>/dev/null | wc -l)
echo "==> Staged $n_basis basis files, $n_aux auxiliary basis files"

# --- 4. Build the thin wheel ----------------------------------------------
pushd "$STAGE_DIR" >/dev/null
python -m build --wheel --no-isolation
RAW_WHEEL=$(ls "$STAGE_DIR"/dist/gansu-*.whl | head -n1)
popd >/dev/null
echo "==> Built raw wheel: $RAW_WHEEL"

# --- 5. Retag the wheel to manylinux -------------------------------------
# We don't run auditwheel here: the wheel contains no native code (the .so
# is fetched at runtime), so there is nothing to bundle. We just need to
# swap the platform tag from the build-host's `linux_x86_64` to the
# distributable `manylinux_2_28_x86_64`. `wheel tags` rewrites the
# WHEEL/RECORD metadata in place and renames the file accordingly.
mkdir -p wheelhouse
find wheelhouse -mindepth 1 -delete 2>/dev/null || true

pushd "$STAGE_DIR/dist" >/dev/null
wheel tags --platform-tag "$PLAT" --remove gansu-*-linux_x86_64.whl
popd >/dev/null

cp "$STAGE_DIR"/dist/gansu-*-${PLAT}.whl wheelhouse/

# --- 6. Stage the native library + checksum for GitHub Release ------------
SO_NAME="libgansu-${VERSION}-${PLATFORM_ID}.so"
cp "$BUILD_DIR/libgansu.so" "wheelhouse/$SO_NAME"
echo "$SHA256  $SO_NAME" > "wheelhouse/${SO_NAME}.sha256"

echo
echo "=========================================================="
echo " Release artefacts ready in wheelhouse/:"
ls -lh wheelhouse/
echo "=========================================================="
echo
echo "Distribution map:"
echo "  Upload to PyPI            : wheelhouse/gansu-*.whl"
echo "  Upload to GitHub Release  : ALL of wheelhouse/* (wheel + .so + sha256)"
echo
echo "Local test in fresh venv:"
echo "  python -m venv /tmp/venv-gansu && source /tmp/venv-gansu/bin/activate"
echo "  pip install wheelhouse/gansu-*.whl"
echo "  python -c 'import gansu; gansu.init()'"
echo "    # First call downloads libgansu.so to ~/.cache/gansu/${VERSION}/"
echo "  python -c 'import gansu; gansu.init(); print(\"OK\"); gansu.finalize()'"
