#!/usr/bin/env bash
# Build the CUDA extensions that have no prebuilt linux-aarch64 wheels
# (spconv + cumm, flash-attn, torch-scatter) into the pixi environment.
# On linux-64, pixi installs prebuilt wheels instead, so this script is not needed.
#
# Usage (from the repo root):
#   pixi run build-cuda-ext                  # all
#   pixi run build-cuda-ext spconv           # one of: spconv | flash-attn | torch-scatter
#
# Env overrides:
#   CUDA_ARCH  compute capability to build for (default 12.0; GB10 is sm_121 and runs sm_120 code)
#   MAX_JOBS   parallel compile jobs (default 8)
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" || -z "${PIXI_PROJECT_ROOT:-}" ]]; then
    echo "Run this through pixi: pixi run build-cuda-ext" >&2
    exit 1
fi

ARCH="${CUDA_ARCH:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CUDA_HOME="$CONDA_PREFIX"

CUMM_VERSION=v0.8.2    # first cumm release with CUDA 12.8 / sm_120 support
SPCONV_VERSION=v2.3.8
FLASH_ATTN_VERSION=2.8.3
TORCH_SCATTER_VERSION=2.1.2

SRC_DIR="$PIXI_PROJECT_ROOT/.pixi/build-src"
mkdir -p "$SRC_DIR"

# Keep pip from replacing the pinned packages while installing build helpers.
CONSTRAINTS="$SRC_DIR/constraints.txt"
python - > "$CONSTRAINTS" <<'PY'
from importlib.metadata import version
for pkg in ("torch", "numpy"):
    print(f"{pkg}=={version(pkg)}")
PY

pip_install() {
    python -m pip install --no-build-isolation -c "$CONSTRAINTS" "$@"
}

fetch() {  # fetch <repo-url> <tag> <dir>
    local url=$1 tag=$2 dir=$3
    if [[ ! -d "$dir" ]]; then
        git clone --depth 1 --branch "$tag" "$url" "$dir"
    fi
}

build_spconv() {
    echo "==> cumm ${CUMM_VERSION} + spconv ${SPCONV_VERSION} (arch ${ARCH})"
    pip_install "pccm>=0.4.16" "ccimport>=0.4.4" "pybind11>=2.6.0" fire sympy

    local cumm_dir="$SRC_DIR/cumm-${CUMM_VERSION}"
    fetch https://github.com/FindDefinition/cumm "$CUMM_VERSION" "$cumm_dir"
    # cumm hardcodes the conda CUDA header dir as targets/x86_64-linux; on aarch64 it is targets/sbsa-linux.
    # Without this it falls back to /usr/local/cuda, which may be a different CUDA version.
    if [[ "$(uname -m)" == "aarch64" ]]; then
        sed -i 's#"targets/x86_64-linux/include"#"targets/sbsa-linux/include"#' "$cumm_dir/cumm/common.py"
    fi

    local spconv_dir="$SRC_DIR/spconv-${SPCONV_VERSION}"
    fetch https://github.com/traveller59/spconv "$SPCONV_VERSION" "$spconv_dir"

    export CUMM_CUDA_ARCH_LIST="${ARCH}+PTX"
    # Always rebuild from scratch. cumm's setup.py imports `cumm` to generate its bindings; if an
    # installed cumm shadows the source tree, core_cc gets nested names and `tensorview_bind` is missing.
    python -m pip uninstall -y spconv cumm || true
    rm -rf "$cumm_dir/build" "$spconv_dir/build"
    CUMM_DISABLE_JIT=1 pip_install --no-deps "$cumm_dir"
    # spconv 2.3.8 pins cumm<0.8 only because cumm 0.8.0 was a CI version bump; the API is unchanged.
    SPCONV_DISABLE_JIT=1 pip_install --no-deps "$spconv_dir"
}

build_flash_attn() {
    echo "==> flash-attn ${FLASH_ATTN_VERSION} (arch ${ARCH})"
    FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="${ARCH/./}" NVCC_THREADS="${NVCC_THREADS:-2}" \
        pip_install --no-deps "flash-attn==${FLASH_ATTN_VERSION}"
}

build_torch_scatter() {
    echo "==> torch-scatter ${TORCH_SCATTER_VERSION} (arch ${ARCH})"
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${ARCH}" \
        pip_install --no-deps "torch-scatter==${TORCH_SCATTER_VERSION}"
}

target="${1:-all}"
case "$target" in
    spconv) build_spconv ;;
    flash-attn) build_flash_attn ;;
    torch-scatter) build_torch_scatter ;;
    all) build_spconv; build_torch_scatter; build_flash_attn ;;
    *) echo "unknown target: $target" >&2; exit 1 ;;
esac
