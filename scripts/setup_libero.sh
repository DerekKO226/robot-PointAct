#!/usr/bin/env bash
# Install the LIBERO simulator into the `libero` pixi environment.
# LIBERO is downloaded into .pixi/build-src/LIBERO (git-ignored), pinned to a fixed commit,
# and installed in editable mode (a regular install misses the bddl/init/asset files).
#
# Usage: pixi install -e libero && pixi run -e libero setup-libero
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" || -z "${PIXI_PROJECT_ROOT:-}" || -z "${LIBERO_CONFIG_PATH:-}" ]]; then
    echo "Run this through pixi: pixi run -e libero setup-libero" >&2
    exit 1
fi

LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_DIR="$PIXI_PROJECT_ROOT/.pixi/build-src/LIBERO"

if [[ "$(cat "$LIBERO_DIR/.pinned_commit" 2>/dev/null)" != "$LIBERO_COMMIT" ]]; then
    # Download a tarball of the pinned commit; curl can retry and resume, unlike git over HTTP/2.
    TARBALL="$PIXI_PROJECT_ROOT/.pixi/build-src/LIBERO-${LIBERO_COMMIT}.tar.gz"
    mkdir -p "$(dirname "$TARBALL")"
    curl -fL --retry 20 --retry-all-errors -C - -o "$TARBALL" \
        "https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/${LIBERO_COMMIT}"
    rm -rf "$LIBERO_DIR"
    mkdir -p "$LIBERO_DIR"
    tar -xzf "$TARBALL" -C "$LIBERO_DIR" --strip-components=1
    echo "$LIBERO_COMMIT" > "$LIBERO_DIR/.pinned_commit"
fi

# Dependencies are declared in pyproject.toml ([tool.pixi.feature.libero]).
# LIBERO's init-state files are pickled numpy arrays; torch>=2.6 defaults to weights_only=True and refuses them.
# They come from the pinned upstream commit above, so allow full unpickling for this one call only.
sed -i 's|init_states = torch.load(init_states_path)$|init_states = torch.load(init_states_path, weights_only=False)|' \
    "$LIBERO_DIR/libero/libero/benchmark/__init__.py"

# Compat mode adds LIBERO_DIR to sys.path; the default editable mode misses the top-level `libero`
# package because it has no __init__.py.
python -m pip install --no-deps -e "$LIBERO_DIR" --config-settings editable_mode=compat

# Write the LIBERO config up front; otherwise `import libero` asks interactive questions.
BENCH_ROOT="$LIBERO_DIR/libero/libero"
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $BENCH_ROOT
bddl_files: $BENCH_ROOT/bddl_files
init_states: $BENCH_ROOT/init_files
datasets: $LIBERO_DIR/libero/datasets
assets: $BENCH_ROOT/assets
EOF

python - <<'PY'
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
print(f"LIBERO ok: libero_spatial has {suite.n_tasks} tasks")
PY
