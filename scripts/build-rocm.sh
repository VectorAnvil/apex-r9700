#!/usr/bin/env bash
set -Eeuo pipefail

readonly LLAMA_DIR="${1:?usage: build-rocm.sh /path/to/llama.cpp [/path/to/build]}"
readonly BUILD_DIR="${2:-$LLAMA_DIR/build-apex-r9700}"
readonly JOBS="${APEX_BUILD_JOBS:-$(nproc)}"

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1201 \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DGGML_NATIVE=ON

cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo "Built binaries: $BUILD_DIR/bin"
