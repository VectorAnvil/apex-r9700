#!/usr/bin/env bash
set -Eeuo pipefail
readonly LLAMA_DIR="${1:?usage: build-rocm.sh /isolated/llama.cpp /new/build-directory}"
readonly BUILD_DIR="${2:?provide a separate new build directory}"
readonly ROCM_ROOT="${APEX_ROCM_ROOT:-/opt/rocm-7.2.0}"
if [[ -e "$BUILD_DIR" ]]; then
    echo "Refusing to overwrite an existing build directory: $BUILD_DIR" >&2
    exit 1
fi
# Source-derived recipe; the measured binary used separately rebuilt objects.
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_HIP_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++" \
    -DCMAKE_HIP_FLAGS="-DAPEX_V2_Q6_MUL24=1 -DAPEX_V2_Q6_VERIFY_ROWS2=1 -DAPEX_V2_AR_PEER_COPY=1" \
    -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 \
    -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_RCCL=OFF \
    -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    '-DGGML_CUDA_FA_QUANTS=q8_0-q8_0;q4_0-q4_0;f16-f16;bf16-bf16' \
    -DGGML_NATIVE=ON
cmake --build "$BUILD_DIR" --parallel "${APEX_BUILD_JOBS:-4}" --target llama-server test-backend-ops
