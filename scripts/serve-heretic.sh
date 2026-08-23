#!/usr/bin/env bash
set -Eeuo pipefail

readonly CONFIG="${1:?usage: serve-heretic.sh /path/to/production.env [/path/to/build]}"
readonly BUILD_DIR="${2:-./build}"

set -a
# shellcheck source=/dev/null
source "$CONFIG"
set +a

readonly SERVER="${APEX_LLAMA_SERVER:-$BUILD_DIR/bin/llama-server}"
: "${APEX_MODEL:?set APEX_MODEL in the configuration file}"
: "${APEX_MMPROJ:?set APEX_MMPROJ in the configuration file}"

[[ -x "$SERVER" ]] || { echo "Missing llama-server: $SERVER" >&2; exit 1; }
[[ -r "$APEX_MODEL" ]] || { echo "Missing model: $APEX_MODEL" >&2; exit 1; }
[[ -r "$APEX_MMPROJ" ]] || { echo "Missing projector: $APEX_MMPROJ" >&2; exit 1; }

check_hash() {
    local file="$1" expected="$2" label="$3" actual
    [[ -n "$expected" ]] || return 0
    actual="$(sha256sum "$file" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "$label hash mismatch: expected $expected, got $actual" >&2
        exit 1
    }
}

check_hash "$SERVER" "${APEX_SERVER_SHA256:-}" "llama-server"
check_hash "$APEX_MODEL" "${APEX_MODEL_SHA256:-}" "model"
check_hash "$APEX_MMPROJ" "${APEX_MMPROJ_SHA256:-}" "projector"

unset ROCR_VISIBLE_DEVICES GGML_CUDA_DISABLE_GRAPHS LLAMA_ARG_FIT
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
export GGML_CUDA_FA1_MIN_NQ=32
export GGML_CUDA_Q6K_SWIGLU_D4_FFN_DOWN=1
export GGML_CUDA_Q6K_SHARED_Q8_GATE_UP=1
export GGML_CUDA_Q6K_CANONICAL_ROWS=1
export GGML_CUDA_Q6K_CANONICAL_STAGE=3
export GGML_CUDA_Q6K_CANONICAL_CONTROL_STAGE=2
export GGML_CUDA_Q6K_CANONICAL_FFN_STAGE=3
export GGML_CUDA_Q6K_CANONICAL_FFN_PACKED=1
export GGML_CUDA_Q6K_CANONICAL_PACKED_ALL=1
export GGML_CUDA_Q6K_CANONICAL_ATTN_STAGE=4
export GGML_CUDA_FA_CANONICAL_VEC=1
export APEX_SPEC_KV_PAD_GUARD=256
export APEX_RECURRENT_NATURAL_SPLIT=1

exec "$SERVER" \
    --model "$APEX_MODEL" \
    --mmproj "$APEX_MMPROJ" \
    --host "${APEX_HOST:-127.0.0.1}" \
    --port "${APEX_PORT:-8083}" \
    --ctx-size "${APEX_CTX_SIZE:-262144}" \
    --parallel 1 \
    --threads "${APEX_THREADS:-12}" \
    --batch-size 2048 \
    --ubatch-size 512 \
    --n-gpu-layers 99 \
    --split-mode tensor \
    --tensor-split 1,1 \
    --flash-attn on \
    --fit off \
    --no-warmup \
    --jinja \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cache-ram "${APEX_CACHE_RAM_MIB:-20480}" \
    --cache-prompt \
    --cache-idle-slots \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --spec-draft-n-min 0
