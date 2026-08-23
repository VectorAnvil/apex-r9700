#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LLAMA_DIR="${1:?usage: apply-series.sh /path/to/llama.cpp}"
readonly EXPECTED_BASE="4695f001fece1660d8bb1b3748f50726ddcc100b"

git -C "$LLAMA_DIR" rev-parse --git-dir >/dev/null 2>&1 || {
    echo "Not a Git checkout: $LLAMA_DIR" >&2
    exit 1
}
[[ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" == "$EXPECTED_BASE" ]] || {
    echo "Expected llama.cpp base $EXPECTED_BASE" >&2
    echo "Found: $(git -C "$LLAMA_DIR" rev-parse HEAD)" >&2
    exit 1
}
[[ -z "$(git -C "$LLAMA_DIR" status --porcelain)" ]] || {
    echo "Refusing to patch a dirty llama.cpp checkout." >&2
    exit 1
}

while IFS= read -r patch_name; do
    [[ -n "$patch_name" && "$patch_name" != \#* ]] || continue
    patch_path="$REPO_ROOT/patches/$patch_name"
    echo "Applying $patch_name"
    git -C "$LLAMA_DIR" apply --check "$patch_path"
    git -C "$LLAMA_DIR" apply "$patch_path"
done < "$REPO_ROOT/patches/series"

echo "Patch series applied successfully."
git -C "$LLAMA_DIR" status --short
