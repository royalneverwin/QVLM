#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

need_cmd python3

python3 "${SCRIPT_DIR}/prepare_local_model.py" \
  --model-id "${MODEL_ID}" \
  --local-model-dir "${LOCAL_MODEL_DIR}" \
  --vendored-code-dir "${VENDORED_QWEN2_VL_CODE_DIR}" \
  --refresh-vendored-code

echo
echo "Local model preparation finished:"
echo "  Local model dir:    ${LOCAL_MODEL_DIR}"
echo "  Vendored code dir:  ${VENDORED_QWEN2_VL_CODE_DIR}"
