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

# 默认 baseline 模式（不使用 vendored code）
# 传 --use-vendored-code 启用自定义代码（VisionZip/QAPruner）
USE_VENDORED_CODE="${USE_VENDORED_CODE:-false}"

VENDORED_ARGS=()
if [[ "${USE_VENDORED_CODE}" == "true" ]]; then
    VENDORED_ARGS+=(
        --vendored-code-dir "${VENDORED_QWEN2_VL_CODE_DIR}"
        --use-vendored-code
        --refresh-vendored-code
    )
fi

python3 "${SCRIPT_DIR}/prepare_local_model.py" \
  --model-id "${MODEL_ID}" \
  --local-model-dir "${LOCAL_MODEL_DIR}" \
  "${VENDORED_ARGS[@]+"${VENDORED_ARGS[@]}"}"

echo
echo "Local model preparation finished:"
echo "  Local model dir:    ${LOCAL_MODEL_DIR}"
if [[ "${USE_VENDORED_CODE}" == "true" ]]; then
    echo "  Vendored code dir:  ${VENDORED_QWEN2_VL_CODE_DIR}"
    echo "  Mode: vendored code (custom model)"
else
    echo "  Mode: baseline (transformers built-in)"
    echo
    echo "  To use vendored code: USE_VENDORED_CODE=true bash $0"
fi
