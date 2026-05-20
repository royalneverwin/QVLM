#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

LLM_BUILD_BIN="${EDGE_LLM_REPO}/build/examples/llm/llm_build"
VISUAL_BUILD_BIN="${EDGE_LLM_REPO}/build/examples/multimodal/visual_build"

if [[ ! -x "${LLM_BUILD_BIN}" ]]; then
    echo "Missing llm_build binary: ${LLM_BUILD_BIN}" >&2
    exit 1
fi

if [[ ! -x "${VISUAL_BUILD_BIN}" ]]; then
    echo "Missing visual_build binary: ${VISUAL_BUILD_BIN}" >&2
    exit 1
fi

mkdir -p "${DEVICE_ENGINE_LLM_DIR}" "${DEVICE_ENGINE_VISUAL_DIR}"

echo "[1/2] Building LLM engine"
"${LLM_BUILD_BIN}" \
  --onnxDir "${DEVICE_ONNX_LLM_DIR}" \
  --engineDir "${DEVICE_ENGINE_LLM_DIR}" \
  --maxBatchSize "${MAX_BATCH_SIZE}" \
  --maxInputLen "${MAX_INPUT_LEN}" \
  --maxKVCacheCapacity "${MAX_KV_CACHE_CAPACITY}"

echo "[2/2] Building visual engine"
"${VISUAL_BUILD_BIN}" \
  --onnxDir "${DEVICE_ONNX_VISUAL_DIR}" \
  --engineDir "${DEVICE_ENGINE_VISUAL_DIR}" \
  --minImageTokens "${MIN_IMAGE_TOKENS}" \
  --maxImageTokens "${MAX_IMAGE_TOKENS}" \
  --maxImageTokensPerImage "${MAX_IMAGE_TOKENS_PER_IMAGE}"

echo
echo "Build finished:"
echo "  LLM engine dir:    ${DEVICE_ENGINE_LLM_DIR}"
echo "  Visual engine dir: ${DEVICE_ENGINE_VISUAL_DIR}"
