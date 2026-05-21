#!/usr/bin/env bash
# Thor 设备上构建 TensorRT engine：LLM 使用原 NVFP4 导出，visual 使用 VisionZip 导出。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

LLM_BUILD_BIN="${EDGE_LLM_REPO}/build/examples/llm/llm_build"
VISUAL_BUILD_BIN="${EDGE_LLM_REPO}/build/examples/multimodal/visual_build"

derive_keep_tokens() {
    if [[ -n "${VISIONZIP_KEEP_TOKENS}" ]]; then
        echo "${VISIONZIP_KEEP_TOKENS}"
    else
        python3 -c 'import sys; r=float(sys.argv[1]); m=int(sys.argv[2]); print(max(1, int(round(m * (1.0 - r)))))' \
          "${VISIONZIP_PRUNE_RATE}" "${MAX_IMAGE_TOKENS_PER_IMAGE}"
    fi
}

if [[ ! -x "${LLM_BUILD_BIN}" ]]; then
    echo "Missing llm_build binary: ${LLM_BUILD_BIN}" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Missing required command: python3" >&2
    exit 1
fi

if [[ ! -x "${VISUAL_BUILD_BIN}" ]]; then
    echo "Missing visual_build binary: ${VISUAL_BUILD_BIN}" >&2
    exit 1
fi

if [[ ! -d "${DEVICE_ONNX_LLM_DIR}" ]]; then
    echo "ONNX LLM directory not found: ${DEVICE_ONNX_LLM_DIR}" >&2
    echo "Run x86_host/01_quantize_export_visionzip.sh and x86_host/02_transfer_onnx.sh first." >&2
    exit 1
fi

if [[ ! -d "${DEVICE_ONNX_VISUAL_VISIONZIP_DIR}" ]]; then
    echo "VisionZip visual ONNX directory not found: ${DEVICE_ONNX_VISUAL_VISIONZIP_DIR}" >&2
    echo "Run x86_host/01_quantize_export_visionzip.sh and x86_host/02_transfer_onnx.sh first." >&2
    exit 1
fi

VISIONZIP_EFFECTIVE_KEEP_TOKENS="$(derive_keep_tokens)"
VISIONZIP_BUILD_MIN_IMAGE_TOKENS="$(python3 -c 'import sys; print(max(int(sys.argv[1]), int(sys.argv[2])))' \
  "${MIN_IMAGE_TOKENS}" "${VISIONZIP_EFFECTIVE_KEEP_TOKENS}")"

if (( VISIONZIP_EFFECTIVE_KEEP_TOKENS > MAX_IMAGE_TOKENS_PER_IMAGE )); then
    echo "VisionZip keep_tokens=${VISIONZIP_EFFECTIVE_KEEP_TOKENS} exceeds MAX_IMAGE_TOKENS_PER_IMAGE=${MAX_IMAGE_TOKENS_PER_IMAGE}" >&2
    exit 1
fi

mkdir -p "${DEVICE_ENGINE_LLM_DIR}" "${DEVICE_ENGINE_VISUAL_VISIONZIP_DIR}"

LLM_ENGINE_EXISTS=false
if find "${DEVICE_ENGINE_LLM_DIR}" -name "*.engine" 2>/dev/null | grep -q .; then
    LLM_ENGINE_EXISTS=true
fi

VISUAL_ENGINE_EXISTS=false
if find "${DEVICE_ENGINE_VISUAL_VISIONZIP_DIR}" -name "*.engine" 2>/dev/null | grep -q .; then
    VISUAL_ENGINE_EXISTS=true
fi

if [[ "${LLM_ENGINE_EXISTS}" == "true" ]]; then
    echo "[1/2] LLM engine already exists, skipping (delete ${DEVICE_ENGINE_LLM_DIR} to rebuild)"
else
    echo "[1/2] Building LLM engine"
    "${LLM_BUILD_BIN}" \
      --onnxDir "${DEVICE_ONNX_LLM_DIR}" \
      --engineDir "${DEVICE_ENGINE_LLM_DIR}" \
      --maxBatchSize "${MAX_BATCH_SIZE}" \
      --maxInputLen "${MAX_INPUT_LEN}" \
      --maxKVCacheCapacity "${MAX_KV_CACHE_CAPACITY}"
fi

if [[ "${VISUAL_ENGINE_EXISTS}" == "true" ]]; then
    echo "[2/2] VisionZip visual engine already exists, skipping (delete ${DEVICE_ENGINE_VISUAL_VISIONZIP_DIR} to rebuild)"
else
    echo "[2/2] Building VisionZip visual engine"
    echo "      prune_rate=${VISIONZIP_PRUNE_RATE}, keep_tokens=${VISIONZIP_EFFECTIVE_KEEP_TOKENS}, minImageTokens=${VISIONZIP_BUILD_MIN_IMAGE_TOKENS}"
    "${VISUAL_BUILD_BIN}" \
      --onnxDir "${DEVICE_ONNX_VISUAL_VISIONZIP_DIR}" \
      --engineDir "${DEVICE_ENGINE_VISUAL_VISIONZIP_DIR}" \
      --minImageTokens "${VISIONZIP_BUILD_MIN_IMAGE_TOKENS}" \
      --maxImageTokens "${MAX_IMAGE_TOKENS}" \
      --maxImageTokensPerImage "${MAX_IMAGE_TOKENS_PER_IMAGE}"
fi

echo
echo "VisionZip build finished:"
echo "  LLM engine dir:       ${DEVICE_ENGINE_LLM_DIR}"
echo "  Visual engine dir:    ${DEVICE_ENGINE_VISUAL_VISIONZIP_DIR}"
echo "  Runtime override:     DEVICE_ENGINE_VISUAL_DIR=${DEVICE_ENGINE_VISUAL_VISIONZIP_DIR}"
