#!/usr/bin/env bash
# x86 主机上本地构建 TensorRT engine（不经过 Thor 设备）
# 前置条件：
#   - 已完成 01_quantize_export.sh（ONNX 已导出）
#   - TensorRT Edge-LLM C++ 项目已在 x86 上编译
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

LLM_BUILD_BIN="${EDGE_LLM_REPO}/build/examples/llm/llm_build"
VISUAL_BUILD_BIN="${EDGE_LLM_REPO}/build/examples/multimodal/visual_build"

if [[ ! -x "${LLM_BUILD_BIN}" ]]; then
    echo "Missing llm_build binary: ${LLM_BUILD_BIN}" >&2
    echo "Please build TensorRT-Edge-LLM C++ project first:" >&2
    echo "  cd ${EDGE_LLM_REPO} && mkdir -p build && cd build" >&2
    echo "  cmake .. -DTRT_PACKAGE_DIR=/usr && make -j\$(nproc)" >&2
    exit 1
fi

if [[ ! -x "${VISUAL_BUILD_BIN}" ]]; then
    echo "Missing visual_build binary: ${VISUAL_BUILD_BIN}" >&2
    exit 1
fi

if [[ ! -d "${EXPORT_ONNX_LLM_DIR}" ]]; then
    echo "ONNX LLM directory not found: ${EXPORT_ONNX_LLM_DIR}" >&2
    echo "Run x86_host/01_quantize_export.sh first." >&2
    exit 1
fi

mkdir -p "${X86_ENGINE_LLM_DIR}" "${X86_ENGINE_VISUAL_DIR}"

# 检查 LLM engine 是否已存在
LLM_ENGINE_EXISTS=false
if find "${X86_ENGINE_LLM_DIR}" -name "*.engine" 2>/dev/null | grep -q .; then
    LLM_ENGINE_EXISTS=true
fi

# 检查 visual engine 是否已存在
VISUAL_ENGINE_EXISTS=false
if find "${X86_ENGINE_VISUAL_DIR}" -name "*.engine" 2>/dev/null | grep -q .; then
    VISUAL_ENGINE_EXISTS=true
fi

if [[ "${LLM_ENGINE_EXISTS}" == "true" ]]; then
    echo "[1/2] LLM engine already exists, skipping (delete ${X86_ENGINE_LLM_DIR} to rebuild)"
else
    echo "[1/2] Building LLM engine on x86"
    "${LLM_BUILD_BIN}" \
      --onnxDir "${EXPORT_ONNX_LLM_DIR}" \
      --engineDir "${X86_ENGINE_LLM_DIR}" \
      --maxBatchSize "${MAX_BATCH_SIZE}" \
      --maxInputLen "${MAX_INPUT_LEN}" \
      --maxKVCacheCapacity "${MAX_KV_CACHE_CAPACITY}"
fi

if [[ "${VISUAL_ENGINE_EXISTS}" == "true" ]]; then
    echo "[2/2] Visual engine already exists, skipping (delete ${X86_ENGINE_VISUAL_DIR} to rebuild)"
else
    echo "[2/2] Building visual engine on x86"
    "${VISUAL_BUILD_BIN}" \
      --onnxDir "${EXPORT_ONNX_VISUAL_DIR}" \
      --engineDir "${X86_ENGINE_VISUAL_DIR}" \
      --minImageTokens "${MIN_IMAGE_TOKENS}" \
      --maxImageTokens "${MAX_IMAGE_TOKENS}" \
      --maxImageTokensPerImage "${MAX_IMAGE_TOKENS_PER_IMAGE}"
fi

echo
echo "Build finished (x86 local):"
echo "  LLM engine dir:    ${X86_ENGINE_LLM_DIR}"
echo "  Visual engine dir: ${X86_ENGINE_VISUAL_DIR}"
