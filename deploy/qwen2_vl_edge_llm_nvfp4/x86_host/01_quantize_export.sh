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

need_cmd tensorrt-edgellm-quantize-llm
need_cmd tensorrt-edgellm-export-llm
need_cmd tensorrt-edgellm-export-visual

if [[ ! -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
    echo "Prepared local model not found: ${LOCAL_MODEL_DIR}/config.json" >&2
    echo "Run x86_host/00_prepare_local_model.sh first." >&2
    exit 1
fi

mkdir -p "${EXPORT_QUANTIZED_LLM_DIR}" "${EXPORT_ONNX_LLM_DIR}" "${EXPORT_ONNX_VISUAL_DIR}"

echo "[1/3] Quantizing Qwen2-VL LLM to ${QUANTIZATION}"
tensorrt-edgellm-quantize-llm \
  --model_dir "${LOCAL_MODEL_DIR}" \
  --quantization "${QUANTIZATION}" \
  --output_dir "${EXPORT_QUANTIZED_LLM_DIR}"

echo "[2/3] Exporting quantized LLM to ONNX"
tensorrt-edgellm-export-llm \
  --model_dir "${EXPORT_QUANTIZED_LLM_DIR}" \
  --output_dir "${EXPORT_ONNX_LLM_DIR}"

echo "[3/3] Exporting visual encoder from original checkpoint"
tensorrt-edgellm-export-visual \
  --model_dir "${LOCAL_MODEL_DIR}" \
  --output_dir "${EXPORT_ONNX_VISUAL_DIR}"

echo
echo "Export finished:"
echo "  Quantized LLM: ${EXPORT_QUANTIZED_LLM_DIR}"
echo "  LLM ONNX:      ${EXPORT_ONNX_LLM_DIR}"
echo "  Visual ONNX:   ${EXPORT_ONNX_VISUAL_DIR}"
