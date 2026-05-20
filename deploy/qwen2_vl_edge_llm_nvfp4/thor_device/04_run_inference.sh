#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

LLM_INFER_BIN="${EDGE_LLM_REPO}/build/examples/llm/llm_inference"
TEMPLATE_INPUT="${DEPLOY_DIR}/templates/input_vlm.json"

if [[ ! -x "${LLM_INFER_BIN}" ]]; then
    echo "Missing llm_inference binary: ${LLM_INFER_BIN}" >&2
    exit 1
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Input file not found: ${INPUT_FILE}" >&2
    echo "Starter template available at: ${TEMPLATE_INPUT}" >&2
    exit 1
fi

"${LLM_INFER_BIN}" \
  --engineDir "${DEVICE_ENGINE_LLM_DIR}" \
  --multimodalEngineDir "${DEVICE_ENGINE_VISUAL_DIR}" \
  --inputFile "${INPUT_FILE}" \
  --outputFile "${OUTPUT_FILE}"

echo
echo "Inference finished:"
echo "  Input:  ${INPUT_FILE}"
echo "  Output: ${OUTPUT_FILE}"
