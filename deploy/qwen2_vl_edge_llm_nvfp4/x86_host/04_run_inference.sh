#!/usr/bin/env bash
# x86 主机上本地运行多模态推理（不经过 Thor 设备）
# 前置条件：
#   - 已完成 x86_host/03_build_engines.sh（engine 已构建）
#   - 输入文件 $X86_INPUT_FILE 已准备好（可从 templates/input_vlm.json 拷贝修改）
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

# 如果输入文件不存在，从模板拷贝一份
if [[ ! -f "${X86_INPUT_FILE}" ]]; then
    echo "Input file not found: ${X86_INPUT_FILE}"
    echo "Copying starter template from: ${TEMPLATE_INPUT}"
    cp "${TEMPLATE_INPUT}" "${X86_INPUT_FILE}"
    echo "请编辑 ${X86_INPUT_FILE} 中的图片路径后重新运行。"
    exit 0
fi

echo "Running inference on x86..."
"${LLM_INFER_BIN}" \
  --engineDir "${X86_ENGINE_LLM_DIR}" \
  --multimodalEngineDir "${X86_ENGINE_VISUAL_DIR}" \
  --inputFile "${X86_INPUT_FILE}" \
  --outputFile "${X86_OUTPUT_FILE}"

echo
echo "Inference finished (x86 local):"
echo "  Input:  ${X86_INPUT_FILE}"
echo "  Output: ${X86_OUTPUT_FILE}"
