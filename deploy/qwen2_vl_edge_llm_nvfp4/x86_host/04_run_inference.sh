#!/usr/bin/env bash
# x86 主机上批量推理 ScienceQA 并评估准确率
# 前置条件：
#   - 已完成 x86_host/03_build_engines.sh（engine 已构建）
#   - 已完成 x86_host/03a_prepare_scienceqa.sh（数据已准备）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

LLM_INFER_BIN="${EDGE_LLM_REPO}/build/examples/llm/llm_inference"

if [[ ! -x "${LLM_INFER_BIN}" ]]; then
    echo "Missing llm_inference binary: ${LLM_INFER_BIN}" >&2
    exit 1
fi

# 检查是否有 ScienceQA 输入
INPUTS_DIR="${SCIENCEQA_EVAL_DIR}/inputs"
OUTPUTS_DIR="${SCIENCEQA_EVAL_DIR}/outputs"

if [[ ! -d "${INPUTS_DIR}" ]]; then
    echo "ScienceQA 输入目录不存在: ${INPUTS_DIR}" >&2
    echo "请先运行: bash x86_host/03a_prepare_scienceqa.sh" >&2
    exit 1
fi

mkdir -p "${OUTPUTS_DIR}"

# 统计输入文件数
INPUT_FILES=("${INPUTS_DIR}"/batch_*.json)
TOTAL=${#INPUT_FILES[@]}

if [[ ${TOTAL} -eq 0 ]]; then
    echo "没有找到输入文件: ${INPUTS_DIR}/batch_*.json" >&2
    exit 1
fi

echo "============================================"
echo " ScienceQA 推理 (TensorRT Edge-LLM)"
echo "============================================"
echo "  Engine LLM:    ${X86_ENGINE_LLM_DIR}"
echo "  Engine Visual: ${X86_ENGINE_VISUAL_DIR}"
echo "  输入目录:      ${INPUTS_DIR}"
echo "  输出目录:      ${OUTPUTS_DIR}"
echo "  总 batch 数:   ${TOTAL}"
echo "============================================"

# 逐 batch 推理
FAILED=0
for ((i=0; i<TOTAL; i++)); do
    INPUT_FILE="${INPUT_FILES[$i]}"
    BASENAME="$(basename "${INPUT_FILE}")"
    OUTPUT_FILE="${OUTPUTS_DIR}/${BASENAME}"

    # 跳过已完成的
    if [[ -f "${OUTPUT_FILE}" ]]; then
        continue
    fi

    printf "\r[%d/%d] %s" $((i+1)) "${TOTAL}" "${BASENAME}"

    if ! "${LLM_INFER_BIN}" \
        --engineDir "${X86_ENGINE_LLM_DIR}" \
        --multimodalEngineDir "${X86_ENGINE_VISUAL_DIR}" \
        --plugin "${EDGELLM_PLUGIN_PATH}" \
        --inputFile "${INPUT_FILE}" \
        --outputFile "${OUTPUT_FILE}" \
        2>> "${OUTPUTS_DIR}/inference_errors.log"; then
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "推理完成: 成功 $((TOTAL - FAILED))/${TOTAL}, 失败 ${FAILED}"
echo

# 评估准确率
echo "============================================"
echo " 评估 ScienceQA 准确率"
echo "============================================"

python3 "${SCRIPT_DIR}/evaluate_scienceqa.py" \
    --metadata "${SCIENCEQA_EVAL_DIR}/metadata.json" \
    --outputs-dir "${OUTPUTS_DIR}" \
    --result-file "${SCIENCEQA_EVAL_DIR}/results.json"
