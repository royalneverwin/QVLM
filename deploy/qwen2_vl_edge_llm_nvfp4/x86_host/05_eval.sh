#!/usr/bin/env bash
# 评估已有的 ScienceQA 推理输出结果
# 可以在推理中途运行，只评估已完成的 batch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

OUTPUTS_DIR="${SCIENCEQA_EVAL_DIR}/outputs"
METADATA_FILE="${SCIENCEQA_EVAL_DIR}/metadata.json"
RESULT_FILE="${SCIENCEQA_EVAL_DIR}/results.json"

if [[ ! -f "${METADATA_FILE}" ]]; then
    echo "Metadata 文件不存在: ${METADATA_FILE}" >&2
    echo "请先运行: bash x86_host/03a_prepare_scienceqa.sh" >&2
    exit 1
fi

if [[ ! -d "${OUTPUTS_DIR}" ]]; then
    echo "输出目录不存在: ${OUTPUTS_DIR}" >&2
    echo "请先运行: bash x86_host/04_run_inference.sh" >&2
    exit 1
fi

# 统计已有输出文件数
COMPLETED=$(find "${OUTPUTS_DIR}" -name "batch_*.json" | wc -l)
TOTAL_INPUTS=$(find "${SCIENCEQA_EVAL_DIR}/inputs" -name "batch_*.json" 2>/dev/null | wc -l)

echo "╔══════════════════════════════════════════════╗"
echo "║         ScienceQA 准确率评估                  ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  已完成输出: ${COMPLETED} / ${TOTAL_INPUTS} batches"
echo "║  输出目录:   ${OUTPUTS_DIR}"
echo "║  结果文件:   ${RESULT_FILE}"
echo "╚══════════════════════════════════════════════╝"
echo

python3 "${SCRIPT_DIR}/evaluate_scienceqa.py" \
    --metadata "${METADATA_FILE}" \
    --outputs-dir "${OUTPUTS_DIR}" \
    --result-file "${RESULT_FILE}"
