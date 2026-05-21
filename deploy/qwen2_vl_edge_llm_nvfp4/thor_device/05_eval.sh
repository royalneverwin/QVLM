#!/usr/bin/env bash
# Thor 设备上评估 ScienceQA 推理结果
# 可以在推理中途运行，只评估已完成的 batch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

OUTPUTS_DIR="${DEVICE_SCIENCEQA_EVAL_DIR}/outputs"
METADATA_FILE="${DEVICE_SCIENCEQA_EVAL_DIR}/metadata.json"
RESULT_FILE="${DEVICE_SCIENCEQA_EVAL_DIR}/results.json"

# evaluate_scienceqa.py 在 x86_host 目录下，Thor 上也能用（仓库共享）
EVAL_SCRIPT="${DEPLOY_DIR}/x86_host/evaluate_scienceqa.py"

if [[ ! -f "${METADATA_FILE}" ]]; then
    echo "Metadata 文件不存在: ${METADATA_FILE}" >&2
    echo "请先运行: bash thor_device/03a_prepare_scienceqa.sh" >&2
    exit 1
fi

if [[ ! -d "${OUTPUTS_DIR}" ]]; then
    echo "输出目录不存在: ${OUTPUTS_DIR}" >&2
    echo "请先运行: bash thor_device/04_run_inference.sh" >&2
    exit 1
fi

# 统计已有输出文件数
COMPLETED=$(find "${OUTPUTS_DIR}" -name "batch_*.json" | wc -l)
TOTAL_INPUTS=$(find "${DEVICE_SCIENCEQA_EVAL_DIR}/inputs" -name "batch_*.json" 2>/dev/null | wc -l)

echo "╔══════════════════════════════════════════════╗"
echo "║      ScienceQA 准确率评估 - Thor              ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  已完成输出: ${COMPLETED} / ${TOTAL_INPUTS} batches"
echo "║  输出目录:   ${OUTPUTS_DIR}"
echo "║  结果文件:   ${RESULT_FILE}"
echo "╚══════════════════════════════════════════════╝"
echo

python3 "${EVAL_SCRIPT}" \
    --metadata "${METADATA_FILE}" \
    --outputs-dir "${OUTPUTS_DIR}" \
    --result-file "${RESULT_FILE}"
