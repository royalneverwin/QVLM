#!/usr/bin/env bash
# 下载 ScienceQA 数据集并生成 TensorRT Edge-LLM 推理输入文件
# 前置条件：pip install datasets pillow
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

echo "============================================"
echo " ScienceQA 数据准备"
echo "============================================"
echo "  数据目录: ${SCIENCEQA_DATA_DIR}"
echo "  输出目录: ${SCIENCEQA_EVAL_DIR}"
echo "  Split:    ${SCIENCEQA_SPLIT}"
echo "  最大样本: ${SCIENCEQA_MAX_SAMPLES} (0=全部)"
echo "============================================"

python3 "${SCRIPT_DIR}/prepare_scienceqa_inputs.py" \
    --data-dir "${SCIENCEQA_DATA_DIR}" \
    --output-dir "${SCIENCEQA_EVAL_DIR}" \
    --split "${SCIENCEQA_SPLIT}" \
    --batch-size 1 \
    --max-samples "${SCIENCEQA_MAX_SAMPLES}" \
    --max-generate-length 32

echo
echo "数据准备完成。下一步运行:"
echo "  bash x86_host/04_run_inference.sh"
