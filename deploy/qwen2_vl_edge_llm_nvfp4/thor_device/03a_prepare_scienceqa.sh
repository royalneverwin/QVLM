#!/usr/bin/env bash
# Thor 设备上修正 ScienceQA 输入文件中的图片路径
# 从 x86 传过来的 batch JSON 中图片路径指向 x86 的目录，
# 需要替换为 Thor 设备上的路径。
#
# 前置条件：x86_host/02_transfer_onnx.sh 已传输数据到 Thor
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

INPUTS_DIR="${DEVICE_SCIENCEQA_EVAL_DIR}/inputs"

if [[ ! -d "${INPUTS_DIR}" ]]; then
    echo "ScienceQA 输入目录不存在: ${INPUTS_DIR}" >&2
    echo "请先通过 x86_host/02_transfer_onnx.sh 传输数据" >&2
    exit 1
fi

echo "修正 ScienceQA 输入文件中的图片路径..."
echo "  x86 数据目录:  ${SCIENCEQA_DATA_DIR}"
echo "  Thor 数据目录: ${DEVICE_SCIENCEQA_DATA_DIR}"

# 替换 batch JSON 文件中的 x86 路径为 Thor 路径
COUNT=0
for f in "${INPUTS_DIR}"/batch_*.json; do
    if grep -q "${SCIENCEQA_DATA_DIR}" "$f" 2>/dev/null; then
        sed -i "s|${SCIENCEQA_DATA_DIR}|${DEVICE_SCIENCEQA_DATA_DIR}|g" "$f"
        COUNT=$((COUNT + 1))
    fi
done

echo "已修正 ${COUNT} 个文件"
echo "完成。"
