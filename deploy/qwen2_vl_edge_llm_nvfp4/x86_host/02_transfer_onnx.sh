#!/usr/bin/env bash
# 从 x86 主机传输 ONNX 文件和 ScienceQA 数据到 Thor 设备
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

need_cmd ssh
need_cmd scp

REMOTE="${DEVICE_USER}@${DEVICE_HOST}"

# ─── 1. 传输 ONNX 文件 ───

if [[ ! -d "${EXPORT_MODEL_DIR}/onnx" ]]; then
    echo "ONNX directory not found: ${EXPORT_MODEL_DIR}/onnx" >&2
    echo "Run x86_host/01_quantize_export.sh first." >&2
    exit 1
fi

echo "[1/3] Creating remote workspace"
ssh "${REMOTE}" "mkdir -p '${DEVICE_MODEL_DIR}'"

echo "[2/3] Transferring ONNX artifacts"
scp -r "${EXPORT_MODEL_DIR}/onnx" "${REMOTE}:${DEVICE_MODEL_DIR}/"

# ─── 2. 传输 ScienceQA 数据（如果存在） ───

if [[ -d "${SCIENCEQA_EVAL_DIR}/inputs" ]]; then
    echo "[3/3] Transferring ScienceQA evaluation data"
    ssh "${REMOTE}" "mkdir -p '${DEVICE_SCIENCEQA_EVAL_DIR}'"

    # 传输 inputs 目录和 metadata.json
    scp -r "${SCIENCEQA_EVAL_DIR}/inputs" "${REMOTE}:${DEVICE_SCIENCEQA_EVAL_DIR}/"
    scp "${SCIENCEQA_EVAL_DIR}/metadata.json" "${REMOTE}:${DEVICE_SCIENCEQA_EVAL_DIR}/"

    # 传输 ScienceQA 图片数据（推理需要）
    if [[ -d "${SCIENCEQA_DATA_DIR}" ]]; then
        ssh "${REMOTE}" "mkdir -p '${DEVICE_SCIENCEQA_DATA_DIR}'"
        scp -r "${SCIENCEQA_DATA_DIR}/images" "${REMOTE}:${DEVICE_SCIENCEQA_DATA_DIR}/"
    fi
else
    echo "[3/3] ScienceQA data not found, skipping (run x86_host/03a_prepare_scienceqa.sh first)"
fi

echo
echo "Transfer finished:"
echo "  Device model dir:     ${DEVICE_MODEL_DIR}"
echo "  Device ScienceQA dir: ${DEVICE_SCIENCEQA_EVAL_DIR}"
