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

need_cmd ssh
need_cmd scp

if [[ ! -d "${EXPORT_MODEL_DIR}/onnx" ]]; then
    echo "ONNX directory not found: ${EXPORT_MODEL_DIR}/onnx" >&2
    echo "Run x86_host/01_quantize_export.sh first." >&2
    exit 1
fi

echo "[1/2] Creating remote workspace"
ssh "${DEVICE_USER}@${DEVICE_HOST}" "mkdir -p ${DEVICE_MODEL_DIR}"

echo "[2/2] Transferring ONNX artifacts to ${DEVICE_USER}@${DEVICE_HOST}:${DEVICE_MODEL_DIR}"
scp -r "${EXPORT_MODEL_DIR}/onnx" "${DEVICE_USER}@${DEVICE_HOST}:${DEVICE_MODEL_DIR}/"

echo
echo "Transfer finished:"
echo "  Device model dir: ${DEVICE_MODEL_DIR}"
