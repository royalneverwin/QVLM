#!/usr/bin/env bash
# Thor 设备上评估 ScienceQA 推理结果
# 可以在推理中途运行，只评估已完成的 batch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

OUTPUTS_DIR="${SCIENCEQA_OUTPUTS_DIR:-${DEVICE_SCIENCEQA_EVAL_DIR}/outputs}"
METADATA_FILE="${SCIENCEQA_METADATA_FILE:-${DEVICE_SCIENCEQA_EVAL_DIR}/metadata.json}"
# 默认结果文件放在对应 outputs 目录里，避免不同实验互相覆盖
RESULT_FILE="${SCIENCEQA_RESULT_FILE:-${OUTPUTS_DIR}/results.json}"

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

TIMING_LOG="${OUTPUTS_DIR}/timing.log"
if [[ -f "${TIMING_LOG}" ]]; then
    echo
    python3 - "${TIMING_LOG}" "${RESULT_FILE}" <<'PY'
import json
import re
import sys
from pathlib import Path

timing_log = Path(sys.argv[1])
result_file = Path(sys.argv[2])

patterns = {
    "visual": re.compile(r"(?:^|[, ])visual=([0-9.]+)\s*ms"),
    "llm": re.compile(r"(?:^|[, ])llm=([0-9.]+)\s*ms"),
    "total": re.compile(r"(?:^|[, ])total=([0-9.]+)\s*ms"),
}

values = {key: [] for key in patterns}
timed_batches = 0

for line in timing_log.read_text().splitlines():
    if "FAILED" in line:
        continue
    matched = {}
    for key, pattern in patterns.items():
        match = pattern.search(line)
        if match:
            matched[key] = float(match.group(1))
    if matched:
        timed_batches += 1
        for key, value in matched.items():
            values[key].append(value)

def average(items):
    return sum(items) / len(items) if items else None

timing_stats = {
    "timed_batches": timed_batches,
    "avg_visual_ms": average(values["visual"]),
    "avg_llm_ms": average(values["llm"]),
    "avg_total_ms": average(values["total"]),
}

print("╔══════════════════════════════════════════════╗")
print("║              推理耗时统计                    ║")
print("╠══════════════════════════════════════════════╣")
print(f"║  有计时 batch:  {timed_batches}")
for label, key in (("Visual", "avg_visual_ms"), ("LLM", "avg_llm_ms"), ("Total", "avg_total_ms")):
    value = timing_stats[key]
    text = "N/A" if value is None else f"{value:.2f} ms"
    print(f"║  平均 {label:6s}: {text}")
print("╚══════════════════════════════════════════════╝")

if result_file.exists():
    with result_file.open() as f:
        result = json.load(f)
    result["timing_stats"] = timing_stats
    with result_file.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
PY
fi
