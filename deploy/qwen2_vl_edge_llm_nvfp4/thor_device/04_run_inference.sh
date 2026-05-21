#!/usr/bin/env bash
# Thor 设备上批量推理 ScienceQA
# 前置条件：
#   - 已完成 thor_device/03_build_engines.sh（engine 已构建）
#   - 已完成 thor_device/03a_prepare_scienceqa.sh（图片路径已修正）

# DEVICE_ENGINE_VISUAL_DIR="/home/vdig/wangxinhao/QVLM/tensorrt-edgellm-workspace/Qwen2-VL-7B-Instruct/engines/visual_visionzip_prune_0p5" \
# SCIENCEQA_OUTPUTS_DIR="/home/vdig/wangxinhao/QVLM/tensorrt-edgellm-workspace/scienceqa_eval/outputs_visionzip_prune_0p5" \
# bash /home/vdig/wangxinhao/QVLM/deploy/qwen2_vl_edge_llm_nvfp4/thor_device/04_run_inference.sh

# SCIENCEQA_OUTPUTS_DIR="/home/vdig/wangxinhao/QVLM/tensorrt-edgellm-workspace/scienceqa_eval/outputs_baseline" \
# bash /home/vdig/wangxinhao/QVLM/deploy/qwen2_vl_edge_llm_nvfp4/thor_device/04_run_inference.sh


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
INPUTS_DIR="${SCIENCEQA_INPUTS_DIR:-${DEVICE_SCIENCEQA_EVAL_DIR}/inputs}"
OUTPUTS_DIR="${SCIENCEQA_OUTPUTS_DIR:-${DEVICE_SCIENCEQA_EVAL_DIR}/outputs}"

if [[ ! -d "${INPUTS_DIR}" ]]; then
    echo "ScienceQA 输入目录不存在: ${INPUTS_DIR}" >&2
    echo "请先运行: bash thor_device/03a_prepare_scienceqa.sh" >&2
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

# 时间记录文件
TIMING_LOG="${OUTPUTS_DIR}/timing.log"
ERROR_LOG="${OUTPUTS_DIR}/inference_errors.log"
TOKEN_LOG="${OUTPUTS_DIR}/visionzip_tokens.log"
> "${TIMING_LOG}"
> "${ERROR_LOG}"
> "${TOKEN_LOG}"

echo "╔══════════════════════════════════════════════╗"
echo "║   ScienceQA 推理 - Thor (TensorRT Edge-LLM)  ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Engine LLM:    ${DEVICE_ENGINE_LLM_DIR}"
echo "║  Engine Visual: ${DEVICE_ENGINE_VISUAL_DIR}"
echo "║  输入目录:      ${INPUTS_DIR}"
echo "║  输出目录:      ${OUTPUTS_DIR}"
echo "║  总 batch 数:   ${TOTAL}"
echo "╚══════════════════════════════════════════════╝"
echo

# 逐 batch 推理
FAILED=0
SUCCEEDED=0
SKIPPED=0
TOTAL_TIME=0

for ((i=0; i<TOTAL; i++)); do
    INPUT_FILE="${INPUT_FILES[$i]}"
    BASENAME="$(basename "${INPUT_FILE}")"
    OUTPUT_FILE="${OUTPUTS_DIR}/${BASENAME}"

    # 跳过已完成的
    if [[ -f "${OUTPUT_FILE}" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "────────────────────────────────────────────────"
    echo "[$((i+1))/${TOTAL}] ${BASENAME}"
    echo "────────────────────────────────────────────────"

    # 打印输入 prompt（提取 user message 中的文字部分）
    PROMPT=$(python3 -c "
import json
with open('${INPUT_FILE}') as f:
    d = json.load(f)
requests = d.get('requests', [d]) if isinstance(d, dict) else d
for r in requests[:1]:
    if 'input_text' in r:
        text = r['input_text']
    elif 'messages' in r:
        text = ''
        for msg in r['messages']:
            if msg.get('role') != 'user':
                continue
            content = msg.get('content', '')
            if isinstance(content, list):
                parts = [p.get('text', '') for p in content if p.get('type') == 'text']
                text = ' '.join(parts)
            elif isinstance(content, str):
                text = content
    else:
        text = r.get('prompt', '')
    print(text)
" 2>/dev/null || echo "[无法解析输入]")
    echo "  Prompt: ${PROMPT}"
    echo

    # 计时推理，抑制 TensorRT INFO 输出；完整 stderr 仍按 batch 保存用于调试
    STDERR_FILE="${OUTPUTS_DIR}/${BASENAME%.json}.stderr"
    START_TIME=$(date +%s%N)

    if "${LLM_INFER_BIN}" \
        --engineDir "${DEVICE_ENGINE_LLM_DIR}" \
        --multimodalEngineDir "${DEVICE_ENGINE_VISUAL_DIR}" \
        --inputFile "${INPUT_FILE}" \
        --outputFile "${OUTPUT_FILE}" \
        > /dev/null \
        2> "${STDERR_FILE}"; then

        END_TIME=$(date +%s%N)
        ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
        TOTAL_TIME=$((TOTAL_TIME + ELAPSED_MS))
        SUCCEEDED=$((SUCCEEDED + 1))

        TOKEN_LINE="$(grep -F "VisionZip tokens:" "${STDERR_FILE}" | tail -n 1 || true)"
        if [[ -n "${TOKEN_LINE}" ]]; then
            echo "  ${TOKEN_LINE}"
            echo "${BASENAME}: ${TOKEN_LINE}" >> "${TOKEN_LOG}"
        fi
        grep -E "\[ERROR\]|ERROR|Error|terminate|what\(\)|std::" "${STDERR_FILE}" >> "${ERROR_LOG}" || true

        # 打印输出
        OUTPUT_TEXT=$(python3 -c "
import json
with open('${OUTPUT_FILE}') as f:
    d = json.load(f)
responses = d.get('responses', [])
if responses:
    text = responses[0].get('text', responses[0].get('output_text', ''))
elif 'output_text' in d:
    text = d['output_text']
elif 'output' in d:
    text = d['output']
else:
    text = json.dumps(d)[:200]
print(text[:500])
" 2>/dev/null || echo "[无法解析输出]")

        echo "  Output: ${OUTPUT_TEXT}"
        echo "  Time: ${ELAPSED_MS} ms"
        echo "${BASENAME}: ${ELAPSED_MS} ms" >> "${TIMING_LOG}"
    else
        END_TIME=$(date +%s%N)
        ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
        TOTAL_TIME=$((TOTAL_TIME + ELAPSED_MS))
        FAILED=$((FAILED + 1))
        TOKEN_LINE="$(grep -F "VisionZip tokens:" "${STDERR_FILE}" | tail -n 1 || true)"
        if [[ -n "${TOKEN_LINE}" ]]; then
            echo "  ${TOKEN_LINE}"
            echo "${BASENAME}: ${TOKEN_LINE}" >> "${TOKEN_LOG}"
        fi
        grep -E "\[ERROR\]|ERROR|Error|terminate|what\(\)|std::" "${STDERR_FILE}" >> "${ERROR_LOG}" || true
        echo "  FAILED (${ELAPSED_MS} ms)"
        echo "${BASENAME}: FAILED (${ELAPSED_MS} ms)" >> "${TIMING_LOG}"
    fi
    echo
done

# 汇总统计
PROCESSED=$((SUCCEEDED + FAILED))
if [[ ${PROCESSED} -gt 0 ]]; then
    AVG_TIME=$((TOTAL_TIME / PROCESSED))
else
    AVG_TIME=0
fi

echo "╔══════════════════════════════════════════════╗"
echo "║              推理完成 - 统计                  ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  成功:    ${SUCCEEDED}"
echo "║  失败:    ${FAILED}"
echo "║  跳过:    ${SKIPPED} (已存在)"
echo "║  总耗时:  ${TOTAL_TIME} ms"
if [[ ${PROCESSED} -gt 0 ]]; then
echo "║  平均:    ${AVG_TIME} ms/batch"
fi
echo "╚══════════════════════════════════════════════╝"
echo
echo "时间日志: ${TIMING_LOG}"
echo "VisionZip token 日志: ${TOKEN_LOG}"
echo "错误日志: ${OUTPUTS_DIR}/inference_errors.log"
