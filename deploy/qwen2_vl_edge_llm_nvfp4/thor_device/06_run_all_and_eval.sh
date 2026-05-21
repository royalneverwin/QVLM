#!/usr/bin/env bash
# 顺序执行 3 组 ScienceQA 推理（visionzip prune 0.875 / 0.5 / baseline），
# 每组结束后立即用 05_eval.sh 计算准确率。
# 最后汇总三组的准确率与平均推理时间。
#
# 使用方法：
#   bash thor_device/06_run_all_and_eval.sh
# 可选环境变量（默认值见 env.sh）：
#   DEVICE_ENGINE_LLM_DIR  —— 共享的 LLM engine 目录
#   SCIENCEQA_INPUTS_DIR   —— ScienceQA 输入 batch 目录

set -uo pipefail  # 注意：去掉 -e，让单个实验失败不影响后续实验

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

RUN_INFER_SH="${SCRIPT_DIR}/04_run_inference.sh"
EVAL_SH="${SCRIPT_DIR}/05_eval.sh"

if [[ ! -f "${RUN_INFER_SH}" || ! -f "${EVAL_SH}" ]]; then
    echo "缺少必要脚本: ${RUN_INFER_SH} 或 ${EVAL_SH}" >&2
    exit 1
fi

WORKSPACE="/home/vdig/wangxinhao/QVLM/tensorrt-edgellm-workspace"
MODEL_WORKSPACE="${WORKSPACE}/Qwen2-VL-7B-Instruct"
SCIENCEQA_BASE="${WORKSPACE}/scienceqa_eval"

# ========== 三组实验配置 ==========
# 每组：标签 | visual engine 目录 | outputs 目录
# baseline 不指定 visual engine，使用 env.sh 默认值
declare -a EXP_TAGS=(
    "visionzip_prune_0p875"
    "visionzip_prune_0p5"
    "baseline"
)
declare -a EXP_VISUAL_DIRS=(
    "${MODEL_WORKSPACE}/engines/visual_visionzip_prune_0p875"
    "${MODEL_WORKSPACE}/engines/visual_visionzip_prune_0p5"
    ""  # baseline：留空走默认
)
declare -a EXP_OUTPUTS_DIRS=(
    "${SCIENCEQA_BASE}/outputs_visionzip_prune_0p875"
    "${SCIENCEQA_BASE}/outputs_visionzip_prune_0p5"
    "${SCIENCEQA_BASE}/outputs_baseline"
)

# 汇总日志
SUMMARY_LOG="${SCIENCEQA_BASE}/summary_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCIENCEQA_BASE}"
: > "${SUMMARY_LOG}"

print_box() {
    local msg="$1"
    echo "================================================="
    echo " ${msg}"
    echo "================================================="
}

NUM_EXP=${#EXP_TAGS[@]}

for ((i=0; i<NUM_EXP; i++)); do
    TAG="${EXP_TAGS[$i]}"
    VISUAL_DIR="${EXP_VISUAL_DIRS[$i]}"
    OUTPUTS_DIR="${EXP_OUTPUTS_DIRS[$i]}"
    RESULT_FILE="${OUTPUTS_DIR}/results.json"

    print_box "[$((i+1))/${NUM_EXP}] 推理: ${TAG}"
    echo "  visual engine: ${VISUAL_DIR:-<默认 baseline>}"
    echo "  outputs:       ${OUTPUTS_DIR}"
    echo

    # ---------- 推理 ----------
    INFER_START=$(date +%s)
    if [[ -n "${VISUAL_DIR}" ]]; then
        DEVICE_ENGINE_VISUAL_DIR="${VISUAL_DIR}" \
        SCIENCEQA_OUTPUTS_DIR="${OUTPUTS_DIR}" \
            bash "${RUN_INFER_SH}"
        INFER_RC=$?
    else
        SCIENCEQA_OUTPUTS_DIR="${OUTPUTS_DIR}" \
            bash "${RUN_INFER_SH}"
        INFER_RC=$?
    fi
    INFER_END=$(date +%s)
    INFER_ELAPSED=$((INFER_END - INFER_START))

    if [[ ${INFER_RC} -ne 0 ]]; then
        echo "[警告] ${TAG} 推理脚本退出码 ${INFER_RC}（继续评估已完成的 batch）" | tee -a "${SUMMARY_LOG}"
    fi

    # ---------- 评估 ----------
    print_box "[$((i+1))/${NUM_EXP}] 评估: ${TAG}"
    SCIENCEQA_OUTPUTS_DIR="${OUTPUTS_DIR}" \
    SCIENCEQA_RESULT_FILE="${RESULT_FILE}" \
        bash "${EVAL_SH}" || echo "[警告] ${TAG} 评估失败"

    # ---------- 计算平均推理时间 ----------
    TIMING_LOG="${OUTPUTS_DIR}/timing.log"
    AVG_VISUAL_MS="N/A"
    AVG_LLM_MS="N/A"
    AVG_MS="N/A"
    SUCCEEDED_CNT=0
    if [[ -f "${TIMING_LOG}" ]]; then
        # 只统计成功行，分别计算 visual / llm / total 平均值
        AVG_VISUAL_MS=$(awk '
            /FAILED/ { next }
            /visual=/ {
                split($0, a, "visual="); split(a[2], b, " ")
                if (b[1] != "N/A") { sum += b[1]; cnt += 1 }
            }
            END {
                if (cnt > 0) printf "%.2f", sum / cnt; else printf "N/A"
            }
        ' "${TIMING_LOG}")
        AVG_LLM_MS=$(awk '
            /FAILED/ { next }
            /llm=/ {
                split($0, a, "llm="); split(a[2], b, " ")
                if (b[1] != "N/A") { sum += b[1]; cnt += 1 }
            }
            END {
                if (cnt > 0) printf "%.2f", sum / cnt; else printf "N/A"
            }
        ' "${TIMING_LOG}")
        AVG_MS=$(awk '
            /FAILED/ { next }
            /total=/ {
                split($0, a, "total="); split(a[2], b, " ")
                if (b[1] != "N/A") { sum += b[1]; cnt += 1 }
            }
            END {
                if (cnt > 0) printf "%.2f", sum / cnt; else printf "N/A"
            }
        ' "${TIMING_LOG}")
        SUCCEEDED_CNT=$(grep -vc "FAILED" "${TIMING_LOG}" | head -n 1 || echo 0)
    fi

    # ---------- 提取准确率 ----------
    ACCURACY="N/A"
    if [[ -f "${RESULT_FILE}" ]]; then
        ACCURACY=$(python3 -c "
import json, sys
try:
    with open('${RESULT_FILE}') as f:
        d = json.load(f)
    # 兼容多种字段名
    for key in ('accuracy', 'overall_accuracy', 'acc'):
        if key in d:
            print(d[key]); sys.exit(0)
    # 嵌套结构尝试
    if 'summary' in d and isinstance(d['summary'], dict):
        for key in ('accuracy', 'overall_accuracy', 'acc'):
            if key in d['summary']:
                print(d['summary'][key]); sys.exit(0)
    print('N/A')
except Exception as e:
    print('N/A')
" 2>/dev/null || echo "N/A")
    fi

    {
        echo "----- ${TAG} -----"
        echo "  outputs_dir:    ${OUTPUTS_DIR}"
        echo "  result_file:    ${RESULT_FILE}"
        echo "  推理总耗时(s):  ${INFER_ELAPSED}"
        echo "  成功 batch 数:  ${SUCCEEDED_CNT}"
        echo "  平均 visual(ms): ${AVG_VISUAL_MS}"
        echo "  平均 llm(ms):    ${AVG_LLM_MS}"
        echo "  平均 total(ms):  ${AVG_MS}"
        echo "  准确率:         ${ACCURACY}"
        echo
    } | tee -a "${SUMMARY_LOG}"
done

print_box "全部完成 - 汇总"
cat "${SUMMARY_LOG}"
echo
echo "汇总日志: ${SUMMARY_LOG}"
