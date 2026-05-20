#!/usr/bin/env bash
# 构建 TensorRT Edge-LLM 量化导出环境（x86 主机端）
# 使用方法：source setup_env.sh 或 bash setup_env.sh
#
# 前置条件：
#   - Linux x86_64 + NVIDIA GPU + CUDA 已安装
#   - Python 3.10+
#   - TensorRT-Edge-LLM 仓库已 clone 到 $EDGE_LLM_REPO_PATH
#
# 已知坑点（本脚本已处理）：
#   1. tensorrt-edgellm 0.6.x 需要 transformers >= 5.2.0（Qwen3.5/Qwen3VL 支持）
#   2. transformers 5.x 需要 timm >= 1.0.0（gemma3n 依赖 ImageNetInfo）
#   3. deepspeed 与 pydantic v2 不兼容，量化不需要 deepspeed，直接不装
#   4. numpy/scikit-learn 二进制不兼容时需要同时升级
#   5. 已废弃的 local_dir_use_symlinks 参数已移除
#   6. venv 不可直接 move 目录，需原地重建

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEPLOY_DIR}/env.sh"

# ==================== 配置区域 ====================
# TensorRT-Edge-LLM 仓库路径（需要先 git clone）
EDGE_LLM_REPO_PATH="${EDGE_LLM_REPO_PATH:-$HOME/TensorRT-Edge-LLM}"

# Python 虚拟环境路径
VENV_DIR="${VENV_DIR:-$(cd "${DEPLOY_DIR}/../.." && pwd)/.venv}"

# ==================== 检查前置条件 ====================
echo "============================================"
echo " TensorRT Edge-LLM 环境构建"
echo "============================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "错误：未找到 python3" >&2
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python 版本: ${PYTHON_VERSION}"

if [[ ! -d "${EDGE_LLM_REPO_PATH}" ]]; then
    echo "错误：TensorRT-Edge-LLM 仓库未找到: ${EDGE_LLM_REPO_PATH}" >&2
    echo "请先执行:" >&2
    echo "  git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git ${EDGE_LLM_REPO_PATH}" >&2
    echo "  cd ${EDGE_LLM_REPO_PATH} && git submodule update --init --recursive" >&2
    exit 1
fi

# ==================== 创建/重建虚拟环境 ====================
echo
echo "[1/5] 创建 Python 虚拟环境: ${VENV_DIR}"

if [[ -d "${VENV_DIR}" ]]; then
    echo "  已存在，跳过创建（如需重建请先 rm -rf ${VENV_DIR}）"
else
    python3 -m venv "${VENV_DIR}"
    echo "  创建完成"
fi

# 激活虚拟环境
source "${VENV_DIR}/bin/activate"
echo "  已激活: $(which python3)"

# ==================== 升级基础工具 ====================
echo
echo "[2/5] 升级 pip/setuptools"
python3 -m pip install --upgrade pip setuptools wheel -q

# ==================== 安装 TensorRT Edge-LLM ====================
echo
echo "[3/5] 安装 TensorRT Edge-LLM（提供量化/导出 CLI 工具）"
python3 -m pip install "${EDGE_LLM_REPO_PATH}" -q

# ==================== 安装/修复依赖 ====================
echo
echo "[4/5] 安装兼容版本的依赖包"

# transformers >= 5.2.0：支持 qwen3_5、qwen3_vl 模块
# timm >= 1.0.0：修复 gemma3n 的 ImageNetInfo 导入
# numpy + scikit-learn：确保二进制兼容
python3 -m pip install \
    "transformers>=5.2.0" \
    "timm>=1.0.0" \
    "numpy>=1.26" \
    "scikit-learn" \
    -q

# deepspeed 与 pydantic v2 冲突，量化/导出不需要它
# 如果被 tensorrt-edgellm 间接安装了，卸载掉
if python3 -c "import deepspeed" 2>/dev/null; then
    echo "  卸载 deepspeed（与 pydantic v2 不兼容，量化不需要）"
    python3 -m pip uninstall deepspeed -y -q
fi

# ==================== 验证安装 ====================
echo
echo "[5/5] 验证 CLI 工具"

TOOLS_OK=true
for cmd in tensorrt-edgellm-quantize-llm tensorrt-edgellm-export-llm tensorrt-edgellm-export-visual; do
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "  ✓ $cmd"
    else
        echo "  ✗ $cmd 未找到!" >&2
        TOOLS_OK=false
    fi
done

if [[ "${TOOLS_OK}" != "true" ]]; then
    echo
    echo "错误：部分工具未安装成功，请检查上面的输出" >&2
    exit 1
fi

# 打印关键包版本
echo
echo "关键包版本："
python3 -c "
import transformers, timm, modelopt, pydantic
print(f'  transformers: {transformers.__version__}')
print(f'  timm:         {timm.__version__}')
print(f'  modelopt:     {modelopt.__version__}')
print(f'  pydantic:     {pydantic.__version__}')
" 2>/dev/null || true

echo
echo "============================================"
echo " 环境构建完成！"
echo "============================================"
echo
echo "激活环境：source ${VENV_DIR}/bin/activate"
echo
echo "下一步："
echo "  1. bash x86_host/00_prepare_local_model.sh   # 下载模型"
echo "  2. bash x86_host/01_quantize_export.sh       # 量化 + 导出 ONNX"
echo "  3. bash x86_host/02_transfer_onnx.sh         # 传输到 Thor 设备"
