#!/usr/bin/env bash
# Thor 设备上的环境构建脚本
# 编译 TensorRT-Edge-LLM C++ 项目，获得 llm_build / visual_build / llm_inference
#
# Thor (Jetson Orin) 特点：
#   - TensorRT 已预装（随 JetPack）
#   - CUDA 已预装
#   - 需要编译 TensorRT-Edge-LLM 的 C++ 部分（native ARM64）
#
# 用法：
#   bash thor_device/setup_env.sh
#
# 可选环境变量：
#   EDGE_LLM_REPO_PATH  - TensorRT-Edge-LLM 仓库克隆路径（默认 /home/vdig/wangxinhao/TensorRT-Edge-LLM）
#   TRT_PACKAGE_DIR     - TensorRT 安装路径（默认 /usr）
#   CUDA_CTK_VERSION    - CUDA Toolkit 版本（Thor/JetPack 默认 13.0）
#   ENABLE_CUTE_DSL     - 是否链接 CuTe DSL kernels（Thor 默认 fmha）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── 配置 ───
EDGE_LLM_REPO_PATH="${EDGE_LLM_REPO_PATH:-/home/vdig/wangxinhao/TensorRT-Edge-LLM}"
TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-/usr}"
CUDA_CTK_VERSION="${CUDA_CTK_VERSION:-13.0}"
ENABLE_CUTE_DSL="${ENABLE_CUTE_DSL:-fmha}"
EDGE_LLM_GIT_URL="https://github.com/NVIDIA/TensorRT-Edge-LLM.git"

echo "╔══════════════════════════════════════════════╗"
echo "║     Thor 设备环境构建                         ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Edge-LLM 仓库: ${EDGE_LLM_REPO_PATH}"
echo "║  TensorRT 路径: ${TRT_PACKAGE_DIR}"
echo "╚══════════════════════════════════════════════╝"
echo

# ─── 1. 检查前置依赖 ───
echo "[1/4] 检查前置依赖..."

# 检查 cmake
if ! command -v cmake >/dev/null 2>&1; then
    echo "缺少 cmake，尝试安装..." >&2
    sudo apt-get update && sudo apt-get install -y cmake
fi

# 检查 make
if ! command -v make >/dev/null 2>&1; then
    echo "缺少 make，尝试安装..." >&2
    sudo apt-get update && sudo apt-get install -y build-essential
fi

# 检查 nvcc
if ! command -v nvcc >/dev/null 2>&1; then
    echo "警告: nvcc 未找到，确保 CUDA 在 PATH 中" >&2
    # 尝试常见路径
    for cuda_dir in /usr/local/cuda /usr/local/cuda-*; do
        if [[ -x "${cuda_dir}/bin/nvcc" ]]; then
            export PATH="${cuda_dir}/bin:${PATH}"
            export CUDA_HOME="${cuda_dir}"
            echo "  找到 CUDA: ${cuda_dir}"
            break
        fi
    done
fi

# 检查 TensorRT
if [[ ! -f "${TRT_PACKAGE_DIR}/include/x86_64-linux-gnu/NvInfer.h" ]] && \
   [[ ! -f "${TRT_PACKAGE_DIR}/include/aarch64-linux-gnu/NvInfer.h" ]] && \
   [[ ! -f "${TRT_PACKAGE_DIR}/include/NvInfer.h" ]]; then
    echo "警告: 未在 ${TRT_PACKAGE_DIR} 找到 NvInfer.h"
    echo "Thor/Jetson 上 TensorRT 通常随 JetPack 预装在 /usr"
    echo "如果路径不同，请设置: export TRT_PACKAGE_DIR=/path/to/tensorrt"
fi

# 检查 python3
if ! command -v python3 >/dev/null 2>&1; then
    echo "缺少 python3，尝试安装..." >&2
    sudo apt-get update && sudo apt-get install -y python3
fi

echo "  cmake: $(cmake --version | head -1)"
echo "  nvcc:  $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "  python3: $(python3 --version)"
echo

# ─── 2. 克隆 TensorRT-Edge-LLM ───
echo "[2/4] 准备 TensorRT-Edge-LLM 仓库..."

if [[ -d "${EDGE_LLM_REPO_PATH}" ]]; then
    echo "  仓库已存在: ${EDGE_LLM_REPO_PATH}"
    cd "${EDGE_LLM_REPO_PATH}"
    echo "  拉取最新代码..."
    git pull --ff-only 2>/dev/null || echo "  (pull 失败，使用现有代码)"
    git submodule update --init --recursive
else
    echo "  克隆到: ${EDGE_LLM_REPO_PATH}"
    git clone "${EDGE_LLM_GIT_URL}" "${EDGE_LLM_REPO_PATH}"
    cd "${EDGE_LLM_REPO_PATH}"
    git submodule update --init --recursive
fi
echo

# ─── 3. 编译 C++ 项目 ───
echo "[3/4] 编译 TensorRT-Edge-LLM C++ 项目..."

BUILD_DIR="${EDGE_LLM_REPO_PATH}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 自动检测 CUDA_HOME
if [[ -z "${CUDA_HOME:-}" ]]; then
    for cuda_dir in /usr/local/cuda /usr/local/cuda-*; do
        if [[ -d "${cuda_dir}/include" ]]; then
            CUDA_HOME="${cuda_dir}"
            break
        fi
    done
fi

echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  TRT_PACKAGE_DIR: ${TRT_PACKAGE_DIR}"

# 自动检测 GPU compute capability，用于编译对应的 CUDA code 和 FMHA cubin
# 注意：TensorRT-Edge-LLM 顶层 CMake 在未设置 AARCH64_BUILD 时会覆盖
# CMAKE_CUDA_ARCHITECTURES 为 x86 默认列表，因此 Thor 必须显式设置
# AARCH64_BUILD/EMBEDDED_TARGET。
GPU_ARCH="${GPU_ARCH:-}"
if [[ -z "${GPU_ARCH}" ]]; then
    # 尝试用 python3 + torch 检测
    GPU_ARCH=$(python3 -c "import torch; cap=torch.cuda.get_device_capability(); print(f'{cap[0]*10+cap[1]}')" 2>/dev/null || true)
fi
if [[ -z "${GPU_ARCH}" ]]; then
    # 尝试 nvidia-smi
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)
fi
if [[ -z "${GPU_ARCH}" ]]; then
    echo "  警告: 无法自动检测 GPU 架构，使用默认值 87"
    GPU_ARCH="87"
fi

# Edge-LLM 的 embedded toolchain 使用 110a；若本机 CUDA/CMake 不接受 110a，
# 可在调用脚本前覆盖：GPU_ARCH=110 bash thor_device/setup_env.sh
if [[ "${GPU_ARCH}" == "110" ]]; then
    GPU_ARCH="110a"
fi
echo "  GPU_ARCH (SM): ${GPU_ARCH}"
echo "  CUDA_CTK_VERSION: ${CUDA_CTK_VERSION}"
echo "  ENABLE_CUTE_DSL: ${ENABLE_CUTE_DSL}"

# SM 110 (Thor/Blackwell) 运行时在 Edge-LLM 中会映射到 SM 101 的 FMHA_v2 cubin。
# 仅把 110 加进 FMHA_ALL_SM_VERSIONS 不够；必须保证 CMake 没有把 SM 101
# cubin 排除掉。下面的 AARCH64_BUILD/EMBEDDED_TARGET 会阻止 x86 默认架构覆盖。

cmake .. \
    -DTRT_PACKAGE_DIR="${TRT_PACKAGE_DIR}" \
    -DCUDA_DIR="${CUDA_HOME:-/usr/local/cuda}" \
    -DCUDA_CTK_VERSION="${CUDA_CTK_VERSION}" \
    -DAARCH64_BUILD=ON \
    -DEMBEDDED_TARGET=jetson-thor \
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
    -DENABLE_CUTE_DSL="${ENABLE_CUTE_DSL}" \
    -DCUTE_DSL_ARTIFACT_TAG=sm_110

make -j"$(nproc)"

echo

# ─── 4. 验证编译结果 ───
echo "[4/4] 验证编译结果..."

BINARIES=(
    "examples/llm/llm_build"
    "examples/llm/llm_inference"
    "examples/multimodal/visual_build"
)

ALL_OK=true
for bin in "${BINARIES[@]}"; do
    full_path="${BUILD_DIR}/${bin}"
    if [[ -x "${full_path}" ]]; then
        echo "  ✓ ${bin}"
    else
        echo "  ✗ ${bin} (NOT FOUND)" >&2
        ALL_OK=false
    fi
done

# 检查插件库
PLUGIN="${BUILD_DIR}/libNvInfer_edgellm_plugin.so"
if [[ -f "${PLUGIN}" ]]; then
    echo "  ✓ libNvInfer_edgellm_plugin.so"
else
    echo "  ✗ libNvInfer_edgellm_plugin.so (NOT FOUND)" >&2
    ALL_OK=false
fi

echo

if [[ "${ALL_OK}" == "true" ]]; then
    echo "╔══════════════════════════════════════════════╗"
    echo "║            构建成功！                         ║"
    echo "╚══════════════════════════════════════════════╝"
    echo
    echo "请在运行 Thor 脚本前设置环境变量："
    echo
    echo "  export EDGE_LLM_REPO=\"${EDGE_LLM_REPO_PATH}\""
    echo
    echo "或者写入 ~/.bashrc："
    echo "  echo 'export EDGE_LLM_REPO=\"${EDGE_LLM_REPO_PATH}\"' >> ~/.bashrc"
else
    echo "构建存在问题，请检查上面的错误信息" >&2
    exit 1
fi
