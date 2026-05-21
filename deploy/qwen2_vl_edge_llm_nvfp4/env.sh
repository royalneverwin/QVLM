#!/usr/bin/env bash

# Shared configuration for the first-stage TensorRT Edge-LLM deployment.
# Override any variable before sourcing/running the scripts if needed.

# Absolute path of this deployment directory inside the QVLM repo.
# It is used to derive local editable-code paths in a stable way.
export DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# Short local name for the deployment artifact directory under the workspace.
# We keep it separate from `MODEL_ID` so paths stay readable and stable.
export MODEL_NAME="Qwen2-VL-7B-Instruct"


############ Thor Deploy Related ##################
# Local checkout path of the TensorRT Edge-LLM repository on the machine
# that runs build/inference commands. On the Thor device this should point
# to the built Edge-LLM C++ repo because `llm_build`, `visual_build`, and
# `llm_inference` are resolved from this directory.
# 可通过环境变量覆盖（Thor 和 x86 路径不同）
export EDGE_LLM_REPO="${EDGE_LLM_REPO:-/data/wangxinhao/TensorRT-Edge-LLM}"

# TensorRT Edge-LLM 自定义插件库路径，build engine 时必须加载
# 编译 TensorRT-Edge-LLM 后生成在 build/ 目录下
export EDGELLM_PLUGIN_PATH="${EDGE_LLM_REPO}/build/libNvInfer_edgellm_plugin.so"

# SSH username used by the host-side transfer script when copying ONNX files
# from the x86 machine to the Thor device.
export DEVICE_USER="vdig"

# SSH hostname or IP address of the Thor device used by the transfer script.
export DEVICE_HOST="222.29.156.86"

# Thor-side workspace root. This single path is shared by:
# 1. x86 host scripts when they copy files to the Thor device via `ssh/scp`
# 2. scripts that run locally on the Thor device for build/inference
# Keep this as an absolute path on Thor so both use cases resolve the same way.
export DEVICE_WORKSPACE_DIR="/home/vdig/wangxinhao/QVLM/tensorrt-edgellm-workspace"

# Maximum batch size baked into the LLM TensorRT engine.
# Higher values improve flexibility but increase engine build/runtime memory.
export MAX_BATCH_SIZE="1"

# Maximum text input length used when building the LLM engine. This controls
# how much prompt text the engine is prepared to accept before generation.
export MAX_INPUT_LEN="1024"

# Maximum KV cache capacity baked into the LLM engine. This affects the total
# context + generation budget and also has direct memory implications.
export MAX_KV_CACHE_CAPACITY="4096"


# Lower bound for image token count accepted by the visual engine.
# This is passed to `visual_build` and should match the model/runtime needs.
# Qwen2-VL 在较小分辨率下单图也可能产生 ~900 tokens，设为 64 避免下限卡住
export MIN_IMAGE_TOKENS="64"

# Upper bound for total image token count accepted by the visual engine.
# Larger values support higher-resolution inputs but cost more memory.
export MAX_IMAGE_TOKENS="4096"

# Upper bound for image tokens contributed by a single image. This matters for
# multimodal requests with one or more images in the same prompt.
# Qwen2-VL 标准模型 720p 图像约 1280 tokens，设 1536 留余量
export MAX_IMAGE_TOKENS_PER_IMAGE="1536"

# Default request JSON path used by `thor_device/04_run_inference.sh`.
# The file is expected to live on the Thor device when inference is run.
export INPUT_FILE="$DEVICE_WORKSPACE_DIR/input_vlm.json"

# Default output JSON path written by `thor_device/04_run_inference.sh`.
export OUTPUT_FILE="$DEVICE_WORKSPACE_DIR/output_vlm.json"



############ X86 Export Related ##################
# Host-side workspace root used to store downloaded/quantized models, ONNX
# exports, and the default input/output JSON files. This is mainly consumed
# by the x86 export scripts and can also be reused on the device if desired.
export WORKSPACE_DIR="/data/wangxinhao/QVLM/tensorrt-edgellm-workspace"

# HuggingFace model id passed into TensorRT Edge-LLM export tools. This can
# be a Hub id such as `Qwen/Qwen2-VL-7B-Instruct` or a local HF checkpoint.
export MODEL_ID="Qwen/Qwen2-VL-7B-Instruct"

# Quantization method used only for the language model in this baseline.
# For the current Thor target we default to `nvfp4`; the visual encoder is
# still exported from the original checkpoint via the official path.
export QUANTIZATION="nvfp4"

# Root directory for all export-stage artifacts of this specific model under
# the x86 workspace. The export scripts create subdirectories under this path.
export EXPORT_MODEL_DIR="$WORKSPACE_DIR/$MODEL_NAME"

# Local editable copy of the HuggingFace model repo. The prepare script
# downloads the checkpoint here and patches it to load local QVLM code.
export LOCAL_MODEL_DIR="$EXPORT_MODEL_DIR/local_model"

# Vendored Qwen2-VL code inside QVLM. This is the codebase you will later edit
# to add `VisionZip` / `QAPruner`.
export VENDORED_QWEN2_VL_CODE_DIR="$DEPLOY_DIR/qwen2_vl_local_code"

# Export-stage directory that stores the NVFP4-quantized language model produced
# by `tensorrt-edgellm-quantize-llm`.
export EXPORT_QUANTIZED_LLM_DIR="$EXPORT_MODEL_DIR/quantized/llm"

# Export-stage ONNX output directory for the language model component.
export EXPORT_ONNX_LLM_DIR="$EXPORT_MODEL_DIR/onnx/llm"

# Export-stage ONNX output directory for the visual encoder component.
export EXPORT_ONNX_VISUAL_DIR="$EXPORT_MODEL_DIR/onnx/visual"

# Device-side root directory for this model's artifacts when scripts run
# locally on the Thor device.
export DEVICE_MODEL_DIR="$DEVICE_WORKSPACE_DIR/$MODEL_NAME"

# Device-side location of the transferred language-model ONNX files.
export DEVICE_ONNX_LLM_DIR="$DEVICE_MODEL_DIR/onnx/llm"

# Device-side location of the transferred visual-encoder ONNX files.
export DEVICE_ONNX_VISUAL_DIR="$DEVICE_MODEL_DIR/onnx/visual"

# Output directory for the built TensorRT language-model engine on Thor.
export DEVICE_ENGINE_LLM_DIR="$DEVICE_MODEL_DIR/engines/llm"

# Output directory for the built TensorRT visual engine on Thor.
export DEVICE_ENGINE_VISUAL_DIR="$DEVICE_MODEL_DIR/engines/visual"


############ X86 Local Build/Inference Related ##################
# x86 侧本地构建 engine 和推理时使用的路径（不经过 Thor 设备）

# x86 侧 TensorRT engine 输出目录
export X86_ENGINE_LLM_DIR="$EXPORT_MODEL_DIR/engines/llm"
export X86_ENGINE_VISUAL_DIR="$EXPORT_MODEL_DIR/engines/visual"

# x86 侧推理输入/输出文件
export X86_INPUT_FILE="$WORKSPACE_DIR/input_vlm.json"
export X86_OUTPUT_FILE="$WORKSPACE_DIR/output_vlm.json"


############ ScienceQA Evaluation Related ##################
# ScienceQA 数据集存放目录
export SCIENCEQA_DATA_DIR="$WORKSPACE_DIR/scienceqa_data"

# ScienceQA 推理工作目录（输入 batch 文件 + 输出 + metadata）
export SCIENCEQA_EVAL_DIR="$WORKSPACE_DIR/scienceqa_eval"

# ScienceQA 评估用的 split
export SCIENCEQA_SPLIT="test"

# 最大评估样本数（0 = 全部）
export SCIENCEQA_MAX_SAMPLES="0"


############ Thor ScienceQA Evaluation Related ##################
# Thor 设备上 ScienceQA 相关路径
export DEVICE_SCIENCEQA_DATA_DIR="$DEVICE_WORKSPACE_DIR/scienceqa_data"
export DEVICE_SCIENCEQA_EVAL_DIR="$DEVICE_WORKSPACE_DIR/scienceqa_eval"
