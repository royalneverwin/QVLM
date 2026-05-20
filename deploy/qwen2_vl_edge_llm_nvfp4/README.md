# Qwen2-VL Edge-LLM NVFP4 Baseline

This directory implements the first deployment step we aligned on:

- language model: TensorRT Edge-LLM `NVFP4`
- visual encoder: exported from the prepared local `Qwen2-VL` checkpoint
- scope: baseline deployment only, without `VisionZip` or `QAPruner`

The workflow follows the official TensorRT Edge-LLM split pipeline:

1. x86 host: download `Qwen2-VL` locally and prepare editable QVLM-linked code
2. x86 host: quantize LLM to `NVFP4`
3. x86 host: export quantized LLM to ONNX
4. x86 host: export visual encoder from the prepared local model
5. Thor device: build `llm` and `visual` TensorRT engines
6. Thor device: run multimodal inference

## Directory Layout

- `env.sh`: shared environment variables and build parameters
- `x86_host/00_prepare_local_model.sh`: download model locally, vendor Qwen2-VL code into QVLM, patch config files
- `x86_host/01_quantize_export.sh`: quantize LLM + export LLM + export visual from the prepared local model
- `x86_host/02_transfer_onnx.sh`: copy ONNX artifacts to the Thor device
- `thor_device/03_build_engines.sh`: build `llm` and `visual` engines on Thor
- `thor_device/04_run_inference.sh`: run multimodal inference on Thor
- `templates/input_vlm.json`: starter request payload for `llm_inference`

## Defaults

The baseline defaults to `Qwen/Qwen2-VL-7B-Instruct` to stay close to your original 7B target, but `env.sh` is parameterized and can also be switched to `Qwen/Qwen2-VL-2B-Instruct`.

Key default choices:

- `QUANTIZATION=nvfp4`
- `MODEL_ID=Qwen/Qwen2-VL-7B-Instruct`
- visual path stays on the original HF checkpoint
- `MAX_BATCH_SIZE=1`
- `MAX_INPUT_LEN=1024`
- `MAX_KV_CACHE_CAPACITY=4096`

## Expected Environment

On the x86 host:

- TensorRT Edge-LLM Python package installed
- `tensorrt-edgellm-quantize-llm`
- `tensorrt-edgellm-export-llm`
- `tensorrt-edgellm-export-visual`

On the Thor device:

- TensorRT Edge-LLM C++ project built
- `llm_build`, `visual_build`, and `llm_inference` available under `build/examples/...`

## Usage

On the x86 host:

```bash
cd deploy/qwen2_vl_edge_llm_nvfp4
bash x86_host/00_prepare_local_model.sh
bash x86_host/01_quantize_export.sh
bash x86_host/02_transfer_onnx.sh
```

On the Thor device:

```bash
cd deploy/qwen2_vl_edge_llm_nvfp4
bash thor_device/03_build_engines.sh
bash thor_device/04_run_inference.sh
```

## Notes

- `x86_host/00_prepare_local_model.sh` downloads the checkpoint into `LOCAL_MODEL_DIR`, vendors Qwen2-VL code into QVLM, rewrites imports for local dynamic loading, and patches `config.json` / `preprocessor_config.json` / `tokenizer_config.json`.
- The later export step uses `LOCAL_MODEL_DIR`, not the raw HuggingFace model id, so you can modify the vendored QVLM files before export.
- This baseline intentionally leaves the visual encoder on the official export path. We can add `VisionZip` and `QAPruner` on top of this directory later.
- `templates/input_vlm.json` contains a placeholder image path and should point to a real image on the Thor device before inference.
- The commands here mirror the official TensorRT Edge-LLM workflow as documented on 2026-02-19.
