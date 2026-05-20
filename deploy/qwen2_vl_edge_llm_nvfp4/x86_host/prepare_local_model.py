#!/usr/bin/env python3
"""
Prepare a local, editable Qwen2-VL checkpoint for QVLM development.

Workflow:
1. Download the HuggingFace checkpoint locally (skip if already present).
2. Vendor Qwen2-VL source files into the QVLM repo.
3. Rewrite imports so the vendored files work as local dynamic modules.
4. Symlink those vendored files into the local model folder.
5. Patch config/preprocessor/tokenizer metadata to prefer the local code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


QWEN2_VL_CODE_FILES = [
    "__init__.py",
    "configuration_qwen2_vl.py",
    "modeling_qwen2_vl.py",
    "image_processing_qwen2_vl.py",
    "image_processing_pil_qwen2_vl.py",
    "processing_qwen2_vl.py",
    "video_processing_qwen2_vl.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--local-model-dir", required=True)
    parser.add_argument("--vendored-code-dir", required=True)
    parser.add_argument(
        "--refresh-vendored-code",
        action="store_true",
        help="Overwrite existing vendored Qwen2-VL code with the local transformers source.",
    )
    return parser.parse_args()


def require_packages() -> None:
    try:
        import huggingface_hub  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This script requires `huggingface_hub` and `transformers` in the current python3 environment. "
            f"Import error: {exc}"
        )


def is_model_downloaded(local_model_dir: Path) -> bool:
    """检查模型是否已经下载完成（通过 config.json 是否存在来判断）。"""
    return (local_model_dir / "config.json").exists()


def download_model_snapshot(model_id: str, local_model_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    # 使用 hf-mirror 镜像加速下载
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    local_model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_model_dir),
        resume_download=True,
    )


def get_qwen2_vl_source_dir() -> Path:
    import transformers

    source_dir = Path(transformers.__file__).resolve().parent / "models" / "qwen2_vl"
    if not source_dir.is_dir():
        raise SystemExit(f"Could not locate transformers qwen2_vl source dir: {source_dir}")
    return source_dir


def rewrite_imports(text: str) -> str:
    text = re.sub(r"from \.\.\.([A-Za-z0-9_\.]+) import ", r"from transformers.\1 import ", text)
    text = re.sub(
        r"from \.\.qwen2\.([A-Za-z0-9_\.]+) import ",
        r"from transformers.models.qwen2.\1 import ",
        text,
    )
    return text


def seed_vendored_code(vendored_code_dir: Path, refresh: bool) -> None:
    source_dir = get_qwen2_vl_source_dir()
    vendored_code_dir.mkdir(parents=True, exist_ok=True)

    for filename in QWEN2_VL_CODE_FILES:
        src = source_dir / filename
        dst = vendored_code_dir / filename

        if dst.exists() and not refresh:
            continue

        text = src.read_text(encoding="utf-8")
        if filename != "__init__.py":
            text = rewrite_imports(text)
        dst.write_text(text, encoding="utf-8")


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        backup = dst.with_name(dst.name + ".qvlm_backup")
        if not backup.exists():
            shutil.move(str(dst), str(backup))
        else:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

    os.symlink(src.resolve(), dst)


def link_vendored_code(vendored_code_dir: Path, local_model_dir: Path) -> None:
    for filename in QWEN2_VL_CODE_FILES:
        ensure_symlink(vendored_code_dir / filename, local_model_dir / filename)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".qvlm_backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_config_json(local_model_dir: Path) -> None:
    path = local_model_dir / "config.json"
    backup_once(path)
    data = load_json(path)

    auto_map = dict(data.get("auto_map", {}))
    auto_map.update(
        {
            "AutoConfig": "configuration_qwen2_vl.Qwen2VLConfig",
            "AutoModel": "modeling_qwen2_vl.Qwen2VLModel",
            "AutoModelForVision2Seq": "modeling_qwen2_vl.Qwen2VLForConditionalGeneration",
            "AutoModelForImageTextToText": "modeling_qwen2_vl.Qwen2VLForConditionalGeneration",
        }
    )
    data["auto_map"] = auto_map

    save_json(path, data)


def patch_preprocessor_config_json(local_model_dir: Path) -> None:
    path = local_model_dir / "preprocessor_config.json"
    if not path.exists():
        return

    backup_once(path)
    data = load_json(path)

    auto_map = dict(data.get("auto_map", {}))
    auto_map.update(
        {
            "AutoImageProcessor": [
                "image_processing_qwen2_vl.Qwen2VLImageProcessor",
                "image_processing_pil_qwen2_vl.Qwen2VLImageProcessorPIL",
            ],
            "AutoProcessor": "processing_qwen2_vl.Qwen2VLProcessor",
            "AutoVideoProcessor": "video_processing_qwen2_vl.Qwen2VLVideoProcessor",
        }
    )
    data["auto_map"] = auto_map
    data["processor_class"] = "Qwen2VLProcessor"

    save_json(path, data)


def patch_tokenizer_config_json(local_model_dir: Path) -> None:
    path = local_model_dir / "tokenizer_config.json"
    if not path.exists():
        return

    backup_once(path)
    data = load_json(path)

    auto_map = dict(data.get("auto_map", {}))
    auto_map.update({"AutoProcessor": "processing_qwen2_vl.Qwen2VLProcessor"})
    data["auto_map"] = auto_map
    data["processor_class"] = "Qwen2VLProcessor"

    save_json(path, data)


def write_prepare_metadata(local_model_dir: Path, model_id: str, vendored_code_dir: Path) -> None:
    metadata = {
        "model_id": model_id,
        "vendored_code_dir": str(vendored_code_dir.resolve()),
        "code_files": QWEN2_VL_CODE_FILES,
        "notes": [
            "The *.py files in this local model dir are symlinks into the QVLM repo vendored code directory.",
            "Edit the vendored QVLM files rather than the symlink targets in this model directory.",
        ],
    }
    save_json(local_model_dir / "qvlm_prepare_metadata.json", metadata)


def main() -> None:
    args = parse_args()
    require_packages()

    local_model_dir = Path(args.local_model_dir).expanduser().resolve()
    vendored_code_dir = Path(args.vendored_code_dir).expanduser().resolve()

    if is_model_downloaded(local_model_dir):
        print(f"[1/4] Model already downloaded at {local_model_dir}, skipping download.")
    else:
        print(f"[1/4] Downloading model snapshot to {local_model_dir}")
        download_model_snapshot(args.model_id, local_model_dir)

    print(f"[2/4] Seeding vendored Qwen2-VL code to {vendored_code_dir}")
    seed_vendored_code(vendored_code_dir, refresh=args.refresh_vendored_code)

    print("[3/4] Linking vendored code into local model directory")
    link_vendored_code(vendored_code_dir, local_model_dir)

    print("[4/4] Patching local model config files for local dynamic loading")
    patch_config_json(local_model_dir)
    patch_preprocessor_config_json(local_model_dir)
    patch_tokenizer_config_json(local_model_dir)
    write_prepare_metadata(local_model_dir, args.model_id, vendored_code_dir)

    print()
    print("Prepared local Qwen2-VL model:")
    print(f"  model_id:          {args.model_id}")
    print(f"  local_model_dir:   {local_model_dir}")
    print(f"  vendored_code_dir: {vendored_code_dir}")


if __name__ == "__main__":
    main()
