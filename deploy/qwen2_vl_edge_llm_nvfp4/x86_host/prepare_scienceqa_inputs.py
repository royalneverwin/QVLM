#!/usr/bin/env python3
"""
下载 ScienceQA 数据集并生成 TensorRT Edge-LLM 推理所需的 input JSON 文件。

用法:
    python prepare_scienceqa_inputs.py \
        --data-dir /path/to/scienceqa_data \
        --output-dir /path/to/output \
        --split test \
        --batch-size 1 \
        --max-samples 0  # 0 表示全部

输出:
    在 output-dir 下生成:
    - inputs/  目录，包含多个 batch_{i}.json 文件
    - metadata.json  记录样本信息，用于后续评估
"""

import argparse
import json
import os
from pathlib import Path


def download_scienceqa(data_dir: str):
    """从 HuggingFace 下载 ScienceQA 数据集"""
    data_path = Path(data_dir)

    problems_file = data_path / "problems.json"
    pid_splits_file = data_path / "pid_splits.json"

    if problems_file.exists() and pid_splits_file.exists():
        print(f"  ScienceQA 数据已存在于 {data_dir}，跳过下载")
        return

    print(f"  正在从 HuggingFace 下载 ScienceQA 到 {data_dir}")
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "需要 datasets 库来下载 ScienceQA。请运行: pip install datasets"
        )

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # 下载 ScienceQA 数据集
    dataset = load_dataset("derek-thomas/ScienceQA", trust_remote_code=True)

    # 构建 problems.json 和 pid_splits.json
    problems = {}
    pid_splits = {"train": [], "val": [], "test": []}

    split_map = {"train": "train", "validation": "val", "test": "test"}

    for hf_split, local_split in split_map.items():
        if hf_split not in dataset:
            continue
        for item in dataset[hf_split]:
            pid = str(item["index"])
            pid_splits[local_split].append(pid)

            problems[pid] = {
                "question": item.get("question", ""),
                "choices": item.get("choices", []),
                "answer": item.get("answer", 0),
                "hint": item.get("hint", ""),
                "image": None,
                "subject": item.get("subject", ""),
                "topic": item.get("topic", ""),
                "category": item.get("category", ""),
                "skill": item.get("skill", ""),
                "lecture": item.get("lecture", ""),
                "solution": item.get("solution", ""),
            }

            # 保存图片
            if item.get("image") is not None:
                img_dir = data_path / "images" / local_split / pid
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / "image.png"
                if not img_path.exists():
                    item["image"].save(str(img_path))
                problems[pid]["image"] = "image.png"

    # 写入 JSON 文件
    with open(problems_file, "w") as f:
        json.dump(problems, f, ensure_ascii=False)
    with open(pid_splits_file, "w") as f:
        json.dump(pid_splits, f, ensure_ascii=False)

    print(f"  下载完成: {len(problems)} 个问题")


def build_prompt(problem: dict) -> str:
    """构建 ScienceQA 的 QCM 格式 prompt"""
    options = "ABCDEFGHIJ"
    question = problem["question"]
    context = problem.get("hint", "")
    if not context:
        context = "N/A"

    choices = problem["choices"]
    choice_text = " ".join(
        f"({options[i]}) {c}" for i, c in enumerate(choices)
    )

    prompt = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Options: {choice_text}\n"
        f"Answer with the option letter only."
    )
    return prompt


def generate_inputs(
    data_dir: str,
    output_dir: str,
    split: str = "test",
    batch_size: int = 1,
    max_samples: int = 0,
    max_generate_length: int = 32,
):
    """生成 TensorRT Edge-LLM 格式的输入 JSON 文件"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    inputs_dir = output_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    with open(data_path / "problems.json") as f:
        problems = json.load(f)
    with open(data_path / "pid_splits.json") as f:
        pid_splits = json.load(f)

    # 获取指定 split 的 pid 列表
    pids = pid_splits.get(split, [])
    if not pids:
        raise ValueError(f"Split '{split}' 为空或不存在")

    if max_samples > 0:
        pids = pids[:max_samples]

    # 只保留有图片的样本（VLM 推理需要图片）
    image_pids = []
    for pid in pids:
        if problems[pid].get("image") is not None:
            img_path = data_path / "images" / split / pid / "image.png"
            if img_path.exists():
                image_pids.append(pid)

    print(f"  Split '{split}': 总共 {len(pids)} 个样本，"
          f"有图片的 {len(image_pids)} 个")

    # 按 batch 生成输入文件
    metadata = []
    batch_idx = 0

    for i in range(0, len(image_pids), batch_size):
        batch_pids = image_pids[i:i + batch_size]
        requests = []

        for pid in batch_pids:
            problem = problems[pid]
            prompt = build_prompt(problem)
            img_path = str(data_path / "images" / split / pid / "image.png")

            request = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer the multiple choice question with just the option letter."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ]
            }
            requests.append(request)

            metadata.append({
                "pid": pid,
                "batch_idx": batch_idx,
                "answer": problem["answer"],
                "choices": problem["choices"],
                "subject": problem.get("subject", ""),
                "has_image": True,
            })

        input_json = {
            "batch_size": len(requests),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_generate_length": max_generate_length,
            "requests": requests,
        }

        batch_file = inputs_dir / f"batch_{batch_idx:05d}.json"
        with open(batch_file, "w") as f:
            json.dump(input_json, f, ensure_ascii=False, indent=2)

        batch_idx += 1

    # 写入 metadata
    meta_file = output_path / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  生成 {batch_idx} 个 batch 文件到 {inputs_dir}")
    print(f"  Metadata 写入 {meta_file}")
    return batch_idx


def main():
    parser = argparse.ArgumentParser(
        description="下载 ScienceQA 并生成 Edge-LLM 推理输入"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="ScienceQA 数据集存放目录"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="输出目录（生成 inputs/ 和 metadata.json）"
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "val", "test"],
        help="使用的数据集划分"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="每个输入文件中的请求数量"
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="最大样本数（0 = 全部）"
    )
    parser.add_argument(
        "--max-generate-length", type=int, default=32,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="跳过下载，直接使用现有数据"
    )

    args = parser.parse_args()

    if not args.skip_download:
        print("[1/2] 下载 ScienceQA 数据集")
        download_scienceqa(args.data_dir)
    else:
        print("[1/2] 跳过下载（使用现有数据）")

    print("[2/2] 生成推理输入文件")
    generate_inputs(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_generate_length=args.max_generate_length,
    )

    print("完成！")


if __name__ == "__main__":
    main()
