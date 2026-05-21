#!/usr/bin/env python3
"""
评估 ScienceQA 推理结果的准确率。

从 llm_inference 输出的 JSON 文件中提取模型回答，
与 metadata.json 中的标准答案对比，计算准确率。
"""

import argparse
import json
import re
from pathlib import Path


def extract_answer_letter(text: str) -> str:
    """从模型输出中提取答案字母"""
    text = text.strip()

    # 尝试匹配 "ANSWER: X" 格式
    match = re.search(r"ANSWER:\s*([A-Z])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 尝试匹配 "(X)" 格式
    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1).upper()

    # 尝试匹配单独的字母（开头）
    match = re.match(r"^\s*([A-Z])\b", text)
    if match:
        return match.group(1).upper()

    # 尝试匹配任何位置的单字母
    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1).upper()

    return ""


def evaluate(metadata_file: str, outputs_dir: str, result_file: str):
    """评估推理结果"""
    with open(metadata_file) as f:
        metadata = json.load(f)

    outputs_path = Path(outputs_dir)
    options = "ABCDEFGHIJ"

    correct = 0
    total = 0
    missing = 0
    results_detail = []

    # 按学科统计
    subject_stats = {}

    for item in metadata:
        batch_idx = item["batch_idx"]
        output_file = outputs_path / f"batch_{batch_idx:05d}.json"

        if not output_file.exists():
            missing += 1
            continue

        try:
            with open(output_file) as f:
                output_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            missing += 1
            continue

        # 从输出中获取生成的文本
        # Edge-LLM 输出格式可能是 {"responses": [{"text": "..."}]}
        generated_text = ""
        if "responses" in output_data:
            responses = output_data["responses"]
            if responses:
                generated_text = responses[0].get("text", "")
        elif "output" in output_data:
            generated_text = output_data["output"]
        elif "text" in output_data:
            generated_text = output_data["text"]
        else:
            # 尝试读取整个文件作为文本
            generated_text = json.dumps(output_data)

        # 提取答案
        pred_letter = extract_answer_letter(generated_text)
        gt_idx = item["answer"]
        gt_letter = options[gt_idx] if gt_idx < len(options) else ""

        is_correct = pred_letter == gt_letter
        if is_correct:
            correct += 1
        total += 1

        # 按学科统计
        subject = item.get("subject", "unknown")
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if is_correct:
            subject_stats[subject]["correct"] += 1

        results_detail.append({
            "pid": item["pid"],
            "predicted": pred_letter,
            "ground_truth": gt_letter,
            "correct": is_correct,
            "generated_text": generated_text[:200],
            "subject": subject,
        })

    # 计算准确率
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n总体准确率: {correct}/{total} = {accuracy:.2f}%")
    if missing > 0:
        print(f"缺失输出: {missing}")

    # 按学科打印
    print("\n按学科:")
    for subject, stats in sorted(subject_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {subject:20s}: {stats['correct']:4d}/{stats['total']:4d} = {acc:.2f}%")

    # 保存结果
    result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "missing": missing,
        "subject_stats": {
            k: {**v, "accuracy": v["correct"] / v["total"] * 100 if v["total"] > 0 else 0}
            for k, v in subject_stats.items()
        },
        "details": results_detail,
    }

    with open(result_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="评估 ScienceQA 推理结果")
    parser.add_argument("--metadata", required=True, help="metadata.json 路径")
    parser.add_argument("--outputs-dir", required=True, help="推理输出目录")
    parser.add_argument("--result-file", required=True, help="评估结果保存路径")
    args = parser.parse_args()

    evaluate(args.metadata, args.outputs_dir, args.result_file)


if __name__ == "__main__":
    main()
