#!/usr/bin/env python3
"""
评估 ScienceQA 推理结果的准确率。

从 llm_inference 输出的 JSON 文件中提取模型回答，
与 metadata.json 中的标准答案对比，计算准确率。
支持部分结果评估（推理未完成时也可运行）。
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

    # 尝试匹配 "The answer is X" 格式
    match = re.search(r"(?:the\s+)?answer\s+is\s+([A-Z])", text, re.IGNORECASE)
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


def get_output_text(output_data: dict) -> str:
    """从不同格式的输出中提取生成文本"""
    if "responses" in output_data:
        responses = output_data["responses"]
        if responses:
            return responses[0].get("text", responses[0].get("output_text", ""))
    if "output_text" in output_data:
        return output_data["output_text"]
    if "output" in output_data:
        return output_data["output"]
    if "text" in output_data:
        return output_data["text"]
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
    failed = 0
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

        # 检查是否推理失败
        generated_text = get_output_text(output_data)
        if "cannot handle this request" in generated_text.lower():
            failed += 1
            continue

        if not generated_text:
            failed += 1
            continue

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
    total_samples = len(metadata)

    # 打印结果
    print("╔══════════════════════════════════════════════╗")
    print("║              评估结果                        ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  数据集总量:    {total_samples}")
    print(f"║  有效评估:      {total}")
    print(f"║  推理失败:      {failed}")
    print(f"║  缺失输出:      {missing}")
    print(f"║  覆盖率:        {(total + failed) / total_samples * 100:.1f}%")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  ✅ 正确:       {correct}/{total}")
    print(f"║  📊 准确率:     {accuracy:.2f}%")
    print("╚══════════════════════════════════════════════╝")

    # 按学科打印
    if subject_stats:
        print("\n┌─────────────────────────────────────────────┐")
        print("│              按学科统计                       │")
        print("├─────────────────────────────────────────────┤")
        for subject, stats in sorted(subject_stats.items(),
                                      key=lambda x: x[1]["total"],
                                      reverse=True):
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"│  {subject:16s} {stats['correct']:4d}/{stats['total']:4d}  "
                  f"{bar} {acc:5.1f}%")
        print("└─────────────────────────────────────────────┘")

    # 保存结果
    result = {
        "accuracy": accuracy,
        "correct": correct,
        "total_evaluated": total,
        "total_samples": total_samples,
        "missing": missing,
        "failed": failed,
        "coverage": (total + failed) / total_samples * 100 if total_samples > 0 else 0,
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
