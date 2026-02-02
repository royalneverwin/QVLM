#!/bin/bash

CHUNKS=4
result_path="/home/wangxinhao/QVLM/results/ScienceQA_Prune_64"
output_file="$result_path/LLaVA-vicuna-7B-v1.3-4bit.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for idx in $(seq 0 $((CHUNKS-1))); do
  cat "$result_path/LLaVA-vicuna-7B-v1.3-4bit-chunk${idx}.jsonl" >> "$output_file"
done

CUDA_VISIBLE_DEVICES=7 python llava/eval/eval_science_qa.py \
    --base-dir /home/wangxinhao/ScienceQA/data/scienceqa \
    --result-file $output_file \
    --output-file $result_path/test_llava-7b_output.json \
    --output-result $result_path/test_llava-7b_result.json 

    