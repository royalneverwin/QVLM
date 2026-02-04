result_path="/home/wangxinhao/QVLM/results/ScienceQA"

CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_science_qa.py \
    --base-dir /home/wangxinhao/ScienceQA/data/scienceqa \
    --result-file $result_path/LLaVA-vicuna-7B-v1.3-4bit.jsonl \
    --output-file $result_path/test_llava-7b_output.json \
    --output-result $result_path/test_llava-7b_result.json 
