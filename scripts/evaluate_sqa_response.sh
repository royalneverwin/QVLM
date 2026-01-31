CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_science_qa.py \
    --base-dir /home/wangxinhao/ScienceQA/data/scienceqa \
    --result-file /home/wangxinhao/QVLM/results/ScienceQA/LLaVA-vicuna-7B-v1.3.jsonl \
    --output-file /home/wangxinhao/QVLM/results/ScienceQA/test_llava-7b_output.json \
    --output-result /home/wangxinhao/QVLM/results/ScienceQA/test_llava-7b_result.json 