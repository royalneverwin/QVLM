CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_science \
    --model-path /data1/public_data/llava_ckpt/LLaVA-vicuna-7B-v1.3-ScienceQA \
    --question-file /home/wangxinhao/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /home/wangxinhao/ScienceQA/data/scienceqa/images/test \
    --question-file-calibrate /home/wangxinhao/ScienceQA/data/scienceqa/llava_train_QCM-LEA.json \
    --image-folder-calibrate /home/wangxinhao/ScienceQA/data/scienceqa/images/train \
    --answers-file /home/wangxinhao/QVLM/results/ScienceQA/LLaVA-vicuna-7B-v1.3-4bit.jsonl \
    --conv-mode llava_v1 \
    --load-4bit