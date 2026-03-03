CUDA_VISIBLE_DEVICES=7 python -m llava.eval.model_vqa_science \
    --model-path /mnt/bn/yufei1900/wangxinhao/paper/checkpoint/LLaVA-vicuna-7B-v1.3-ScienceQA \
    --question-file /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/images/test \
    --question-file-calibrate /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/llava_train_QCM-LEA.json \
    --image-folder-calibrate /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/images/train \
    --answers-file /mnt/bn/yufei1900/wangxinhao/paper/QVLM/results/tmp/LLaVA-vicuna-7B-v1.3-4bit.jsonl \
    --conv-mode llava_v1 \
    --load-4bit