CHUNKS=8
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$((IDX+0)) python -m llava.eval.model_vqa_science \
    --model-path /mnt/bn/yufei1900/wangxinhao/paper/checkpoint/LLaVA-vicuna-7B-v1.3-ScienceQA \
    --question-file /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/images/test \
    --question-file-calibrate /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/llava_train_QCM-LEA.json \
    --image-folder-calibrate /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/images/train \
    --answers-file /mnt/bn/yufei1900/wangxinhao/paper/QVLM/results/4_Bit_32_tokens/LLaVA-vicuna-7B-v1.3-4bit-chunk$CHUNKS_$IDX.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --load-4bit \
    --visual_token_num 32 \
    --conv-mode llava_v1  &
done
