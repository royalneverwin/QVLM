GPU_IDS=(0 1 2 3 4 5 6 7)
CHUNKS=${#GPU_IDS[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_ID=${GPU_IDS[$IDX]}
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.model_vqa_science \
    --model-path /mnt/bn/yufei1900/wangxinhao/paper/checkpoint/LLaVA-vicuna-7B-v1.3-ScienceQA \
    --question-file /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/images/test \
    --question-file-calibrate /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/llava_train_QCM-LEA.json \
    --image-folder-calibrate /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/images/train \
    --answers-file /mnt/bn/yufei1900/wangxinhao/paper/QVLM/results/Full_Bit/LLaVA-vicuna-7B-v1.3-4bit-chunk$CHUNKS_$IDX.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode llava_v1  &
done
