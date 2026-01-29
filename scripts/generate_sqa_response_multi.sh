CHUNKS=8
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$((IDX+0)) python -m llava.eval.model_vqa_science \
    --model-path /data1/public_data/llava_ckpt/LLaVA-vicuna-7B-v1.3-ScienceQA \
    --question-file /home/wangxinhao/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json \
    --image-folder /home/wangxinhao/ScienceQA/data/scienceqa/images/test \
    --question-file-calibrate /home/wangxinhao/ScienceQA/data/scienceqa/llava_train_QCM-LEPA.json \
    --image-folder-calibrate /home/wangxinhao/data/scienceqa/images/train \
    --answers-file /home/wangxinhao/QVLM/results/ScienceQA/LLaVA-vicuna-7B-v1.3-4bit-chunk$CHUNKS_$IDX.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --load-4bit \
    --conv-mode llava_v1  &
done
