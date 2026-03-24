CHUNKS=8
ALPHA=0.55
VISUAL_TOKEN_NUM=32
QUANT_METHOD="quant_error_group" # l2_norm l1_norm l_inf variance l2 quant_error quant_error_clip quant_error_group dynamic_range complex complex_l1 complex_mul
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$((IDX+0)) python -m llava.eval.model_vqa_science \
    --model-path /mnt/bn/yufei1900/wangxinhao/paper/checkpoint/LLaVA-vicuna-7B-v1.3-ScienceQA \
    --question-file /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/images/test \
    --question-file-calibrate /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/llava_train_QCM-LEA.json \
    --image-folder-calibrate /mnt/bn/yufei1900/wangxinhao/paper/ScienceQA/data/scienceqa/images/train \
    --answers-file /mnt/bn/yufei1900/wangxinhao/paper/QVLM_2/results/4_Bit_${VISUAL_TOKEN_NUM}_tokens_withquant_alpha_dynamic_method_${QUANT_METHOD}/LLaVA-vicuna-7B-v1.3-4bit-chunk$CHUNKS_$IDX.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --load-4bit \
    --visual_token_num $VISUAL_TOKEN_NUM \
    --add_quant \
    --alpha $ALPHA \
    --dynamic_alpha \
    --quant_method $QUANT_METHOD \
    --conv-mode llava_v1  &
done
