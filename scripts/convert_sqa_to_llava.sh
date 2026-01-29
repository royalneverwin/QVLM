for split in train val minival test minitest; do
    python scripts/convert_sqa_to_llava.py convert_to_llava \
    --base-dir /home/wangxinhao/ScienceQA/data/scienceqa \
    --prompt-format "QCM-LEA" \
    --split $split
done