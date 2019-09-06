#!/bin/bash

echo "labor-wwm"
python3 bert-multilabel-classification.py \
        --task_type_name labor \
        --model_name wwm \
        --learning_rate 3e-5 \
        --data_prefix merge_processed_aug