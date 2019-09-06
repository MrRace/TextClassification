#!/bin/bash

echo "labor-bert"
python3 off_line_predict_judge.py \
        --task_type_name labor \
        --model_name bert \
        --learning_rate 3e-5 \
        --data_prefix merge_processed_aug
