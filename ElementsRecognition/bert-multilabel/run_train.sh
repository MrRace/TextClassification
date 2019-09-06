#!/bin/bash

echo "divorce-bert"
python3 bert-multilabel-classification.py \
        --task_type_name divorce \
        --model_name bert \
        --learning_rate 3e-5